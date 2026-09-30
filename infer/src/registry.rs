//! Model registry: scans `models/<name>/<version>/` artifact directories,
//! loads boosters, and resolves model references.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use anyhow::{anyhow, bail, Context, Result};
use serde::Serialize;
use serde_json::Value;

use crate::features::{manifest_from_path, Manifest};
use crate::ffi;

/// One loaded model version.
pub struct Entry {
    pub name: String,
    pub version: String,
    pub key: String,
    pub manifest: Manifest,
    booster: Mutex<ffi::XgbBooster>,
}

/// All loaded model entries plus the latest entry per model name.
pub struct Registry {
    entries: HashMap<String, Arc<Entry>>,
    latest: HashMap<String, Arc<Entry>>,
}

/// Serializable model listing entry.
#[derive(Debug, Clone, Serialize)]
pub struct ModelInfo {
    pub name: String,
    pub version: String,
    pub key: String,
    pub metrics: HashMap<String, Value>,
}

/// Resolve failures classified for HTTP status mapping.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResolveError {
    UnknownModel(String),
    InvalidReference(String),
}

impl std::fmt::Display for ResolveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ResolveError::UnknownModel(m) => write!(f, "{m}"),
            ResolveError::InvalidReference(m) => write!(f, "{m}"),
        }
    }
}

impl std::error::Error for ResolveError {}

/// Options for [`load_registry`].
#[derive(Debug, Clone, Copy)]
pub struct LoadOptions {
    /// Reject (instead of warn-and-load) artifacts whose manifest lacks
    /// `xgboost_version` or whose runtime library version is unknown.
    pub strict_xgboost_version: bool,
}

/// Lexicographically greatest version string, i.e. the latest for
/// "%Y-%m-%dT%H%M%S" style versions with an optional zero-padded `-NN`/`-NNN`
/// run suffix. The suffix is compared numerically so `-10` beats `-9`.
pub fn pick_latest(versions: &[String]) -> Option<String> {
    fn sort_key(v: &str) -> (String, u64) {
        match v.rsplit_once('-') {
            Some((base, suffix))
                if !suffix.is_empty() && suffix.bytes().all(|b| b.is_ascii_digit()) =>
            {
                (base.to_string(), suffix.parse::<u64>().unwrap_or(0))
            }
            _ => (v.to_string(), 0),
        }
    }
    versions.iter().max_by_key(|v| sort_key(v)).cloned()
}

/// Compare the manifest's recorded xgboost version against the runtime
/// library; refuse to load on mismatch. In strict mode, missing version
/// information is a hard error instead of a warning.
pub fn check_xgboost_version(
    manifest: &Manifest,
    runtime: Option<(i32, i32, i32)>,
    artifact: &Path,
    strict: bool,
) -> Result<()> {
    let manifest_ver = match &manifest.xgboost_version {
        None => {
            if strict {
                bail!(
                    "model artifact {} has no xgboost_version in its manifest and strict version checking is enabled (XGBOOSTER_STRICT_VERSION); refusing to load",
                    artifact.display()
                );
            }
            eprintln!(
                "warn: {} has no xgboost_version; skipping runtime check",
                artifact.display()
            );
            return Ok(());
        }
        Some(v) => v,
    };
    let (major, minor, patch) = match runtime {
        None => {
            if strict {
                bail!(
                    "runtime xgboost version unknown and strict version checking is enabled (XGBOOSTER_STRICT_VERSION); refusing to load {}",
                    artifact.display()
                );
            }
            eprintln!(
                "warn: runtime xgboost version unknown; skipping check for {}",
                artifact.display()
            );
            return Ok(());
        }
        Some(t) => t,
    };
    let runtime_ver = format!("{major}.{minor}.{patch}");
    if &runtime_ver != manifest_ver {
        bail!(
            "model artifact {} was trained with xgboost {} but runtime library is {}.{}.{}; refusing to load (retrain the artifact or deploy a matching libxgboost)",
            artifact.display(),
            manifest_ver,
            major,
            minor,
            patch
        );
    }
    Ok(())
}

/// Scan two directory levels below `models_dir`. Returns
/// (name, version, version_dir) triples for complete artifacts.
/// A version directory holding only one of manifest.json / model.ubj is an
/// error; directories with neither are skipped.
fn scan_versions(models_dir: &Path) -> Result<Vec<(String, String, PathBuf)>> {
    let mut found = Vec::new();
    for name in sorted_dir_names(models_dir)? {
        let name_dir = models_dir.join(&name);
        if !name_dir.is_dir() {
            continue;
        }
        for version in sorted_dir_names(&name_dir)? {
            let version_dir = name_dir.join(&version);
            if !version_dir.is_dir() {
                continue;
            }
            let has_manifest = version_dir.join("manifest.json").exists();
            let has_model = version_dir.join("model.ubj").exists();
            if has_manifest && has_model {
                found.push((name.clone(), version, version_dir));
            } else if has_manifest || has_model {
                bail!(
                    "incomplete model artifact at {}: both manifest.json and model.ubj are required",
                    version_dir.display()
                );
            }
        }
    }
    Ok(found)
}

fn sorted_dir_names(dir: &Path) -> Result<Vec<String>> {
    let mut names: Vec<String> = std::fs::read_dir(dir)
        .with_context(|| format!("cannot read directory {}", dir.display()))?
        .filter_map(|e| e.ok())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect();
    names.sort();
    Ok(names)
}

/// Build a registry from all artifacts under `models_dir`, honoring
/// `options` (see [`LoadOptions`]).
pub fn load_registry(
    lib: &Arc<ffi::Lib>,
    models_dir: &Path,
    options: &LoadOptions,
) -> Result<Registry> {
    if !models_dir.is_dir() {
        bail!("models directory not found: {}", models_dir.display());
    }
    let versions = scan_versions(models_dir)?;
    let mut entries: HashMap<String, Arc<Entry>> = HashMap::new();
    let mut by_name: HashMap<String, Vec<String>> = HashMap::new();
    for (name, version, version_dir) in versions {
        let manifest = manifest_from_path(&version_dir.join("manifest.json"))
            .with_context(|| format!("bad manifest at {}", version_dir.display()))?;
        if manifest.name != name {
            bail!(
                "manifest name {:?} does not match artifact directory {:?} at {}",
                manifest.name,
                name,
                version_dir.display()
            );
        }
        if manifest.version != version {
            bail!(
                "manifest version {:?} does not match artifact directory {:?} at {}",
                manifest.version,
                version,
                version_dir.display()
            );
        }
        let model_path = version_dir.join("model.ubj");
        check_xgboost_version(
            &manifest,
            ffi::library_version(lib),
            &model_path,
            options.strict_xgboost_version,
        )?;
        let booster = ffi::booster_from_model_file(lib, &model_path)
            .with_context(|| format!("failed to load booster at {}", version_dir.display()))?;
        let key = format!("{}@{}", name, version);
        entries.insert(
            key.clone(),
            Arc::new(Entry {
                name: name.clone(),
                version: version.clone(),
                key,
                manifest,
                booster: Mutex::new(booster),
            }),
        );
        by_name.entry(name).or_default().push(version);
    }
    if entries.is_empty() {
        bail!("no model artifacts found under {}", models_dir.display());
    }
    let mut latest: HashMap<String, Arc<Entry>> = HashMap::new();
    for (name, versions) in by_name {
        let version = pick_latest(&versions).expect("version list is non-empty");
        let key = format!("{}@{}", name, version);
        let entry = entries.get(&key).expect("entry exists").clone();
        latest.insert(name, entry);
    }
    Ok(Registry { entries, latest })
}

/// Resolve `"name"` (latest version) or `"name@version"` (exact).
pub fn resolve<'a>(reg: &'a Registry, model_ref: &str) -> Result<&'a Arc<Entry>, ResolveError> {
    let missing = || ResolveError::UnknownModel(format!("unknown model \"{}\"", model_ref));
    match model_ref.split_once('@') {
        None => reg.latest.get(model_ref).ok_or_else(missing),
        Some((name, version)) if name.is_empty() || version.is_empty() => {
            Err(ResolveError::InvalidReference(format!(
                "invalid model reference \"{}\" (expected \"name\" or \"name@version\")",
                model_ref
            )))
        }
        Some(_) => reg.entries.get(model_ref).ok_or_else(missing),
    }
}

/// List all loaded models, sorted by registry key.
pub fn list_models(reg: &Registry) -> Vec<ModelInfo> {
    let mut infos: Vec<ModelInfo> = reg
        .entries
        .values()
        .map(|entry| ModelInfo {
            name: entry.name.clone(),
            version: entry.version.clone(),
            key: entry.key.clone(),
            metrics: entry.manifest.metrics.clone(),
        })
        .collect();
    infos.sort_by(|a, b| a.key.cmp(&b.key));
    infos
}

/// Number of loaded model versions.
pub fn model_count(reg: &Registry) -> usize {
    reg.entries.len()
}

/// Lock and expose the entry's booster for prediction.
pub fn lock_booster(entry: &Entry) -> Result<std::sync::MutexGuard<'_, ffi::XgbBooster>> {
    entry
        .booster
        .lock()
        .map_err(|_| anyhow!("booster mutex poisoned for model {}", entry.key))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pick_latest_prefers_lexicographic_max() {
        let versions = vec![
            "2026-09-30T2116".to_string(),
            "2025-01-01T0000".to_string(),
            "2026-10-01T0000".to_string(),
        ];
        assert_eq!(pick_latest(&versions).as_deref(), Some("2026-10-01T0000"));
        assert_eq!(pick_latest(&[]), None);
        assert_eq!(pick_latest(&["v1".to_string()]).as_deref(), Some("v1"));
    }

    #[test]
    fn pick_latest_compares_numeric_run_suffix() {
        let versions = vec!["a-9".to_string(), "a-10".to_string()];
        assert_eq!(pick_latest(&versions).as_deref(), Some("a-10"));

        let versions = vec![
            "2026-09-30T152800".to_string(),
            "2026-09-30T152800-01".to_string(),
        ];
        assert_eq!(
            pick_latest(&versions).as_deref(),
            Some("2026-09-30T152800-01")
        );

        // non-numeric suffixes stay plain lexicographic
        let versions = vec![
            "2026-09-30T152800".to_string(),
            "2026-09-30T152800-dev".to_string(),
        ];
        assert_eq!(
            pick_latest(&versions).as_deref(),
            Some("2026-09-30T152800-dev")
        );
    }

    fn version_manifest(ver: Option<&str>) -> Manifest {
        Manifest {
            name: "m".to_string(),
            version: "v".to_string(),
            created_at: None,
            xgboost_version: ver.map(str::to_string),
            objective: None,
            n_features: 0,
            feature_schema: vec![],
            metrics: HashMap::new(),
        }
    }

    #[test]
    fn check_xgboost_version_match_ok() {
        let manifest = version_manifest(Some("3.4.1"));
        let artifact = Path::new("models/m/v/model.ubj");
        assert!(check_xgboost_version(&manifest, Some((3, 4, 1)), artifact, false).is_ok());
    }

    #[test]
    fn check_xgboost_version_mismatch_err() {
        let manifest = version_manifest(Some("3.4.1"));
        let artifact = Path::new("models/m/v/model.ubj");
        let err = check_xgboost_version(&manifest, Some((3, 3, 0)), artifact, false).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("trained with xgboost 3.4.1"), "{msg}");
        assert!(msg.contains("runtime library is 3.3.0"), "{msg}");
        assert!(msg.contains("refusing to load"), "{msg}");
    }

    #[test]
    fn check_xgboost_version_missing_fields_ok() {
        let artifact = Path::new("models/m/v/model.ubj");
        assert!(
            check_xgboost_version(&version_manifest(None), Some((3, 4, 1)), artifact, false)
                .is_ok()
        );
        assert!(
            check_xgboost_version(&version_manifest(Some("3.4.1")), None, artifact, false).is_ok()
        );
    }

    #[test]
    fn check_xgboost_version_strict_rejects_missing_fields() {
        let artifact = Path::new("models/m/v/model.ubj");

        // manifest without a recorded version is refused under strict mode
        let err = check_xgboost_version(&version_manifest(None), Some((3, 4, 1)), artifact, true)
            .unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("strict"), "{msg}");
        assert!(msg.contains("refusing to load"), "{msg}");

        // unknown runtime version is refused under strict mode too
        let err = check_xgboost_version(&version_manifest(Some("3.4.1")), None, artifact, true)
            .unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("strict"), "{msg}");
        assert!(msg.contains("refusing to load"), "{msg}");

        // complete and matching versions still load under strict mode
        assert!(check_xgboost_version(
            &version_manifest(Some("3.4.1")),
            Some((3, 4, 1)),
            artifact,
            true
        )
        .is_ok());
    }

    #[test]
    fn scan_versions_rejects_incomplete_artifacts() {
        let dir = std::env::temp_dir().join(format!("xgb-registry-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("good/v1")).unwrap();
        std::fs::create_dir_all(dir.join("empty/v1")).unwrap();

        std::fs::write(dir.join("good/v1/manifest.json"), "{}").unwrap();
        std::fs::write(dir.join("good/v1/model.ubj"), "x").unwrap();

        // directories with neither file are skipped
        let scanned = scan_versions(&dir).unwrap();
        assert_eq!(scanned.len(), 1);
        assert_eq!(scanned[0].0, "good");
        assert_eq!(scanned[0].1, "v1");
        assert_eq!(scanned[0].2.file_name().unwrap().to_string_lossy(), "v1");

        // manifest-only artifact is an error
        std::fs::create_dir_all(dir.join("partial/v1")).unwrap();
        std::fs::write(dir.join("partial/v1/manifest.json"), "{}").unwrap();
        let err = scan_versions(&dir).unwrap_err();
        assert!(
            err.to_string().contains("incomplete model artifact"),
            "{err}"
        );

        // model-only artifact is an error too
        std::fs::remove_file(dir.join("partial/v1/manifest.json")).unwrap();
        std::fs::write(dir.join("partial/v1/model.ubj"), "x").unwrap();
        let err = scan_versions(&dir).unwrap_err();
        assert!(
            err.to_string().contains("incomplete model artifact"),
            "{err}"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn load_registry_integration_or_skip() {
        let base = Path::new(env!("CARGO_MANIFEST_DIR"));
        let models_dir = std::env::var("XGBOOSTER_MODELS_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|_| base.join("../models"));
        let lib_path = std::env::var("XGBOOSTER_LIB")
            .map(PathBuf::from)
            .unwrap_or_else(|_| base.join("lib/libxgboost.so"));
        if !models_dir.is_dir() || !lib_path.is_file() {
            eprintln!(
                "SKIP: no real models artifacts at {} or library at {}",
                models_dir.display(),
                lib_path.display()
            );
            return;
        }
        let lib = Arc::new(crate::ffi::load_lib(&lib_path).expect("library loads"));
        let reg = load_registry(
            &lib,
            &models_dir,
            &LoadOptions {
                strict_xgboost_version: false,
            },
        )
        .expect("registry loads");
        assert!(model_count(&reg) > 0);
        let infos = list_models(&reg);
        assert_eq!(infos.len(), model_count(&reg));
        assert!(infos.windows(2).all(|w| w[0].key <= w[1].key));
    }
}
