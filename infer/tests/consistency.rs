//! Cross-check Rust inference against holdout.csv scores produced by the
//! Python training side, for every model that has artifacts under models/.
//! Skips when artifacts or the shared library are not present yet.

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde_json::{Map, Number, Value};

use xgbooster_infer::registry::{self, pick_latest, Entry, Registry};
use xgbooster_infer::{ffi, predict};

fn models_dir() -> PathBuf {
    std::env::var("XGBOOSTER_MODELS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| Path::new(env!("CARGO_MANIFEST_DIR")).join("../models"))
}

fn lib_path() -> PathBuf {
    std::env::var("XGBOOSTER_LIB")
        .map(PathBuf::from)
        .unwrap_or_else(|_| Path::new(env!("CARGO_MANIFEST_DIR")).join("lib/libxgboost.so"))
}

/// Model names (sorted) having at least one version with a model.ubj.
fn discover_models(models: &Path) -> Vec<String> {
    let mut names: Vec<String> = std::fs::read_dir(models)
        .into_iter()
        .flatten()
        .filter_map(|e| e.ok())
        .filter(|e| latest_version_dir(models, &e.file_name().to_string_lossy()).is_some())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect();
    names.sort();
    names
}

fn latest_version_dir(models: &Path, name: &str) -> Option<PathBuf> {
    let dir = models.join(name);
    let versions: Vec<String> = std::fs::read_dir(&dir)
        .ok()?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().join("model.ubj").is_file())
        .map(|e| e.file_name().to_string_lossy().into_owned())
        .collect();
    pick_latest(&versions).map(|v| dir.join(v))
}

/// Column expected to hold the reference score: a non-feature column whose
/// header mentions score/pred/prob, else the last non-feature column.
fn score_column(header: &[String], feature_names: &HashSet<&str>) -> Option<usize> {
    let candidates: Vec<usize> = header
        .iter()
        .enumerate()
        .filter(|(_, h)| !feature_names.contains(h.as_str()))
        .map(|(i, _)| i)
        .collect();
    for &i in candidates.iter().rev() {
        let lower = header[i].to_ascii_lowercase();
        if lower.contains("score") || lower.contains("pred") || lower.contains("prob") {
            return Some(i);
        }
    }
    candidates.last().copied()
}

/// One holdout.csv cell as JSON. Empty cells (Python wrote None -> "") become
/// Null for numbers and an unmapped empty category for categoricals, both of
/// which extract to NaN downstream - matching Python's None -> NaN semantics.
fn cell_value(cell: &str, categorical: bool) -> Value {
    if categorical {
        return Value::String(cell.to_string());
    }
    match cell.trim().parse::<f64>() {
        Ok(n) => Number::from_f64(n)
            .map(Value::Number)
            .unwrap_or(Value::Null),
        Err(_) => Value::Null,
    }
}

fn parse_holdout(text: &str) -> (Vec<String>, Vec<Vec<String>>) {
    let mut lines = text.lines().filter(|l| !l.trim().is_empty());
    let header: Vec<String> = lines
        .next()
        .expect("holdout.csv has a header")
        .split(',')
        .map(|h| h.trim().to_string())
        .collect();
    let rows: Vec<Vec<String>> = lines
        .map(|l| l.split(',').map(|c| c.trim().to_string()).collect())
        .collect();
    (header, rows)
}

/// Replay one model's latest holdout.csv through predict::predict and return
/// (n_rows, max_abs_diff); asserts every row matches within 1e-6.
fn check_model(reg: &Registry, models: &Path, name: &str) -> (usize, f64) {
    let version_dir = latest_version_dir(models, name)
        .unwrap_or_else(|| panic!("model {} lost its artifacts mid-run", name));
    let entry: &Arc<Entry> = registry::resolve(reg, name)
        .unwrap_or_else(|e| panic!("latest {} resolves: {:?}", name, e));
    assert_eq!(
        entry.version,
        version_dir.file_name().unwrap().to_string_lossy(),
        "resolved latest must match the newest artifact directory"
    );

    let holdout_text =
        std::fs::read_to_string(version_dir.join("holdout.csv")).expect("holdout.csv readable");
    let (header, rows) = parse_holdout(&holdout_text);
    let schema = &entry.manifest.feature_schema;
    let feature_names: HashSet<&str> = schema.iter().map(|s| s.name.as_str()).collect();
    let score_idx = score_column(&header, &feature_names).expect("a score column exists");
    assert!(!rows.is_empty(), "holdout.csv has data rows");

    let mut max_abs_diff: f64 = 0.0;
    for (row_no, row) in rows.iter().enumerate() {
        assert_eq!(row.len(), header.len(), "row {} width mismatch", row_no + 1);
        let mut features = Map::new();
        for spec in schema {
            let col = header
                .iter()
                .position(|h| h == &spec.name)
                .unwrap_or_else(|| panic!("feature {} missing from holdout header", spec.name));
            features.insert(
                spec.name.clone(),
                cell_value(&row[col], spec.ftype == "categorical"),
            );
        }
        let outcome = predict::predict(entry, &features, None)
            .unwrap_or_else(|e| panic!("row {} predict failed: {:?}", row_no + 1, e));
        let expected: f64 = row[score_idx].trim().parse().expect("score cell parses");
        let diff = (outcome.score as f64 - expected).abs();
        max_abs_diff = max_abs_diff.max(diff);
        assert!(
            diff < 1e-6,
            "{} row {}: rust {} vs holdout {} (diff {diff})",
            name,
            row_no + 1,
            outcome.score,
            expected
        );
    }
    println!(
        "model {}@{}: n_rows={} max_abs_diff={:.3e} PASS",
        name, entry.version, rows.len(), max_abs_diff
    );
    (rows.len(), max_abs_diff)
}

#[test]
fn holdout_consistency() {
    let models = models_dir();
    let lib_file = lib_path();
    if !lib_file.is_file() {
        eprintln!("SKIP: {} not present yet", lib_file.display());
        return;
    }
    let names = discover_models(&models);
    if names.is_empty() {
        eprintln!("SKIP: no model artifacts under {}", models.display());
        return;
    }

    let lib = Arc::new(ffi::load_lib(&lib_file).expect("library loads"));
    let reg = registry::load_registry(
        &lib,
        &models,
        &registry::LoadOptions {
            strict_xgboost_version: false,
        },
    )
    .expect("registry loads");

    let mut total_rows = 0usize;
    for name in &names {
        let (n_rows, _) = check_model(&reg, &models, name);
        total_rows += n_rows;
    }
    println!(
        "all models consistent: {} models, {} rows total",
        names.len(),
        total_rows
    );
}
