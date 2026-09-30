//! Feature schema types and JSON -> dense f32 vector extraction.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{bail, Context, Result};
use serde::Deserialize;
use serde_json::{Map, Value};

/// One entry of `manifest.json`'s `feature_schema`.
#[derive(Debug, Clone, Deserialize)]
pub struct FeatureSpec {
    pub idx: usize,
    pub name: String,
    #[serde(rename = "type")]
    pub ftype: String,
    #[serde(default)]
    pub mapping: Option<HashMap<String, f64>>,
}

/// Parsed `manifest.json` for one model artifact.
#[derive(Debug, Clone, Deserialize)]
pub struct Manifest {
    pub name: String,
    pub version: String,
    pub created_at: Option<String>,
    pub xgboost_version: Option<String>,
    pub objective: Option<String>,
    pub n_features: usize,
    pub feature_schema: Vec<FeatureSpec>,
    #[serde(default)]
    pub metrics: HashMap<String, Value>,
}

/// Read and parse a `manifest.json` from disk.
pub fn manifest_from_path(path: &Path) -> Result<Manifest> {
    let raw = std::fs::read_to_string(path)
        .with_context(|| format!("cannot read manifest {}", path.display()))?;
    let manifest: Manifest = serde_json::from_str(&raw)
        .with_context(|| format!("invalid manifest JSON in {}", path.display()))?;
    validate_schema(&manifest.feature_schema, manifest.n_features)
        .with_context(|| format!("invalid feature schema in {}", path.display()))?;
    Ok(manifest)
}

/// Check that schema idx values are exactly 0..n_features-1 and types are known.
pub fn validate_schema(schema: &[FeatureSpec], n_features: usize) -> Result<()> {
    if schema.len() != n_features {
        bail!(
            "feature schema has {} entries but n_features is {}",
            schema.len(),
            n_features
        );
    }
    let mut seen = vec![false; n_features];
    for spec in schema {
        if spec.idx >= n_features {
            bail!(
                "feature \"{}\" has out-of-range idx {}",
                spec.name,
                spec.idx
            );
        }
        if seen[spec.idx] {
            bail!(
                "duplicate idx {} in feature schema (feature \"{}\")",
                spec.idx,
                spec.name
            );
        }
        seen[spec.idx] = true;
        if spec.ftype != "number" && spec.ftype != "categorical" {
            bail!(
                "feature \"{}\" has unsupported type \"{}\" (expected \"number\" or \"categorical\")",
                spec.name,
                spec.ftype
            );
        }
        if spec.ftype == "categorical" && spec.mapping.is_none() {
            bail!(
                "categorical feature \"{}\" is missing its mapping",
                spec.name
            );
        }
    }
    Ok(())
}

/// Non-fatal feature extraction anomalies surfaced to callers/logs.
#[derive(Debug, Default, Clone, PartialEq)]
pub struct ExtractionNotes {
    pub missing: Vec<String>,
    pub unknown_categorical: Vec<String>,
    pub unexpected_fields: Vec<String>,
}

impl ExtractionNotes {
    pub fn is_empty(&self) -> bool {
        self.missing.is_empty()
            && self.unknown_categorical.is_empty()
            && self.unexpected_fields.is_empty()
    }
}

/// Extract a dense feature vector from a JSON feature object.
///
/// Output slot i is filled from the schema entry with idx == i.
/// Missing/null and unknown-categorical inputs become NaN and are
/// reported in the returned notes; type errors hard-fail.
pub fn to_vector(
    schema: &[FeatureSpec],
    n_features: usize,
    json: &Map<String, Value>,
) -> Result<(Vec<f32>, ExtractionNotes)> {
    validate_schema(schema, n_features)?;
    let mut out = vec![f32::NAN; n_features];
    let mut notes = ExtractionNotes::default();
    for spec in schema {
        out[spec.idx] = extract_one(spec, json, &mut notes)?;
    }
    let known: std::collections::HashSet<&str> = schema.iter().map(|s| s.name.as_str()).collect();
    let mut unexpected: Vec<String> = json
        .keys()
        .filter(|k| !known.contains(k.as_str()))
        .cloned()
        .collect();
    unexpected.sort();
    notes.unexpected_fields = unexpected;
    Ok((out, notes))
}

fn extract_one(
    spec: &FeatureSpec,
    json: &Map<String, Value>,
    notes: &mut ExtractionNotes,
) -> Result<f32> {
    match json.get(&spec.name) {
        None | Some(Value::Null) => {
            notes.missing.push(spec.name.clone());
            Ok(f32::NAN)
        }
        Some(value) => match (spec.ftype.as_str(), value) {
            ("categorical", Value::String(s)) => {
                match spec.mapping.as_ref().and_then(|m| m.get(s)) {
                    Some(v) => Ok(*v as f32),
                    None => {
                        notes
                            .unknown_categorical
                            .push(format!("{}={}", spec.name, s));
                        Ok(f32::NAN)
                    }
                }
            }
            _ => convert(spec, value),
        },
    }
}

fn convert(spec: &FeatureSpec, value: &Value) -> Result<f32> {
    match spec.ftype.as_str() {
        "number" => match value {
            Value::Number(n) => Ok(n.as_f64().unwrap_or(f64::NAN) as f32),
            Value::Bool(b) => Ok(if *b { 1.0 } else { 0.0 }),
            _ => bail!(
                "feature \"{}\" expects a JSON number, got {}",
                spec.name,
                describe(value)
            ),
        },
        "categorical" => match value {
            Value::String(s) => Ok(spec
                .mapping
                .as_ref()
                .and_then(|m| m.get(s))
                .map(|v| *v as f32)
                .unwrap_or(f32::NAN)),
            _ => bail!(
                "feature \"{}\" expects a JSON string, got {}",
                spec.name,
                describe(value)
            ),
        },
        other => bail!(
            "unsupported feature type \"{}\" for \"{}\"",
            other,
            spec.name
        ),
    }
}

fn describe(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "a boolean",
        Value::Number(_) => "a number",
        Value::String(_) => "a string",
        Value::Array(_) => "an array",
        Value::Object(_) => "an object",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn schema_entry(idx: usize, name: &str, ftype: &str) -> FeatureSpec {
        serde_json::from_value(json!({
            "idx": idx,
            "name": name,
            "type": ftype,
            "mapping": if ftype == "categorical" {
                json!({"red": 0.0, "blue": 1.0})
            } else {
                Value::Null
            }
        }))
        .unwrap()
    }

    fn features(value: Value) -> Map<String, Value> {
        value.as_object().expect("object").clone()
    }

    fn vec_of(schema: &[FeatureSpec], value: Value) -> Vec<f32> {
        to_vector(schema, schema.len(), &features(value)).unwrap().0
    }

    fn notes_of(schema: &[FeatureSpec], value: Value) -> ExtractionNotes {
        to_vector(schema, schema.len(), &features(value)).unwrap().1
    }

    #[test]
    fn number_features() {
        let schema = [
            schema_entry(0, "amount", "number"),
            schema_entry(1, "flag", "number"),
        ];
        let v = vec_of(&schema, json!({"amount": 12.5, "flag": 7}));
        assert_eq!(v[0], 12.5);
        assert_eq!(v[1], 7.0);

        let v = vec_of(&schema, json!({"amount": true, "flag": false}));
        assert_eq!(v[0], 1.0);
        assert_eq!(v[1], 0.0);
    }

    #[test]
    fn number_missing_becomes_nan() {
        let schema = [
            schema_entry(0, "amount", "number"),
            schema_entry(1, "flag", "number"),
        ];
        let v = vec_of(&schema, json!({"amount": 1.0}));
        assert_eq!(v[0], 1.0);
        assert!(v[1].is_nan());

        let v = vec_of(&schema, json!({"amount": Value::Null, "flag": Value::Null}));
        assert!(v[0].is_nan());
        assert!(v[1].is_nan());
    }

    #[test]
    fn number_wrong_type_errors() {
        let schema = [schema_entry(0, "amount", "number")];
        for bad in [
            json!({"amount": "12.5"}),
            json!({"amount": [1]}),
            json!({"amount": {"v": 1}}),
        ] {
            let err = to_vector(&schema, 1, &features(bad)).unwrap_err();
            assert!(err.to_string().contains("expects a JSON number"), "{err}");
        }
    }

    #[test]
    fn categorical_known_and_unknown() {
        let schema = [schema_entry(0, "color", "categorical")];
        assert_eq!(vec_of(&schema, json!({"color": "red"}))[0], 0.0);
        assert_eq!(vec_of(&schema, json!({"color": "blue"}))[0], 1.0);
        assert!(vec_of(&schema, json!({"color": "green"}))[0].is_nan());
    }

    #[test]
    fn categorical_missing_and_wrong_type() {
        let schema = [schema_entry(0, "color", "categorical")];
        assert!(vec_of(&schema, json!({}))[0].is_nan());
        assert!(vec_of(&schema, json!({"color": Value::Null}))[0].is_nan());
        for bad in [json!({"color": 3}), json!({"color": true})] {
            let err = to_vector(&schema, 1, &features(bad)).unwrap_err();
            assert!(err.to_string().contains("expects a JSON string"), "{err}");
        }
    }

    #[test]
    fn idx_controls_output_position() {
        let schema = [
            schema_entry(2, "c", "number"),
            schema_entry(0, "a", "number"),
            schema_entry(1, "b", "number"),
        ];
        let v = vec_of(&schema, json!({"a": 1.0, "b": 2.0, "c": 3.0}));
        assert_eq!(v, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn notes_report_missing_in_schema_order() {
        let schema = [
            schema_entry(0, "b", "number"),
            schema_entry(1, "a", "number"),
            schema_entry(2, "color", "categorical"),
        ];
        let notes = notes_of(&schema, json!({"color": "red"}));
        assert_eq!(notes.missing, vec!["b".to_string(), "a".to_string()]);

        let notes = notes_of(
            &schema,
            json!({"b": Value::Null, "a": 1.0, "color": Value::Null}),
        );
        assert_eq!(notes.missing, vec!["b".to_string(), "color".to_string()]);
    }

    #[test]
    fn notes_report_unknown_categorical() {
        let schema = [schema_entry(0, "color", "categorical")];
        let notes = notes_of(&schema, json!({"color": "green"}));
        assert_eq!(notes.unknown_categorical, vec!["color=green".to_string()]);
        assert!(notes.missing.is_empty());
    }

    #[test]
    fn notes_report_unexpected_fields_sorted() {
        let schema = [schema_entry(0, "a", "number")];
        let notes = notes_of(&schema, json!({"a": 1.0, "zz": 2, "bb": 3}));
        assert_eq!(
            notes.unexpected_fields,
            vec!["bb".to_string(), "zz".to_string()]
        );
    }

    #[test]
    fn notes_empty_when_all_features_resolve() {
        let schema = [
            schema_entry(0, "a", "number"),
            schema_entry(1, "color", "categorical"),
        ];
        let notes = notes_of(&schema, json!({"a": 1.0, "color": "red"}));
        assert_eq!(notes, ExtractionNotes::default());
        assert!(notes.is_empty());
    }

    #[test]
    fn schema_validation_rejects_bad_idx() {
        let dup = [
            schema_entry(0, "a", "number"),
            schema_entry(0, "b", "number"),
        ];
        assert!(to_vector(&dup, 2, &features(json!({}))).is_err());

        let oob = [schema_entry(3, "a", "number")];
        assert!(validate_schema(&oob, 1).is_err());

        let short = [schema_entry(0, "a", "number")];
        assert!(validate_schema(&short, 2).is_err());

        let bad_type = [schema_entry(0, "a", "string")];
        assert!(validate_schema(&bad_type, 1).is_err());

        let no_mapping = [FeatureSpec {
            idx: 0,
            name: "a".to_string(),
            ftype: "categorical".to_string(),
            mapping: None,
        }];
        assert!(validate_schema(&no_mapping, 1).is_err());
    }

    #[test]
    fn manifest_roundtrip_and_errors() {
        let dir = std::env::temp_dir().join(format!("xgb-manifest-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();

        let good = dir.join("manifest.json");
        std::fs::write(
            &good,
            json!({
                "name": "risk_score",
                "version": "2026-09-30T2116",
                "created_at": "2026-09-30T21:16:00",
                "xgboost_version": "3.4.1",
                "objective": "binary:logistic",
                "n_features": 2,
                "feature_schema": [
                    {"idx": 0, "name": "amount", "type": "number"},
                    {"idx": 1, "name": "color", "type": "categorical", "mapping": {"red": 0.0}}
                ],
                "metrics": {"auc": 0.91}
            })
            .to_string(),
        )
        .unwrap();
        let manifest = manifest_from_path(&good).unwrap();
        assert_eq!(manifest.name, "risk_score");
        assert_eq!(manifest.version, "2026-09-30T2116");
        assert_eq!(manifest.n_features, 2);
        assert_eq!(manifest.metrics["auc"], json!(0.91));
        assert_eq!(manifest.objective.as_deref(), Some("binary:logistic"));

        let bad = dir.join("bad.json");
        std::fs::write(&bad, "{not json").unwrap();
        assert!(manifest_from_path(&bad).is_err());

        let mismatch = dir.join("mismatch.json");
        std::fs::write(
            &mismatch,
            json!({"name": "m", "version": "v", "n_features": 1, "feature_schema": [
                {"idx": 5, "name": "a", "type": "number"}
            ]})
            .to_string(),
        )
        .unwrap();
        assert!(manifest_from_path(&mismatch).is_err());

        let _ = std::fs::remove_dir_all(&dir);
    }
}
