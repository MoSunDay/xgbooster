//! Single-row prediction over a registry entry.

use std::time::Instant;

use serde::Serialize;
use serde_json::{Map, Value};

use crate::features::{to_vector, ExtractionNotes};
use crate::ffi;
use crate::registry::{lock_booster, Entry};
use crate::throttle::WarnThrottle;

/// Prediction failures classified for HTTP mapping.
#[derive(Debug)]
pub enum PredictError {
    UnknownModel(String),
    BadRequest(String),
    Internal(String),
}

/// Human-readable message for a prediction failure.
pub fn error_message(err: &PredictError) -> String {
    match err {
        PredictError::UnknownModel(m) => m.clone(),
        PredictError::BadRequest(m) => m.clone(),
        PredictError::Internal(m) => m.clone(),
    }
}

/// Result of a successful prediction.
#[derive(Debug, Serialize)]
pub struct PredictionOutcome {
    pub score: f32,
    #[serde(rename = "model")]
    pub model_key: String,
    pub latency_us: u64,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub missing_features: Vec<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub unknown_categories: Vec<String>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub unexpected_fields: Vec<String>,
}

/// Deterministic throttle key for one warning signature: the model plus the
/// exact extraction notes, so distinct problems stay independently visible.
fn warn_key(model: &str, notes: &ExtractionNotes) -> String {
    format!(
        "{model}|missing={:?}|unknown={:?}|unexpected={:?}",
        notes.missing, notes.unknown_categorical, notes.unexpected_fields
    )
}

/// Extract features, run the booster, return the first output as the score.
/// `warn_throttle` (when given) rate-limits repeated extraction warnings.
pub fn predict(
    entry: &Entry,
    features: &Map<String, Value>,
    warn_throttle: Option<&WarnThrottle>,
) -> Result<PredictionOutcome, PredictError> {
    let start = Instant::now();
    let (vector, notes) = to_vector(
        &entry.manifest.feature_schema,
        entry.manifest.n_features,
        features,
    )
    .map_err(|e| PredictError::BadRequest(format!("invalid features: {}", e)))?;
    if !notes.is_empty() {
        let key = warn_key(&entry.key, &notes);
        let emit = match warn_throttle {
            Some(throttle) => crate::throttle::warn_allowed(throttle, &key),
            None => true,
        };
        if emit {
            eprintln!(
                "warn: model={} missing={:?} unknown_categories={:?} unexpected_fields={:?}",
                entry.key, notes.missing, notes.unknown_categorical, notes.unexpected_fields
            );
        }
    }
    let outputs = {
        let booster = lock_booster(entry).map_err(|e| PredictError::Internal(e.to_string()))?;
        ffi::predict(&booster, &vector, entry.manifest.n_features)
            .map_err(|e| PredictError::Internal(format!("booster predict failed: {}", e)))?
    };
    let score = outputs
        .first()
        .copied()
        .ok_or_else(|| PredictError::Internal("booster returned no outputs".to_string()))?;
    Ok(PredictionOutcome {
        score,
        model_key: entry.key.clone(),
        latency_us: start.elapsed().as_micros() as u64,
        missing_features: notes.missing,
        unknown_categories: notes.unknown_categorical,
        unexpected_fields: notes.unexpected_fields,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn warn_key_is_deterministic_and_model_scoped() {
        let notes = ExtractionNotes {
            missing: vec!["a".to_string()],
            unknown_categorical: vec!["color=green".to_string()],
            unexpected_fields: vec!["extra".to_string()],
        };
        assert_eq!(warn_key("m@v1", &notes), warn_key("m@v1", &notes));
        assert_ne!(warn_key("m@v1", &notes), warn_key("m@v2", &notes));
        assert!(warn_key("m@v1", &notes).starts_with("m@v1|"));
    }
}
