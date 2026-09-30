//! HTTP API (axum 0.8): model listing, prediction, admin reload.
//! `/predict` is guarded by admission control (token-bucket rate limit plus
//! an in-flight concurrency cap) so overload sheds requests with 429
//! instead of piling up.

use std::path::PathBuf;
use std::sync::{Arc, RwLock};
use std::time::Instant;

use axum::extract::State;
use axum::http::{header, HeaderMap, StatusCode};
use axum::response::{IntoResponse, Json, Response};
use axum::routing::{get, post};
use axum::Router;
use serde_json::{json, Map, Value};

use crate::ffi;
use crate::guard::{self, Admission, AdmissionDenied};
use crate::predict::{error_message, predict, PredictError};
use crate::registry::{self, LoadOptions, Registry, ResolveError};
use crate::throttle::WarnThrottle;

/// Shared application state.
pub struct AppShared {
    pub lib: Arc<ffi::Lib>,
    pub models_dir: PathBuf,
    pub registry: RwLock<Arc<Registry>>,
    pub admin_token: Option<String>,
    pub admission: Admission,
    pub warn_throttle: Arc<WarnThrottle>,
    pub strict_version: bool,
}

fn registry_snapshot(shared: &AppShared) -> Arc<Registry> {
    match shared.registry.read() {
        Ok(guard) => Arc::clone(&guard),
        Err(poisoned) => Arc::clone(&poisoned.into_inner()),
    }
}

fn error_response(status: StatusCode, message: String) -> Response {
    (status, Json(json!({ "error": message }))).into_response()
}

/// 429 with a short `Retry-After`, used for rate limiting and overload.
fn too_many_requests(message: &str) -> Response {
    (
        StatusCode::TOO_MANY_REQUESTS,
        [(header::RETRY_AFTER, "1")],
        Json(json!({ "error": message })),
    )
        .into_response()
}

fn predict_status(err: &PredictError) -> StatusCode {
    match err {
        PredictError::UnknownModel(_) => StatusCode::NOT_FOUND,
        PredictError::BadRequest(_) => StatusCode::BAD_REQUEST,
        PredictError::Internal(_) => StatusCode::INTERNAL_SERVER_ERROR,
    }
}

/// Byte equality without early exit on first difference.
fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff: u8 = 0;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

/// Check admin credentials: `None` means dev mode (main enforces loopback);
/// otherwise accept `Authorization: Bearer <t>` or `X-Admin-Token: <t>`.
fn authorize_admin(headers: &HeaderMap, expected: &Option<String>) -> Result<(), String> {
    let expected = match expected {
        None => return Ok(()),
        Some(t) => t,
    };
    let bearer = headers
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "))
        .map(str::trim);
    let custom = headers
        .get("x-admin-token")
        .and_then(|v| v.to_str().ok())
        .map(str::trim);
    match bearer.or(custom) {
        Some(p) if constant_time_eq(p.as_bytes(), expected.as_bytes()) => Ok(()),
        _ => Err("admin token missing or invalid".to_string()),
    }
}

/// Build the application router.
pub fn router(shared: Arc<AppShared>) -> Router {
    Router::new()
        .route("/models", get(list_models_handler))
        .route("/predict", post(predict_handler))
        .route("/admin/reload", post(reload_handler))
        .with_state(shared)
}

async fn list_models_handler(state: State<Arc<AppShared>>) -> Json<Value> {
    let registry = registry_snapshot(&state);
    Json(json!({ "models": registry::list_models(&registry) }))
}

async fn predict_handler(state: State<Arc<AppShared>>, body: String) -> Response {
    let inflight = match guard::try_admit(&state.admission, Instant::now()) {
        Ok(g) => g,
        Err(AdmissionDenied::RateLimited) => return too_many_requests("rate limit exceeded"),
        Err(AdmissionDenied::Overloaded) => {
            return too_many_requests("too many concurrent predict requests")
        }
    };
    let parsed: Value = match serde_json::from_str(&body) {
        Ok(v) => v,
        Err(e) => {
            return error_response(StatusCode::BAD_REQUEST, format!("invalid JSON body: {e}"))
        }
    };
    let request = match parsed.as_object() {
        Some(obj) => obj,
        None => {
            return error_response(
                StatusCode::BAD_REQUEST,
                "request body must be a JSON object".to_string(),
            )
        }
    };
    let model_ref = match request.get("model").and_then(Value::as_str) {
        Some(m) if !m.is_empty() => m.to_string(),
        _ => {
            return error_response(
                StatusCode::BAD_REQUEST,
                "field \"model\" must be a non-empty string".to_string(),
            )
        }
    };
    let features: Map<String, Value> = match request.get("features").and_then(Value::as_object) {
        Some(f) => f.clone(),
        None => {
            return error_response(
                StatusCode::BAD_REQUEST,
                "field \"features\" must be a JSON object".to_string(),
            )
        }
    };

    let registry = registry_snapshot(&state);
    let entry = match registry::resolve(&registry, &model_ref) {
        Ok(entry) => entry,
        Err(ResolveError::UnknownModel(m)) => return error_response(StatusCode::NOT_FOUND, m),
        Err(ResolveError::InvalidReference(m)) => {
            return error_response(StatusCode::BAD_REQUEST, m)
        }
    };
    let entry = Arc::clone(entry);
    let throttle = Arc::clone(&state.warn_throttle);
    // The in-flight slot stays held for the whole request and is released
    // once the blocking prediction has finished.
    let started = {
        let _inflight = inflight;
        tokio::task::spawn_blocking(move || predict(&entry, &features, Some(&throttle))).await
    };
    match started {
        Ok(Ok(outcome)) => (StatusCode::OK, Json(json!(outcome))).into_response(),
        Ok(Err(err)) => error_response(predict_status(&err), error_message(&err)),
        Err(e) => error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("predict task failed: {e}"),
        ),
    }
}

async fn reload_handler(state: State<Arc<AppShared>>, headers: HeaderMap) -> Response {
    if let Err(e) = authorize_admin(&headers, &state.admin_token) {
        return error_response(StatusCode::UNAUTHORIZED, format!("unauthorized: {e}"));
    }
    let lib = state.lib.clone();
    let models_dir = state.models_dir.clone();
    let options = LoadOptions {
        strict_xgboost_version: state.strict_version,
    };
    let loaded =
        tokio::task::spawn_blocking(move || registry::load_registry(&lib, &models_dir, &options))
            .await;
    match loaded {
        Ok(Ok(new_registry)) => {
            let count = registry::model_count(&new_registry);
            let mut guard = state
                .registry
                .write()
                .unwrap_or_else(|poisoned| poisoned.into_inner());
            *guard = Arc::new(new_registry);
            (
                StatusCode::OK,
                Json(json!({ "status": "ok", "models": count })),
            )
                .into_response()
        }
        Ok(Err(e)) => error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("reload failed, keeping previous registry: {e}"),
        ),
        Err(e) => error_response(
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("reload task failed: {e}"),
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn header(key: &'static str, value: &str) -> HeaderMap {
        let mut headers = HeaderMap::new();
        headers.insert(key, value.parse().expect("valid header value"));
        headers
    }

    #[test]
    fn constant_time_eq_basics() {
        assert!(constant_time_eq(b"secret", b"secret"));
        assert!(constant_time_eq(b"", b""));
        assert!(!constant_time_eq(b"secret", b"secreT"));
        assert!(!constant_time_eq(b"secret", b"secre"));
        assert!(!constant_time_eq(b"secret", b"secrets"));
    }

    #[test]
    fn authorize_admin_open_when_no_token_configured() {
        assert!(authorize_admin(&HeaderMap::new(), &None).is_ok());
        assert!(authorize_admin(&header("authorization", "Bearer x"), &None).is_ok());
    }

    #[test]
    fn authorize_admin_accepts_bearer_and_custom_header() {
        let token = Some("test-token-123".to_string());
        let headers = header("authorization", "Bearer test-token-123");
        assert!(authorize_admin(&headers, &token).is_ok());

        // surrounding whitespace is trimmed
        let headers = header("x-admin-token", " test-token-123 ");
        assert!(authorize_admin(&headers, &token).is_ok());
    }

    #[test]
    fn authorize_admin_rejects_wrong_or_missing_token() {
        let token = Some("test-token-123".to_string());
        let err = authorize_admin(&HeaderMap::new(), &token).unwrap_err();
        assert_eq!(err, "admin token missing or invalid");

        let headers = header("authorization", "Bearer wrong");
        assert!(authorize_admin(&headers, &token).is_err());

        // wrong scheme is not accepted
        let headers = header("authorization", "Basic test-token-123");
        assert!(authorize_admin(&headers, &token).is_err());
    }
}
