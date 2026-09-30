//! XGBoost inference side (Rust).
//!
//! Loads model artifacts produced by the Python training pipeline
//! (`models/<name>/<version>/{model.ubj, manifest.json}`) and serves
//! low-latency predictions. Pure functional style: free functions plus
//! plain data records only. All unsafe FFI code is isolated in `ffi`.

pub mod features;
pub mod ffi;
pub mod guard;
pub mod http;
pub mod predict;
pub mod registry;
pub mod throttle;
