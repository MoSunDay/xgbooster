//! HTTP server entry point.

use std::net::SocketAddr;
use std::path::PathBuf;
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

use anyhow::{anyhow, bail, Context, Result};

use xgbooster_infer::{ffi, guard, http, registry, throttle};

const MANIFEST_DIR: &str = env!("CARGO_MANIFEST_DIR");

struct Config {
    models_dir: PathBuf,
    lib_path: PathBuf,
    addr: SocketAddr,
    admin_token: Option<String>,
    rate_limit_rps: Option<f64>,
    rate_burst: f64,
    max_inflight: Option<usize>,
    strict_version: bool,
}

fn flag_value(args: &[String], i: usize) -> Result<&str> {
    args.get(i)
        .map(String::as_str)
        .ok_or_else(|| anyhow!("missing value for {}", args[i - 1]))
}

/// Raw environment value; unset or non-UTF-8 variables read as empty.
fn env_raw(name: &str) -> String {
    std::env::var(name).unwrap_or_default()
}

/// Admin token from `XGBOOSTER_ADMIN_TOKEN`; empty string counts as unset.
fn admin_token_from_env() -> Option<String> {
    match std::env::var("XGBOOSTER_ADMIN_TOKEN") {
        Ok(v) if !v.is_empty() => Some(v),
        _ => None,
    }
}

/// Admission settings from the environment.
///
/// - `XGBOOSTER_RATE_LIMIT_RPS`: sustained `/predict` rate limit in
///   requests per second; empty or invalid disables rate limiting.
/// - `XGBOOSTER_RATE_BURST`: token-bucket capacity for the rate limit
///   (clamped to at least 1); only applies when a rate limit is set,
///   otherwise it defaults along with it.
/// - `XGBOOSTER_MAX_INFLIGHT`: hard cap on concurrent `/predict` requests;
///   empty or invalid disables the cap.
/// - `XGBOOSTER_STRICT_VERSION`: when truthy ("1"/"true"/"yes"/"on"),
///   refuse to load artifacts lacking version information.
fn admission_from_env() -> (Option<f64>, f64, Option<usize>, bool) {
    let rate_limit_rps = guard::parse_positive_f64(&env_raw("XGBOOSTER_RATE_LIMIT_RPS"));
    let rate_burst = match guard::parse_positive_f64(&env_raw("XGBOOSTER_RATE_BURST")) {
        Some(v) => v.max(1.0),
        None => rate_limit_rps.map(|r| r.max(1.0)).unwrap_or(1.0),
    };
    let max_inflight = guard::parse_max(&env_raw("XGBOOSTER_MAX_INFLIGHT"));
    let strict_version = guard::parse_bool_flag(&env_raw("XGBOOSTER_STRICT_VERSION"));
    (rate_limit_rps, rate_burst, max_inflight, strict_version)
}

fn parse_args(args: &[String]) -> Result<Config> {
    let mut models_dir = None;
    let mut lib_path = None;
    let mut addr = None;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--models-dir" => models_dir = Some(PathBuf::from(flag_value(args, i + 1)?)),
            "--lib" => lib_path = Some(PathBuf::from(flag_value(args, i + 1)?)),
            "--addr" => addr = Some(flag_value(args, i + 1)?.parse()?),
            other => return Err(anyhow!("unknown argument: {other}")),
        }
        i += 2;
    }
    let (rate_limit_rps, rate_burst, max_inflight, strict_version) = admission_from_env();
    Ok(Config {
        models_dir: models_dir.unwrap_or_else(|| PathBuf::from(MANIFEST_DIR).join("../models")),
        lib_path: lib_path.unwrap_or_else(|| PathBuf::from(MANIFEST_DIR).join("lib/libxgboost.so")),
        addr: addr.unwrap_or_else(|| "127.0.0.1:8080".parse().expect("default addr")),
        admin_token: admin_token_from_env(),
        rate_limit_rps,
        rate_burst,
        max_inflight,
        strict_version,
    })
}

#[tokio::main]
async fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let config = parse_args(&args)?;
    if !config.addr.ip().is_loopback() {
        let mut missing = Vec::new();
        if config.admin_token.is_none() {
            missing.push("XGBOOSTER_ADMIN_TOKEN");
        }
        if config.rate_limit_rps.is_none() {
            missing.push("XGBOOSTER_RATE_LIMIT_RPS");
        }
        if config.max_inflight.is_none() {
            missing.push("XGBOOSTER_MAX_INFLIGHT");
        }
        if !missing.is_empty() {
            bail!(
                "refusing to bind non-loopback address {} without {}",
                config.addr,
                missing.join(" / ")
            );
        }
    }
    println!("loading xgboost library: {}", config.lib_path.display());
    let lib = Arc::new(ffi::load_lib(&config.lib_path)?);
    if let Some((major, minor, patch)) = ffi::library_version(&lib) {
        println!("xgboost runtime: {major}.{minor}.{patch}");
    }
    let load_options = registry::LoadOptions {
        strict_xgboost_version: config.strict_version,
    };
    let registry = registry::load_registry(&lib, &config.models_dir, &load_options)
        .with_context(|| format!("failed to load models from {}", config.models_dir.display()))?;
    let keys = registry::list_models(&registry)
        .iter()
        .map(|m| m.key.clone())
        .collect::<Vec<_>>()
        .join(", ");
    println!(
        "serving {} model(s) [{}] on {}",
        registry::model_count(&registry),
        keys,
        config.addr
    );
    let rate_limit_rps = config.rate_limit_rps;
    let rate_burst = config.rate_burst;
    let max_inflight = config.max_inflight;
    let strict_version = config.strict_version;
    let shared = Arc::new(http::AppShared {
        lib,
        models_dir: config.models_dir.clone(),
        registry: RwLock::new(Arc::new(registry)),
        admin_token: config.admin_token.clone(),
        admission: guard::admission_from(rate_limit_rps, rate_burst, max_inflight, Instant::now()),
        warn_throttle: Arc::new(throttle::new_throttle(Duration::from_secs(60))),
        strict_version,
    });
    let listener = tokio::net::TcpListener::bind(config.addr).await?;
    println!(
        "admin auth: {}",
        match config.admin_token {
            Some(_) => "enabled".to_string(),
            None => "disabled (loopback-only)".to_string(),
        }
    );
    println!(
        "admission: rate_limit_rps={} burst={:.0} max_inflight={} | version gate strict={}",
        rate_limit_rps
            .map(|v| v.to_string())
            .unwrap_or_else(|| "off".to_string()),
        rate_burst,
        max_inflight
            .map(|v| v.to_string())
            .unwrap_or_else(|| "off".to_string()),
        strict_version
    );
    axum::serve(listener, http::router(shared)).await?;
    Ok(())
}
