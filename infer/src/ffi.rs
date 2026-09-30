//! All unsafe FFI against the XGBoost C shared library lives here.
//!
//! Function pointers are resolved once at load time; the loaded library
//! stays alive for as long as `Lib` is alive. Handle wrappers free the
//! underlying XGBoost resources on drop.

use std::ffi::{c_char, c_int, c_uint, c_void, CStr, CString};
use std::os::unix::ffi::OsStrExt;
use std::path::Path;
use std::sync::Arc;

use anyhow::{anyhow, Context, Result};

type BoosterHandle = *mut c_void;
type DMatrixHandle = *mut c_void;
type BstUlong = u64;
type BstFloat = f32;

type BoosterCreateFn =
    unsafe extern "C" fn(*const DMatrixHandle, BstUlong, *mut BoosterHandle) -> c_int;
type BoosterFreeFn = unsafe extern "C" fn(BoosterHandle) -> c_int;
type LoadModelFn = unsafe extern "C" fn(BoosterHandle, *const c_char) -> c_int;
type PredictFn = unsafe extern "C" fn(
    BoosterHandle,
    DMatrixHandle,
    c_int,
    c_uint,
    c_int,
    *mut BstUlong,
    *mut *const BstFloat,
) -> c_int;
type DMatCreateFn = unsafe extern "C" fn(
    *const BstFloat,
    BstUlong,
    BstUlong,
    BstFloat,
    *mut DMatrixHandle,
) -> c_int;
type DMatFreeFn = unsafe extern "C" fn(DMatrixHandle) -> c_int;
type LastErrorFn = unsafe extern "C" fn() -> *const c_char;
type VersionFn = unsafe extern "C" fn(*mut c_int, *mut c_int, *mut c_int);

struct Fns {
    booster_create: BoosterCreateFn,
    booster_free: BoosterFreeFn,
    booster_load_model: LoadModelFn,
    booster_predict: PredictFn,
    dmatrix_create_from_mat: DMatCreateFn,
    dmatrix_free: DMatFreeFn,
    get_last_error: LastErrorFn,
    xgboost_version: Option<VersionFn>,
}

/// Loaded XGBoost shared library plus resolved C API entry points.
pub struct Lib {
    #[allow(dead_code)]
    library: libloading::Library,
    #[allow(dead_code)]
    preloads: Vec<libloading::os::unix::Library>,
    fns: Fns,
}

/// Owned booster handle; freed by `XGBoosterFree` on drop.
///
/// Holds a clone of the owning `Arc<Lib>` so the resolved function
/// pointers outlive the handle.
pub struct XgbBooster {
    raw: BoosterHandle,
    lib: Arc<Lib>,
}

impl Drop for XgbBooster {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe { (self.lib.fns.booster_free)(self.raw) };
        }
    }
}

// The booster handle is used from multiple threads behind a mutex.
unsafe impl Send for XgbBooster {}

/// Owned DMatrix handle; freed by `XGDMatrixFree` on drop.
pub struct XgbDMatrix {
    raw: DMatrixHandle,
    lib: Arc<Lib>,
}

impl Drop for XgbDMatrix {
    fn drop(&mut self) {
        if !self.raw.is_null() {
            unsafe { (self.lib.fns.dmatrix_free)(self.raw) };
        }
    }
}

fn last_error(fns: &Fns) -> String {
    let ptr = unsafe { (fns.get_last_error)() };
    if ptr.is_null() {
        return "unknown XGBoost error".to_string();
    }
    unsafe { CStr::from_ptr(ptr) }
        .to_string_lossy()
        .into_owned()
}

fn require_symbol<T>(library: &libloading::Library, name: &[u8]) -> Result<T>
where
    T: Copy,
{
    let symbol = unsafe { library.get::<T>(name) }.with_context(|| {
        format!(
            "symbol {} not found in xgboost library",
            String::from_utf8_lossy(&name[..name.len() - 1])
        )
    })?;
    Ok(*symbol)
}

fn dlopen_preload(path: &Path) -> Option<libloading::os::unix::Library> {
    use libloading::os::unix::{Library as UnixLibrary, RTLD_GLOBAL, RTLD_LAZY};
    unsafe { UnixLibrary::open(Some(path.as_os_str()), RTLD_LAZY | RTLD_GLOBAL).ok() }
}

/// Load the XGBoost shared library from `path` (must be absolute).
///
/// Sibling shared libraries in the same directory are dlopened first so
/// bundled dependencies resolve; failures on those preloads are ignored.
pub fn load_lib(path: &Path) -> Result<Lib> {
    if !path.is_file() {
        return Err(anyhow!(
            "xgboost shared library not found at {} (absolute path required)",
            path.display()
        ));
    }
    let abs = std::fs::canonicalize(path)
        .with_context(|| format!("failed to canonicalize {}", path.display()))?;
    let dir = abs
        .parent()
        .ok_or_else(|| anyhow!("library path has no parent directory: {}", abs.display()))?;

    let mut preloads = Vec::new();
    if let Ok(entries) = std::fs::read_dir(dir) {
        let mut names: Vec<String> = entries
            .filter_map(|e| e.ok())
            .filter(|e| e.path() != abs)
            .filter(|e| {
                let n = e.file_name();
                let n = n.to_string_lossy();
                e.path().is_file() && (n.ends_with(".so") || n.contains(".so."))
            })
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .collect();
        names.sort();
        for name in names {
            if let Some(lib) = dlopen_preload(&dir.join(&name)) {
                preloads.push(lib);
            }
        }
    }

    // SAFETY: dlopen of an absolute path resolved just above.
    let library = unsafe { libloading::Library::new(&abs) }
        .with_context(|| format!("dlopen failed for {}", abs.display()))?;
    let xgboost_version: Option<VersionFn> =
        unsafe { library.get::<VersionFn>(b"XGBoostVersion\0") }
            .ok()
            .map(|s| *s);
    let fns = Fns {
        booster_create: require_symbol(&library, b"XGBoosterCreate\0")?,
        booster_free: require_symbol(&library, b"XGBoosterFree\0")?,
        booster_load_model: require_symbol(&library, b"XGBoosterLoadModel\0")?,
        booster_predict: require_symbol(&library, b"XGBoosterPredict\0")?,
        dmatrix_create_from_mat: require_symbol(&library, b"XGDMatrixCreateFromMat\0")?,
        dmatrix_free: require_symbol(&library, b"XGDMatrixFree\0")?,
        get_last_error: require_symbol(&library, b"XGBGetLastError\0")?,
        xgboost_version,
    };
    Ok(Lib {
        library,
        preloads,
        fns,
    })
}

/// Runtime XGBoost version reported by the loaded library, if exported.
pub fn library_version(lib: &Lib) -> Option<(i32, i32, i32)> {
    let f = lib.fns.xgboost_version?;
    let (mut major, mut minor, mut patch) = (0i32, 0i32, 0i32);
    unsafe { f(&mut major, &mut minor, &mut patch) };
    Some((major, minor, patch))
}

/// Create a booster and load a model file (`.ubj` or `.json`) into it.
pub fn booster_from_model_file(lib: &Arc<Lib>, path: &Path) -> Result<XgbBooster> {
    let cpath = CString::new(path.as_os_str().as_bytes())
        .with_context(|| format!("invalid model path {}", path.display()))?;
    let mut handle: BoosterHandle = std::ptr::null_mut();
    unsafe {
        let rc = (lib.fns.booster_create)(std::ptr::null(), 0, &mut handle);
        if rc != 0 || handle.is_null() {
            return Err(anyhow!("XGBoosterCreate failed: {}", last_error(&lib.fns)));
        }
        let rc = (lib.fns.booster_load_model)(handle, cpath.as_ptr());
        if rc != 0 {
            let msg = last_error(&lib.fns);
            (lib.fns.booster_free)(handle);
            return Err(anyhow!(
                "XGBoosterLoadModel failed for {}: {}",
                path.display(),
                msg
            ));
        }
    }
    Ok(XgbBooster {
        raw: handle,
        lib: Arc::clone(lib),
    })
}

fn dmatrix_from_row(lib: &Arc<Lib>, row: &[f32]) -> Result<XgbDMatrix> {
    let mut handle: DMatrixHandle = std::ptr::null_mut();
    unsafe {
        let rc = (lib.fns.dmatrix_create_from_mat)(
            row.as_ptr(),
            1,
            row.len() as BstUlong,
            f32::NAN,
            &mut handle,
        );
        if rc != 0 || handle.is_null() {
            return Err(anyhow!(
                "XGDMatrixCreateFromMat failed: {}",
                last_error(&lib.fns)
            ));
        }
    }
    Ok(XgbDMatrix {
        raw: handle,
        lib: Arc::clone(lib),
    })
}

/// Predict a single 1 x n_features row; returns the raw margin/probability
/// outputs from the booster.
pub fn predict(booster: &XgbBooster, row: &[f32], n_features: usize) -> Result<Vec<f32>> {
    if row.len() != n_features {
        return Err(anyhow!(
            "feature vector length {} does not match model n_features {}",
            row.len(),
            n_features
        ));
    }
    let lib = &booster.lib;
    let dmat = dmatrix_from_row(lib, row)?;
    let mut out_len: BstUlong = 0;
    let mut out_ptr: *const BstFloat = std::ptr::null();
    unsafe {
        let rc = (lib.fns.booster_predict)(
            booster.raw,
            dmat.raw,
            0, // option_mask
            0, // ntree_limit
            0, // training (xgboost >= 3.0; was strict_shape in 2.x)
            &mut out_len,
            &mut out_ptr,
        );
        if rc != 0 {
            return Err(anyhow!("XGBoosterPredict failed: {}", last_error(&lib.fns)));
        }
        if out_ptr.is_null() {
            return Err(anyhow!("XGBoosterPredict returned a null result pointer"));
        }
        // Copy before `dmat` is dropped: the output buffer is owned by the
        // DMatrix prediction cache.
        Ok(std::slice::from_raw_parts(out_ptr, out_len as usize).to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lib_path() -> std::path::PathBuf {
        std::env::var("XGBOOSTER_LIB")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|_| {
                std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("lib/libxgboost.so")
            })
    }

    #[test]
    fn load_lib_or_skip() {
        let path = lib_path();
        if !path.is_file() {
            eprintln!("SKIP: {} not present yet", path.display());
            return;
        }
        let lib = load_lib(&path).expect("library loads");
        if let Some((major, minor, patch)) = library_version(&lib) {
            eprintln!("xgboost runtime version: {}.{}.{}", major, minor, patch);
        } else {
            eprintln!("XGBoostVersion symbol not exported; continuing");
        }
    }

    #[test]
    fn load_lib_missing_file_errors() {
        let err = match load_lib(Path::new("/nonexistent/definitely-missing.so")) {
            Err(e) => e,
            Ok(_) => panic!("loading a missing file must fail"),
        };
        assert!(err.to_string().contains("not found"));
    }
}
