//! Black-box contracts for the reusable native backend framework.

use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int};
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

unsafe extern "C" {
    fn apxinf_framework_test_registry_contract(error: *mut c_char, capacity: usize) -> c_int;
    fn apxinf_framework_test_recipe_db_contract(
        directory: *const c_char,
        error: *mut c_char,
        capacity: usize,
    ) -> c_int;
    fn apxinf_framework_test_autotune_contract(
        graph_safe: c_int,
        error: *mut c_char,
        capacity: usize,
    ) -> c_int;
}

fn run_contract(name: &str, invoke: impl FnOnce(*mut c_char, usize) -> c_int) {
    let mut error = vec![0 as c_char; 4096];
    let passed = invoke(error.as_mut_ptr(), error.len());
    let detail = unsafe { CStr::from_ptr(error.as_ptr()) }.to_string_lossy();
    assert_eq!(passed, 1, "{name} failed: {detail}");
}

struct TestDirectory(PathBuf);

impl TestDirectory {
    fn new() -> Self {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock predates UNIX epoch")
            .as_nanos();
        let path = std::env::temp_dir().join(format!(
            "apxinf-framework-recipe-contract-{}-{nonce}",
            std::process::id()
        ));
        std::fs::create_dir(&path)
            .unwrap_or_else(|error| panic!("create {}: {error}", path.display()));
        Self(path)
    }
}

impl Drop for TestDirectory {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

#[test]
fn registry_requires_the_exact_identity_triple() {
    run_contract("registry exact identity", |error, capacity| unsafe {
        apxinf_framework_test_registry_contract(error, capacity)
    });
}

#[test]
fn recipe_db_round_trips_updates_and_rejects_invalid_reads() {
    let directory = TestDirectory::new();
    let path = CString::new(directory.0.to_string_lossy().as_bytes()).unwrap();
    run_contract("recipe DB persistence", |error, capacity| unsafe {
        apxinf_framework_test_recipe_db_contract(path.as_ptr(), error, capacity)
    });
}

#[test]
fn autotune_filters_failures_and_times_every_valid_configuration() {
    run_contract("CUDA autotune selection", |error, capacity| unsafe {
        apxinf_framework_test_autotune_contract(0, error, capacity)
    });
}

#[test]
fn graph_safe_autotune_captures_the_winner() {
    run_contract("CUDA Graph-safe autotune", |error, capacity| unsafe {
        apxinf_framework_test_autotune_contract(1, error, capacity)
    });
}
