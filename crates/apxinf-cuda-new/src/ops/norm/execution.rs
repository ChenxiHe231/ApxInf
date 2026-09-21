use std::ffi::CString;

use apxinf_core::{Error, Result};

use super::contracts::Normalized;
use crate::ffi::abi::{norm as abi, status};
use crate::CudaContext;

fn cache_string(normalized: &Normalized) -> Result<Option<CString>> {
    normalized
        .policy
        .cache_dir
        .as_ref()
        .map(|path| CString::new(path.as_str()))
        .transpose()
        .map_err(|_| Error::Other("cache path contains NUL".into()))
}

fn policy_abi(normalized: &Normalized, cache: &Option<CString>) -> abi::Policy {
    abi::Policy {
        workspace_limit: normalized.policy.workspace_limit as u64,
        online_tune: normalized.policy.online_tune as u32,
        allow_fallback: normalized.policy.allow_fallback as u32,
        graph_safe: normalized.policy.graph_safe as u32,
        deterministic: normalized.policy.deterministic as u32,
        cache_dir: cache.as_ref().map_or(std::ptr::null(), |path| path.as_ptr()),
    }
}

pub(crate) fn execute(ctx: &CudaContext, normalized: Normalized) -> Result<()> {
    let _storage = &normalized.storage;
    crate::workspace::validate_capture_target(
        ctx.device_id(),
        normalized.bindings.stream as usize,
    )?;
    ctx.stream().set_current_device().map_err(Error::Cuda)?;
    let cache = cache_string(&normalized)?;
    let policy = policy_abi(&normalized, &cache);
    unsafe {
        status::check(abi::apxinf_norm_launch(
            ctx.runtime(),
            &normalized.spec,
            &policy,
            &normalized.bindings,
        ))
    }
}

#[cfg(test)]
pub(crate) fn validate_candidates(
    ctx: &CudaContext,
    normalized: &Normalized,
    expected: &[f32],
) -> Result<()> {
    let cache = cache_string(normalized)?;
    let policy = policy_abi(normalized, &cache);
    unsafe {
        status::check(abi::apxinf_norm_test_validate_candidates(
            ctx.runtime(),
            &normalized.spec,
            &policy,
            &normalized.bindings,
            expected.as_ptr(),
            expected.len() as u64,
        ))
    }
}
