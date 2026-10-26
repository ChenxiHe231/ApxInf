use apxinf_core::Result;

use super::cuda_contracts::{NormalizedCombine, NormalizedNvfp4Experts, NormalizedRoute};
use crate::ffi::abi::{moe as abi, status};
use crate::CudaContext;

pub(crate) fn route(ctx: &CudaContext, normalized: NormalizedRoute) -> Result<()> {
    let _storage = &normalized.storage;
    crate::workspace::validate_capture_target(
        ctx.device_id(),
        normalized.bindings.stream as usize,
    )?;
    unsafe {
        status::check(abi::apxinf_moe_route_topk(
            ctx.runtime(),
            &normalized.spec,
            &normalized.bindings,
        ))
    }
}

pub(crate) fn combine(ctx: &CudaContext, normalized: NormalizedCombine) -> Result<()> {
    let _storage = &normalized.storage;
    crate::workspace::validate_capture_target(
        ctx.device_id(),
        normalized.bindings.stream as usize,
    )?;
    unsafe {
        status::check(abi::apxinf_moe_combine(
            ctx.runtime(),
            &normalized.spec,
            &normalized.bindings,
        ))
    }
}

pub(crate) fn nvfp4_experts(ctx: &CudaContext, normalized: NormalizedNvfp4Experts) -> Result<()> {
    let _storage = &normalized.storage;
    crate::workspace::validate_capture_target(
        ctx.device_id(),
        normalized.bindings.stream as usize,
    )?;
    unsafe {
        status::check(abi::apxinf_moe_nvfp4_experts(
            ctx.runtime(),
            &normalized.spec,
            &normalized.bindings,
        ))
    }
}
