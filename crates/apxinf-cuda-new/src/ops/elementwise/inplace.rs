//! Caller-owned elementwise operations.

use apxinf_core::{DType, Result, Tensor};

use crate::ffi::abi::{pointwise as abi, status};
use crate::ops::gemm::contracts::{invalid, tensor_storage};
use crate::CudaContext;

/// `accumulator += addend`, elementwise BF16.
pub fn add_into(ctx: &CudaContext, addend: &Tensor, accumulator: &Tensor) -> Result<()> {
    let dims = addend.shape().dims().to_vec();
    if dims != accumulator.shape().dims() {
        return Err(invalid("residual add requires matching shapes"));
    }
    let addend_buffer = tensor_storage(ctx, addend, DType::BF16, &dims)?;
    let accumulator_buffer = tensor_storage(ctx, accumulator, DType::BF16, &dims)?;
    unsafe {
        status::check(abi::apxinf_pointwise_add_into(
            addend_buffer.ptr(),
            accumulator_buffer.ptr(),
            dims.iter().product::<usize>() as i64,
            ctx.stream().handle(),
        ))
    }
}
