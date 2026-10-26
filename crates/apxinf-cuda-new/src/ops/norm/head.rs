//! Per-head normalization helpers.

use apxinf_core::{DType, Result, Tensor};

use crate::ffi::abi::{norm as abi, status};
use crate::ops::gemm::contracts::{invalid, tensor_storage};
use crate::CudaContext;

/// Per-head RMSNorm in place over `[rows, head_dim]`, for q_norm and k_norm.
pub fn head_rms_norm(
    ctx: &CudaContext,
    data: &Tensor,
    weight: &Tensor,
    epsilon: f32,
) -> Result<()> {
    let dims = data.shape().dims().to_vec();
    if dims.len() != 2 {
        return Err(invalid("head RMSNorm expects [rows, head_dim]"));
    }
    let data_buffer = tensor_storage(ctx, data, DType::BF16, &dims)?;
    let weight_buffer = tensor_storage(ctx, weight, DType::BF16, &[dims[1]])?;
    unsafe {
        status::check(abi::apxinf_norm_head_rms(
            data_buffer.ptr(),
            weight_buffer.ptr(),
            dims[0] as i64,
            dims[1] as i64,
            epsilon,
            ctx.stream().handle(),
        ))
    }
}
