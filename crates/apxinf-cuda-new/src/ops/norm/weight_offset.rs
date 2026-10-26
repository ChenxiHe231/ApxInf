//! RMS normalization whose checkpoint weight is zero-centered.

use apxinf_core::{DType, Result, Tensor};

use crate::ffi::abi::{norm as abi, status};
use crate::ops::gemm::contracts::{invalid, tensor_storage};
use crate::CudaContext;

/// `y = x / rms(x) * (1 + weight)`.
///
/// The dedicated entry keeps the existing reduction and BF16 rounding order
/// unchanged while placing the operator in the norm family.
pub fn rms_norm_weight_offset(
    ctx: &CudaContext,
    input: &Tensor,
    weight: &Tensor,
    output: &Tensor,
    epsilon: f32,
) -> Result<()> {
    let dims = input.shape().dims().to_vec();
    if dims.len() != 2 {
        return Err(invalid("RMSNorm requires a rank-2 tensor"));
    }
    let (rows, width) = (dims[0], dims[1]);
    let input_buffer = tensor_storage(ctx, input, DType::BF16, &dims)?;
    let weight_buffer = tensor_storage(ctx, weight, DType::BF16, &[width])?;
    let output_buffer = tensor_storage(ctx, output, DType::BF16, &dims)?;
    unsafe {
        status::check(abi::apxinf_norm_rms_weight_offset(
            input_buffer.ptr(),
            weight_buffer.ptr(),
            output_buffer.ptr(),
            rows as i64,
            width as i64,
            epsilon,
            ctx.stream().handle(),
        ))
    }
}
