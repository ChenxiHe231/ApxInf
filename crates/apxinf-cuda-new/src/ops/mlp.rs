//! MLP building blocks: RMSNorm, SwiGLU and residual addition.
//!
//! These have a single implementation each and nothing to select between, so
//! they are direct entry points rather than registry-backed operators
//! (`doc/adding-new-kernels.md` section 6).

use apxinf_core::{DType, Result, Tensor};

use crate::ffi::abi::{mlp as abi, status};
use crate::ops::gemm::contracts::{invalid, tensor_storage};
use crate::CudaContext;

/// `y[r,c] = x[r,c] / sqrt(mean(x[r,:]^2) + epsilon) * weight[c]`
///
/// The reduction runs in f32 regardless of the BF16 storage.
pub fn rms_norm(
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
        status::check(abi::apxinf_rms_norm_bf16(
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

/// `y[r,c] = silu(fused[r,c]) * fused[r,width+c]`
///
/// `fused` is `[rows, 2*width]` with gate first, the layout one fused gate/up
/// GEMM produces. This is SwiGLU; `gemm_geglu` computes the GELU variant and
/// is not interchangeable.
pub fn swiglu(ctx: &CudaContext, fused: &Tensor, output: &Tensor) -> Result<()> {
    let fused_dims = fused.shape().dims().to_vec();
    let output_dims = output.shape().dims().to_vec();
    if fused_dims.len() != 2 || output_dims.len() != 2 {
        return Err(invalid("SwiGLU requires rank-2 tensors"));
    }
    if fused_dims[0] != output_dims[0] || fused_dims[1] != output_dims[1] * 2 {
        return Err(invalid(
            "SwiGLU input must be [rows, 2*width] for an output of [rows, width]",
        ));
    }
    let fused_buffer = tensor_storage(ctx, fused, DType::BF16, &fused_dims)?;
    let output_buffer = tensor_storage(ctx, output, DType::BF16, &output_dims)?;
    unsafe {
        status::check(abi::apxinf_swiglu_bf16(
            fused_buffer.ptr(),
            output_buffer.ptr(),
            output_dims[0] as i64,
            output_dims[1] as i64,
            ctx.stream().handle(),
        ))
    }
}

/// `accumulator += addend`, elementwise. Residual connections.
pub fn add_into(ctx: &CudaContext, addend: &Tensor, accumulator: &Tensor) -> Result<()> {
    let dims = addend.shape().dims().to_vec();
    if dims != accumulator.shape().dims() {
        return Err(invalid("residual add requires matching shapes"));
    }
    let addend_buffer = tensor_storage(ctx, addend, DType::BF16, &dims)?;
    let accumulator_buffer = tensor_storage(ctx, accumulator, DType::BF16, &dims)?;
    unsafe {
        status::check(abi::apxinf_add_bf16(
            addend_buffer.ptr(),
            accumulator_buffer.ptr(),
            dims.iter().product::<usize>() as i64,
            ctx.stream().handle(),
        ))
    }
}

/// Quantize BF16 to E4M3 against a single per-tensor scale.
///
/// The FP8 projections in a ModelOpt checkpoint carry scalar `weight_scale`
/// and `input_scale`, not the `[M]`/`[N]` vectors [`GemmQuantization::Fp8`]
/// expects. Quantizing here against `input_scale` lets the projection run as
/// `Fp8UnitScale` with `alpha = weight_scale * input_scale`, so attention and
/// GDN need no new quantization contract.
pub fn quantize_fp8_per_tensor(
    ctx: &CudaContext,
    input: &Tensor,
    output: &Tensor,
    input_scale: f32,
) -> Result<()> {
    let dims = input.shape().dims().to_vec();
    if dims != output.shape().dims() {
        return Err(invalid("FP8 quantization requires matching shapes"));
    }
    if !(input_scale > 0.0) || !input_scale.is_finite() {
        return Err(invalid("FP8 input_scale must be finite and positive"));
    }
    let source = tensor_storage(ctx, input, DType::BF16, &dims)?;
    let destination = tensor_storage(ctx, output, DType::F8E4M3, &dims)?;
    unsafe {
        status::check(abi::apxinf_quantize_fp8_per_tensor(
            source.ptr(),
            destination.ptr(),
            dims.iter().product::<usize>() as i64,
            input_scale,
            ctx.stream().handle(),
        ))
    }
}
