use apxinf_core::{DType, Error, Result, Tensor};

use crate::context::CudaContext;
use crate::ffi::abi::quantization as quantization_abi;
use crate::ops::validation::{check_cuda, gpu_ptr};

pub fn cast(ctx: &CudaContext, input: &Tensor, output: &Tensor) -> Result<()> {
    match (input.dtype(), output.dtype()) {
        (DType::F32, DType::BF16) => cast_f32_to_bf16(ctx, input, output),
        (DType::BF16, DType::F32) => cast_bf16_to_f32(ctx, input, output),
        _ => Err(Error::Other(
            "cast supports F32->BF16 and BF16->F32".into(),
        )),
    }
}

fn cast_f32_to_bf16(ctx: &CudaContext, input: &Tensor, output: &Tensor) -> Result<()> {
    if input.dtype() != DType::F32
        || output.dtype() != DType::BF16
        || input.numel() != output.numel()
    {
        return Err(Error::Other("cast f32->bf16 shape/dtype mismatch".into()));
    }
    unsafe {
        check_cuda(quantization_abi::apxinf_quantization_cast_f32_bf16(
            gpu_ptr(input)?,
            gpu_ptr(output)?,
            input.numel() as i64,
            ctx.stream().handle(),
        ))
    }
}

/// Cast a BF16 tensor to F32 (exact widening).
fn cast_bf16_to_f32(ctx: &CudaContext, input: &Tensor, output: &Tensor) -> Result<()> {
    if input.dtype() != DType::BF16
        || output.dtype() != DType::F32
        || input.numel() != output.numel()
    {
        return Err(Error::Other("cast bf16->f32 shape/dtype mismatch".into()));
    }
    unsafe {
        check_cuda(quantization_abi::apxinf_quantization_cast_bf16_f32(
            gpu_ptr(input)?,
            gpu_ptr(output)?,
            input.numel() as i64,
            ctx.stream().handle(),
        ))
    }
}
