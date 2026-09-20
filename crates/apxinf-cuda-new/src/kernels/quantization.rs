//! Quantization operator contracts.

use apxinf_core::{DType, Error, Result, Tensor};

use super::contracts::{gpu_ptr, make_gpu_tensor};
use crate::context::CudaContext;
use crate::ffi;
use crate::workspace::output_buffer;

pub struct DynamicFp8Tensor {
    pub values: Tensor,
    pub scales: Tensor,
}

/// Quantize an FP16 device tensor to E4M3 using a pre-calibrated scale.
pub fn quantize_f16_e4m3(ctx: &CudaContext, input: &Tensor, scale: f32) -> Result<Tensor> {
    if input.dtype() != DType::F16 {
        return Err(Error::DTypeMismatch {
            expected: DType::F16,
            got: input.dtype(),
        });
    }
    if !scale.is_finite() || scale <= 0.0 {
        return Err(Error::Other(format!("invalid FP8 scale {scale}")));
    }
    let output = output_buffer(ctx, input.numel())?;
    unsafe {
        ffi::check_cuda(ffi::apxinf_static_quantize_f16_e4m3(
            gpu_ptr(input)?,
            output.ptr(),
            input.numel() as i64,
            scale,
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)?;
    }
    Ok(make_gpu_tensor(
        input.shape().clone(),
        DType::F8E4M3,
        ctx.device_id(),
        output,
    ))
}

/// Quantize a BF16 device tensor to E4M3 using a pre-calibrated scale.
pub fn quantize_bf16_e4m3(ctx: &CudaContext, input: &Tensor, scale: f32) -> Result<Tensor> {
    if input.dtype() != DType::BF16 {
        return Err(Error::DTypeMismatch {
            expected: DType::BF16,
            got: input.dtype(),
        });
    }
    if !scale.is_finite() || scale <= 0.0 {
        return Err(Error::Other(format!("invalid FP8 scale {scale}")));
    }
    let output = output_buffer(ctx, input.numel())?;
    unsafe {
        ffi::check_cuda(ffi::apxinf_static_quantize_bf16_e4m3(
            gpu_ptr(input)?,
            output.ptr(),
            input.numel() as i64,
            scale,
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)?;
    }
    Ok(make_gpu_tensor(
        input.shape().clone(),
        DType::F8E4M3,
        ctx.device_id(),
        output,
    ))
}

/// Quantize each BF16 row independently and return E4M3 values plus FP32 row scales.
pub fn quantize_rows_bf16_e4m3(ctx: &CudaContext, input: &Tensor) -> Result<DynamicFp8Tensor> {
    let cols = input.shape().dims().get(1).copied().unwrap_or(0);
    quantize_rows_bf16_e4m3_padded(ctx, input, cols)
}

/// Quantize each BF16 row independently and append zero-valued FP8 columns.
pub fn quantize_rows_bf16_e4m3_padded(
    ctx: &CudaContext,
    input: &Tensor,
    output_cols: usize,
) -> Result<DynamicFp8Tensor> {
    if input.dtype() != DType::BF16 {
        return Err(Error::DTypeMismatch {
            expected: DType::BF16,
            got: input.dtype(),
        });
    }
    let shape = input.shape().dims();
    if shape.len() != 2 || shape[0] == 0 || shape[1] == 0 {
        return Err(Error::Other(format!(
            "dynamic FP8 row quantization expects a non-empty 2D tensor, got {shape:?}"
        )));
    }
    let (rows, cols) = (shape[0], shape[1]);
    if output_cols < cols {
        return Err(Error::Other(format!(
            "dynamic FP8 padded row width {output_cols} is smaller than input width {cols}"
        )));
    }
    let values = output_buffer(
        ctx,
        rows.checked_mul(output_cols)
            .ok_or_else(|| Error::Other("dynamic FP8 padded size overflow".into()))?,
    )?;
    let scales = output_buffer(ctx, rows * DType::F32.size_in_bytes())?;
    unsafe {
        ffi::check_cuda(ffi::apxinf_dynamic_quantize_rows_bf16_e4m3(
            gpu_ptr(input)?,
            values.ptr(),
            scales.ptr(),
            rows as i32,
            cols as i32,
            output_cols as i32,
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)?;
    }
    Ok(DynamicFp8Tensor {
        values: make_gpu_tensor(
            apxinf_core::Shape::new(vec![rows, output_cols]),
            DType::F8E4M3,
            ctx.device_id(),
            values,
        ),
        scales: make_gpu_tensor(
            apxinf_core::Shape::new(vec![rows]),
            DType::F32,
            ctx.device_id(),
            scales,
        ),
    })
}

/// Keep the leading columns of a contiguous BF16 matrix.
pub fn slice_columns_bf16(ctx: &CudaContext, input: &Tensor, output_cols: usize) -> Result<Tensor> {
    if input.dtype() != DType::BF16 {
        return Err(Error::DTypeMismatch {
            expected: DType::BF16,
            got: input.dtype(),
        });
    }
    let shape = input.shape().dims();
    if shape.len() != 2 || shape[0] == 0 || output_cols == 0 || output_cols > shape[1] {
        return Err(Error::Other(format!(
            "BF16 column slice expects [rows,input_cols] with 0 < output_cols <= input_cols, got {shape:?} -> {output_cols}"
        )));
    }
    if output_cols == shape[1] {
        return Ok(input.clone());
    }
    let (rows, input_cols) = (shape[0], shape[1]);
    let output = output_buffer(
        ctx,
        rows.checked_mul(output_cols)
            .and_then(|elements| elements.checked_mul(DType::BF16.size_in_bytes()))
            .ok_or_else(|| Error::Other("BF16 column slice size overflow".into()))?,
    )?;
    unsafe {
        ffi::check_cuda(ffi::apxinf_slice_columns_bf16(
            gpu_ptr(input)?,
            output.ptr(),
            rows as i32,
            input_cols as i32,
            output_cols as i32,
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)?;
    }
    Ok(make_gpu_tensor(
        apxinf_core::Shape::new(vec![rows, output_cols]),
        DType::BF16,
        ctx.device_id(),
        output,
    ))
}

/// Convert an FP16 GEMM result into the BF16 WallOSS residual stream.
pub fn cast_f16_bf16(ctx: &CudaContext, input: &Tensor) -> Result<Tensor> {
    if input.dtype() != DType::F16 {
        return Err(Error::DTypeMismatch {
            expected: DType::F16,
            got: input.dtype(),
        });
    }
    let output = output_buffer(ctx, input.numel() * DType::BF16.size_in_bytes())?;
    unsafe {
        ffi::check_cuda(ffi::apxinf_static_cast_f16_bf16(
            gpu_ptr(input)?,
            output.ptr(),
            input.numel() as i64,
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)?;
    }
    Ok(make_gpu_tensor(
        input.shape().clone(),
        DType::BF16,
        ctx.device_id(),
        output,
    ))
}
