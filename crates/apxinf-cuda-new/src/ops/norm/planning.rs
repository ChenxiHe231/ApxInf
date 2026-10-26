use apxinf_core::{DType, Error, Result, Shape, Tensor};

use crate::context::CudaContext;
use crate::ffi::abi::norm as norm_abi;
use crate::ops::validation::{
    bf16_output, check_cuda, expect_bf16, gpu_ptr, make_gpu_tensor, matrix_shape,
};

/// Gated RMSNorm with SiLU gating.
pub fn gated_rms_silu(
    ctx: &CudaContext,
    x: &Tensor,
    z: &Tensor,
    z_col_offset: usize,
    z_heads: usize,
    weight: &Tensor,
    eps: f32,
) -> Result<Tensor> {
    let (rows, cols) = matrix_shape(x, "gated rms silu")?;
    let (z_rows, z_width) = matrix_shape(z, "gated rms silu")?;
    if rows == 0
        || z_heads == 0
        || rows % z_heads != 0
        || z_rows != rows / z_heads
        || z_col_offset + z_heads * cols > z_width
        || weight.shape().dims() != [cols]
        || !eps.is_finite()
        || eps <= 0.0
    {
        return Err(Error::Other("gated rms silu shape mismatch".into()));
    }
    for tensor in [x, z, weight] {
        expect_bf16(tensor, "gated rms silu")?;
    }
    let output = bf16_output(ctx, rows, cols)?;
    let z_ptr = gpu_ptr(z)?;
    unsafe {
        check_cuda(norm_abi::apxinf_norm_gated_rms_silu_bf16(
            gpu_ptr(x)?,
            z_ptr,
            gpu_ptr(weight)?,
            output.ptr(),
            rows as i32,
            cols as i32,
            z_heads as i32,
            z_width as i64,
            z_col_offset as i64,
            eps,
            ctx.stream().handle(),
        ))
    }?;
    Ok(make_gpu_tensor(
        Shape::new(vec![rows, cols]),
        DType::BF16,
        ctx.device_id(),
        output,
    ))
}

/// RMSNorm with zero-init (1 + weight) semantics, fp32 compute, BF16 storage.
/// `sum = a + b` and `rms_norm_plus1(sum)` in one launch, returning both.
///
/// The pair it replaces is the tail of every text block: the residual add,
/// then the norm at the head of what follows. Returns the sum as well as the
/// normed rows because the sum is the next block's residual.
pub fn add_rms_norm_plus1(
    ctx: &CudaContext,
    a: &Tensor,
    b: &Tensor,
    weight: &Tensor,
    eps: f32,
) -> Result<(Tensor, Tensor)> {
    let (rows, cols) = matrix_shape(a, "add rms norm plus1")?;
    if rows == 0
        || cols == 0
        || b.shape().dims() != a.shape().dims()
        || weight.shape().dims() != [cols]
        || !eps.is_finite()
        || eps <= 0.0
    {
        return Err(Error::Other("add rms norm plus1 shape mismatch".into()));
    }
    expect_bf16(a, "add rms norm plus1")?;
    expect_bf16(b, "add rms norm plus1")?;
    expect_bf16(weight, "add rms norm plus1")?;
    let sum = bf16_output(ctx, rows, cols)?;
    let output = bf16_output(ctx, rows, cols)?;
    unsafe {
        check_cuda(norm_abi::apxinf_norm_add_rms_norm_plus1_bf16(
            gpu_ptr(a)?,
            gpu_ptr(b)?,
            gpu_ptr(weight)?,
            sum.ptr(),
            output.ptr(),
            rows as i32,
            cols as i32,
            eps,
            ctx.stream().handle(),
        ))
    }?;
    let shape = || Shape::new(vec![rows, cols]);
    Ok((
        make_gpu_tensor(shape(), DType::BF16, ctx.device_id(), sum),
        make_gpu_tensor(shape(), DType::BF16, ctx.device_id(), output),
    ))
}

pub fn rms_norm_plus1(
    ctx: &CudaContext,
    input: &Tensor,
    weight: &Tensor,
    eps: f32,
) -> Result<Tensor> {
    let (rows, cols) = matrix_shape(input, "rms norm plus1")?;
    if rows == 0 || cols == 0 || weight.shape().dims() != [cols] || !eps.is_finite() || eps <= 0.0 {
        return Err(Error::Other("rms norm plus1 shape mismatch".into()));
    }
    expect_bf16(input, "rms norm plus1")?;
    expect_bf16(weight, "rms norm plus1")?;
    let output = bf16_output(ctx, rows, cols)?;
    unsafe {
        check_cuda(norm_abi::apxinf_norm_rms_norm_plus1_bf16(
            gpu_ptr(input)?,
            gpu_ptr(weight)?,
            output.ptr(),
            rows as i32,
            cols as i32,
            eps,
            ctx.stream().handle(),
        ))
    }?;
    Ok(make_gpu_tensor(
        Shape::new(vec![rows, cols]),
        DType::BF16,
        ctx.device_id(),
        output,
    ))
}

/// adaLN normalization with RMSNorm affine parameters.
pub fn adaln_rms_norm(
    ctx: &CudaContext,
    x: &Tensor,
    weight: &Tensor,
    scale: &Tensor,
    shift: &Tensor,
    eps: f32,
) -> Result<Tensor> {
    let (rows, cols) = matrix_shape(x, "adaln rms norm")?;
    if rows == 0
        || weight.shape().dims() != [cols]
        || scale.shape().dims() != [cols]
        || shift.shape().dims() != [cols]
        || !eps.is_finite()
        || eps <= 0.0
    {
        return Err(Error::Other("adaln rms norm shape mismatch".into()));
    }
    for tensor in [x, weight, scale, shift] {
        expect_bf16(tensor, "adaln rms norm")?;
    }
    let output = bf16_output(ctx, rows, cols)?;
    unsafe {
        check_cuda(norm_abi::apxinf_norm_adaln_rms_norm_bf16(
            gpu_ptr(x)?,
            gpu_ptr(weight)?,
            gpu_ptr(scale)?,
            gpu_ptr(shift)?,
            output.ptr(),
            rows as i32,
            cols as i32,
            eps,
            ctx.stream().handle(),
        ))
    }?;
    Ok(make_gpu_tensor(
        Shape::new(vec![rows, cols]),
        DType::BF16,
        ctx.device_id(),
        output,
    ))
}

/// Apply the adaLN gate and add the residual.
pub fn adaln_gate_residual(
    ctx: &CudaContext,
    proj: &Tensor,
    residual: &Tensor,
    gate: &Tensor,
) -> Result<Tensor> {
    let (rows, cols) = matrix_shape(proj, "adaln gate residual")?;
    if rows == 0 || residual.shape().dims() != [rows, cols] || gate.shape().dims() != [cols] {
        return Err(Error::Other("adaln gate residual shape mismatch".into()));
    }
    for tensor in [proj, residual, gate] {
        expect_bf16(tensor, "adaln gate residual")?;
    }
    let output = bf16_output(ctx, rows, cols)?;
    unsafe {
        check_cuda(norm_abi::apxinf_norm_adaln_gate_residual_bf16(
            gpu_ptr(proj)?,
            gpu_ptr(residual)?,
            gpu_ptr(gate)?,
            output.ptr(),
            (rows * cols) as i64,
            cols as i32,
            ctx.stream().handle(),
        ))
    }?;
    Ok(make_gpu_tensor(
        Shape::new(vec![rows, cols]),
        DType::BF16,
        ctx.device_id(),
        output,
    ))
}
