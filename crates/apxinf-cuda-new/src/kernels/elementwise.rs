//! Legacy `kernels::elementwise` names over cuda-new operators.

use apxinf_core::{DType, Result, Shape, Tensor};

use crate::{ops, CudaContext};

/// `add`: elementwise `a + b` into a fresh tensor.
pub fn add(ctx: &CudaContext, a: &Tensor, b: &Tensor) -> Result<Tensor> {
    let output = ctx.allocate_output(Shape::new(a.shape().dims().to_vec()), DType::BF16)?;
    ops::elementwise_add(ctx, a, b, &output)?;
    Ok(output)
}

/// `scale`: elementwise `input * factor` into a fresh tensor.
pub fn scale(ctx: &CudaContext, input: &Tensor, factor: f32) -> Result<Tensor> {
    let output = ctx.allocate_output(Shape::new(input.shape().dims().to_vec()), DType::BF16)?;
    ops::elementwise_scale(ctx, input, &output, factor)?;
    Ok(output)
}

/// `bias_bf16`: broadcast-add an optional `[cols]` bias over rows. A missing
/// bias returns the input unchanged, matching the legacy contract.
pub fn bias_bf16(ctx: &CudaContext, input: &Tensor, value: Option<&Tensor>) -> Result<Tensor> {
    let Some(bias) = value else {
        return Ok(input.clone());
    };
    let output = ctx.allocate_output(Shape::new(input.shape().dims().to_vec()), DType::BF16)?;
    ops::elementwise_add_bias(ctx, input, bias, &output)?;
    Ok(output)
}

/// `concat_rows_bf16`: stack two row-major matrices with equal columns.
pub fn concat_rows_bf16(ctx: &CudaContext, first: &Tensor, second: &Tensor) -> Result<Tensor> {
    ops::concat_rows(ctx, first, second)
}
