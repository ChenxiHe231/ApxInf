//! Legacy `kernels::norm` names over cuda-new norm operators.

use apxinf_core::{DType, Result, Shape, Tensor};

use crate::{ops, CudaContext};

/// `rms_bf16`: RMS-normalize a `[rows, cols]` BF16 activation.
pub fn rms_bf16(ctx: &CudaContext, input: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
    let output = ctx.allocate_output(Shape::new(input.shape().dims().to_vec()), DType::BF16)?;
    ops::mlp::rms_norm(ctx, input, weight, &output, eps)?;
    Ok(output)
}

/// `layer_bf16`: LayerNorm with weight and bias.
pub fn layer_bf16(
    ctx: &CudaContext,
    input: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    eps: f32,
) -> Result<Tensor> {
    let mut output = ctx.allocate_output(Shape::new(input.shape().dims().to_vec()), DType::BF16)?;
    ops::layer_norm(ctx, ops::LayerNormArgs::new(input, weight, bias, &mut output, eps))?;
    Ok(output)
}
