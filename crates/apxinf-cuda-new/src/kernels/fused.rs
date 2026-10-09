//! Legacy `kernels::fused` names over cuda-new norm operators.

use apxinf_core::{DType, Result, Shape, Tensor};

use crate::{ops, CudaContext};

/// Output pair of a fused residual + norm: the updated residual stream and
/// its normalized view. Mirrors the legacy struct of the same name.
pub struct ResidualNormTensors {
    pub hidden: Tensor,
    pub normalized: Tensor,
}

/// `bias_residual_bf16`: `hidden = projection + bias? + residual`.
pub fn bias_residual_bf16(
    ctx: &CudaContext,
    projection: &Tensor,
    bias: Option<&Tensor>,
    residual: &Tensor,
) -> Result<Tensor> {
    let mut hidden =
        ctx.allocate_output(Shape::new(projection.shape().dims().to_vec()), DType::BF16)?;
    ops::bias_residual(
        ctx,
        ops::BiasResidualArgs::new(projection, bias, residual, &mut hidden),
    )?;
    Ok(hidden)
}

/// `bias_residual_rms_bf16`: residual update plus an RMS-normalized view.
pub fn bias_residual_rms_bf16(
    ctx: &CudaContext,
    projection: &Tensor,
    bias: Option<&Tensor>,
    residual: &Tensor,
    weight: &Tensor,
    eps: f32,
) -> Result<ResidualNormTensors> {
    let shape = Shape::new(projection.shape().dims().to_vec());
    let mut hidden = ctx.allocate_output(shape.clone(), DType::BF16)?;
    let mut normalized = ctx.allocate_output(shape, DType::BF16)?;
    ops::bias_residual_rms_norm(
        ctx,
        ops::BiasResidualRmsNormArgs::new(
            projection,
            bias,
            residual,
            weight,
            &mut hidden,
            &mut normalized,
            eps,
        ),
    )?;
    Ok(ResidualNormTensors { hidden, normalized })
}

/// `bias_residual_layer_bf16`: residual update plus a LayerNorm-normalized view.
#[allow(clippy::too_many_arguments)]
pub fn bias_residual_layer_bf16(
    ctx: &CudaContext,
    projection: &Tensor,
    projection_bias: Option<&Tensor>,
    residual: &Tensor,
    norm_weight: &Tensor,
    norm_bias: &Tensor,
    eps: f32,
) -> Result<ResidualNormTensors> {
    let shape = Shape::new(projection.shape().dims().to_vec());
    let mut hidden = ctx.allocate_output(shape.clone(), DType::BF16)?;
    let mut normalized = ctx.allocate_output(shape, DType::BF16)?;
    ops::bias_residual_layer_norm(
        ctx,
        ops::BiasResidualLayerNormArgs::new(
            projection,
            projection_bias,
            residual,
            norm_weight,
            norm_bias,
            &mut hidden,
            &mut normalized,
            eps,
        ),
    )?;
    Ok(ResidualNormTensors { hidden, normalized })
}
