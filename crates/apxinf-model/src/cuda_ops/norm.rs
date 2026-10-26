use apxinf_core::{DType, Result, Shape, Tensor};
use apxinf_cuda_new::{ops, CudaContext};

pub(crate) fn rms(
    ctx: &CudaContext,
    input: &Tensor,
    weight: &Tensor,
    eps: f32,
) -> Result<Tensor> {
    let mut output = ctx.allocate_output(input.shape().clone(), input.dtype())?;
    ops::rms_norm(ctx, ops::RmsNormArgs::new(input, weight, &mut output, eps))?;
    Ok(output)
}

pub(crate) fn layer(
    ctx: &CudaContext,
    input: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    eps: f32,
) -> Result<Tensor> {
    let mut output = ctx.allocate_output(input.shape().clone(), input.dtype())?;
    ops::layer_norm(
        ctx,
        ops::LayerNormArgs::new(input, weight, bias, &mut output, eps),
    )?;
    Ok(output)
}

pub(crate) fn bias_residual(
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

pub(crate) fn bias_residual_rms_norm(
    ctx: &CudaContext,
    projection: &Tensor,
    bias: Option<&Tensor>,
    residual: &Tensor,
    weight: &Tensor,
    eps: f32,
) -> Result<ops::norm::ResidualNormTensors> {
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
    Ok(ops::norm::ResidualNormTensors { hidden, normalized })
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn bias_residual_layer_norm(
    ctx: &CudaContext,
    projection: &Tensor,
    projection_bias: Option<&Tensor>,
    residual: &Tensor,
    norm_weight: &Tensor,
    norm_bias: &Tensor,
    eps: f32,
) -> Result<ops::norm::ResidualNormTensors> {
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
    Ok(ops::norm::ResidualNormTensors { hidden, normalized })
}
