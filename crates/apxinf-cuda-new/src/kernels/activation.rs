//! Legacy `kernels::activation` names over cuda-new pointwise operators.

use apxinf_core::{DType, Error, Result, Shape, Tensor};

use crate::{ops, CudaContext};

/// `silu`: elementwise SiLU into a fresh tensor.
pub fn silu(ctx: &CudaContext, input: &Tensor) -> Result<Tensor> {
    let output = ctx.allocate_output(Shape::new(input.shape().dims().to_vec()), DType::BF16)?;
    ops::elementwise_activation(ctx, input, &output, ops::ElementwiseActivation::Silu)?;
    Ok(output)
}

/// `gelu_tanh`: elementwise tanh-approximated GELU into a fresh tensor.
pub fn gelu_tanh(ctx: &CudaContext, input: &Tensor) -> Result<Tensor> {
    let output = ctx.allocate_output(Shape::new(input.shape().dims().to_vec()), DType::BF16)?;
    ops::elementwise_activation(ctx, input, &output, ops::ElementwiseActivation::GeluTanh)?;
    Ok(output)
}

/// `geglu_bf16`: `[rows, 2*cols]` gate/up projection to `[rows, cols]` GeGLU.
pub fn geglu_bf16(ctx: &CudaContext, gate_up: &Tensor) -> Result<Tensor> {
    let dims = gate_up.shape().dims();
    if dims.len() != 2 || dims[1] % 2 != 0 {
        return Err(Error::Other("GeGLU requires a [rows, 2*cols] input".into()));
    }
    let mut output = ctx.allocate_output(Shape::new(vec![dims[0], dims[1] / 2]), DType::BF16)?;
    let args = ops::PointwiseArgs::new(ops::PointwiseSemantic::Geglu, gate_up, &mut output);
    ops::pointwise(ctx, args)?;
    Ok(output)
}

/// `swiglu_bf16`: `[rows, 2*cols]` gate/up projection to `[rows, cols]` SwiGLU.
pub fn swiglu_bf16(ctx: &CudaContext, gate_up: &Tensor) -> Result<Tensor> {
    let dims = gate_up.shape().dims();
    if dims.len() != 2 || dims[1] % 2 != 0 {
        return Err(Error::Other("SwiGLU requires a [rows, 2*cols] input".into()));
    }
    let output = ctx.allocate_output(Shape::new(vec![dims[0], dims[1] / 2]), DType::BF16)?;
    ops::mlp::swiglu(ctx, gate_up, &output)?;
    Ok(output)
}

/// `bias_gelu_bf16`: broadcast bias then tanh-GELU. A missing bias is a plain
/// GELU, matching the legacy contract.
pub fn bias_gelu_bf16(ctx: &CudaContext, input: &Tensor, value: Option<&Tensor>) -> Result<Tensor> {
    let mut output = ctx.allocate_output(Shape::new(input.shape().dims().to_vec()), DType::BF16)?;
    let mut args =
        ops::PointwiseArgs::new(ops::PointwiseSemantic::BiasActivation, input, &mut output);
    args.bias = value;
    args.activation = ops::PointwiseActivation::Gelu;
    ops::pointwise(ctx, args)?;
    Ok(output)
}
