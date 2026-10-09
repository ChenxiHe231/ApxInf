//! Legacy `kernels::gemm` names over the cuda-new GEMM operator.

use apxinf_core::{DType, Error, Result, Shape, Tensor};

use crate::{ops, CudaContext};

fn output_for(ctx: &CudaContext, a: &Tensor, b: &Tensor, what: &str) -> Result<Tensor> {
    let a_dims = a.shape().dims();
    let b_dims = b.shape().dims();
    if a_dims.len() != 2 || b_dims.len() != 2 || a_dims[1] != b_dims[0] {
        return Err(Error::Other(format!(
            "{what} shape mismatch: {a_dims:?} @ {b_dims:?}"
        )));
    }
    ctx.allocate_output(Shape::new(vec![a_dims[0], b_dims[1]]), DType::BF16)
}

/// `bf16`: plain BF16 GEMM, `[m, k] @ [k, n] -> [m, n]`.
pub fn bf16(ctx: &CudaContext, activation: &Tensor, weight: &Tensor) -> Result<Tensor> {
    let mut output = output_for(ctx, activation, weight, "BF16 GEMM")?;
    ops::gemm(ctx, ops::GemmArgs::new(activation, weight, &mut output))?;
    Ok(output)
}

/// `matmul`: alias of [`bf16`] under the legacy generic name.
pub fn matmul(ctx: &CudaContext, activation: &Tensor, weight: &Tensor) -> Result<Tensor> {
    bf16(ctx, activation, weight)
}

/// `bf16_bias`: BF16 GEMM with a fused `[n]` bias epilogue.
pub fn bf16_bias(
    ctx: &CudaContext,
    activation: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
) -> Result<Tensor> {
    let mut output = output_for(ctx, activation, weight, "BF16 bias GEMM")?;
    let gemm = ops::GemmArgs::new(activation, weight, &mut output);
    ops::gemm_bias(ctx, ops::GemmBiasArgs { gemm, bias })?;
    Ok(output)
}
