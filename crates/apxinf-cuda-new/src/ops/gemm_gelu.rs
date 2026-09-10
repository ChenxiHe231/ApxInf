use apxinf_core::{Result, Tensor};

use super::contracts::{normalize, GemmArgs, Semantic};
use super::execution;
use crate::CudaContext;

pub struct GemmBiasGeluArgs<'a> {
    pub gemm: GemmArgs<'a>,
    pub bias: &'a Tensor,
}

pub fn gemm_bias_gelu(ctx: &CudaContext, args: GemmBiasGeluArgs<'_>) -> Result<()> {
    let mut execution = prepare_gemm_bias_gelu(ctx, args)?;
    execution.enqueue()?;
    ctx.synchronize().map_err(apxinf_core::Error::Cuda)
}

pub fn prepare_gemm_bias_gelu(
    ctx: &CudaContext,
    args: GemmBiasGeluArgs<'_>,
) -> Result<execution::PreparedExecution> {
    execution::prepare(
        ctx,
        normalize(
            ctx,
            args.gemm,
            Semantic::GemmBiasGelu,
            execution::PlanApi::gemm_bias_gelu(),
            Some(args.bias),
        )?,
    )
}
