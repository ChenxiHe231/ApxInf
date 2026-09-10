use apxinf_core::{Result, Tensor};

use super::contracts::{normalize, GemmArgs, Semantic};
use super::execution;
use crate::CudaContext;

pub struct GemmBiasArgs<'a> {
    pub gemm: GemmArgs<'a>,
    pub bias: &'a Tensor,
}

pub fn gemm_bias(ctx: &CudaContext, args: GemmBiasArgs<'_>) -> Result<()> {
    let mut execution = prepare_gemm_bias(ctx, args)?;
    execution.enqueue()?;
    ctx.synchronize().map_err(apxinf_core::Error::Cuda)
}

pub fn prepare_gemm_bias(
    ctx: &CudaContext,
    args: GemmBiasArgs<'_>,
) -> Result<execution::PreparedExecution> {
    execution::prepare(
        ctx,
        normalize(
            ctx,
            args.gemm,
            Semantic::GemmBias,
            execution::PlanApi::gemm_bias(),
            Some(args.bias),
        )?,
    )
}
