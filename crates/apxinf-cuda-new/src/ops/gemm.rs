use apxinf_core::Result;

use super::contracts::{normalize, GemmArgs, Semantic};
use super::execution;
use crate::CudaContext;

pub fn gemm(ctx: &CudaContext, args: GemmArgs<'_>) -> Result<()> {
    let mut execution = prepare_gemm(ctx, args)?;
    execution.enqueue()?;
    ctx.synchronize().map_err(apxinf_core::Error::Cuda)
}

pub fn prepare_gemm(
    ctx: &CudaContext,
    args: GemmArgs<'_>,
) -> Result<execution::PreparedExecution> {
    execution::prepare(
        ctx,
        normalize(ctx, args, Semantic::Gemm, execution::PlanApi::gemm(), None)?,
    )
}
