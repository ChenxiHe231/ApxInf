use apxinf_core::Result;

use super::contracts::{normalize, GemmArgs, Semantic};
use super::execution;
use crate::CudaContext;

pub struct GemmGegluArgs<'a> {
    pub gemm: GemmArgs<'a>,
}

pub fn gemm_geglu(ctx: &CudaContext, args: GemmGegluArgs<'_>) -> Result<()> {
    let mut execution = prepare_gemm_geglu(ctx, args)?;
    execution.enqueue()?;
    ctx.synchronize().map_err(apxinf_core::Error::Cuda)
}

pub fn prepare_gemm_geglu(
    ctx: &CudaContext,
    args: GemmGegluArgs<'_>,
) -> Result<execution::PreparedExecution> {
    execution::prepare(
        ctx,
        normalize(
            ctx,
            args.gemm,
            Semantic::GemmGeglu,
            execution::PlanApi::gemm_geglu(),
            None,
        )?,
    )
}
