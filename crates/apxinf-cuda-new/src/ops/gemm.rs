use apxinf_core::Result;

use super::contracts::{normalize, GemmArgs, Semantic};
use super::execution;
use crate::CudaContext;

pub fn gemm(ctx: &CudaContext, args: GemmArgs<'_>) -> Result<()> {
    execution::execute(
        ctx,
        normalize(ctx, args, Semantic::Gemm, execution::PlanApi::gemm(), None)?,
    )
}

pub fn prepare_gemm(ctx: &CudaContext, args: GemmArgs<'_>) -> Result<execution::PreparedExecution> {
    execution::prepare(
        ctx,
        normalize(ctx, args, Semantic::Gemm, execution::PlanApi::gemm(), None)?,
    )
}
