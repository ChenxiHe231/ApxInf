use apxinf_core::{Result, Tensor};

use super::contracts::{normalize, GemmArgs, Semantic};
use super::execution;
use crate::CudaContext;

pub struct GemmBiasGeluArgs<'a> {
    pub gemm: GemmArgs<'a>,
    pub bias: &'a Tensor,
}

pub fn gemm_bias_gelu(ctx: &CudaContext, args: GemmBiasGeluArgs<'_>) -> Result<()> {
    execution::execute(
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
