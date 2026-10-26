use apxinf_core::Result;

use super::cuda_contracts::{
    normalize_combine, normalize_nvfp4_experts, normalize_route, RoutedMoeCombineArgs,
    RoutedMoeNvfp4ExpertsArgs, RoutedMoeRouteArgs,
};
use super::launch;
use crate::CudaContext;

/// Execute deterministic softmax/top-k routing and build expert-major
/// assignments on the context stream.
pub fn routed_moe_route(ctx: &CudaContext, args: RoutedMoeRouteArgs<'_>) -> Result<()> {
    launch::route(ctx, normalize_route(ctx, args)?)
}

/// Combine expert-major outputs and add the sigmoid-gated shared expert.
pub fn routed_moe_combine(ctx: &CudaContext, args: RoutedMoeCombineArgs<'_>) -> Result<()> {
    launch::combine(ctx, normalize_combine(ctx, args)?)
}

/// Run routed and shared NVFP4 SwiGLU experts without a host route round trip.
///
/// This fixed correctness candidate reads checkpoint-native row-major block
/// scales and writes BF16 outputs for [`routed_moe_combine`](super::routed_moe_combine).
pub fn routed_moe_nvfp4_experts(
    ctx: &CudaContext,
    args: RoutedMoeNvfp4ExpertsArgs<'_>,
) -> Result<()> {
    launch::nvfp4_experts(ctx, normalize_nvfp4_experts(ctx, args)?)
}
