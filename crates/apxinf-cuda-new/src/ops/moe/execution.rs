use apxinf_core::Result;

use super::cuda_contracts::{
    edge_nvfp4_workspace_size as workspace_size, normalize_add_shared, normalize_combine,
    normalize_edge_nvfp4, normalize_nvfp4_experts, normalize_route, normalize_route_topk,
    EdgeNvfp4MoeArgs, RoutedMoeAddSharedArgs, RoutedMoeCombineArgs, RoutedMoeNvfp4ExpertsArgs,
    RoutedMoeRouteArgs, RoutedMoeRouteTopkArgs,
};
use super::launch;
use super::RoutedMoeConfig;
use crate::CudaContext;

/// Execute deterministic softmax/top-k routing and build expert-major
/// assignments on the context stream.
pub fn routed_moe_route(ctx: &CudaContext, args: RoutedMoeRouteArgs<'_>) -> Result<()> {
    launch::route(ctx, normalize_route(ctx, args)?)
}

/// Execute only token-major softmax/top-k routing without expert-major dispatch.
pub fn routed_moe_route_topk(ctx: &CudaContext, args: RoutedMoeRouteTopkArgs<'_>) -> Result<()> {
    launch::route_topk(ctx, normalize_route_topk(ctx, args)?)
}

/// Combine expert-major outputs and add the sigmoid-gated shared expert.
pub fn routed_moe_combine(ctx: &CudaContext, args: RoutedMoeCombineArgs<'_>) -> Result<()> {
    launch::combine(ctx, normalize_combine(ctx, args)?)
}

/// Add the sigmoid-gated shared expert to a token-major routed contribution.
pub fn routed_moe_add_shared(ctx: &CudaContext, args: RoutedMoeAddSharedArgs<'_>) -> Result<()> {
    launch::add_shared(ctx, normalize_add_shared(ctx, args)?)
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

pub const fn edge_nvfp4_moe_available() -> bool {
    cfg!(apxinf_edge_nvfp4_moe)
}

pub fn edge_nvfp4_moe_workspace_size(
    config: RoutedMoeConfig,
    tokens: usize,
    hidden: usize,
    intermediate: usize,
) -> Result<usize> {
    workspace_size(config, tokens, hidden, intermediate)
}

/// Execute the SM110 TensorRT-Edge-LLM CuTeDSL routed NVFP4 MoE kernel.
///
/// The result already includes top-k routing weights and is token-major. The
/// caller remains responsible for adding the gated shared expert.
pub fn edge_nvfp4_moe(ctx: &CudaContext, args: EdgeNvfp4MoeArgs<'_>) -> Result<()> {
    launch::edge_nvfp4(ctx, normalize_edge_nvfp4(ctx, args)?)
}
