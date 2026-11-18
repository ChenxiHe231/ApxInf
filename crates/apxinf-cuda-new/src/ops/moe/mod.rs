//! Model-neutral routed-MoE semantic contracts and CPU reference behavior.
//!
//! This module intentionally contains no CUDA launch or candidate selection.
//! It fixes routing IDs, weights, dispatch order, and combine semantics before
//! those details are encoded in an ABI.

mod contracts;
mod cuda_contracts;
mod execution;
mod launch;
mod packing;
mod reference;

pub use contracts::{
    ExpertAssignment, RoutePlan, RoutedMoeConfig, RoutedMoeSemantic, SharedExpertOutput,
};
pub use cuda_contracts::{
    EdgeBf16DecodeMoeArgs, EdgeBf16DecodeMoeWeights, EdgeNvfp4MoeArgs, EdgeNvfp4MoeWeights,
    Nvfp4ExpertProjection, RoutedMoeAddSharedArgs, RoutedMoeCombineArgs, RoutedMoeNvfp4ExpertsArgs,
    RoutedMoeRouteArgs, RoutedMoeRouteTopkArgs,
};
pub use execution::{
    fused_routed_moe, fused_routed_moe_available, fused_routed_moe_workspace_size,
    routed_moe_add_shared, routed_moe_combine, routed_moe_quantized_experts, routed_moe_route,
    routed_moe_route_topk, small_routed_moe, small_routed_moe_available,
    small_routed_moe_workspace_size,
};
pub use packing::{prepare_expert_fc1, prepare_expert_scales, rescale_expert_weights};
pub use reference::{combine_reference, route_topk_reference};
