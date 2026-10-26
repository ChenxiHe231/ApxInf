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
    edge_bf16_decode_moe, edge_bf16_decode_moe_available, edge_bf16_decode_moe_workspace_size,
    edge_nvfp4_moe, edge_nvfp4_moe_available, edge_nvfp4_moe_workspace_size, routed_moe_add_shared,
    routed_moe_combine, routed_moe_nvfp4_experts, routed_moe_route, routed_moe_route_topk,
};
pub use packing::{edge_nvfp4_interleave_fc1, edge_nvfp4_rescale_e4m3, edge_nvfp4_swizzle_scales};
pub use reference::{combine_reference, route_topk_reference};
