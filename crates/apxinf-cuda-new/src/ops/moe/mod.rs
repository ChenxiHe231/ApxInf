//! Model-neutral routed-MoE semantic contracts and CPU reference behavior.
//!
//! This module intentionally contains no CUDA launch or candidate selection.
//! It fixes routing IDs, weights, dispatch order, and combine semantics before
//! those details are encoded in an ABI.

mod contracts;
mod cuda_contracts;
mod execution;
mod launch;
mod reference;

pub use contracts::{
    ExpertAssignment, RoutePlan, RoutedMoeConfig, RoutedMoeSemantic, SharedExpertOutput,
};
pub use cuda_contracts::{
    Nvfp4ExpertProjection, RoutedMoeCombineArgs, RoutedMoeNvfp4ExpertsArgs, RoutedMoeRouteArgs,
};
pub use execution::{routed_moe_combine, routed_moe_nvfp4_experts, routed_moe_route};
pub use reference::{combine_reference, route_topk_reference};
