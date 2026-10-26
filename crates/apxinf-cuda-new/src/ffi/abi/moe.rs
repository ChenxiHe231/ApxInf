use std::ffi::c_void;

pub(crate) use super::types::CudaStream;
use super::types::Runtime;

pub(crate) const ROUTE_SPEC_VERSION: u32 = 1;
pub(crate) const COMBINE_SPEC_VERSION: u32 = 1;
pub(crate) const NVFP4_EXPERTS_SPEC_VERSION: u32 = 1;
pub(crate) const EDGE_NVFP4_SPEC_VERSION: u32 = 1;

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct RouteSpec {
    pub version: u32,
    pub logits_dtype: u32,
    pub tokens: u32,
    pub experts: u32,
    pub top_k: u32,
    pub renormalize: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct RouteBindings {
    pub logits: *const c_void,
    pub expert_ids: *mut i32,
    pub expert_weights: *mut f32,
    pub assignment_expert_ids: *mut i32,
    pub assignment_token_ids: *mut i32,
    pub assignment_slots: *mut i32,
    pub token_assignment_indices: *mut i32,
    pub assignment_weights: *mut f32,
    pub expert_offsets: *mut i32,
    pub stream: CudaStream,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct CombineSpec {
    pub version: u32,
    pub dtype: u32,
    pub tokens: u32,
    pub routes: u32,
    pub top_k: u32,
    pub hidden: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct CombineBindings {
    pub assignment_output: *const c_void,
    pub assignment_token_ids: *const i32,
    pub assignment_weights: *const f32,
    pub token_assignment_indices: *const i32,
    pub shared_output: *const c_void,
    pub shared_gate_logits: *const c_void,
    pub output: *mut c_void,
    pub stream: CudaStream,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct AddSharedBindings {
    pub routed_output: *const c_void,
    pub shared_output: *const c_void,
    pub shared_gate_logits: *const c_void,
    pub output: *mut c_void,
    pub stream: CudaStream,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct Nvfp4ExpertsSpec {
    pub version: u32,
    pub dtype: u32,
    pub tokens: u32,
    pub routes: u32,
    pub experts: u32,
    pub hidden: u32,
    pub intermediate: u32,
    pub block_size: u32,
    pub scale_layout: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct Nvfp4ExpertsBindings {
    pub input: *const c_void,
    pub assignment_expert_ids: *const i32,
    pub assignment_token_ids: *const i32,

    pub expert_gate_weight: *const c_void,
    pub expert_gate_scales: *const c_void,
    pub expert_gate_alpha: *const f32,
    pub expert_up_weight: *const c_void,
    pub expert_up_scales: *const c_void,
    pub expert_up_alpha: *const f32,
    pub expert_down_weight: *const c_void,
    pub expert_down_scales: *const c_void,
    pub expert_down_alpha: *const f32,

    pub shared_gate_weight: *const c_void,
    pub shared_gate_scales: *const c_void,
    pub shared_gate_alpha: *const f32,
    pub shared_up_weight: *const c_void,
    pub shared_up_scales: *const c_void,
    pub shared_up_alpha: *const f32,
    pub shared_down_weight: *const c_void,
    pub shared_down_scales: *const c_void,
    pub shared_down_alpha: *const f32,

    pub route_intermediate: *mut c_void,
    pub shared_intermediate: *mut c_void,
    pub assignment_output: *mut c_void,
    pub shared_output: *mut c_void,
    pub stream: CudaStream,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct EdgeNvfp4Spec {
    pub version: u32,
    pub dtype: u32,
    pub tokens: u32,
    pub experts: u32,
    pub top_k: u32,
    pub hidden: u32,
    pub intermediate: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct EdgeNvfp4Bindings {
    pub input: *const c_void,
    pub expert_ids: *const i32,
    pub expert_weights: *const f32,
    pub fc1_weight: *const c_void,
    pub fc1_scales: *const c_void,
    pub fc1_alpha: *const f32,
    pub input_global_scale: *const f32,
    pub fc2_weight: *const c_void,
    pub fc2_scales: *const c_void,
    pub fc2_alpha: *const f32,
    pub down_input_scale: *const f32,
    pub workspace: *mut c_void,
    pub workspace_bytes: usize,
    pub output: *mut c_void,
    pub stream: CudaStream,
}

unsafe extern "C" {
    pub(crate) fn apxinf_moe_route_topk(
        runtime: Runtime,
        spec: *const RouteSpec,
        bindings: *const RouteBindings,
    ) -> i32;

    pub(crate) fn apxinf_moe_route_topk_only(
        runtime: Runtime,
        spec: *const RouteSpec,
        bindings: *const RouteBindings,
    ) -> i32;

    pub(crate) fn apxinf_moe_combine(
        runtime: Runtime,
        spec: *const CombineSpec,
        bindings: *const CombineBindings,
    ) -> i32;

    pub(crate) fn apxinf_moe_add_shared(
        runtime: Runtime,
        spec: *const CombineSpec,
        bindings: *const AddSharedBindings,
    ) -> i32;

    pub(crate) fn apxinf_moe_nvfp4_experts(
        runtime: Runtime,
        spec: *const Nvfp4ExpertsSpec,
        bindings: *const Nvfp4ExpertsBindings,
    ) -> i32;

    pub(crate) fn apxinf_moe_edge_nvfp4_workspace_size(
        spec: *const EdgeNvfp4Spec,
        workspace_bytes: *mut usize,
    ) -> i32;

    pub(crate) fn apxinf_moe_edge_nvfp4(
        runtime: Runtime,
        spec: *const EdgeNvfp4Spec,
        bindings: *const EdgeNvfp4Bindings,
    ) -> i32;
}
