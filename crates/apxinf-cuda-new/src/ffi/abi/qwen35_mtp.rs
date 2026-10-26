use std::ffi::c_void;

pub(crate) use super::types::CudaStream;
use super::types::Runtime;

pub(crate) const MTP_MOE_SPEC_VERSION: u32 = 1;

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct MtpMoeSpec {
    pub version: u32,
    pub dtype: u32,
    pub experts: u32,
    pub top_k: u32,
    pub hidden: u32,
    pub intermediate: u32,
    pub vocab_size: u32,
    pub mtp_layers: u32,
    pub draft_steps: u32,
    pub draft_step: u32,
    pub base_committed_position: u64,
    pub mtp_committed_position: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct MtpMoeBindings {
    pub input: *const c_void,
    pub router_weight: *const c_void,
    pub expert_gate_weight: *const c_void,
    pub expert_up_weight: *const c_void,
    pub expert_down_weight: *const c_void,
    pub shared_gate_weight: *const c_void,
    pub shared_up_weight: *const c_void,
    pub shared_down_weight: *const c_void,
    pub shared_expert_gate_weight: *const c_void,
    pub router_logits: *mut c_void,
    pub expert_ids: *mut i32,
    pub expert_weights: *mut f32,
    pub route_intermediate: *mut c_void,
    pub route_output: *mut c_void,
    pub shared_intermediate: *mut c_void,
    pub shared_output: *mut c_void,
    pub shared_gate_logit: *mut c_void,
    pub output: *mut c_void,
    pub stream: CudaStream,
}

unsafe extern "C" {
    pub(crate) fn apxinf_qwen35_mtp_moe(
        runtime: Runtime,
        spec: *const MtpMoeSpec,
        bindings: *const MtpMoeBindings,
    ) -> i32;
}
