use std::ffi::c_void;

use super::types::{CudaStream, Runtime};

pub(crate) const SPEC_VERSION: u32 = 2;

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Spec {
    pub version: u32,
    pub tokens: u32,
    pub cache_offset: u32,
    pub cache_capacity: u32,
    pub rope_theta: f32,
    pub rms_epsilon: f32,
    pub key_scale: f32,
    pub value_scale: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct Bindings {
    pub projected_query_gate: *const c_void,
    pub projected_key: *const c_void,
    pub projected_value: *const c_void,
    pub query_norm: *const c_void,
    pub key_norm: *const c_void,
    pub positions: *const c_void,
    pub key_cache: *mut c_void,
    pub value_cache: *mut c_void,
    pub query: *mut c_void,
    pub gate: *mut c_void,
    pub output: *mut c_void,
    pub scores: *mut c_void,
    pub partial_output: *mut c_void,
    pub partial_stats: *mut c_void,
    pub stream: CudaStream,
}

unsafe extern "C" {
    pub(crate) fn apxinf_qwen35_attention_fp8(
        runtime: Runtime,
        spec: *const Spec,
        bindings: *const Bindings,
    ) -> i32;
}
