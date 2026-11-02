use std::ffi::c_void;

use super::types::{CudaStream, Runtime};

pub(crate) const SPEC_VERSION: u32 = 2;
pub(crate) const DYNAMIC_KV_SPEC_VERSION: u32 = 1;
pub(crate) const DYNAMIC_DECODE_SPEC_VERSION: u32 = 2;
pub(crate) const QUERY_PREPARE_SPEC_VERSION: u32 = 1;
pub(crate) const DECODE_SPLIT_TOKENS: usize = 256;
pub(crate) const MIN_DECODE_SPLIT_TOKENS: usize = 128;

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

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct DynamicKvSpec {
    pub version: u32,
    pub tokens: u32,
    pub cache_offset: u32,
    pub cache_capacity: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct DynamicKvBindings {
    pub key: *const c_void,
    pub value: *const c_void,
    pub key_cache: *mut c_void,
    pub value_cache: *mut c_void,
    pub key_scales: *mut c_void,
    pub value_scales: *mut c_void,
    pub stream: CudaStream,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct DynamicPrepareSpec {
    pub kv: DynamicKvSpec,
    pub rope_theta: f32,
    pub rms_epsilon: f32,
}
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct DynamicPrepareBindings {
    pub kv: DynamicKvBindings,
    pub norm_weight: *const c_void,
    pub positions: *const c_void,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct QueryPrepareSpec {
    pub version: u32,
    pub tokens: u32,
    pub rope_theta: f32,
    pub rms_epsilon: f32,
}
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct QueryPrepareBindings {
    pub projected_query_gate: *const c_void,
    pub norm_weight: *const c_void,
    pub positions: *const c_void,
    pub query: *mut c_void,
    pub gate: *mut c_void,
    pub stream: CudaStream,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct DynamicDecodeSpec {
    pub version: u32,
    pub valid_tokens: u32,
    pub cache_capacity: u32,
    pub max_splits: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct DynamicDecodeBindings {
    pub query: *const c_void,
    pub key_cache: *const c_void,
    pub value_cache: *const c_void,
    pub key_scales: *const c_void,
    pub value_scales: *const c_void,
    pub partial_max: *mut c_void,
    pub partial_sum: *mut c_void,
    pub partial_output: *mut c_void,
    pub gate: *const c_void,
    pub output: *mut c_void,
    pub stream: CudaStream,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct DynamicDequantSpec {
    pub version: u32,
    pub valid_tokens: u32,
    pub cache_capacity: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct DynamicDequantBindings {
    pub key_cache: *const c_void,
    pub value_cache: *const c_void,
    pub key_scales: *const c_void,
    pub value_scales: *const c_void,
    pub key: *mut c_void,
    pub value: *mut c_void,
    pub stream: CudaStream,
}

unsafe extern "C" {
    pub(crate) fn apxinf_qwen35_query_prepare(
        runtime: Runtime,
        spec: *const QueryPrepareSpec,
        bindings: *const QueryPrepareBindings,
    ) -> i32;

    pub(crate) fn apxinf_qwen35_attention_fp8(
        runtime: Runtime,
        spec: *const Spec,
        bindings: *const Bindings,
    ) -> i32;

    pub(crate) fn apxinf_qwen35_dynamic_kv_append(
        runtime: Runtime,
        spec: *const DynamicKvSpec,
        bindings: *const DynamicKvBindings,
    ) -> i32;

    pub(crate) fn apxinf_qwen35_dynamic_kv_prepare(
        runtime: Runtime,
        spec: *const DynamicPrepareSpec,
        bindings: *const DynamicPrepareBindings,
    ) -> i32;

    pub(crate) fn apxinf_qwen35_dynamic_decode_attention(
        runtime: Runtime,
        spec: *const DynamicDecodeSpec,
        bindings: *const DynamicDecodeBindings,
    ) -> i32;

    pub(crate) fn apxinf_qwen35_dynamic_kv_dequant(
        runtime: Runtime,
        spec: *const DynamicDequantSpec,
        bindings: *const DynamicDequantBindings,
    ) -> i32;
}
