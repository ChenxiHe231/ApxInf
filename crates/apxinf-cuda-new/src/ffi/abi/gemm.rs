use std::ffi::{c_char, c_void};

use super::types::{CudaStream, Runtime};

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Spec {
    pub version: u32,
    pub a_dtype: u32,
    pub b_dtype: u32,
    pub accumulation_dtype: u32,
    pub output_dtype: u32,
    pub quantization: u32,
    pub a_alignment: u32,
    pub b_alignment: u32,
    pub bias_alignment: u32,
    pub a_scales_alignment: u32,
    pub b_scales_alignment: u32,
    pub output_alignment: u32,
    pub m: i64,
    pub n: i64,
    pub k: i64,
    /// Structural predicates only: the scale values live in [`Bindings`].
    pub alpha_is_unit: u32,
    pub output_scale_is_unit: u32,
}

#[repr(C)]
pub(crate) struct Policy {
    pub workspace_limit: u64,
    pub online_tune: u32,
    pub allow_fallback: u32,
    pub graph_safe: u32,
    pub deterministic: u32,
    pub execution_mode: u32,
    pub cache_dir: *const c_char,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct Bindings {
    pub a: *const c_void,
    pub b: *const c_void,
    pub bias: *const c_void,
    pub a_scales: *const f32,
    pub b_scales: *const f32,
    pub output: *mut c_void,
    pub stream: CudaStream,
    pub alpha: f32,
    pub output_scale: f32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct TuningBindings {
    pub execution: Bindings,
    pub original_a: *const f32,
    pub original_a_len: u64,
    pub original_b: *const f32,
    pub original_b_len: u64,
    pub original_bias: *const f32,
    pub original_bias_len: u64,
    /// 0 reconstructs inputs from the inference bindings; 1 uses originals.
    pub reference_kind: u32,
}

pub(crate) type Plan = *mut c_void;
pub(crate) type Instance = *mut c_void;

unsafe extern "C" {
    pub(crate) fn apxinf_gemm_plan_create(
        runtime: Runtime,
        spec: *const Spec,
        policy: *const Policy,
        tuning_bindings: *const TuningBindings,
        plan: *mut Plan,
    ) -> i32;
    pub(crate) fn apxinf_gemm_instance_create(
        plan: Plan,
        bindings: *const Bindings,
        instance: *mut Instance,
    ) -> i32;
    pub(crate) fn apxinf_gemm_enqueue(instance: Instance) -> i32;
    pub(crate) fn apxinf_gemm_instance_destroy(instance: Instance);
    pub(crate) fn apxinf_gemm_plan_destroy(plan: Plan);
    pub(crate) fn apxinf_gemm_plan_summary(plan: Plan) -> *const c_char;
}
