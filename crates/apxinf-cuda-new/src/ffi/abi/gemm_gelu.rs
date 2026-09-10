use std::ffi::{c_char, c_void};

use super::gemm::{Bindings, Policy, Spec, TuningBindings};
use super::types::Runtime;

pub(crate) type Plan = *mut c_void;
pub(crate) type Instance = *mut c_void;

unsafe extern "C" {
    pub(crate) fn apxinf_gemm_bias_gelu_plan_create(
        runtime: Runtime,
        spec: *const Spec,
        policy: *const Policy,
        tuning_bindings: *const TuningBindings,
        plan: *mut Plan,
    ) -> i32;
    pub(crate) fn apxinf_gemm_bias_gelu_instance_create(
        plan: Plan,
        bindings: *const Bindings,
        instance: *mut Instance,
    ) -> i32;
    pub(crate) fn apxinf_gemm_bias_gelu_enqueue(instance: Instance) -> i32;
    pub(crate) fn apxinf_gemm_bias_gelu_instance_destroy(instance: Instance);
    pub(crate) fn apxinf_gemm_bias_gelu_plan_destroy(plan: Plan);
    pub(crate) fn apxinf_gemm_bias_gelu_plan_summary(plan: Plan) -> *const c_char;
}
