use std::ffi::{c_char, c_void};

use super::gemm::{Bindings, Policy, Spec};
use super::types::Runtime;

pub(crate) type Plan = *mut c_void;
pub(crate) type Instance = *mut c_void;

unsafe extern "C" {
    pub(crate) fn apxinf_gemm_geglu_plan_create(
        runtime: Runtime,
        spec: *const Spec,
        policy: *const Policy,
        tuning_bindings: *const Bindings,
        plan: *mut Plan,
    ) -> i32;
    pub(crate) fn apxinf_gemm_geglu_instance_create(
        plan: Plan,
        bindings: *const Bindings,
        instance: *mut Instance,
    ) -> i32;
    pub(crate) fn apxinf_gemm_geglu_enqueue(instance: Instance) -> i32;
    pub(crate) fn apxinf_gemm_geglu_instance_destroy(instance: Instance);
    pub(crate) fn apxinf_gemm_geglu_plan_destroy(plan: Plan);
    pub(crate) fn apxinf_gemm_geglu_plan_summary(plan: Plan) -> *const c_char;
}
