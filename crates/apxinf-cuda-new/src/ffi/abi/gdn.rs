use std::ffi::c_void;

use super::types::CudaStream;

unsafe extern "C" {
    pub(crate) fn apxinf_gdn_recurrent_step(
        state: *mut c_void,
        q: *const c_void,
        k: *const c_void,
        v: *const c_void,
        decay: *const c_void,
        beta: *const c_void,
        output: *mut c_void,
        v_heads: i64,
        k_heads: i64,
        v_dim: i64,
        k_dim: i64,
        stream: CudaStream,
    ) -> i32;
    pub(crate) fn apxinf_gdn_gated_norm(
        input: *const c_void,
        gate: *const c_void,
        weight: *const c_void,
        output: *mut c_void,
        heads: i64,
        head_dim: i64,
        epsilon: f32,
        stream: CudaStream,
    ) -> i32;
    pub(crate) fn apxinf_gdn_causal_conv_step(
        window: *mut c_void,
        input: *const c_void,
        weight: *const c_void,
        output: *mut c_void,
        channels: i64,
        kernel_width: i64,
        stream: CudaStream,
    ) -> i32;
    pub(crate) fn apxinf_gdn_l2_normalize_heads(
        data: *mut c_void,
        heads: i64,
        head_dim: i64,
        epsilon: f32,
        stream: CudaStream,
    ) -> i32;
    pub(crate) fn apxinf_gdn_decay_and_beta(
        a: *const c_void,
        b: *const c_void,
        a_log: *const c_void,
        dt_bias: *const c_void,
        decay: *mut c_void,
        beta: *mut c_void,
        heads: i64,
        stream: CudaStream,
    ) -> i32;
}
