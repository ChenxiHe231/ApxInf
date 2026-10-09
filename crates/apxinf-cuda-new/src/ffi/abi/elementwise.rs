use std::ffi::c_void;

use super::types::CudaStream;

unsafe extern "C" {
    pub(crate) fn apxinf_elementwise_activation_bf16(
        input: *const c_void,
        output: *mut c_void,
        count: i64,
        activation: i32,
        stream: CudaStream,
    ) -> i32;
    pub(crate) fn apxinf_elementwise_mul_bf16(
        a: *const c_void,
        b: *const c_void,
        output: *mut c_void,
        count: i64,
        stream: CudaStream,
    ) -> i32;
    pub(crate) fn apxinf_elementwise_add_bf16(
        a: *const c_void,
        b: *const c_void,
        output: *mut c_void,
        count: i64,
        stream: CudaStream,
    ) -> i32;
    pub(crate) fn apxinf_elementwise_scale_bf16(
        input: *const c_void,
        output: *mut c_void,
        count: i64,
        factor: f32,
        stream: CudaStream,
    ) -> i32;
    pub(crate) fn apxinf_elementwise_add_bias_bf16(
        input: *const c_void,
        bias: *const c_void,
        output: *mut c_void,
        rows: i64,
        cols: i64,
        stream: CudaStream,
    ) -> i32;
}
