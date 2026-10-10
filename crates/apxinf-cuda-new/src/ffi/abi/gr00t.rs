//! Raw bindings for the GR00T-family fused activation primitives, ported from
//! the legacy crate with the apxinf_gr00t_ prefix. Defined in
//! native/adapters/gr00t/ops.cu.

use std::ffi::c_void;

use crate::ffi::raw::cuda_runtime::{cudaError_t, cudaStream_t};

unsafe extern "C" {
    pub(crate) fn apxinf_gr00t_silu_mul_separate_bf16(
        gate: *const c_void,
        up: *const c_void,
        output: *mut c_void,
        count: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn apxinf_gr00t_silu_mul_quant_bf16_e4m3(
        gate: *const c_void,
        up: *const c_void,
        output: *mut c_void,
        count: i64,
        scale: f32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn apxinf_gr00t_bias_gelu_quant_bf16_e4m3(
        input: *const c_void,
        bias: *const c_void,
        output: *mut c_void,
        rows: i32,
        cols: i32,
        scale: f32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn apxinf_gr00t_bias_gelu_bf16_packed8(
        input: *const c_void,
        bias: *const c_void,
        output: *mut c_void,
        rows: i32,
        cols: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn apxinf_gr00t_scatter_rows_bf16(
        source: *const c_void,
        rows: *const c_void,
        output: *mut c_void,
        row_count: i32,
        cols: i32,
        add: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn apxinf_gr00t_bias_qkv_in_place_bf16(
        query: *mut c_void,
        key: *mut c_void,
        value: *mut c_void,
        query_bias: *const c_void,
        key_bias: *const c_void,
        value_bias: *const c_void,
        rows: i32,
        cols: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn apxinf_gr00t_adaptive_layer_norm_bf16(
        input: *const c_void,
        modulation: *const c_void,
        output: *mut c_void,
        rows: u32,
        cols: u32,
        eps: f32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn apxinf_gr00t_adaptive_layer_norm_quant_bf16_e4m3(
        input: *const c_void,
        modulation: *const c_void,
        output: *mut c_void,
        quantized: *mut c_void,
        rows: u32,
        cols: u32,
        eps: f32,
        scale: f32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn apxinf_gr00t_rms_norm_quant_bf16_e4m3(
        input: *const c_void,
        weight: *const c_void,
        output: *mut c_void,
        rows: i32,
        cols: i32,
        eps: f32,
        scale: f32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn apxinf_gr00t_layer_norm_quant_bf16_e4m3(
        input: *const c_void,
        weight: *const c_void,
        bias: *const c_void,
        output: *mut c_void,
        rows: i32,
        cols: i32,
        eps: f32,
        scale: f32,
        stream: cudaStream_t,
    ) -> cudaError_t;
}
