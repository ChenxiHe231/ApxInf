use std::ffi::c_void;

pub(crate) use super::types::CudaStream;
use super::types::Runtime;
use crate::ffi::raw::cuda_runtime::{cudaError_t, cudaStream_t};

pub(crate) const SPEC_VERSION: u32 = 1;

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct Spec {
    pub version: u32,
    pub semantic: u32,
    pub dtype: u32,
    pub output_dtype: u32,
    pub has_bias: u32,
    pub input_alignment: u32,
    pub weight_alignment: u32,
    pub bias_alignment: u32,
    pub residual_alignment: u32,
    pub style_alignment: u32,
    pub hidden_alignment: u32,
    pub normalized_alignment: u32,
    pub rows: i64,
    pub cols: i64,
    /// Structural predicate only: the scale value lives in [`Bindings`].
    pub output_scale_is_unit: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct Bindings {
    pub input: *const c_void,
    pub bias: *const c_void,
    pub residual: *const c_void,
    pub weight: *const c_void,
    pub norm_bias: *const c_void,
    pub norm_style: *const c_void,
    pub gate_style: *const c_void,
    pub hidden: *mut c_void,
    pub normalized: *mut c_void,
    pub stream: CudaStream,
    pub eps: f32,
    pub output_scale: f32,
}

unsafe extern "C" {
    pub(crate) fn apxinf_norm_head_rms(
        data: *mut c_void,
        weight: *const c_void,
        rows: i64,
        head_dim: i64,
        epsilon: f32,
        stream: CudaStream,
    ) -> i32;
    pub(crate) fn apxinf_norm_rms_weight_offset(
        input: *const c_void,
        weight: *const c_void,
        output: *mut c_void,
        rows: i64,
        width: i64,
        epsilon: f32,
        stream: CudaStream,
    ) -> i32;
    pub(crate) fn apxinf_norm_rms_norm_rows_bf16_e4m3(
        input: *const c_void,
        weight: *const c_void,
        output: *mut c_void,
        scales: *mut c_void,
        rows: i32,
        input_cols: i32,
        output_cols: i32,
        eps: f32,
        stream: CudaStream,
    ) -> i32;
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn apxinf_norm_bias_residual_rms_norm_rows_bf16_e4m3(
        projection: *const c_void,
        bias: *const c_void,
        residual: *const c_void,
        weight: *const c_void,
        hidden: *mut c_void,
        normalized: *mut c_void,
        scales: *mut c_void,
        rows: i32,
        cols: i32,
        output_cols: i32,
        eps: f32,
        stream: CudaStream,
    ) -> i32;
    pub(crate) fn apxinf_norm_launch(
        runtime: Runtime,
        spec: *const Spec,
        bindings: *const Bindings,
    ) -> i32;
}

unsafe extern "C" {
    pub(crate) fn apxinf_norm_fixed_decode_rms_norm_f32(
        input: *const c_void,
        weight: *const c_void,
        output: *mut c_void,
        cols: u32,
        rows: u32,
        eps: f32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn apxinf_norm_fixed_decode_rms_norm_bf16(
        input: *const c_void,
        weight: *const c_void,
        output: *mut c_void,
        cols: u32,
        rows: u32,
        eps: f32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn apxinf_norm_fixed_decode_rms_norm_add_bf16(
        x_inout: *mut c_void,
        delta: *const c_void,
        weight: *const c_void,
        output: *mut c_void,
        cols: u32,
        rows: u32,
        eps: f32,
        stream: cudaStream_t,
    ) -> cudaError_t;
}

mod fixed_launchers {
    use std::ffi::c_void;

    use crate::ffi::raw::cuda_runtime::{cudaError_t, cudaStream_t};

    unsafe extern "C" {
        #[allow(clippy::too_many_arguments)]
        pub(crate) fn apxinf_norm_gated_rms_silu_bf16(
            x: *const c_void,
            z: *const c_void,
            weight: *const c_void,
            out: *mut c_void,
            rows: i32,
            cols: i32,
            z_heads: i32,
            z_row_stride: i64,
            z_col_offset: i64,
            eps: f32,
            stream: cudaStream_t,
        ) -> cudaError_t;
        #[allow(clippy::too_many_arguments)]
        pub(crate) fn apxinf_norm_add_rms_norm_plus1_bf16(
            a: *const c_void,
            b: *const c_void,
            weight: *const c_void,
            sum_out: *mut c_void,
            output: *mut c_void,
            rows: i32,
            cols: i32,
            eps: f32,
            stream: cudaStream_t,
        ) -> cudaError_t;
        pub(crate) fn apxinf_norm_rms_norm_plus1_bf16(
            input: *const c_void,
            weight: *const c_void,
            output: *mut c_void,
            rows: i32,
            cols: i32,
            eps: f32,
            stream: cudaStream_t,
        ) -> cudaError_t;
        #[allow(clippy::too_many_arguments)]
        pub(crate) fn apxinf_norm_adaln_rms_norm_bf16(
            x: *const c_void,
            weight: *const c_void,
            scale: *const c_void,
            shift: *const c_void,
            out: *mut c_void,
            rows: i32,
            cols: i32,
            eps: f32,
            stream: cudaStream_t,
        ) -> cudaError_t;
        pub(crate) fn apxinf_norm_adaln_gate_residual_bf16(
            proj: *const c_void,
            residual: *const c_void,
            gate: *const c_void,
            out: *mut c_void,
            count: i64,
            cols: i32,
            stream: cudaStream_t,
        ) -> cudaError_t;
        pub(crate) fn apxinf_norm_adaln_gate_residual_rms_bf16(
            projection: *const c_void,
            residual: *const c_void,
            gate: *const c_void,
            weight: *const c_void,
            scale: *const c_void,
            shift: *const c_void,
            hidden: *mut c_void,
            normalized: *mut c_void,
            rows: i32,
            cols: i32,
            eps: f32,
            stream: cudaStream_t,
        ) -> cudaError_t;
        pub(crate) fn apxinf_norm_adaptive_layer_norm_bf16(
            input: *const c_void,
            modulation: *const c_void,
            output: *mut c_void,
            rows: u32,
            cols: u32,
            eps: f32,
            stream: cudaStream_t,
        ) -> cudaError_t;
        pub(crate) fn apxinf_norm_adaptive_layer_norm_quant_bf16_e4m3(
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
        pub(crate) fn apxinf_norm_rms_norm_quant_bf16_e4m3(
            input: *const c_void,
            weight: *const c_void,
            output: *mut c_void,
            rows: i32,
            cols: i32,
            eps: f32,
            scale: f32,
            stream: cudaStream_t,
        ) -> cudaError_t;
        pub(crate) fn apxinf_norm_layer_norm_quant_bf16_e4m3(
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
        pub(crate) fn apxinf_norm_bias_then_residual_bf16(
            projection: *const c_void,
            bias: *const c_void,
            residual: *const c_void,
            output: *mut c_void,
            rows: i32,
            cols: i32,
            stream: cudaStream_t,
        ) -> cudaError_t;
        pub(crate) fn apxinf_norm_bias_residual_bf16_packed4(
            projection: *const c_void,
            bias: *const c_void,
            residual: *const c_void,
            output: *mut c_void,
            rows: i32,
            cols: i32,
            stream: cudaStream_t,
        ) -> cudaError_t;
        pub(crate) fn apxinf_norm_bias_then_residual_bf16_packed4(
            projection: *const c_void,
            bias: *const c_void,
            residual: *const c_void,
            output: *mut c_void,
            rows: i32,
            cols: i32,
            stream: cudaStream_t,
        ) -> cudaError_t;
        pub(crate) fn apxinf_norm_bias_residual_layer_norm_bf16_cached_1024(
            projection: *const c_void,
            projection_bias: *const c_void,
            residual: *const c_void,
            norm_weight: *const c_void,
            norm_bias: *const c_void,
            hidden: *mut c_void,
            normalized: *mut c_void,
            rows: i32,
            cols: i32,
            eps: f32,
            stream: cudaStream_t,
        ) -> cudaError_t;
        pub(crate) fn apxinf_norm_bias_then_residual_layer_norm_bf16_cached_1536(
            projection: *const c_void,
            projection_bias: *const c_void,
            residual: *const c_void,
            norm_weight: *const c_void,
            norm_bias: *const c_void,
            hidden: *mut c_void,
            normalized: *mut c_void,
            rows: i32,
            cols: i32,
            eps: f32,
            stream: cudaStream_t,
        ) -> cudaError_t;
        pub(crate) fn apxinf_norm_bias_then_residual_adaptive_layer_norm_bf16_cached_1536(
            projection: *const c_void,
            projection_bias: *const c_void,
            residual: *const c_void,
            modulation: *const c_void,
            hidden: *mut c_void,
            normalized: *mut c_void,
            rows: i32,
            cols: i32,
            eps: f32,
            stream: cudaStream_t,
        ) -> cudaError_t;
        pub(crate) fn apxinf_norm_bias_residual_layer_norm_quant_bf16_e4m3(
            projection: *const c_void,
            projection_bias: *const c_void,
            residual: *const c_void,
            norm_weight: *const c_void,
            norm_bias: *const c_void,
            hidden: *mut c_void,
            normalized: *mut c_void,
            rows: i32,
            cols: i32,
            eps: f32,
            scale: f32,
            stream: cudaStream_t,
        ) -> cudaError_t;
    }
}
pub(crate) use fixed_launchers::*;
