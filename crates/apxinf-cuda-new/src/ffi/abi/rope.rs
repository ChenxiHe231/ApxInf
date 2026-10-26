use std::ffi::c_void;

pub(crate) use super::types::CudaStream;
use super::types::Runtime;
use crate::ffi::raw::cuda_runtime::{cudaError_t, cudaStream_t};

pub(crate) const SPEC_VERSION: u32 = 3;

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct Spec {
    pub version: u32,
    pub semantic: u32,
    pub dtype: u32,
    pub has_bias: u32,
    pub q_heads: u32,
    pub kv_heads: u32,
    pub head_dim: u32,
    pub rotary_dim: u32,
    pub section_t: u32,
    pub section_h: u32,
    pub section_w: u32,
    pub qkv_alignment: u32,
    pub bias_alignment: u32,
    pub q_alignment: u32,
    pub kv_alignment: u32,
    pub position_alignment: u32,
    pub tokens: i64,
    pub cache_capacity: i64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct Bindings {
    pub qkv: *const c_void,
    pub bias: *const c_void,
    pub q: *mut c_void,
    pub k: *mut c_void,
    pub v: *mut c_void,
    pub key_input: *const c_void,
    pub value_input: *const c_void,
    pub position: *const u32,
    pub stream: CudaStream,
    pub theta: f32,
    pub position_offset: i32,
    pub kv_output_offset: i32,
}

unsafe extern "C" {
    pub(crate) fn apxinf_rope_partial(
        data: *mut c_void,
        positions: *const c_void,
        tokens: i64,
        heads: i64,
        head_dim: i64,
        rotary_dim: i64,
        theta: f32,
        stream: CudaStream,
    ) -> i32;
    pub(crate) fn apxinf_rope_launch(
        runtime: Runtime,
        spec: *const Spec,
        bindings: *const Bindings,
    ) -> i32;
    pub(crate) fn apxinf_rope_apply_batched_bf16(
        input: *const c_void,
        output: *mut c_void,
        n_heads: i32,
        head_dim: i32,
        seq_len: i32,
        theta: f32,
        pos_offset: i32,
        stream: CudaStream,
    ) -> i32;
    pub(crate) fn apxinf_rope_apply_mrope_bf16(
        input: *const c_void,
        output: *mut c_void,
        n_heads: i32,
        head_dim: i32,
        seq_len: i32,
        theta: f32,
        pos_ids: *const c_void,
        sec_h: i32,
        sec_w: i32,
        stream: CudaStream,
    ) -> i32;
    pub(crate) fn apxinf_rope_apply_vision_2d_bf16(
        input: *const c_void,
        output: *mut c_void,
        n_heads: i32,
        head_dim: i32,
        seq_len: i32,
        theta: f32,
        pos_ids: *const c_void,
        stream: CudaStream,
    ) -> i32;
}

unsafe extern "C" {
    pub(crate) fn apxinf_rope_fixed_decode_decode_f32(
        input: *const c_void,
        output: *mut c_void,
        head_dim: u32,
        n_heads: u32,
        rope_theta: f32,
        pos_ptr: *const c_void,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn apxinf_rope_fixed_decode_decode_bf16(
        input: *const c_void,
        output: *mut c_void,
        head_dim: u32,
        n_heads: u32,
        rope_theta: f32,
        pos_ptr: *const c_void,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn apxinf_rope_fixed_decode_k_write_bf16(
        k_in: *const c_void,
        k_cache: *mut c_void,
        head_dim: u32,
        n_kv_heads: u32,
        max_seq_len: u32,
        rope_theta: f32,
        pos_ptr: *const c_void,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn apxinf_rope_fixed_decode_mrope_decode_bf16(
        input: *const c_void,
        output: *mut c_void,
        head_dim: u32,
        n_heads: u32,
        theta: f32,
        pos_ids: *const c_void,
        sec_h: u32,
        sec_w: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
}

mod fixed_launchers {
    use std::ffi::c_void;

    use super::Runtime;
    use crate::ffi::raw::cuda_runtime::{cudaError_t, cudaStream_t};

    unsafe extern "C" {
        pub(crate) fn apxinf_rope_mrope_bf16(
            input: *const c_void,
            output: *mut c_void,
            head_dim: u32,
            n_heads: u32,
            seq_len: u32,
            theta: f32,
            pos_ids: *const c_void,
            sec_h: u32,
            sec_w: u32,
            stream: cudaStream_t,
        ) -> cudaError_t;
        pub(crate) fn apxinf_rope_build_vision_rotation_table_f32(
            pos_ids: *const c_void,
            rotation_table: *mut c_void,
            head_dim: u32,
            seq_len: u32,
            theta: f32,
            stream: cudaStream_t,
        ) -> cudaError_t;
        pub(crate) fn apxinf_rope_vision_2d_pair_bf16(
            q: *const c_void,
            k: *const c_void,
            q_out: *mut c_void,
            k_out: *mut c_void,
            head_dim: u32,
            n_heads: u32,
            seq_len: u32,
            theta: f32,
            pos_ids: *const c_void,
            stream: cudaStream_t,
        ) -> cudaError_t;
        pub(crate) fn apxinf_rope_qk_rms_norm_mrope_bf16(
            runtime: Runtime,
            query_input: *const c_void,
            query_weight: *const c_void,
            query_output: *mut c_void,
            key_input: *const c_void,
            key_weight: *const c_void,
            key_output: *mut c_void,
            head_dim: u32,
            query_heads: u32,
            key_heads: u32,
            seq_len: u32,
            eps: f32,
            theta: f32,
            pos_ids: *const c_void,
            sec_h: u32,
            sec_w: u32,
            stream: cudaStream_t,
        ) -> cudaError_t;
        pub(crate) fn apxinf_rope_qkv_split_bias_vision_rope_bf16(
            qkv: *const c_void,
            bias: *const c_void,
            q_out: *mut c_void,
            k_out: *mut c_void,
            v_out: *mut c_void,
            head_dim: u32,
            n_heads: u32,
            seq_len: u32,
            theta: f32,
            pos_ids: *const c_void,
            stream: cudaStream_t,
        ) -> cudaError_t;
        pub(crate) fn apxinf_rope_qkv_split_bias_vision_rope_precomputed_bf16(
            qkv: *const c_void,
            bias: *const c_void,
            q_out: *mut c_void,
            k_out: *mut c_void,
            v_out: *mut c_void,
            head_dim: u32,
            n_heads: u32,
            seq_len: u32,
            rotation_table: *const c_void,
            stream: cudaStream_t,
        ) -> cudaError_t;
        pub(crate) fn apxinf_rope_qkv_split_bias_vision_rope_precomputed_vec2_bf16(
            qkv: *const c_void,
            bias: *const c_void,
            q_out: *mut c_void,
            k_out: *mut c_void,
            v_out: *mut c_void,
            head_dim: u32,
            n_heads: u32,
            seq_len: u32,
            rotation_table: *const c_void,
            stream: cudaStream_t,
        ) -> cudaError_t;
        pub(crate) fn apxinf_rope_split_strided_qkv_bf16(
            qkv: *const c_void,
            q: *mut c_void,
            k: *mut c_void,
            v: *mut c_void,
            tokens: i32,
            hidden: i32,
            stream: cudaStream_t,
        ) -> cudaError_t;
        pub(crate) fn apxinf_rope_half_split_f16(
            input: *const c_void,
            output: *mut c_void,
            head_dim: u32,
            n_heads: u32,
            seq_len: u32,
            rope_theta: f32,
            pos_offset: u32,
            stream: cudaStream_t,
        ) -> cudaError_t;
    }
}
pub(crate) use fixed_launchers::*;
