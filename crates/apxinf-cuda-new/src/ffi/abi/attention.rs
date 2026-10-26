use std::ffi::{c_char, c_void};

pub(crate) use super::types::Policy;
use super::types::{CudaStream, Runtime};
use crate::ffi::raw::cuda_runtime::{cudaError_t, cudaStream_t};

pub(crate) const SPEC_VERSION: u32 = 4;
pub(crate) const SEMANTIC_DENSE: u32 = 0;
pub(crate) const SEMANTIC_KV_CACHE: u32 = 1;
pub(crate) const SEMANTIC_SEGMENTED: u32 = 2;
pub(crate) const SEMANTIC_PACKED_QKV: u32 = 3;

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub(crate) struct Spec {
    pub version: u32,
    pub semantic: u32,
    pub dtype: u32,
    pub output_dtype: u32,
    pub mask: u32,
    pub q_alignment: u32,
    pub k_alignment: u32,
    pub v_alignment: u32,
    pub output_alignment: u32,
    pub offsets_alignment: u32,
    pub batch: i64,
    pub query_tokens: i64,
    pub key_tokens: i64,
    pub key_capacity: i64,
    pub query_heads: i64,
    pub kv_heads: i64,
    pub head_dim: i64,
    pub query_start: i64,
    pub segments: i64,
    pub max_segment_tokens: i64,
    pub offsets_hash: u64,
    pub scale_is_default: u32,
    pub dynamic_decode: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub(crate) struct Bindings {
    pub query: *const c_void,
    pub key: *const c_void,
    pub value: *const c_void,
    pub offsets: *const c_void,
    pub decode_meta: *const u32,
    pub output: *mut c_void,
    pub stream: CudaStream,
    pub scale: f32,
    pub output_scale: f32,
}

pub(crate) type Execution = *mut c_void;

unsafe extern "C" {
    pub(crate) fn apxinf_attention_split_query_and_gate(
        fused: *const c_void,
        query: *mut c_void,
        gate: *mut c_void,
        tokens: i64,
        heads: i64,
        head_dim: i64,
        stream: CudaStream,
    ) -> i32;
    pub(crate) fn apxinf_attention_apply_output_gate(
        data: *mut c_void,
        gate: *const c_void,
        count: i64,
        stream: CudaStream,
    ) -> i32;
    pub(crate) fn apxinf_decode_attention_workspace_bytes() -> i64;
    pub(crate) fn apxinf_decode_attention(
        runtime: Runtime,
        query: *const c_void,
        key_cache: *const c_void,
        value_cache: *const c_void,
        output: *mut c_void,
        workspace: *mut c_void,
        workspace_bytes: i64,
        key_tokens: i64,
        scale: f32,
        stream: CudaStream,
    ) -> i32;
    pub(crate) fn apxinf_attention_prepare(
        runtime: Runtime,
        spec: *const Spec,
        policy: *const Policy,
        bindings: *const Bindings,
        execution: *mut Execution,
    ) -> i32;
    pub(crate) fn apxinf_attention_enqueue(execution: Execution) -> i32;
    pub(crate) fn apxinf_attention_destroy(execution: Execution);
    #[cfg(test)]
    pub(crate) fn apxinf_attention_summary(execution: Execution) -> *const c_char;
    #[cfg(test)]
    pub(crate) fn apxinf_attention_test_validate_candidates(
        runtime: Runtime,
        spec: *const Spec,
        policy: *const Policy,
        bindings: *const Bindings,
        expected_output: *const f32,
        expected_output_len: u64,
    ) -> i32;
}

unsafe extern "C" {
    pub(crate) fn apxinf_attention_fixed_decode_softmax_decode_f32(
        scores: *const c_void,
        output: *mut c_void,
        cols: u32,
        n_heads: u32,
        pos_ptr: *const c_void,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn apxinf_attention_fixed_decode_softmax_f32(
        scores: *const c_void,
        output: *mut c_void,
        cols: u32,
        rows: u32,
        kv_offset: u32,
        n_heads: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn apxinf_attention_fixed_decode_softmax_bf16(
        scores: *const c_void,
        output: *mut c_void,
        cols: u32,
        rows: u32,
        kv_offset: u32,
        n_heads: u32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    pub(crate) fn apxinf_attention_fixed_decode_flash_attn_decode_bf16(
        q: *const c_void,
        k_cache: *const c_void,
        v_cache: *const c_void,
        out: *mut c_void,
        n_heads: u32,
        n_kv_heads: u32,
        head_dim: u32,
        bucket_kv_len: u32,
        max_seq_len: u32,
        scale: f32,
        pos_ptr: *const c_void,
        stream: cudaStream_t,
    ) -> cudaError_t;
}

mod fixed_launchers {
    use std::ffi::c_void;

    use crate::ffi::raw::cuda_runtime::{cudaError_t, cudaStream_t};

    unsafe extern "C" {
        // FIX (implement_r8): tiled block-per-row softmax (fp32 in, bf16 out) for the
        // route-3 composed vision attention; defined in adapters/custom_kernels.cu.
        pub(crate) fn apxinf_attention_row_softmax_f32_bf16(
            input: *const c_void,
            output: *mut c_void,
            cols: u32,
            rows: u32,
            stream: cudaStream_t,
        ) -> cudaError_t;
        // FIX (implement_final_r20): tiled block-per-row CAUSAL fp32 softmax (fp32 in, fp32
        // out, in-place safe) for the Option A composed-causal budget repair; defined in
        // adapters/custom_kernels.cu with the attention.cuh row = seq_pos*n_heads + head
        // contract (valid_cols = min(seq_pos + kv_offset + 1, cols), masked cells exact 0.0f).
        #[allow(clippy::too_many_arguments)]
        pub(crate) fn apxinf_attention_row_softmax_causal_f32(
            input: *const c_void,
            output: *mut c_void,
            cols: u32,
            rows: u32,
            kv_offset: u32,
            n_heads: u32,
            stream: cudaStream_t,
        ) -> cudaError_t;
        #[allow(clippy::too_many_arguments)]
        pub(crate) fn apxinf_attention_full_prepare_bf16(
            fused: *const c_void,
            q_norm_w: *const c_void,
            k_norm_w: *const c_void,
            cos: *const c_void,
            sin: *const c_void,
            q_out: *mut c_void,
            k_cache: *mut c_void,
            v_cache: *mut c_void,
            seq: i32,
            cache_offset: i32,
            q_heads: i32,
            kv_heads: i32,
            head_dim: i32,
            rotary_dim: i32,
            fused_width: i64,
            cache_width: i64,
            eps: f32,
            stream: cudaStream_t,
        ) -> cudaError_t;
        pub(crate) fn apxinf_attention_sigmoid_gate_mul_bf16(
            attn: *mut c_void,
            fused: *const c_void,
            rows: i32,
            heads: i32,
            head_dim: i32,
            fused_width: i64,
            stream: cudaStream_t,
        ) -> cudaError_t;
        #[allow(clippy::too_many_arguments)]
        pub(crate) fn apxinf_attention_expert_qkv_prepare_bf16(
            fused: *const c_void,
            q_norm_w: *const c_void,
            k_norm_w: *const c_void,
            cos: *const c_void,
            sin: *const c_void,
            q_out: *mut c_void,
            gate_out: *mut c_void,
            k_out: *mut c_void,
            v_out: *mut c_void,
            seq: i32,
            q_heads: i32,
            kv_heads: i32,
            head_dim: i32,
            rotary_dim: i32,
            fused_width: i64,
            eps: f32,
            stream: cudaStream_t,
        ) -> cudaError_t;
        pub(crate) fn apxinf_attention_expert_sigmoid_gate_mul_bf16(
            attn: *mut c_void,
            gate: *const c_void,
            count: i64,
            stream: cudaStream_t,
        ) -> cudaError_t;
        pub(crate) fn apxinf_attention_cublas_gqa_f16(
            q: *const c_void,
            k: *const c_void,
            v: *const c_void,
            output: *mut c_void,
            query_tokens: i32,
            key_tokens: i32,
            q_heads: i32,
            kv_heads: i32,
            head_dim: i32,
            stream: cudaStream_t,
        ) -> i32;
        pub(crate) fn apxinf_attention_cublas_gqa_causal_f16(
            q: *const c_void,
            k: *const c_void,
            v: *const c_void,
            output: *mut c_void,
            query_tokens: i32,
            key_tokens: i32,
            q_heads: i32,
            kv_heads: i32,
            head_dim: i32,
            key_offset: i32,
            stream: cudaStream_t,
        ) -> i32;
    }
}
pub(crate) use fixed_launchers::*;
