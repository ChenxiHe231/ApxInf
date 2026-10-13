//! Raw bindings for the FlashAttention 2 adapter.

use std::ffi::c_void;

use super::cuda::{cudaError_t, cudaStream_t};

extern "C" {
    // --- action-attention: AOT arm (feature `aot-action-attention`) ---------
    #[cfg(apxinf_aot_action_attention)]
    pub fn apxinf_static_fa4_split_batch_aot_init(stream: cudaStream_t) -> i32;

    #[cfg(apxinf_aot_action_attention)]
    pub fn apxinf_static_fa4_split_batch_aot_ready() -> i32;

    #[cfg(apxinf_aot_action_attention)]
    pub fn apxinf_static_fa4_split_batch_aot_forward(
        q: *const c_void,
        k: *const c_void,
        v: *const c_void,
        q2: *mut c_void,
        partial: *mut c_void,
        lse: *mut f32,
        output: *mut c_void,
        key_tokens: i32,
        stream: cudaStream_t,
    ) -> i32;

    // --- action-attention: in-tree CUDA arm ---------------------------------
    #[cfg(apxinf_fa4_attention_cuda)]
    pub fn apxinf_static_fa4_split_batch_cuda_init(stream: cudaStream_t) -> i32;

    #[cfg(apxinf_fa4_attention_cuda)]
    pub fn apxinf_static_fa4_split_batch_cuda_ready() -> i32;

    #[cfg(apxinf_fa4_attention_cuda)]
    pub fn apxinf_static_fa4_split_batch_cuda_forward(
        q: *const c_void,
        k: *const c_void,
        v: *const c_void,
        q2: *mut c_void,
        partial: *mut c_void,
        lse: *mut f32,
        output: *mut c_void,
        key_tokens: i32,
        stream: cudaStream_t,
    ) -> i32;

    /// Index into the CUTLASS arm's native packed-M LSE buffer. Exported so a
    /// test can check the repack rather than re-deriving the layout.
    #[cfg(apxinf_fa4_attention_cuda)]
    pub fn apxinf_d256_action_lse_index(
        seq_q: i32,
        kv_heads: i32,
        batch_index: i32,
        q_head: i32,
        token: i32,
    ) -> usize;

    // --- language-attention: AOT arm (feature `aot-language-attention`) -----
    #[cfg(apxinf_aot_language_attention)]
    pub fn apxinf_static_fa4_d256_aot_init() -> i32;

    #[cfg(apxinf_aot_language_attention)]
    pub fn apxinf_static_fa4_d256_aot_forward(
        q: *const c_void,
        k: *const c_void,
        v: *const c_void,
        output: *mut c_void,
        stream: cudaStream_t,
    ) -> i32;

    // --- language-attention: in-tree CUDA arm -------------------------------
    #[cfg(apxinf_fa4_attention_cuda)]
    pub fn apxinf_static_fa4_d256_cuda_init() -> i32;

    #[cfg(apxinf_fa4_attention_cuda)]
    pub fn apxinf_static_fa4_d256_cuda_forward(
        q: *const c_void,
        k: *const c_void,
        v: *const c_void,
        output: *mut c_void,
        stream: cudaStream_t,
    ) -> i32;

    // --- vision-attention: AOT arm (feature `aot-vision-attention`) ---------
    #[cfg(apxinf_aot_vision_attention)]
    pub fn apxinf_static_fa4_bf16_vfixed_aot_init(stream: cudaStream_t) -> i32;

    #[cfg(apxinf_aot_vision_attention)]
    pub fn apxinf_static_fa4_bf16_vfixed_aot_ready() -> i32;

    #[cfg(apxinf_aot_vision_attention)]
    pub fn apxinf_static_fa4_bf16_vfixed_aot_forward(
        q: *const c_void,
        k: *const c_void,
        v_strided: *const c_void,
        output: *mut c_void,
        stream: cudaStream_t,
    ) -> i32;

    // --- vision-attention: in-tree CUDA arm --------------------------------
    #[cfg(apxinf_fa4_attention_cuda)]
    pub fn apxinf_static_fa4_bf16_vfixed_cuda_init(stream: cudaStream_t) -> i32;

    #[cfg(apxinf_fa4_attention_cuda)]
    pub fn apxinf_static_fa4_bf16_vfixed_cuda_ready() -> i32;

    #[cfg(apxinf_fa4_attention_cuda)]
    pub fn apxinf_static_fa4_bf16_vfixed_cuda_forward(
        q: *const c_void,
        k: *const c_void,
        v_strided: *const c_void,
        output: *mut c_void,
        stream: cudaStream_t,
    ) -> i32;

    #[cfg(apxinf_fa2_f16_sm100)]
    pub fn apxinf_static_fa2_f16(
        q: *const c_void,
        k: *const c_void,
        v: *const c_void,
        output: *mut c_void,
        softmax_lse: *mut c_void,
        batches: i32,
        query_tokens: i32,
        key_tokens: i32,
        query_heads: i32,
        kv_heads: i32,
        head_dim: i32,
        softmax_scale: f32,
        stream: cudaStream_t,
    ) -> cudaError_t;

    #[cfg(apxinf_fa2_f16_sm100)]
    pub fn apxinf_static_fa2_f16_strided_qkv(
        qkv: *const c_void,
        output: *mut c_void,
        softmax_lse: *mut c_void,
        batch: i32,
        tokens: i32,
        heads: i32,
        head_dim: i32,
        softmax_scale: f32,
        stream: cudaStream_t,
    ) -> cudaError_t;

    #[cfg(apxinf_fa2_direct_e4m3_sm100)]
    pub fn apxinf_static_fa2_f16_direct_e4m3_522(
        q: *const c_void,
        k: *const c_void,
        v: *const c_void,
        output: *mut c_void,
        softmax_lse: *mut c_void,
        batches: i32,
        query_tokens: i32,
        key_tokens: i32,
        query_heads: i32,
        kv_heads: i32,
        head_dim: i32,
        output_scale: f32,
        stream: cudaStream_t,
    ) -> cudaError_t;

    #[cfg(any(apxinf_fa2_sm80, apxinf_fa2_f16_sm100))]
    pub fn apxinf_static_fa2_bf16(
        q: *const c_void,
        k: *const c_void,
        v: *const c_void,
        output: *mut c_void,
        softmax_lse: *mut c_void,
        batches: i32,
        query_tokens: i32,
        key_tokens: i32,
        query_heads: i32,
        kv_heads: i32,
        head_dim: i32,
        softmax_scale: f32,
        stream: cudaStream_t,
    ) -> cudaError_t;

    #[cfg(any(apxinf_fa2_sm80, apxinf_fa2_f16_sm100))]
    pub fn apxinf_static_fa2_bf16_hdim96_bm64(
        q: *const c_void,
        k: *const c_void,
        v: *const c_void,
        output: *mut c_void,
        softmax_lse: *mut c_void,
        batches: i32,
        query_tokens: i32,
        key_tokens: i32,
        query_heads: i32,
        kv_heads: i32,
        head_dim: i32,
        softmax_scale: f32,
        stream: cudaStream_t,
    ) -> cudaError_t;

    #[cfg(any(apxinf_fa2_sm80, apxinf_fa2_f16_sm100))]
    pub fn apxinf_static_fa2_bf16_strided_qkv(
        qkv: *const c_void,
        output: *mut c_void,
        softmax_lse: *mut c_void,
        batch: i32,
        tokens: i32,
        heads: i32,
        head_dim: i32,
        softmax_scale: f32,
        stream: cudaStream_t,
    ) -> cudaError_t;

    #[cfg(any(apxinf_fa2_sm80, apxinf_fa2_f16_sm100))]
    pub fn apxinf_static_fa2_bf16_strided_qkv_hdim96_bm64(
        qkv: *const c_void,
        output: *mut c_void,
        softmax_lse: *mut c_void,
        batch: i32,
        tokens: i32,
        heads: i32,
        head_dim: i32,
        softmax_scale: f32,
        stream: cudaStream_t,
    ) -> cudaError_t;

    #[cfg(any(apxinf_fa2_sm80, apxinf_fa2_f16_sm100))]
    pub fn apxinf_static_fa2_bf16_causal(
        q: *const c_void,
        k: *const c_void,
        v: *const c_void,
        output: *mut c_void,
        softmax_lse: *mut c_void,
        batches: i32,
        query_tokens: i32,
        key_tokens: i32,
        query_heads: i32,
        kv_heads: i32,
        head_dim: i32,
        softmax_scale: f32,
        stream: cudaStream_t,
    ) -> cudaError_t;

    #[cfg(any(apxinf_fa2_sm80, apxinf_fa2_f16_sm100))]
    pub fn apxinf_static_fa2_bf16_splitkv(
        q: *const c_void,
        k: *const c_void,
        v: *const c_void,
        output: *mut c_void,
        softmax_lse: *mut c_void,
        softmax_lse_accum: *mut c_void,
        o_accum: *mut c_void,
        batches: i32,
        query_tokens: i32,
        key_tokens: i32,
        query_heads: i32,
        kv_heads: i32,
        head_dim: i32,
        softmax_scale: f32,
        num_splits: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;
    #[cfg(any(apxinf_fa2_sm80, apxinf_fa2_f16_sm100))]
    pub fn apxinf_static_fa2_bf16_causal_splitkv(
        q: *const c_void,
        k: *const c_void,
        v: *const c_void,
        output: *mut c_void,
        softmax_lse: *mut c_void,
        softmax_lse_accum: *mut c_void,
        o_accum: *mut c_void,
        batches: i32,
        query_tokens: i32,
        key_tokens: i32,
        query_heads: i32,
        kv_heads: i32,
        head_dim: i32,
        softmax_scale: f32,
        num_splits: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;
}
