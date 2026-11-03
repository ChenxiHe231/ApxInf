#pragma once

#include "gdn_types.h"

#ifdef __cplusplus
extern "C" {
#endif

apxinf_status_t apxinf_gdn_launch(
    apxinf_runtime_t runtime, const apxinf_gdn_spec_t* spec,
    const apxinf_gdn_bindings_t* bindings);

apxinf_status_t apxinf_gdn_prepare_fixed(
    int32_t device, int32_t head_k_dim, int32_t head_v_dim,
    int32_t chunk_size);

int32_t apxinf_gdn_qk_bf16_supported(
    int32_t device, int32_t recurrent, int32_t head_k_dim,
    int32_t head_v_dim, int32_t chunk_size);

apxinf_status_t apxinf_gdn_chunk_scan_prepare(int32_t device);

// Scratch bytes apxinf_gdn_prefill needs for TMA rewrites.
int64_t apxinf_gdn_prefill_workspace_bytes(int64_t v_heads,
                                           int64_t num_seqs);
apxinf_status_t apxinf_gdn_recurrent_verify(
    void* state, const void* q, const void* k, const void* v,
    const void* decay, const void* beta, void* output, void* checkpoints,
    int64_t tokens, int64_t v_heads, int64_t k_heads, int64_t v_dim,
    int64_t k_dim, int64_t q_row_stride, int64_t k_row_stride,
    int64_t v_row_stride, apxinf_cuda_stream_t stream);

apxinf_status_t apxinf_gdn_causal_conv_verify(
    void* window, const void* input, const void* weight, void* output,
    void* checkpoints, int64_t tokens, int64_t channels,
    int64_t kernel_width, apxinf_cuda_stream_t stream);

apxinf_status_t apxinf_gdn_l2_normalize_heads_seq(
    void* data, int64_t tokens, int64_t heads, int64_t head_dim,
    int64_t row_stride, float epsilon, apxinf_cuda_stream_t stream);


#ifdef __cplusplus
}
#endif
