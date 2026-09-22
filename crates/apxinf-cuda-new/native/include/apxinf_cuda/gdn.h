#pragma once

#include "types.h"
#include "status.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Gated DeltaNet primitives. See kernels/custom/gdn_ops.h for the recurrence.

   The update rule is implemented from the architecture's documented form and
   has not yet been checked against a reference engine running this
   checkpoint. */

apxinf_status_t apxinf_gdn_recurrent_step(
    void* state, const void* q, const void* k, const void* v,
    const void* decay, const void* beta, void* output, int64_t v_heads,
    int64_t k_heads, int64_t v_dim, int64_t k_dim,
    apxinf_cuda_stream_t stream);

apxinf_status_t apxinf_gdn_gated_norm(const void* input, const void* gate,
                                      const void* weight, void* output,
                                      int64_t heads, int64_t head_dim,
                                      float epsilon,
                                      apxinf_cuda_stream_t stream);

apxinf_status_t apxinf_gdn_causal_conv_step(void* window, const void* input,
                                            const void* weight, void* output,
                                            int64_t channels,
                                            int64_t kernel_width,
                                            apxinf_cuda_stream_t stream);

apxinf_status_t apxinf_gdn_causal_conv_forward(
    const void* input, const void* weight, void* output, void* window,
    int64_t tokens, int64_t channels, int64_t kernel_width,
    apxinf_cuda_stream_t stream);

apxinf_status_t apxinf_gdn_decay_and_beta_seq(
    const void* a, const void* b, const void* a_log, const void* dt_bias,
    void* decay, void* beta, int64_t tokens, int64_t heads,
    apxinf_cuda_stream_t stream);

apxinf_status_t apxinf_gdn_gated_norm_seq(
    const void* input, const void* gate, const void* weight, void* output,
    int64_t tokens, int64_t heads, int64_t head_dim, float epsilon,
    apxinf_cuda_stream_t stream);

apxinf_status_t apxinf_gdn_l2_normalize_heads(void* data, int64_t heads,
                                              int64_t head_dim, float epsilon,
                                              apxinf_cuda_stream_t stream);

apxinf_status_t apxinf_gdn_decay_and_beta(const void* a, const void* b,
                                          const void* a_log,
                                          const void* dt_bias, void* decay,
                                          void* beta, int64_t heads,
                                          apxinf_cuda_stream_t stream);


apxinf_status_t apxinf_gdn_chunk_scan(
    const void* q, const void* k, const void* v, const void* g,
    const void* beta, void* out, void* state, int64_t seq_padded,
    int64_t v_heads, int64_t k_heads, int64_t chunk_size, int64_t k_dim,
    int64_t num_chunks, apxinf_cuda_stream_t stream);

#ifdef __cplusplus
}
#endif
