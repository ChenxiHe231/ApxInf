// Copyright 2026 ApxInf contributors.
#pragma once

#include <cuda_runtime_api.h>

namespace apxinf::cuda::gdn_ops {

// Gated DeltaNet single-token recurrent step.
//
// `state` is [v_heads, v_dim, k_dim] f32 and is updated in place. `q` and `k`
// are [k_heads, k_dim] BF16; `v` is [v_heads, v_dim] BF16. Value head h reads
// k-head `h / (v_heads / k_heads)`, the GQA-style sharing this architecture
// uses (48 value heads over 16 key heads).
//
// `decay` holds log decay per value head (<= 0) and `beta` the delta-rule
// gate; both f32, produced by gdn_decay_and_beta.
int gdn_recurrent_step(void* state, const void* q, const void* k,
                       const void* v, const void* decay, const void* beta,
                       void* output, int v_heads, int k_heads, int v_dim,
                       int k_dim, cudaStream_t stream);

// Per-head RMSNorm followed by the swish output gate.
//
// Qwen3.5 sets `output_gate_type: swish`, so the gate is silu(z); a
// sigmoid-gated implementation would be silently wrong here.
int gdn_gated_norm(const void* input, const void* gate, const void* weight,
                   void* output, int heads, int head_dim, float epsilon,
                   cudaStream_t stream);

// Causal depthwise conv1d advanced by one token, then SiLU.
//
// `window` is [channels, kernel_width] f32 recurrent state holding the last
// `kernel_width` inputs per channel.
int gdn_causal_conv_step(void* window, const void* input, const void* weight,
                         void* output, int channels, int kernel_width,
                         cudaStream_t stream);

// L2-normalize each head in place; the delta rule needs unit-norm q and k.
int gdn_l2_normalize_heads(void* data, int heads, int head_dim, float epsilon,
                           cudaStream_t stream);

// decay = -exp(A_log) * softplus(a + dt_bias), beta = sigmoid(b).
int gdn_decay_and_beta(const void* a, const void* b, const void* a_log,
                       const void* dt_bias, void* decay, void* beta, int heads,
                       cudaStream_t stream);

}  // namespace apxinf::cuda::gdn_ops
