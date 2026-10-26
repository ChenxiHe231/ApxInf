// Copyright 2026 ApxInf contributors.
// Fixed decode kernels, moved without arithmetic changes from the legacy port.
#pragma once
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
namespace {
#define FIXED_DECODE_BLOCK_SIZE 256
#include "../primitives/rope.cuh"
// Fused single-token RoPE + K-cache write, head-major destination.
__global__ void llmdec_rope_k_write_bf16_kernel(
    const __nv_bfloat16* k_in,
    __nv_bfloat16* k_cache,
    uint32_t head_dim, uint32_t n_kv_heads, uint32_t max_seq_len,
    float rope_theta, const uint32_t* pos_ptr)
{
    uint32_t pair_idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t head_idx = blockIdx.y;
    if (pair_idx >= head_dim / 2) return;

    uint32_t pos = *pos_ptr;
    float freq    = 1.0f / powf(rope_theta, 2.0f * (float)pair_idx / (float)head_dim);
    float angle   = (float)pos * freq;
    float cos_val = cosf(angle);
    float sin_val = sinf(angle);

    uint32_t src_base = head_idx * head_dim;
    uint32_t half = head_dim / 2;
    float x0 = __bfloat162float(k_in[src_base + pair_idx]);
    float x1 = __bfloat162float(k_in[src_base + half + pair_idx]);

    uint32_t dst_base = head_idx * max_seq_len * head_dim + pos * head_dim;
    k_cache[dst_base + pair_idx]        = __float2bfloat16(x0 * cos_val - x1 * sin_val);
    k_cache[dst_base + half + pair_idx] = __float2bfloat16(x0 * sin_val + x1 * cos_val);
}

}  // namespace
