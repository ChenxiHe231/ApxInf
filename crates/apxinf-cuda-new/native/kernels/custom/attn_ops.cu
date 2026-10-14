// Copyright 2026 ApxInf contributors.
//
// Full-attention primitives for Qwen3.5: partial rotary embedding, per-head
// query/key normalization, and the sigmoid output gate.
//
// Two scope notes.
//
// mRoPE: the config specifies mrope_section [11, 11, 10] over the 32 rotary
// pairs, splitting them across temporal, height and width position ids. For
// text-only input all three ids are the token position, so every section
// rotates by the same angle and the result is identical to plain partial
// RoPE. That equivalence is exactly why this kernel is correct for text and
// exactly why it stops being correct once image or video tokens appear -- at
// that point the sections carry different positions and must be applied
// separately.
//
// Pairing convention: this implements the rotate-half form, where pair i is
// (x[i], x[i + rotary_dim/2]). The alternative interleaved form pairs
// (x[2i], x[2i+1]). Both run and both produce plausible logits, so this
// choice must be confirmed against a reference engine before any accuracy
// claim.

#include "attn_ops.h"

#include <cuda_bf16.h>

#include <cstdint>

namespace apxinf::cuda_new::attn_ops {
namespace {

// One block per (token, head). Only the first `rotary_dim` elements rotate;
// partial_rotary_factor 0.25 over head_dim 256 leaves 192 untouched.
__global__ void partial_rope_kernel(__nv_bfloat16* __restrict__ data,
                                    const int32_t* __restrict__ positions,
                                    int tokens, int heads, int head_dim,
                                    int rotary_dim, float theta) {
  const int token = blockIdx.x;
  const int head = blockIdx.y;
  if (token >= tokens || head >= heads) return;

  const int half = rotary_dim / 2;
  const long long base =
      ((long long)token * heads + head) * head_dim;
  const float position = static_cast<float>(positions[token]);

  for (int index = threadIdx.x; index < half; index += blockDim.x) {
    const float inverse_frequency =
        __powf(theta, -2.0f * static_cast<float>(index) /
                          static_cast<float>(rotary_dim));
    const float angle = position * inverse_frequency;
    float sine;
    float cosine;
    __sincosf(angle, &sine, &cosine);

    const float low = __bfloat162float(data[base + index]);
    const float high = __bfloat162float(data[base + half + index]);
    data[base + index] = __float2bfloat16(low * cosine - high * sine);
    data[base + half + index] = __float2bfloat16(high * cosine + low * sine);
  }
}

__device__ __forceinline__ int partial_mrope_axis(int pair, int section_h,
                                                  int section_w) {
  const int remainder = pair % 3;
  if (remainder == 1 && pair < section_h * 3) return 1;
  if (remainder == 2 && pair < section_w * 3) return 2;
  return 0;
}

// Qwen3.5 interleaves the three position axes across the rotary frequency
// pairs. Pair i still rotates the half-split values (i, i + rotary_dim/2).
__global__ void partial_mrope_kernel(__nv_bfloat16* __restrict__ data,
                                     const int32_t* __restrict__ positions,
                                     int tokens, int heads, int head_dim,
                                     int rotary_dim, int section_h,
                                     int section_w, float theta) {
  const int token = blockIdx.x;
  const int head = blockIdx.y;
  if (token >= tokens || head >= heads) return;

  const int half = rotary_dim / 2;
  const long long base = ((long long)token * heads + head) * head_dim;
  for (int pair = threadIdx.x; pair < half; pair += blockDim.x) {
    const int axis = partial_mrope_axis(pair, section_h, section_w);
    const float position = static_cast<float>(positions[token * 3 + axis]);
    const float inverse_frequency =
        __powf(theta, -2.0f * static_cast<float>(pair) /
                          static_cast<float>(rotary_dim));
    float sine;
    float cosine;
    __sincosf(position * inverse_frequency, &sine, &cosine);

    const float low = __bfloat162float(data[base + pair]);
    const float high = __bfloat162float(data[base + half + pair]);
    data[base + pair] = __float2bfloat16(low * cosine - high * sine);
    data[base + half + pair] = __float2bfloat16(high * cosine + low * sine);
  }
}

// Qwen vision uses the first quarter of each half-split rotary head for the
// height axis and the second quarter for width. The explicit BF16 round after
// adding bias preserves the existing Qwen vision kernel's numerical contract.
__global__ void vision_qkv_rope_kernel(
    const __nv_bfloat16* __restrict__ qkv,
    const __nv_bfloat16* __restrict__ bias,
    const int32_t* __restrict__ positions,
    __nv_bfloat16* __restrict__ query,
    __nv_bfloat16* __restrict__ key,
    __nv_bfloat16* __restrict__ value,
    int tokens, int heads, int head_dim, float theta) {
  const int token = blockIdx.x;
  const int half = head_dim / 2;
  const int projection_width = heads * head_dim;
  const int fused_width = 3 * projection_width;
  const int pairs_per_projection = heads * half;

  extern __shared__ float rope[];
  const int quarter = half / 2;
  float* sine = rope;
  float* cosine = rope + 2 * quarter;
  for (int slot = threadIdx.x; slot < 2 * quarter; slot += blockDim.x) {
    const int axis = slot / quarter;
    const int pair = slot - axis * quarter;
    const float position = static_cast<float>(positions[token * 2 + axis]);
    const float frequency =
        powf(theta, -2.0f * static_cast<float>(pair) / half);
    sincosf(position * frequency, &sine[slot], &cosine[slot]);
  }
  __syncthreads();

  for (int work = threadIdx.x; work < 2 * pairs_per_projection;
       work += blockDim.x) {
    const bool is_key = work >= pairs_per_projection;
    const int local = is_key ? work - pairs_per_projection : work;
    const int head = local / half;
    const int pair = local - head * half;
    const int projection_offset = is_key ? projection_width : 0;
    const int source = token * fused_width + projection_offset + head * head_dim;
    const int bias_base = projection_offset + head * head_dim;
    float first = __bfloat162float(qkv[source + pair]);
    float second = __bfloat162float(qkv[source + half + pair]);
    if (bias != nullptr) {
      first += __bfloat162float(bias[bias_base + pair]);
      second += __bfloat162float(bias[bias_base + half + pair]);
    }
    first = __bfloat162float(__float2bfloat16(first));
    second = __bfloat162float(__float2bfloat16(second));
    const int axis = pair < quarter ? 0 : 1;
    const int slot = axis * quarter + (pair < quarter ? pair : pair - quarter);
    __nv_bfloat16* destination = (is_key ? key : query) +
        (token * heads + head) * head_dim;
    destination[pair] =
        __float2bfloat16(first * cosine[slot] - second * sine[slot]);
    destination[half + pair] =
        __float2bfloat16(first * sine[slot] + second * cosine[slot]);
  }

  for (int column = threadIdx.x; column < projection_width;
       column += blockDim.x) {
    const int source = token * fused_width + 2 * projection_width + column;
    float result = __bfloat162float(qkv[source]);
    if (bias != nullptr) {
      result += __bfloat162float(bias[2 * projection_width + column]);
    }
    value[token * projection_width + column] = __float2bfloat16(result);
  }
}

// Per-head RMSNorm without a gate, for q_norm and k_norm.
__global__ void head_rms_norm_kernel(__nv_bfloat16* __restrict__ data,
                                     const __nv_bfloat16* __restrict__ weight,
                                     int rows, int head_dim, float epsilon) {
  const int row = blockIdx.x;
  if (row >= rows) return;
  const long long base = (long long)row * head_dim;

  float sum = 0.0f;
  for (int index = threadIdx.x; index < head_dim; index += blockDim.x) {
    const float value = __bfloat162float(data[base + index]);
    sum += value * value;
  }
  for (int offset = 16; offset > 0; offset >>= 1) {
    sum += __shfl_down_sync(0xFFFFFFFFu, sum, offset);
  }
  __shared__ float partial[32];
  const int warp = threadIdx.x >> 5;
  if ((threadIdx.x & 31) == 0) partial[warp] = sum;
  __syncthreads();
  if (threadIdx.x == 0) {
    const int warps = (blockDim.x + 31) / 32;
    float total = 0.0f;
    for (int index = 0; index < warps; ++index) total += partial[index];
    partial[0] = total;
  }
  __syncthreads();

  const float scale = rsqrtf(partial[0] / static_cast<float>(head_dim) + epsilon);
  for (int index = threadIdx.x; index < head_dim; index += blockDim.x) {
    data[base + index] = __float2bfloat16(
        __bfloat162float(data[base + index]) * scale *
        (1.0f + __bfloat162float(weight[index])));
  }
}

// q_proj emits [tokens, 2 * heads * head_dim]: the query in the first half of
// each head's slot and its gate in the second (`attn_output_gate: true`).
// See apply_output_gate_kernel for what is done with that gate.
__global__ void split_query_and_gate_kernel(
    const __nv_bfloat16* __restrict__ fused, __nv_bfloat16* __restrict__ query,
    __nv_bfloat16* __restrict__ gate, int tokens, int heads, int head_dim) {
  const long long index = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  const long long total = (long long)tokens * heads * head_dim;
  if (index >= total) return;
  const int element = static_cast<int>(index % head_dim);
  const long long head_index = index / head_dim;
  const int head = static_cast<int>(head_index % heads);
  const int token = static_cast<int>(head_index / heads);

  const long long slot =
      ((long long)token * heads + head) * 2 * head_dim + element;
  query[index] = fused[slot];
  gate[index] = fused[slot + head_dim];
}

// data *= sigmoid(gate), elementwise.
//
// `config.json` says `output_gate_type: "swish"`, which reads as silu. It is
// not what the model does. `grep -rn output_gate` over the reference
// implementation finds nothing -- transformers never reads that key -- and
// Qwen3_5Attention.forward (modeling_qwen3_5.py:818) is literally
//
//     attn_output = attn_output * torch.sigmoid(gate)
//
// with no extra factor of `gate`. Where the two official artifacts disagree,
// the code that produced the checkpoint's behaviour wins. Do not "fix" this
// back to silu on the strength of the config key.
//
// The GDN output gate *is* silu (Qwen3_5RMSNormGated sets activation="silu");
// that one lives in gdn_ops.cu and is correct as written.
__global__ void apply_output_gate_kernel(__nv_bfloat16* __restrict__ data,
                                         const __nv_bfloat16* __restrict__ gate,
                                         long long count) {
  const long long index = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  if (index >= count) return;
  const float z = __bfloat162float(gate[index]);
  data[index] =
      __float2bfloat16(__bfloat162float(data[index]) / (1.0f + __expf(-z)));
}

}  // namespace

int partial_rope(void* data, const void* positions, int tokens, int heads,
                 int head_dim, int rotary_dim, float theta,
                 cudaStream_t stream) {
  if (tokens <= 0 || heads <= 0 || head_dim <= 0) return -1;
  if (rotary_dim <= 0 || rotary_dim > head_dim || rotary_dim % 2 != 0) return -2;
  const int half = rotary_dim / 2;
  const int threads = half >= 128 ? 128 : ((half + 31) / 32) * 32;
  partial_rope_kernel<<<dim3(tokens, heads), threads, 0, stream>>>(
      static_cast<__nv_bfloat16*>(data),
      static_cast<const int32_t*>(positions), tokens, heads, head_dim,
      rotary_dim, theta);
  return cudaGetLastError() == cudaSuccess ? 0 : -3;
}

int partial_mrope(void* data, const void* positions, int tokens, int heads,
                  int head_dim, int rotary_dim, int section_t, int section_h,
                  int section_w, float theta, cudaStream_t stream) {
  if (tokens <= 0 || heads <= 0 || head_dim <= 0) return -1;
  if (rotary_dim <= 0 || rotary_dim > head_dim || rotary_dim % 2 != 0) return -2;
  if (section_t < 0 || section_h < 0 || section_w < 0 ||
      section_t + section_h + section_w != rotary_dim / 2) return -3;
  const int half = rotary_dim / 2;
  const int threads = half >= 128 ? 128 : ((half + 31) / 32) * 32;
  partial_mrope_kernel<<<dim3(tokens, heads), threads, 0, stream>>>(
      static_cast<__nv_bfloat16*>(data),
      static_cast<const int32_t*>(positions), tokens, heads, head_dim,
      rotary_dim, section_h, section_w, theta);
  return cudaGetLastError() == cudaSuccess ? 0 : -4;
}

int vision_qkv_rope(const void* qkv, const void* bias,
                    const void* positions, void* query, void* key,
                    void* value, int tokens, int heads, int head_dim,
                    float theta, cudaStream_t stream) {
  if (qkv == nullptr || positions == nullptr || query == nullptr ||
      key == nullptr || value == nullptr || tokens <= 0 || heads <= 0 ||
      head_dim <= 0 || head_dim > 256 || head_dim % 4 != 0 ||
      !(theta > 0.0f)) {
    return -1;
  }
  const size_t shared = static_cast<size_t>(head_dim) * sizeof(float);
  vision_qkv_rope_kernel<<<tokens, 256, shared, stream>>>(
      static_cast<const __nv_bfloat16*>(qkv),
      static_cast<const __nv_bfloat16*>(bias),
      static_cast<const int32_t*>(positions),
      static_cast<__nv_bfloat16*>(query),
      static_cast<__nv_bfloat16*>(key),
      static_cast<__nv_bfloat16*>(value), tokens, heads, head_dim, theta);
  return cudaGetLastError() == cudaSuccess ? 0 : -2;
}

int head_rms_norm(void* data, const void* weight, int rows, int head_dim,
                  float epsilon, cudaStream_t stream) {
  if (rows <= 0 || head_dim <= 0) return -1;
  const int threads = head_dim >= 256 ? 256 : ((head_dim + 31) / 32) * 32;
  head_rms_norm_kernel<<<rows, threads, 0, stream>>>(
      static_cast<__nv_bfloat16*>(data),
      static_cast<const __nv_bfloat16*>(weight), rows, head_dim, epsilon);
  return cudaGetLastError() == cudaSuccess ? 0 : -2;
}

int split_query_and_gate(const void* fused, void* query, void* gate,
                         int tokens, int heads, int head_dim,
                         cudaStream_t stream) {
  if (tokens <= 0 || heads <= 0 || head_dim <= 0) return -1;
  const long long total = (long long)tokens * heads * head_dim;
  const int threads = 256;
  const long long blocks = (total + threads - 1) / threads;
  split_query_and_gate_kernel<<<static_cast<int>(blocks), threads, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(fused),
      static_cast<__nv_bfloat16*>(query), static_cast<__nv_bfloat16*>(gate),
      tokens, heads, head_dim);
  return cudaGetLastError() == cudaSuccess ? 0 : -2;
}

int apply_output_gate(void* data, const void* gate, long long count,
                      cudaStream_t stream) {
  if (count <= 0) return -1;
  const int threads = 256;
  const long long blocks = (count + threads - 1) / threads;
  apply_output_gate_kernel<<<static_cast<int>(blocks), threads, 0, stream>>>(
      static_cast<__nv_bfloat16*>(data),
      static_cast<const __nv_bfloat16*>(gate), count);
  return cudaGetLastError() == cudaSuccess ? 0 : -2;
}

}  // namespace apxinf::cuda_new::attn_ops
