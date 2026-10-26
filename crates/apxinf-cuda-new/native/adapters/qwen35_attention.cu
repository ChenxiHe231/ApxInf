// Copyright 2026 ApxInf contributors.
//
// Exact Qwen3.5-35B-A3B full-attention core for batch size one.  The cache is
// genuinely E4M3: BF16 K/V are divided by their calibrated dequant scales on
// write and E4M3 codes are multiplied by those scales on read.

#include "../include/apxinf_cuda/qwen35_attention.h"
#include "../framework/runtime_internal.h"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#include <cmath>
#include <climits>
#include <cstdint>

namespace {

using apxinf::framework::Failure;

constexpr int kQHeads = APXINF_QWEN35_Q_HEADS;
constexpr int kKvHeads = APXINF_QWEN35_KV_HEADS;
constexpr int kHeadDim = APXINF_QWEN35_HEAD_DIM;
constexpr int kRotaryDim = APXINF_QWEN35_ROTARY_DIM;
constexpr int kRotaryPairs = kRotaryDim / 2;
constexpr int kThreads = kHeadDim;
constexpr int kWarps = kThreads / 32;

__device__ float block_sum(float value, float* partial) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffffU, value, offset);
  }
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  if (lane == 0) partial[warp] = value;
  __syncthreads();
  if (threadIdx.x == 0) {
    float total = 0.0F;
#pragma unroll
    for (int index = 0; index < kWarps; ++index) total += partial[index];
    partial[0] = total;
  }
  __syncthreads();
  return partial[0];
}
__device__ float rotary_value(const float* normalized, int dim, int token,
                              int tokens, const int32_t* positions,
                              float theta) {
  if (dim >= kRotaryDim) return normalized[dim];
  const int pair = dim < kRotaryPairs ? dim : dim - kRotaryPairs;
  // Qwen3.5 mrope_interleaved=true with mrope_section=[11,11,10]: pair
  // indices 0,3,... use T; 1,4,... use H; 2,5,... use W.
  const int position_channel = pair % 3;
  const float position =
      static_cast<float>(positions[position_channel * tokens + token]);
  const float inverse_frequency =
      __powf(theta, -2.0F * static_cast<float>(pair) /
                        static_cast<float>(kRotaryDim));
  float sine;
  float cosine;
  __sincosf(position * inverse_frequency, &sine, &cosine);
  const float low = normalized[pair];
  const float high = normalized[kRotaryPairs + pair];
  return dim < kRotaryPairs ? low * cosine - high * sine
                            : high * cosine + low * sine;
}

__global__ void prepare_query_kernel(
    const __nv_bfloat16* __restrict__ projected,
    const __nv_bfloat16* __restrict__ norm_weight,
    const int32_t* __restrict__ positions, __nv_bfloat16* __restrict__ query,
    __nv_bfloat16* __restrict__ gate, int tokens, float theta,
    float epsilon) {
  const int token = blockIdx.x;
  const int head = blockIdx.y;
  const int dim = threadIdx.x;
  if (token >= tokens || head >= kQHeads || dim >= kHeadDim) return;

  __shared__ float partial[kWarps];
  __shared__ float normalized[kHeadDim];
  const int64_t source =
      (static_cast<int64_t>(token) * kQHeads + head) * (2 * kHeadDim);
  const int64_t destination =
      (static_cast<int64_t>(token) * kQHeads + head) * kHeadDim;
  const float value = __bfloat162float(projected[source + dim]);
  const float total = block_sum(value * value, partial);
  const float scale = rsqrtf(total / static_cast<float>(kHeadDim) + epsilon);
  // Qwen3.5 Q/K norms are zero-centered: effective weight is 1 + w.
  normalized[dim] = __bfloat162float(__float2bfloat16(
      value * scale * (1.0F + __bfloat162float(norm_weight[dim]))));
  gate[destination + dim] = projected[source + kHeadDim + dim];
  __syncthreads();
  query[destination + dim] = __float2bfloat16(
      rotary_value(normalized, dim, token, tokens, positions, theta));
}

__global__ void append_kv_kernel(
    const __nv_bfloat16* __restrict__ projected_key,
    const __nv_bfloat16* __restrict__ projected_value,
    const __nv_bfloat16* __restrict__ norm_weight,
    const int32_t* __restrict__ positions,
    __nv_fp8_e4m3* __restrict__ key_cache,
    __nv_fp8_e4m3* __restrict__ value_cache, int tokens, int cache_offset,
    float theta, float epsilon, float key_scale, float value_scale) {
  const int token = blockIdx.x;
  const int head = blockIdx.y;
  const int dim = threadIdx.x;
  if (token >= tokens || head >= kKvHeads || dim >= kHeadDim) return;

  __shared__ float partial[kWarps];
  __shared__ float normalized[kHeadDim];
  const int64_t source =
      (static_cast<int64_t>(token) * kKvHeads + head) * kHeadDim;
  const float key = __bfloat162float(projected_key[source + dim]);
  const float total = block_sum(key * key, partial);
  const float norm_scale =
      rsqrtf(total / static_cast<float>(kHeadDim) + epsilon);
  normalized[dim] = __bfloat162float(__float2bfloat16(
      key * norm_scale * (1.0F + __bfloat162float(norm_weight[dim]))));
  __syncthreads();

  const int64_t cache_index =
      (static_cast<int64_t>(cache_offset + token) * kKvHeads + head) *
          kHeadDim +
      dim;
  const float rotated =
      rotary_value(normalized, dim, token, tokens, positions, theta);
  key_cache[cache_index] =
      __nv_fp8_e4m3(__bfloat162float(__float2bfloat16(rotated)) / key_scale);
  value_cache[cache_index] = __nv_fp8_e4m3(
      __bfloat162float(projected_value[source + dim]) / value_scale);
}

__global__ void causal_attention_kernel(
    const __nv_bfloat16* __restrict__ query,
    const __nv_bfloat16* __restrict__ gate,
    const __nv_fp8_e4m3* __restrict__ key_cache,
    const __nv_fp8_e4m3* __restrict__ value_cache,
    __nv_bfloat16* __restrict__ output, int tokens, int cache_offset,
    float key_scale, float value_scale) {
  const int token = blockIdx.x;
  const int query_head = blockIdx.y;
  const int dim = threadIdx.x;
  if (token >= tokens || query_head >= kQHeads || dim >= kHeadDim) return;

  __shared__ float partial[kWarps];
  __shared__ float online_max;
  __shared__ float denominator;
  __shared__ float old_weight;
  __shared__ float new_weight;
  const int kv_head = query_head / (kQHeads / kKvHeads);
  const int64_t query_index =
      (static_cast<int64_t>(token) * kQHeads + query_head) * kHeadDim + dim;
  const float q = __bfloat162float(query[query_index]);
  float accumulator = 0.0F;

  if (dim == 0) {
    online_max = -CUDART_INF_F;
    denominator = 0.0F;
  }
  __syncthreads();

  const int inclusive_length = cache_offset + token + 1;
  for (int position = 0; position < inclusive_length; ++position) {
    const int64_t cache_index =
        (static_cast<int64_t>(position) * kKvHeads + kv_head) * kHeadDim +
        dim;
    const float k = static_cast<float>(key_cache[cache_index]) * key_scale;
    // head_dim^-0.5 = 1/16 for head_dim 256.
    const float score = block_sum(q * k, partial) * 0.0625F;
    if (dim == 0) {
      const float next_max = fmaxf(online_max, score);
      old_weight = __expf(online_max - next_max);
      new_weight = __expf(score - next_max);
      denominator = denominator * old_weight + new_weight;
      online_max = next_max;
    }
    __syncthreads();
    const float v = static_cast<float>(value_cache[cache_index]) * value_scale;
    accumulator = accumulator * old_weight + v * new_weight;
    __syncthreads();
  }

  const float gate_value = __bfloat162float(gate[query_index]);
  const float sigmoid = 1.0F / (1.0F + __expf(-gate_value));
  output[query_index] =
      __float2bfloat16((accumulator / denominator) * sigmoid);
}

bool valid_spec(const apxinf_qwen35_attention_spec_t& spec) {
  return spec.version == APXINF_QWEN35_ATTENTION_SPEC_VERSION &&
         spec.tokens > 0 && spec.tokens <= INT_MAX &&
         spec.cache_capacity > 0 && spec.cache_capacity <= INT_MAX &&
         spec.cache_offset <= INT_MAX &&
         static_cast<uint64_t>(spec.cache_offset) + spec.tokens <=
             spec.cache_capacity &&
         std::isfinite(spec.rope_theta) && spec.rope_theta > 0.0F &&
         std::isfinite(spec.rms_epsilon) && spec.rms_epsilon > 0.0F &&
         std::isfinite(spec.key_scale) && spec.key_scale > 0.0F &&
         std::isfinite(spec.value_scale) && spec.value_scale > 0.0F;
}

bool valid_bindings(const apxinf_qwen35_attention_bindings_t& bindings) {
  return bindings.projected_query_gate != nullptr &&
         bindings.projected_key != nullptr &&
         bindings.projected_value != nullptr &&
         bindings.query_norm != nullptr && bindings.key_norm != nullptr &&
         bindings.positions != nullptr && bindings.key_cache != nullptr &&
         bindings.value_cache != nullptr && bindings.query != nullptr &&
         bindings.gate != nullptr && bindings.output != nullptr;
}

}  // namespace

extern "C" apxinf_status_t apxinf_qwen35_attention_fp8(
    apxinf_runtime_t runtime, const apxinf_qwen35_attention_spec_t* spec,
    const apxinf_qwen35_attention_bindings_t* bindings) {
  return apxinf::framework::abi_boundary([&] {
    if (runtime == nullptr || spec == nullptr || bindings == nullptr ||
        !valid_spec(*spec) || !valid_bindings(*bindings)) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                    "invalid Qwen3.5 FP8 attention arguments");
    }
    apxinf::framework::check_cuda(cudaSetDevice(runtime->device));
    auto stream = static_cast<cudaStream_t>(bindings->stream);
    const float key_scale = spec->key_scale;
    const float value_scale = spec->value_scale;
    const dim3 query_grid(spec->tokens, kQHeads);
    prepare_query_kernel<<<query_grid, kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(bindings->projected_query_gate),
        static_cast<const __nv_bfloat16*>(bindings->query_norm),
        static_cast<const int32_t*>(bindings->positions),
        static_cast<__nv_bfloat16*>(bindings->query),
        static_cast<__nv_bfloat16*>(bindings->gate), spec->tokens,
        spec->rope_theta, spec->rms_epsilon);
    apxinf::framework::check_cuda(cudaGetLastError());

    const dim3 kv_grid(spec->tokens, kKvHeads);
    append_kv_kernel<<<kv_grid, kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(bindings->projected_key),
        static_cast<const __nv_bfloat16*>(bindings->projected_value),
        static_cast<const __nv_bfloat16*>(bindings->key_norm),
        static_cast<const int32_t*>(bindings->positions),
        static_cast<__nv_fp8_e4m3*>(bindings->key_cache),
        static_cast<__nv_fp8_e4m3*>(bindings->value_cache), spec->tokens,
        spec->cache_offset, spec->rope_theta, spec->rms_epsilon, key_scale,
        value_scale);
    apxinf::framework::check_cuda(cudaGetLastError());

    causal_attention_kernel<<<query_grid, kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(bindings->query),
        static_cast<const __nv_bfloat16*>(bindings->gate),
        static_cast<const __nv_fp8_e4m3*>(bindings->key_cache),
        static_cast<const __nv_fp8_e4m3*>(bindings->value_cache),
        static_cast<__nv_bfloat16*>(bindings->output), spec->tokens,
        spec->cache_offset, key_scale, value_scale);
    apxinf::framework::check_cuda(cudaGetLastError());
  });
}
