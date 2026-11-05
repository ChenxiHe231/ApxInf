// Copyright 2026 ApxInf contributors.
//
// Exact Qwen3.5-35B-A3B full-attention kernels for batch size one.  The legacy
// entry point retains static-scale E4M3 semantics.  The dynamic path stores
// post-RoPE K and projected V with dynamic per-token-per-KV-head scales, then
// decodes directly from FP8 without materializing a BF16 cache.

#include "../include/apxinf_cuda/qwen35_attention.h"
#include "../framework/runtime_internal.h"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>
#include <mma.h>
#include <math_constants.h>
#include <mma.h>

#include <cfloat>
#include <cmath>
#include <climits>
#include <cstdint>
#include <cstdlib>
#include <cstring>

namespace {

using apxinf::framework::Failure;

constexpr int kQHeads = APXINF_QWEN35_Q_HEADS;
constexpr int kKvHeads = APXINF_QWEN35_KV_HEADS;
constexpr int kHeadDim = APXINF_QWEN35_HEAD_DIM;
constexpr int kRotaryDim = APXINF_QWEN35_ROTARY_DIM;
constexpr int kRotaryPairs = kRotaryDim / 2;
constexpr int kThreads = kHeadDim;
constexpr int kWarps = kThreads / 32;
constexpr int kQueriesPerKvHead = kQHeads / kKvHeads;
constexpr int kScorePositionsPerBlock = 32;
constexpr int kValuePositionsPerTile = 128;

__device__ float warp_sum(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffffU, value, offset);
  }
  return value;
}
constexpr int kDecodeSplitTokens = APXINF_QWEN35_DECODE_SPLIT_TOKENS;
constexpr float kFp8Max = 448.0F;
constexpr float kMinDynamicScale = 1.0e-12F;

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

__device__ float block_max(float value, float* partial) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    value = fmaxf(value, __shfl_down_sync(0xffffffffU, value, offset));
  }
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  if (lane == 0) partial[warp] = value;
  __syncthreads();
  if (threadIdx.x == 0) {
    float maximum = 0.0F;
#pragma unroll
    for (int index = 0; index < kWarps; ++index) {
      maximum = fmaxf(maximum, partial[index]);
    }
    partial[0] = maximum;
  }
  __syncthreads();
  const float result = partial[0];
  __syncthreads();  // All warps must read before a later reduction reuses partial.
  return result;
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

// Materialize one FP32 score per query/head/cache position. One block owns a
// KV head and all eight GQA query heads that share it. K is staged once for 32
// positions rather than being reread by eight independent query-head blocks.
__global__ void attention_scores_kernel(
    const __nv_bfloat16* __restrict__ query,
    const __nv_fp8_e4m3* __restrict__ key_cache,
    float* __restrict__ scores, int tokens, int cache_offset,
    int cache_capacity, float key_scale) {
  const int token = blockIdx.z;
  const int kv_head = blockIdx.y;
  const int lane = threadIdx.x & 31;
  const int local_query_head = threadIdx.x >> 5;
  if (token >= tokens || kv_head >= kKvHeads) return;

  __shared__ float shared_query[kQueriesPerKvHead][kHeadDim];
  __shared__ float shared_key[kScorePositionsPerBlock][kHeadDim];
  const int dim = threadIdx.x;
  const int query_head = kv_head * kQueriesPerKvHead + local_query_head;
  const int64_t query_base =
      (static_cast<int64_t>(token) * kQHeads + query_head) * kHeadDim;
#pragma unroll
  for (int component = lane; component < kHeadDim; component += 32) {
    shared_query[local_query_head][component] =
        __bfloat162float(query[query_base + component]);
  }

  const int first = blockIdx.x * kScorePositionsPerBlock;
  const int inclusive_length = cache_offset + token + 1;
#pragma unroll
  for (int item = 0; item < kScorePositionsPerBlock; ++item) {
    const int position = first + item;
    shared_key[item][dim] =
        position < inclusive_length
            ? static_cast<float>(key_cache[
                  (static_cast<int64_t>(position) * kKvHeads + kv_head) *
                      kHeadDim +
                  dim]) *
                  key_scale
            : 0.0F;
  }
  __syncthreads();

#pragma unroll
  for (int item = 0; item < kScorePositionsPerBlock; ++item) {
    const int position = first + item;
    if (position >= inclusive_length) continue;
    float dot = 0.0F;
#pragma unroll
    for (int component = lane; component < kHeadDim; component += 32) {
      dot += shared_query[local_query_head][component] *
             shared_key[item][component];
    }
    dot = warp_sum(dot);
    if (lane == 0) {
      const int64_t score_index =
          (static_cast<int64_t>(token) * kQHeads + query_head) *
              cache_capacity +
          position;
      // head_dim^-0.5 = 1/16 for head_dim 256.
      scores[score_index] = dot * 0.0625F;
    }
  }
}

__global__ void attention_split_output_kernel(
    const __nv_fp8_e4m3* __restrict__ value_cache,
    const float* __restrict__ scores, float* __restrict__ partial_output,
    float* __restrict__ partial_stats, int tokens, int cache_offset,
    int cache_capacity, int splits) {
  const int split = blockIdx.x;
  const int kv_head = blockIdx.y;
  const int token = blockIdx.z;
  const int dim = threadIdx.x;
  if (token >= tokens || kv_head >= kKvHeads || split >= splits ||
      dim >= kHeadDim)
    return;

  __shared__ float reduction[kThreads];
  __shared__ float weights[kQueriesPerKvHead][kValuePositionsPerTile];
  __shared__ float maxima[kQueriesPerKvHead];
  const int inclusive_length = cache_offset + token + 1;
  const int split_width = (inclusive_length + splits - 1) / splits;
  const int split_begin = split * split_width;
  const int split_end = min(inclusive_length, split_begin + split_width);

#pragma unroll
  for (int local_head = 0; local_head < kQueriesPerKvHead; ++local_head) {
    const int query_head = kv_head * kQueriesPerKvHead + local_head;
    const int64_t score_base =
        (static_cast<int64_t>(token) * kQHeads + query_head) *
        cache_capacity;
    float local_max = -CUDART_INF_F;
    for (int position = split_begin + dim; position < split_end;
         position += kThreads) {
      local_max = fmaxf(local_max, scores[score_base + position]);
    }
    reduction[dim] = local_max;
    __syncthreads();
#pragma unroll
    for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
      if (dim < stride)
        reduction[dim] = fmaxf(reduction[dim], reduction[dim + stride]);
      __syncthreads();
    }
    if (dim == 0) maxima[local_head] = reduction[0];
    __syncthreads();

    float local_sum = 0.0F;
    for (int position = split_begin + dim; position < split_end;
         position += kThreads) {
      local_sum += __expf(scores[score_base + position] - maxima[local_head]);
    }
    reduction[dim] = local_sum;
    __syncthreads();
#pragma unroll
    for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
      if (dim < stride) reduction[dim] += reduction[dim + stride];
      __syncthreads();
    }
    if (dim == 0) {
      const int64_t stats_base =
          ((((static_cast<int64_t>(token) * kKvHeads + kv_head) *
                 APXINF_QWEN35_ATTENTION_MAX_SPLITS +
             split) *
                kQueriesPerKvHead +
            local_head) *
           2);
      partial_stats[stats_base] = maxima[local_head];
      partial_stats[stats_base + 1] = reduction[0];
    }
    __syncthreads();
  }

  float accumulators[kQueriesPerKvHead] = {};
  for (int tile = split_begin; tile < split_end;
       tile += kValuePositionsPerTile) {
    if (dim < kValuePositionsPerTile) {
      const int position = tile + dim;
#pragma unroll
      for (int local_head = 0; local_head < kQueriesPerKvHead; ++local_head) {
        const int query_head = kv_head * kQueriesPerKvHead + local_head;
        const int64_t score_base =
            (static_cast<int64_t>(token) * kQHeads + query_head) *
            cache_capacity;
        weights[local_head][dim] =
            position < split_end
                ? __expf(scores[score_base + position] - maxima[local_head]) *
                      1.0F
                : 0.0F;
      }
    }
    __syncthreads();
    const int count = min(kValuePositionsPerTile, split_end - tile);
    for (int item = 0; item < count; ++item) {
      const int64_t cache_index =
          (static_cast<int64_t>(tile + item) * kKvHeads + kv_head) *
              kHeadDim +
          dim;
      const float value = static_cast<float>(value_cache[cache_index]);
#pragma unroll
      for (int local_head = 0; local_head < kQueriesPerKvHead; ++local_head) {
        accumulators[local_head] += value * weights[local_head][item];
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (int local_head = 0; local_head < kQueriesPerKvHead; ++local_head) {
    const int64_t output_index =
        ((((static_cast<int64_t>(token) * kKvHeads + kv_head) *
               APXINF_QWEN35_ATTENTION_MAX_SPLITS +
           split) *
              kQueriesPerKvHead +
          local_head) *
             kHeadDim +
         dim);
    partial_output[output_index] = accumulators[local_head];
  }
}

// Fuse score formation, online softmax, and value accumulation for one
// split/KV-head. A 32-position tile stages K once for all eight GQA heads and V
// once for all eight output vectors. This removes the full
// [tokens,heads,capacity] score round-trip while preserving the split-wise
// online-softmax contract consumed by attention_combine_kernel.
__global__ void attention_fused_split_kernel(
    const __nv_bfloat16* __restrict__ query,
    const __nv_fp8_e4m3* __restrict__ key_cache,
    const __nv_fp8_e4m3* __restrict__ value_cache,
    float* __restrict__ partial_output, float* __restrict__ partial_stats,
    int tokens, int cache_offset, int splits, float key_scale) {
  const int split = blockIdx.x;
  const int kv_head = blockIdx.y;
  const int token = blockIdx.z;
  const int dim = threadIdx.x;
  const int lane = dim & 31;
  const int local_query_head = dim >> 5;
  if (token >= tokens || kv_head >= kKvHeads || split >= splits ||
      dim >= kHeadDim)
    return;

  // Keep Q/K in BF16. FP8 values (and the unit TRT scale used by this model)
  // are represented exactly, while halving shared storage from 64 KiB to
  // 32 KiB and allowing multiple split blocks to reside on one SM.
  // The QK product is a 8x256 by 256x32 GEMM. Pad Q to 16 rows so two
  // warp-level BF16 tensor-core tiles can produce all 8x32 scores. The old
  // path evaluated the same product with scalar FP32 FMAs, leaving the tensor
  // cores idle during the dominant part of long-context verification.
  __shared__ __align__(16) __nv_bfloat16 shared_query[16][kHeadDim];
  __shared__ __nv_bfloat16 shared_key[kScorePositionsPerBlock][kHeadDim];
  __shared__ __align__(16) float shared_scores[16][kScorePositionsPerBlock];
  __shared__ float shared_weights[kQueriesPerKvHead][kScorePositionsPerBlock];
  __shared__ float maxima[kQueriesPerKvHead];
  __shared__ float denominators[kQueriesPerKvHead];
  __shared__ float rescale[kQueriesPerKvHead];

  for (int index = dim; index < 16 * kHeadDim; index += kThreads) {
    const int head = index / kHeadDim;
    const int component = index % kHeadDim;
    if (head < kQueriesPerKvHead) {
      const int query_head = kv_head * kQueriesPerKvHead + head;
      const int64_t query_index =
          (static_cast<int64_t>(token) * kQHeads + query_head) * kHeadDim +
          component;
      shared_query[head][component] = query[query_index];
    } else {
      shared_query[head][component] = __float2bfloat16(0.0F);
    }
  }
  if (dim < kQueriesPerKvHead) {
    maxima[dim] = -CUDART_INF_F;
    denominators[dim] = 0.0F;
  }
  __syncthreads();

  const int inclusive_length = cache_offset + token + 1;
  const int split_width = (inclusive_length + splits - 1) / splits;
  const int split_begin = split * split_width;
  const int split_end = min(inclusive_length, split_begin + split_width);
  float accumulators[kQueriesPerKvHead] = {};

  for (int tile = split_begin; tile < split_end;
       tile += kScorePositionsPerBlock) {
    const int count = min(kScorePositionsPerBlock, split_end - tile);
    for (int index = dim; index < count * kHeadDim; index += kThreads) {
      const int item = index / kHeadDim;
      const int component = index % kHeadDim;
      const int64_t cache_index =
          (static_cast<int64_t>(tile + item) * kKvHeads + kv_head) *
              kHeadDim +
          component;
      shared_key[item][component] = __float2bfloat16(
          static_cast<float>(key_cache[cache_index]) * key_scale);
    }
    __syncthreads();

    const int warp = dim >> 5;
    if (warp < 2) {
      using namespace nvcuda;
      wmma::fragment<wmma::accumulator, 16, 16, 16, float> accumulator;
      wmma::fill_fragment(accumulator, 0.0F);
#pragma unroll
      for (int component = 0; component < kHeadDim; component += 16) {
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16,
                       wmma::row_major>
            query_fragment;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16,
                       wmma::col_major>
            key_fragment;
        wmma::load_matrix_sync(query_fragment, &shared_query[0][component],
                               kHeadDim);
        wmma::load_matrix_sync(key_fragment, &shared_key[warp * 16][component],
                               kHeadDim);
        wmma::mma_sync(accumulator, query_fragment, key_fragment, accumulator);
      }
#pragma unroll
      for (int element = 0; element < accumulator.num_elements; ++element)
        accumulator.x[element] *= 0.0625F;
      wmma::store_matrix_sync(&shared_scores[0][warp * 16], accumulator,
                              kScorePositionsPerBlock, wmma::mem_row_major);
    }
    __syncthreads();

    if (dim < kQueriesPerKvHead) {
      const int head = dim;
      float tile_max = -CUDART_INF_F;
      for (int item = 0; item < count; ++item)
        tile_max = fmaxf(tile_max, shared_scores[head][item]);
      const float next_max = fmaxf(maxima[head], tile_max);
      const float alpha =
          maxima[head] == -CUDART_INF_F ? 0.0F : __expf(maxima[head] - next_max);
      float next_denominator = denominators[head] * alpha;
      for (int item = 0; item < count; ++item) {
        const float weight = __expf(shared_scores[head][item] - next_max);
        shared_weights[head][item] = weight;
        next_denominator += weight;
      }
      maxima[head] = next_max;
      denominators[head] = next_denominator;
      rescale[head] = alpha;
    }
    __syncthreads();

#pragma unroll
    for (int head = 0; head < kQueriesPerKvHead; ++head)
      accumulators[head] *= rescale[head];
    for (int item = 0; item < count; ++item) {
      const int64_t cache_index =
          (static_cast<int64_t>(tile + item) * kKvHeads + kv_head) *
              kHeadDim +
          dim;
      const float value = static_cast<float>(value_cache[cache_index]);
#pragma unroll
      for (int head = 0; head < kQueriesPerKvHead; ++head)
        accumulators[head] += value * shared_weights[head][item];
    }
    __syncthreads();
  }

#pragma unroll
  for (int head = 0; head < kQueriesPerKvHead; ++head) {
    const int64_t output_index =
        ((((static_cast<int64_t>(token) * kKvHeads + kv_head) *
               APXINF_QWEN35_ATTENTION_MAX_SPLITS +
           split) *
              kQueriesPerKvHead +
          head) *
             kHeadDim +
         dim);
    partial_output[output_index] = accumulators[head];
    if (dim == 0) {
      const int64_t stats_base =
          ((((static_cast<int64_t>(token) * kKvHeads + kv_head) *
                 APXINF_QWEN35_ATTENTION_MAX_SPLITS +
             split) *
                kQueriesPerKvHead +
            head) *
           2);
      partial_stats[stats_base] = maxima[head];
      partial_stats[stats_base + 1] = denominators[head];
    }
  }
}

__global__ void attention_combine_kernel(
    const __nv_bfloat16* __restrict__ gate,
    const float* __restrict__ partial_output,
    const float* __restrict__ partial_stats,
    __nv_bfloat16* __restrict__ output, int tokens, int splits,
    float value_scale) {
  const int token = blockIdx.x;
  const int kv_head = blockIdx.y;
  const int dim = threadIdx.x;
  if (token >= tokens || kv_head >= kKvHeads || dim >= kHeadDim) return;

  __shared__ float global_maxima[kQueriesPerKvHead];
  __shared__ float global_denominators[kQueriesPerKvHead];
  if (dim == 0) {
#pragma unroll
    for (int local_head = 0; local_head < kQueriesPerKvHead; ++local_head) {
      float maximum = -CUDART_INF_F;
      for (int split = 0; split < splits; ++split) {
        const int64_t stats_base =
            ((((static_cast<int64_t>(token) * kKvHeads + kv_head) *
                   APXINF_QWEN35_ATTENTION_MAX_SPLITS +
               split) *
                  kQueriesPerKvHead +
              local_head) *
             2);
        maximum = fmaxf(maximum, partial_stats[stats_base]);
      }
      float denominator = 0.0F;
      for (int split = 0; split < splits; ++split) {
        const int64_t stats_base =
            ((((static_cast<int64_t>(token) * kKvHeads + kv_head) *
                   APXINF_QWEN35_ATTENTION_MAX_SPLITS +
               split) *
                  kQueriesPerKvHead +
              local_head) *
             2);
        denominator += partial_stats[stats_base + 1] *
                       __expf(partial_stats[stats_base] - maximum);
      }
      global_maxima[local_head] = maximum;
      global_denominators[local_head] = denominator;
    }
  }
  __syncthreads();

#pragma unroll
  for (int local_head = 0; local_head < kQueriesPerKvHead; ++local_head) {
    float accumulator = 0.0F;
    for (int split = 0; split < splits; ++split) {
      const int64_t stats_base =
          ((((static_cast<int64_t>(token) * kKvHeads + kv_head) *
                 APXINF_QWEN35_ATTENTION_MAX_SPLITS +
             split) *
                kQueriesPerKvHead +
            local_head) *
           2);
      const int64_t output_index =
          ((((static_cast<int64_t>(token) * kKvHeads + kv_head) *
                 APXINF_QWEN35_ATTENTION_MAX_SPLITS +
             split) *
                kQueriesPerKvHead +
            local_head) *
               kHeadDim +
           dim);
      accumulator += partial_output[output_index] *
                     __expf(partial_stats[stats_base] -
                            global_maxima[local_head]);
    }
    accumulator /= global_denominators[local_head];
    const int query_head = kv_head * kQueriesPerKvHead + local_head;
    const int64_t query_index =
        (static_cast<int64_t>(token) * kQHeads + query_head) * kHeadDim +
        dim;
    const float gate_value = __bfloat162float(gate[query_index]);
    const float sigmoid = 1.0F / (1.0F + __expf(-gate_value));
    output[query_index] =
        __float2bfloat16(accumulator * value_scale * sigmoid);
  }
}

__global__ void dynamic_kv_append_kernel(
    const __nv_bfloat16* __restrict__ key,
    const __nv_bfloat16* __restrict__ value,
    __nv_fp8_e4m3* __restrict__ key_cache,
    __nv_fp8_e4m3* __restrict__ value_cache,
    float* __restrict__ key_scales, float* __restrict__ value_scales,
    int tokens, int cache_offset) {
  const int token = blockIdx.x;
  const int head = blockIdx.y;
  const int dim = threadIdx.x;
  if (token >= tokens || head >= kKvHeads || dim >= kHeadDim) return;

  __shared__ float partial[kWarps];
  __shared__ float key_values[kHeadDim];
  __shared__ float value_values[kHeadDim];
  const int64_t source =
      (static_cast<int64_t>(token) * kKvHeads + head) * kHeadDim + dim;
  const float key_value = __bfloat162float(key[source]);
  const float value_value = __bfloat162float(value[source]);
  key_values[dim] = key_value;
  value_values[dim] = value_value;

  const float key_absmax = block_max(fabsf(key_value), partial);
  const float value_absmax = block_max(fabsf(value_value), partial);
  const float key_scale = fmaxf(key_absmax / kFp8Max, kMinDynamicScale);
  const float value_scale = fmaxf(value_absmax / kFp8Max, kMinDynamicScale);
  const int64_t scale_index =
      static_cast<int64_t>(cache_offset + token) * kKvHeads + head;
  if (dim == 0) {
    key_scales[scale_index] = key_scale;
    value_scales[scale_index] = value_scale;
  }
  const int64_t cache_index = scale_index * kHeadDim + dim;
  key_cache[cache_index] = __nv_fp8_e4m3(key_values[dim] / key_scale);
  value_cache[cache_index] =
      __nv_fp8_e4m3(value_values[dim] / value_scale);
}

// Fused K-Norm -> RoPE -> dynamic quantize/write. Preserve both BF16
// rounding boundaries of the separate kernels, without global intermediates.
__global__ void dynamic_kv_prepare_kernel(
    const __nv_bfloat16* __restrict__ key,
    const __nv_bfloat16* __restrict__ value,
    const __nv_bfloat16* __restrict__ norm_weight,
    const int32_t* __restrict__ positions,
    __nv_fp8_e4m3* __restrict__ key_cache,
    __nv_fp8_e4m3* __restrict__ value_cache,
    float* __restrict__ key_scales, float* __restrict__ value_scales,
    int tokens, int cache_offset, float theta, float epsilon) {
  const int token = blockIdx.x;
  const int head = blockIdx.y;
  const int dim = threadIdx.x;
  __shared__ float partial[kWarps];
  __shared__ float normalized[kHeadDim];
  const int64_t source =
      (static_cast<int64_t>(token) * kKvHeads + head) * kHeadDim + dim;
  const float k = __bfloat162float(key[source]);
  const float total = block_sum(k * k, partial);
  const float norm_scale = rsqrtf(total / kHeadDim + epsilon);
  normalized[dim] = __bfloat162float(__float2bfloat16(
      k * norm_scale * (1.0F + __bfloat162float(norm_weight[dim]))));
  __syncthreads();
  const float rotated = __bfloat162float(__float2bfloat16(
      rotary_value(normalized, dim, token, tokens, positions, theta)));
  const float v = __bfloat162float(value[source]);
  const float ks = fmaxf(block_max(fabsf(rotated), partial) / kFp8Max,
                        kMinDynamicScale);
  const float vs = fmaxf(block_max(fabsf(v), partial) / kFp8Max,
                        kMinDynamicScale);
  const int64_t row = static_cast<int64_t>(cache_offset + token) * kKvHeads + head;
  if (dim == 0) { key_scales[row] = ks; value_scales[row] = vs; }
  key_cache[row * kHeadDim + dim] = __nv_fp8_e4m3(rotated / ks);
  value_cache[row * kHeadDim + dim] = __nv_fp8_e4m3(v / vs);
}

// E4M3 normal values are a BF16 exponent rebias and mantissa shift. The
// small subnormal range is also exact in BF16. This avoids the sm_87 CUDA
// scalar FP8->half normalization loop; four contiguous cache bytes share a load.
__device__ __forceinline__ unsigned short e4m3_bf16_bits(unsigned int code) {
  const unsigned int magnitude = code & 0x7fU;
  const unsigned short sign = (code & 0x80U) << 8;
  if (magnitude == 0x7fU) return 0x7fffU;
  if (magnitude < 8U) {
    const __nv_bfloat16 small = __float2bfloat16(__uint2float_rn(magnitude) * 0x1p-9F);
    return sign | __bfloat16_as_ushort(small);
  }
  return sign | (0x3c00U + (magnitude << 4));
}

__device__ __forceinline__ void store_e4m3x4_bf16(unsigned int bytes, unsigned int* out) {
  out[0] = e4m3_bf16_bits(bytes & 0xffU) |
      (static_cast<unsigned int>(e4m3_bf16_bits((bytes >> 8) & 0xffU)) << 16);
  out[1] = e4m3_bf16_bits((bytes >> 16) & 0xffU) |
      (static_cast<unsigned int>(e4m3_bf16_bits(bytes >> 24)) << 16);
}

// One CTA owns one KV head and one 128/256-token split.  Its eight warps own the
// eight GQA query heads sharing that KV head.  K/V are loaded once into shared
// memory and reused by all eight query heads, avoiding eight redundant global
// cache reads. Synchronization and online softmax rescaling are amortized across tiles.
template <int kSplitTokens>
__device__ __forceinline__ void dynamic_decode_simt(
    const __nv_bfloat16* __restrict__ query,
    const __nv_fp8_e4m3* __restrict__ key_cache,
    const __nv_fp8_e4m3* __restrict__ value_cache,
    const float* __restrict__ key_scales,
    const float* __restrict__ value_scales,
    float* __restrict__ partial_max, float* __restrict__ partial_sum,
    float* __restrict__ partial_output, int valid_tokens, int max_splits) {
  const int split = blockIdx.x;
  const int kv_head = blockIdx.y;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int query_head = kv_head * kQueriesPerKvHead + warp;
  const int begin = split * kSplitTokens;
  const int end = begin + min(kSplitTokens, valid_tokens - begin);
  if (begin >= end || kv_head >= kKvHeads || warp >= kQueriesPerKvHead) return;

  constexpr int kTile = 8;
  static_assert(kTile <= 32);
  __shared__ float shared_key[kTile][kHeadDim];
  __shared__ float shared_value[kTile][kHeadDim];
  __shared__ float shared_key_scale[kTile];
  __shared__ float shared_value_scale[kTile];
  float q[8];
  float accumulator[8];
#pragma unroll
  for (int item = 0; item < 8; ++item) {
    q[item] = __bfloat162float(query[query_head * kHeadDim + lane + item * 32]);
    accumulator[item] = 0.0F;
  }
  float running_max = -CUDART_INF_F;
  float running_sum = 0.0F;
  // Load eight cache rows per tile; synchronize per tile instead of per token.
  // Every query warp reuses the same coalesced K/V tile.
  for (int tile_begin = begin; tile_begin < end; tile_begin += kTile) {
    const int tile_count = min(kTile, end - tile_begin);
    for (int t = 0; t < tile_count; ++t) {
      const int64_t row = static_cast<int64_t>(tile_begin + t) * kKvHeads + kv_head;
      shared_key[t][threadIdx.x] = __bfloat162float(__ushort_as_bfloat16(
          e4m3_bf16_bits(key_cache[row * kHeadDim + threadIdx.x].__x)));
      shared_value[t][threadIdx.x] = __bfloat162float(__ushort_as_bfloat16(
          e4m3_bf16_bits(value_cache[row * kHeadDim + threadIdx.x].__x)));
      if (threadIdx.x == 0) {
        shared_key_scale[t] = key_scales[row];
        shared_value_scale[t] = value_scales[row];
      }
    }
    __syncthreads();
    // Each query warp computes a tile of scores, then distributes one score
    // to each active lane. Softmax exponentials run in parallel, and the
    // running accumulator is rescaled once per tile rather than per token.
    float lane_score = -CUDART_INF_F;
    for (int t = 0; t < tile_count; ++t) {
      float dot = 0.0F;
#pragma unroll
      for (int item = 0; item < 8; ++item) {
        dot += q[item] * shared_key[t][lane + item * 32];
      }
      dot = __shfl_sync(0xffffffffU, warp_sum(dot), 0);
      if (lane == t) lane_score = dot * shared_key_scale[t] * 0.0625F;
    }
    float tile_max = lane_score;
#pragma unroll
    for (int delta = 16; delta > 0; delta >>= 1) {
      tile_max = fmaxf(tile_max, __shfl_down_sync(0xffffffffU, tile_max, delta));
    }
    tile_max = __shfl_sync(0xffffffffU, tile_max, 0);
    const float next_max = fmaxf(running_max, tile_max);
    const float old_weight = __expf(running_max - next_max);
    const float probability = lane < tile_count ? __expf(lane_score - next_max) : 0.0F;
    const float tile_sum = __shfl_sync(0xffffffffU, warp_sum(probability), 0);
    running_sum = running_sum * old_weight + tile_sum;
    running_max = next_max;
    const float lane_weight = lane < tile_count ? probability * shared_value_scale[lane] : 0.0F;
#pragma unroll
    for (int item = 0; item < 8; ++item) accumulator[item] *= old_weight;
    for (int t = 0; t < tile_count; ++t) {
      const float weighted_value_scale = __shfl_sync(0xffffffffU, lane_weight, t);
#pragma unroll
      for (int item = 0; item < 8; ++item) {
        accumulator[item] += shared_value[t][lane + item * 32] * weighted_value_scale;
      }
    }
    __syncthreads();
  }

  const int64_t partial_index =
      static_cast<int64_t>(query_head) * max_splits + split;
  if (lane == 0) {
    partial_max[partial_index] = running_max;
    partial_sum[partial_index] = running_sum;
  }
#pragma unroll
  for (int item = 0; item < 8; ++item) {
    const int dim = lane + item * 32;
    partial_output[partial_index * kHeadDim + dim] = accumulator[item];
  }
}

template <int kSplitTokens>
__global__ void dynamic_decode_split_kernel(
    const __nv_bfloat16* query, const __nv_fp8_e4m3* key_cache,
    const __nv_fp8_e4m3* value_cache, const float* key_scales,
    const float* value_scales, float* partial_max, float* partial_sum,
    float* partial_output, int valid_tokens, int max_splits) {
  dynamic_decode_simt<kSplitTokens>(query, key_cache, value_cache, key_scales,
      value_scales, partial_max, partial_sum, partial_output, valid_tokens, max_splits);
}

// Every nonzero finite E4M3 magnitude lies in [2^-9,448]. A margin of
// 512 keeps each BF16 tensor product normal and finite. Inputs outside this
// range use SIMT, including normal operands whose product could underflow.
__device__ __forceinline__ bool safe_tensor_operand(float value) {
  const float magnitude = fabsf(value);
  return isfinite(value) && (value == 0.0F ||
      (magnitude >= FLT_MIN * 512.0F && magnitude <= FLT_MAX / 512.0F));
}

// BF16 Tensor Cores consume raw E4M3 values exactly. QK applies each token's
// scale after the matrix product. PV decomposes the FP32 probability*V-scale
// into three BF16 components, rather than rounding the probability once.
// Unsafe components/products use the FP32 SIMT path for that tile.
template <int kSplitTokens, int kQkWarps = 1>
__global__ void dynamic_decode_tensor_kernel(
    const __nv_bfloat16* __restrict__ query,
    const __nv_fp8_e4m3* __restrict__ key_cache,
    const __nv_fp8_e4m3* __restrict__ value_cache,
    const float* __restrict__ key_scales,
    const float* __restrict__ value_scales,
    float* __restrict__ partial_max, float* __restrict__ partial_sum,
    float* __restrict__ partial_output, int valid_tokens, int max_splits) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  namespace wmma = nvcuda::wmma;
  constexpr int kTile = 32;
  const int split = blockIdx.x;
  const int kv_head = blockIdx.y;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int query_head = kv_head * kQueriesPerKvHead + warp;
  const int begin = split * kSplitTokens;
  const int end = begin + min(kSplitTokens, valid_tokens - begin);
  if (begin >= end || kv_head >= kKvHeads) return;

  // K, V and the tile's FP32 PV output share storage across separate phases.
  // All WMMA pointers/leading dimensions meet the 32-byte alignment contract.
  __shared__ __align__(32) unsigned char data[kTile * kHeadDim * sizeof(__nv_bfloat16)];
  __shared__ __align__(32) __nv_bfloat16 shared_query[kQueriesPerKvHead][kHeadDim];
  __shared__ __align__(32) unsigned char weights[3 * kQueriesPerKvHead * kTile * sizeof(__nv_bfloat16)];
  __shared__ float ks[kTile];
  __shared__ float vs[kTile];
  auto* raw = reinterpret_cast<__nv_bfloat16*>(data);
  auto* scores = reinterpret_cast<float*>(weights);
  auto* parts = reinterpret_cast<__nv_bfloat16*>(weights);
  constexpr int kPartElements = kQueriesPerKvHead * kTile;
  static_assert(kQkWarps == 1 || kQkWarps == 4 || kQkWarps == 8);
  static_assert(kPartElements == kThreads);
  // One full 8x32 partial per QK producer warp. The single-warp path keeps the
  // original zero-extra-shared-memory schedule.
  __shared__ __align__(32)
      float qk_partial[kQkWarps > 1 ? kQkWarps * kPartElements : 1];
  bool unsafe_query = false;
  for (int i = threadIdx.x; i < kQueriesPerKvHead * kHeadDim; i += kThreads) {
    const __nv_bfloat16 q = query[kv_head * kQueriesPerKvHead * kHeadDim + i];
    shared_query[i / kHeadDim][i % kHeadDim] = q;
    const float qf = __bfloat162float(q);
    unsafe_query |= !safe_tensor_operand(qf);
  }
  const bool scalar_qk = __syncthreads_or(unsafe_query);
  float accumulator[8] = {};
  float running_max = -CUDART_INF_F;
  float running_sum = 0.0F;

  for (int first = begin; first < end; first += kTile) {
    const int count = min(kTile, end - first);
    for (int group = threadIdx.x; group < kTile * kHeadDim / 4; group += kThreads) {
      const int t = group / (kHeadDim / 4);
      const int d = (group % (kHeadDim / 4)) * 4;
      const int64_t index = (static_cast<int64_t>(first + t) * kKvHeads + kv_head) * kHeadDim + d;
      const unsigned int bytes = t < count ?
          *reinterpret_cast<const unsigned int*>(key_cache + index) : 0U;
      store_e4m3x4_bf16(bytes, reinterpret_cast<unsigned int*>(raw) + group * 2);
    }
    if (threadIdx.x < kTile) {
      const int t = threadIdx.x;
      const int64_t row = static_cast<int64_t>(first + t) * kKvHeads + kv_head;
      ks[t] = t < count ? key_scales[row] : 0.0F;
      vs[t] = t < count ? value_scales[row] : 0.0F;
    }
    __syncthreads();
    if (scalar_qk) {
      for (int t = 0; t < kTile; ++t) {
        float dot = 0.0F;
#pragma unroll
        for (int item = 0; item < 8; ++item) {
          const int d = lane + item * 32;
          dot += __bfloat162float(shared_query[warp][d]) * __bfloat162float(raw[t * kHeadDim + d]);
        }
        dot = warp_sum(dot);
        if (lane == 0) scores[warp * kTile + t] = dot;
      }
    } else if (warp < kQkWarps) {
      wmma::fragment<wmma::matrix_a, 8, 32, 16, __nv_bfloat16, wmma::row_major> a;
      wmma::fragment<wmma::matrix_b, 8, 32, 16, __nv_bfloat16, wmma::col_major> b;
      wmma::fragment<wmma::accumulator, 8, 32, 16, float> c;
      wmma::fill_fragment(c, 0.0F);
      constexpr int kDimsPerWarp = kHeadDim / kQkWarps;
#pragma unroll
      for (int d = warp * kDimsPerWarp; d < (warp + 1) * kDimsPerWarp; d += 16) {
        wmma::load_matrix_sync(a, &shared_query[0][d], kHeadDim);
        wmma::load_matrix_sync(b, raw + d, kHeadDim);
        wmma::mma_sync(c, a, b, c);
      }
      wmma::store_matrix_sync(
          kQkWarps == 1 ? scores : qk_partial + warp * kPartElements, c, kTile,
          wmma::mem_row_major);
    }
    __syncthreads(); // QK finished; K storage may become V storage.
    if (kQkWarps > 1 && !scalar_qk) {
      // Every thread owns exactly one score; the fold order is fixed so the
      // reduction is bit-reproducible across runs and warp counts.
      const int index = threadIdx.x;
      float value = qk_partial[index];
#pragma unroll
      for (int producer = 1; producer < kQkWarps; ++producer) {
        value += qk_partial[producer * kPartElements + index];
      }
      scores[index] = value;
    }
    float score = lane < count ? scores[warp * kTile + lane] * ks[lane] * 0.0625F : -CUDART_INF_F;
    float maximum = score;
#pragma unroll
    for (int delta = 16; delta > 0; delta >>= 1) {
      maximum = fmaxf(maximum, __shfl_down_sync(0xffffffffU, maximum, delta));
    }
    maximum = __shfl_sync(0xffffffffU, maximum, 0);
    const float next_max = fmaxf(running_max, maximum);
    const float old_weight = __expf(running_max - next_max);
    const float probability = lane < count ? __expf(score - next_max) : 0.0F;
    const float tile_sum = __shfl_sync(0xffffffffU, warp_sum(probability), 0);
    running_sum = running_sum * old_weight + tile_sum;
    running_max = next_max;
    const float weight = probability * vs[lane];
    const __nv_bfloat16 high = __float2bfloat16(weight);
    const float hf = __bfloat162float(high);
    const float r1 = weight - hf;
    const __nv_bfloat16 middle = __float2bfloat16(r1);
    const float mf = __bfloat162float(middle);
    const float r2 = r1 - mf;
    const __nv_bfloat16 low = __float2bfloat16(r2);
    const float lf = __bfloat162float(low);
    const bool unsafe_weight = !isfinite(weight) || r2 - lf != 0.0F ||
        !safe_tensor_operand(hf) || !safe_tensor_operand(mf) ||
        !safe_tensor_operand(lf);
    // Also ensures every warp has read FP32 scores before BF16 parts reuse
    // the same bytes. The fallback keeps tiny FP32 weights intact.
    const bool scalar_pv = __syncthreads_or(unsafe_weight);
    const int wi = warp * kTile + lane;
    if (scalar_pv) {
      scores[wi] = weight;
    } else {
      parts[wi] = high;
      parts[kPartElements + wi] = middle;
      parts[2 * kPartElements + wi] = low;
    }
    for (int group = threadIdx.x; group < kTile * kHeadDim / 4; group += kThreads) {
      const int t = group / (kHeadDim / 4);
      const int d = (group % (kHeadDim / 4)) * 4;
      const int64_t index = (static_cast<int64_t>(first + t) * kKvHeads + kv_head) * kHeadDim + d;
      const unsigned int bytes = t < count ?
          *reinterpret_cast<const unsigned int*>(value_cache + index) : 0U;
      store_e4m3x4_bf16(bytes, reinterpret_cast<unsigned int*>(raw) + group * 2);
    }
    __syncthreads();
    if (scalar_pv) {
#pragma unroll
      for (int item = 0; item < 8; ++item) accumulator[item] *= old_weight;
      for (int t = 0; t < count; ++t) {
        const float w = scores[warp * kTile + t];
#pragma unroll
        for (int item = 0; item < 8; ++item) {
          accumulator[item] += __bfloat162float(raw[t * kHeadDim + lane + item * 32]) * w;
        }
      }
    } else {
      wmma::fragment<wmma::matrix_a, 8, 32, 16, __nv_bfloat16, wmma::row_major> a;
      wmma::fragment<wmma::matrix_b, 8, 32, 16, __nv_bfloat16, wmma::row_major> b;
      wmma::fragment<wmma::accumulator, 8, 32, 16, float> c;
      wmma::fill_fragment(c, 0.0F);
#pragma unroll
      for (int t = 0; t < kTile; t += 16) {
        wmma::load_matrix_sync(b, raw + t * kHeadDim + warp * 32, kHeadDim);
#pragma unroll
        for (int part = 0; part < 3; ++part) {
          wmma::load_matrix_sync(a, parts + part * kPartElements + t, kTile);
          wmma::mma_sync(c, a, b, c);
        }
      }
      __syncthreads(); // All V fragments are loaded before output reuse.
      auto* tile_output = reinterpret_cast<float*>(data);
      wmma::store_matrix_sync(tile_output + warp * 32, c, kHeadDim, wmma::mem_row_major);
      __syncthreads();
#pragma unroll
      for (int item = 0; item < 8; ++item) {
        accumulator[item] = accumulator[item] * old_weight +
            tile_output[warp * kHeadDim + lane + item * 32];
      }
    }
    __syncthreads(); // All PV consumers finish before the next K tile.
  }
  const int64_t pi = static_cast<int64_t>(query_head) * max_splits + split;
  if (lane == 0) {
    partial_max[pi] = running_max;
    partial_sum[pi] = running_sum;
  }
#pragma unroll
  for (int item = 0; item < 8; ++item) {
    partial_output[pi * kHeadDim + lane + item * 32] = accumulator[item];
  }
#else
  dynamic_decode_simt<kSplitTokens>(query, key_cache, value_cache, key_scales,
      value_scales, partial_max, partial_sum, partial_output, valid_tokens, max_splits);
#endif
}

__device__ __forceinline__ uint4 load_decode_raw16(
    const __nv_fp8_e4m3* cache, int first, int count, int kv_head,
    int group) {
  const int token = group >> 4;
  const int dim = (group & 15) * 16;
  const int cache_token = token < count ? first + token : first;
  const int64_t index =
      (static_cast<int64_t>(cache_token) * kKvHeads + kv_head) *
          kHeadDim +
      dim;
  return *reinterpret_cast<const uint4*>(cache + index);
}

__device__ __forceinline__ void store_decode_raw16(
    uint4 bytes, __nv_bfloat16* data, int group) {
  auto* output = reinterpret_cast<unsigned int*>(data) + group * 8;
  store_e4m3x4_bf16(bytes.x, output);
  store_e4m3x4_bf16(bytes.y, output + 2);
  store_e4m3x4_bf16(bytes.z, output + 4);
  store_e4m3x4_bf16(bytes.w, output + 6);
}

// The vector path has exactly two uint4 loads per thread for each 32x256 FP8
// tile. Invalid tail rows reload the tile's first valid row; their scores are
// masked to -inf, so their probabilities and PV weights are exactly zero.
template <int kSplitTokens, int kQkWarps>
__global__ void dynamic_decode_register_prefetch_kernel(
    const __nv_bfloat16* __restrict__ query,
    const __nv_fp8_e4m3* __restrict__ key_cache,
    const __nv_fp8_e4m3* __restrict__ value_cache,
    const float* __restrict__ key_scales,
    const float* __restrict__ value_scales,
    float* __restrict__ partial_max, float* __restrict__ partial_sum,
    float* __restrict__ partial_output, int valid_tokens, int max_splits) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  namespace wmma = nvcuda::wmma;
  static_assert(kQkWarps == 1 || kQkWarps == 4 || kQkWarps == 8);
  constexpr int kTile = 32;
  constexpr int kPartElements = kQueriesPerKvHead * kTile;
  const int split = blockIdx.x;
  const int kv_head = blockIdx.y;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const int query_head = kv_head * kQueriesPerKvHead + warp;
  const int begin = split * static_cast<int>(kSplitTokens);
  const int end = begin + min(static_cast<int>(kSplitTokens), valid_tokens - begin);
  if (begin >= end || kv_head >= kKvHeads) return;

  __shared__ __align__(32) unsigned char data[kTile * kHeadDim * sizeof(__nv_bfloat16)];
  __shared__ __align__(32) __nv_bfloat16 shared_query[kQueriesPerKvHead][kHeadDim];
  __shared__ __align__(32) unsigned char weights[3 * kPartElements * sizeof(__nv_bfloat16)];
  // QK partial scores and the PV tile output have disjoint lifetimes, so they
  // share one buffer that never aliases the K/V data tile.
  __shared__ __align__(32) float exchange[kQueriesPerKvHead * kHeadDim];
  static_assert(kQkWarps * kPartElements <= kQueriesPerKvHead * kHeadDim);
  float* qk_partial = exchange;
  float* tile_output = exchange;
  __shared__ float ks[kTile];
  __shared__ float vs[kTile];
  auto* raw = reinterpret_cast<__nv_bfloat16*>(data);
  auto* scores = reinterpret_cast<float*>(weights);
  auto* parts = reinterpret_cast<__nv_bfloat16*>(weights);

  bool unsafe_query = false;
  for (int i = threadIdx.x; i < kQueriesPerKvHead * kHeadDim; i += kThreads) {
    const __nv_bfloat16 q = query[kv_head * kQueriesPerKvHead * kHeadDim + i];
    shared_query[i / kHeadDim][i % kHeadDim] = q;
    unsafe_query |= !safe_tensor_operand(__bfloat162float(q));
  }
  const bool scalar_qk = __syncthreads_or(unsafe_query);
  float accumulator[8] = {};
  float running_max = -CUDART_INF_F;
  float running_sum = 0.0F;

  const int initial_group0 = threadIdx.x;
  const int initial_group1 = threadIdx.x + kThreads;
  const int initial_count = min(kTile, end - begin);
  const uint4 initial_k0 =
      load_decode_raw16(key_cache, begin, initial_count, kv_head, initial_group0);
  const uint4 initial_k1 =
      load_decode_raw16(key_cache, begin, initial_count, kv_head, initial_group1);
  store_decode_raw16(initial_k0, raw, initial_group0);
  store_decode_raw16(initial_k1, raw, initial_group1);
  if (threadIdx.x < kTile) {
    const int token = threadIdx.x;
    const int64_t row = static_cast<int64_t>(begin + token) * kKvHeads + kv_head;
    ks[token] = token < initial_count ? key_scales[row] : 0.0F;
    vs[token] = token < initial_count ? value_scales[row] : 0.0F;
  }
  __syncthreads();

  for (int first = begin; first < end; first += kTile) {
    const int count = min(kTile, end - first);
    const bool has_next = end - first > kTile;
    const int next = has_next ? first + kTile : end;
    const int next_count = has_next ? min(kTile, end - next) : 0;
    const int group0 = threadIdx.x;
    const int group1 = threadIdx.x + kThreads;

    // V(current) is resident before QK starts. The bytes stay in registers
    // until every QK reader of the reused smem buffer has reached the barrier.
    const uint4 v0 = load_decode_raw16(value_cache, first, count, kv_head, group0);
    const uint4 v1 = load_decode_raw16(value_cache, first, count, kv_head, group1);

    if (scalar_qk) {
      for (int token = 0; token < kTile; ++token) {
        float dot = 0.0F;
#pragma unroll
        for (int item = 0; item < 8; ++item) {
          const int dim = lane + item * 32;
          dot += __bfloat162float(shared_query[warp][dim]) *
              __bfloat162float(raw[token * kHeadDim + dim]);
        }
        dot = warp_sum(dot);
        if (lane == 0) scores[warp * kTile + token] = dot;
      }
    } else if (warp < kQkWarps) {
      wmma::fragment<wmma::matrix_a, 8, 32, 16, __nv_bfloat16, wmma::row_major> a;
      wmma::fragment<wmma::matrix_b, 8, 32, 16, __nv_bfloat16, wmma::col_major> b;
      wmma::fragment<wmma::accumulator, 8, 32, 16, float> c;
      wmma::fill_fragment(c, 0.0F);
      constexpr int kDimsPerWarp = kHeadDim / kQkWarps;
#pragma unroll
      for (int dim = warp * kDimsPerWarp;
           dim < (warp + 1) * kDimsPerWarp; dim += 16) {
        wmma::load_matrix_sync(a, &shared_query[0][dim], kHeadDim);
        wmma::load_matrix_sync(b, raw + dim, kHeadDim);
        wmma::mma_sync(c, a, b, c);
      }
      wmma::store_matrix_sync(qk_partial + warp * kPartElements, c, kTile,
          wmma::mem_row_major);
    }
    __syncthreads();
    if (!scalar_qk) {
      const int index = threadIdx.x;
      float value = qk_partial[index];
#pragma unroll
      for (int producer = 1; producer < kQkWarps; ++producer) {
        value += qk_partial[producer * kPartElements + index];
      }
      scores[index] = value;
    }

    const float score =
        lane < count ? scores[warp * kTile + lane] * ks[lane] * 0.0625F
                     : -CUDART_INF_F;
    float maximum = score;
#pragma unroll
    for (int delta = 16; delta > 0; delta >>= 1) {
      maximum = fmaxf(maximum, __shfl_down_sync(0xffffffffU, maximum, delta));
    }
    maximum = __shfl_sync(0xffffffffU, maximum, 0);
    const float next_max = fmaxf(running_max, maximum);
    const float old_weight = __expf(running_max - next_max);
    const float probability = lane < count ? __expf(score - next_max) : 0.0F;
    running_sum = running_sum * old_weight +
        __shfl_sync(0xffffffffU, warp_sum(probability), 0);
    running_max = next_max;

    const float weight = probability * vs[lane];
    const __nv_bfloat16 high = __float2bfloat16(weight);
    const float hf = __bfloat162float(high);
    const float r1 = weight - hf;
    const __nv_bfloat16 middle = __float2bfloat16(r1);
    const float mf = __bfloat162float(middle);
    const float r2 = r1 - mf;
    const __nv_bfloat16 low = __float2bfloat16(r2);
    const float lf = __bfloat162float(low);
    const bool unsafe_weight = !isfinite(weight) || r2 - lf != 0.0F ||
        !safe_tensor_operand(hf) || !safe_tensor_operand(mf) ||
        !safe_tensor_operand(lf);
    const bool scalar_pv = __syncthreads_or(unsafe_weight);
    const int weight_index = warp * kTile + lane;
    if (scalar_pv) {
      scores[weight_index] = weight;
    } else {
      parts[weight_index] = high;
      parts[kPartElements + weight_index] = middle;
      parts[2 * kPartElements + weight_index] = low;
    }

    store_decode_raw16(v0, raw, group0);
    store_decode_raw16(v1, raw, group1);
    const uint4 next_k0 = has_next
        ? load_decode_raw16(key_cache, next, next_count, kv_head, group0)
        : make_uint4(0, 0, 0, 0);
    const uint4 next_k1 = has_next
        ? load_decode_raw16(key_cache, next, next_count, kv_head, group1)
        : make_uint4(0, 0, 0, 0);
    __syncthreads();

    if (scalar_pv) {
#pragma unroll
      for (int item = 0; item < 8; ++item) accumulator[item] *= old_weight;
      for (int token = 0; token < count; ++token) {
        const float token_weight = scores[warp * kTile + token];
#pragma unroll
        for (int item = 0; item < 8; ++item) {
          accumulator[item] += __bfloat162float(
              raw[token * kHeadDim + lane + item * 32]) * token_weight;
        }
      }
    } else {
      wmma::fragment<wmma::matrix_a, 8, 32, 16, __nv_bfloat16, wmma::row_major> a;
      wmma::fragment<wmma::matrix_b, 8, 32, 16, __nv_bfloat16, wmma::row_major> b;
      wmma::fragment<wmma::accumulator, 8, 32, 16, float> c;
      wmma::fill_fragment(c, 0.0F);
#pragma unroll
      for (int token = 0; token < kTile; token += 16) {
        wmma::load_matrix_sync(
            b, raw + token * kHeadDim + warp * 32, kHeadDim);
#pragma unroll
        for (int part = 0; part < 3; ++part) {
          wmma::load_matrix_sync(
              a, parts + part * kPartElements + token, kTile);
          wmma::mma_sync(c, a, b, c);
        }
      }
      wmma::store_matrix_sync(
          tile_output + warp * 32, c, kHeadDim, wmma::mem_row_major);
    }
    // Every warp has finished reading V and PV weights, and each column
    // owner's output for all eight query heads is visible.
    __syncthreads();
    if (!scalar_pv) {
#pragma unroll
      for (int item = 0; item < 8; ++item) {
        accumulator[item] = accumulator[item] * old_weight +
            tile_output[warp * kHeadDim + lane + item * 32];
      }
    }
    if (!has_next) break;

    // The data tile has no remaining V readers. Tile output lives in the
    // separate exchange buffer, which the next QK reuses only after the
    // following barrier.
    store_decode_raw16(next_k0, raw, group0);
    store_decode_raw16(next_k1, raw, group1);
    if (threadIdx.x < kTile) {
      const int token = threadIdx.x;
      const int64_t row =
          static_cast<int64_t>(next + token) * kKvHeads + kv_head;
      ks[token] = token < next_count ? key_scales[row] : 0.0F;
      vs[token] = token < next_count ? value_scales[row] : 0.0F;
    }
    __syncthreads();
  }
  const int64_t partial_index =
      static_cast<int64_t>(query_head) * max_splits + split;
  if (lane == 0) {
    partial_max[partial_index] = running_max;
    partial_sum[partial_index] = running_sum;
  }
#pragma unroll
  for (int item = 0; item < 8; ++item) {
    partial_output[partial_index * kHeadDim + lane + item * 32] =
        accumulator[item];
  }
#else
  dynamic_decode_simt<kSplitTokens>(query, key_cache, value_cache, key_scales,
      value_scales, partial_max, partial_sum, partial_output, valid_tokens,
      max_splits);
#endif
}

__global__ void dynamic_decode_reduce_kernel(
    const float* __restrict__ partial_max,
    const float* __restrict__ partial_sum,
    const float* __restrict__ partial_output,
    const __nv_bfloat16* __restrict__ gate,
    __nv_bfloat16* __restrict__ output, int num_splits, int max_splits) {
  const int query_head = blockIdx.x;
  const int dim = threadIdx.x;
  if (dim >= kHeadDim) return;
  __shared__ float partial[kWarps];
  __shared__ float weights[kThreads];
  const int64_t base = static_cast<int64_t>(query_head) * max_splits;
  float maximum = -CUDART_INF_F;
  for (int split = dim; split < num_splits; split += kThreads) {
    maximum = fmaxf(maximum, partial_max[base + split]);
  }
  // Unlike absmax reduction, attention scores can all be negative.
  for (int delta = 16; delta > 0; delta >>= 1) {
    maximum = fmaxf(maximum, __shfl_down_sync(0xffffffffU, maximum, delta));
  }
  if ((dim & 31) == 0) partial[dim >> 5] = maximum;
  __syncthreads();
  if (dim == 0) {
    float m = -CUDART_INF_F;
#pragma unroll
    for (int i = 0; i < kWarps; ++i) m = fmaxf(m, partial[i]);
    partial[0] = m;
  }
  __syncthreads();
  const float global_max = partial[0];
  __syncthreads();
  float sum = 0.0F;
  float value = 0.0F;
  // Compute each split weight once for all 256 output dimensions. Chunking
  // keeps shared memory bounded even for contexts with >256 splits.
  for (int first = 0; first < num_splits; first += kThreads) {
    const int count = min(kThreads, num_splits - first);
    if (dim < count) {
      const int64_t index = base + first + dim;
      const float weight = __expf(partial_max[index] - global_max);
      weights[dim] = weight;
      sum += partial_sum[index] * weight;
    }
    __syncthreads();
    for (int split = 0; split < count; ++split) {
      value += partial_output[(base + first + split) * kHeadDim + dim] * weights[split];
    }
    __syncthreads();
  }
  const float denominator = block_sum(sum, partial);
  // Preserve the decomposed path's BF16 attention rounding before the gate.
  const __nv_bfloat16 attention = __float2bfloat16(value / denominator);
  const int index = query_head * kHeadDim + dim;
  output[index] = gate == nullptr ? attention : __float2bfloat16(
      __bfloat162float(attention) / (1.0F + __expf(-__bfloat162float(gate[index]))));
}

__global__ void dynamic_kv_dequant_kernel(
    const __nv_fp8_e4m3* __restrict__ key_cache,
    const __nv_fp8_e4m3* __restrict__ value_cache,
    const float* __restrict__ key_scales,
    const float* __restrict__ value_scales,
    __nv_bfloat16* __restrict__ key, __nv_bfloat16* __restrict__ value,
    int elements) {
  const int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index >= elements) return;
  const int scale_index = index / kHeadDim;
  key[index] = __float2bfloat16(static_cast<float>(key_cache[index]) *
                               key_scales[scale_index]);
  value[index] = __float2bfloat16(static_cast<float>(value_cache[index]) *
                                 value_scales[scale_index]);
}

// Multi-row direct dynamic-FP8 attention: rows = 1 (decode) .. 4 (MTP verifier).
__device__ __forceinline__ void e4m3x4_to_bf16(unsigned int w, uint2& out) {
  // E4M3 bit pattern placed in BF16 fields equals value * 2^-120 (subnormals included).
  const __nv_bfloat162 k = __halves2bfloat162(__ushort_as_bfloat16(0x7b80), __ushort_as_bfloat16(0x7b80));
  unsigned int lo = __byte_perm(w, 0, 0x1404), hi = __byte_perm(w, 0, 0x3424);
  lo = (lo & 0x80008000U) | ((lo & 0x7f007f00U) >> 4);
  hi = (hi & 0x80008000U) | ((hi & 0x7f007f00U) >> 4);
  __nv_bfloat162 a = __hmul2(*reinterpret_cast<__nv_bfloat162*>(&lo), k);
  __nv_bfloat162 b = __hmul2(*reinterpret_cast<__nv_bfloat162*>(&hi), k);
  out.x = *reinterpret_cast<unsigned int*>(&a);
  out.y = *reinterpret_cast<unsigned int*>(&b);
}
constexpr int kMrTile = 32, kMrLd = 264, kMrPld = 40;
template <int kSplit, int kRows> constexpr size_t mr_smem() {
  constexpr int MP = (kRows * 8 + 15) / 16 * 16;
  constexpr size_t parts = 2 * MP * kMrPld * 2 > 8192 ? 2 * MP * kMrPld * 2 : 8192;
  return size_t(MP) * kMrLd * 2 + size_t(kMrTile) * kMrLd * 2 + size_t(MP) * (kSplit + 4) * 4 + parts + 2 * kSplit * 4 + MP * 4;
}
__device__ __forceinline__ void mr_load(const __nv_fp8_e4m3* base, int first, int count, int kv, uint4* r) {
#pragma unroll
  for (int j = 0; j < 2; ++j) {
    const int c = threadIdx.x + j * 256, t = c >> 4, d = (c & 15) * 16;
    r[j] = t < count ? *reinterpret_cast<const uint4*>(base + (static_cast<int64_t>(first + t) * kKvHeads + kv) * kHeadDim + d)
                     : make_uint4(0, 0, 0, 0);
  }
}
__device__ __forceinline__ void mr_store(__nv_bfloat16* kvs, const uint4* r) {
#pragma unroll
  for (int j = 0; j < 2; ++j) {
    const int c = threadIdx.x + j * 256, t = c >> 4, d = (c & 15) * 16;
    uint2 o[4];
    e4m3x4_to_bf16(r[j].x, o[0]); e4m3x4_to_bf16(r[j].y, o[1]);
    e4m3x4_to_bf16(r[j].z, o[2]); e4m3x4_to_bf16(r[j].w, o[3]);
    uint4* dst = reinterpret_cast<uint4*>(kvs + t * kMrLd + d);
    dst[0] = make_uint4(o[0].x, o[0].y, o[1].x, o[1].y);
    dst[1] = make_uint4(o[2].x, o[2].y, o[3].x, o[3].y);
  }
}
template <int kSplit, int kRows>
__global__ __launch_bounds__(256) void dynamic_mr_kernel(
    const __nv_bfloat16* __restrict__ query, const __nv_fp8_e4m3* __restrict__ key_cache,
    const __nv_fp8_e4m3* __restrict__ value_cache, const float* __restrict__ key_scales,
    const float* __restrict__ value_scales, float* __restrict__ partial_max,
    float* __restrict__ partial_sum, float* __restrict__ partial_output,
    int valid_tokens, int max_splits) {
  namespace wmma = nvcuda::wmma;
  constexpr int M = kRows * 8, MP = (M + 15) / 16 * 16, MT = MP / 16, NT = MT * 2, NP = 8 / NT;
  constexpr int SLD = kSplit + 4;
  extern __shared__ __align__(128) unsigned char sm[];
  auto* q = reinterpret_cast<__nv_bfloat16*>(sm);
  auto* kvs = q + MP * kMrLd;
  auto* sc = reinterpret_cast<float*>(kvs + kMrTile * kMrLd);
  auto* pr = reinterpret_cast<__nv_bfloat16*>(sc + MP * SLD);
  auto* red = reinterpret_cast<float*>(pr);
  constexpr int kPartBytes = 2 * MP * kMrPld * 2 > 8192 ? 2 * MP * kMrPld * 2 : 8192;
  auto* ks = reinterpret_cast<float*>(reinterpret_cast<unsigned char*>(pr) + kPartBytes);
  auto* vs = ks + kSplit;
  auto* wsc = vs + kSplit;
  const int split = blockIdx.x, kv = blockIdx.y, tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int begin = split * kSplit;
  const int count = min(kSplit, valid_tokens - begin);
  if (count <= 0) return;
  const int tiles = (count + kMrTile - 1) / kMrTile;
  uint4 reg[2];
  mr_load(key_cache, begin, min(kMrTile, count), kv, reg);
  for (int i = tid; i < MP * kHeadDim; i += 256) {
    const int m = i >> 8, d = i & 255, r = m >> 3, h = m & 7;
    q[m * kMrLd + d] = m < M ? query[(r * kQHeads + kv * kQueriesPerKvHead + h) * kHeadDim + d] : __float2bfloat16(0.0F);
  }
  for (int i = tid; i < kSplit; i += 256) {
    const int64_t row = static_cast<int64_t>(begin + i) * kKvHeads + kv;
    ks[i] = i < count ? key_scales[row] : 0.0F;
    vs[i] = i < count ? value_scales[row] : 0.0F;
  }
  for (int tile = 0; tile < tiles; ++tile) {
    const int t0 = tile * kMrTile;
    mr_store(kvs, reg);
    if (tile + 1 < tiles) mr_load(key_cache, begin + t0 + kMrTile, min(kMrTile, count - t0 - kMrTile), kv, reg);
    else mr_load(value_cache, begin, min(kMrTile, count), kv, reg);
    __syncthreads();
    {
      const int ot = warp % NT, kp = warp / NT, mt = ot >> 1, nt = ot & 1;
      wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a;
      wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b;
      wmma::fragment<wmma::accumulator, 16, 16, 16, float> c;
      wmma::fill_fragment(c, 0.0F);
#pragma unroll
      for (int d = kp * (kHeadDim / NP); d < (kp + 1) * (kHeadDim / NP); d += 16) {
        wmma::load_matrix_sync(a, q + mt * 16 * kMrLd + d, kMrLd);
        wmma::load_matrix_sync(b, kvs + nt * 16 * kMrLd + d, kMrLd);
        wmma::mma_sync(c, a, b, c);
      }
      wmma::store_matrix_sync(red + warp * 256, c, 16, wmma::mem_row_major);
    }
    __syncthreads();
    for (int e = tid; e < MP * kMrTile; e += 256) {
      const int m = e >> 5, tok = e & 31, ot = (m >> 4) * 2 + (tok >> 4), idx = (m & 15) * 16 + (tok & 15);
      float s = 0.0F;
#pragma unroll
      for (int kp = 0; kp < NP; ++kp) s += red[(kp * NT + ot) * 256 + idx];
      sc[m * SLD + t0 + tok] = s * ks[t0 + tok] * 0.0625F;
    }
  }
  __syncthreads();
  for (int m = warp; m < M; m += 8) {
    const int r = m >> 3, h = m & 7;
    const int lim = min(valid_tokens - kRows + 1 + r - begin, kSplit);
    float mx = -CUDART_INF_F;
    for (int j = lane; j < lim; j += 32) mx = fmaxf(mx, sc[m * SLD + j]);
#pragma unroll
    for (int delta = 16; delta > 0; delta >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffU, mx, delta));
    float sum = 0.0F, wm = 0.0F;
    for (int j = lane; j < tiles * kMrTile; j += 32) {
      const float p = j < lim ? __expf(sc[m * SLD + j] - mx) : 0.0F;
      const float w = p * vs[j];
      sum += p;
      wm = fmaxf(wm, w);
      sc[m * SLD + j] = w;
    }
#pragma unroll
    for (int delta = 16; delta > 0; delta >>= 1) {
      sum += __shfl_xor_sync(0xffffffffU, sum, delta);
      wm = fmaxf(wm, __shfl_xor_sync(0xffffffffU, wm, delta));
    }
    // Tensor cores flush BF16 subnormals: keep PV weights in [0,1] per row and
    // restore the magnitude in FP32 when the split result is written.
    wm = wm > 0.0F ? wm : 1.0F;
    const float inv = 1.0F / wm;
    const bool use_inv = isfinite(inv);
    for (int j = lane; j < tiles * kMrTile; j += 32) {
      const float w = sc[m * SLD + j];
      sc[m * SLD + j] = use_inv ? w * inv : w / wm;
    }
    if (lane == 0) {
      const int64_t pi = static_cast<int64_t>(r * kQHeads + kv * kQueriesPerKvHead + h) * max_splits + split;
      partial_max[pi] = mx;
      partial_sum[pi] = sum;
      wsc[m] = wm;
    }
  }
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[MT][2];
#pragma unroll
  for (int mt = 0; mt < MT; ++mt) { wmma::fill_fragment(acc[mt][0], 0.0F); wmma::fill_fragment(acc[mt][1], 0.0F); }
  for (int tile = 0; tile < tiles; ++tile) {
    const int t0 = tile * kMrTile;
    __syncthreads();
    mr_store(kvs, reg);
    if (tile + 1 < tiles) mr_load(value_cache, begin + t0 + kMrTile, min(kMrTile, count - t0 - kMrTile), kv, reg);
    for (int e = tid; e < MP * kMrTile; e += 256) {
      const int m = e >> 5, t = e & 31;
      const float w = m < M ? sc[m * SLD + t0 + t] : 0.0F;
      const __nv_bfloat16 hi = __float2bfloat16(w);
      const float r1 = w - __bfloat162float(hi);
      const __nv_bfloat16 mid = __float2bfloat16(r1);
      pr[m * kMrPld + t] = hi;
      pr[MP * kMrPld + m * kMrPld + t] = mid;
    }
    __syncthreads();
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> b;
#pragma unroll
    for (int ts = 0; ts < kMrTile; ts += 16) {
#pragma unroll
      for (int n = 0; n < 2; ++n) {
        wmma::load_matrix_sync(b, kvs + ts * kMrLd + warp * 32 + n * 16, kMrLd);
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
#pragma unroll
          for (int part = 0; part < 2; ++part) {
            wmma::load_matrix_sync(a, pr + part * MP * kMrPld + mt * 16 * kMrPld + ts, kMrPld);
            wmma::mma_sync(acc[mt][n], a, b, acc[mt][n]);
          }
        }
      }
    }
  }
  __syncthreads();
  auto* out = reinterpret_cast<float*>(kvs);
#pragma unroll
  for (int mt = 0; mt < MT; ++mt)
#pragma unroll
    for (int n = 0; n < 2; ++n)
      wmma::store_matrix_sync(out + mt * 16 * kHeadDim + warp * 32 + n * 16, acc[mt][n], kHeadDim, wmma::mem_row_major);
  __syncthreads();
  for (int e = tid; e < M * kHeadDim; e += 256) {
    const int m = e >> 8, d = e & 255, r = m >> 3, h = m & 7;
    const int64_t pi = static_cast<int64_t>(r * kQHeads + kv * kQueriesPerKvHead + h) * max_splits + split;
    partial_output[pi * kHeadDim + d] = out[e] * wsc[m];
  }
}

template <int kSplit, int kRows>
void launch_dynamic_mr(const apxinf_qwen35_dynamic_decode_spec_t& spec,
                       const apxinf_qwen35_dynamic_decode_bindings_t& bindings,
                       cudaStream_t stream, int num_splits) {
  static const cudaError_t attribute = cudaFuncSetAttribute(
      dynamic_mr_kernel<kSplit, kRows>,
      cudaFuncAttributeMaxDynamicSharedMemorySize,
      static_cast<int>(mr_smem<kSplit, kRows>()));
  apxinf::framework::check_cuda(attribute);
  dynamic_mr_kernel<kSplit, kRows>
      <<<dim3(num_splits, kKvHeads), 256, mr_smem<kSplit, kRows>(), stream>>>(
          static_cast<const __nv_bfloat16*>(bindings.query),
          static_cast<const __nv_fp8_e4m3*>(bindings.key_cache),
          static_cast<const __nv_fp8_e4m3*>(bindings.value_cache),
          static_cast<const float*>(bindings.key_scales),
          static_cast<const float*>(bindings.value_scales),
          static_cast<float*>(bindings.partial_max),
          static_cast<float*>(bindings.partial_sum),
          static_cast<float*>(bindings.partial_output),
          static_cast<int>(spec.valid_tokens),
          static_cast<int>(spec.max_splits));
}

template <int kSplit>
void launch_dynamic_mr_rows(const apxinf_qwen35_dynamic_decode_spec_t& spec,
                            const apxinf_qwen35_dynamic_decode_bindings_t& bindings,
                            cudaStream_t stream, int num_splits) {
  switch (spec.rows) {
    case 1: return launch_dynamic_mr<kSplit, 1>(spec, bindings, stream, num_splits);
    case 2: return launch_dynamic_mr<kSplit, 2>(spec, bindings, stream, num_splits);
    case 3: return launch_dynamic_mr<kSplit, 3>(spec, bindings, stream, num_splits);
    default: return launch_dynamic_mr<kSplit, 4>(spec, bindings, stream, num_splits);
  }
}

enum class DecodeKernel {
  MultiRow,
  Baseline,
  RegisterQk1,
  RegisterQk4,
  RegisterQk8,
  TensorQk1,
  TensorQk4,
  TensorQk8,
};

struct DecodeOptions {
  DecodeKernel kernel = DecodeKernel::MultiRow;
  int split_tokens = kDecodeSplitTokens;
  const char* error = nullptr;
};

const DecodeOptions& decode_options() {
  // Frozen per process so a captured CUDA Graph always launches one schedule.
  static const DecodeOptions options = [] {
    DecodeOptions result;
    const char* kernel = std::getenv("APXINF_QWEN35_DECODE_KERNEL");
    if (kernel && *kernel) {
      if (!std::strcmp(kernel, "mr")) {
        result.kernel = DecodeKernel::MultiRow;
      } else if (!std::strcmp(kernel, "baseline")) {
        result.kernel = DecodeKernel::Baseline;
      } else if (!std::strcmp(kernel, "reg-qk1")) {
        result.kernel = DecodeKernel::RegisterQk1;
      } else if (!std::strcmp(kernel, "reg-qk4")) {
        result.kernel = DecodeKernel::RegisterQk4;
      } else if (!std::strcmp(kernel, "reg-qk8")) {
        result.kernel = DecodeKernel::RegisterQk8;
      } else if (!std::strcmp(kernel, "tensor-qk1")) {
        result.kernel = DecodeKernel::TensorQk1;
      } else if (!std::strcmp(kernel, "tensor-qk4")) {
        result.kernel = DecodeKernel::TensorQk4;
      } else if (!std::strcmp(kernel, "tensor-qk8")) {
        result.kernel = DecodeKernel::TensorQk8;
      } else {
        result.error = "invalid APXINF_QWEN35_DECODE_KERNEL";
      }
    }
    const char* split = std::getenv("APXINF_QWEN35_DECODE_SPLIT_TOKENS");
    if (split && *split) {
      if (!std::strcmp(split, "256")) {
        result.split_tokens = 256;
      } else if (!std::strcmp(split, "128")) {
        result.split_tokens = 128;
      } else if (!std::strcmp(split, "512")) {
        result.split_tokens = 512;
      } else if (!std::strcmp(split, "1024")) {
        result.split_tokens = 1024;
      } else {
        result.error = "invalid APXINF_QWEN35_DECODE_SPLIT_TOKENS";
      }
    }
    return result;
  }();
  return options;
}

template <int kSplitTokens, int kQkWarps>
void launch_dynamic_decode_register(
    const apxinf_qwen35_dynamic_decode_spec_t& spec,
    const apxinf_qwen35_dynamic_decode_bindings_t& bindings,
    cudaStream_t stream, int num_splits) {
  dynamic_decode_register_prefetch_kernel<kSplitTokens, kQkWarps>
      <<<dim3(num_splits, kKvHeads), kThreads, 0, stream>>>(
          static_cast<const __nv_bfloat16*>(bindings.query),
          static_cast<const __nv_fp8_e4m3*>(bindings.key_cache),
          static_cast<const __nv_fp8_e4m3*>(bindings.value_cache),
          static_cast<const float*>(bindings.key_scales),
          static_cast<const float*>(bindings.value_scales),
          static_cast<float*>(bindings.partial_max),
          static_cast<float*>(bindings.partial_sum),
          static_cast<float*>(bindings.partial_output), spec.valid_tokens,
          spec.max_splits);
}

template <int kSplitTokens, int kQkWarps>
void launch_dynamic_decode_tensor(
    const apxinf_qwen35_dynamic_decode_spec_t& spec,
    const apxinf_qwen35_dynamic_decode_bindings_t& bindings,
    cudaStream_t stream, int num_splits) {
  dynamic_decode_tensor_kernel<kSplitTokens, kQkWarps>
      <<<dim3(num_splits, kKvHeads), kThreads, 0, stream>>>(
          static_cast<const __nv_bfloat16*>(bindings.query),
          static_cast<const __nv_fp8_e4m3*>(bindings.key_cache),
          static_cast<const __nv_fp8_e4m3*>(bindings.value_cache),
          static_cast<const float*>(bindings.key_scales),
          static_cast<const float*>(bindings.value_scales),
          static_cast<float*>(bindings.partial_max),
          static_cast<float*>(bindings.partial_sum),
          static_cast<float*>(bindings.partial_output), spec.valid_tokens,
          spec.max_splits);
}

template <int kSplitTokens>
void launch_dynamic_decode_split(
    const apxinf_qwen35_dynamic_decode_spec_t& spec,
    const apxinf_qwen35_dynamic_decode_bindings_t& bindings,
    cudaStream_t stream, int num_splits, DecodeKernel kernel) {
  if (spec.valid_tokens >= 256) {
    const bool aligned =
        (reinterpret_cast<uintptr_t>(bindings.key_cache) & 15U) == 0 &&
        (reinterpret_cast<uintptr_t>(bindings.value_cache) & 15U) == 0;
    if (aligned) {
      switch (kernel) {
        case DecodeKernel::RegisterQk1:
          return launch_dynamic_decode_register<kSplitTokens, 1>(
              spec, bindings, stream, num_splits);
        case DecodeKernel::RegisterQk4:
          return launch_dynamic_decode_register<kSplitTokens, 4>(
              spec, bindings, stream, num_splits);
        case DecodeKernel::RegisterQk8:
          return launch_dynamic_decode_register<kSplitTokens, 8>(
              spec, bindings, stream, num_splits);
        case DecodeKernel::MultiRow:
        case DecodeKernel::Baseline:
        case DecodeKernel::TensorQk1:
        case DecodeKernel::TensorQk4:
        case DecodeKernel::TensorQk8:
          break;
      }
    }
    switch (kernel) {
      case DecodeKernel::TensorQk4:
        return launch_dynamic_decode_tensor<kSplitTokens, 4>(
            spec, bindings, stream, num_splits);
      case DecodeKernel::TensorQk8:
        return launch_dynamic_decode_tensor<kSplitTokens, 8>(
            spec, bindings, stream, num_splits);
      default:
        break;
    }
    launch_dynamic_decode_tensor<kSplitTokens, 1>(
        spec, bindings, stream, num_splits);
  } else {
    dynamic_decode_split_kernel<kSplitTokens><<<dim3(num_splits, kKvHeads), kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(bindings.query),
        static_cast<const __nv_fp8_e4m3*>(bindings.key_cache),
        static_cast<const __nv_fp8_e4m3*>(bindings.value_cache),
        static_cast<const float*>(bindings.key_scales),
        static_cast<const float*>(bindings.value_scales),
        static_cast<float*>(bindings.partial_max),
        static_cast<float*>(bindings.partial_sum),
        static_cast<float*>(bindings.partial_output), spec.valid_tokens, spec.max_splits);
  }
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
         bindings.gate != nullptr && bindings.output != nullptr &&
         bindings.scores != nullptr && bindings.partial_output != nullptr &&
         bindings.partial_stats != nullptr;
}

bool valid_dynamic_kv_spec(const apxinf_qwen35_dynamic_kv_spec_t& spec) {
  return spec.version == APXINF_QWEN35_DYNAMIC_KV_SPEC_VERSION &&
         spec.tokens > 0 && spec.tokens <= INT_MAX &&
         spec.cache_capacity > 0 && spec.cache_capacity <= INT_MAX &&
         spec.cache_offset <= INT_MAX &&
         static_cast<uint64_t>(spec.cache_offset) + spec.tokens <=
             spec.cache_capacity;
}

bool valid_dynamic_kv_bindings(
    const apxinf_qwen35_dynamic_kv_bindings_t& bindings) {
  return bindings.key != nullptr && bindings.value != nullptr &&
         bindings.key_cache != nullptr && bindings.value_cache != nullptr &&
         bindings.key_scales != nullptr && bindings.value_scales != nullptr;
}

bool valid_dynamic_decode_spec(
    const apxinf_qwen35_dynamic_decode_spec_t& spec) {
  const uint64_t splits =
      (static_cast<uint64_t>(spec.valid_tokens) + APXINF_QWEN35_MIN_DECODE_SPLIT_TOKENS - 1) /
      APXINF_QWEN35_MIN_DECODE_SPLIT_TOKENS;
  return spec.version == APXINF_QWEN35_DYNAMIC_DECODE_SPEC_VERSION &&
         spec.rows > 0 && spec.rows <= APXINF_QWEN35_DYNAMIC_DECODE_MAX_ROWS &&
         spec.valid_tokens >= spec.rows && spec.valid_tokens <= spec.cache_capacity &&
         spec.cache_capacity <= INT_MAX && spec.max_splits > 0 &&
         spec.max_splits <= INT_MAX && splits <= spec.max_splits;
}

bool valid_dynamic_decode_bindings(
    const apxinf_qwen35_dynamic_decode_bindings_t& bindings) {
  return bindings.query != nullptr && bindings.key_cache != nullptr &&
         bindings.value_cache != nullptr && bindings.key_scales != nullptr &&
         bindings.value_scales != nullptr && bindings.partial_max != nullptr &&
         bindings.partial_sum != nullptr && bindings.partial_output != nullptr &&
         bindings.output != nullptr;
}

bool valid_dynamic_dequant_spec(
    const apxinf_qwen35_dynamic_dequant_spec_t& spec) {
  return spec.version == APXINF_QWEN35_DYNAMIC_KV_SPEC_VERSION &&
         spec.valid_tokens > 0 && spec.valid_tokens <= spec.cache_capacity &&
         spec.cache_capacity <= INT_MAX &&
         spec.valid_tokens <= INT_MAX / (kKvHeads * kHeadDim);
}

bool valid_dynamic_dequant_bindings(
    const apxinf_qwen35_dynamic_dequant_bindings_t& bindings) {
  return bindings.key_cache != nullptr && bindings.value_cache != nullptr &&
         bindings.key_scales != nullptr && bindings.value_scales != nullptr &&
         bindings.key != nullptr && bindings.value != nullptr;
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

    const int valid_tokens = spec->cache_offset + spec->tokens;
    const int automatic_splits = min(
        static_cast<int>(APXINF_QWEN35_ATTENTION_MAX_SPLITS),
        max(1, (valid_tokens + kValuePositionsPerTile - 1) /
                   kValuePositionsPerTile));
    int splits = automatic_splits;
    if (const char* configured =
            std::getenv("APXINF_QWEN35_FP8_ATTENTION_MAX_SPLITS")) {
      const int limit = std::atoi(configured);
      if (limit > 0) splits = min(automatic_splits, limit);
    }
    const dim3 split_grid(splits, kKvHeads, spec->tokens);
    const char* fused_attention =
        std::getenv("APXINF_QWEN35_FP8_FUSED_ATTENTION");
    if (fused_attention == nullptr || fused_attention[0] != '0') {
      attention_fused_split_kernel<<<split_grid, kThreads, 0, stream>>>(
          static_cast<const __nv_bfloat16*>(bindings->query),
          static_cast<const __nv_fp8_e4m3*>(bindings->key_cache),
          static_cast<const __nv_fp8_e4m3*>(bindings->value_cache),
          static_cast<float*>(bindings->partial_output),
          static_cast<float*>(bindings->partial_stats), spec->tokens,
          spec->cache_offset, splits, key_scale);
    } else {
      const dim3 score_grid(
          (valid_tokens + kScorePositionsPerBlock - 1) /
              kScorePositionsPerBlock,
          kKvHeads, spec->tokens);
      attention_scores_kernel<<<score_grid, kThreads, 0, stream>>>(
          static_cast<const __nv_bfloat16*>(bindings->query),
          static_cast<const __nv_fp8_e4m3*>(bindings->key_cache),
          static_cast<float*>(bindings->scores), spec->tokens,
          spec->cache_offset, spec->cache_capacity, key_scale);
      apxinf::framework::check_cuda(cudaGetLastError());
      attention_split_output_kernel<<<split_grid, kThreads, 0, stream>>>(
          static_cast<const __nv_fp8_e4m3*>(bindings->value_cache),
          static_cast<const float*>(bindings->scores),
          static_cast<float*>(bindings->partial_output),
          static_cast<float*>(bindings->partial_stats), spec->tokens,
          spec->cache_offset, spec->cache_capacity, splits);
    }
    apxinf::framework::check_cuda(cudaGetLastError());

    const dim3 output_grid(spec->tokens, kKvHeads);
    attention_combine_kernel<<<output_grid, kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(bindings->gate),
        static_cast<const float*>(bindings->partial_output),
        static_cast<const float*>(bindings->partial_stats),
        static_cast<__nv_bfloat16*>(bindings->output), spec->tokens,
        splits, value_scale);
    apxinf::framework::check_cuda(cudaGetLastError());
  });
}

extern "C" apxinf_status_t apxinf_qwen35_query_prepare(
    apxinf_runtime_t runtime, const apxinf_qwen35_query_prepare_spec_t* spec,
    const apxinf_qwen35_query_prepare_bindings_t* bindings) {
  return apxinf::framework::abi_boundary([&] {
    if (runtime == nullptr || spec == nullptr || bindings == nullptr ||
        spec->version != APXINF_QWEN35_QUERY_PREPARE_SPEC_VERSION ||
        spec->tokens == 0 || spec->tokens > INT_MAX ||
        !std::isfinite(spec->rope_theta) || spec->rope_theta <= 0.0F ||
        !std::isfinite(spec->rms_epsilon) || spec->rms_epsilon <= 0.0F ||
        bindings->projected_query_gate == nullptr || bindings->norm_weight == nullptr ||
        bindings->positions == nullptr || bindings->query == nullptr || bindings->gate == nullptr) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT, "invalid fused query preparation arguments");
    }
    apxinf::framework::check_cuda(cudaSetDevice(runtime->device));
    prepare_query_kernel<<<dim3(spec->tokens, kQHeads), kThreads, 0,
        static_cast<cudaStream_t>(bindings->stream)>>>(
        static_cast<const __nv_bfloat16*>(bindings->projected_query_gate),
        static_cast<const __nv_bfloat16*>(bindings->norm_weight),
        static_cast<const int32_t*>(bindings->positions),
        static_cast<__nv_bfloat16*>(bindings->query),
        static_cast<__nv_bfloat16*>(bindings->gate), spec->tokens,
        spec->rope_theta, spec->rms_epsilon);
    apxinf::framework::check_cuda(cudaGetLastError());
  });
}

extern "C" apxinf_status_t apxinf_qwen35_dynamic_kv_append(
    apxinf_runtime_t runtime, const apxinf_qwen35_dynamic_kv_spec_t* spec,
    const apxinf_qwen35_dynamic_kv_bindings_t* bindings) {
  return apxinf::framework::abi_boundary([&] {
    if (runtime == nullptr || spec == nullptr || bindings == nullptr ||
        !valid_dynamic_kv_spec(*spec) ||
        !valid_dynamic_kv_bindings(*bindings)) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                    "invalid Qwen3.5 dynamic KV append arguments");
    }
    apxinf::framework::check_cuda(cudaSetDevice(runtime->device));
    auto stream = static_cast<cudaStream_t>(bindings->stream);
    const dim3 grid(spec->tokens, kKvHeads);
    dynamic_kv_append_kernel<<<grid, kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(bindings->key),
        static_cast<const __nv_bfloat16*>(bindings->value),
        static_cast<__nv_fp8_e4m3*>(bindings->key_cache),
        static_cast<__nv_fp8_e4m3*>(bindings->value_cache),
        static_cast<float*>(bindings->key_scales),
        static_cast<float*>(bindings->value_scales), spec->tokens,
        spec->cache_offset);
    apxinf::framework::check_cuda(cudaGetLastError());
  });
}

extern "C" apxinf_status_t apxinf_qwen35_dynamic_kv_prepare(
    apxinf_runtime_t runtime, const apxinf_qwen35_dynamic_prepare_spec_t* spec,
    const apxinf_qwen35_dynamic_prepare_bindings_t* bindings) {
  return apxinf::framework::abi_boundary([&] {
    if (runtime == nullptr || spec == nullptr || bindings == nullptr ||
        !valid_dynamic_kv_spec(spec->kv) ||
        !valid_dynamic_kv_bindings(bindings->kv) ||
        bindings->norm_weight == nullptr || bindings->positions == nullptr ||
        !std::isfinite(spec->rope_theta) || spec->rope_theta <= 0.0F ||
        !std::isfinite(spec->rms_epsilon) || spec->rms_epsilon <= 0.0F) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT, "invalid fused KV prepare arguments");
    }
    apxinf::framework::check_cuda(cudaSetDevice(runtime->device));
    const auto& b = bindings->kv;
    dynamic_kv_prepare_kernel<<<dim3(spec->kv.tokens, kKvHeads), kThreads, 0,
        static_cast<cudaStream_t>(b.stream)>>>(
        static_cast<const __nv_bfloat16*>(b.key),
        static_cast<const __nv_bfloat16*>(b.value),
        static_cast<const __nv_bfloat16*>(bindings->norm_weight),
        static_cast<const int32_t*>(bindings->positions),
        static_cast<__nv_fp8_e4m3*>(b.key_cache),
        static_cast<__nv_fp8_e4m3*>(b.value_cache),
        static_cast<float*>(b.key_scales), static_cast<float*>(b.value_scales),
        spec->kv.tokens, spec->kv.cache_offset, spec->rope_theta, spec->rms_epsilon);
    apxinf::framework::check_cuda(cudaGetLastError());
  });
}

extern "C" apxinf_status_t apxinf_qwen35_dynamic_decode_attention(
    apxinf_runtime_t runtime, const apxinf_qwen35_dynamic_decode_spec_t* spec,
    const apxinf_qwen35_dynamic_decode_bindings_t* bindings) {
  return apxinf::framework::abi_boundary([&] {
    if (runtime == nullptr || spec == nullptr || bindings == nullptr ||
        !valid_dynamic_decode_spec(*spec) ||
        !valid_dynamic_decode_bindings(*bindings)) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                    "invalid Qwen3.5 dynamic FP8 decode arguments");
    }
    apxinf::framework::check_cuda(cudaSetDevice(runtime->device));
    auto stream = static_cast<cudaStream_t>(bindings->stream);
    const auto& options = decode_options();
    if (options.error) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT, options.error);
    }
    // Short prefixes need more CTAs to fill the device. Long prefixes default
    // to 256-token splits, with process-level alternatives for device tuning.
    const int split_tokens =
        spec->valid_tokens <= 8192 ? 128 : options.split_tokens;
    const int num_splits = static_cast<int>(
        (static_cast<uint64_t>(spec->valid_tokens) + split_tokens - 1) /
        split_tokens);
    const bool multi_row =
        spec->rows > 1 || options.kernel == DecodeKernel::MultiRow;
    if (multi_row) {
      switch (split_tokens) {
        case 128:
          launch_dynamic_mr_rows<128>(*spec, *bindings, stream, num_splits);
          break;
        case 256:
          launch_dynamic_mr_rows<256>(*spec, *bindings, stream, num_splits);
          break;
        case 512:
          launch_dynamic_mr_rows<512>(*spec, *bindings, stream, num_splits);
          break;
        default:
          throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                        "unsupported multi-row decode split tokens");
      }
    } else switch (split_tokens) {
      case 128:
        launch_dynamic_decode_split<128>(
            *spec, *bindings, stream, num_splits, options.kernel);
        break;
      case 256:
        launch_dynamic_decode_split<256>(
            *spec, *bindings, stream, num_splits, options.kernel);
        break;
      case 512:
        launch_dynamic_decode_split<512>(
            *spec, *bindings, stream, num_splits, options.kernel);
        break;
      case 1024:
        launch_dynamic_decode_split<1024>(
            *spec, *bindings, stream, num_splits, options.kernel);
        break;
      default:
        throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
            "unsupported Qwen3.5 decode split tokens");
    }
    apxinf::framework::check_cuda(cudaGetLastError());
    dynamic_decode_reduce_kernel<<<kQHeads * spec->rows, kThreads, 0, stream>>>(
        static_cast<const float*>(bindings->partial_max),
        static_cast<const float*>(bindings->partial_sum),
        static_cast<const float*>(bindings->partial_output),
        static_cast<const __nv_bfloat16*>(bindings->gate),
        static_cast<__nv_bfloat16*>(bindings->output), num_splits,
        spec->max_splits);
    apxinf::framework::check_cuda(cudaGetLastError());
  });
}

extern "C" apxinf_status_t apxinf_qwen35_dynamic_kv_dequant(
    apxinf_runtime_t runtime, const apxinf_qwen35_dynamic_dequant_spec_t* spec,
    const apxinf_qwen35_dynamic_dequant_bindings_t* bindings) {
  return apxinf::framework::abi_boundary([&] {
    if (runtime == nullptr || spec == nullptr || bindings == nullptr ||
        !valid_dynamic_dequant_spec(*spec) ||
        !valid_dynamic_dequant_bindings(*bindings)) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                    "invalid Qwen3.5 dynamic KV dequant arguments");
    }
    apxinf::framework::check_cuda(cudaSetDevice(runtime->device));
    auto stream = static_cast<cudaStream_t>(bindings->stream);
    const int elements = spec->valid_tokens * kKvHeads * kHeadDim;
    constexpr int threads = 256;
    const int blocks = (elements + threads - 1) / threads;
    dynamic_kv_dequant_kernel<<<blocks, threads, 0, stream>>>(
        static_cast<const __nv_fp8_e4m3*>(bindings->key_cache),
        static_cast<const __nv_fp8_e4m3*>(bindings->value_cache),
        static_cast<const float*>(bindings->key_scales),
        static_cast<const float*>(bindings->value_scales),
        static_cast<__nv_bfloat16*>(bindings->key),
        static_cast<__nv_bfloat16*>(bindings->value), elements);
    apxinf::framework::check_cuda(cudaGetLastError());
  });
}
