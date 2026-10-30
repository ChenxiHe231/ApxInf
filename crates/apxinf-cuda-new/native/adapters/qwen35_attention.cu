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
#include <cstdlib>

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
  __shared__ __nv_bfloat16 shared_query[kQueriesPerKvHead][kHeadDim];
  __shared__ __nv_bfloat16 shared_key[kScorePositionsPerBlock][kHeadDim];
  __shared__ float shared_scores[kQueriesPerKvHead][kScorePositionsPerBlock];
  __shared__ float shared_weights[kQueriesPerKvHead][kScorePositionsPerBlock];
  __shared__ float maxima[kQueriesPerKvHead];
  __shared__ float denominators[kQueriesPerKvHead];
  __shared__ float rescale[kQueriesPerKvHead];

  for (int index = dim; index < kQueriesPerKvHead * kHeadDim;
       index += kThreads) {
    const int head = index / kHeadDim;
    const int component = index % kHeadDim;
    const int query_head = kv_head * kQueriesPerKvHead + head;
    const int64_t query_index =
        (static_cast<int64_t>(token) * kQHeads + query_head) * kHeadDim +
        component;
    shared_query[head][component] = query[query_index];
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

    const int query_head = local_query_head;
    for (int item = 0; item < count; ++item) {
      float dot = 0.0F;
#pragma unroll
      for (int pair = lane; pair < kHeadDim / 2; pair += 32) {
        const float2 query_pair = __bfloat1622float2(
            *reinterpret_cast<const __nv_bfloat162*>(
                &shared_query[query_head][2 * pair]));
        const float2 key_pair = __bfloat1622float2(
            *reinterpret_cast<const __nv_bfloat162*>(
                &shared_key[item][2 * pair]));
        dot = fmaf(query_pair.x, key_pair.x,
                   fmaf(query_pair.y, key_pair.y, dot));
      }
      dot = warp_sum(dot);
      if (lane == 0) shared_scores[query_head][item] = dot * 0.0625F;
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
    const int splits = min(
        static_cast<int>(APXINF_QWEN35_ATTENTION_MAX_SPLITS),
        max(1, (valid_tokens + kValuePositionsPerTile - 1) /
                   kValuePositionsPerTile));
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
