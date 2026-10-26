#include "../include/apxinf_cuda/moe.h"
#include "../framework/runtime_internal.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cfloat>
#include <climits>
#include <cstddef>
#include <cstdint>

namespace {

using apxinf::framework::Failure;

constexpr int kThreads = 256;
constexpr int kMaxBlocks = 4096;
constexpr int kMaxExperts = 256;
constexpr int kMaxTopK = 8;
constexpr int kNvfp4RowMajorScales = 1;
constexpr int kWarpSize = 32;
constexpr int kWarpsPerBlock = kThreads / kWarpSize;
constexpr size_t kMaxCachedActivationBytes = 32 * 1024;
static_assert(kThreads % kWarpSize == 0,
              "routed NVFP4 tile must contain whole warps");

template <class T>
__device__ float as_float(T value);

template <>
__device__ float as_float<float>(float value) {
  return value;
}

template <>
__device__ float as_float<__half>(__half value) {
  return __half2float(value);
}

template <>
__device__ float as_float<__nv_bfloat16>(__nv_bfloat16 value) {
  return __bfloat162float(value);
}

template <class T>
__device__ T from_float(float value);

template <>
__device__ float from_float<float>(float value) {
  return value;
}

template <>
__device__ __half from_float<__half>(float value) {
  return __float2half_rn(value);
}

template <>
__device__ __nv_bfloat16 from_float<__nv_bfloat16>(float value) {
  return __float2bfloat16_rn(value);
}

template <class T>
__global__ void route_topk_kernel(const T* __restrict__ logits,
                                  int32_t* __restrict__ expert_ids,
                                  float* __restrict__ expert_weights,
                                  int tokens, int experts, int top_k,
                                  bool renormalize) {
  const int token = blockIdx.x;
  if (threadIdx.x != 0 || token >= tokens) return;

  const T* row = logits + static_cast<int64_t>(token) * experts;
  float maximum = -FLT_MAX;
  for (int expert = 0; expert < experts; ++expert) {
    maximum = fmaxf(maximum, as_float(row[expert]));
  }
  float denominator = 0.0f;
  for (int expert = 0; expert < experts; ++expert) {
    denominator += expf(as_float(row[expert]) - maximum);
  }

  float selected_values[kMaxTopK];
  int selected_ids[kMaxTopK];
#pragma unroll
  for (int slot = 0; slot < kMaxTopK; ++slot) {
    selected_values[slot] = -FLT_MAX;
    selected_ids[slot] = INT_MAX;
  }
  for (int expert = 0; expert < experts; ++expert) {
    const float value = as_float(row[expert]);
    int insertion = top_k;
    for (int slot = 0; slot < top_k; ++slot) {
      if (value > selected_values[slot] ||
          (value == selected_values[slot] && expert < selected_ids[slot])) {
        insertion = slot;
        break;
      }
    }
    if (insertion == top_k) continue;
    for (int slot = top_k - 1; slot > insertion; --slot) {
      selected_values[slot] = selected_values[slot - 1];
      selected_ids[slot] = selected_ids[slot - 1];
    }
    selected_values[insertion] = value;
    selected_ids[insertion] = expert;
  }

  float selected_sum = 0.0f;
  float weights[kMaxTopK];
  for (int slot = 0; slot < top_k; ++slot) {
    weights[slot] = expf(selected_values[slot] - maximum) / denominator;
    selected_sum += weights[slot];
  }
  const int64_t output = static_cast<int64_t>(token) * top_k;
  for (int slot = 0; slot < top_k; ++slot) {
    expert_ids[output + slot] = selected_ids[slot];
    expert_weights[output + slot] =
        renormalize ? weights[slot] / selected_sum : weights[slot];
  }
}

__global__ void build_assignments_kernel(
    const int32_t* __restrict__ expert_ids,
    const float* __restrict__ expert_weights,
    int32_t* __restrict__ assignment_expert_ids,
    int32_t* __restrict__ assignment_token_ids,
    int32_t* __restrict__ assignment_slots,
    int32_t* __restrict__ token_assignment_indices,
    float* __restrict__ assignment_weights,
    int32_t* __restrict__ expert_offsets, int tokens, int experts, int top_k) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  int cursor = 0;
  for (int expert = 0; expert < experts; ++expert) {
    expert_offsets[expert] = cursor;
    for (int token = 0; token < tokens; ++token) {
      const int token_offset = token * top_k;
      for (int slot = 0; slot < top_k; ++slot) {
        const int route = token_offset + slot;
        if (expert_ids[route] != expert) continue;
        assignment_expert_ids[cursor] = expert;
        assignment_token_ids[cursor] = token;
        assignment_slots[cursor] = slot;
        token_assignment_indices[token_offset + slot] = cursor;
        assignment_weights[cursor] = expert_weights[route];
        ++cursor;
      }
    }
  }
  expert_offsets[experts] = cursor;
}

__device__ __forceinline__ float decode_e2m1(uint8_t code) {
  float magnitude = 0.0f;
  switch (code & 0x7U) {
    case 1:
      magnitude = 0.5f;
      break;
    case 2:
      magnitude = 1.0f;
      break;
    case 3:
      magnitude = 1.5f;
      break;
    case 4:
      magnitude = 2.0f;
      break;
    case 5:
      magnitude = 3.0f;
      break;
    case 6:
      magnitude = 4.0f;
      break;
    case 7:
      magnitude = 6.0f;
      break;
  }
  return (code & 0x8U) != 0 ? -magnitude : magnitude;
}

__device__ __forceinline__ float decode_e4m3_nonnegative(uint8_t code) {
  code &= 0x7fU;
  const int exponent = (code >> 3) & 0x0f;
  const float fraction = static_cast<float>(code & 0x07U);
  if (exponent == 0) return fraction * 0.001953125f;
  return (1.0f + fraction * 0.125f) *
         exp2f(static_cast<float>(exponent - 7));
}

// Accumulate the two E2M1 values stored in one byte.  Adjacent warp lanes read
// adjacent packed bytes, unlike the scalar-output candidate where adjacent
// threads walked distinct output rows and generated strided global loads.
__device__ __forceinline__ float nvfp4_pair_dot(
    const uint8_t* __restrict__ packed,
    const uint8_t* __restrict__ scales, const float* __restrict__ alpha,
    int expert, int output, int pair, int outputs, int inputs, int block_size,
    float activation_low, float activation_high) {
  const int64_t row = static_cast<int64_t>(expert) * outputs + output;
  const uint8_t codes = packed[row * (inputs / 2) + pair];
  const uint8_t scale_code =
      scales[row * (inputs / block_size) + (pair * 2) / block_size];
  const float scale = decode_e4m3_nonnegative(scale_code) * alpha[expert];
  return activation_low * (decode_e2m1(codes & 0x0fU) * scale) +
         activation_high * (decode_e2m1(codes >> 4) * scale);
}

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
  for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
    value += __shfl_down_sync(0xffffffffU, value, offset);
  }
  return value;
}

__device__ __forceinline__ float stable_sigmoid(float value) {
  if (value >= 0.0f) return 1.0f / (1.0f + expf(-value));
  const float exponential = expf(value);
  return exponential / (1.0f + exponential);
}

// One warp owns one output element and reduces K across its 32 lanes.  A block
// therefore computes an eight-output tile.  This keeps the checkpoint's native
// [expert, output, packed-K] layout and avoids materializing dequantized weights.
__global__ void nvfp4_swiglu_warp_kernel(
    const __nv_bfloat16* __restrict__ input,
    const int32_t* __restrict__ assignment_expert_ids,
    const int32_t* __restrict__ assignment_token_ids,
    const uint8_t* __restrict__ gate_weight,
    const uint8_t* __restrict__ gate_scales,
    const float* __restrict__ gate_alpha,
    const uint8_t* __restrict__ up_weight,
    const uint8_t* __restrict__ up_scales,
    const float* __restrict__ up_alpha,
    __nv_bfloat16* __restrict__ intermediate_output, int rows, int tokens,
    int experts, int hidden, int intermediate, int block_size, bool routed,
    bool cache_activation) {
  extern __shared__ __nv_bfloat16 shared_activation[];
  const int warp = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x % kWarpSize;
  const int output_tiles =
      (intermediate + kWarpsPerBlock - 1) / kWarpsPerBlock;
  const int64_t tile_count = static_cast<int64_t>(rows) * output_tiles;
  for (int64_t tile = blockIdx.x; tile < tile_count; tile += gridDim.x) {
    const int row = static_cast<int>(tile / output_tiles);
    const int output =
        static_cast<int>(tile % output_tiles) * kWarpsPerBlock + warp;
    const int expert = routed ? assignment_expert_ids[row] : 0;
    const int token = routed ? assignment_token_ids[row] : row;
    const bool valid_route = expert >= 0 && expert < experts && token >= 0 &&
                             token < tokens;
    if (cache_activation) {
      for (int feature = threadIdx.x; feature < hidden; feature += blockDim.x) {
        shared_activation[feature] =
            valid_route
                ? input[static_cast<int64_t>(token) * hidden + feature]
                : __float2bfloat16(0.0f);
      }
      __syncthreads();
    }
    if (output < intermediate) {
      if (!valid_route) {
        if (lane == 0) {
          intermediate_output[static_cast<int64_t>(row) * intermediate +
                              output] = __float2bfloat16(0.0f);
        }
      } else {
        float gate = 0.0f;
        float up = 0.0f;
        const int pairs = hidden / 2;
        const __nv_bfloat16* activation =
            cache_activation
                ? shared_activation
                : input + static_cast<int64_t>(token) * hidden;
        for (int pair = lane; pair < pairs; pair += kWarpSize) {
          const int feature = pair * 2;
          const float activation_low = __bfloat162float(activation[feature]);
          const float activation_high =
              __bfloat162float(activation[feature + 1]);
          gate += nvfp4_pair_dot(
              gate_weight, gate_scales, gate_alpha, expert, output, pair,
              intermediate, hidden, block_size, activation_low,
              activation_high);
          up += nvfp4_pair_dot(
              up_weight, up_scales, up_alpha, expert, output, pair,
              intermediate, hidden, block_size, activation_low,
              activation_high);
        }
        gate = warp_sum(gate);
        up = warp_sum(up);
        if (lane == 0) {
          gate = __bfloat162float(__float2bfloat16(gate));
          up = __bfloat162float(__float2bfloat16(up));
          intermediate_output[static_cast<int64_t>(row) * intermediate +
                              output] =
              __float2bfloat16((gate * stable_sigmoid(gate)) * up);
        }
      }
    }
    if (cache_activation) __syncthreads();
  }
}

__global__ void nvfp4_down_warp_kernel(
    const __nv_bfloat16* __restrict__ intermediate_input,
    const int32_t* __restrict__ assignment_expert_ids,
    const uint8_t* __restrict__ down_weight,
    const uint8_t* __restrict__ down_scales,
    const float* __restrict__ down_alpha,
    __nv_bfloat16* __restrict__ output, int rows, int experts, int hidden,
    int intermediate, int block_size, bool routed, bool cache_activation) {
  extern __shared__ __nv_bfloat16 shared_activation[];
  const int warp = threadIdx.x / kWarpSize;
  const int lane = threadIdx.x % kWarpSize;
  const int output_tiles =
      (hidden + kWarpsPerBlock - 1) / kWarpsPerBlock;
  const int64_t tile_count = static_cast<int64_t>(rows) * output_tiles;
  for (int64_t tile = blockIdx.x; tile < tile_count; tile += gridDim.x) {
    const int row = static_cast<int>(tile / output_tiles);
    const int output_feature =
        static_cast<int>(tile % output_tiles) * kWarpsPerBlock + warp;
    const int expert = routed ? assignment_expert_ids[row] : 0;
    if (cache_activation) {
      for (int feature = threadIdx.x; feature < intermediate;
           feature += blockDim.x) {
        shared_activation[feature] =
            intermediate_input[static_cast<int64_t>(row) * intermediate +
                               feature];
      }
      __syncthreads();
    }
    if (output_feature < hidden) {
      if (expert < 0 || expert >= experts) {
        if (lane == 0) {
          output[static_cast<int64_t>(row) * hidden + output_feature] =
              __float2bfloat16(0.0f);
        }
      } else {
        float result = 0.0f;
        const int pairs = intermediate / 2;
        const __nv_bfloat16* activation =
            cache_activation
                ? shared_activation
                : intermediate_input +
                      static_cast<int64_t>(row) * intermediate;
        for (int pair = lane; pair < pairs; pair += kWarpSize) {
          const int feature = pair * 2;
          result += nvfp4_pair_dot(
              down_weight, down_scales, down_alpha, expert, output_feature,
              pair, hidden, intermediate, block_size,
              __bfloat162float(activation[feature]),
              __bfloat162float(activation[feature + 1]));
        }
        result = warp_sum(result);
        if (lane == 0) {
          output[static_cast<int64_t>(row) * hidden + output_feature] =
              __float2bfloat16(result);
        }
      }
    }
    if (cache_activation) __syncthreads();
  }
}

template <class T>
__global__ void combine_kernel(
    const T* __restrict__ assignment_output,
    const float* __restrict__ assignment_weights,
    const int32_t* __restrict__ token_assignment_indices,
    const T* __restrict__ shared_output,
    const T* __restrict__ shared_gate_logits, T* __restrict__ output,
    int tokens, int top_k, int hidden) {
  const int token = static_cast<int>(blockIdx.x);
  if (token >= tokens) return;

  // The inverse map points from the token-major route slots back into the
  // expert-major output. Sorting those row indices reproduces the legacy
  // expert-major accumulation order exactly while reducing each output from
  // a scan over every route in the batch to just this token's top-k routes.
  __shared__ int route_rows[kMaxTopK];
  if (threadIdx.x < top_k) {
    route_rows[threadIdx.x] =
        token_assignment_indices[token * top_k + threadIdx.x];
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    for (int index = 1; index < top_k; ++index) {
      const int value = route_rows[index];
      int cursor = index;
      while (cursor > 0 && route_rows[cursor - 1] > value) {
        route_rows[cursor] = route_rows[cursor - 1];
        --cursor;
      }
      route_rows[cursor] = value;
    }
  }
  __syncthreads();

  for (int dimension = threadIdx.x; dimension < hidden;
       dimension += blockDim.x) {
    float result = 0.0f;
    for (int slot = 0; slot < top_k; ++slot) {
      const int route = route_rows[slot];
      result += assignment_weights[route] *
                as_float(assignment_output[static_cast<int64_t>(route) *
                                           hidden + dimension]);
    }
    const float gate_logit = as_float(shared_gate_logits[token]);
    const float gate = gate_logit >= 0.0f
                           ? 1.0f / (1.0f + expf(-gate_logit))
                           : expf(gate_logit) / (1.0f + expf(gate_logit));
    result += gate * as_float(
                         shared_output[static_cast<int64_t>(token) * hidden +
                                       dimension]);
    output[static_cast<int64_t>(token) * hidden + dimension] =
        from_float<T>(result);
  }
}

void validate_route_spec(const apxinf_moe_route_spec_t& spec) {
  if (spec.version != APXINF_MOE_ROUTE_SPEC_VERSION ||
      (spec.logits_dtype != APXINF_DTYPE_F32 &&
       spec.logits_dtype != APXINF_DTYPE_F16 &&
       spec.logits_dtype != APXINF_DTYPE_BF16) ||
      spec.tokens == 0 || spec.tokens > INT32_MAX || spec.experts == 0 ||
      spec.experts > kMaxExperts || spec.top_k == 0 ||
      spec.top_k > kMaxTopK || spec.top_k > spec.experts ||
      static_cast<uint64_t>(spec.tokens) * spec.top_k > INT32_MAX ||
      spec.renormalize > 1) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT, "invalid routed-MoE route Spec");
  }
}

void validate_combine_spec(const apxinf_moe_combine_spec_t& spec) {
  if (spec.version != APXINF_MOE_COMBINE_SPEC_VERSION ||
      (spec.dtype != APXINF_DTYPE_F32 && spec.dtype != APXINF_DTYPE_F16 &&
       spec.dtype != APXINF_DTYPE_BF16) ||
      spec.tokens == 0 || spec.tokens > INT32_MAX || spec.routes == 0 ||
      spec.routes > INT32_MAX || spec.top_k == 0 || spec.top_k > kMaxTopK ||
      static_cast<uint64_t>(spec.tokens) * spec.top_k != spec.routes ||
      spec.hidden == 0 || spec.hidden > INT32_MAX) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                  "invalid routed-MoE combine Spec");
  }
}

void validate_nvfp4_experts_spec(
    const apxinf_moe_nvfp4_experts_spec_t& spec) {
  if (spec.version != APXINF_MOE_NVFP4_EXPERTS_SPEC_VERSION ||
      spec.dtype != APXINF_DTYPE_BF16 || spec.tokens == 0 ||
      spec.tokens > INT32_MAX || spec.routes == 0 || spec.routes > INT32_MAX ||
      spec.experts == 0 || spec.experts > kMaxExperts || spec.hidden == 0 ||
      spec.hidden > INT32_MAX || spec.intermediate == 0 ||
      spec.intermediate > INT32_MAX || spec.block_size != 16 ||
      spec.hidden % 16 != 0 || spec.intermediate % 16 != 0 ||
      spec.scale_layout != kNvfp4RowMajorScales) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                  "invalid routed-MoE NVFP4 experts Spec");
  }
}

int blocks_for(int64_t count) {
  const int64_t blocks = (count + kThreads - 1) / kThreads;
  if (blocks < 1) return 1;
  return static_cast<int>(blocks < kMaxBlocks ? blocks : kMaxBlocks);
}

int blocks_for_output_tiles(int rows, int outputs) {
  const int64_t tiles_per_row =
      (static_cast<int64_t>(outputs) + kWarpsPerBlock - 1) /
      kWarpsPerBlock;
  const int64_t blocks = static_cast<int64_t>(rows) * tiles_per_row;
  if (blocks < 1) return 1;
  return static_cast<int>(blocks < kMaxBlocks ? blocks : kMaxBlocks);
}

template <class T>
cudaError_t launch_route(const apxinf_moe_route_spec_t& spec,
                         const apxinf_moe_route_bindings_t& bindings) {
  auto stream = static_cast<cudaStream_t>(bindings.stream);
  route_topk_kernel<T><<<static_cast<int>(spec.tokens), 1, 0, stream>>>(
      static_cast<const T*>(bindings.logits), bindings.expert_ids,
      bindings.expert_weights, static_cast<int>(spec.tokens),
      static_cast<int>(spec.experts), static_cast<int>(spec.top_k),
      spec.renormalize != 0);
  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess) return status;
  build_assignments_kernel<<<1, 1, 0, stream>>>(
      bindings.expert_ids, bindings.expert_weights,
      bindings.assignment_expert_ids, bindings.assignment_token_ids,
      bindings.assignment_slots, bindings.token_assignment_indices,
      bindings.assignment_weights,
      bindings.expert_offsets, static_cast<int>(spec.tokens),
      static_cast<int>(spec.experts), static_cast<int>(spec.top_k));
  return cudaGetLastError();
}

template <class T>
cudaError_t launch_combine(const apxinf_moe_combine_spec_t& spec,
                           const apxinf_moe_combine_bindings_t& bindings) {
  combine_kernel<T><<<static_cast<int>(spec.tokens), kThreads, 0,
                      static_cast<cudaStream_t>(bindings.stream)>>>(
      static_cast<const T*>(bindings.assignment_output),
      bindings.assignment_weights, bindings.token_assignment_indices,
      static_cast<const T*>(bindings.shared_output),
      static_cast<const T*>(bindings.shared_gate_logits),
      static_cast<T*>(bindings.output), static_cast<int>(spec.tokens),
      static_cast<int>(spec.top_k), static_cast<int>(spec.hidden));
  return cudaGetLastError();
}

cudaError_t launch_nvfp4_experts(
    const apxinf_moe_nvfp4_experts_spec_t& spec,
    const apxinf_moe_nvfp4_experts_bindings_t& bindings) {
  auto stream = static_cast<cudaStream_t>(bindings.stream);
  const int tokens = static_cast<int>(spec.tokens);
  const int routes = static_cast<int>(spec.routes);
  const int experts = static_cast<int>(spec.experts);
  const int hidden = static_cast<int>(spec.hidden);
  const int intermediate = static_cast<int>(spec.intermediate);
  const int block_size = static_cast<int>(spec.block_size);
  const size_t hidden_bytes =
      static_cast<size_t>(hidden) * sizeof(__nv_bfloat16);
  const size_t intermediate_bytes =
      static_cast<size_t>(intermediate) * sizeof(__nv_bfloat16);
  const bool cache_hidden = hidden_bytes <= kMaxCachedActivationBytes;
  const bool cache_intermediate =
      intermediate_bytes <= kMaxCachedActivationBytes;

  nvfp4_swiglu_warp_kernel<<<blocks_for_output_tiles(routes, intermediate),
                               kThreads, cache_hidden ? hidden_bytes : 0,
                               stream>>>(
      static_cast<const __nv_bfloat16*>(bindings.input),
      bindings.assignment_expert_ids, bindings.assignment_token_ids,
      static_cast<const uint8_t*>(bindings.expert_gate_weight),
      static_cast<const uint8_t*>(bindings.expert_gate_scales),
      bindings.expert_gate_alpha,
      static_cast<const uint8_t*>(bindings.expert_up_weight),
      static_cast<const uint8_t*>(bindings.expert_up_scales),
      bindings.expert_up_alpha,
      static_cast<__nv_bfloat16*>(bindings.route_intermediate), routes, tokens,
      experts, hidden, intermediate, block_size, true, cache_hidden);
  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess) return status;

  nvfp4_down_warp_kernel<<<blocks_for_output_tiles(routes, hidden), kThreads,
                            cache_intermediate ? intermediate_bytes : 0,
                            stream>>>(
      static_cast<const __nv_bfloat16*>(bindings.route_intermediate),
      bindings.assignment_expert_ids,
      static_cast<const uint8_t*>(bindings.expert_down_weight),
      static_cast<const uint8_t*>(bindings.expert_down_scales),
      bindings.expert_down_alpha,
      static_cast<__nv_bfloat16*>(bindings.assignment_output), routes, experts,
      hidden, intermediate, block_size, true, cache_intermediate);
  status = cudaGetLastError();
  if (status != cudaSuccess) return status;

  nvfp4_swiglu_warp_kernel<<<blocks_for_output_tiles(tokens, intermediate),
                               kThreads, cache_hidden ? hidden_bytes : 0,
                               stream>>>(
      static_cast<const __nv_bfloat16*>(bindings.input), nullptr, nullptr,
      static_cast<const uint8_t*>(bindings.shared_gate_weight),
      static_cast<const uint8_t*>(bindings.shared_gate_scales),
      bindings.shared_gate_alpha,
      static_cast<const uint8_t*>(bindings.shared_up_weight),
      static_cast<const uint8_t*>(bindings.shared_up_scales),
      bindings.shared_up_alpha,
      static_cast<__nv_bfloat16*>(bindings.shared_intermediate), tokens, tokens,
      1, hidden, intermediate, block_size, false, cache_hidden);
  status = cudaGetLastError();
  if (status != cudaSuccess) return status;

  nvfp4_down_warp_kernel<<<blocks_for_output_tiles(tokens, hidden), kThreads,
                            cache_intermediate ? intermediate_bytes : 0,
                            stream>>>(
      static_cast<const __nv_bfloat16*>(bindings.shared_intermediate), nullptr,
      static_cast<const uint8_t*>(bindings.shared_down_weight),
      static_cast<const uint8_t*>(bindings.shared_down_scales),
      bindings.shared_down_alpha,
      static_cast<__nv_bfloat16*>(bindings.shared_output), tokens, 1, hidden,
      intermediate, block_size, false, cache_intermediate);
  return cudaGetLastError();
}

}  // namespace

extern "C" apxinf_status_t apxinf_moe_route_topk(
    apxinf_runtime_t runtime, const apxinf_moe_route_spec_t* spec,
    const apxinf_moe_route_bindings_t* bindings) {
  return apxinf::framework::abi_boundary([&] {
    if (runtime == nullptr || spec == nullptr || bindings == nullptr) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                    "null routed-MoE route argument");
    }
    validate_route_spec(*spec);
    if (bindings->logits == nullptr || bindings->expert_ids == nullptr ||
        bindings->expert_weights == nullptr ||
        bindings->assignment_expert_ids == nullptr ||
        bindings->assignment_token_ids == nullptr ||
        bindings->assignment_slots == nullptr ||
        bindings->token_assignment_indices == nullptr ||
        bindings->assignment_weights == nullptr ||
        bindings->expert_offsets == nullptr) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                    "missing routed-MoE route binding");
    }
    apxinf::framework::check_cuda(cudaSetDevice(runtime->device));
    cudaError_t status = cudaErrorInvalidValue;
    if (spec->logits_dtype == APXINF_DTYPE_F32) {
      status = launch_route<float>(*spec, *bindings);
    } else if (spec->logits_dtype == APXINF_DTYPE_BF16) {
      status = launch_route<__nv_bfloat16>(*spec, *bindings);
    } else {
      status = launch_route<__half>(*spec, *bindings);
    }
    apxinf::framework::check_cuda(status);
  });
}

extern "C" apxinf_status_t apxinf_moe_combine(
    apxinf_runtime_t runtime, const apxinf_moe_combine_spec_t* spec,
    const apxinf_moe_combine_bindings_t* bindings) {
  return apxinf::framework::abi_boundary([&] {
    if (runtime == nullptr || spec == nullptr || bindings == nullptr) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                    "null routed-MoE combine argument");
    }
    validate_combine_spec(*spec);
    if (bindings->assignment_output == nullptr ||
        bindings->assignment_token_ids == nullptr ||
        bindings->assignment_weights == nullptr ||
        bindings->token_assignment_indices == nullptr ||
        bindings->shared_output == nullptr ||
        bindings->shared_gate_logits == nullptr || bindings->output == nullptr) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                    "missing routed-MoE combine binding");
    }
    apxinf::framework::check_cuda(cudaSetDevice(runtime->device));
    cudaError_t status = cudaErrorInvalidValue;
    if (spec->dtype == APXINF_DTYPE_F32) {
      status = launch_combine<float>(*spec, *bindings);
    } else if (spec->dtype == APXINF_DTYPE_BF16) {
      status = launch_combine<__nv_bfloat16>(*spec, *bindings);
    } else {
      status = launch_combine<__half>(*spec, *bindings);
    }
    apxinf::framework::check_cuda(status);
  });
}

extern "C" apxinf_status_t apxinf_moe_nvfp4_experts(
    apxinf_runtime_t runtime, const apxinf_moe_nvfp4_experts_spec_t* spec,
    const apxinf_moe_nvfp4_experts_bindings_t* bindings) {
  return apxinf::framework::abi_boundary([&] {
    if (runtime == nullptr || spec == nullptr || bindings == nullptr) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                    "null routed-MoE NVFP4 experts argument");
    }
    validate_nvfp4_experts_spec(*spec);
    if (bindings->input == nullptr ||
        bindings->assignment_expert_ids == nullptr ||
        bindings->assignment_token_ids == nullptr ||
        bindings->expert_gate_weight == nullptr ||
        bindings->expert_gate_scales == nullptr ||
        bindings->expert_gate_alpha == nullptr ||
        bindings->expert_up_weight == nullptr ||
        bindings->expert_up_scales == nullptr ||
        bindings->expert_up_alpha == nullptr ||
        bindings->expert_down_weight == nullptr ||
        bindings->expert_down_scales == nullptr ||
        bindings->expert_down_alpha == nullptr ||
        bindings->shared_gate_weight == nullptr ||
        bindings->shared_gate_scales == nullptr ||
        bindings->shared_gate_alpha == nullptr ||
        bindings->shared_up_weight == nullptr ||
        bindings->shared_up_scales == nullptr ||
        bindings->shared_up_alpha == nullptr ||
        bindings->shared_down_weight == nullptr ||
        bindings->shared_down_scales == nullptr ||
        bindings->shared_down_alpha == nullptr ||
        bindings->route_intermediate == nullptr ||
        bindings->shared_intermediate == nullptr ||
        bindings->assignment_output == nullptr ||
        bindings->shared_output == nullptr) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                    "missing routed-MoE NVFP4 experts binding");
    }
    apxinf::framework::check_cuda(cudaSetDevice(runtime->device));
    apxinf::framework::check_cuda(launch_nvfp4_experts(*spec, *bindings));
  });
}
