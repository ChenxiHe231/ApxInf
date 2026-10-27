/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Adapted from TensorRT-Edge-LLM's NVFP4 CuTeDSL MoE runner and FP4 support
 * kernels.  The ApxInf adapter removes TensorRT tensor/plugin dependencies,
 * keeps the raw-pointer CUDA ABI, and emits BF16 output.
 */

#include "../../include/apxinf_cuda/moe.h"
#include "../../framework/runtime_internal.h"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>

using apxinf::framework::Failure;

#define CUTE_DSL_CUDA_ERROR_CHECK(error)                                      \
  do {                                                                         \
    cudaError_t status_ = (error);                                              \
    if (status_ != cudaSuccess) {                                               \
      throw Failure(APXINF_STATUS_CUDA_ERROR, cudaGetErrorString(status_));     \
    }                                                                          \
  } while (0)
#include "generated/nvfp4_moe_sm110_fc1_swiglu_n128.h"
#include "generated/nvfp4_moe_sm110_fc2_n128_bf16.h"
#undef CUTE_DSL_CUDA_ERROR_CHECK

namespace {

constexpr int kSfVecSize = 16;
constexpr int kRowTile = 128;
constexpr int kThreads = 512;
constexpr int kMaxDecodeExperts = 512;
constexpr int kMaxResidentThreadsPerSm = 2048;
constexpr float kSfScaleEpsilon = 1.0e-20f;
constexpr size_t kAlignment = 256;

struct WorkspaceLayout {
  size_t input_fp4{};
  size_t input_sf{};
  size_t tile_group{};
  size_t tile_limit{};
  size_t permuted_to_expanded{};
  size_t num_tiles{};
  size_t fc1_fp4{};
  size_t fc1_sf{};
  size_t total{};
  int32_t permuted_m{};
  int32_t tile_entries{};
  int32_t padded_inter_sf{};
};

size_t align_up(size_t value, size_t alignment) {
  return ((value + alignment - 1) / alignment) * alignment;
}

int64_t pad_up(int64_t value, int64_t alignment) {
  return ((value + alignment - 1) / alignment) * alignment;
}

WorkspaceLayout workspace_layout(const apxinf_moe_edge_nvfp4_spec_t& spec) {
  WorkspaceLayout layout{};
  const int64_t routes = static_cast<int64_t>(spec.tokens) * spec.top_k;
  layout.permuted_m = static_cast<int32_t>(
      pad_up(routes + static_cast<int64_t>(spec.experts) * (kRowTile - 1),
             kRowTile));
  layout.tile_entries = std::max(1, layout.permuted_m / kRowTile);
  layout.padded_inter_sf =
      static_cast<int32_t>(pad_up(spec.intermediate / kSfVecSize, 4));
  size_t cursor = 0;
  auto place = [&](size_t& offset, size_t bytes) {
    offset = cursor;
    cursor = align_up(cursor + bytes, kAlignment);
  };
  place(layout.input_fp4, static_cast<size_t>(routes) * spec.hidden / 2);
  place(layout.input_sf,
        static_cast<size_t>(routes) * spec.hidden / kSfVecSize);
  place(layout.tile_group,
        static_cast<size_t>(layout.tile_entries) * sizeof(int32_t));
  place(layout.tile_limit,
        static_cast<size_t>(layout.tile_entries) * sizeof(int32_t));
  place(layout.permuted_to_expanded,
        static_cast<size_t>(layout.permuted_m) * sizeof(int32_t));
  place(layout.num_tiles, sizeof(int32_t));
  place(layout.fc1_fp4,
        static_cast<size_t>(layout.permuted_m) * spec.intermediate / 2);
  place(layout.fc1_sf,
        static_cast<size_t>(layout.permuted_m) * layout.padded_inter_sf);
  layout.total = cursor + kAlignment;
  return layout;
}

void validate_spec(const apxinf_moe_edge_nvfp4_spec_t* spec) {
  if (spec == nullptr || spec->version != APXINF_MOE_EDGE_NVFP4_SPEC_VERSION) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                  "invalid Edge NVFP4 MoE spec version");
  }
  if (spec->dtype != APXINF_DTYPE_BF16 || spec->tokens == 0 ||
      spec->experts != 256 || spec->top_k != 8 ||
      spec->hidden == 0 || spec->hidden % 128 != 0 ||
      spec->intermediate == 0 || spec->intermediate % 64 != 0) {
    throw Failure(
        APXINF_STATUS_UNSUPPORTED,
        "Edge NVFP4 MoE requires BF16, 256 experts, top-8, hidden multiple "
        "of 128, and intermediate multiple of 64");
  }
  const uint64_t routes = static_cast<uint64_t>(spec->tokens) * spec->top_k;
  const uint64_t padded =
      pad_up(routes + static_cast<uint64_t>(spec->experts) * (kRowTile - 1),
             kRowTile);
  if (routes > INT32_MAX || padded > INT32_MAX) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                  "Edge NVFP4 MoE route geometry exceeds int32");
  }
}

template <typename T>
T* at(void* base, size_t offset) {
  return reinterpret_cast<T*>(static_cast<std::byte*>(base) + offset);
}

__global__ void build_layout_kernel(
    const int32_t* __restrict__ selected_experts,
    int32_t* __restrict__ permuted_idx,
    int32_t* __restrict__ tile_group_idx,
    int32_t* __restrict__ tile_mn_limit,
    int32_t* __restrict__ num_tiles, int32_t routes, int32_t experts,
    int32_t max_padded) {
  extern __shared__ int32_t shared[];
  int32_t* counts = shared;
  int32_t* offsets = shared + experts;
  int32_t* scatter = shared + 2 * experts;
  const int tid = threadIdx.x;
  for (int e = tid; e < experts; e += blockDim.x) counts[e] = 0;
  __syncthreads();
  for (int row = tid; row < routes; row += blockDim.x) {
    const int expert = selected_experts[row];
    if (expert >= 0 && expert < experts) atomicAdd_block(&counts[expert], 1);
  }
  __syncthreads();
  if (tid == 0) {
    int running_offset = 0;
    int running_tiles = 0;
    for (int expert = 0; expert < experts; ++expert) {
      const int count = counts[expert];
      const int padded = count > 0 ? ((count + kRowTile - 1) / kRowTile) * kRowTile : 0;
      const int expert_tiles = padded / kRowTile;
      offsets[expert] = running_offset;
      for (int tile = 0; tile < expert_tiles; ++tile) {
        tile_group_idx[running_tiles + tile] = expert;
        tile_mn_limit[running_tiles + tile] =
            running_offset + min((tile + 1) * kRowTile, count);
      }
      running_offset += padded;
      running_tiles += expert_tiles;
    }
    num_tiles[0] = running_tiles;
  }
  __syncthreads();
  for (int row = tid; row < max_padded; row += blockDim.x) permuted_idx[row] = -1;
  for (int e = tid; e < experts; e += blockDim.x) scatter[e] = 0;
  // All padded entries must be initialized before any live route is written.
  // Without this barrier a slower initialization lane can overwrite a route
  // inserted by another lane, leaving the generated MoE kernel with a missing
  // or invalid permutation entry.
  __syncthreads();
  // Preserve route order inside each expert. Besides making prefill bitwise
  // reproducible, this avoids run-to-run accumulation-order drift in FC2 that
  // can eventually change greedy decoding after many tokens.
  if (tid == 0) {
    for (int row = 0; row < routes; ++row) {
      const int expert = selected_experts[row];
      if (expert >= 0 && expert < experts) {
        const int position = scatter[expert]++;
        permuted_idx[offsets[expert] + position] = row;
      }
    }
  }
}

struct Bf16Vector {
  __nv_bfloat162 values[4];
};

__device__ __forceinline__ float reciprocal_approx(float value) {
  float result;
  asm volatile("rcp.approx.ftz.f32 %0, %1;" : "=f"(result) : "f"(value));
  return result;
}

__device__ __forceinline__ uint8_t quantize_e2m1(float value) {
  const bool negative = signbit(value);
  const float magnitude = fminf(fabsf(value), 6.0f);
  uint8_t code;
  // E2M1 finite magnitudes are {0, .5, 1, 1.5, 2, 3, 4, 6}.
  // Inclusive boundaries implement round-to-nearest-even at midpoints.
  if (magnitude <= 0.25f) {
    code = 0;
  } else if (magnitude < 0.75f) {
    code = 1;
  } else if (magnitude <= 1.25f) {
    code = 2;
  } else if (magnitude < 1.75f) {
    code = 3;
  } else if (magnitude <= 2.5f) {
    code = 4;
  } else if (magnitude < 3.5f) {
    code = 5;
  } else if (magnitude <= 5.0f) {
    code = 6;
  } else {
    code = 7;
  }
  return code | (negative ? 0x8 : 0);
}

__device__ __forceinline__ uint32_t pack_e2m1(float2 (&values)[4]) {
  uint32_t packed = 0;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const uint8_t byte = quantize_e2m1(values[i].x) |
                         (quantize_e2m1(values[i].y) << 4);
    packed |= static_cast<uint32_t>(byte) << (8 * i);
  }
  return packed;
}

__device__ void quantize_row(const __nv_bfloat16* input, int columns,
                             float global_scale, uint32_t* output_fp4,
                             uint8_t* output_sf) {
  const int sf_columns = columns / kSfVecSize;
  const int sf_threads = sf_columns * 2;
  const float inverse_global = 1.0f / fmaxf(global_scale, kSfScaleEpsilon);
  for (int column = threadIdx.x; column < sf_threads; column += blockDim.x) {
    const Bf16Vector vector =
        *reinterpret_cast<const Bf16Vector*>(input + static_cast<int64_t>(column) * 8);
    __nv_bfloat162 local = __habs2(vector.values[0]);
#pragma unroll
    for (int i = 1; i < 4; ++i) local = __hmax2(local, __habs2(vector.values[i]));
    local = __hmax2(local, __shfl_xor_sync(0xffffffff, local, 1));
    const float maximum = __bfloat162float(__hmax(local.x, local.y));
    const float sf_value = inverse_global * (maximum * reciprocal_approx(6.0f));
    const __nv_fp8_e4m3 sf(sf_value);
    const float sf_back = static_cast<float>(sf);
    const float scale = maximum != 0.0f ? reciprocal_approx(sf_back * global_scale) : 0.0f;
    float2 values[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      values[i] = __bfloat1622float2(vector.values[i]);
      values[i].x *= scale;
      values[i].y *= scale;
    }
    output_fp4[column] = pack_e2m1(values);
    if ((column & 1) == 0) output_sf[column / 2] = sf.__x;
  }
}

__global__ __launch_bounds__(512, 4) void quantize_routed_kernel(
    int tokens, int top_k, int hidden, const __nv_bfloat16* input,
    const int32_t* expert_ids, const float* expert_scales,
    uint32_t* output_fp4, uint8_t* output_sf) {
  const int routes = tokens * top_k;
  for (int row = blockIdx.x; row < routes; row += gridDim.x) {
    const int token = row / top_k;
    const int expert = expert_ids[row];
    quantize_row(input + static_cast<int64_t>(token) * hidden, hidden,
                 expert_scales[expert],
                 output_fp4 + static_cast<int64_t>(row) * hidden / 8,
                 output_sf + static_cast<int64_t>(row) * hidden / kSfVecSize);
  }
}

__global__ __launch_bounds__(512, 4) void small_batch_setup_kernel(
    int tokens, int top_k, int hidden, int experts, const __nv_bfloat16* input,
    const int32_t* expert_ids, const float* expert_scales,
    int32_t* permuted_idx, int32_t* tile_group, int32_t* tile_limit,
    int32_t* num_tiles, uint32_t* output_fp4, uint8_t* output_sf) {
  __shared__ int32_t counts[kMaxDecodeExperts];
  __shared__ int32_t offsets[kMaxDecodeExperts];
  __shared__ int32_t scatter[kMaxDecodeExperts];
  const int tid = threadIdx.x;
  const int routes = tokens * top_k;
  if (blockIdx.x == 0) {
    for (int e = tid; e < experts; e += blockDim.x) counts[e] = scatter[e] = 0;
    __syncthreads();
    for (int row = tid; row < routes; row += blockDim.x)
      atomicAdd_block(&counts[expert_ids[row]], 1);
    __syncthreads();
    if (tid == 0) {
      int row = 0;
      int tile = 0;
      for (int e = 0; e < experts; ++e) {
        offsets[e] = row;
        if (counts[e] > 0) {
          tile_group[tile] = e;
          tile_limit[tile] = row + counts[e];
          row += kRowTile;
          ++tile;
        }
      }
      num_tiles[0] = tile;
    }
    __syncthreads();
    for (int row = tid; row < routes; row += blockDim.x) {
      const int expert = expert_ids[row];
      permuted_idx[offsets[expert] + atomicAdd_block(&scatter[expert], 1)] = row;
    }
    return;
  }
  const int quant_blocks = gridDim.x - 1;
  for (int row = blockIdx.x - 1; row < routes; row += quant_blocks) {
    const int token = row / top_k;
    quantize_row(input + static_cast<int64_t>(token) * hidden, hidden,
                 expert_scales[expert_ids[row]],
                 output_fp4 + static_cast<int64_t>(row) * hidden / 8,
                 output_sf + static_cast<int64_t>(row) * hidden / kSfVecSize);
  }
}

std::once_flag modules_once;
nvfp4_moe_sm110_fc1_swiglu_n128_Kernel_Module_t fc1_module{};
nvfp4_moe_sm110_fc2_n128_bf16_Kernel_Module_t fc2_module{};

void ensure_modules() {
  std::call_once(modules_once, [] {
    nvfp4_moe_sm110_fc1_swiglu_n128_Kernel_Module_Load(&fc1_module);
    nvfp4_moe_sm110_fc2_n128_bf16_Kernel_Module_Load(&fc2_module);
  });
}

void launch(apxinf_runtime_t runtime, const apxinf_moe_edge_nvfp4_spec_t& spec,
            const apxinf_moe_edge_nvfp4_bindings_t& bindings) {
  if (runtime == nullptr) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT, "null Edge NVFP4 MoE runtime");
  }
  if (runtime->device < 0) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT, "invalid CUDA device");
  }
  apxinf::framework::check_cuda(cudaSetDevice(runtime->device));
  cudaDeviceProp properties{};
  apxinf::framework::check_cuda(cudaGetDeviceProperties(&properties, runtime->device));
  const int sm = properties.major * 10 + properties.minor;
  if (sm != 101) {
    throw Failure(APXINF_STATUS_UNSUPPORTED,
                  "Edge NVFP4 MoE CUDA-12 artifact requires Thor SM101 compatibility encoding, got SM" +
                      std::to_string(sm));
  }
  if (bindings.input == nullptr || bindings.expert_ids == nullptr ||
      bindings.expert_weights == nullptr || bindings.fc1_weight == nullptr ||
      bindings.fc1_scales == nullptr || bindings.fc1_alpha == nullptr ||
      bindings.input_global_scale == nullptr || bindings.fc2_weight == nullptr ||
      bindings.fc2_scales == nullptr || bindings.fc2_alpha == nullptr ||
      bindings.down_input_scale == nullptr || bindings.workspace == nullptr ||
      bindings.output == nullptr) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                  "missing Edge NVFP4 MoE binding");
  }
  const WorkspaceLayout layout = workspace_layout(spec);
  if (bindings.workspace_bytes < layout.total) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                  "Edge NVFP4 MoE workspace is too small");
  }
  ensure_modules();

  auto* input_fp4 = at<uint32_t>(bindings.workspace, layout.input_fp4);
  auto* input_sf = at<uint8_t>(bindings.workspace, layout.input_sf);
  auto* tile_group = at<int32_t>(bindings.workspace, layout.tile_group);
  auto* tile_limit = at<int32_t>(bindings.workspace, layout.tile_limit);
  auto* permutation = at<int32_t>(bindings.workspace, layout.permuted_to_expanded);
  auto* num_tiles = at<int32_t>(bindings.workspace, layout.num_tiles);
  auto* fc1_fp4 = at<void>(bindings.workspace, layout.fc1_fp4);
  auto* fc1_sf = at<void>(bindings.workspace, layout.fc1_sf);
  const int routes = static_cast<int>(spec.tokens * spec.top_k);
  const auto stream = reinterpret_cast<cudaStream_t>(bindings.stream);

  if (spec.tokens <= 4) {
    const int quant_blocks = std::min<int>(routes, properties.multiProcessorCount * 4);
    small_batch_setup_kernel<<<quant_blocks + 1, kThreads, 0, stream>>>(
        spec.tokens, spec.top_k, spec.hidden, spec.experts,
        static_cast<const __nv_bfloat16*>(bindings.input), bindings.expert_ids,
        bindings.input_global_scale, permutation, tile_group, tile_limit,
        num_tiles, input_fp4, input_sf);
  } else {
    build_layout_kernel<<<1, 256, 3 * spec.experts * sizeof(int32_t), stream>>>(
        bindings.expert_ids, permutation, tile_group, tile_limit, num_tiles,
        routes, spec.experts, layout.permuted_m);
    const int column_threads = spec.hidden / 8;
    const int blocks_per_sm = std::max(1, kMaxResidentThreadsPerSm /
                                             std::min(column_threads, kThreads));
    const int grid = std::min(routes, properties.multiProcessorCount * blocks_per_sm);
    quantize_routed_kernel<<<grid, std::min(column_threads, kThreads), 0, stream>>>(
        spec.tokens, spec.top_k, spec.hidden,
        static_cast<const __nv_bfloat16*>(bindings.input), bindings.expert_ids,
        bindings.input_global_scale, input_fp4, input_sf);
  }
  apxinf::framework::check_cuda(cudaGetLastError());

  const int64_t m = layout.permuted_m;
  const int64_t n1 = 2 * static_cast<int64_t>(spec.intermediate);
  const int32_t pdl = 0;
  int result = cute_dsl_nvfp4_moe_sm110_fc1_swiglu_n128_wrapper(
      &fc1_module, input_fp4, const_cast<void*>(bindings.fc1_weight), input_sf,
      const_cast<void*>(bindings.fc1_scales), fc1_fp4, fc1_sf, bindings.output,
      static_cast<int64_t>(spec.tokens) * spec.hidden, 1,
      const_cast<float*>(bindings.fc1_alpha),
      const_cast<float*>(bindings.input_global_scale),
      const_cast<float*>(bindings.down_input_scale), tile_group, tile_limit,
      permutation, num_tiles, routes, m, n1, spec.hidden, spec.experts,
      properties.multiProcessorCount, pdl, stream);
  if (result != 0) {
    throw Failure(APXINF_STATUS_PROVIDER_ERROR,
                  "Edge NVFP4 MoE FC1 returned " + std::to_string(result));
  }
  apxinf::framework::check_cuda(cudaGetLastError());

  result = cute_dsl_nvfp4_moe_sm110_fc2_n128_bf16_wrapper(
      &fc2_module, fc1_fp4, const_cast<void*>(bindings.fc2_weight), fc1_sf,
      const_cast<void*>(bindings.fc2_scales), bindings.output,
      const_cast<float*>(bindings.fc2_alpha),
      const_cast<float*>(bindings.down_input_scale), tile_group, tile_limit,
      permutation, num_tiles, const_cast<float*>(bindings.expert_weights), m,
      spec.hidden, spec.intermediate, spec.experts, spec.tokens, spec.top_k,
      properties.multiProcessorCount, pdl, stream);
  if (result != 0) {
    throw Failure(APXINF_STATUS_PROVIDER_ERROR,
                  "Edge NVFP4 MoE FC2 returned " + std::to_string(result));
  }
  apxinf::framework::check_cuda(cudaGetLastError());
}

}  // namespace

extern "C" apxinf_status_t apxinf_moe_edge_nvfp4_workspace_size(
    const apxinf_moe_edge_nvfp4_spec_t* spec, size_t* workspace_bytes) {
  return apxinf::framework::abi_boundary([&] {
    validate_spec(spec);
    if (workspace_bytes == nullptr) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                    "null Edge NVFP4 MoE workspace-size output");
    }
    *workspace_bytes = workspace_layout(*spec).total;
  });
}

extern "C" apxinf_status_t apxinf_moe_edge_nvfp4(
    apxinf_runtime_t runtime, const apxinf_moe_edge_nvfp4_spec_t* spec,
    const apxinf_moe_edge_nvfp4_bindings_t* bindings) {
  return apxinf::framework::abi_boundary([&] {
    validate_spec(spec);
    if (bindings == nullptr) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                    "null Edge NVFP4 MoE bindings");
    }
    launch(runtime, *spec, *bindings);
  });
}
