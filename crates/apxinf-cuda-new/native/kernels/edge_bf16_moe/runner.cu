/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Adapted from TensorRT-Edge-LLM v0.11.0 f16MoeSupportKernels and
 * cuteDslF16MoeRunner. ApxInf keeps activations and weights in BF16, removes
 * plugin dependencies, and specializes the public ABI to one-token decode.
 */

#include "../../include/apxinf_cuda/moe.h"
#include "../../framework/runtime_internal.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <mutex>

using apxinf::framework::Failure;

#define CUTE_DSL_CUDA_ERROR_CHECK(error)                                  \
  do {                                                                     \
    cudaError_t status_ = (error);                                          \
    if (status_ != cudaSuccess) {                                           \
      throw Failure(APXINF_STATUS_CUDA_ERROR, cudaGetErrorString(status_)); \
    }                                                                      \
  } while (0)
#include "generated/edge_bf16_moe_sm110_grouped.h"

namespace {

constexpr int32_t kThreads = 256;
constexpr int32_t kSwiGluChunk = 64;
constexpr int32_t kTensorMapsPerBlock = 3;
constexpr int32_t kBytesPerTensorMap = 128;
constexpr int32_t kMaxPersistentBlocks = 256;
constexpr size_t kAlignment = 256;

size_t align_up(size_t value) {
  if (value > std::numeric_limits<size_t>::max() - (kAlignment - 1)) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT, "BF16 MoE workspace overflow");
  }
  return ((value + kAlignment - 1) / kAlignment) * kAlignment;
}

size_t checked_mul(size_t lhs, size_t rhs) {
  if (rhs != 0 && lhs > std::numeric_limits<size_t>::max() / rhs) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT, "BF16 MoE workspace overflow");
  }
  return lhs * rhs;
}

struct WorkspaceLayout {
  size_t expert_counts{};
  size_t expert_offsets{};
  size_t expert_write_offsets{};
  size_t sorted_to_expanded{};
  size_t expanded_to_sorted{};
  size_t fc1_shapes{};
  size_t fc1_strides{};
  size_t fc1_addresses{};
  size_t fc2_shapes{};
  size_t fc2_strides{};
  size_t fc2_addresses{};
  size_t tensor_maps{};
  size_t gathered_input{};
  size_t raw_fc1{};
  size_t activated_fc1{};
  size_t routed_fc2{};
  size_t total{};
};

WorkspaceLayout workspace_layout(
    const apxinf_moe_edge_bf16_decode_spec_t& spec) {
  const size_t routes = spec.top_k;
  const size_t experts = spec.experts;
  const size_t bf16 = sizeof(__nv_bfloat16);
  WorkspaceLayout layout{};
  size_t cursor = 0;
  auto place = [&](size_t& offset, size_t bytes) {
    offset = cursor;
    cursor = align_up(cursor + bytes);
  };
  place(layout.expert_counts, checked_mul(experts, sizeof(int32_t)));
  place(layout.expert_offsets, checked_mul(experts + 1, sizeof(int32_t)));
  place(layout.expert_write_offsets, checked_mul(experts, sizeof(int32_t)));
  place(layout.sorted_to_expanded, checked_mul(routes, sizeof(int32_t)));
  place(layout.expanded_to_sorted, checked_mul(routes, sizeof(int32_t)));
  const size_t shapes = checked_mul(experts * 4, sizeof(int32_t));
  const size_t strides = checked_mul(experts * 6, sizeof(int32_t));
  const size_t addresses = checked_mul(experts * 3, sizeof(int64_t));
  place(layout.fc1_shapes, shapes);
  place(layout.fc1_strides, strides);
  place(layout.fc1_addresses, addresses);
  place(layout.fc2_shapes, shapes);
  place(layout.fc2_strides, strides);
  place(layout.fc2_addresses, addresses);
  place(layout.tensor_maps,
        static_cast<size_t>(kMaxPersistentBlocks) * kTensorMapsPerBlock *
            kBytesPerTensorMap);
  place(layout.gathered_input,
        checked_mul(routes * spec.hidden, bf16));
  place(layout.raw_fc1,
        checked_mul(routes * 2 * spec.intermediate, bf16));
  place(layout.activated_fc1,
        checked_mul(routes * spec.intermediate, bf16));
  place(layout.routed_fc2,
        checked_mul(routes * spec.hidden, bf16));
  layout.total = align_up(cursor);
  return layout;
}

template <typename T>
T* at(void* base, size_t offset) {
  return reinterpret_cast<T*>(static_cast<std::byte*>(base) + offset);
}

void validate_spec(const apxinf_moe_edge_bf16_decode_spec_t* spec) {
  if (spec == nullptr ||
      spec->version != APXINF_MOE_EDGE_BF16_DECODE_SPEC_VERSION ||
      spec->dtype != APXINF_DTYPE_BF16 || spec->experts != 256 ||
      spec->top_k != 8 || spec->hidden == 0 || spec->hidden % 128 != 0 ||
      spec->intermediate == 0 || spec->intermediate % 64 != 0) {
    throw Failure(APXINF_STATUS_UNSUPPORTED,
                  "Edge BF16 decode MoE requires BF16, 256 experts, top-8, "
                  "hidden multiple of 128, and intermediate multiple of 64");
  }
}

struct GemmMetadata {
  int32_t* shapes;
  int32_t* strides;
  int64_t* addresses;
};

__device__ void populate_metadata(GemmMetadata metadata, int32_t expert,
                                  int32_t row_offset, int32_t rows,
                                  int32_t n, int32_t k,
                                  const __nv_bfloat16* input,
                                  const __nv_bfloat16* weights,
                                  __nv_bfloat16* output) {
  const int shape = expert * 4;
  metadata.shapes[shape] = rows;
  metadata.shapes[shape + 1] = n;
  metadata.shapes[shape + 2] = k;
  metadata.shapes[shape + 3] = 1;
  const int stride = expert * 6;
  metadata.strides[stride] = k;
  metadata.strides[stride + 1] = 1;
  metadata.strides[stride + 2] = k;
  metadata.strides[stride + 3] = 1;
  metadata.strides[stride + 4] = n;
  metadata.strides[stride + 5] = 1;
  const int address = expert * 3;
  metadata.addresses[address] = reinterpret_cast<int64_t>(
      input + static_cast<int64_t>(row_offset) * k);
  metadata.addresses[address + 1] = reinterpret_cast<int64_t>(
      weights + static_cast<int64_t>(expert) * n * k);
  metadata.addresses[address + 2] = reinterpret_cast<int64_t>(
      output + static_cast<int64_t>(row_offset) * n);
}

__global__ void setup_kernel(
    const int32_t* expert_ids, int32_t experts, int32_t top_k,
    int32_t hidden, int32_t intermediate, int32_t* expert_counts,
    int32_t* expert_offsets, int32_t* expert_write_offsets,
    int32_t* sorted_to_expanded, int32_t* expanded_to_sorted,
    GemmMetadata fc1, GemmMetadata fc2, const __nv_bfloat16* input,
    const __nv_bfloat16* fc1_weights, const __nv_bfloat16* fc2_weights,
    __nv_bfloat16* raw_fc1, __nv_bfloat16* activated_fc1,
    __nv_bfloat16* routed_fc2) {
  __shared__ int32_t counts[kThreads];
  __shared__ int32_t writes[kThreads];
  const int thread = threadIdx.x;
  if (thread < experts) counts[thread] = 0;
  __syncthreads();
  if (thread < top_k) {
    const int expert = expert_ids[thread];
    if (expert >= 0 && expert < experts) atomicAdd(&counts[expert], 1);
  }
  __syncthreads();
  if (thread == 0) {
    int prefix = 0;
    expert_offsets[0] = 0;
    for (int expert = 0; expert < experts; ++expert) {
      prefix += counts[expert];
      expert_offsets[expert + 1] = prefix;
    }
  }
  __syncthreads();
  if (thread < experts) {
    const int row_offset = expert_offsets[thread];
    const int rows = counts[thread];
    expert_counts[thread] = rows;
    writes[thread] = row_offset;
    populate_metadata(fc1, thread, row_offset, rows, 2 * intermediate,
                      hidden, input, fc1_weights, raw_fc1);
    populate_metadata(fc2, thread, row_offset, rows, hidden, intermediate,
                      activated_fc1, fc2_weights, routed_fc2);
  }
  __syncthreads();
  if (thread < top_k) {
    const int expert = expert_ids[thread];
    if (expert >= 0 && expert < experts) {
      const int sorted = atomicAdd(&writes[expert], 1);
      sorted_to_expanded[sorted] = thread;
      expanded_to_sorted[thread] = sorted;
    }
  }
  __syncthreads();
  if (thread < experts) expert_write_offsets[thread] = writes[thread];
}

struct Bf16Vector {
  __nv_bfloat162 values[4];
};

__global__ void gather_kernel(const Bf16Vector* input, Bf16Vector* gathered,
                              int32_t routes, int32_t vectors_per_row) {
  const int64_t element = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                          threadIdx.x;
  const int64_t total = static_cast<int64_t>(routes) * vectors_per_row;
  if (element < total) gathered[element] = input[element % vectors_per_row];
}

__global__ void swiglu_kernel(const __nv_bfloat16* raw,
                              __nv_bfloat16* activated, int32_t routes,
                              int32_t intermediate) {
  const int64_t element = static_cast<int64_t>(blockIdx.x) * blockDim.x +
                          threadIdx.x;
  const int64_t total = static_cast<int64_t>(routes) * intermediate;
  if (element >= total) return;
  const int row = element / intermediate;
  const int column = element % intermediate;
  const int chunk = column / kSwiGluChunk;
  const int within = column % kSwiGluChunk;
  const int up_column = chunk * (2 * kSwiGluChunk) + within;
  const int gate_column = up_column + kSwiGluChunk;
  const int width = 2 * intermediate;
  const float up = __bfloat162float(
      raw[static_cast<int64_t>(row) * width + up_column]);
  const float gate = __bfloat162float(
      raw[static_cast<int64_t>(row) * width + gate_column]);
  activated[element] = __float2bfloat16_rn(
      up * gate / (1.0f + expf(-gate)));
}

__global__ void reduce_kernel(const __nv_bfloat16* routed,
                              const int32_t* expanded_to_sorted,
                              const float* expert_weights,
                              __nv_bfloat16* output, int32_t top_k,
                              int32_t hidden) {
  const int feature = blockIdx.x * blockDim.x + threadIdx.x;
  if (feature >= hidden) return;
  float value = 0.0f;
  for (int slot = 0; slot < top_k; ++slot) {
    value += expert_weights[slot] *
             __bfloat162float(
                 routed[static_cast<int64_t>(expanded_to_sorted[slot]) *
                            hidden +
                        feature]);
  }
  output[feature] = __float2bfloat16_rn(value);
}

std::once_flag module_once;
edge_bf16_moe_sm110_grouped_Kernel_Module_t module{};
int32_t active_blocks = 0;

void ensure_module() {
  std::call_once(module_once, [] {
    edge_bf16_moe_sm110_grouped_Kernel_Module_Load(&module);
    int device = 0;
    cudaDeviceProp properties{};
    CUTE_DSL_CUDA_ERROR_CHECK(cudaGetDevice(&device));
    CUTE_DSL_CUDA_ERROR_CHECK(cudaGetDeviceProperties(&properties, device));
    active_blocks =
        std::min(kMaxPersistentBlocks, properties.multiProcessorCount);
    if (active_blocks <= 0) {
      throw Failure(APXINF_STATUS_UNSUPPORTED,
                    "Edge BF16 grouped GEMM found no resident SMs");
    }
  });
}

void run(const apxinf_moe_edge_bf16_decode_spec_t& spec,
         const apxinf_moe_edge_bf16_decode_bindings_t& bindings) {
  const WorkspaceLayout layout = workspace_layout(spec);
  if (bindings.workspace_bytes < layout.total) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                  "Edge BF16 decode MoE workspace is too small");
  }
  auto* workspace = bindings.workspace;
  GemmMetadata fc1{at<int32_t>(workspace, layout.fc1_shapes),
                   at<int32_t>(workspace, layout.fc1_strides),
                   at<int64_t>(workspace, layout.fc1_addresses)};
  GemmMetadata fc2{at<int32_t>(workspace, layout.fc2_shapes),
                   at<int32_t>(workspace, layout.fc2_strides),
                   at<int64_t>(workspace, layout.fc2_addresses)};
  auto* gathered = at<__nv_bfloat16>(workspace, layout.gathered_input);
  auto* raw_fc1 = at<__nv_bfloat16>(workspace, layout.raw_fc1);
  auto* activated = at<__nv_bfloat16>(workspace, layout.activated_fc1);
  auto* routed = at<__nv_bfloat16>(workspace, layout.routed_fc2);
  auto stream = static_cast<cudaStream_t>(bindings.stream);

  setup_kernel<<<1, kThreads, 0, stream>>>(
      bindings.expert_ids, spec.experts, spec.top_k, spec.hidden,
      spec.intermediate, at<int32_t>(workspace, layout.expert_counts),
      at<int32_t>(workspace, layout.expert_offsets),
      at<int32_t>(workspace, layout.expert_write_offsets),
      at<int32_t>(workspace, layout.sorted_to_expanded),
      at<int32_t>(workspace, layout.expanded_to_sorted), fc1, fc2, gathered,
      static_cast<const __nv_bfloat16*>(bindings.fc1_weight),
      static_cast<const __nv_bfloat16*>(bindings.fc2_weight), raw_fc1,
      activated, routed);
  CUTE_DSL_CUDA_ERROR_CHECK(cudaGetLastError());
  const int vectors = spec.hidden / 8;
  const int gather_elements = spec.top_k * vectors;
  gather_kernel<<<(gather_elements + kThreads - 1) / kThreads, kThreads, 0,
                  stream>>>(
      static_cast<const Bf16Vector*>(bindings.input),
      reinterpret_cast<Bf16Vector*>(gathered), spec.top_k, vectors);
  CUTE_DSL_CUDA_ERROR_CHECK(cudaGetLastError());

  ensure_module();
  auto launch_grouped = [&](GemmMetadata metadata, void* input, void* weights,
                            void* output, int n, int k) {
    const int result = cute_dsl_edge_bf16_moe_sm110_grouped_wrapper(
        &module, input, weights, output, metadata.shapes, metadata.strides,
        metadata.addresses, at<void>(workspace, layout.tensor_maps),
        spec.top_k, n, k, spec.experts, active_blocks, stream);
    if (result != 0) {
      throw Failure(APXINF_STATUS_CUDA_ERROR,
                    "Edge BF16 grouped GEMM wrapper failed");
    }
    CUTE_DSL_CUDA_ERROR_CHECK(cudaGetLastError());
  };
  launch_grouped(fc1, gathered, const_cast<void*>(bindings.fc1_weight),
                 raw_fc1, 2 * spec.intermediate, spec.hidden);
  const int activation_elements = spec.top_k * spec.intermediate;
  swiglu_kernel<<<(activation_elements + kThreads - 1) / kThreads, kThreads,
                  0, stream>>>(raw_fc1, activated, spec.top_k,
                              spec.intermediate);
  CUTE_DSL_CUDA_ERROR_CHECK(cudaGetLastError());
  launch_grouped(fc2, activated, const_cast<void*>(bindings.fc2_weight),
                 routed, spec.hidden, spec.intermediate);
  reduce_kernel<<<(spec.hidden + kThreads - 1) / kThreads, kThreads, 0,
                  stream>>>(
      routed, at<int32_t>(workspace, layout.expanded_to_sorted),
      bindings.expert_weights,
      static_cast<__nv_bfloat16*>(bindings.output), spec.top_k, spec.hidden);
  CUTE_DSL_CUDA_ERROR_CHECK(cudaGetLastError());
}

}  // namespace

extern "C" apxinf_status_t apxinf_moe_edge_bf16_decode_workspace_size(
    const apxinf_moe_edge_bf16_decode_spec_t* spec, size_t* bytes) {
  return apxinf::framework::abi_boundary([&] {
    validate_spec(spec);
    if (bytes == nullptr) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                    "null Edge BF16 workspace output");
    }
    ensure_module();
    *bytes = workspace_layout(*spec).total;
  });
}

extern "C" apxinf_status_t apxinf_moe_edge_bf16_decode(
    apxinf_runtime_t runtime, const apxinf_moe_edge_bf16_decode_spec_t* spec,
    const apxinf_moe_edge_bf16_decode_bindings_t* bindings) {
  return apxinf::framework::abi_boundary([&] {
    validate_spec(spec);
    if (runtime == nullptr || bindings == nullptr || bindings->input == nullptr ||
        bindings->expert_ids == nullptr || bindings->expert_weights == nullptr ||
        bindings->fc1_weight == nullptr || bindings->fc2_weight == nullptr ||
        bindings->workspace == nullptr || bindings->output == nullptr) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                    "missing Edge BF16 decode MoE binding");
    }
    apxinf::framework::check_cuda(cudaSetDevice(runtime->device));
    run(*spec, *bindings);
  });
}

#undef CUTE_DSL_CUDA_ERROR_CHECK
