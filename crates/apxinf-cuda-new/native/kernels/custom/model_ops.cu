// Copyright 2026 ApxInf contributors.

#include "model_ops.h"

#include <cuda_bf16.h>

namespace apxinf::cuda_new::model_ops {
namespace {

__global__ void embedding_gather_kernel(const __nv_bfloat16* __restrict__ table,
                                        const int32_t* __restrict__ ids,
                                        __nv_bfloat16* __restrict__ output,
                                        int tokens, int hidden, int vocab) {
  const long long index = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  const long long total = (long long)tokens * hidden;
  if (index >= total) return;
  const int token = static_cast<int>(index / hidden);
  const int element = static_cast<int>(index % hidden);
  int id = ids[token];
  // Clamp rather than fault: an out-of-range id is a caller bug, but reading
  // arbitrary device memory would turn it into a silent wrong answer.
  if (id < 0 || id >= vocab) id = 0;
  output[index] = table[(long long)id * hidden + element];
}

__global__ void argmax_kernel(const __nv_bfloat16* __restrict__ logits,
                              int32_t* __restrict__ index, int count) {
  __shared__ float best_value[32];
  __shared__ int best_index[32];

  float local_value = -INFINITY;
  int local_index = 0;
  for (int i = threadIdx.x; i < count; i += blockDim.x) {
    const float value = __bfloat162float(logits[i]);
    if (value > local_value) {
      local_value = value;
      local_index = i;
    }
  }
  for (int offset = 16; offset > 0; offset >>= 1) {
    const float other_value =
        __shfl_down_sync(0xFFFFFFFFu, local_value, offset);
    const int other_index = __shfl_down_sync(0xFFFFFFFFu, local_index, offset);
    if (other_value > local_value) {
      local_value = other_value;
      local_index = other_index;
    }
  }
  const int warp = threadIdx.x >> 5;
  if ((threadIdx.x & 31) == 0) {
    best_value[warp] = local_value;
    best_index[warp] = local_index;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    const int warps = (blockDim.x + 31) / 32;
    float value = best_value[0];
    int chosen = best_index[0];
    for (int i = 1; i < warps; ++i) {
      if (best_value[i] > value) {
        value = best_value[i];
        chosen = best_index[i];
      }
    }
    *index = chosen;
  }
}

__global__ void cast_f32_to_bf16_kernel(
    const float* __restrict__ input, __nv_bfloat16* __restrict__ output,
    long long count) {
  long long index = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
  const long long stride = static_cast<long long>(blockDim.x) * gridDim.x;
  for (; index < count; index += stride) {
    output[index] = __float2bfloat16(input[index]);
  }
}

__global__ void gelu_exact_bf16_kernel(
    const __nv_bfloat16* __restrict__ input,
    __nv_bfloat16* __restrict__ output, long long count) {
  long long index = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
  const long long stride = static_cast<long long>(blockDim.x) * gridDim.x;
  for (; index < count; index += stride) {
    const float value = __bfloat162float(input[index]);
    output[index] = __float2bfloat16(
        value * 0.5f * (1.0f + erff(value * 0.70710678f)));
  }
}

__global__ void replace_rows_bf16_kernel(
    const __nv_bfloat16* __restrict__ base,
    const __nv_bfloat16* __restrict__ replacement,
    const int32_t* __restrict__ row_map,
    __nv_bfloat16* __restrict__ output, int rows, int columns,
    int replacement_rows) {
  const long long index =
      blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
  const long long count = static_cast<long long>(rows) * columns;
  const long long stride = static_cast<long long>(blockDim.x) * gridDim.x;
  for (long long current = index; current < count; current += stride) {
    const int row = static_cast<int>(current / columns);
    const int column = static_cast<int>(current % columns);
    const int replacement_row = row_map[row];
    output[current] = replacement_row >= 0 && replacement_row < replacement_rows
        ? replacement[static_cast<long long>(replacement_row) * columns + column]
        : base[current];
  }
}

}  // namespace

int embedding_gather(const void* table, const int32_t* ids, void* output,
                     int tokens, int hidden, int vocab, cudaStream_t stream) {
  if (tokens <= 0 || hidden <= 0 || vocab <= 0) return -1;
  const long long total = (long long)tokens * hidden;
  const int threads = 256;
  const long long blocks = (total + threads - 1) / threads;
  embedding_gather_kernel<<<static_cast<int>(blocks), threads, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(table), ids,
      static_cast<__nv_bfloat16*>(output), tokens, hidden, vocab);
  return cudaGetLastError() == cudaSuccess ? 0 : -2;
}

int argmax_bf16(const void* logits, int32_t* index, int count,
                cudaStream_t stream) {
  if (count <= 0) return -1;
  argmax_kernel<<<1, 1024, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(logits), index, count);
  return cudaGetLastError() == cudaSuccess ? 0 : -2;
}

int cast_f32_to_bf16(const void* input, void* output, long long count,
                     cudaStream_t stream) {
  if (input == nullptr || output == nullptr || count <= 0) return -1;
  const int threads = 256;
  const long long needed = (count + threads - 1) / threads;
  const int blocks = static_cast<int>(needed < 4096 ? needed : 4096);
  cast_f32_to_bf16_kernel<<<blocks, threads, 0, stream>>>(
      static_cast<const float*>(input),
      static_cast<__nv_bfloat16*>(output), count);
  return cudaGetLastError() == cudaSuccess ? 0 : -2;
}

int gelu_exact_bf16(const void* input, void* output, long long count,
                    cudaStream_t stream) {
  if (input == nullptr || output == nullptr || count <= 0) return -1;
  const int threads = 256;
  const long long needed = (count + threads - 1) / threads;
  const int blocks = static_cast<int>(needed < 1024 ? needed : 1024);
  gelu_exact_bf16_kernel<<<blocks, threads, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(input),
      static_cast<__nv_bfloat16*>(output), count);
  return cudaGetLastError() == cudaSuccess ? 0 : -2;
}

int replace_rows_bf16(const void* base, const void* replacement,
                      const int32_t* row_map, void* output, int rows,
                      int columns, int replacement_rows,
                      cudaStream_t stream) {
  if (base == nullptr || replacement == nullptr || row_map == nullptr ||
      output == nullptr || rows <= 0 || columns <= 0 ||
      replacement_rows <= 0) {
    return -1;
  }
  const long long count = static_cast<long long>(rows) * columns;
  const int threads = 256;
  const long long needed = (count + threads - 1) / threads;
  const int blocks = static_cast<int>(needed < 65535 ? needed : 65535);
  replace_rows_bf16_kernel<<<blocks, threads, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(base),
      static_cast<const __nv_bfloat16*>(replacement), row_map,
      static_cast<__nv_bfloat16*>(output), rows, columns, replacement_rows);
  return cudaGetLastError() == cudaSuccess ? 0 : -2;
}

}  // namespace apxinf::cuda_new::model_ops
