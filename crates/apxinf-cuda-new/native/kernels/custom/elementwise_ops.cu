// Copyright 2026 ApxInf contributors.
//
// Out-of-place elementwise and activation operators for the portable
// `Backend` trait. See `elementwise_ops.h` for the contract; the arithmetic
// is deliberately identical to the legacy `apxinf-cuda` kernels so a model
// migrated between backends produces identical bytes.

#include "elementwise_ops.h"

#include <cuda_bf16.h>

#include <cstdint>

namespace apxinf::cuda_new::elementwise_ops {
namespace {

constexpr int kThreads = 256;

__device__ __forceinline__ float gelu_tanh(float x) {
  // sqrt(2/pi) ~= 0.7978845608028654; matches the legacy kernel and
  // PyTorch's `gelu_pytorch_tanh`.
  const float kBeta = 0.7978845608028654f;
  const float kAlpha = 0.044715f;
  const float inner = kBeta * (x + kAlpha * x * x * x);
  return 0.5f * x * (1.0f + tanhf(inner));
}

__device__ __forceinline__ float apply_activation(float value, int activation) {
  if (activation == 1) return gelu_tanh(value);
  if (activation == 2) return value / (1.0f + expf(-value));
  return value;
}

__global__ void activation_bf16_kernel(const __nv_bfloat16* __restrict__ input,
                                       __nv_bfloat16* __restrict__ output,
                                       long long count, int activation) {
  const long long index = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  if (index >= count) return;
  const float value = __bfloat162float(input[index]);
  output[index] = __float2bfloat16(apply_activation(value, activation));
}

__global__ void mul_bf16_kernel(const __nv_bfloat16* __restrict__ a,
                                const __nv_bfloat16* __restrict__ b,
                                __nv_bfloat16* __restrict__ output,
                                long long count) {
  const long long index = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  if (index >= count) return;
  output[index] = __float2bfloat16(__bfloat162float(a[index]) *
                                   __bfloat162float(b[index]));
}

__global__ void add_bf16_kernel(const __nv_bfloat16* __restrict__ a,
                                const __nv_bfloat16* __restrict__ b,
                                __nv_bfloat16* __restrict__ output,
                                long long count) {
  const long long index = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  if (index >= count) return;
  output[index] = __float2bfloat16(__bfloat162float(a[index]) +
                                   __bfloat162float(b[index]));
}

__global__ void scale_bf16_kernel(const __nv_bfloat16* __restrict__ input,
                                  __nv_bfloat16* __restrict__ output,
                                  long long count, float factor) {
  const long long index = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  if (index >= count) return;
  output[index] = __float2bfloat16(__bfloat162float(input[index]) * factor);
}

// One thread per (row, col); the y grid dimension carries the row so a long
// column axis still maps to whole warps.
__global__ void add_bias_bf16_kernel(const __nv_bfloat16* __restrict__ input,
                                     const __nv_bfloat16* __restrict__ bias,
                                     __nv_bfloat16* __restrict__ output,
                                     long long cols) {
  const long long col = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  const long long row = blockIdx.y;
  if (col >= cols) return;
  const long long index = row * cols + col;
  output[index] = __float2bfloat16(__bfloat162float(input[index]) +
                                   __bfloat162float(bias[col]));
}

__global__ void gather_rows_bf16_kernel(const __nv_bfloat16* __restrict__ input,
                                        const unsigned int* __restrict__ indices,
                                        __nv_bfloat16* __restrict__ output,
                                        long long rows, long long cols) {
  const long long count = rows * cols;
  long long index = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  const long long stride = (long long)blockDim.x * gridDim.x;
  for (; index < count; index += stride) {
    const long long row = index / cols;
    const long long col = index % cols;
    output[index] = input[(long long)indices[row] * cols + col];
  }
}

__global__ void bias_position_f32_bf16_kernel(
    const float* __restrict__ projection, const float* __restrict__ bias,
    const float* __restrict__ position, __nv_bfloat16* __restrict__ output,
    long long count, int cols, int tokens_per_view) {
  long long index = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  const long long stride = (long long)blockDim.x * gridDim.x;
  for (; index < count; index += stride) {
    const int col = static_cast<int>(index % cols);
    const int token = static_cast<int>((index / cols) % tokens_per_view);
    float value = projection[index] + position[(long long)token * cols + col];
    if (bias != nullptr) value += bias[col];
    output[index] = __float2bfloat16(value);
  }
}

long long block_count(long long work, long long per_thread) {
  const long long threads = work / per_thread + (work % per_thread != 0);
  return (threads + kThreads - 1) / kThreads;
}

}  // namespace

int activation_bf16(const void* input, void* output, long long count,
                    int activation, cudaStream_t stream) {
  if (input == nullptr || output == nullptr || count <= 0 || activation < 0 ||
      activation > 2) {
    return -1;
  }
  activation_bf16_kernel<<<static_cast<int>(block_count(count, 1)), kThreads, 0,
                           stream>>>(
      static_cast<const __nv_bfloat16*>(input),
      static_cast<__nv_bfloat16*>(output), count, activation);
  return cudaGetLastError() == cudaSuccess ? 0 : -2;
}

int mul_bf16(const void* a, const void* b, void* output, long long count,
             cudaStream_t stream) {
  if (a == nullptr || b == nullptr || output == nullptr || count <= 0) {
    return -1;
  }
  mul_bf16_kernel<<<static_cast<int>(block_count(count, 1)), kThreads, 0,
                    stream>>>(static_cast<const __nv_bfloat16*>(a),
                              static_cast<const __nv_bfloat16*>(b),
                              static_cast<__nv_bfloat16*>(output), count);
  return cudaGetLastError() == cudaSuccess ? 0 : -2;
}

int add_bf16(const void* a, const void* b, void* output, long long count,
             cudaStream_t stream) {
  if (a == nullptr || b == nullptr || output == nullptr || count <= 0) {
    return -1;
  }
  add_bf16_kernel<<<static_cast<int>(block_count(count, 1)), kThreads, 0,
                    stream>>>(static_cast<const __nv_bfloat16*>(a),
                              static_cast<const __nv_bfloat16*>(b),
                              static_cast<__nv_bfloat16*>(output), count);
  return cudaGetLastError() == cudaSuccess ? 0 : -2;
}

int scale_bf16(const void* input, void* output, long long count, float factor,
               cudaStream_t stream) {
  if (input == nullptr || output == nullptr || count <= 0) return -1;
  scale_bf16_kernel<<<static_cast<int>(block_count(count, 1)), kThreads, 0,
                      stream>>>(static_cast<const __nv_bfloat16*>(input),
                                static_cast<__nv_bfloat16*>(output), count,
                                factor);
  return cudaGetLastError() == cudaSuccess ? 0 : -2;
}

int add_bias_bf16(const void* input, const void* bias, void* output,
                  long long rows, long long cols, cudaStream_t stream) {
  if (input == nullptr || bias == nullptr || output == nullptr || rows <= 0 ||
      cols <= 0) {
    return -1;
  }
  const long long blocks_x = (cols + kThreads - 1) / kThreads;
  if (blocks_x > 65535 || rows > 65535) return -1;
  const dim3 grid(static_cast<unsigned>(blocks_x), static_cast<unsigned>(rows));
  add_bias_bf16_kernel<<<grid, kThreads, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(input),
      static_cast<const __nv_bfloat16*>(bias),
      static_cast<__nv_bfloat16*>(output), cols);
  return cudaGetLastError() == cudaSuccess ? 0 : -2;
}

int gather_rows_bf16(const void* input, const void* indices, void* output,
                     long long rows, long long cols, cudaStream_t stream) {
  if (input == nullptr || indices == nullptr || output == nullptr ||
      rows <= 0 || cols <= 0) {
    return -1;
  }
  gather_rows_bf16_kernel<<<static_cast<int>(block_count(rows * cols, 1)),
                            kThreads, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(input),
      static_cast<const unsigned int*>(indices),
      static_cast<__nv_bfloat16*>(output), rows, cols);
  return cudaGetLastError() == cudaSuccess ? 0 : -2;
}

int bias_position_f32_bf16(const void* projection, const void* bias,
                           const void* position, void* output, long long count,
                           int cols, int tokens_per_view, cudaStream_t stream) {
  if (projection == nullptr || position == nullptr || output == nullptr ||
      count <= 0 || cols <= 0 || tokens_per_view <= 0) {
    return -1;
  }
  bias_position_f32_bf16_kernel<<<static_cast<int>(block_count(count, 1)),
                                  kThreads, 0, stream>>>(
      static_cast<const float*>(projection), static_cast<const float*>(bias),
      static_cast<const float*>(position),
      static_cast<__nv_bfloat16*>(output), count, cols, tokens_per_view);
  return cudaGetLastError() == cudaSuccess ? 0 : -2;
}

}  // namespace apxinf::cuda_new::elementwise_ops
