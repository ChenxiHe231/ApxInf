// Copyright 2026 ApxInf contributors.
//
// Elementwise and reduction operators the Qwen3.8 MLP needs, kept alongside
// the NVFP4 GEMM they feed. Each has exactly one implementation and no
// persisted selection, so per `doc/adding-new-kernels.md` section 6 they carry
// no recipe: there is nothing to tune between.

#include "mlp_ops.h"

#include <cuda_bf16.h>

#include <cstdint>

namespace apxinf::cuda::mlp_ops {
namespace {

constexpr int kWarpSize = 32;

__device__ __forceinline__ float warp_sum(float value) {
  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xFFFFFFFFu, value, offset);
  }
  return value;
}

// One block per row. The reduction is in f32 regardless of the BF16 storage:
// summing 5120 squares in BF16 loses the small terms entirely.
__global__ void rms_norm_kernel(const __nv_bfloat16* __restrict__ input,
                                const __nv_bfloat16* __restrict__ weight,
                                __nv_bfloat16* __restrict__ output, int rows,
                                int width, float epsilon) {
  extern __shared__ float partials[];
  const int row = blockIdx.x;
  if (row >= rows) return;
  const long long base = (long long)row * width;

  float sum = 0.0f;
  for (int index = threadIdx.x; index < width; index += blockDim.x) {
    const float value = __bfloat162float(input[base + index]);
    sum += value * value;
  }
  sum = warp_sum(sum);
  const int lane = threadIdx.x % kWarpSize;
  const int warp = threadIdx.x / kWarpSize;
  if (lane == 0) partials[warp] = sum;
  __syncthreads();

  if (threadIdx.x < kWarpSize) {
    const int warps = (blockDim.x + kWarpSize - 1) / kWarpSize;
    float total = threadIdx.x < warps ? partials[threadIdx.x] : 0.0f;
    total = warp_sum(total);
    if (threadIdx.x == 0) partials[0] = total;
  }
  __syncthreads();

  const float scale = rsqrtf(partials[0] / static_cast<float>(width) + epsilon);
  for (int index = threadIdx.x; index < width; index += blockDim.x) {
    const float value = __bfloat162float(input[base + index]) * scale *
                        __bfloat162float(weight[index]);
    output[base + index] = __float2bfloat16(value);
  }
}

__device__ __forceinline__ float silu(float value) {
  return value / (1.0f + __expf(-value));
}

// `fused` holds gate and up side by side: [rows, 2*width] with gate first.
// Qwen's MLP is SwiGLU -- silu(gate) * up -- which is not what gemm_geglu
// computes, so it cannot be reused.
__global__ void swiglu_kernel(const __nv_bfloat16* __restrict__ fused,
                              __nv_bfloat16* __restrict__ output, int rows,
                              int width) {
  const long long index = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  const long long total = (long long)rows * width;
  if (index >= total) return;
  const int row = static_cast<int>(index / width);
  const int column = static_cast<int>(index % width);
  const long long base = (long long)row * 2 * width;
  const float gate = __bfloat162float(fused[base + column]);
  const float up = __bfloat162float(fused[base + width + column]);
  output[index] = __float2bfloat16(silu(gate) * up);
}

__global__ void add_kernel(const __nv_bfloat16* __restrict__ addend,
                           __nv_bfloat16* __restrict__ accumulator,
                           long long count) {
  const long long index = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  if (index >= count) return;
  accumulator[index] = __float2bfloat16(__bfloat162float(accumulator[index]) +
                                        __bfloat162float(addend[index]));
}

// E4M3 with bias 7, saturating at +-448. Values are pre-divided by the
// per-tensor scale, so the representable range maps onto the calibrated one.
__device__ __forceinline__ uint8_t to_e4m3(float value) {
  const uint8_t sign = value < 0.0f ? 0x80 : 0x00;
  float magnitude = fabsf(value);
  if (!(magnitude > 0.0f)) return sign;
  if (magnitude >= 448.0f) return sign | 0x7E;  // saturate, never NaN
  int exponent;
  float mantissa = frexpf(magnitude, &exponent);
  mantissa *= 2.0f;
  exponent -= 1;
  int biased = exponent + 7;
  int fraction = __float2int_rn((mantissa - 1.0f) * 8.0f);
  if (fraction > 7) {
    fraction = 0;
    ++biased;
  }
  if (biased <= 0) return sign;                 // flush subnormals to zero
  if (biased > 15) return sign | 0x7E;
  return sign | static_cast<uint8_t>((biased << 3) | fraction);
}

__global__ void quantize_fp8_kernel(const __nv_bfloat16* __restrict__ input,
                                    uint8_t* __restrict__ output,
                                    long long count, float inverse_scale) {
  const long long index = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  if (index >= count) return;
  output[index] = to_e4m3(__bfloat162float(input[index]) * inverse_scale);
}

}  // namespace

int quantize_fp8_per_tensor(const void* input, void* output, long long count,
                            float input_scale, cudaStream_t stream) {
  if (count <= 0 || !(input_scale > 0.0f)) return -1;
  const int threads = 256;
  const long long blocks = (count + threads - 1) / threads;
  quantize_fp8_kernel<<<static_cast<int>(blocks), threads, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(input), static_cast<uint8_t*>(output),
      count, 1.0f / input_scale);
  return cudaGetLastError() == cudaSuccess ? 0 : -2;
}

int rms_norm_bf16(const void* input, const void* weight, void* output,
                  int rows, int width, float epsilon, cudaStream_t stream) {
  if (rows <= 0 || width <= 0) return -1;
  const int threads = width >= 1024 ? 1024 : ((width + 31) / 32) * 32;
  const int warps = (threads + kWarpSize - 1) / kWarpSize;
  rms_norm_kernel<<<rows, threads, warps * sizeof(float), stream>>>(
      static_cast<const __nv_bfloat16*>(input),
      static_cast<const __nv_bfloat16*>(weight),
      static_cast<__nv_bfloat16*>(output), rows, width, epsilon);
  return cudaGetLastError() == cudaSuccess ? 0 : -2;
}

int swiglu_bf16(const void* fused_gate_up, void* output, int rows, int width,
                cudaStream_t stream) {
  if (rows <= 0 || width <= 0) return -1;
  const long long total = (long long)rows * width;
  const int threads = 256;
  const long long blocks = (total + threads - 1) / threads;
  swiglu_kernel<<<static_cast<int>(blocks), threads, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(fused_gate_up),
      static_cast<__nv_bfloat16*>(output), rows, width);
  return cudaGetLastError() == cudaSuccess ? 0 : -2;
}

int add_bf16(const void* addend, void* accumulator, long long count,
             cudaStream_t stream) {
  if (count <= 0) return -1;
  const int threads = 256;
  const long long blocks = (count + threads - 1) / threads;
  add_kernel<<<static_cast<int>(blocks), threads, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(addend),
      static_cast<__nv_bfloat16*>(accumulator), count);
  return cudaGetLastError() == cudaSuccess ? 0 : -2;
}

}  // namespace apxinf::cuda::mlp_ops
