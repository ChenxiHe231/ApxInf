// Copyright 2026 ApxInf contributors.
//
// Elementwise and reduction operators the Qwen3.8 MLP needs, kept alongside
// the NVFP4 GEMM they feed. Each has exactly one implementation and no
// persisted selection, so per `doc/adding-new-kernels.md` section 6 they carry
// no recipe: there is nothing to tune between.

#include "mlp_ops.h"

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>

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
                        (1.0f + __bfloat162float(weight[index]));
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

__device__ __forceinline__ uint8_t to_e4m3_bits(float value) {
  const uint8_t sign = value < 0.0f ? 0x80 : 0;
  const float magnitude = fabsf(value);
  if (!(magnitude > 0.0f)) return sign;
  if (magnitude >= 448.0f) return sign | 0x7e;
  const uint32_t bits = __float_as_uint(magnitude);
  const uint32_t mantissa = bits & 0x7fffff;
  int biased = static_cast<int>((bits >> 23) & 255) - 120;
  uint32_t fraction = (mantissa + 0x7ffff + ((mantissa >> 20) & 1)) >> 20;
  if (fraction == 8) {
    fraction = 0;
    ++biased;
  }
  if (biased <= 0) return sign;
  if (biased > 15) return sign | 0x7e;
  return sign | static_cast<uint8_t>((biased << 3) | fraction);
}

__global__ void quantize_fp8_vector_kernel(const uint4* __restrict__ input,
                                           uint2* __restrict__ output,
                                           long long vectors, float inverse_scale) {
  const long long index = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  if (index >= vectors) return;
  const uint4 packed_input = input[index];
  const auto* elements = reinterpret_cast<const __nv_bfloat16*>(&packed_input);
  uint2 packed_output;
  auto* codes = reinterpret_cast<uint8_t*>(&packed_output);
#pragma unroll
  for (int offset = 0; offset < 8; ++offset) {
    codes[offset] = to_e4m3_bits(__bfloat162float(elements[offset]) * inverse_scale);
  }
  output[index] = packed_output;
}

__device__ __forceinline__ uint16_t to_e4m3_native_pair(float low, float high) {
  uint16_t packed = __nv_cvt_float2_to_fp8x2(make_float2(low, high), __NV_SATFINITE, __NV_E4M3);
  if (!(fabsf(low) >= 0.01513671875f))
    packed = (packed & 0xff00) | (low < 0.0f ? 0x80 : 0);
  if (!(fabsf(high) >= 0.01513671875f))
    packed = (packed & 0x00ff) | (high < 0.0f ? 0x8000 : 0);
  return packed;
}

__global__ void quantize_fp8_native_pair_kernel(const uint4* __restrict__ input,
                                               uint2* __restrict__ output,
                                               long long vectors, float inverse_scale) {
  const long long index = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  if (index >= vectors) return;
  const uint4 packed_input = input[index];
  const auto* elements = reinterpret_cast<const __nv_bfloat16*>(&packed_input);
  uint2 packed_output;
  auto* pairs = reinterpret_cast<uint16_t*>(&packed_output);
#pragma unroll
  for (int offset = 0; offset < 4; ++offset)
    pairs[offset] = to_e4m3_native_pair(
        __bfloat162float(elements[2 * offset]) * inverse_scale,
        __bfloat162float(elements[2 * offset + 1]) * inverse_scale);
  output[index] = packed_output;
}

__global__ void add_vector_kernel(const uint4* __restrict__ addend,
                                  uint4* __restrict__ accumulator,
                                  long long vectors) {
  const long long index = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  if (index >= vectors) return;
  const uint4 source = addend[index];
  uint4 destination = accumulator[index];
  const auto* source_pairs = reinterpret_cast<const __nv_bfloat162*>(&source);
  auto* destination_pairs = reinterpret_cast<__nv_bfloat162*>(&destination);
#pragma unroll
  for (int offset = 0; offset < 4; ++offset) {
    destination_pairs[offset] = __hadd2(destination_pairs[offset], source_pairs[offset]);
  }
  accumulator[index] = destination;
}

bool vector_elementwise_enabled(const void* source, const void* destination,
                                long long count) {
  const char* flag = std::getenv("APXINF_QWEN38_VECTOR_ELEMENTWISE");
  return flag != nullptr && std::strcmp(flag, "1") == 0 && count % 8 == 0 &&
         reinterpret_cast<uintptr_t>(source) % 16 == 0 &&
         reinterpret_cast<uintptr_t>(destination) % 16 == 0;
}

// One warp per output row, four rows per block, 16-byte loads.
//
// The shape of this kernel is set by what the first attempt got wrong. One
// block per row with scalar loads gave each thread ~20 bytes of work over
// 10240 blocks, and launch plus per-element conversion overhead buried the
// memory traffic it was trying to optimize. Here each thread issues 16-byte
// vector loads and carries ~160 bytes, and the block count drops 4x.
//
// The activation is read from global rather than staged in shared: staging
// makes every block re-read the same K bytes, which at N=10240 is 52 MB of
// redundant traffic -- as much as the weights.
__global__ void fp8_gemv_kernel(const uint4* __restrict__ weight,
                                const uint4* __restrict__ activation,
                                __nv_bfloat16* __restrict__ output, int n,
                                int k_vectors, float alpha) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int row = blockIdx.x * (blockDim.x >> 5) + warp;
  if (row >= n) return;

  const uint4* weight_row = weight + (long long)row * k_vectors;
  float sum = 0.0f;
  for (int index = lane; index < k_vectors; index += 32) {
    const uint4 w = weight_row[index];
    const uint4 a = activation[index];
    const __nv_fp8_storage_t* wb =
        reinterpret_cast<const __nv_fp8_storage_t*>(&w);
    const __nv_fp8_storage_t* ab =
        reinterpret_cast<const __nv_fp8_storage_t*>(&a);
#pragma unroll
    for (int element = 0; element < 16; ++element) {
      sum += __half2float(__nv_cvt_fp8_to_halfraw(wb[element], __NV_E4M3)) *
             __half2float(__nv_cvt_fp8_to_halfraw(ab[element], __NV_E4M3));
    }
  }
  for (int offset = 16; offset > 0; offset >>= 1) {
    sum += __shfl_down_sync(0xFFFFFFFFu, sum, offset);
  }
  if (lane == 0) output[row] = __float2bfloat16(alpha * sum);
}

__global__ void fp8_gemv_shared_kernel(const uint4* __restrict__ weight,
                                      const uint4* __restrict__ activation,
                                      __nv_bfloat16* __restrict__ output, int rows,
                                      int vectors, float alpha) {
  extern __shared__ float activation_shared[];
  for (int index = threadIdx.x; index < vectors; index += blockDim.x) {
    const uint4 packed = activation[index];
    const auto* codes = reinterpret_cast<const __nv_fp8_storage_t*>(&packed);
#pragma unroll
    for (int element = 0; element < 16; ++element)
      activation_shared[(index / 32) * 512 + element * 32 + index % 32] =
          __half2float(__nv_cvt_fp8_to_halfraw(codes[element], __NV_E4M3));
  }
  __syncthreads();
  const int lane = threadIdx.x % 32;
  const int row = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
  if (row >= rows) return;
  float sum = 0.0f;
  for (int index = lane; index < vectors; index += 32) {
    const uint4 packed = weight[(long long)row * vectors + index];
    const auto* codes = reinterpret_cast<const __nv_fp8_storage_t*>(&packed);
#pragma unroll
    for (int element = 0; element < 16; ++element)
      sum += __half2float(__nv_cvt_fp8_to_halfraw(codes[element], __NV_E4M3)) *
             activation_shared[(index / 32) * 512 + element * 32 + lane];
  }
  for (int offset = 16; offset > 0; offset >>= 1)
    sum += __shfl_down_sync(0xFFFFFFFFu, sum, offset);
  if (lane == 0) output[row] = __float2bfloat16(alpha * sum);
}

// E2M1 magnitudes indexed by the low three bits; bit 3 is the sign.
__constant__ float kE2M1Table[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};

__device__ __forceinline__ float from_e4m3(uint8_t code) {
  return __half2float(__nv_cvt_fp8_to_halfraw(code, __NV_E4M3));
}

__device__ __forceinline__ int unpack_e2m1_int8(unsigned selector) {
  const unsigned magnitudes = __byte_perm(0x03020100, 0x0c080604, selector & 0x7777);
  const unsigned signs = __byte_perm(0x0000ff00, 0, (selector >> 3) & 0x1111);
  return static_cast<int>(__vsub4(magnitudes ^ signs, signs));
}

// One warp per output row, four rows per block, 16-byte loads.
//
// Each uint4 carries 32 FP4 values, which is exactly two scale blocks, so the
// scale lookup is two loads per vector rather than one per element.
__global__ void nvfp4_gemv_kernel(const uint4* __restrict__ weight,
                                  const uint8_t* __restrict__ weight_scales,
                                  const uint4* __restrict__ activation,
                                  const uint8_t* __restrict__ activation_scales,
                                  __nv_bfloat16* __restrict__ output, int n,
                                  int k_vectors, float alpha) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int row = blockIdx.x * (blockDim.x >> 5) + warp;
  if (row >= n) return;

  const uint4* weight_row = weight + (long long)row * k_vectors;
  const uint8_t* weight_scale_row =
      weight_scales + (long long)row * k_vectors * 2;

  float sum = 0.0f;
  for (int index = lane; index < k_vectors; index += 32) {
    const uint4 w = weight_row[index];
    const uint4 a = activation[index];
    const uint8_t* wb = reinterpret_cast<const uint8_t*>(&w);
    const uint8_t* ab = reinterpret_cast<const uint8_t*>(&a);

#pragma unroll
    for (int block = 0; block < 2; ++block) {
      float accumulator = 0.0f;
#pragma unroll
      for (int byte = 0; byte < 8; ++byte) {
        const int offset = block * 8 + byte;
        const uint8_t wpair = wb[offset];
        const uint8_t apair = ab[offset];
        accumulator += kE2M1Table[wpair & 0x0F] * kE2M1Table[apair & 0x0F];
        accumulator += kE2M1Table[wpair >> 4] * kE2M1Table[apair >> 4];
      }
      const int scale_index = index * 2 + block;
      sum += accumulator * from_e4m3(weight_scale_row[scale_index]) *
             from_e4m3(activation_scales[scale_index]);
    }
  }
  for (int offset = 16; offset > 0; offset >>= 1) {
    sum += __shfl_down_sync(0xFFFFFFFFu, sum, offset);
  }
  if (lane == 0) output[row] = __float2bfloat16(alpha * sum);
}

__global__ void nvfp4_dp4a_gemv_kernel(
    const uint4* __restrict__ weight, const uint8_t* __restrict__ weight_scales,
    const uint4* __restrict__ activation, const uint8_t* __restrict__ activation_scales,
    __nv_bfloat16* __restrict__ output, int n, int k_vectors, float alpha) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int row = blockIdx.x * (blockDim.x >> 5) + warp;
  if (row >= n) return;
  const uint4* weight_row = weight + (long long)row * k_vectors;
  const uint8_t* weight_scale_row = weight_scales + (long long)row * k_vectors * 2;
  float sum = 0.0f;
  for (int index = lane; index < k_vectors; index += 32) {
    const uint4 w = weight_row[index];
    const uint4 a = activation[index];
    const auto* weight_words = reinterpret_cast<const uint16_t*>(&w);
    const auto* activation_words = reinterpret_cast<const uint16_t*>(&a);
#pragma unroll
    for (int block = 0; block < 2; ++block) {
      int accumulator = 0;
#pragma unroll
      for (int word = 0; word < 4; ++word) {
        accumulator = __dp4a(
            unpack_e2m1_int8(weight_words[block * 4 + word]),
            unpack_e2m1_int8(activation_words[block * 4 + word]), accumulator);
      }
      const int scale_index = index * 2 + block;
      sum += static_cast<float>(accumulator) * 0.25f *
             from_e4m3(weight_scale_row[scale_index]) *
             from_e4m3(activation_scales[scale_index]);
    }
  }
  for (int offset = 16; offset > 0; offset >>= 1) {
    sum += __shfl_down_sync(0xFFFFFFFFu, sum, offset);
  }
  if (lane == 0) output[row] = __float2bfloat16(alpha * sum);
}

}  // namespace

int nvfp4_gemv(const void* weight, const void* weight_scales,
               const void* activation, const void* activation_scales,
               void* output, int n, int k, float alpha, cudaStream_t stream) {
  if (n <= 0 || k <= 0) return -1;
  // 16-byte loads cover 32 FP4 values, which is two whole scale blocks.
  if (k % 32 != 0) return -2;
  constexpr int kRowsPerBlock = 4;
  const int threads = kRowsPerBlock * 32;
  const int blocks = (n + kRowsPerBlock - 1) / kRowsPerBlock;
  const char* dp4a_flag = std::getenv("APXINF_QWEN38_NVFP4_DP4A");
  if (dp4a_flag != nullptr && std::strcmp(dp4a_flag, "1") == 0 &&
      reinterpret_cast<uintptr_t>(weight) % 16 == 0 &&
      reinterpret_cast<uintptr_t>(activation) % 16 == 0) {
    nvfp4_dp4a_gemv_kernel<<<blocks, threads, 0, stream>>>(
        static_cast<const uint4*>(weight),
        static_cast<const uint8_t*>(weight_scales),
        static_cast<const uint4*>(activation),
        static_cast<const uint8_t*>(activation_scales),
        static_cast<__nv_bfloat16*>(output), n, k / 32, alpha);
    return cudaGetLastError() == cudaSuccess ? 0 : -3;
  }
  nvfp4_gemv_kernel<<<blocks, threads, 0, stream>>>(
      static_cast<const uint4*>(weight),
      static_cast<const uint8_t*>(weight_scales),
      static_cast<const uint4*>(activation),
      static_cast<const uint8_t*>(activation_scales),
      static_cast<__nv_bfloat16*>(output), n, k / 32, alpha);
  return cudaGetLastError() == cudaSuccess ? 0 : -3;
}

int fp8_gemv(const void* weight, const void* activation, void* output, int n,
             int k, float alpha, cudaStream_t stream) {
  if (n <= 0 || k <= 0) return -1;
  // 16-byte loads need K to be a multiple of 16. Every projection in this
  // model satisfies that; reject rather than silently scalarize.
  if (k % 16 != 0) return -2;
  // FP8 decode projections are memory-bound and launch thousands of small
  // output-row blocks. Eight independent warps per block cuts the grid in
  // half while preserving the one-warp-per-row reduction.
  constexpr int kRowsPerBlock = 8;
  const int threads = kRowsPerBlock * 32;
  const int blocks = (n + kRowsPerBlock - 1) / kRowsPerBlock;
  const char* shared = std::getenv("APXINF_FP8_GEMV_SHARED");
  if (shared != nullptr && std::strcmp(shared, "1") == 0 &&
      k % 512 == 0 && k <= 8192 && n <= 6144) {
    fp8_gemv_shared_kernel<<<blocks, threads, k * sizeof(float), stream>>>(
        static_cast<const uint4*>(weight), static_cast<const uint4*>(activation),
        static_cast<__nv_bfloat16*>(output), n, k / 16, alpha);
    return cudaGetLastError() == cudaSuccess ? 0 : -3;
  }
  fp8_gemv_kernel<<<blocks, threads, 0, stream>>>(
      static_cast<const uint4*>(weight), static_cast<const uint4*>(activation),
      static_cast<__nv_bfloat16*>(output), n, k / 16, alpha);
  return cudaGetLastError() == cudaSuccess ? 0 : -3;
}

int quantize_fp8_per_tensor(const void* input, void* output, long long count,
                            float input_scale, cudaStream_t stream) {
  if (count <= 0 || !(input_scale > 0.0f)) return -1;
  const int threads = 256;
  if (vector_elementwise_enabled(input, output, count)) {
    const long long vectors = count / 8;
    const char* native_pair = std::getenv("APXINF_FP8_NATIVE_PAIR");
    if (native_pair != nullptr && std::strcmp(native_pair, "1") == 0) {
      quantize_fp8_native_pair_kernel<<<static_cast<int>((vectors + threads - 1) / threads), threads, 0, stream>>>(
          static_cast<const uint4*>(input), static_cast<uint2*>(output), vectors, 1.0f / input_scale);
      return cudaGetLastError() == cudaSuccess ? 0 : -2;
    }
    quantize_fp8_vector_kernel<<<static_cast<int>((vectors + threads - 1) / threads), threads, 0, stream>>>(
        static_cast<const uint4*>(input), static_cast<uint2*>(output), vectors, 1.0f / input_scale);
    return cudaGetLastError() == cudaSuccess ? 0 : -2;
  }
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
  if (vector_elementwise_enabled(addend, accumulator, count)) {
    const long long vectors = count / 8;
    add_vector_kernel<<<static_cast<int>((vectors + threads - 1) / threads), threads, 0, stream>>>(
        static_cast<const uint4*>(addend), static_cast<uint4*>(accumulator), vectors);
    return cudaGetLastError() == cudaSuccess ? 0 : -2;
  }
  const long long blocks = (count + threads - 1) / threads;
  add_kernel<<<static_cast<int>(blocks), threads, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(addend),
      static_cast<__nv_bfloat16*>(accumulator), count);
  return cudaGetLastError() == cudaSuccess ? 0 : -2;
}

}  // namespace apxinf::cuda::mlp_ops
