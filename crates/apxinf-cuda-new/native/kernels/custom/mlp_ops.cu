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

namespace apxinf::cuda_new::mlp_ops {
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

constexpr int kResidualNormRows = 4096;
constexpr int kResidualNormWidth = 2048;
constexpr int kResidualNormThreads = 128;
constexpr int kResidualNormPartials = 32;
static_assert(kResidualNormThreads * 16 == kResidualNormWidth);

__device__ __forceinline__ float unpack_residual_bf16(uint32_t word, int half) {
  return __bfloat162float(__ushort_as_bfloat16(
      static_cast<unsigned short>(word >> (half * 16))));
}

__device__ __forceinline__ uint32_t pack_residual_bf16(float value) {
  return __bfloat16_as_ushort(__float2bfloat16_rn(value));
}

__device__ __forceinline__ float residual_norm_warp_sum(float value) {
  for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
    value = __fadd_rn(value, __shfl_down_sync(0xffffffffu, value, offset));
  }
  return value;
}

// FIB/001's fused residual/RMSNorm, adapted to preserve ApxInf's BF16
// residual writeback and FP32 (1 + weight) affine. Four physical lanes
// carry 32 virtual lanes of the original RMSNorm reduction tree.
__global__ void add_rms_norm_kernel(
    const __nv_bfloat16* __restrict__ input,
    __nv_bfloat16* __restrict__ residual,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ output, float epsilon) {
  __shared__ float partials[kResidualNormPartials + 1];
  const int tid = threadIdx.x;
  const int64_t row_offset = static_cast<int64_t>(blockIdx.x) * kResidualNormWidth;
  input += row_offset;
  residual += row_offset;
  output += row_offset;
  const auto* input_vectors = reinterpret_cast<const uint4*>(input);
  const uint4 a0 = input_vectors[tid];
  const uint4 a1 = input_vectors[tid + kResidualNormThreads];
  uint32_t low[4] = {a0.x, a0.y, a0.z, a0.w};
  uint32_t high[4] = {a1.x, a1.y, a1.z, a1.w};
  auto* residual_vectors = reinterpret_cast<uint4*>(residual);
  const uint4 b0 = residual_vectors[tid];
  const uint4 b1 = residual_vectors[tid + kResidualNormThreads];
  const uint32_t old_low[4] = {b0.x, b0.y, b0.z, b0.w};
  const uint32_t old_high[4] = {b1.x, b1.y, b1.z, b1.w};
#pragma unroll
  for (int word = 0; word < 4; ++word) {
    const auto add_pair = [](uint32_t old, uint32_t next) {
      const auto lo = pack_residual_bf16(__fadd_rn(unpack_residual_bf16(old, 0), unpack_residual_bf16(next, 0)));
      const auto hi = pack_residual_bf16(__fadd_rn(unpack_residual_bf16(old, 1), unpack_residual_bf16(next, 1)));
      return lo | (hi << 16);
    };
    low[word] = add_pair(old_low[word], low[word]);
    high[word] = add_pair(old_high[word], high[word]);
  }
  residual_vectors[tid] = make_uint4(low[0], low[1], low[2], low[3]);
  residual_vectors[tid + kResidualNormThreads] = make_uint4(high[0], high[1], high[2], high[3]);

  float squares[8];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const float a = unpack_residual_bf16(low[i / 2], i % 2);
    const float b = unpack_residual_bf16(high[i / 2], i % 2);
    float sum = __fmaf_rn(b, b, __fmul_rn(a, a));
    sum = __fadd_rn(sum, __shfl_down_sync(0xffffffffu, sum, 2, 4));
    squares[i] = __fadd_rn(sum, __shfl_down_sync(0xffffffffu, sum, 1, 4));
  }
  if (tid % 4 == 0) {
    const float even = __fadd_rn(__fadd_rn(squares[0], squares[4]),
                                 __fadd_rn(squares[2], squares[6]));
    const float odd = __fadd_rn(__fadd_rn(squares[1], squares[5]),
                                __fadd_rn(squares[3], squares[7]));
    partials[tid / 4] = __fadd_rn(even, odd);
  }
  __syncthreads();
  if (tid < kWarpSize) {
    const float total = residual_norm_warp_sum(partials[tid]);
    if (tid == 0) {
      const float mean = __fmul_rn(total, 1.0f / static_cast<float>(kResidualNormWidth));
      partials[kResidualNormPartials] = rsqrtf(__fadd_rn(mean, epsilon));
    }
  }
  __syncthreads();

  const float scale = partials[kResidualNormPartials];
  const auto* weight_vectors = reinterpret_cast<const uint4*>(weight);
  const uint4 w0 = weight_vectors[tid];
  const uint4 w1 = weight_vectors[tid + kResidualNormThreads];
  const uint32_t weight_low[4] = {w0.x, w0.y, w0.z, w0.w};
  const uint32_t weight_high[4] = {w1.x, w1.y, w1.z, w1.w};
#pragma unroll
  for (int word = 0; word < 4; ++word) {
    const auto normalize_pair = [scale](uint32_t values, uint32_t weights) {
      const auto normalize = [&](int half) {
        const float scaled = __fmul_rn(unpack_residual_bf16(values, half), scale);
        const float gain = __fadd_rn(1.0f, unpack_residual_bf16(weights, half));
        return pack_residual_bf16(__fmul_rn(scaled, gain));
      };
      return normalize(0) | (normalize(1) << 16);
    };
    low[word] = normalize_pair(low[word], weight_low[word]);
    high[word] = normalize_pair(high[word], weight_high[word]);
  }
  auto* output_vectors = reinterpret_cast<uint4*>(output);
  output_vectors[tid] = make_uint4(low[0], low[1], low[2], low[3]);
  output_vectors[tid + kResidualNormThreads] = make_uint4(high[0], high[1], high[2], high[3]);
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
  // Reference semantics are bf16(bf16(silu(gate)) * up): the SiLU output
  // materializes in BF16 before the multiply (vLLM parity defect D4,
  // report 64 / Codex round 63 -- rounding only the product is not enough).
  const float activated = __bfloat162float(__float2bfloat16(silu(gate)));
  output[index] = __float2bfloat16(activated * up);
}

// Gate and up are separate contiguous [rows, width] matrices. This is used
// when two independent projections share one scratch allocation as
// [all-gate-rows][all-up-rows], which is not the row-interleaved layout
// accepted by swiglu_kernel for rows > 1.
__global__ void swiglu_split_kernel(
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ up,
    __nv_bfloat16* __restrict__ output, long long total) {
  const long long index = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  if (index >= total) return;
  const float gate_value = __bfloat162float(gate[index]);
  const float up_value = __bfloat162float(up[index]);
  const float activated =
      __bfloat162float(__float2bfloat16(silu(gate_value)));
  output[index] = __float2bfloat16(activated * up_value);
}

__global__ void swiglu_round_product_kernel(
    const __nv_bfloat16* __restrict__ fused,
    __nv_bfloat16* __restrict__ output, int rows, int width) {
  const long long index = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  const long long total = (long long)rows * width;
  if (index >= total) return;
  const int row = static_cast<int>(index / width);
  const int column = static_cast<int>(index % width);
  const long long base = (long long)row * 2 * width;
  const float gate = __bfloat162float(fused[base + column]);
  const float up = __bfloat162float(fused[base + width + column]);
  output[index] = __float2bfloat16((gate / (1.0f + expf(-gate))) * up);
}

// Canonical GEMM weights are [K, N]. One block owns 32 adjacent output
// columns while its eight warps split K. Every iteration therefore issues a
// coalesced weight load and broadcasts one activation value per warp. The
// fixed eight-way reduction avoids the 2K-long dependent FMA chain of the old
// one-thread-per-column implementation.
__global__ void bf16_gemv_kernel(
    const __nv_bfloat16* __restrict__ activation,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ output, int n, int k) {
  __shared__ float partials[8][32];
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int column = blockIdx.x * 32 + lane;
  float sum = 0.0f;
  if (column < n) {
    for (int index = warp; index < k; index += 8) {
      sum = fmaf(__bfloat162float(activation[index]),
                 __bfloat162float(weight[(long long)index * n + column]), sum);
    }
  }
  partials[warp][lane] = sum;
  __syncthreads();
  if (warp == 0 && column < n) {
    float total = partials[0][lane];
#pragma unroll
    for (int split = 1; split < 8; ++split) total += partials[split][lane];
    output[column] = __float2bfloat16(total);
  }
}

// Very narrow projections (the shared-expert gate is [K, 1]) need a K-wise
// warp reduction rather than the N-wise tiled kernel above. One warp owns one
// output and reads contiguous [K, N] elements when N is at most four.
__global__ void bf16_narrow_gemv_kernel(
    const __nv_bfloat16* __restrict__ activation,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ output, int n, int k) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  if (warp >= n) return;
  float sum = 0.0f;
  for (int index = lane; index < k; index += 32) {
    sum = fmaf(__bfloat162float(activation[index]),
               __bfloat162float(weight[(long long)index * n + warp]), sum);
  }
  sum = warp_sum(sum);
  if (lane == 0) output[warp] = __float2bfloat16(sum);
}

// Eight elements per thread through 16-byte vectors. One bf16 per thread
// (round-05 nsys: 46.7 ms of a 874 ms prefill across 130 calls) leaves the
// kernel launch-and-index bound; the tail elements run scalar so any count
// works. Same per-element arithmetic, bit-identical.
__global__ void add_kernel(const __nv_bfloat16* __restrict__ addend,
                           __nv_bfloat16* __restrict__ accumulator,
                           long long count) {
  const long long vectors = count / 8;
  const long long index = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  if (index < vectors) {
    uint4 a = reinterpret_cast<const uint4*>(addend)[index];
    uint4 c = reinterpret_cast<uint4*>(accumulator)[index];
    const __nv_bfloat16* av = reinterpret_cast<const __nv_bfloat16*>(&a);
    __nv_bfloat16* cv = reinterpret_cast<__nv_bfloat16*>(&c);
#pragma unroll
    for (int element = 0; element < 8; ++element) {
      cv[element] = __float2bfloat16(__bfloat162float(cv[element]) +
                                     __bfloat162float(av[element]));
    }
    reinterpret_cast<uint4*>(accumulator)[index] = c;
    return;
  }
  const long long tail = vectors * 8 + (index - vectors);
  if (tail < count) {
    accumulator[tail] = __float2bfloat16(__bfloat162float(accumulator[tail]) +
                                         __bfloat162float(addend[tail]));
  }
}

// E4M3 with bias 7, saturating at +-448. Values are pre-divided by the
// per-tensor scale, so the representable range maps onto the calibrated one.
// The hardware SATFINITE convert keeps subnormals (the old frexpf ladder
// flushed them to signed zero -- vLLM parity defect D3, report 64).
__device__ __forceinline__ uint8_t to_e4m3(float value) {
  return __nv_cvt_float_to_fp8(value, __NV_SATFINITE, __NV_E4M3);
}

// Eight elements per thread, 16-byte input vectors and 8-byte output
// stores. Same to_e4m3 per element in element order; bit-identical.
__global__ void quantize_fp8_kernel(const __nv_bfloat16* __restrict__ input,
                                    uint8_t* __restrict__ output,
                                    long long count, float inverse_scale) {
  const long long vectors = count / 8;
  const long long index = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  if (index < vectors) {
    const uint4 in = reinterpret_cast<const uint4*>(input)[index];
    const __nv_bfloat16* iv = reinterpret_cast<const __nv_bfloat16*>(&in);
    uint64_t packed = 0;
#pragma unroll
    for (int element = 0; element < 8; ++element) {
      packed |= static_cast<uint64_t>(
                    to_e4m3(__bfloat162float(iv[element]) * inverse_scale))
                << (8 * element);
    }
    reinterpret_cast<uint64_t*>(output)[index] = packed;
    return;
  }
  const long long tail = vectors * 8 + (index - vectors);
  if (tail < count) {
    output[tail] = to_e4m3(__bfloat162float(input[tail]) * inverse_scale);
  }
}

// Same conversion as to_e4m3; kept as a separate name so the vector kernel's
// dispatch identity survives (it used to be a bit-twiddling variant with the
// same D3 subnormal flush).
__device__ __forceinline__ uint8_t to_e4m3_bits(float value) {
  return to_e4m3(value);
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
  // Bare SATFINITE convert: the old small-magnitude zeroing branch replicated
  // the D3 subnormal flush (report 64); the hardware keeps subnormals.
  return __nv_cvt_float2_to_fp8x2(make_float2(low, high), __NV_SATFINITE, __NV_E4M3);
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
//
// Paired conversion (fp8x2 -> half2): one cvt per two elements. Round-00
// ncu had the scalar-convert version at 58% SM; pairing dropped it to 24.5%
// (141.7 vs 153 us). fp8->half is exact and the f32 accumulation order is
// unchanged, so the sum is bit-identical to the scalar version. A second
// accumulator chain (ILP) was tried on top and rejected: 89.38 vs 89.79
// ms/tok end to end -- the kernel is latency-bound on loads, not FMA -- and
// the reordered sum moved generated tokens, breaking the bit-identity gate
// (round-02).
//
// Software prefetch instead: issue the next iteration's two loads before
// converting the current one, so the load latency overlaps the 32 FMAs.
// The values and their accumulation order are untouched -- only the issue
// distance moves -- so this stays bit-identical.
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
  int index = lane;
  uint4 w = index < k_vectors ? weight_row[index] : uint4{0, 0, 0, 0};
  uint4 a = index < k_vectors ? activation[index] : uint4{0, 0, 0, 0};
  while (index < k_vectors) {
    const int next = index + 32;
    uint4 w_next{0, 0, 0, 0};
    uint4 a_next{0, 0, 0, 0};
    if (next < k_vectors) {
      w_next = weight_row[next];
      a_next = activation[next];
    }
    const __nv_fp8x2_storage_t* wp =
        reinterpret_cast<const __nv_fp8x2_storage_t*>(&w);
    const __nv_fp8x2_storage_t* ap =
        reinterpret_cast<const __nv_fp8x2_storage_t*>(&a);
#pragma unroll
    for (int pair = 0; pair < 8; ++pair) {
      const __half2 wh =
          __half2(__nv_cvt_fp8x2_to_halfraw2(wp[pair], __NV_E4M3));
      const __half2 ah =
          __half2(__nv_cvt_fp8x2_to_halfraw2(ap[pair], __NV_E4M3));
      sum += __low2float(wh) * __low2float(ah);
      sum += __high2float(wh) * __high2float(ah);
    }
    w = w_next;
    a = a_next;
    index = next;
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
//
// FP4 decodes by bit placement, not by table or cvt. sm_110 has no FP4
// hardware convert: __nv_cvt_fp4x2_to_halfraw2 compiles to a LOP3 sea that
// left this kernel at 96% SM / 6% memory, 994 us for the down projection
// (round-01 ncu, 01-gemv-hwcvt.ncu-rep). A __constant__ table per nibble is
// as bad through a different pipe: 32 divergent constant-cache reads per
// warp (round-01, 01-gemv-slow.ncu-rep, 4.7 ms).
//
// The placement trick: an E2M1 nibble scattered into a binary16 as
// [sign<<15 | exp<<10..9 | man<<9] reads as the FP4 value scaled by 2^-14,
// for normals and subnormals both. One shift-mask pair per element plus one
// exact power-of-two multiply recovers the value; all of it dual-issues as
// plain ALU.
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

  // Both nibbles of one packed byte, placed into a half2. Exact.
  const auto pair_to_half2 = [](uint32_t byte) -> __half2 {
    const uint32_t bits = ((byte & 0x07u) << 9) | ((byte & 0x08u) << 12) |
                          ((byte & 0x70u) << 21) | ((byte & 0x80u) << 24);
    return *reinterpret_cast<const __half2*>(&bits);
  };
  constexpr float kRescale = 16384.0f;  // 2^14, exact

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
        const __half2 wh = pair_to_half2(wb[offset]);
        const __half2 ah = pair_to_half2(ab[offset]);
        // One 2^28 rescale per product pair keeps the f32 accumulation
        // order of the reference kernel: (w*2^-14)*(a*2^-14)*2^28 == w*a
        // exactly, because every intermediate is a power-of-two scaling
        // of an exactly-representable value.
        accumulator = fmaf(__low2float(wh) * __low2float(ah),
                           kRescale * kRescale, accumulator);
        accumulator = fmaf(__high2float(wh) * __high2float(ah),
                           kRescale * kRescale, accumulator);
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

int bf16_gemv(const void* activation, const void* weight, void* output, int n,
              int k, cudaStream_t stream) {
  if (n <= 0 || k <= 0) return -1;
  constexpr int kThreads = 256;
  if (n <= 4) {
    bf16_narrow_gemv_kernel<<<1, 128, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(activation),
        static_cast<const __nv_bfloat16*>(weight),
        static_cast<__nv_bfloat16*>(output), n, k);
    return cudaGetLastError() == cudaSuccess ? 0 : -2;
  }
  const int blocks = (n + 31) / 32;
  bf16_gemv_kernel<<<blocks, kThreads, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(activation),
      static_cast<const __nv_bfloat16*>(weight),
      static_cast<__nv_bfloat16*>(output), n, k);
  return cudaGetLastError() == cudaSuccess ? 0 : -2;
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
  // One thread per 8-element vector plus one per tail element.
  const long long work = count / 8 + count % 8;
  const long long blocks = (work + threads - 1) / threads;
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

int add_rms_norm_bf16(const void* addend, void* residual, const void* weight,
                      void* normalized, float epsilon, cudaStream_t stream) {
  add_rms_norm_kernel<<<kResidualNormRows, kResidualNormThreads, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(addend),
      static_cast<__nv_bfloat16*>(residual),
      static_cast<const __nv_bfloat16*>(weight),
      static_cast<__nv_bfloat16*>(normalized), epsilon);
  return static_cast<int>(cudaGetLastError());
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

int swiglu_split_bf16(const void* gate, const void* up, void* output, int rows,
                      int width, cudaStream_t stream) {
  if (gate == nullptr || up == nullptr || output == nullptr || rows <= 0 ||
      width <= 0) {
    return -1;
  }
  const long long total = static_cast<long long>(rows) * width;
  const int threads = 256;
  const int blocks = static_cast<int>((total + threads - 1) / threads);
  swiglu_split_kernel<<<blocks, threads, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(gate),
      static_cast<const __nv_bfloat16*>(up),
      static_cast<__nv_bfloat16*>(output), total);
  return cudaGetLastError() == cudaSuccess ? 0 : -2;
}

int swiglu_bf16_round_product(const void* fused_gate_up, void* output, int rows,
                              int width, cudaStream_t stream) {
  const int threads = 256;
  const long long count = static_cast<long long>(rows) * width;
  const int blocks = static_cast<int>((count + threads - 1) / threads);
  swiglu_round_product_kernel<<<blocks, threads, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(fused_gate_up),
      static_cast<__nv_bfloat16*>(output), rows, width);
  return static_cast<int>(cudaGetLastError());
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
  // One thread per 8-element vector plus one per tail element.
  const long long work = count / 8 + count % 8;
  const long long blocks = (work + threads - 1) / threads;
  add_kernel<<<static_cast<int>(blocks), threads, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(addend),
      static_cast<__nv_bfloat16*>(accumulator), count);
  return cudaGetLastError() == cudaSuccess ? 0 : -2;
}

}  // namespace apxinf::cuda_new::mlp_ops
