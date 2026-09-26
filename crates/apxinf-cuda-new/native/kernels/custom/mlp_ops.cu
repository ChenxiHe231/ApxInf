// Copyright 2026 ApxInf contributors.
//
// Elementwise and reduction operators the Qwen3.8 MLP needs, kept alongside
// the NVFP4 GEMM they feed. Each has exactly one implementation and no
// persisted selection, so per `doc/adding-new-kernels.md` section 6 they carry
// no recipe: there is nothing to tune between.

#include "mlp_ops.h"

#include <cuda_bf16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>

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

__device__ __forceinline__ float from_e4m3(uint8_t code) {
  return __half2float(__nv_cvt_fp8_to_halfraw(code, __NV_E4M3));
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
  constexpr int kRowsPerBlock = 4;
  const int threads = kRowsPerBlock * 32;
  const int blocks = (n + kRowsPerBlock - 1) / kRowsPerBlock;
  fp8_gemv_kernel<<<blocks, threads, 0, stream>>>(
      static_cast<const uint4*>(weight), static_cast<const uint4*>(activation),
      static_cast<__nv_bfloat16*>(output), n, k / 16, alpha);
  return cudaGetLastError() == cudaSuccess ? 0 : -3;
}

int quantize_fp8_per_tensor(const void* input, void* output, long long count,
                            float input_scale, cudaStream_t stream) {
  if (count <= 0 || !(input_scale > 0.0f)) return -1;
  const int threads = 256;
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
  // One thread per 8-element vector plus one per tail element.
  const long long work = count / 8 + count % 8;
  const long long blocks = (work + threads - 1) / threads;
  add_kernel<<<static_cast<int>(blocks), threads, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(addend),
      static_cast<__nv_bfloat16*>(accumulator), count);
  return cudaGetLastError() == cudaSuccess ? 0 : -2;
}

}  // namespace apxinf::cuda::mlp_ops
