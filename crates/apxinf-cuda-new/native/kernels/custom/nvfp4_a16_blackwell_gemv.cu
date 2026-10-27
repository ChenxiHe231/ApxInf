/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Adapted from TensorRT-Edge-LLM's nvfp4A16BlackwellGemv JIT source.  This
 * translation unit keeps the exact BLACKWELL_N128_K64_V1 data path, but takes
 * N and K at launch time so one ApxInf binary can serve every Qwen projection.
 */

#include "mlp_ops.h"

#include <cuda_bf16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

namespace apxinf::cuda_new::mlp_ops {
namespace {

constexpr int kWarpSize = 32;
constexpr int kWarpsPerBlock = 8;
constexpr int kThreadsPerBlock = kWarpSize * kWarpsPerBlock;
constexpr int kNTile = 128;
constexpr int kKTile = 64;
constexpr int kRowsPerWarp = 16;
constexpr int kPackedBytesPerRowTile = kKTile / 2;
constexpr int kScalesPerRowTile = kKTile / 16;
constexpr int kPackedBytesPerThread = kPackedBytesPerRowTile / 2;
constexpr int kSmemHalfStride = kKTile / 2 + 4;
constexpr int kSmemRowStride = kKTile + 4;

struct __align__(16) Uint4 {
  unsigned int x;
  unsigned int y;
  unsigned int z;
  unsigned int w;
};

__device__ __forceinline__ Uint4 load_global_128_no_allocate(
    const void* pointer) {
  Uint4 value{};
  asm volatile("ld.global.L1::no_allocate.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(value.x), "=r"(value.y), "=r"(value.z), "=r"(value.w)
               : "l"(pointer));
  return value;
}

__device__ __forceinline__ Uint4 load_global_128_retain(const void* pointer) {
  Uint4 value{};
  asm volatile("ld.global.L1::evict_last.v4.u32 {%0, %1, %2, %3}, [%4];"
               : "=r"(value.x), "=r"(value.y), "=r"(value.z), "=r"(value.w)
               : "l"(pointer));
  return value;
}

__device__ __forceinline__ float2 fp4_pair_to_float2(
    unsigned int packed_byte) {
  const __half2_raw raw = __nv_cvt_fp4x2_to_halfraw2(
      static_cast<__nv_fp4x2_storage_t>(packed_byte), __NV_E2M1);
  return __half22float2(static_cast<half2>(raw));
}

__device__ __forceinline__ float2 fp8_pair_to_float2(
    unsigned short packed_bytes) {
  const __half2_raw raw = __nv_cvt_fp8x2_to_halfraw2(
      static_cast<__nv_fp8x2_storage_t>(packed_bytes), __NV_E4M3);
  return __half22float2(static_cast<half2>(raw));
}

template <int kRows, int kStaticKBlocks>
__global__ __launch_bounds__(kThreadsPerBlock, 1)
void nvfp4_a16_blackwell_gemv_bf16(
    const __nv_bfloat16* __restrict__ activation,
    const unsigned char* __restrict__ qweights,
    const unsigned char* __restrict__ block_scales,
    const float* __restrict__ global_scale,
    __nv_bfloat16* __restrict__ output, int runtime_k_blocks) {
  __shared__ __align__(16) float shared_activation[kRows * kSmemRowStride];

  const int lane = static_cast<int>(threadIdx.x) & (kWarpSize - 1);
  const int warp = static_cast<int>(threadIdx.x) / kWarpSize;
  const int row_in_tile = warp * kRowsPerWarp + lane / 2;
  const int k_half = lane & 1;
  const int n_block = static_cast<int>(blockIdx.x);
  const int output_column = n_block * kNTile + row_in_tile;
  const int k_blocks =
      kStaticKBlocks == 0 ? runtime_k_blocks : kStaticKBlocks;
  const int k = k_blocks * kKTile;
  const int n = static_cast<int>(gridDim.x) * kNTile;
  float accumulators[kRows]{};

  for (int k_block = 0; k_block < k_blocks; ++k_block) {
    constexpr int kElementsPerVector =
        static_cast<int>(sizeof(Uint4) / sizeof(__nv_bfloat16));
    constexpr int kVectorsPerTile =
        kRows * kKTile / kElementsPerVector;
    for (int vector = static_cast<int>(threadIdx.x); vector < kVectorsPerTile;
         vector += kThreadsPerBlock) {
      const int activation_row = vector / (kKTile / kElementsPerVector);
      const int vector_in_row = vector % (kKTile / kElementsPerVector);
      const __nv_bfloat16* source =
          activation + static_cast<long long>(k_block) * kKTile +
          static_cast<long long>(activation_row) * k +
          vector_in_row * kElementsPerVector;
      const Uint4 packed_activation = load_global_128_retain(source);
      const __nv_bfloat16* packed_elements =
          reinterpret_cast<const __nv_bfloat16*>(&packed_activation);
      const int logical_k = vector_in_row * kElementsPerVector;
      const int staged_k = logical_k + (logical_k >= kKTile / 2 ? 4 : 0);
      float2* staged_pairs = reinterpret_cast<float2*>(
          shared_activation + activation_row * kSmemRowStride + staged_k);
#pragma unroll
      for (int pair = 0; pair < kElementsPerVector / 2; ++pair) {
        staged_pairs[pair] = __bfloat1622float2(
            *reinterpret_cast<const __nv_bfloat162*>(packed_elements + pair * 2));
      }
    }
    __syncthreads();

    const long long row_tile_index =
        (static_cast<long long>(n_block) * k_blocks + k_block) * kNTile +
        row_in_tile;
    const unsigned char* packed_pointer =
        qweights + row_tile_index * kPackedBytesPerRowTile +
        k_half * kPackedBytesPerThread;
    const Uint4 packed_weights = load_global_128_no_allocate(packed_pointer);
    const unsigned short packed_scales =
        *reinterpret_cast<const unsigned short*>(
            block_scales + row_tile_index * kScalesPerRowTile + k_half * 2);
    const float2 scales = fp8_pair_to_float2(packed_scales);
    const unsigned int words[4]{packed_weights.x, packed_weights.y,
                                packed_weights.z, packed_weights.w};

#pragma unroll
    for (int word = 0; word < 4; ++word) {
#pragma unroll
      for (int byte = 0; byte < 4; ++byte) {
        const int pair = word * 4 + byte;
        const unsigned int packed_byte =
            (words[word] >> (byte * 8)) & 0xFFU;
        const float2 weights = fp4_pair_to_float2(packed_byte);
        const float scale = pair < 8 ? scales.x : scales.y;
        const int k_in_tile = k_half * kSmemHalfStride + pair * 2;
#pragma unroll
        for (int activation_row = 0; activation_row < kRows;
             ++activation_row) {
          const float2 values = *reinterpret_cast<const float2*>(
              shared_activation + activation_row * kSmemRowStride +
              k_in_tile);
          const float dot = fmaf(values.x, weights.x, values.y * weights.y);
          accumulators[activation_row] =
              fmaf(dot, scale, accumulators[activation_row]);
        }
      }
    }
    __syncthreads();
  }

#pragma unroll
  for (int activation_row = 0; activation_row < kRows; ++activation_row) {
    accumulators[activation_row] += __shfl_xor_sync(
        0xFFFFFFFFU, accumulators[activation_row], 1, kWarpSize);
    if (k_half == 0) {
      output[static_cast<long long>(activation_row) * n + output_column] =
          __float2bfloat16_rn(accumulators[activation_row] * global_scale[0]);
    }
  }
}

template <int kRows>
void launch_nvfp4_a16_blackwell_gemv(
    const __nv_bfloat16* activation, const unsigned char* qweights,
    const unsigned char* block_scales, const float* global_scale,
    __nv_bfloat16* output, int n, int k, cudaStream_t stream) {
  if (k == 2048) {
    nvfp4_a16_blackwell_gemv_bf16<kRows, 32>
        <<<n / kNTile, kThreadsPerBlock, 0, stream>>>(
            activation, qweights, block_scales, global_scale, output, 32);
  } else {
    nvfp4_a16_blackwell_gemv_bf16<kRows, 0>
        <<<n / kNTile, kThreadsPerBlock, 0, stream>>>(
            activation, qweights, block_scales, global_scale, output,
            k / kKTile);
  }
}

}  // namespace

int nvfp4_a16_blackwell_gemv(const void* activation, const void* qweights,
                              const void* block_scales,
                              const void* global_scale, void* output, int n,
                              int k, int rows, cudaStream_t stream) {
  if (activation == nullptr || qweights == nullptr || block_scales == nullptr ||
      global_scale == nullptr || output == nullptr || n <= 0 || k <= 0 ||
      n % kNTile != 0 || k % kKTile != 0 ||
      (rows != 1 && rows != 2 && rows != 4 && rows != 8 && rows != 16)) {
    return -1;
  }
  const auto* a = static_cast<const __nv_bfloat16*>(activation);
  const auto* w = static_cast<const unsigned char*>(qweights);
  const auto* s = static_cast<const unsigned char*>(block_scales);
  const auto* g = static_cast<const float*>(global_scale);
  auto* o = static_cast<__nv_bfloat16*>(output);
  switch (rows) {
    case 1:
      launch_nvfp4_a16_blackwell_gemv<1>(a, w, s, g, o, n, k, stream);
      break;
    case 2:
      launch_nvfp4_a16_blackwell_gemv<2>(a, w, s, g, o, n, k, stream);
      break;
    case 4:
      launch_nvfp4_a16_blackwell_gemv<4>(a, w, s, g, o, n, k, stream);
      break;
    case 8:
      launch_nvfp4_a16_blackwell_gemv<8>(a, w, s, g, o, n, k, stream);
      break;
    case 16:
      launch_nvfp4_a16_blackwell_gemv<16>(a, w, s, g, o, n, k, stream);
      break;
  }
  return cudaGetLastError() == cudaSuccess ? 0 : -2;
}

}  // namespace apxinf::cuda_new::mlp_ops
