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
#include <mma.h>
#include <cstdlib>
#include <cstring>

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

// Experimental LM-head M1/K2048 only. No new arithmetic/quantization path:
// the original per-lane K-block/pair FMA sequence is retained. GPU validation
// and register/occupancy checks are mandatory before any performance claim.
__global__ __launch_bounds__(kThreadsPerBlock, 1)
void nvfp4_a16_blackwell_m1_stage_all_bf16(
    const __nv_bfloat16* __restrict__ activation,
    const unsigned char* __restrict__ qweights,
    const unsigned char* __restrict__ block_scales,
    const float* __restrict__ global_scale,
    __nv_bfloat16* __restrict__ output, int runtime_k_blocks) {
  constexpr int kRows = 1;
  constexpr int kStaticKBlocks = 32;
  __shared__ __align__(16) float shared_activation[32 * kSmemRowStride];

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

  // Keep the original 128-bit retain load and BF16-pair conversion, but
  // stage all K blocks once. Each K64 keeps its original four-float half pad.
  constexpr int kElementsPerVector = sizeof(Uint4) / sizeof(__nv_bfloat16);
  constexpr int kVectors = 2048 / kElementsPerVector;
  for (int vector = static_cast<int>(threadIdx.x); vector < kVectors;
       vector += kThreadsPerBlock) {
    const Uint4 packed_activation = load_global_128_retain(
        activation + vector * kElementsPerVector);
    const __nv_bfloat16* packed_elements =
        reinterpret_cast<const __nv_bfloat16*>(&packed_activation);
    const int element = vector * kElementsPerVector;
    const int k_block = element / kKTile;
    const int logical_k = element % kKTile;
    const int padded_k = logical_k + (logical_k >= kKTile / 2 ? 4 : 0);
    float2* staged_pairs = reinterpret_cast<float2*>(
        shared_activation + k_block * kSmemRowStride + padded_k);
#pragma unroll
    for (int pair = 0; pair < kElementsPerVector / 2; ++pair) {
      staged_pairs[pair] = __bfloat1622float2(
          *reinterpret_cast<const __nv_bfloat162*>(packed_elements + pair * 2));
    }
  }
  __syncthreads();
  for (int k_block = 0; k_block < k_blocks; ++k_block) {
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
              shared_activation + k_block * kSmemRowStride + k_in_tile);
          const float dot = fmaf(values.x, weights.x, values.y * weights.y);
          accumulators[activation_row] =
              fmaf(dot, scale, accumulators[activation_row]);
        }
      }
    }
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

// BF16 A and exact BF16(E2M1 * E4M3-block-scale) B; no activation
// quantization. The separate F32 global scale remains AFTER the full dot.
// Pad only the WMMA M dimension to16. N tiling divides the original N128
// packing; each CTA stages exactly one contiguous tile's packed rows.
template <int kRows, int kTileN, int kStaticKBlocks>
__global__ __launch_bounds__(kTileN * 2, 1)
void nvfp4_a16_blackwell_wmma_bf16(
    const __nv_bfloat16* __restrict__ activation,
    const unsigned char* __restrict__ qweights,
    const unsigned char* __restrict__ block_scales,
    const float* __restrict__ global_scale,
    __nv_bfloat16* __restrict__ output, int n, int runtime_k_blocks) {
  constexpr int kBlockThreads = kTileN * 2;
  constexpr int kWmmaStride = kKTile + 8; // multiple8, rotate shared-memory banks
  __shared__ __align__(32) __nv_bfloat16 shared_a[16][kWmmaStride];
  __shared__ __align__(32) __nv_bfloat16 shared_b[kTileN][kWmmaStride];
  __shared__ __align__(32) float shared_output[16][kTileN];
  const int tid = threadIdx.x;
  const int warp = tid / 32;
  const int local_n = tid / 2;
  const int k_half = tid & 1;
  const int column = static_cast<int>(blockIdx.x) * kTileN + local_n;
  const int k_blocks = kStaticKBlocks == 0 ? runtime_k_blocks : kStaticKBlocks;
  const int k = k_blocks * kKTile;
  // Inactive M rows stay zero throughout all K tiles. Initialization is
  // outside the K loop and fenced before active-row stores/WMMA reads.
  for (int i = tid; i < 16 * kKTile; i += kBlockThreads)
    shared_a[i / kKTile][i % kKTile] = __float2bfloat16_rn(0.0F);
  __syncthreads();
  using namespace nvcuda;
  wmma::fragment<wmma::accumulator, 16, 16, 16, float> accumulator;
  wmma::fill_fragment(accumulator, 0.0F);
#pragma unroll 1
  for (int kb = 0; kb < k_blocks; ++kb) {
    for (int i = tid; i < kRows * kKTile; i += kBlockThreads) {
      const int row = i / kKTile;
      const int component = i % kKTile;
      shared_a[row][component] = activation[
          static_cast<long long>(row) * k + kb * kKTile + component];
    }
    const long long tile_row =
        (static_cast<long long>(column / kNTile) * k_blocks + kb) *
            kNTile + column % kNTile;
    const Uint4 packed = load_global_128_no_allocate(
        qweights + tile_row * kPackedBytesPerRowTile + k_half * 16);
    const unsigned short sf = *reinterpret_cast<const unsigned short*>(
        block_scales + tile_row * kScalesPerRowTile + k_half * 2);
    const float2 scales = fp8_pair_to_float2(sf);
    const unsigned int words[4]{packed.x, packed.y, packed.z, packed.w};
#pragma unroll
    for (int word = 0; word < 4; ++word) {
#pragma unroll
      for (int byte = 0; byte < 4; ++byte) {
        const int pair = word * 4 + byte;
        const float2 w = fp4_pair_to_float2((words[word] >> (byte * 8)) & 0xffU);
        const float scale = pair < 8 ? scales.x : scales.y;
        // Every finite FP4*E4M3 product has <=6 significant binary bits and
        // magnitude between2^-10 and2688 (or zero), exactly representable BF16.
        // global_scale is intentionally excluded from this conversion.
        shared_b[local_n][k_half * 32 + pair * 2] =
            __float2bfloat16_rn(w.x * scale);
        shared_b[local_n][k_half * 32 + pair * 2 + 1] =
            __float2bfloat16_rn(w.y * scale);
      }
    }
    __syncthreads();
#pragma unroll
    for (int component = 0; component < kKTile; component += 16) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16,
                     wmma::row_major> a_fragment;
      wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16,
                     wmma::col_major> b_fragment;
      wmma::load_matrix_sync(a_fragment, &shared_a[0][component], kWmmaStride);
      wmma::load_matrix_sync(b_fragment, &shared_b[warp * 16][component], kWmmaStride);
      wmma::mma_sync(accumulator, a_fragment, b_fragment, accumulator);
    }
    __syncthreads(); // all WMMA reads finish before next tile overwrites sharedB/A
  }
  wmma::store_matrix_sync(&shared_output[0][warp * 16], accumulator,
                          kTileN, wmma::mem_row_major);
  __syncthreads();
  for (int i = tid; i < kRows * kTileN; i += kBlockThreads) {
    const int row = i / kTileN;
    const int col = i % kTileN;
    output[static_cast<long long>(row) * n +
           static_cast<long long>(blockIdx.x) * kTileN + col] =
        __float2bfloat16_rn(shared_output[row][col] * global_scale[0]);
  }
}

template <int Rows, int TileN>
void launch_nvfp4_a16_blackwell_wmma(
    const __nv_bfloat16* a, const unsigned char* w,
    const unsigned char* s, const float* g, __nv_bfloat16* o,
    int n, int k, cudaStream_t stream) {
  if (k == 2048) {
    nvfp4_a16_blackwell_wmma_bf16<Rows, TileN, 32>
        <<<n / TileN, TileN * 2, 0, stream>>>(a, w, s, g, o, n, 32);
  } else {
    nvfp4_a16_blackwell_wmma_bf16<Rows, TileN, 0>
        <<<n / TileN, TileN * 2, 0, stream>>>(a, w, s, g, o, n, k / kKTile);
  }
}

template <int Rows>
void dispatch_nvfp4_a16_wmma_tile(
    const __nv_bfloat16* a, const unsigned char* w,
    const unsigned char* s, const float* g, __nv_bfloat16* o,
    int n, int k, int tile_n, cudaStream_t stream) {
  switch (tile_n) {
    case 32: launch_nvfp4_a16_blackwell_wmma<Rows, 32>(a,w,s,g,o,n,k,stream); break;
    case 64: launch_nvfp4_a16_blackwell_wmma<Rows, 64>(a,w,s,g,o,n,k,stream); break;
    case 128: launch_nvfp4_a16_blackwell_wmma<Rows, 128>(a,w,s,g,o,n,k,stream); break;
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
      (rows != 1 && rows != 2 && rows != 3 && rows != 4 && rows != 8 &&
       rows != 16)) {
    return -1;
  }
  const auto* a = static_cast<const __nv_bfloat16*>(activation);
  const auto* w = static_cast<const unsigned char*>(qweights);
  const auto* s = static_cast<const unsigned char*>(block_scales);
  const auto* g = static_cast<const float*>(global_scale);
  auto* o = static_cast<__nv_bfloat16*>(output);
  switch (rows) {
    case 1: {
      static const int stage_all_mode = [] {
        const char* value = std::getenv("APXINF_NVFP4_A16_M1_STAGE_ALL");
        if (!value || std::strcmp(value, "0") == 0) return 0;
        if (std::strcmp(value, "1") == 0) return 1;
        return -1;
      }();
      if (stage_all_mode < 0) return -3;
      if (stage_all_mode == 1 && n == 248320 && k == 2048) {
        nvfp4_a16_blackwell_m1_stage_all_bf16
            <<<n / kNTile, kThreadsPerBlock, 0, stream>>>(a, w, s, g, o, 32);
      } else {
        launch_nvfp4_a16_blackwell_gemv<1>(a, w, s, g, o, n, k, stream);
      }
      break;
    }
    case 2:
      launch_nvfp4_a16_blackwell_gemv<2>(a, w, s, g, o, n, k, stream);
      break;
    case 3:
      launch_nvfp4_a16_blackwell_gemv<3>(a, w, s, g, o, n, k, stream);
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


// Internal provider entry selected only in the L1 adapter. Public L3/C ABI is
// unchanged, old Rows8/16 remain available through the original implementation.
int nvfp4_a16_blackwell_wmma(const void* activation, const void* qweights,
                           const void* block_scales, const void* global_scale,
                           void* output, int n, int k, int rows, int tile_n,
                           cudaStream_t stream) {
  if (!activation || !qweights || !block_scales || !global_scale || !output ||
      n <= 0 || k <= 0 || n % kNTile || k % kKTile ||
      rows < 1 || rows > 4 || (tile_n != 32 && tile_n != 64 && tile_n != 128)) return -1;
  const auto* a = static_cast<const __nv_bfloat16*>(activation);
  const auto* w = static_cast<const unsigned char*>(qweights);
  const auto* s = static_cast<const unsigned char*>(block_scales);
  const auto* g = static_cast<const float*>(global_scale);
  auto* o = static_cast<__nv_bfloat16*>(output);
  switch (rows) {
    case 1: dispatch_nvfp4_a16_wmma_tile<1>(a,w,s,g,o,n,k,tile_n,stream); break;
    case 2: dispatch_nvfp4_a16_wmma_tile<2>(a,w,s,g,o,n,k,tile_n,stream); break;
    case 3: dispatch_nvfp4_a16_wmma_tile<3>(a,w,s,g,o,n,k,tile_n,stream); break;
    case 4: dispatch_nvfp4_a16_wmma_tile<4>(a,w,s,g,o,n,k,tile_n,stream); break;
  }
  return cudaGetLastError() == cudaSuccess ? 0 : -2;
}

}  // namespace apxinf::cuda_new::mlp_ops
