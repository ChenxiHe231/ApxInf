#pragma once

#include "../primitives/bf16_pairwise.cuh"

namespace apxinf::gemm::kernels {

// FTZ instructions implement the contract's input/output subnormal flushing
// without masking every operand in the inner loop. Directed bounds still
// detect inexact additions; zero must stay flushed after setting the odd bit.
__device__ __forceinline__ float pairwise_add_round_odd(float a, float b) {
  float result, up, down;
  asm("add.rz.ftz.f32 %0, %1, %2;" : "=f"(result) : "f"(a), "f"(b));
  asm("add.rp.ftz.f32 %0, %1, %2;" : "=f"(up) : "f"(a), "f"(b));
  asm("add.rm.ftz.f32 %0, %1, %2;" : "=f"(down) : "f"(a), "f"(b));
  if (isnan(result)) return __uint_as_float(0x7fc00000u);
  unsigned bits = __float_as_uint(result);
  if (up != down) bits |= 1u;
  return (bits & 0x7f800000u) == 0
      ? __uint_as_float(bits & 0x80000000u) : __uint_as_float(bits);
}

__device__ __forceinline__ float pairwise_bf16_product(float a, float b) {
  float result;
  asm("mul.rn.ftz.f32 %0, %1, %2;" : "=f"(result) : "f"(a), "f"(b));
  return result;
}

// NK BF16 weights, a single ordered K reduction and a final FP32 bias addition.
// Independent register accumulators reuse shared loads without reassociating
// any dot product. The padded shared stride avoids conflicts across columns.
template <int RowsPerThread, int ColumnsPerThread>
__global__ void pairwise_bf16_bias(
    const __nv_bfloat16* a, const __nv_bfloat16* b,
    const __nv_bfloat16* bias, __nv_bfloat16* out, int m, int n, int k) {
  constexpr int Rows = 8 * RowsPerThread;
  constexpr int Columns = 16 * ColumnsPerThread;
  const int n_tiles = int((int64_t(n) + Columns - 1) / Columns);
  const int tile_row = int(blockIdx.x) / n_tiles;
  const int tile_column = int(blockIdx.x) % n_tiles;
  const int thread = threadIdx.y * 16 + threadIdx.x;
  __shared__ float x[Rows][65], w[Columns][65];
  float sum[RowsPerThread][ColumnsPerThread] = {};
  for (int64_t begin = 0; begin < k; begin += 64) {
    for (int i = thread; i < Rows * 64; i += 128) {
      const int r = tile_row * Rows + i / 64;
      const int64_t c = begin + i % 64;
      x[i / 64][i % 64] = r < m && c < k
          ? __bfloat162float(a[int64_t(r) * k + c]) : 0.f;
    }
    for (int i = thread; i < Columns * 64; i += 128) {
      const int r = tile_column * Columns + i / 64;
      const int64_t c = begin + i % 64;
      w[i / 64][i % 64] = r < n && c < k
          ? __bfloat162float(b[int64_t(r) * k + c]) : 0.f;
    }
    __syncthreads();
    #pragma unroll 4
    for (int i = 0; i < 64 && begin + i < k; i += 2) {
      #pragma unroll
      for (int rr = 0; rr < RowsPerThread; ++rr) {
        #pragma unroll
        for (int cc = 0; cc < ColumnsPerThread; ++cc) {
          const int r = threadIdx.y + rr * 8;
          const int c = threadIdx.x + cc * 16;
          const float p0 = pairwise_bf16_product(x[r][i], w[c][i]);
          const float p1 = pairwise_bf16_product(x[r][i + 1], w[c][i + 1]);
          sum[rr][cc] = pairwise_add_round_odd(
              sum[rr][cc], pairwise_add_round_odd(p0, p1));
        }
      }
    }
    __syncthreads();
  }
  #pragma unroll
  for (int rr = 0; rr < RowsPerThread; ++rr) {
    #pragma unroll
    for (int cc = 0; cc < ColumnsPerThread; ++cc) {
      const int row = tile_row * Rows + threadIdx.y + rr * 8;
      const int column = tile_column * Columns + threadIdx.x + cc * 16;
      if (row < m && column < n) {
        out[int64_t(row) * n + column] = __float2bfloat16(
            __fadd_rn(sum[rr][cc], __bfloat162float(bias[column])));
      }
    }
  }
}

}  // namespace apxinf::gemm::kernels
