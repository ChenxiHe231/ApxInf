#pragma once

#include "../primitives/bf16_pairwise.cuh"

namespace apxinf::gemm::kernels {

// NK BF16 weights, a single ordered K reduction and a final FP32 bias addition.
// Shared tiles preserve coalesced reads without changing each dot's arithmetic.
__global__ void pairwise_bf16_bias(
    const __nv_bfloat16* a, const __nv_bfloat16* b,
    const __nv_bfloat16* bias, __nv_bfloat16* out, int m, int n, int k) {
  const int n_tiles = int((int64_t(n) + 15) / 16);
  const int tile_row = int(blockIdx.x) / n_tiles;
  const int tile_column = int(blockIdx.x) % n_tiles;
  const int column = tile_column * 16 + threadIdx.x;
  const int row = tile_row * 8 + threadIdx.y;
  const int thread = threadIdx.y * 16 + threadIdx.x;
  __shared__ float x[8][64], w[16][64];
  float sum = 0.f;
  for (int64_t begin = 0; begin < k; begin += 64) {
    for (int i = thread; i < 8 * 64; i += 128) {
      const int r = tile_row * 8 + i / 64;
      const int64_t c = begin + i % 64;
      x[i / 64][i % 64] = r < m && c < k
          ? __bfloat162float(a[int64_t(r) * k + c]) : 0.f;
    }
    for (int i = thread; i < 16 * 64; i += 128) {
      const int r = tile_column * 16 + i / 64;
      const int64_t c = begin + i % 64;
      w[i / 64][i % 64] = r < n && c < k
          ? __bfloat162float(b[int64_t(r) * k + c]) : 0.f;
    }
    __syncthreads();
    for (int i = 0; i < 64 && begin + i < k; i += 2) {
      const float p0 = numerics::bf16_product(x[threadIdx.y][i], w[threadIdx.x][i]);
      const float p1 = numerics::bf16_product(x[threadIdx.y][i + 1], w[threadIdx.x][i + 1]);
      sum = numerics::add_round_odd(sum, numerics::add_round_odd(p0, p1));
    }
    __syncthreads();
  }
  if (row < m && column < n) {
    out[int64_t(row) * n + column] =
        __float2bfloat16(__fadd_rn(sum, __bfloat162float(bias[column])));
  }
}

}  // namespace apxinf::gemm::kernels
