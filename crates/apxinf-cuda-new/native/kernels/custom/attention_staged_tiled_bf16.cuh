#pragma once

#include "../primitives/bf16_pairwise.cuh"

namespace apxinf::attention::kernels::staged_tiled {
#include "../../vendor/sleef/sleefinline_cuda.h"

using apxinf::numerics::add_round_odd_native;

__device__ __forceinline__ bool span(const uint32_t* offsets, int segment,
                                   int tokens, int maximum,
                                   int& begin, int& count) {
  const uint32_t a = offsets[segment], b = offsets[segment + 1];
  if (b <= a || b > uint32_t(tokens) || b - a > uint32_t(maximum)) return false;
  begin = int(a);
  count = int(b - a);
  return true;
}

// Workspace is [segment*heads, query chunk, maximum keys]. F32 stores the
// BF16-rounded scores/probabilities exactly, preserving all staged boundaries.
__global__ void qk(const __nv_bfloat16* q, const __nv_bfloat16* k,
                   const uint32_t* offsets, float* p, int tokens, int maximum,
                   int heads, int dim, int first, int chunk, float scale) {
  const int head = blockIdx.y, segment = blockIdx.z;
  int begin, count;
  if (!span(offsets, segment, tokens, maximum, begin, count)) return;
  const int columns = (maximum + 31) / 32;
  const int tr = int(blockIdx.x) / columns, tc = int(blockIdx.x) % columns;
  if (first + tr * 16 >= count || tr * 16 >= chunk || tc * 32 >= count) return;
  const int tx = threadIdx.x, ty = threadIdx.y, t = ty * 16 + tx;
  __shared__ float a[16][65], b[32][65];
  float sum[2][2] = {{0.f, 0.f}, {0.f, 0.f}};
  for (int base = 0; base < dim; base += 64) {
    for (int i = t; i < 16 * 64; i += 128) {
      const int row = first + tr * 16 + i / 64, c = base + i % 64;
      a[i / 64][i % 64] = row < count && row < first + chunk && c < dim
          ? __bfloat162float(q[(int64_t(begin + row) * heads + head) * dim + c])
          : 0.f;
    }
    for (int i = t; i < 32 * 64; i += 128) {
      const int row = tc * 32 + i / 64, c = base + i % 64;
      b[i / 64][i % 64] = row < count && c < dim
          ? __bfloat162float(k[(int64_t(begin + row) * heads + head) * dim + c])
          : 0.f;
    }
    __syncthreads();
    #pragma unroll 4
    for (int i = 0; i < 64 && base + i < dim; i += 2) {
      #pragma unroll
      for (int rr = 0; rr < 2; ++rr) {
        #pragma unroll
        for (int cc = 0; cc < 2; ++cc) {
          // Products retain subnormal operands; only the pair additions flush.
          const float p0 = __fmul_rn(a[ty + rr * 8][i], b[tx + cc * 16][i]);
          const float p1 = __fmul_rn(a[ty + rr * 8][i + 1], b[tx + cc * 16][i + 1]);
          sum[rr][cc] = add_round_odd_native(
              sum[rr][cc], add_round_odd_native(p0, p1));
        }
      }
    }
    __syncthreads();
  }
  #pragma unroll
  for (int rr = 0; rr < 2; ++rr) {
    #pragma unroll
    for (int cc = 0; cc < 2; ++cc) {
      const int row = tr * 16 + ty + rr * 8, col = tc * 32 + tx + cc * 16;
      if (row < chunk && first + row < count && col < count) {
        const float rounded = __bfloat162float(__float2bfloat16(sum[rr][cc]));
        p[((int64_t(segment) * heads + head) * chunk + row) * maximum + col] =
            __bfloat162float(__float2bfloat16(__fmul_rn(rounded, scale)));
      }
    }
  }
}

__global__ void softmax(float* p, const uint32_t* offsets, int tokens,
                       int maximum, int heads, int first, int chunk) {
  const int row = blockIdx.x, head = blockIdx.y, segment = blockIdx.z;
  int begin, count;
  if (!span(offsets, segment, tokens, maximum, begin, count) || first + row >= count)
    return;
  float* data = p + ((int64_t(segment) * heads + head) * chunk + row) * maximum;
  extern __shared__ float scores[];
  __shared__ float maximum_score, reciprocal;
  for (int i = threadIdx.x; i < count; i += blockDim.x) scores[i] = data[i];
  __syncthreads();
  if (threadIdx.x == 0) {
    float v = -CUDART_INF_F;
    for (int i = 0; i < count; ++i) v = fmaxf(v, scores[i]);
    maximum_score = v;
  }
  __syncthreads();
  for (int i = threadIdx.x; i < count; i += blockDim.x)
    scores[i] = Sleef_expf1_u10cuda(__fsub_rn(scores[i], maximum_score));
  __syncthreads();
  // Each lane keeps its reference addition order: (s0+s2)+(s1+s3).
  if (threadIdx.x < 4) {
    float sum = 0.f;
    for (int i = threadIdx.x; i < count; i += 4) sum = __fadd_rn(sum, scores[i]);
    const float pair = __fadd_rn(sum, __shfl_xor_sync(0xf, sum, 2));
    const float other = __shfl_xor_sync(0xf, pair, 1);
    if (threadIdx.x == 0) reciprocal = __fdiv_rn(1.f, __fadd_rn(pair, other));
  }
  __syncthreads();
  for (int i = threadIdx.x; i < count; i += blockDim.x)
    data[i] = __bfloat162float(__float2bfloat16(__fmul_rn(scores[i], reciprocal)));
}

__global__ void pv(const float* p, const __nv_bfloat16* v,
                   const uint32_t* offsets, __nv_bfloat16* out,
                   int tokens, int maximum, int heads, int dim,
                   int first, int chunk) {
  const int head = blockIdx.y, segment = blockIdx.z;
  int begin, count;
  if (!span(offsets, segment, tokens, maximum, begin, count)) return;
  const int columns = (dim + 31) / 32;
  const int tr = int(blockIdx.x) / columns, tc = int(blockIdx.x) % columns;
  if (first + tr * 16 >= count || tr * 16 >= chunk) return;
  const int tx = threadIdx.x, ty = threadIdx.y, t = ty * 16 + tx;
  __shared__ float a[16][65], b[32][65];
  float sum[2][2] = {{0.f, 0.f}, {0.f, 0.f}};
  for (int base = 0; base < count; base += 64) {
    for (int i = t; i < 16 * 64; i += 128) {
      const int row = tr * 16 + i / 64, c = base + i % 64;
      a[i / 64][i % 64] = row < chunk && first + row < count && c < count
          ? p[((int64_t(segment) * heads + head) * chunk + row) * maximum + c]
          : 0.f;
    }
    for (int i = t; i < 64 * 32; i += 128) {
      const int d = tc * 32 + i % 32, c = base + i / 32;
      b[i % 32][i / 32] = c < count && d < dim
          ? __bfloat162float(v[(int64_t(begin + c) * heads + head) * dim + d])
          : 0.f;
    }
    __syncthreads();
    #pragma unroll 4
    for (int i = 0; i < 64 && base + i < count; i += 2) {
      #pragma unroll
      for (int rr = 0; rr < 2; ++rr) {
        #pragma unroll
        for (int cc = 0; cc < 2; ++cc) {
          const float p0 = __fmul_rn(a[ty + rr * 8][i], b[tx + cc * 16][i]);
          const float p1 = __fmul_rn(a[ty + rr * 8][i + 1], b[tx + cc * 16][i + 1]);
          sum[rr][cc] = add_round_odd_native(
              sum[rr][cc], add_round_odd_native(p0, p1));
        }
      }
    }
    __syncthreads();
  }
  #pragma unroll
  for (int rr = 0; rr < 2; ++rr) {
    #pragma unroll
    for (int cc = 0; cc < 2; ++cc) {
      const int row = tr * 16 + ty + rr * 8, d = tc * 32 + tx + cc * 16;
      if (row < chunk && first + row < count && d < dim)
        out[(int64_t(begin + first + row) * heads + head) * dim + d] =
            __float2bfloat16(sum[rr][cc]);
    }
  }
}
}  // namespace apxinf::attention::kernels::staged_tiled
