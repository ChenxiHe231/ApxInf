#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cfloat>

namespace apxinf::numerics {

// Standard BF16 dot arithmetic flushes subnormal operands/products and rounds
// each pair sum and accumulator update to odd FP32. Directed FP32 bounds differ
// exactly when the sum is inexact, including operand exponent gaps above 53.
// No wider accumulator changes the declared FP32 reduction order.
__device__ __forceinline__ float flush_subnormal(float value) {
  unsigned bits = __float_as_uint(value);
  return (bits & 0x7f800000u) == 0
      ? __uint_as_float(bits & 0x80000000u) : value;
}

__device__ __forceinline__ float add_round_odd(float a, float b) {
  a = flush_subnormal(a);
  b = flush_subnormal(b);
  float result = __fadd_rz(a, b);
  if (isnan(result)) return __uint_as_float(0x7fc00000u);
  const bool inexact = __fadd_ru(a, b) != __fadd_rd(a, b);
  if (inexact) result = __uint_as_float(__float_as_uint(result) | 1u);
  return flush_subnormal(result);
}

__device__ __forceinline__ float bf16_product(float a, float b) {
  return flush_subnormal(__fmul_rn(flush_subnormal(a), flush_subnormal(b)));
}

}  // namespace apxinf::numerics
