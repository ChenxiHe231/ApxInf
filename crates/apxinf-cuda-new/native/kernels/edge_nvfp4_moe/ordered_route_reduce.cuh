// Experimental arithmetic policy only. Not wired into production/AOT/public API.
#pragma once
#include <cuda_runtime.h>
#include <cstddef>
#include <cstdint>

namespace apxinf::experimental::ordered_route_reduce {
constexpr unsigned kSlots = 8;
constexpr unsigned kThreads = 256;

// Preconditions: token_slot0_column addresses one column in a valid [T,8,H]
// BF16-bit buffer; hidden>0, all eight source words exist and are finite BF16.
// Explicit PTX add.rn.f32 (without FTZ) fixes each rounding point even when the
// enclosing translation unit uses --use_fast_math. volatile prevents reassociation.
static __device__ __forceinline__ float ordered_eight_sum(
    const uint16_t* token_slot0_column, size_t hidden) {
  float accumulator = __uint_as_float(0U);  // Explicit +0 initial state.
#pragma unroll
  for (unsigned slot = 0; slot < kSlots; ++slot) {
    const float value = __uint_as_float(
        static_cast<uint32_t>(token_slot0_column[slot * hidden]) << 16);
    float next;
    asm volatile("add.rn.f32 %0, %1, %2;"
                 : "=f"(next) : "f"(accumulator), "f"(value));
    accumulator = next;
  }
  return accumulator;
}

static __device__ __forceinline__ uint16_t bf16_rne(float value) {
  uint16_t bits;
  // No saturation or flush-to-zero. Overflow remains IEEE infinity.
  asm volatile("cvt.rn.bf16.f32 %0, %1;" : "=h"(bits) : "f"(value));
  return bits;
}

// Private experiment kernel. The caller validates T*H and8*T*H byte arithmetic,
// nonaliasing, capacities and lifetime before launch. Source is [token,slot,H],
// destination and optional diagnostic FP32 bits are [token,H]. No atomics.
// Rounded and raw writes each cover exactly live_elements, including tails.
static __global__ void ordered_route_reduce_kernel(
    const uint16_t* __restrict__ expanded,
    uint16_t* __restrict__ output,
    uint32_t* __restrict__ fp32_bits,
    size_t live_elements, size_t hidden) {
  const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= live_elements) return;
  const size_t token = index / hidden;
  const size_t column = index % hidden;
  const float value = ordered_eight_sum(expanded + token * kSlots * hidden + column, hidden);
  output[index] = bf16_rne(value);
  if (fp32_bits != nullptr) fp32_bits[index] = __float_as_uint(value);
}
}  // namespace apxinf::experimental::ordered_route_reduce

