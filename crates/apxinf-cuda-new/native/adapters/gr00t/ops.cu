// Copyright 2026 ApxInf contributors.
// GR00T-family fused activation primitives, ported verbatim from the legacy
// apxinf-cuda adapters (core_kernels_adapter.cu, custom_kernels.cu,
// static_bf16_adapter.cu) and kernels/custom/activation.cuh. Symbols carry the
// apxinf_gr00t_ prefix because the legacy crate links into the same binary and
// its own launchers are already present. The copied kernels live in an
// anonymous namespace so their names cannot collide with the identically named
// kernels that cuda-new already builds in pointwise.cu / mlp_ops.cu.

#include "../../include/apxinf_cuda/gr00t.h"
#include "../../kernels/custom/reduction.cuh"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace {

constexpr int kGr00tBlockSize = 256;

__device__ __forceinline__ float gelu_tanh(float value) {
  constexpr float kAlpha = 0.7978845608028654f;
  return 0.5f * value *
         (1.0f + tanhf(kAlpha * (value + 0.044715f * value * value * value)));
}

struct alignas(8) Bf16x4 {
  __nv_bfloat162 low;
  __nv_bfloat162 high;
};

struct alignas(16) Bf16Pairx8 {
  Bf16x4 first;
  Bf16x4 second;
};

// Separate-input variant used when gate/up projections are not packed.  The
// intermediate SiLU value is explicitly rounded to BF16 so this remains
// bit-compatible with the former `silu` then `mul` kernel sequence.
__global__ void silu_mul_separate_bf16_kernel(
    const __nv_bfloat16* gate, const __nv_bfloat16* up,
    __nv_bfloat16* output, uint32_t count) {
  const uint32_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= count) return;
  const float g = __bfloat162float(gate[gid]);
  const __nv_bfloat16 activated = __float2bfloat16(g / (1.0f + expf(-g)));
  output[gid] = __float2bfloat16(
      __bfloat162float(activated) * __bfloat162float(up[gid]));
}

__global__ void silu_mul_separate_bf16_packed4_kernel(
    const Bf16x4* gate, const Bf16x4* up, Bf16x4* output,
    uint32_t quad_count) {
  uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t stride = blockDim.x * gridDim.x;
  for (; index < quad_count; index += stride) {
    const Bf16x4 g = gate[index];
    const Bf16x4 u = up[index];
    float values[4] = {
        __bfloat162float(g.low.x), __bfloat162float(g.low.y),
        __bfloat162float(g.high.x), __bfloat162float(g.high.y)};
    const float ups[4] = {
        __bfloat162float(u.low.x), __bfloat162float(u.low.y),
        __bfloat162float(u.high.x), __bfloat162float(u.high.y)};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const __nv_bfloat16 activated =
          __float2bfloat16(values[i] / (1.0f + expf(-values[i])));
      values[i] = __bfloat162float(activated) * ups[i];
    }
    output[index] = Bf16x4{
        __floats2bfloat162_rn(values[0], values[1]),
        __floats2bfloat162_rn(values[2], values[3])};
  }
}

// Preserve the existing three-stage numerical contract while avoiding the
// two BF16 intermediates: SiLU rounds to BF16, multiplication rounds to BF16,
// then the calibrated value converts to saturating E4M3.
__global__ void silu_mul_quant_bf16_e4m3_kernel(
    const __nv_bfloat16* gate, const __nv_bfloat16* up,
    __nv_fp8_e4m3* output, int64_t count, float inverse_scale) {
  int64_t pair = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t pair_count = count / 2;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (; pair < pair_count; pair += stride) {
    const int64_t index = pair * 2;
    const __nv_bfloat162 gate2 =
        reinterpret_cast<const __nv_bfloat162*>(gate)[pair];
    const __nv_bfloat162 up2 =
        reinterpret_cast<const __nv_bfloat162*>(up)[pair];
    const float gx = __bfloat162float(gate2.x);
    const float gy = __bfloat162float(gate2.y);
    const __nv_bfloat16 sx =
        __float2bfloat16(gx / (1.0f + expf(-gx)));
    const __nv_bfloat16 sy =
        __float2bfloat16(gy / (1.0f + expf(-gy)));
    const __nv_bfloat16 px = __float2bfloat16(
        __bfloat162float(sx) * __bfloat162float(up2.x));
    const __nv_bfloat16 py = __float2bfloat16(
        __bfloat162float(sy) * __bfloat162float(up2.y));
    reinterpret_cast<__nv_fp8x2_e4m3*>(output)[pair] =
        __nv_fp8x2_e4m3(make_float2(
            __bfloat162float(px) * inverse_scale,
            __bfloat162float(py) * inverse_scale));
  }
  if (count % 2 != 0 && pair == pair_count) {
    const float g = __bfloat162float(gate[count - 1]);
    const __nv_bfloat16 s = __float2bfloat16(g / (1.0f + expf(-g)));
    const __nv_bfloat16 product = __float2bfloat16(
        __bfloat162float(s) * __bfloat162float(up[count - 1]));
    output[count - 1] = static_cast<__nv_fp8_e4m3>(
        __bfloat162float(product) * inverse_scale);
  }
}

__global__ void bias_gelu_quant_bf16_e4m3_packed4_kernel(
    const __nv_bfloat16* input, const __nv_bfloat16* bias,
    __nv_fp8_e4m3* output, int64_t quad_count, int cols,
    float inverse_scale) {
  int64_t quad_index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  const int quads_per_row = cols / 4;
  const Bf16x4* input4 = reinterpret_cast<const Bf16x4*>(input);
  const Bf16x4* bias4 = reinterpret_cast<const Bf16x4*>(bias);
  for (; quad_index < quad_count; quad_index += stride) {
    const Bf16x4 x = input4[quad_index];
    const Bf16x4 b = bias4[quad_index % quads_per_row];
    const float y0 = __bfloat162float(__float2bfloat16(gelu_tanh(
        __bfloat162float(x.low.x) + __bfloat162float(b.low.x))));
    const float y1 = __bfloat162float(__float2bfloat16(gelu_tanh(
        __bfloat162float(x.low.y) + __bfloat162float(b.low.y))));
    const float y2 = __bfloat162float(__float2bfloat16(gelu_tanh(
        __bfloat162float(x.high.x) + __bfloat162float(b.high.x))));
    const float y3 = __bfloat162float(__float2bfloat16(gelu_tanh(
        __bfloat162float(x.high.y) + __bfloat162float(b.high.y))));
    const __nv_fp8x2_e4m3 first(make_float2(
        y0 * inverse_scale, y1 * inverse_scale));
    const __nv_fp8x2_e4m3 second(make_float2(
        y2 * inverse_scale, y3 * inverse_scale));
    reinterpret_cast<uint32_t*>(output)[quad_index] =
        static_cast<uint32_t>(first.__x) |
        (static_cast<uint32_t>(second.__x) << 16);
  }
}

__device__ __forceinline__ Bf16x4 bias_gelu_bf16_quad(
    const Bf16x4 input, const Bf16x4 bias) {
  const float v0 = gelu_tanh(
      __bfloat162float(input.low.x) + __bfloat162float(bias.low.x));
  const float v1 = gelu_tanh(
      __bfloat162float(input.low.y) + __bfloat162float(bias.low.y));
  const float v2 = gelu_tanh(
      __bfloat162float(input.high.x) + __bfloat162float(bias.high.x));
  const float v3 = gelu_tanh(
      __bfloat162float(input.high.y) + __bfloat162float(bias.high.y));
  return Bf16x4{
      __floats2bfloat162_rn(v0, v1), __floats2bfloat162_rn(v2, v3)};
}

__global__ void bias_gelu_bf16_packed8_kernel(
    const Bf16Pairx8* input, const Bf16Pairx8* bias, Bf16Pairx8* output,
    int64_t octet_count, int octets_per_row) {
  int64_t index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (; index < octet_count; index += stride) {
    const Bf16Pairx8 value = input[index];
    const Bf16Pairx8 row_bias = bias[index % octets_per_row];
    output[index] = Bf16Pairx8{
        bias_gelu_bf16_quad(value.first, row_bias.first),
        bias_gelu_bf16_quad(value.second, row_bias.second)};
  }
}

// ── Elementwise / selection primitives ───────────────────────────────────
// Copied verbatim from the legacy kernels/custom/elementwise.cuh and
// kernels/custom/activation.cuh.

__global__ void scatter_rows_bf16_kernel(
    const __nv_bfloat16* source, const uint32_t* rows,
    __nv_bfloat16* output, int64_t count, int cols, bool add) {
  int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (; index < count; index += stride) {
    const int source_row = static_cast<int>(index / cols);
    const int column = static_cast<int>(index % cols);
    const int64_t output_index =
        static_cast<int64_t>(rows[source_row]) * cols + column;
    if (add) {
      output[output_index] = __float2bfloat16(
          __bfloat162float(output[output_index]) + __bfloat162float(source[index]));
    } else {
      output[output_index] = source[index];
    }
  }
}

__global__ void bias_qkv_in_place_bf16_packed4_kernel(
    __nv_bfloat16* query, __nv_bfloat16* key, __nv_bfloat16* value,
    const __nv_bfloat16* query_bias, const __nv_bfloat16* key_bias,
    const __nv_bfloat16* value_bias, int64_t group_count,
    int groups_per_row) {
  int64_t group = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t stride = static_cast<int64_t>(blockDim.x) * gridDim.x;
  for (; group < group_count; group += stride) {
    const int bias_group = static_cast<int>(group % groups_per_row);
    const __nv_bfloat162* qb = reinterpret_cast<const __nv_bfloat162*>(query_bias) + 2 * bias_group;
    const __nv_bfloat162* kb = reinterpret_cast<const __nv_bfloat162*>(key_bias) + 2 * bias_group;
    const __nv_bfloat162* vb = reinterpret_cast<const __nv_bfloat162*>(value_bias) + 2 * bias_group;
    __nv_bfloat162* q = reinterpret_cast<__nv_bfloat162*>(query) + 2 * group;
    __nv_bfloat162* k = reinterpret_cast<__nv_bfloat162*>(key) + 2 * group;
    __nv_bfloat162* v = reinterpret_cast<__nv_bfloat162*>(value) + 2 * group;
    const float2 q0 = __bfloat1622float2(q[0]);
    const float2 q1 = __bfloat1622float2(q[1]);
    const float2 k0 = __bfloat1622float2(k[0]);
    const float2 k1 = __bfloat1622float2(k[1]);
    const float2 v0 = __bfloat1622float2(v[0]);
    const float2 v1 = __bfloat1622float2(v[1]);
    const float2 qb0 = __bfloat1622float2(qb[0]);
    const float2 qb1 = __bfloat1622float2(qb[1]);
    const float2 kb0 = __bfloat1622float2(kb[0]);
    const float2 kb1 = __bfloat1622float2(kb[1]);
    const float2 vb0 = __bfloat1622float2(vb[0]);
    const float2 vb1 = __bfloat1622float2(vb[1]);
    q[0] = __floats2bfloat162_rn(q0.x + qb0.x, q0.y + qb0.y);
    q[1] = __floats2bfloat162_rn(q1.x + qb1.x, q1.y + qb1.y);
    k[0] = __floats2bfloat162_rn(k0.x + kb0.x, k0.y + kb0.y);
    k[1] = __floats2bfloat162_rn(k1.x + kb1.x, k1.y + kb1.y);
    v[0] = __floats2bfloat162_rn(v0.x + vb0.x, v0.y + vb0.y);
    v[1] = __floats2bfloat162_rn(v1.x + vb1.x, v1.y + vb1.y);
  }
}

// ── Norm kernels (BF16-input variants) ───────────────────────────────────
// cuda-new's own norm kernels take F16 input for the E4M3-output forms; the
// gr00t path needs BF16-input versions, so these are copied verbatim from the
// legacy normalization.cuh. `block_sum_parallel_unsafe` is cuda-new's own
// reduction helper (reduction.cuh), identical to the legacy one.

__global__ void gr00t_adaptive_layer_norm_bf16_kernel(
    const __nv_bfloat16* input, const __nv_bfloat16* modulation,
    __nv_bfloat16* output, uint32_t rows, uint32_t cols, float eps) {
  __shared__ float scratch[16];
  const uint32_t row = blockIdx.x;
  if (row >= rows) return;

  float sum = 0.0f;
  for (uint32_t col = threadIdx.x; col < cols; col += blockDim.x)
    sum += __bfloat162float(input[(uint64_t)row * cols + col]);
  const float mean = block_sum_parallel_unsafe(sum, scratch) / cols;

  float variance_sum = 0.0f;
  for (uint32_t col = threadIdx.x; col < cols; col += blockDim.x) {
    const float centered =
        __bfloat162float(input[(uint64_t)row * cols + col]) - mean;
    variance_sum += centered * centered;
  }
  __syncthreads();
  const float inverse_std =
      rsqrtf(block_sum_parallel_unsafe(variance_sum, scratch) / cols + eps);

  for (uint32_t col = threadIdx.x; col < cols; col += blockDim.x) {
    const uint64_t index = (uint64_t)row * cols + col;
    const float normalized =
        (__bfloat162float(input[index]) - mean) * inverse_std;
    const float scale = __bfloat162float(modulation[col]);
    const float shift = __bfloat162float(modulation[cols + col]);
    output[index] = __float2bfloat16(normalized * (1.0f + scale) + shift);
  }
}

__global__ void gr00t_adaptive_layer_norm_quant_bf16_e4m3_kernel(
    const __nv_bfloat16* input, const __nv_bfloat16* modulation,
    __nv_bfloat16* output, __nv_fp8_e4m3* quantized, uint32_t rows,
    uint32_t cols, float eps, float inverse_scale) {
  __shared__ float scratch[16];
  const uint32_t row = blockIdx.x;
  if (row >= rows) return;

  float sum = 0.0f;
  for (uint32_t col = threadIdx.x; col < cols; col += blockDim.x)
    sum += __bfloat162float(input[(uint64_t)row * cols + col]);
  const float mean = block_sum_parallel_unsafe(sum, scratch) / cols;

  float variance_sum = 0.0f;
  for (uint32_t col = threadIdx.x; col < cols; col += blockDim.x) {
    const float centered =
        __bfloat162float(input[(uint64_t)row * cols + col]) - mean;
    variance_sum += centered * centered;
  }
  __syncthreads();
  const float inverse_std =
      rsqrtf(block_sum_parallel_unsafe(variance_sum, scratch) / cols + eps);

  for (uint32_t col = threadIdx.x; col < cols; col += blockDim.x) {
    const uint64_t index = (uint64_t)row * cols + col;
    const float normalized =
        (__bfloat162float(input[index]) - mean) * inverse_std;
    const float scale = __bfloat162float(modulation[col]);
    const float shift = __bfloat162float(modulation[cols + col]);
    const __nv_bfloat16 rounded =
        __float2bfloat16(normalized * (1.0f + scale) + shift);
    output[index] = rounded;
    float value = __bfloat162float(rounded) * inverse_scale;
    value = fminf(448.0f, fmaxf(-448.0f, value));
    quantized[index] = static_cast<__nv_fp8_e4m3>(value);
  }
}

__global__ void gr00t_rms_norm_quant_bf16_e4m3_kernel(
    const __nv_bfloat16* input, const __nv_bfloat16* weight,
    __nv_fp8_e4m3* output, int rows, int cols, float eps,
    float inverse_scale) {
  __shared__ float scratch[8];
  const int row = blockIdx.x;
  float square_sum = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const float value =
        __bfloat162float(input[static_cast<int64_t>(row) * cols + col]);
    square_sum += value * value;
  }
  const float inverse_rms =
      rsqrtf(block_sum_parallel_unsafe(square_sum, scratch) / cols + eps);
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const int64_t index = static_cast<int64_t>(row) * cols + col;
    float value = __bfloat162float(input[index]) * inverse_rms *
                  __bfloat162float(weight[col]) * inverse_scale;
    value = fminf(448.0f, fmaxf(-448.0f, value));
    output[index] = static_cast<__nv_fp8_e4m3>(value);
  }
}

__global__ void gr00t_layer_norm_quant_bf16_e4m3_kernel(
    const __nv_bfloat16* input, const __nv_bfloat16* weight,
    const __nv_bfloat16* bias, __nv_fp8_e4m3* output, int rows, int cols,
    float eps, float inverse_scale) {
  extern __shared__ float x_buf[];
  __shared__ float scratch[16];
  const int row = blockIdx.x;
  if (row >= rows) return;
  const int offset = row * cols;

  float sum = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const float value = __bfloat162float(input[offset + col]);
    x_buf[col] = value;
    sum += value;
  }
  const float mean = block_sum_parallel_unsafe(sum, scratch) / cols;

  float variance_sum = 0.0f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    const float centered = x_buf[col] - mean;
    variance_sum += centered * centered;
  }
  __syncthreads();
  const float inverse_std =
      rsqrtf(block_sum_parallel_unsafe(variance_sum, scratch) / cols + eps);

  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    float value = (x_buf[col] - mean) * inverse_std;
    value = value * __bfloat162float(weight[col]) +
            __bfloat162float(bias[col]);
    value = __bfloat162float(__float2bfloat16(value));
    value = fminf(448.0f, fmaxf(-448.0f, value * inverse_scale));
    output[offset + col] = static_cast<__nv_fp8_e4m3>(value);
  }
}

}  // namespace

extern "C" cudaError_t apxinf_gr00t_silu_mul_separate_bf16(
    const void* gate, const void* up, void* output, uint32_t count,
    cudaStream_t stream) {
  const bool packed =
      (count & 3U) == 0 &&
      (reinterpret_cast<uintptr_t>(gate) & 7U) == 0 &&
      (reinterpret_cast<uintptr_t>(up) & 7U) == 0 &&
      (reinterpret_cast<uintptr_t>(output) & 7U) == 0;
  if (packed) {
    const uint32_t quad_count = count / 4;
    int blocks = static_cast<int>(
        (quad_count + kGr00tBlockSize - 1) / kGr00tBlockSize);
    blocks = blocks > 512 ? 512 : blocks;
    silu_mul_separate_bf16_packed4_kernel<<<blocks, kGr00tBlockSize, 0,
        stream>>>(
        (const Bf16x4*)gate, (const Bf16x4*)up, (Bf16x4*)output, quad_count);
    return cudaGetLastError();
  }
  dim3 grid((count + kGr00tBlockSize - 1) / kGr00tBlockSize, 1, 1);
  dim3 block(kGr00tBlockSize, 1, 1);
  silu_mul_separate_bf16_kernel<<<grid, block, 0, stream>>>(
      (const __nv_bfloat16*)gate, (const __nv_bfloat16*)up,
      (__nv_bfloat16*)output, count);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_gr00t_silu_mul_quant_bf16_e4m3(
    const void* gate, const void* up, void* output, int64_t count,
    float scale, cudaStream_t stream) {
  if (gate == nullptr || up == nullptr || output == nullptr || count <= 0 ||
      !(scale > 0.0f)) return cudaErrorInvalidValue;
  constexpr int threads = 256;
  const int64_t pairs = (count + 1) / 2;
  int blocks = static_cast<int>((pairs + threads - 1) / threads);
  blocks = blocks > 1024 ? 1024 : blocks;
  silu_mul_quant_bf16_e4m3_kernel<<<blocks, threads, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(gate),
      static_cast<const __nv_bfloat16*>(up),
      static_cast<__nv_fp8_e4m3*>(output), count, 1.0f / scale);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_gr00t_bias_gelu_quant_bf16_e4m3(
    const void* input, const void* bias, void* output, int rows, int cols,
    float scale, cudaStream_t stream) {
  if (!input || !bias || !output || rows <= 0 || cols <= 0 || cols % 4 != 0 ||
      !(scale > 0.0f) || reinterpret_cast<uintptr_t>(input) % alignof(Bf16x4) != 0 ||
      reinterpret_cast<uintptr_t>(bias) % alignof(Bf16x4) != 0 ||
      reinterpret_cast<uintptr_t>(output) % alignof(uint32_t) != 0)
    return cudaErrorInvalidValue;
  constexpr int threads = 256;
  const int64_t quad_count = static_cast<int64_t>(rows) * cols / 4;
  int blocks = static_cast<int>((quad_count + threads - 1) / threads);
  blocks = blocks > 1024 ? 1024 : blocks;
  bias_gelu_quant_bf16_e4m3_packed4_kernel<<<blocks, threads, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(input),
      static_cast<const __nv_bfloat16*>(bias),
      static_cast<__nv_fp8_e4m3*>(output), quad_count, cols, 1.0f / scale);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_gr00t_bias_gelu_bf16_packed8(
    const void* input, const void* bias, void* output, int rows, int cols,
    cudaStream_t stream) {
  if (input == nullptr || bias == nullptr || output == nullptr || rows <= 0 ||
      cols <= 0 || cols % 8 != 0 ||
      reinterpret_cast<uintptr_t>(input) % alignof(Bf16Pairx8) != 0 ||
      reinterpret_cast<uintptr_t>(bias) % alignof(Bf16Pairx8) != 0 ||
      reinterpret_cast<uintptr_t>(output) % alignof(Bf16Pairx8) != 0)
    return cudaErrorInvalidValue;
  const int64_t octet_count = static_cast<int64_t>(rows) * cols / 8;
  const int threads = rows >= 512 ? 256 : 128;
  const int blocks = static_cast<int>((octet_count + threads - 1) / threads);
  bias_gelu_bf16_packed8_kernel<<<blocks, threads, 0, stream>>>(
      static_cast<const Bf16Pairx8*>(input), static_cast<const Bf16Pairx8*>(bias),
      static_cast<Bf16Pairx8*>(output), octet_count, cols / 8);
  return cudaGetLastError();
}

// ── Elementwise / selection launchers ────────────────────────────────────
// Copied verbatim from the legacy static_bf16_adapter.cu, renamed with the
// apxinf_gr00t_ prefix.

extern "C" cudaError_t apxinf_gr00t_scatter_rows_bf16(
    const void* source, const void* rows, void* output,
    int row_count, int cols, int add, cudaStream_t stream) {
  if (source == nullptr || rows == nullptr || output == nullptr ||
      row_count <= 0 || cols <= 0 || (add != 0 && add != 1))
    return cudaErrorInvalidValue;
  const int64_t count = static_cast<int64_t>(row_count) * cols;
  const int blocks = static_cast<int>((count + kGr00tBlockSize - 1) / kGr00tBlockSize);
  scatter_rows_bf16_kernel<<<blocks, kGr00tBlockSize, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(source),
      static_cast<const uint32_t*>(rows),
      static_cast<__nv_bfloat16*>(output), count, cols, add != 0);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_gr00t_bias_qkv_in_place_bf16(
    void* query, void* key, void* value, const void* query_bias,
    const void* key_bias, const void* value_bias, int rows, int cols,
    cudaStream_t stream) {
  if (!query || !key || !value || !query_bias || !key_bias || !value_bias ||
      rows <= 0 || cols <= 0 || cols % 4 != 0)
    return cudaErrorInvalidValue;
  constexpr int threads = 256;
  const int64_t groups = static_cast<int64_t>(rows) * cols / 4;
  int blocks = static_cast<int>((groups + threads - 1) / threads);
  blocks = blocks > 1024 ? 1024 : blocks;
  bias_qkv_in_place_bf16_packed4_kernel<<<blocks, threads, 0, stream>>>(
      static_cast<__nv_bfloat16*>(query),
      static_cast<__nv_bfloat16*>(key),
      static_cast<__nv_bfloat16*>(value),
      static_cast<const __nv_bfloat16*>(query_bias),
      static_cast<const __nv_bfloat16*>(key_bias),
      static_cast<const __nv_bfloat16*>(value_bias), groups, cols / 4);
  return cudaGetLastError();
}

// ── Norm launchers ───────────────────────────────────────────────────────

extern "C" cudaError_t apxinf_gr00t_adaptive_layer_norm_bf16(
    const void* input, const void* modulation, void* output, uint32_t rows,
    uint32_t cols, float eps, cudaStream_t stream) {
  if (rows == 0 || cols == 0 || !(eps > 0.0f)) return cudaErrorInvalidValue;
  gr00t_adaptive_layer_norm_bf16_kernel<<<rows, kGr00tBlockSize, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(input),
      static_cast<const __nv_bfloat16*>(modulation),
      static_cast<__nv_bfloat16*>(output), rows, cols, eps);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_gr00t_adaptive_layer_norm_quant_bf16_e4m3(
    const void* input, const void* modulation, void* output, void* quantized,
    uint32_t rows, uint32_t cols, float eps, float scale, cudaStream_t stream) {
  if (rows == 0 || cols == 0 || !(eps > 0.0f) || !(scale > 0.0f))
    return cudaErrorInvalidValue;
  gr00t_adaptive_layer_norm_quant_bf16_e4m3_kernel<<<rows, kGr00tBlockSize, 0,
                                                     stream>>>(
      static_cast<const __nv_bfloat16*>(input),
      static_cast<const __nv_bfloat16*>(modulation),
      static_cast<__nv_bfloat16*>(output),
      static_cast<__nv_fp8_e4m3*>(quantized), rows, cols, eps, 1.0f / scale);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_gr00t_rms_norm_quant_bf16_e4m3(
    const void* input, const void* weight, void* output, int rows, int cols,
    float eps, float scale, cudaStream_t stream) {
  if (input == nullptr || weight == nullptr || output == nullptr || rows <= 0 ||
      cols <= 0 || !std::isfinite(scale) || scale <= 0.0f)
    return cudaErrorInvalidValue;
  gr00t_rms_norm_quant_bf16_e4m3_kernel<<<rows, kGr00tBlockSize, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(input),
      static_cast<const __nv_bfloat16*>(weight),
      static_cast<__nv_fp8_e4m3*>(output), rows, cols, eps, 1.0f / scale);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_gr00t_layer_norm_quant_bf16_e4m3(
    const void* input, const void* weight, const void* bias, void* output,
    int rows, int cols, float eps, float scale, cudaStream_t stream) {
  if (input == nullptr || weight == nullptr || bias == nullptr ||
      output == nullptr || rows <= 0 || cols <= 0 || !std::isfinite(scale) ||
      scale <= 0.0f)
    return cudaErrorInvalidValue;
  const size_t shared_bytes = static_cast<size_t>(cols) * sizeof(float);
  gr00t_layer_norm_quant_bf16_e4m3_kernel<<<rows, kGr00tBlockSize,
                                            shared_bytes, stream>>>(
      static_cast<const __nv_bfloat16*>(input),
      static_cast<const __nv_bfloat16*>(weight),
      static_cast<const __nv_bfloat16*>(bias),
      static_cast<__nv_fp8_e4m3*>(output), rows, cols, eps, 1.0f / scale);
  return cudaGetLastError();
}
