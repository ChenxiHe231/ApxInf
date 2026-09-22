// Copyright 2026 ApxInf contributors.
//
// Gated DeltaNet recurrent step, the linear-attention half of Qwen3.5.
//
// IMPORTANT: the update rule below is implemented from the architecture's
// documented form. It has NOT been checked against a reference engine running
// this checkpoint, so it validates as "this kernel computes the stated
// recurrence" and not yet as "this recurrence is the model's". Layer-wise
// comparison against a reference is still required before any accuracy claim.
//
// Per value head h, with k-head kh = h / (num_v_heads / num_k_heads):
//
//   S <- S * exp(g)                  decay, g <= 0
//   m  = S @ k                       [v_dim], what the state already predicts
//   d  = (v - m) * beta              the delta rule's correction
//   S <- S + outer(d, k)             rank-1 update
//   o  = S @ q
//
// State is [v_heads, v_dim, k_dim] in f32, which the checkpoint's
// `mamba_ssm_dtype: float32` asks for.

#include "gdn_ops.h"

#include <cuda_bf16.h>

#include <cstdint>

namespace apxinf::cuda::gdn_ops {
namespace {

// One warp per state row, four rows per block, float4 loads.
//
// The recurrence has a dependency that looks like it forces two passes over
// each row: the delta needs `predicted = S @ k`, which needs the whole row,
// and only then can the row be updated. Reading the row twice costs 2x the
// state traffic, and at 151 MiB of state per token across 48 layers that is
// the dominant cost of this kernel.
//
// A warp can hold one row in registers instead: 32 lanes x float4 = 128
// elements. Decay and the S@k reduction happen on the registers, the warp
// reduces to `predicted`, the delta is applied to the same registers, and the
// row is written back once. One read, one write.
__global__ void recurrent_step_kernel(
    float* __restrict__ state, const __nv_bfloat16* __restrict__ q,
    const __nv_bfloat16* __restrict__ k, const __nv_bfloat16* __restrict__ v,
    const float* __restrict__ decay, const float* __restrict__ beta,
    __nv_bfloat16* __restrict__ output, int v_heads, int k_heads, int v_dim,
    int k_dim) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int global_row = blockIdx.x * (blockDim.x >> 5) + warp;
  const int total_rows = v_heads * v_dim;
  if (global_row >= total_rows) return;

  const int head = global_row / v_dim;
  const int v_index = global_row % v_dim;
  const int k_head = head / (v_heads / k_heads);

  // Each lane owns four consecutive elements of the row.
  const int chunk = lane * 4;
  if (chunk >= k_dim) return;

  float4* row = reinterpret_cast<float4*>(
      state + (long long)global_row * k_dim);
  float4 s = row[lane];

  const float head_decay = expf(decay[head]);
  const __nv_bfloat16* k_row = k + (long long)k_head * k_dim + chunk;
  const __nv_bfloat16* q_row = q + (long long)k_head * k_dim + chunk;

  const float k0 = __bfloat162float(k_row[0]);
  const float k1 = __bfloat162float(k_row[1]);
  const float k2 = __bfloat162float(k_row[2]);
  const float k3 = __bfloat162float(k_row[3]);

  s.x *= head_decay;
  s.y *= head_decay;
  s.z *= head_decay;
  s.w *= head_decay;

  float predicted = s.x * k0 + s.y * k1 + s.z * k2 + s.w * k3;
  for (int offset = 16; offset > 0; offset >>= 1) {
    predicted += __shfl_down_sync(0xFFFFFFFFu, predicted, offset);
  }
  predicted = __shfl_sync(0xFFFFFFFFu, predicted, 0);

  const float delta =
      (__bfloat162float(v[head * v_dim + v_index]) - predicted) * beta[head];

  s.x += delta * k0;
  s.y += delta * k1;
  s.z += delta * k2;
  s.w += delta * k3;
  row[lane] = s;

  float out = s.x * __bfloat162float(q_row[0]) +
              s.y * __bfloat162float(q_row[1]) +
              s.z * __bfloat162float(q_row[2]) +
              s.w * __bfloat162float(q_row[3]);
  for (int offset = 16; offset > 0; offset >>= 1) {
    out += __shfl_down_sync(0xFFFFFFFFu, out, offset);
  }
  if (lane == 0) {
    output[head * v_dim + v_index] = __float2bfloat16(out);
  }
}

// Gated RMSNorm over each value head, then the z gate.
//
// Qwen3.5 uses `output_gate_type: swish`, so the gate is silu(z) rather than
// the sigmoid some GDN variants use -- reusing a sigmoid-gated path would be
// silently wrong.
__global__ void gated_norm_kernel(const __nv_bfloat16* __restrict__ input,
                                  const __nv_bfloat16* __restrict__ gate,
                                  const __nv_bfloat16* __restrict__ weight,
                                  __nv_bfloat16* __restrict__ output,
                                  int heads, int head_dim, float epsilon) {
  const int head = blockIdx.x;
  if (head >= heads) return;
  const long long base = (long long)head * head_dim;

  float sum = 0.0f;
  for (int index = threadIdx.x; index < head_dim; index += blockDim.x) {
    const float value = __bfloat162float(input[base + index]);
    sum += value * value;
  }
  for (int offset = 16; offset > 0; offset >>= 1) {
    sum += __shfl_down_sync(0xFFFFFFFFu, sum, offset);
  }
  __shared__ float total;
  if (threadIdx.x == 0) total = sum;
  __syncthreads();

  const float scale = rsqrtf(total / static_cast<float>(head_dim) + epsilon);
  for (int index = threadIdx.x; index < head_dim; index += blockDim.x) {
    const float normalized = __bfloat162float(input[base + index]) * scale *
                             __bfloat162float(weight[index]);
    const float z = __bfloat162float(gate[base + index]);
    output[base + index] = __float2bfloat16(normalized * z / (1.0f + __expf(-z)));
  }
}

// Causal depthwise conv1d over the recurrent window, advancing the ring of
// past inputs. Decode sees one token at a time, so the window is state.
__global__ void causal_conv_step_kernel(
    float* __restrict__ window, const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ weight, __nv_bfloat16* __restrict__ output,
    int channels, int kernel_width) {
  const int channel = blockIdx.x * blockDim.x + threadIdx.x;
  if (channel >= channels) return;

  float* slot = window + (long long)channel * kernel_width;
  // Shift the window by one and append the new sample. kernel_width is 4 here,
  // so an explicit shift beats any ring-buffer indexing.
  for (int index = 0; index < kernel_width - 1; ++index) {
    slot[index] = slot[index + 1];
  }
  slot[kernel_width - 1] = __bfloat162float(input[channel]);

  float accumulator = 0.0f;
  for (int index = 0; index < kernel_width; ++index) {
    accumulator +=
        slot[index] * __bfloat162float(weight[channel * kernel_width + index]);
  }
  // The conv is followed by SiLU in this architecture.
  output[channel] =
      __float2bfloat16(accumulator / (1.0f + __expf(-accumulator)));
}

// L2-normalize each head of q and k, which the delta rule requires for
// stability.
__global__ void l2_normalize_heads_kernel(__nv_bfloat16* __restrict__ data,
                                          int heads, int head_dim,
                                          float epsilon) {
  const int head = blockIdx.x;
  if (head >= heads) return;
  const long long base = (long long)head * head_dim;

  float sum = 0.0f;
  for (int index = threadIdx.x; index < head_dim; index += blockDim.x) {
    const float value = __bfloat162float(data[base + index]);
    sum += value * value;
  }
  for (int offset = 16; offset > 0; offset >>= 1) {
    sum += __shfl_down_sync(0xFFFFFFFFu, sum, offset);
  }
  __shared__ float total;
  if (threadIdx.x == 0) total = sum;
  __syncthreads();

  const float scale = rsqrtf(total + epsilon);
  for (int index = threadIdx.x; index < head_dim; index += blockDim.x) {
    data[base + index] =
        __float2bfloat16(__bfloat162float(data[base + index]) * scale);
  }
}

// g = -exp(A_log) * softplus(a + dt_bias), the log decay; beta = sigmoid(b).
__global__ void decay_and_beta_kernel(const __nv_bfloat16* __restrict__ a,
                                      const __nv_bfloat16* __restrict__ b,
                                      const __nv_bfloat16* __restrict__ a_log,
                                      const __nv_bfloat16* __restrict__ dt_bias,
                                      float* __restrict__ decay,
                                      float* __restrict__ beta, int heads) {
  const int head = blockIdx.x * blockDim.x + threadIdx.x;
  if (head >= heads) return;
  const float shifted =
      __bfloat162float(a[head]) + __bfloat162float(dt_bias[head]);
  // softplus, guarded so large inputs do not overflow the exponential.
  const float softplus =
      shifted > 20.0f ? shifted : log1pf(__expf(shifted));
  decay[head] = -__expf(__bfloat162float(a_log[head])) * softplus;
  const float raw = __bfloat162float(b[head]);
  beta[head] = 1.0f / (1.0f + __expf(-raw));
}

}  // namespace

int gdn_recurrent_step(void* state, const void* q, const void* k,
                       const void* v, const void* decay, const void* beta,
                       void* output, int v_heads, int k_heads, int v_dim,
                       int k_dim, cudaStream_t stream) {
  if (v_heads <= 0 || k_heads <= 0 || v_dim <= 0 || k_dim <= 0) return -1;
  if (v_heads % k_heads != 0) return -2;
  // One warp holds a row as 32 lanes x float4, so k_dim must be exactly 128.
  // Rejecting is better than silently taking a slower general path.
  if (k_dim != 128) return -4;
  constexpr int kRowsPerBlock = 4;
  const int threads = kRowsPerBlock * 32;
  const int blocks = (v_heads * v_dim + kRowsPerBlock - 1) / kRowsPerBlock;
  recurrent_step_kernel<<<blocks, threads, 0, stream>>>(
      static_cast<float*>(state), static_cast<const __nv_bfloat16*>(q),
      static_cast<const __nv_bfloat16*>(k),
      static_cast<const __nv_bfloat16*>(v), static_cast<const float*>(decay),
      static_cast<const float*>(beta), static_cast<__nv_bfloat16*>(output),
      v_heads, k_heads, v_dim, k_dim);
  return cudaGetLastError() == cudaSuccess ? 0 : -3;
}

int gdn_gated_norm(const void* input, const void* gate, const void* weight,
                   void* output, int heads, int head_dim, float epsilon,
                   cudaStream_t stream) {
  if (heads <= 0 || head_dim <= 0) return -1;
  const int threads = head_dim >= 32 ? 32 : head_dim;
  gated_norm_kernel<<<heads, threads, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(input),
      static_cast<const __nv_bfloat16*>(gate),
      static_cast<const __nv_bfloat16*>(weight),
      static_cast<__nv_bfloat16*>(output), heads, head_dim, epsilon);
  return cudaGetLastError() == cudaSuccess ? 0 : -2;
}

int gdn_causal_conv_step(void* window, const void* input, const void* weight,
                         void* output, int channels, int kernel_width,
                         cudaStream_t stream) {
  if (channels <= 0 || kernel_width <= 0) return -1;
  const int threads = 256;
  const int blocks = (channels + threads - 1) / threads;
  causal_conv_step_kernel<<<blocks, threads, 0, stream>>>(
      static_cast<float*>(window), static_cast<const __nv_bfloat16*>(input),
      static_cast<const __nv_bfloat16*>(weight),
      static_cast<__nv_bfloat16*>(output), channels, kernel_width);
  return cudaGetLastError() == cudaSuccess ? 0 : -2;
}

int gdn_l2_normalize_heads(void* data, int heads, int head_dim, float epsilon,
                           cudaStream_t stream) {
  if (heads <= 0 || head_dim <= 0) return -1;
  const int threads = head_dim >= 32 ? 32 : head_dim;
  l2_normalize_heads_kernel<<<heads, threads, 0, stream>>>(
      static_cast<__nv_bfloat16*>(data), heads, head_dim, epsilon);
  return cudaGetLastError() == cudaSuccess ? 0 : -2;
}

int gdn_decay_and_beta(const void* a, const void* b, const void* a_log,
                       const void* dt_bias, void* decay, void* beta, int heads,
                       cudaStream_t stream) {
  if (heads <= 0) return -1;
  const int threads = 128;
  const int blocks = (heads + threads - 1) / threads;
  decay_and_beta_kernel<<<blocks, threads, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(a),
      static_cast<const __nv_bfloat16*>(b),
      static_cast<const __nv_bfloat16*>(a_log),
      static_cast<const __nv_bfloat16*>(dt_bias), static_cast<float*>(decay),
      static_cast<float*>(beta), heads);
  return cudaGetLastError() == cudaSuccess ? 0 : -2;
}

}  // namespace apxinf::cuda::gdn_ops
