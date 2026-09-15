#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math_constants.h>

namespace apxinf::attention::kernels {

template <class T>
__device__ float to_float(T value);

template <>
__device__ inline float to_float(__half value) {
  return __half2float(value);
}

template <>
__device__ inline float to_float(__nv_bfloat16 value) {
  return __bfloat162float(value);
}

template <class T>
__device__ T from_float(float value);

template <>
__device__ inline __half from_float(float value) {
  return __float2half(value);
}

template <>
__device__ inline __nv_bfloat16 from_float(float value) {
  return __float2bfloat16(value);
}

// Correctness fallback. One thread owns one softmax row, keeping the
// implementation shape-general while optimized providers handle throughput.
template <class T>
__global__ void attention_scores_softmax(
    const T* query, const T* key, float* probabilities, int query_tokens,
    int key_tokens, int query_heads, int kv_heads, int head_dim, int batch_size,
    bool causal, float scale) {
  const int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t rows =
      static_cast<int64_t>(batch_size) * query_tokens * query_heads;
  if (row >= rows) return;

  const int q_head = row % query_heads;
  const int query_token = (row / query_heads) % query_tokens;
  const int batch = row / (static_cast<int64_t>(query_heads) * query_tokens);
  const int kv_head = q_head / (query_heads / kv_heads);
  const int valid_keys = causal
      ? min(key_tokens, query_token + key_tokens - query_tokens + 1)
      : key_tokens;
  const T* q = query +
      ((static_cast<int64_t>(batch) * query_tokens + query_token) * query_heads +
       q_head) * head_dim;
  float* scores = probabilities + row * key_tokens;

  float maximum = -CUDART_INF_F;
  for (int key_token = 0; key_token < valid_keys; ++key_token) {
    const T* k = key +
        ((static_cast<int64_t>(batch) * key_tokens + key_token) * kv_heads +
         kv_head) * head_dim;
    float dot = 0.0F;
    for (int dimension = 0; dimension < head_dim; ++dimension) {
      dot += to_float(q[dimension]) * to_float(k[dimension]);
    }
    scores[key_token] = dot * scale;
    maximum = fmaxf(maximum, scores[key_token]);
  }
  float sum = 0.0F;
  for (int key_token = 0; key_token < valid_keys; ++key_token) {
    const float value = expf(scores[key_token] - maximum);
    scores[key_token] = value;
    sum += value;
  }
  for (int key_token = 0; key_token < valid_keys; ++key_token) {
    scores[key_token] /= sum;
  }
  for (int key_token = valid_keys; key_token < key_tokens; ++key_token) {
    scores[key_token] = 0.0F;
  }
}

template <class T>
__global__ void attention_values(const float* probabilities, const T* value,
                                 T* output, int query_tokens, int key_tokens,
                                 int query_heads, int kv_heads, int head_dim,
                                 int batch_size) {
  const int64_t index =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int64_t elements = static_cast<int64_t>(batch_size) * query_tokens *
                           query_heads * head_dim;
  if (index >= elements) return;

  const int dimension = index % head_dim;
  const int q_head = (index / head_dim) % query_heads;
  const int query_token =
      (index / (static_cast<int64_t>(head_dim) * query_heads)) % query_tokens;
  const int batch = index /
      (static_cast<int64_t>(head_dim) * query_heads * query_tokens);
  const int kv_head = q_head / (query_heads / kv_heads);
  const int64_t row =
      (static_cast<int64_t>(batch) * query_tokens + query_token) * query_heads +
      q_head;
  const float* weights = probabilities + row * key_tokens;
  float result = 0.0F;
  for (int key_token = 0; key_token < key_tokens; ++key_token) {
    const T* v = value +
        ((static_cast<int64_t>(batch) * key_tokens + key_token) * kv_heads +
         kv_head) * head_dim;
    result += weights[key_token] * to_float(v[dimension]);
  }
  output[index] = from_float<T>(result);
}

template <class T>
cudaError_t launch_attention(const void* query, const void* key,
                             const void* value, void* output,
                             float* probabilities, int batch,
                             int query_tokens, int key_tokens,
                             int query_heads, int kv_heads, int head_dim,
                             bool causal, float scale, cudaStream_t stream) {
  constexpr int threads = 128;
  const int64_t rows =
      static_cast<int64_t>(batch) * query_tokens * query_heads;
  attention_scores_softmax<T><<<
      static_cast<unsigned>((rows + threads - 1) / threads), threads, 0,
      stream>>>(static_cast<const T*>(query), static_cast<const T*>(key),
                probabilities, query_tokens, key_tokens, query_heads,
                kv_heads, head_dim, batch, causal, scale);
  auto status = cudaGetLastError();
  if (status != cudaSuccess) return status;
  const int64_t elements = rows * head_dim;
  attention_values<T><<<
      static_cast<unsigned>((elements + threads - 1) / threads), threads, 0,
      stream>>>(probabilities, static_cast<const T*>(value),
                static_cast<T*>(output), query_tokens, key_tokens, query_heads,
                kv_heads, head_dim, batch);
  return cudaGetLastError();
}

}  // namespace apxinf::attention::kernels
