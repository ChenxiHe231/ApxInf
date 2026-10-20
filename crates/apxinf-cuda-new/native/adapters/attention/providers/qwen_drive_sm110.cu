// SPDX-License-Identifier: Apache-2.0
// Process-lifetime L1 bindings for Qwen-Drive's three SM110 attention
// candidates. The translated CUTLASS kernels own the arithmetic; this file
// owns stable provider state, fixed geometry, and split-batch glue.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <mutex>

extern "C" size_t apxinf_attention_d256_causal_state_size();
extern "C" int apxinf_attention_d256_causal_forward(
    const void*, const void*, const void*, void*, int, int, int, float,
    cudaStream_t, void*, int);

extern "C" size_t apxinf_attention_d256_split_batch_state_size();
extern "C" size_t apxinf_attention_d256_split_batch_lse_elements(int, int,
                                                                    int);
extern "C" int apxinf_attention_d256_split_batch_forward(
    const void*, const void*, const void*, void*, float*, int, int, int, int*,
    int, int, int, int, int, float, cudaStream_t, void*, int);

extern "C" size_t apxinf_attention_strided_groups_state_size();
extern "C" int apxinf_attention_strided_groups_forward(
    const void*, const void*, const void*, void*, int, int, int, int, int,
    int, int, float, cudaStream_t, void*, int);

namespace {
constexpr int kUnsupported = 0x20000;

bool aligned(const void* pointer) {
  return pointer != nullptr &&
         (reinterpret_cast<uintptr_t>(pointer) & 15u) == 0;
}

int current_sm110_device(int* device) {
  cudaError_t error = cudaGetDevice(device);
  if (error != cudaSuccess) return int(error);
  int major = 0, minor = 0, sms = 0;
  if ((error = cudaDeviceGetAttribute(
           &major, cudaDevAttrComputeCapabilityMajor, *device)) != cudaSuccess)
    return int(error);
  if ((error = cudaDeviceGetAttribute(
           &minor, cudaDevAttrComputeCapabilityMinor, *device)) != cudaSuccess)
    return int(error);
  if ((error = cudaDeviceGetAttribute(
           &sms, cudaDevAttrMultiProcessorCount, *device)) != cudaSuccess)
    return int(error);
  return major == 11 && minor == 0 && sms == 20 ? 0 : kUnsupported;
}

void* causal_state = nullptr;
std::mutex causal_mutex;
std::atomic<bool> causal_ready{false};
int causal_device = -1;

constexpr int kBatch = 2;
constexpr int kQueryTokens = 50;
constexpr int kQueryHeads = 16;
constexpr int kKvHeads = 4;
constexpr int kDim = 256;
constexpr int kKvViewRows = 1720;
constexpr int kKvBatchRows = 1718;
constexpr int kMinKeyTokens = 3433;
constexpr int kMaxKeyTokens = 3437;
constexpr int kQueryWords = kQueryTokens * kQueryHeads * kDim;
constexpr int kMergeRows = kQueryTokens * kQueryHeads;

void* split_state = nullptr;
int32_t* split_cu_seqlens = nullptr;
float* split_native_lse = nullptr;
std::mutex split_mutex;
std::atomic<bool> split_ready{false};
int split_device = -1;

__global__ void duplicate_q_vec8(const uint4* source, uint4* destination) {
  constexpr int vectors = kQueryWords / 8;
  int index = int(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < vectors) {
    uint4 value = source[index];
    destination[index] = value;
    destination[index + vectors] = value;
  }
}

__global__ void repack_lse(const float* native_lse, float* dense_lse) {
  int index = int(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= kBatch * kMergeRows) return;
  int batch = index / kMergeRows;
  int row = index % kMergeRows;
  int query_head = row / kQueryTokens;
  int token = row % kQueryTokens;
  constexpr int rounded_query = 256;
  int kv_head = query_head / 4;
  int lane = query_head % 4;
  dense_lse[index] = native_lse[
      batch * kKvHeads * rounded_query + kv_head * rounded_query +
      token * 4 + lane];
}

union Bf16x8 {
  uint4 vector;
  uint16_t words[8];
};

__global__ void merge_partial_vec8(const uint4* partial, const float* lse,
                                   uint4* output) {
  constexpr int vectors_per_row = kDim / 8;
  constexpr int vectors_per_batch = kQueryWords / 8;
  int warp = (int(blockIdx.x) * blockDim.x + threadIdx.x) / 32;
  if (warp >= kMergeRows) return;
  int lane = threadIdx.x & 31;
  int token = warp / kQueryHeads;
  int head = warp % kQueryHeads;
  int lse_index = head * kQueryTokens + token;
  float weight0 = 0.0f, weight1 = 0.0f, sum = 0.0f;
  if (lane == 0) {
    float first = lse[lse_index];
    float second = lse[kMergeRows + lse_index];
    float maximum = fmaxf(first, second);
    weight0 = expf(first - maximum);
    weight1 = expf(second - maximum);
    sum = weight0 + weight1;
  }
  weight0 = __shfl_sync(0xffffffff, weight0, 0);
  weight1 = __shfl_sync(0xffffffff, weight1, 0);
  sum = __shfl_sync(0xffffffff, sum, 0);
  int vector = warp * vectors_per_row + lane;
  Bf16x8 first{}, second{}, result{};
  first.vector = partial[vector];
  second.vector = partial[vectors_per_batch + vector];
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    float a = __bfloat162float(__ushort_as_bfloat16(first.words[i]));
    float b = __bfloat162float(__ushort_as_bfloat16(second.words[i]));
    result.words[i] = __bfloat16_as_ushort(
        __float2bfloat16_rn((weight0 * a + weight1 * b) / sum));
  }
  output[vector] = result.vector;
}

void* vision_state_624 = nullptr;
void* vision_state_2200 = nullptr;
std::mutex vision_mutex;
std::atomic<bool> vision_ready{false};
int vision_device = -1;

int launch_vision_group(int group, const void* q, const void* k,
                        const void* v, void* output, cudaStream_t stream,
                        int prepare_only) {
  constexpr int starts[4] = {0, 624, 1248, 1872};
  constexpr int lengths[4] = {624, 624, 624, 2200};
  constexpr int qko_stride = 16 * 64;
  constexpr int v_stride = 3 * 16 * 64;
  constexpr int batch_tokens = 4072;
  int start = starts[group];
  int length = lengths[group];
  const char* qp = static_cast<const char*>(q) + int64_t(start) * qko_stride * 2;
  const char* kp = static_cast<const char*>(k) + int64_t(start) * qko_stride * 2;
  const char* vp = static_cast<const char*>(v) + int64_t(start) * v_stride * 2;
  char* op = static_cast<char*>(output) + int64_t(start) * qko_stride * 2;
  return apxinf_attention_strided_groups_forward(
      qp, kp, vp, op, 3, length, 16, 64, qko_stride, v_stride,
      batch_tokens, 0.125f, stream,
      length == 624 ? vision_state_624 : vision_state_2200, prepare_only);
}
}  // namespace

extern "C" int apxinf_qwen_drive_d256_causal_init() {
  int device = -1;
  int status = current_sm110_device(&device);
  if (status != 0) return status;
  if (causal_ready.load(std::memory_order_acquire))
    return device == causal_device ? 0 : kUnsupported;
  std::lock_guard<std::mutex> lock(causal_mutex);
  if (!causal_ready.load(std::memory_order_relaxed)) {
    causal_state = std::calloc(1, apxinf_attention_d256_causal_state_size());
    if (causal_state == nullptr) return int(cudaErrorMemoryAllocation);
    causal_device = device;
    causal_ready.store(true, std::memory_order_release);
  }
  return device == causal_device ? 0 : kUnsupported;
}

extern "C" int apxinf_qwen_drive_d256_causal_forward(
    const void* q, const void* k, const void* v, void* output, int seq,
    cudaStream_t stream) {
  if (!causal_ready.load(std::memory_order_acquire))
    return int(cudaErrorNotReady);
  if (!aligned(q) || !aligned(k) || !aligned(v) || !aligned(output))
    return int(cudaErrorInvalidValue);
  for (int prepare = 1; prepare >= 0; --prepare) {
    int status = apxinf_attention_d256_causal_forward(
        q, k, v, output, seq, 16, 4, 0.0625f, stream, causal_state,
        prepare);
    if (status != 0) return status;
  }
  return int(cudaGetLastError());
}

extern "C" int apxinf_qwen_drive_d256_split_init() {
  int device = -1;
  int status = current_sm110_device(&device);
  if (status != 0) return status;
  if (split_ready.load(std::memory_order_acquire))
    return device == split_device ? 0 : kUnsupported;
  std::lock_guard<std::mutex> lock(split_mutex);
  if (split_ready.load(std::memory_order_relaxed))
    return device == split_device ? 0 : kUnsupported;
  split_state = std::calloc(1, apxinf_attention_d256_split_batch_state_size());
  if (split_state == nullptr) return int(cudaErrorMemoryAllocation);
  size_t lse_bytes = apxinf_attention_d256_split_batch_lse_elements(
                         kQueryTokens, kKvHeads, kBatch) *
                     sizeof(float);
  cudaError_t error = cudaMalloc(&split_native_lse, lse_bytes);
  if (error != cudaSuccess) return int(error);
  int32_t host[3 * (kMaxKeyTokens - kMinKeyTokens + 1)] = {};
  for (int i = 0; i <= kMaxKeyTokens - kMinKeyTokens; ++i) {
    host[3 * i] = 0;
    host[3 * i + 1] = kKvBatchRows;
    host[3 * i + 2] = kMinKeyTokens + i;
  }
  if ((error = cudaMalloc(&split_cu_seqlens, sizeof(host))) != cudaSuccess)
    return int(error);
  if ((error = cudaMemcpy(split_cu_seqlens, host, sizeof(host),
                          cudaMemcpyHostToDevice)) != cudaSuccess)
    return int(error);
  split_device = device;
  split_ready.store(true, std::memory_order_release);
  return 0;
}

extern "C" int apxinf_qwen_drive_d256_split_forward(
    const void* q, const void* k, const void* v, void* q2, void* partial,
    float* lse, void* output, int key_tokens, cudaStream_t stream) {
  if (!split_ready.load(std::memory_order_acquire))
    return int(cudaErrorNotReady);
  if (key_tokens < kMinKeyTokens || key_tokens > kMaxKeyTokens ||
      !aligned(q) || !aligned(k) || !aligned(v) || !aligned(q2) ||
      !aligned(partial) || !aligned(output) || lse == nullptr)
    return int(cudaErrorInvalidValue);
  duplicate_q_vec8<<<100, 256, 0, stream>>>(
      static_cast<const uint4*>(q), static_cast<uint4*>(q2));
  cudaError_t error = cudaGetLastError();
  if (error != cudaSuccess) return int(error);
  int32_t* cu = split_cu_seqlens + 3 * (key_tokens - kMinKeyTokens);
  for (int prepare = 1; prepare >= 0; --prepare) {
    int status = apxinf_attention_d256_split_batch_forward(
        q2, k, v, partial, split_native_lse, kQueryTokens, kKvViewRows,
        key_tokens, cu, kQueryHeads, kKvHeads, kBatch,
        kQueryTokens * kQueryHeads * kDim, kKvBatchRows * kKvHeads * kDim,
        0.0625f, stream, split_state, prepare);
    if (status != 0) return status;
  }
  repack_lse<<<(kBatch * kMergeRows + 255) / 256, 256, 0, stream>>>(
      split_native_lse, lse);
  if ((error = cudaGetLastError()) != cudaSuccess) return int(error);
  merge_partial_vec8<<<100, 256, 0, stream>>>(
      static_cast<const uint4*>(partial), lse, static_cast<uint4*>(output));
  return int(cudaGetLastError());
}

extern "C" int apxinf_qwen_drive_vision_groups_init() {
  int device = -1;
  int status = current_sm110_device(&device);
  if (status != 0) return status;
  if (vision_ready.load(std::memory_order_acquire))
    return device == vision_device ? 0 : kUnsupported;
  std::lock_guard<std::mutex> lock(vision_mutex);
  if (!vision_ready.load(std::memory_order_relaxed)) {
    size_t bytes = apxinf_attention_strided_groups_state_size();
    vision_state_624 = std::calloc(1, bytes);
    vision_state_2200 = std::calloc(1, bytes);
    if (vision_state_624 == nullptr || vision_state_2200 == nullptr)
      return int(cudaErrorMemoryAllocation);
    vision_device = device;
    vision_ready.store(true, std::memory_order_release);
  }
  return device == vision_device ? 0 : kUnsupported;
}

extern "C" int apxinf_qwen_drive_vision_groups_forward(
    const void* q, const void* k, const void* v, void* output,
    cudaStream_t stream) {
  if (!vision_ready.load(std::memory_order_acquire))
    return int(cudaErrorNotReady);
  if (!aligned(q) || !aligned(k) || !aligned(v) || !aligned(output))
    return int(cudaErrorInvalidValue);
  for (int prepare = 1; prepare >= 0; --prepare) {
    for (int group = 0; group < 4; ++group) {
      int status = launch_vision_group(group, q, k, v, output, stream,
                                       prepare);
      if (status != 0) return status;
    }
  }
  return int(cudaGetLastError());
}
