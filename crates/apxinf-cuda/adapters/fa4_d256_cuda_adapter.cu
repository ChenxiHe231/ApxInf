// Copyright 2025 ApxInf contributors.
// SPDX-License-Identifier: Apache-2.0
//
// CUDA arm of `language-attention`, against the in-tree CUTLASS translation in
// `kernels/cutlass/fmha_d256_language_sm100.cu`. Same C ABI shape as
// `fa4_d256_native_adapter.cu` (the AOT arm) so the Rust dispatch differs only
// in which symbol it calls.

#include <cuda_runtime.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <mutex>

#include "fixed_profile.h"

extern "C" size_t apxinf_d256_language_state_size();
extern "C" void apxinf_d256_language_state_destroy(void* state_ptr);
extern "C" int apxinf_d256_language_forward_cuda(
    const void* q, const void* k, const void* v, void* output, int seq,
    int query_heads, int kv_heads, float scale, cudaStream_t stream,
    void* state_ptr, int prepare_only);

namespace {
// Contract, not tunables: the AOT object's own specialisation constants.
constexpr int kQueryHeads = 16;
constexpr int kKvHeads = 4;
constexpr float kScale = 0.0625f;  // 1 / sqrt(256)
constexpr int kUnsupported = 0x20000;

void* state = nullptr;
std::mutex init_mutex;
std::atomic<bool> ready{false};
int initialized_device = -1;
}  // namespace

extern "C" int32_t apxinf_static_fa4_d256_cuda_init(void) {
  int device = -1;
  cudaError_t error = cudaGetDevice(&device);
  if (error != cudaSuccess) return int32_t(error);
  if (ready.load(std::memory_order_acquire))
    return device == initialized_device ? 0 : int32_t(cudaErrorInvalidDevice);
  std::lock_guard<std::mutex> lock(init_mutex);
  if (ready.load(std::memory_order_relaxed))
    return device == initialized_device ? 0 : int32_t(cudaErrorInvalidDevice);
  int major = 0, minor = 0, device_count = 0;
  error =
      cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device);
  if (error != cudaSuccess) return int32_t(error);
  error =
      cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device);
  if (error != cudaSuccess) return int32_t(error);
  if (major != 11 || minor != 0) return int32_t(cudaErrorInvalidDeviceFunction);
  error = cudaGetDeviceCount(&device_count);
  if (error != cudaSuccess) return int32_t(error);
  if (device_count != 1) return int32_t(cudaErrorInvalidDevice);
  state = std::calloc(1, apxinf_d256_language_state_size());
  if (state == nullptr) return int32_t(cudaErrorMemoryAllocation);
  initialized_device = device;
  ready.store(true, std::memory_order_release);
  return 0;
}

extern "C" int32_t apxinf_static_fa4_d256_cuda_ready(void) {
  if (!ready.load(std::memory_order_acquire)) return kUnsupported;
  int device = -1;
  const cudaError_t error = cudaGetDevice(&device);
  if (error != cudaSuccess) return int32_t(error);
  return device == initialized_device ? 0 : kUnsupported;
}

extern "C" int32_t apxinf_static_fa4_d256_cuda_forward(const void* q,
                                                       const void* k,
                                                       const void* v,
                                                       void* out,
                                                       cudaStream_t stream) {
  if (!ready.load(std::memory_order_acquire)) return int32_t(cudaErrorNotReady);
  int device = -1;
  cudaError_t error = cudaGetDevice(&device);
  if (error != cudaSuccess) return int32_t(error);
  if (device != initialized_device) return int32_t(cudaErrorInvalidDevice);
  if (!q || !k || !v || !out) return int32_t(cudaErrorInvalidValue);
  int seq = APXINF_FIXED_SCENE_TOKENS;
  if (const char* value = std::getenv("APX_LANGUAGE_TEST_TOKENS")) {
    const int parsed = std::atoi(value);
    if (parsed > 0) seq = parsed;
  }
  // `ptr_LSE` is null in this configuration, so the prepare pass only sizes
  // the (empty) CUTLASS workspace and issues no allocation.
  int rc = apxinf_d256_language_forward_cuda(
      q, k, v, out, seq, kQueryHeads, kKvHeads, kScale,
      stream, state, 1);
  if (rc != 0) return rc;
  rc = apxinf_d256_language_forward_cuda(
      q, k, v, out, seq, kQueryHeads, kKvHeads, kScale,
      stream, state, 0);
  if (rc != 0) return rc;
  return int32_t(cudaGetLastError());
}

// Test-only: release the CUTLASS state. Production never calls it.
extern "C" int32_t apxinf_static_fa4_d256_cuda_release(void) {
  std::lock_guard<std::mutex> lock(init_mutex);
  if (!ready.load(std::memory_order_relaxed)) return 0;
  apxinf_d256_language_state_destroy(state);
  std::free(state);
  state = nullptr;
  initialized_device = -1;
  ready.store(false, std::memory_order_release);
  return 0;
}
