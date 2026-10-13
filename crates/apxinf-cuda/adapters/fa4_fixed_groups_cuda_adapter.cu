// Copyright 2025 ApxInf contributors.
// SPDX-License-Identifier: Apache-2.0
//
// CUDA arm of `vision-attention-624` / `vision-attention-2200`: the same four
// fixed-group launches `fa4_fixed_groups_adapter.cu` makes against the AOT
// objects, issued instead against the in-tree CUTLASS translation in
// `kernels/cutlass/fmha_vision_sm100.cu`.
//
// The two arms are required to be bit-identical and both are linked when the
// `aot-vision-attention` feature is on, so the group table, the strides and
// the softmax scale live HERE ONLY ONCE each and are compared by
// `tests/qwen_drive_vision_attention_equivalence.rs`.

#include <cuda_runtime.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <mutex>

extern "C" size_t apxinf_vision_fmha_state_size();
extern "C" void apxinf_vision_fmha_state_destroy(void* state_ptr);
extern "C" int apxinf_vision_mha_bf16_cuda(
    const void* q, const void* k, const void* v, void* output, int batches,
    int tokens, int heads, int head_dim, int qko_token_stride,
    int v_token_stride, int batch_token_count, float scale,
    cudaStream_t stream, void* state_ptr, int prepare_only);

namespace {
// Contract, not tunables: these are the AOT objects' own specialisation
// constants. They are duplicated from `fa4_fixed_groups_adapter.cu` because
// that file only exists in the AOT arm.
constexpr int kHeads = 16;
constexpr int kDim = 64;
constexpr int kBatches = 3;
constexpr int kBatchTokens = 4072;
constexpr int kQkoTokenStride = kHeads * kDim;      // 1024
constexpr int kVTokenStride = 3 * kHeads * kDim;    // 3072, the retained QKV
constexpr float kScale = 0.125f;                    // 1 / sqrt(64)
constexpr int kGroupStarts[4] = {0, 624, 1248, 1872};
constexpr int kGroupLengths[4] = {624, 624, 624, 2200};
constexpr int kUnsupported = 0x20000;

// One `State` per group length: the CUTLASS operation caches a workspace whose
// size depends on the problem shape, and the three 624 groups share one.
void* state_624 = nullptr;
void* state_2200 = nullptr;
std::mutex init_mutex;
std::atomic<bool> ready{false};
int initialized_device = -1;  // Published before ready.store(release).

bool aligned(const void* ptr) {
  return ptr != nullptr && (reinterpret_cast<uintptr_t>(ptr) & 15) == 0;
}

int32_t launch_group(int group, const void* q, const void* k, const void* v,
                     void* out, cudaStream_t stream, int prepare_only) {
  const int start = kGroupStarts[group];
  const int length = kGroupLengths[group];
  const int64_t q_offset = int64_t(start) * kQkoTokenStride;
  const int64_t v_offset = int64_t(start) * kVTokenStride;
  const char* qp = static_cast<const char*>(q) + q_offset * 2;
  const char* kp = static_cast<const char*>(k) + q_offset * 2;
  const char* vp = static_cast<const char*>(v) + v_offset * 2;
  char* op = static_cast<char*>(out) + q_offset * 2;
  void* state = length == 624 ? state_624 : state_2200;
  const int rc = apxinf_vision_mha_bf16_cuda(
      qp, kp, vp, op, kBatches, length, kHeads, kDim, kQkoTokenStride,
      kVTokenStride, kBatchTokens, kScale, stream, state, prepare_only);
  if (rc != 0) return rc;
  return int32_t(cudaGetLastError());
}
}  // namespace

extern "C" int32_t apxinf_static_fa4_bf16_vfixed_cuda_init(
    cudaStream_t stream) {
  cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
  cudaError_t error = cudaStreamIsCapturing(stream, &capture);
  if (error != cudaSuccess) return int32_t(error);
  if (capture != cudaStreamCaptureStatusNone)
    return int32_t(cudaErrorStreamCaptureUnsupported);
  int device = -1;
  error = cudaGetDevice(&device);
  if (error != cudaSuccess) return int32_t(error);
  if (ready.load(std::memory_order_acquire))
    return device == initialized_device ? 0 : kUnsupported;
  std::lock_guard<std::mutex> lock(init_mutex);
  if (ready.load(std::memory_order_relaxed))
    return device == initialized_device ? 0 : kUnsupported;
  int count = 0, major = 0, minor = 0, sms = 0;
  error = cudaGetDeviceCount(&count);
  if (error != cudaSuccess) return int32_t(error);
  if (count != 1) return kUnsupported;
  error = cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor,
                                 device);
  if (error != cudaSuccess) return int32_t(error);
  error = cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor,
                                 device);
  if (error != cudaSuccess) return int32_t(error);
  error = cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device);
  if (error != cudaSuccess) return int32_t(error);
  if (major != 11 || minor != 0 || sms != 20) return kUnsupported;

  const size_t state_bytes = apxinf_vision_fmha_state_size();
  state_624 = std::calloc(1, state_bytes);
  state_2200 = std::calloc(1, state_bytes);
  if (state_624 == nullptr || state_2200 == nullptr) {
    std::free(state_624);
    std::free(state_2200);
    state_624 = nullptr;
    state_2200 = nullptr;
    return int32_t(cudaErrorMemoryAllocation);
  }
  initialized_device = device;
  ready.store(true, std::memory_order_release);
  return 0;
}

extern "C" int32_t apxinf_static_fa4_bf16_vfixed_cuda_ready(void) {
  if (!ready.load(std::memory_order_acquire)) return kUnsupported;
  int device = -1;
  cudaError_t error = cudaGetDevice(&device);
  if (error != cudaSuccess) return int32_t(error);
  return device == initialized_device ? 0 : kUnsupported;
}

extern "C" int32_t apxinf_static_fa4_bf16_vfixed_cuda_forward(
    const void* q, const void* k, const void* v, void* out,
    cudaStream_t stream) {
  if (!ready.load(std::memory_order_acquire)) return int32_t(cudaErrorNotReady);
  if (!aligned(q) || !aligned(k) || !aligned(v) || !aligned(out))
    return int32_t(cudaErrorInvalidValue);
  int device = -1;
  cudaError_t error = cudaGetDevice(&device);
  if (error != cudaSuccess) return int32_t(error);
  if (device != initialized_device) return kUnsupported;
  // The first call per group length sizes and allocates the CUTLASS workspace.
  // It cannot happen under graph capture; the Rust layer only reaches
  // `forward` after `init` has run outside capture, and `prepare_only` is
  // idempotent once the workspace is large enough.
  for (int group = 0; group < 4; ++group) {
    int32_t status = launch_group(group, q, k, v, out, stream, 1);
    if (status != 0) return status;
  }
  for (int group = 0; group < 4; ++group) {
    int32_t status = launch_group(group, q, k, v, out, stream, 0);
    if (status != 0) return status;
  }
  return 0;
}

// Test-only: release the two CUTLASS states. Production never calls it -- the
// allocations are process lifetime, like the AOT arm's loaded modules.
extern "C" int32_t apxinf_static_fa4_bf16_vfixed_cuda_release(void) {
  std::lock_guard<std::mutex> lock(init_mutex);
  if (!ready.load(std::memory_order_relaxed)) return 0;
  apxinf_vision_fmha_state_destroy(state_624);
  apxinf_vision_fmha_state_destroy(state_2200);
  std::free(state_624);
  std::free(state_2200);
  state_624 = nullptr;
  state_2200 = nullptr;
  initialized_device = -1;
  ready.store(false, std::memory_order_release);
  return 0;
}
