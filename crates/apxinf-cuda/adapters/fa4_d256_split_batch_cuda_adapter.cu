// Copyright 2025 ApxInf contributors.
// SPDX-License-Identifier: Apache-2.0
//
// CUDA arm of `action-attention`, against the in-tree CUTLASS translation in
// `kernels/cutlass/fmha_d256_action_sm100.cu`. Same C ABI as
// `fa4_d256_split_batch_adapter.cu` (the AOT arm), including the caller-owned
// `q2` / `partial` / `lse` scratch, so the Rust dispatch differs only in which
// symbol it calls, and `apxinf_split_batch_duplicate_q` /
// `apxinf_split_batch_merge` are the SAME kernels in both arms -- which is
// what isolates any difference to the attention launch itself.
//
// The attention launch, the LSE and the merged output are ALL bit-identical
// between the arms, at every one of the five admissible `key_tokens`. That was
// not true before report 27: the epilogue computed
// `cutlass::fast_log(row_sum) + scale_softmax*row_max` where FA4 computes
// `(row_max*scale_log2 + lg2.approx.ftz(row_sum)) * float(ln 2)`, which left
// 986 of 1600 LSE values off by 1-3 fp32 ULP. That is harmless in every other
// operator -- the LSE does not feed O -- but here it is the merge weight
// `sigmoid(lse0 - lse1)`, and at `key_tokens = 3436`, the ONE shape whose two
// batches have exactly equal KV length, `lse0 - lse1` collapses to ~2e-5, so
// the 1e-6 error was a 5-100% error in the weight and flipped the sign of the
// output wherever `p1 == -p0`: 12,112 BF16 ULP at `absmax` 7.6e-6.
// `-DAPXINF_LSE_FA4=1` in `build.rs` is what closes it; the gate asserts all
// three buffers by integer ULP, never by `absmax`.

#include <cuda_runtime.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <mutex>

extern "C" int32_t apxinf_split_batch_duplicate_q(const void*, void*,
                                                  cudaStream_t);
extern "C" int32_t apxinf_split_batch_merge(const void*, const float*, void*,
                                            cudaStream_t);

extern "C" size_t apxinf_d256_action_state_size();
extern "C" void apxinf_d256_action_state_destroy(void* state_ptr);
extern "C" size_t apxinf_d256_action_lse_elements(int seq_q, int kv_heads,
                                                  int batch);
extern "C" int apxinf_d256_action_forward_cuda(
    const void* q, const void* k, const void* v, void* output, float* lse,
    int seq_q, int max_seq_k, int total_seq_k, int* cu_seqlens_k,
    int query_heads, int kv_heads, int batch, int q_batch_stride,
    int kv_batch_stride, float scale, cudaStream_t stream, void* state_ptr,
    int prepare_only);

namespace {
// Contract, not tunables: the AOT object's own specialisation constants. The
// two KV batch views OVERLAP physically -- the batch stride is batch 0's USED
// row count (1718), not the viewed one (1720) -- and `cu_seqlens_k` cuts them
// apart, exactly as the AOT's `seqused_k = {1718, key_tokens - 1718}` does.
constexpr int kBatch = 2;
constexpr int kQueryTokens = 50;
constexpr int kQueryHeads = 16;
constexpr int kKvHeads = 4;
constexpr int kDim = 256;
constexpr int kKvViewRows = 1720;
constexpr int kKvBatchRows = 1718;
constexpr int kKvRowStride = kKvHeads * kDim;    // 1024
constexpr int kQueryRowStride = kQueryHeads * kDim;  // 4096
constexpr float kScale = 0.0625f;                // 1 / sqrt(256)
constexpr int kMinKeyTokens = 3433;
constexpr int kMaxKeyTokens = 3437;
constexpr int kUnsupported = 0x20000;
constexpr int kMergeRows = kQueryTokens * kQueryHeads;

void* state = nullptr;
// Process/device lifetime: a captured graph retains these addresses across
// replays, and neither is allocated outside `_cuda_init`.
int32_t* cu_seqlens_device = nullptr;
float* lse_native = nullptr;
std::mutex init_mutex;
std::atomic<bool> ready{false};
int initialized_device = -1;

// `lse_native` is the packed-M layout the CUTLASS epilogue writes:
// index = batch * kv_heads * 256 + kv_head * 256 + token * 4 + lane, with
// query head = kv_head * 4 + lane. `apxinf_split_batch_merge` wants a dense
// [batch, query_head, token] buffer with a 800-element batch stride. Pure
// gather; the index expression is the same one
// `apxinf_d256_action_lse_index()` exports for the test.
__global__ void repack_lse(const float* __restrict__ native,
                           float* __restrict__ dense) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= kBatch * kMergeRows) return;
  const int batch = i / kMergeRows;
  const int row = i % kMergeRows;
  const int q_head = row / kQueryTokens;
  const int token = row % kQueryTokens;
  constexpr int kRoundedQuery = 256;  // round_up(4 * 50, 128)
  const int kv_head = q_head / (kQueryHeads / kKvHeads);
  const int lane = q_head % (kQueryHeads / kKvHeads);
  dense[batch * kMergeRows + row] =
      native[batch * kKvHeads * kRoundedQuery + kv_head * kRoundedQuery +
             token * (kQueryHeads / kKvHeads) + lane];
}

bool aligned(const void* ptr) {
  return ptr != nullptr && (reinterpret_cast<uintptr_t>(ptr) & 15) == 0;
}
}  // namespace

extern "C" int32_t apxinf_static_fa4_split_batch_cuda_init(
    cudaStream_t stream) {
  cudaStreamCaptureStatus capture_status{};
  cudaError_t error = cudaStreamIsCapturing(stream, &capture_status);
  if (error != cudaSuccess) return int32_t(error);
  if (capture_status != cudaStreamCaptureStatusNone)
    return int32_t(cudaErrorStreamCaptureUnsupported);
  int device = -1;
  error = cudaGetDevice(&device);
  if (error != cudaSuccess) return int32_t(error);
  if (ready.load(std::memory_order_acquire))
    return device == initialized_device ? 0 : kUnsupported;
  std::lock_guard<std::mutex> lock(init_mutex);
  if (ready.load(std::memory_order_relaxed))
    return device == initialized_device ? 0 : kUnsupported;
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
  if (device_count != 1) return kUnsupported;

  state = std::calloc(1, apxinf_d256_action_state_size());
  if (state == nullptr) return int32_t(cudaErrorMemoryAllocation);
  const size_t lse_bytes =
      apxinf_d256_action_lse_elements(kQueryTokens, kKvHeads, kBatch) *
      sizeof(float);
  error = cudaMalloc(&lse_native, lse_bytes);
  if (error != cudaSuccess) {
    std::free(state);
    state = nullptr;
    return int32_t(error);
  }
  // cu_seqlens for all five admissible key_tokens, laid out contiguously as
  // five 3-int rows so `forward` only has to offset into it.
  int32_t host[3 * (kMaxKeyTokens - kMinKeyTokens + 1)] = {};
  for (int i = 0; i <= kMaxKeyTokens - kMinKeyTokens; ++i) {
    host[3 * i + 0] = 0;
    host[3 * i + 1] = kKvBatchRows;
    host[3 * i + 2] = kMinKeyTokens + i;
  }
  error = cudaMalloc(&cu_seqlens_device, sizeof(host));
  if (error == cudaSuccess) {
    error = cudaMemcpy(cu_seqlens_device, host, sizeof(host),
                       cudaMemcpyHostToDevice);
  }
  if (error != cudaSuccess) {
    cudaFree(cu_seqlens_device);
    cudaFree(lse_native);
    cu_seqlens_device = nullptr;
    lse_native = nullptr;
    std::free(state);
    state = nullptr;
    return int32_t(error);
  }
  initialized_device = device;
  ready.store(true, std::memory_order_release);
  return 0;
}

extern "C" int32_t apxinf_static_fa4_split_batch_cuda_ready(void) {
  if (!ready.load(std::memory_order_acquire)) return kUnsupported;
  int device = -1;
  const cudaError_t error = cudaGetDevice(&device);
  if (error != cudaSuccess) return int32_t(error);
  return device == initialized_device ? 0 : kUnsupported;
}

extern "C" int32_t apxinf_static_fa4_split_batch_cuda_forward(
    const void* q, const void* k, const void* v, void* q2, void* partial,
    float* lse, void* out, int32_t key_tokens, cudaStream_t stream) {
  if (key_tokens < kMinKeyTokens || key_tokens > kMaxKeyTokens)
    return int32_t(cudaErrorInvalidValue);
  if (!ready.load(std::memory_order_acquire)) return int32_t(cudaErrorNotReady);
  int device = -1;
  cudaError_t error = cudaGetDevice(&device);
  if (error != cudaSuccess) return int32_t(error);
  if (device != initialized_device) return kUnsupported;
  if (!aligned(q) || !aligned(k) || !aligned(v) || !aligned(q2) ||
      !aligned(partial) || !aligned(out) || lse == nullptr) {
    return int32_t(cudaErrorInvalidValue);
  }

  int32_t status = apxinf_split_batch_duplicate_q(q, q2, stream);
  if (status != 0) return status;

  int32_t* cu = cu_seqlens_device + 3 * (key_tokens - kMinKeyTokens);
  for (int prepare = 1; prepare >= 0; --prepare) {
    const int rc = apxinf_d256_action_forward_cuda(
        q2, k, v, partial, lse_native, kQueryTokens, kKvViewRows, key_tokens,
        cu, kQueryHeads, kKvHeads, kBatch, kQueryTokens * kQueryRowStride,
        kKvBatchRows * kKvRowStride, kScale, stream, state, prepare);
    if (rc != 0) return rc;
  }
  repack_lse<<<(kBatch * kMergeRows + 255) / 256, 256, 0, stream>>>(
      lse_native, lse);
  error = cudaGetLastError();
  if (error != cudaSuccess) return int32_t(error);
  status = apxinf_split_batch_merge(partial, lse, out, stream);
  if (status != 0) return status;
  return int32_t(cudaGetLastError());
}

// Test-only: release the CUTLASS state and the two device scratch buffers.
extern "C" int32_t apxinf_static_fa4_split_batch_cuda_release(void) {
  std::lock_guard<std::mutex> lock(init_mutex);
  if (!ready.load(std::memory_order_relaxed)) return 0;
  apxinf_d256_action_state_destroy(state);
  std::free(state);
  state = nullptr;
  cudaFree(cu_seqlens_device);
  cudaFree(lse_native);
  cu_seqlens_device = nullptr;
  lse_native = nullptr;
  initialized_device = -1;
  ready.store(false, std::memory_order_release);
  return 0;
}
