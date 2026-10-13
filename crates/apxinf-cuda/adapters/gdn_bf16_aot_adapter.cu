// SPDX-License-Identifier: Apache-2.0
// BF16 first-prefill recurrent attention, backed by the pinned FI64 AOT kernel.
//
// Only the GDN prefill launch lives here; the conv/pack/gate prologue and the
// state transpose are shared with the in-tree arm through
// `gdn_bf16_prefill_common.cuh` so the two arms differ in exactly one kernel.
#include <atomic>
#include <mutex>

#include "gdn_bf16_prefill_common.cuh"
#include "flashinfer_gdn_bf16_t64_sm110_real3392.h"

namespace {
using namespace apxinf_gdn_bf16;
constexpr int AOT_ERROR_BASE = 0x10000;
flashinfer_gdn_bf16_t64_sm110_real3392_Kernel_Module_t flash_module{};
std::mutex flash_init_mutex;
std::atomic<int> initialized_device{-1};
std::atomic<bool> flash_ready{false};
}  // namespace

extern "C" int32_t apxinf_static_gdn_flashinfer64_init() {
  int device = -1;
  cudaError_t error = cudaGetDevice(&device);
  if (error != cudaSuccess) return status(error);
  if (flash_ready.load(std::memory_order_acquire))
    return device == initialized_device.load() ? 0 : status(cudaErrorInvalidDevice);
  std::lock_guard<std::mutex> lock(flash_init_mutex);
  if (flash_ready.load(std::memory_order_relaxed))
    return device == initialized_device.load() ? 0 : status(cudaErrorInvalidDevice);
  int major = 0, minor = 0, sms = 0, devices = 0;
  if ((error = cudaGetDeviceCount(&devices)) != cudaSuccess) return status(error);
  if (devices != 1) return status(cudaErrorInvalidDevice);
  if ((error = cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, device)) != cudaSuccess)
    return status(error);
  if ((error = cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, device)) != cudaSuccess)
    return status(error);
  if ((error = cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device)) != cudaSuccess)
    return status(error);
  if (major != 11 || minor != 0 || sms != 20) return status(cudaErrorInvalidDeviceFunction);
  cudaLibrary_t* library = &flash_module.module;
  struct InitArgs { cudaLibrary_t** library; cudaError_t* error; } init{&library, &error};
  _mlir_flashinfer_gdn_bf16_t64_sm110_real3392_cuda_init(reinterpret_cast<void**>(&init));
  if (error != cudaSuccess) return status(error);
  struct LoadArgs { cudaLibrary_t** library; int32_t* device; cudaError_t* error; }
      load{&library, &device, &error};
  _mlir_flashinfer_gdn_bf16_t64_sm110_real3392_cuda_load_to_device(reinterpret_cast<void**>(&load));
  if (error != cudaSuccess) {
    if (flash_module.module) cudaLibraryUnload(flash_module.module);
    flash_module.module = nullptr;
    return status(error);
  }
  if (!flash_module.module) return status(cudaErrorInvalidResourceHandle);
  initialized_device.store(device);
  flash_ready.store(true, std::memory_order_release);
  return 0;
}

extern "C" int32_t apxinf_static_gdn_flashinfer64_prefill_compact_v_aot(
    const void* zba, const void* conv_weight, void* new_conv_state,
    const float* A_log, const void* dt_bias_bf16,
    void* raw_qk_scratch, void* q, void* k, void* v, void* a, void* b,
    float* h_output_vmajor, float* final_state_kmajor,
    void* output_pad, int32_t* cu_seqlens, int32_t seq, int32_t seq_pad,
    cudaStream_t stream) {
  if (!flash_ready.load(std::memory_order_acquire)) return status(cudaErrorNotReady);
  int device = -1;
  cudaError_t error = cudaGetDevice(&device);
  if (error != cudaSuccess) return status(error);
  if (device != initialized_device.load(std::memory_order_relaxed))
    return status(cudaErrorInvalidDevice);
  if (!arguments_valid(zba, conv_weight, new_conv_state, A_log, dt_bias_bf16,
                       raw_qk_scratch, q, k, v, a, b, h_output_vmajor,
                       final_state_kmajor, output_pad, cu_seqlens, seq,
                       seq_pad))
    return status(cudaErrorInvalidValue);
  float* alpha = nullptr;
  float* beta = nullptr;
  void* descriptor_workspace = nullptr;
  int32_t prologue = apxinf_static_gdn_bf16_prologue(zba, conv_weight, new_conv_state, A_log,
                                  dt_bias_bf16, raw_qk_scratch, q, k, v, a, b,
                                  cu_seqlens, seq, seq_pad, &alpha, &beta,
                                  &descriptor_workspace, stream);
  if (prologue != 0) return prologue;
  flashinfer_gdn_bf16_t64_sm110_real3392_Tensor_q_t tq{q};
  flashinfer_gdn_bf16_t64_sm110_real3392_Tensor_k_t tk{k};
  flashinfer_gdn_bf16_t64_sm110_real3392_Tensor_v_t tv{v};
  flashinfer_gdn_bf16_t64_sm110_real3392_Tensor_gate_t tg{alpha};
  flashinfer_gdn_bf16_t64_sm110_real3392_Tensor_beta_t tb{beta};
  flashinfer_gdn_bf16_t64_sm110_real3392_Tensor_o_t to{output_pad};
  flashinfer_gdn_bf16_t64_sm110_real3392_Tensor_cu_seqlens_t ts{cu_seqlens};
  flashinfer_gdn_bf16_t64_sm110_real3392_Tensor_s_out_t th{h_output_vmajor};
  flashinfer_gdn_bf16_t64_sm110_real3392_Tensor_tensormap_workspace_t tw{
      descriptor_workspace};
  int32_t aot_status = cute_dsl_flashinfer_gdn_bf16_t64_sm110_real3392_wrapper(
      &flash_module, &tq, &tk, &tv, &tg, &tb, &to, &ts, &th, 0, SCALE, &tw,
      stream);
  if (aot_status != 0) return AOT_ERROR_BASE | (aot_status & 0xffff);
  return apxinf_static_gdn_bf16_state_transpose(h_output_vmajor,
                                                final_state_kmajor, stream);
}
