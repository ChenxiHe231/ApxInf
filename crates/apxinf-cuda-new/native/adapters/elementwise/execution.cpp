// Elementwise / MLP kernels: RMSNorm, SwiGLU, residual add, FP8 quantize, FP8 GEMV, NVFP4 GEMV.
//
// These ops have a single fixed implementation and no persisted selection,
// so per doc/adding-new-kernels.md §6 they carry no candidate registry,
// tuning key, or autotuner; they are direct C-ABI forwarders.

#include "../../include/apxinf_cuda/mlp.h"

#include "../../framework/runtime_internal.h"
#include "../../kernels/custom/mlp_ops.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>

namespace {

using apxinf::framework::Failure;
using apxinf::framework::abi_boundary;

void check(int status, const char* what) {
  if (status != 0) {
    throw Failure(APXINF_STATUS_PROVIDER_ERROR,
                  std::string(what) + " failed with status " +
                      std::to_string(status));
  }
}
bool valid_extent(int64_t value) {
  return value > 0 && value <= INT32_MAX;
}

struct Nvfp4A16WmmaPolicy { bool enabled; int tile_n; int min_rows; bool valid; };
const Nvfp4A16WmmaPolicy& nvfp4_a16_wmma_policy() {
  static const Nvfp4A16WmmaPolicy policy = [] {
    const char* enabled = std::getenv("APXINF_NVFP4_A16_WMMA");
    if (enabled == nullptr || std::strcmp(enabled,"0") == 0)
      return Nvfp4A16WmmaPolicy{false,64,4,true};
    if (std::strcmp(enabled,"1") != 0)
      return Nvfp4A16WmmaPolicy{false,64,4,false};
    int tile_n = 64;
    if (const char* tile = std::getenv("APXINF_NVFP4_A16_WMMA_TILE_N")) {
      if (std::strcmp(tile,"32") == 0) tile_n = 32;
      else if (std::strcmp(tile,"64") == 0) tile_n = 64;
      else if (std::strcmp(tile,"128") == 0) tile_n = 128;
      else return Nvfp4A16WmmaPolicy{false,64,4,false};
    }
    int min_rows = 4;
    if (const char* minimum = std::getenv("APXINF_NVFP4_A16_WMMA_MIN_ROWS")) {
      if (std::strcmp(minimum,"1") == 0) min_rows = 1;
      else if (std::strcmp(minimum,"2") == 0) min_rows = 2;
      else if (std::strcmp(minimum,"3") == 0) min_rows = 3;
      else if (std::strcmp(minimum,"4") == 0) min_rows = 4;
      else return Nvfp4A16WmmaPolicy{false,64,4,false};
    }
    return Nvfp4A16WmmaPolicy{true,tile_n,min_rows,true};
  }();
  if (!policy.valid) throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
      "WMMA requires0/1; enabled TILE_N requires32/64/128 and MIN_ROWS requires1/2/3/4");
  return policy;
}

}  // namespace

extern "C" apxinf_status_t apxinf_new_rms_norm_bf16(const void* input,
                                                const void* weight,
                                                void* output, int64_t rows,
                                                int64_t width, float epsilon,
                                                apxinf_cuda_stream_t stream) {
  return abi_boundary([&] {
    if (input == nullptr || weight == nullptr || output == nullptr ||
        !valid_extent(rows) || !valid_extent(width) || !(epsilon >= 0.0F)) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT, "invalid RMSNorm arguments");
    }
    check(apxinf::cuda_new::mlp_ops::rms_norm_bf16(
              input, weight, output, static_cast<int>(rows),
              static_cast<int>(width), epsilon,
              static_cast<cudaStream_t>(stream)),
          "RMSNorm");
  });
}

extern "C" apxinf_status_t apxinf_swiglu_bf16(const void* fused_gate_up,
                                              void* output, int64_t rows,
                                              int64_t width,
                                              apxinf_cuda_stream_t stream) {
  return abi_boundary([&] {
    if (fused_gate_up == nullptr || output == nullptr || !valid_extent(rows) ||
        !valid_extent(width)) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT, "invalid SwiGLU arguments");
    }
    check(apxinf::cuda_new::mlp_ops::swiglu_bf16(
              fused_gate_up, output, static_cast<int>(rows),
              static_cast<int>(width), static_cast<cudaStream_t>(stream)),
          "SwiGLU");
  });
}

extern "C" apxinf_status_t apxinf_swiglu_split_bf16(
    const void* gate, const void* up, void* output, int64_t rows,
    int64_t width, apxinf_cuda_stream_t stream) {
  return abi_boundary([&] {
    if (gate == nullptr || up == nullptr || output == nullptr ||
        !valid_extent(rows) || !valid_extent(width)) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                    "invalid split-input SwiGLU arguments");
    }
    check(apxinf::cuda_new::mlp_ops::swiglu_split_bf16(
              gate, up, output, static_cast<int>(rows),
              static_cast<int>(width), static_cast<cudaStream_t>(stream)),
          "split-input SwiGLU");
  });
}

extern "C" apxinf_status_t apxinf_swiglu_bf16_round_product(
    const void* fused_gate_up, void* output, int64_t rows, int64_t width,
    apxinf_cuda_stream_t stream) {
  return abi_boundary([&] {
    if (fused_gate_up == nullptr || output == nullptr || !valid_extent(rows) ||
        !valid_extent(width)) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                    "invalid round-product SwiGLU arguments");
    }
    check(apxinf::cuda_new::mlp_ops::swiglu_bf16_round_product(
              fused_gate_up, output, static_cast<int>(rows),
              static_cast<int>(width), static_cast<cudaStream_t>(stream)),
          "round-product SwiGLU");
  });
}

extern "C" apxinf_status_t apxinf_new_add_bf16(const void* addend,
                                           void* accumulator, int64_t count,
                                           apxinf_cuda_stream_t stream) {
  return abi_boundary([&] {
    if (addend == nullptr || accumulator == nullptr || count <= 0) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT, "invalid add arguments");
    }
    check(apxinf::cuda_new::mlp_ops::add_bf16(addend, accumulator, count,
                                          static_cast<cudaStream_t>(stream)),
          "add");
  });
}

extern "C" apxinf_status_t apxinf_quantize_fp8_per_tensor(
    const void* input, void* output, int64_t count, float input_scale,
    apxinf_cuda_stream_t stream) {
  return abi_boundary([&] {
    if (input == nullptr || output == nullptr || count <= 0 ||
        !(input_scale > 0.0F)) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                    "invalid FP8 quantization arguments");
    }
    check(apxinf::cuda_new::mlp_ops::quantize_fp8_per_tensor(
              input, output, count, input_scale,
              static_cast<cudaStream_t>(stream)),
          "FP8 quantization");
  });
}

extern "C" apxinf_status_t apxinf_bf16_gemv(
    const void* activation, const void* weight, void* output, int64_t n,
    int64_t k, apxinf_cuda_stream_t stream) {
  return abi_boundary([&] {
    if (activation == nullptr || weight == nullptr || output == nullptr ||
        !valid_extent(n) || !valid_extent(k)) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                    "invalid BF16 GEMV arguments");
    }
    check(apxinf::cuda_new::mlp_ops::bf16_gemv(
              activation, weight, output, static_cast<int>(n),
              static_cast<int>(k), static_cast<cudaStream_t>(stream)),
          "BF16 GEMV");
  });
}

extern "C" apxinf_status_t apxinf_fp8_gemv(const void* weight,
                                           const void* activation,
                                           void* output, int64_t n, int64_t k,
                                           float alpha,
                                           apxinf_cuda_stream_t stream) {
  return abi_boundary([&] {
    if (weight == nullptr || activation == nullptr || output == nullptr ||
        !valid_extent(n) || !valid_extent(k)) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT, "invalid FP8 GEMV arguments");
    }
    check(apxinf::cuda_new::mlp_ops::fp8_gemv(weight, activation, output,
                                          static_cast<int>(n),
                                          static_cast<int>(k), alpha,
                                          static_cast<cudaStream_t>(stream)),
          "FP8 GEMV");
  });
}

extern "C" apxinf_status_t apxinf_nvfp4_gemv(
    const void* weight, const void* weight_scales, const void* activation,
    const void* activation_scales, void* output, int64_t n, int64_t k,
    float alpha, apxinf_cuda_stream_t stream) {
  return abi_boundary([&] {
    if (weight == nullptr || weight_scales == nullptr ||
        activation == nullptr || activation_scales == nullptr ||
        output == nullptr || !valid_extent(n) || !valid_extent(k)) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                    "invalid NVFP4 GEMV arguments");
    }
    check(apxinf::cuda_new::mlp_ops::nvfp4_gemv(
              weight, weight_scales, activation, activation_scales, output,
              static_cast<int>(n), static_cast<int>(k), alpha,
              static_cast<cudaStream_t>(stream)),
          "NVFP4 GEMV");
  });
}

extern "C" apxinf_status_t apxinf_nvfp4_a16_blackwell_gemv(
    const void* activation, const void* qweights, const void* block_scales,
    const void* global_scale, void* output, int64_t n, int64_t k, int64_t rows,
    apxinf_cuda_stream_t stream) {
  return abi_boundary([&] {
    if (activation == nullptr || qweights == nullptr ||
        block_scales == nullptr || global_scale == nullptr ||
        output == nullptr || !valid_extent(n) || !valid_extent(k) ||
        !valid_extent(rows) || n % 128 != 0 || k % 64 != 0) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                    "invalid NVFP4-A16 Blackwell GEMV arguments");
    }
    const auto& policy = nvfp4_a16_wmma_policy();
    if (policy.enabled && rows >= policy.min_rows && rows <= 4) {
      check(apxinf::cuda_new::mlp_ops::nvfp4_a16_blackwell_wmma(
                activation,qweights,block_scales,global_scale,output,
                static_cast<int>(n),static_cast<int>(k),static_cast<int>(rows),
                policy.tile_n,static_cast<cudaStream_t>(stream)),
            "NVFP4-A16 BF16 WMMA");
      return;
    }
    check(apxinf::cuda_new::mlp_ops::nvfp4_a16_blackwell_gemv(
              activation, qweights, block_scales, global_scale, output,
              static_cast<int>(n), static_cast<int>(k), static_cast<int>(rows),
              static_cast<cudaStream_t>(stream)),
          "NVFP4-A16 Blackwell GEMV");
  });
}
