// Elementwise / MLP kernels: RMSNorm, SwiGLU, residual add, FP8 quantize, FP8 GEMV, NVFP4 GEMV.
//
// These ops have a single fixed implementation and no persisted selection,
// so per doc/adding-new-kernels.md §6 they carry no candidate registry,
// tuning key, or autotuner; they are direct C-ABI forwarders.

#include "../../include/apxinf_cuda/mlp.h"

#include "../../framework/runtime_internal.h"
#include "../../kernels/custom/mlp_ops.h"

#include <cmath>
#include <cstdint>
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

extern "C" apxinf_status_t apxinf_new_add_rms_norm_bf16(
    const void* addend, void* residual, const void* weight, void* normalized,
    float epsilon, apxinf_cuda_stream_t stream) {
  return abi_boundary([&] {
    const auto aligned = [](const void* ptr) {
      return ptr != nullptr && reinterpret_cast<uintptr_t>(ptr) % 16 == 0;
    };
    if (!aligned(addend) || !aligned(residual) || !aligned(weight) ||
        !aligned(normalized) || !std::isfinite(epsilon) || epsilon < 0.0f) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT, "invalid fused BF16 residual RMSNorm binding");
    }
    constexpr size_t weight_bytes = 2048 * sizeof(uint16_t);
    constexpr size_t matrix_bytes = 4096 * weight_bytes;
    const auto overlaps = [](const void* a, size_t a_bytes, const void* b, size_t b_bytes) {
      const auto first = reinterpret_cast<uintptr_t>(a);
      const auto second = reinterpret_cast<uintptr_t>(b);
      return first < second ? second - first < a_bytes : first - second < b_bytes;
    };
    if (overlaps(residual, matrix_bytes, normalized, matrix_bytes) ||
        overlaps(residual, matrix_bytes, addend, matrix_bytes) ||
        overlaps(residual, matrix_bytes, weight, weight_bytes) ||
        overlaps(normalized, matrix_bytes, addend, matrix_bytes) ||
        overlaps(normalized, matrix_bytes, weight, weight_bytes)) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT, "fused BF16 residual RMSNorm buffers overlap");
    }
    check(apxinf::cuda_new::mlp_ops::add_rms_norm_bf16(
              addend, residual, weight, normalized, epsilon,
              static_cast<cudaStream_t>(stream)),
          "fused BF16 residual RMSNorm");
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
    check(apxinf::cuda_new::mlp_ops::nvfp4_a16_blackwell_gemv(
              activation, qweights, block_scales, global_scale, output,
              static_cast<int>(n), static_cast<int>(k), static_cast<int>(rows),
              static_cast<cudaStream_t>(stream)),
          "NVFP4-A16 Blackwell GEMV");
  });
}
