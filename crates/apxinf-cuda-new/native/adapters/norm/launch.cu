#include "../../include/apxinf_cuda/norm.h"
#include "../../framework/runtime_internal.h"
#include "../../kernels/custom/norm.cuh"

#include <climits>
#include <cmath>
#include <cstdint>
#include <cuda_fp8.h>

extern "C" cudaError_t apxinf_norm_rms_norm_f16(
    const void* input, const void* weight, void* output, int rows, int cols,
    float eps, cudaStream_t stream);
extern "C" cudaError_t apxinf_norm_layer_norm_f16(
    const void* input, const void* weight, const void* bias, void* output,
    int rows, int cols, float eps, cudaStream_t stream);
extern "C" cudaError_t apxinf_norm_bias_then_residual_bf16(
    const void* projection, const void* bias, const void* residual,
    void* output, int rows, int cols, cudaStream_t stream);
extern "C" cudaError_t apxinf_norm_bias_residual_bf16_packed4(
    const void* projection, const void* bias, const void* residual,
    void* output, int rows, int cols, cudaStream_t stream);
extern "C" cudaError_t apxinf_norm_bias_then_residual_bf16_packed4(
    const void* projection, const void* bias, const void* residual,
    void* output, int rows, int cols, cudaStream_t stream);
extern "C" cudaError_t apxinf_norm_bias_residual_layer_norm_bf16_cached_1024(
    const void* projection, const void* projection_bias, const void* residual,
    const void* norm_weight, const void* norm_bias, void* hidden,
    void* normalized, int rows, int cols, float eps, cudaStream_t stream);
extern "C" cudaError_t apxinf_norm_bias_then_residual_layer_norm_bf16_cached_1536(
    const void* projection, const void* projection_bias, const void* residual,
    const void* norm_weight, const void* norm_bias, void* hidden,
    void* normalized, int rows, int cols, float eps, cudaStream_t stream);
extern "C" cudaError_t
apxinf_norm_bias_then_residual_adaptive_layer_norm_bf16_cached_1536(
    const void* projection, const void* projection_bias, const void* residual,
    const void* modulation, void* hidden, void* normalized, int rows, int cols,
    float eps, cudaStream_t stream);
extern "C" cudaError_t apxinf_norm_adaptive_layer_norm_bf16(
    const void* input, const void* modulation, void* output, uint32_t rows,
    uint32_t cols, float eps, cudaStream_t stream);

namespace {

using apxinf::framework::Failure;
namespace kernels = apxinf::norm::kernels;

constexpr int kThreads = 256;
constexpr int kMaxElementwiseBlocks = 4096;

bool valid_alignment(uint32_t alignment) {
  return alignment <= 256 && alignment != 0 &&
         (alignment & (alignment - 1)) == 0;
}

bool writes_hidden(uint32_t semantic) {
  return semantic != APXINF_NORM_SEMANTIC_RMS &&
         semantic != APXINF_NORM_SEMANTIC_LAYER &&
         semantic != APXINF_NORM_SEMANTIC_ADAPTIVE_RMS;
}

bool writes_normalized(uint32_t semantic) {
  return semantic != APXINF_NORM_SEMANTIC_BIAS_RESIDUAL &&
         semantic != APXINF_NORM_SEMANTIC_ADA_GATE_RESIDUAL;
}

bool reads_residual(uint32_t semantic) { return writes_hidden(semantic); }

bool reads_weight(uint32_t semantic) {
  return semantic == APXINF_NORM_SEMANTIC_RMS ||
         semantic == APXINF_NORM_SEMANTIC_LAYER ||
         semantic == APXINF_NORM_SEMANTIC_BIAS_RESIDUAL_RMS ||
         semantic == APXINF_NORM_SEMANTIC_BIAS_RESIDUAL_LAYER;
}

bool reads_norm_bias(uint32_t semantic) {
  return semantic == APXINF_NORM_SEMANTIC_LAYER ||
         semantic == APXINF_NORM_SEMANTIC_BIAS_RESIDUAL_LAYER;
}

bool reads_norm_style(uint32_t semantic) {
  return semantic == APXINF_NORM_SEMANTIC_ADAPTIVE_RMS ||
         semantic == APXINF_NORM_SEMANTIC_ADA_GATE_RESIDUAL_RMS ||
         semantic == APXINF_NORM_SEMANTIC_BIAS_RESIDUAL_ADAPTIVE_LAYER;
}

bool reads_gate_style(uint32_t semantic) {
  return semantic == APXINF_NORM_SEMANTIC_ADA_GATE_RESIDUAL ||
         semantic == APXINF_NORM_SEMANTIC_ADA_GATE_RESIDUAL_RMS;
}

bool may_have_bias(uint32_t semantic) {
  return semantic == APXINF_NORM_SEMANTIC_BIAS_RESIDUAL ||
         semantic == APXINF_NORM_SEMANTIC_BIAS_RESIDUAL_RMS ||
         semantic == APXINF_NORM_SEMANTIC_BIAS_RESIDUAL_LAYER ||
         semantic == APXINF_NORM_SEMANTIC_BIAS_RESIDUAL_ADAPTIVE_LAYER;
}

void validate_spec(const apxinf_norm_spec_t& spec) {
  const bool quantized_output =
      spec.dtype == APXINF_DTYPE_F16 &&
      spec.output_dtype == APXINF_DTYPE_E4M3 &&
      writes_normalized(spec.semantic);
  if (spec.version != APXINF_NORM_SPEC_VERSION ||
      spec.semantic > APXINF_NORM_SEMANTIC_BIAS_RESIDUAL_ADAPTIVE_LAYER ||
      (spec.dtype != APXINF_DTYPE_F16 && spec.dtype != APXINF_DTYPE_BF16) ||
      (spec.output_dtype != spec.dtype && !quantized_output) ||
      spec.rows <= 0 || spec.cols <= 0 ||
      spec.rows > INT32_MAX || spec.cols > INT32_MAX || spec.has_bias > 1 ||
      spec.bias_order > APXINF_NORM_BIAS_ORDER_THEN_RESIDUAL ||
      spec.output_scale_is_unit > 1 ||
      !valid_alignment(spec.input_alignment) ||
      !valid_alignment(spec.weight_alignment) ||
      !valid_alignment(spec.bias_alignment) ||
      !valid_alignment(spec.residual_alignment) ||
      !valid_alignment(spec.style_alignment) ||
      !valid_alignment(spec.hidden_alignment) ||
      !valid_alignment(spec.normalized_alignment)) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT, "invalid Norm Spec");
  }
  if (spec.bias_order == APXINF_NORM_BIAS_ORDER_THEN_RESIDUAL &&
      spec.dtype != APXINF_DTYPE_BF16) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                  "BiasThenResidual order requires BF16 input and output");
  }
  if (spec.semantic == APXINF_NORM_SEMANTIC_BIAS_RESIDUAL_ADAPTIVE_LAYER &&
      spec.dtype != APXINF_DTYPE_BF16) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                  "BiasResidualAdaptiveLayer requires BF16 input and output");
  }
  if (spec.bias_order == APXINF_NORM_BIAS_ORDER_THEN_RESIDUAL &&
      spec.semantic != APXINF_NORM_SEMANTIC_BIAS_RESIDUAL &&
      spec.semantic != APXINF_NORM_SEMANTIC_BIAS_RESIDUAL_LAYER &&
      spec.semantic != APXINF_NORM_SEMANTIC_BIAS_RESIDUAL_ADAPTIVE_LAYER) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                  "BiasThenResidual order requires residual combine");
  }
  if (spec.has_bias != 0 && !may_have_bias(spec.semantic)) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                  "Norm semantic does not take a bias");
  }
}

void validate_bindings(const apxinf_norm_spec_t& spec,
                       const apxinf_norm_bindings_t& bindings) {
  const uint32_t semantic = spec.semantic;
  const bool missing =
      bindings.input == nullptr ||
      (writes_hidden(semantic) && bindings.hidden == nullptr) ||
      (writes_normalized(semantic) && bindings.normalized == nullptr) ||
      (reads_residual(semantic) && bindings.residual == nullptr) ||
      (reads_weight(semantic) && bindings.weight == nullptr) ||
      (reads_norm_bias(semantic) && bindings.norm_bias == nullptr) ||
      (reads_norm_style(semantic) && bindings.norm_style == nullptr) ||
      (reads_gate_style(semantic) && bindings.gate_style == nullptr);
  if (missing) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                  "missing Norm binding for semantic");
  }
  if ((spec.has_bias != 0) != (bindings.bias != nullptr)) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                  "Norm bias binding disagrees with Spec.has_bias");
  }
  const bool scale_matches_spec =
      (bindings.output_scale == 1.0f) ==
      (spec.output_scale_is_unit != 0);
  const bool valid_output_scale =
      std::isfinite(bindings.output_scale) && bindings.output_scale > 0.0f &&
      scale_matches_spec &&
      (spec.output_dtype == APXINF_DTYPE_E4M3 ||
       bindings.output_scale == 1.0f);
  if (!valid_output_scale || !(bindings.eps > 0.0f) ||
      !std::isfinite(bindings.eps)) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT, "invalid Norm scalar");
  }
}

int elementwise_blocks(int64_t count) {
  const int64_t blocks = (count + kThreads - 1) / kThreads;
  if (blocks < 1) return 1;
  return static_cast<int>(blocks < kMaxElementwiseBlocks
                              ? blocks
                              : kMaxElementwiseBlocks);
}

cudaError_t launch_quantized_f16(const apxinf_norm_spec_t& spec,
                                 const apxinf_norm_bindings_t& bindings) {
  const auto* input = static_cast<const half*>(bindings.input);
  const auto* bias = static_cast<const half*>(bindings.bias);
  const auto* residual = static_cast<const half*>(bindings.residual);
  const auto* weight = static_cast<const half*>(bindings.weight);
  const auto* norm_bias = static_cast<const half*>(bindings.norm_bias);
  const auto* norm_style = static_cast<const half*>(bindings.norm_style);
  const auto* gate_style = static_cast<const half*>(bindings.gate_style);
  auto* hidden = static_cast<half*>(bindings.hidden);
  auto* normalized = static_cast<__nv_fp8_e4m3*>(bindings.normalized);
  const int rows = static_cast<int>(spec.rows);
  const int cols = static_cast<int>(spec.cols);
  const float inverse_scale = 1.0f / bindings.output_scale;
  auto stream = static_cast<cudaStream_t>(bindings.stream);

  switch (spec.semantic) {
    case APXINF_NORM_SEMANTIC_RMS:
      kernels::rms_norm_quant_f16_e4m3<<<rows, kThreads, 0, stream>>>(
          input, weight, normalized, rows, cols, bindings.eps, inverse_scale);
      break;
    case APXINF_NORM_SEMANTIC_LAYER:
      kernels::layer_norm_quant_f16_e4m3<<<rows, kThreads, 0, stream>>>(
          input, weight, norm_bias, normalized, rows, cols, bindings.eps,
          inverse_scale);
      break;
    case APXINF_NORM_SEMANTIC_ADAPTIVE_RMS:
      kernels::ada_rms_norm_quant_f16_e4m3<<<rows, kThreads, 0, stream>>>(
          input, norm_style, normalized, rows, cols, bindings.eps,
          inverse_scale);
      break;
    case APXINF_NORM_SEMANTIC_BIAS_RESIDUAL_RMS:
      kernels::bias_residual_rms_norm_quant_f16_e4m3
          <<<rows, kThreads, 0, stream>>>(
              input, bias, residual, weight, hidden, normalized, rows, cols,
              bindings.eps, inverse_scale);
      break;
    case APXINF_NORM_SEMANTIC_BIAS_RESIDUAL_LAYER:
      kernels::bias_residual_layer_norm_quant_f16_e4m3
          <<<rows, kThreads, 0, stream>>>(
              input, bias, residual, weight, norm_bias, hidden, normalized,
              rows, cols, bindings.eps, inverse_scale);
      break;
    case APXINF_NORM_SEMANTIC_ADA_GATE_RESIDUAL_RMS:
      // The 10x1024 specialization performs uint2 loads on norm_style and a
      // uint32 store on normalized; gate on the actual pointer alignment.
      if (rows == 10 && cols == 1024 &&
          reinterpret_cast<uintptr_t>(norm_style) % 8 == 0 &&
          reinterpret_cast<uintptr_t>(normalized) % 4 == 0) {
        kernels::ada_gate_residual_rms_norm_quant_f16_e4m3_10x1024
            <<<rows, kThreads, 0, stream>>>(
                input, residual, gate_style, norm_style, hidden, normalized,
                bindings.eps, inverse_scale);
      } else {
        kernels::ada_gate_residual_rms_norm_quant_f16_e4m3
            <<<rows, kThreads, 0, stream>>>(
                input, residual, gate_style, norm_style, hidden, normalized,
                rows, cols, bindings.eps, inverse_scale);
      }
      break;
    default:
      return cudaErrorInvalidValue;
  }
  return cudaGetLastError();
}

template <class T>
cudaError_t launch(const apxinf_norm_spec_t& spec,
                   const apxinf_norm_bindings_t& bindings);

cudaError_t launch_fixed_f16(const apxinf_norm_spec_t& spec,
                             const apxinf_norm_bindings_t& bindings) {
  auto stream = static_cast<cudaStream_t>(bindings.stream);
  const int rows = static_cast<int>(spec.rows);
  const int cols = static_cast<int>(spec.cols);
  if (spec.semantic == APXINF_NORM_SEMANTIC_RMS) {
    return apxinf_norm_rms_norm_f16(
        bindings.input, bindings.weight, bindings.normalized, rows, cols,
        bindings.eps, stream);
  }
  if (spec.semantic == APXINF_NORM_SEMANTIC_LAYER) {
    return apxinf_norm_layer_norm_f16(
        bindings.input, bindings.weight, bindings.norm_bias,
        bindings.normalized, rows, cols, bindings.eps, stream);
  }
  return launch<__half>(spec, bindings);
}

cudaError_t launch_fixed_bf16(const apxinf_runtime& runtime,
                              const apxinf_norm_spec_t& spec,
                              const apxinf_norm_bindings_t& bindings) {
  auto stream = static_cast<cudaStream_t>(bindings.stream);
  const int rows = static_cast<int>(spec.rows);
  const int cols = static_cast<int>(spec.cols);
  const int sm = runtime.compute_major * 10 + runtime.compute_minor;
  const bool bias_then_residual =
      spec.bias_order == APXINF_NORM_BIAS_ORDER_THEN_RESIDUAL;
  const bool packed4 = (cols & 3) == 0 && spec.input_alignment >= 8 &&
      spec.bias_alignment >= 8 && spec.residual_alignment >= 8 &&
      spec.hidden_alignment >= 8;

  if (spec.semantic == APXINF_NORM_SEMANTIC_BIAS_RESIDUAL) {
    if (bias_then_residual && packed4) {
      return apxinf_norm_bias_then_residual_bf16_packed4(
          bindings.input, bindings.bias, bindings.residual, bindings.hidden,
          rows, cols, stream);
    }
    if (bias_then_residual) {
      return apxinf_norm_bias_then_residual_bf16(
          bindings.input, bindings.bias, bindings.residual, bindings.hidden,
          rows, cols, stream);
    }
    if (bindings.bias != nullptr && packed4) {
      return apxinf_norm_bias_residual_bf16_packed4(
          bindings.input, bindings.bias, bindings.residual, bindings.hidden,
          rows, cols, stream);
    }
    return launch<__nv_bfloat16>(spec, bindings);
  }

  if (spec.semantic == APXINF_NORM_SEMANTIC_BIAS_RESIDUAL_LAYER) {
    if (!bias_then_residual && sm == 87 && rows == 512 && cols == 1024) {
      return apxinf_norm_bias_residual_layer_norm_bf16_cached_1024(
          bindings.input, bindings.bias, bindings.residual, bindings.weight,
          bindings.norm_bias, bindings.hidden, bindings.normalized, rows, cols,
          bindings.eps, stream);
    }
    if (bias_then_residual && (sm == 87 || sm == 110) && rows == 41 &&
        cols == 1536 && bindings.bias != nullptr) {
      return apxinf_norm_bias_then_residual_layer_norm_bf16_cached_1536(
          bindings.input, bindings.bias, bindings.residual, bindings.weight,
          bindings.norm_bias, bindings.hidden, bindings.normalized, rows, cols,
          bindings.eps, stream);
    }
    if (!bias_then_residual) {
      return launch<__nv_bfloat16>(spec, bindings);
    }
    cudaError_t error = apxinf_norm_bias_then_residual_bf16(
        bindings.input, bindings.bias, bindings.residual, bindings.hidden,
        rows, cols, stream);
    if (error != cudaSuccess) return error;
    kernels::layer_norm<__nv_bfloat16><<<rows, kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(bindings.hidden),
        static_cast<const __nv_bfloat16*>(bindings.weight),
        static_cast<const __nv_bfloat16*>(bindings.norm_bias),
        static_cast<__nv_bfloat16*>(bindings.normalized), rows, cols,
        bindings.eps);
    return cudaGetLastError();
  }

  if (spec.semantic == APXINF_NORM_SEMANTIC_BIAS_RESIDUAL_ADAPTIVE_LAYER) {
    if (bias_then_residual && (sm == 87 || sm == 110) && rows == 41 &&
        cols == 1536 && bindings.bias != nullptr) {
      return apxinf_norm_bias_then_residual_adaptive_layer_norm_bf16_cached_1536(
          bindings.input, bindings.bias, bindings.residual,
          bindings.norm_style, bindings.hidden, bindings.normalized, rows,
          cols, bindings.eps, stream);
    }
    cudaError_t error;
    if (bias_then_residual) {
      error = apxinf_norm_bias_then_residual_bf16(
          bindings.input, bindings.bias, bindings.residual, bindings.hidden,
          rows, cols, stream);
    } else {
      const int64_t count = spec.rows * spec.cols;
      kernels::bias_residual<__nv_bfloat16>
          <<<elementwise_blocks(count), kThreads, 0, stream>>>(
              static_cast<const __nv_bfloat16*>(bindings.input),
              static_cast<const __nv_bfloat16*>(bindings.bias),
              static_cast<const __nv_bfloat16*>(bindings.residual),
              static_cast<__nv_bfloat16*>(bindings.hidden), count, cols);
      error = cudaGetLastError();
    }
    if (error != cudaSuccess) return error;
    return apxinf_norm_adaptive_layer_norm_bf16(
        bindings.hidden, bindings.norm_style, bindings.normalized,
        static_cast<uint32_t>(rows), static_cast<uint32_t>(cols),
        bindings.eps, stream);
  }
  return launch<__nv_bfloat16>(spec, bindings);
}

template <class T>
cudaError_t launch(const apxinf_norm_spec_t& spec,
                   const apxinf_norm_bindings_t& bindings) {
  const auto* input = static_cast<const T*>(bindings.input);
  const auto* bias = static_cast<const T*>(bindings.bias);
  const auto* residual = static_cast<const T*>(bindings.residual);
  const auto* weight = static_cast<const T*>(bindings.weight);
  const auto* norm_bias = static_cast<const T*>(bindings.norm_bias);
  const auto* norm_style = static_cast<const T*>(bindings.norm_style);
  const auto* gate_style = static_cast<const T*>(bindings.gate_style);
  auto* hidden = static_cast<T*>(bindings.hidden);
  auto* normalized = static_cast<T*>(bindings.normalized);
  const int rows = static_cast<int>(spec.rows);
  const int cols = static_cast<int>(spec.cols);
  const int64_t count = spec.rows * spec.cols;
  auto stream = static_cast<cudaStream_t>(bindings.stream);

  switch (spec.semantic) {
    case APXINF_NORM_SEMANTIC_RMS:
      kernels::rms_norm<T><<<rows, kThreads, 0, stream>>>(
          input, weight, normalized, rows, cols, bindings.eps);
      break;
    case APXINF_NORM_SEMANTIC_LAYER:
      kernels::layer_norm<T><<<rows, kThreads, 0, stream>>>(
          input, weight, norm_bias, normalized, rows, cols, bindings.eps);
      break;
    case APXINF_NORM_SEMANTIC_ADAPTIVE_RMS:
      kernels::ada_rms_norm<T><<<rows, kThreads, 0, stream>>>(
          input, norm_style, normalized, rows, cols, bindings.eps);
      break;
    case APXINF_NORM_SEMANTIC_BIAS_RESIDUAL:
      kernels::bias_residual<T>
          <<<elementwise_blocks(count), kThreads, 0, stream>>>(
              input, bias, residual, hidden, count, cols);
      break;
    case APXINF_NORM_SEMANTIC_BIAS_RESIDUAL_RMS:
      kernels::bias_residual_rms_norm<T><<<rows, kThreads, 0, stream>>>(
          input, bias, residual, weight, hidden, normalized, rows, cols,
          bindings.eps);
      break;
    case APXINF_NORM_SEMANTIC_BIAS_RESIDUAL_LAYER:
      kernels::bias_residual_layer_norm<T><<<rows, kThreads, 0, stream>>>(
          input, bias, residual, weight, norm_bias, hidden, normalized, rows,
          cols, bindings.eps);
      break;
    case APXINF_NORM_SEMANTIC_ADA_GATE_RESIDUAL:
      kernels::ada_gate_residual<T>
          <<<elementwise_blocks(count), kThreads, 0, stream>>>(
              input, residual, gate_style, hidden, count, cols);
      break;
    case APXINF_NORM_SEMANTIC_ADA_GATE_RESIDUAL_RMS:
      kernels::ada_gate_residual_rms_norm<T><<<rows, kThreads, 0, stream>>>(
          input, residual, gate_style, norm_style, hidden, normalized, rows,
          cols, bindings.eps);
      break;
    case APXINF_NORM_SEMANTIC_BIAS_RESIDUAL_ADAPTIVE_LAYER:
      return cudaErrorInvalidValue;
    default:
      return cudaErrorInvalidValue;
  }
  return cudaGetLastError();
}

}  // namespace

extern "C" apxinf_status_t apxinf_norm_launch(
    apxinf_runtime_t runtime, const apxinf_norm_spec_t* spec,
    const apxinf_norm_bindings_t* bindings) {
  return apxinf::framework::abi_boundary([&] {
    if (runtime == nullptr || spec == nullptr || bindings == nullptr) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT, "null Norm argument");
    }
    validate_spec(*spec);
    validate_bindings(*spec, *bindings);
    apxinf::framework::check_cuda(cudaSetDevice(runtime->device));
    apxinf::framework::check_cuda(
        spec->output_dtype == APXINF_DTYPE_E4M3
            ? launch_quantized_f16(*spec, *bindings)
            : (spec->dtype == APXINF_DTYPE_BF16
                   ? launch_fixed_bf16(*runtime, *spec, *bindings)
                   : launch_fixed_f16(*spec, *bindings)));
  });
}
