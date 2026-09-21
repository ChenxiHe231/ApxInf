#include "../include/apxinf_cuda/quantization.h"
#include "../framework/runtime_internal.h"

#include <climits>
#include <cmath>
#include <cstdint>
#include <string>

extern "C" cudaError_t apxinf_static_quantize_f16_e4m3(
    const void*, void*, int64_t, float, cudaStream_t);
extern "C" cudaError_t apxinf_static_quantize_bf16_e4m3(
    const void*, void*, int64_t, float, cudaStream_t);
extern "C" cudaError_t apxinf_dynamic_quantize_rows_bf16_e4m3(
    const void*, void*, void*, int, int, int, cudaStream_t);
extern "C" cudaError_t apxinf_static_cast_f16_bf16(
    const void*, void*, int64_t, cudaStream_t);
extern "C" cudaError_t apxinf_slice_columns_bf16(
    const void*, void*, int, int, int, cudaStream_t);
extern "C" cudaError_t apxinf_static_quantize_rows_bf16_int8(
    const void*, void*, void*, int, int, cudaStream_t);

namespace {

using apxinf::framework::Failure;

bool valid_alignment(uint32_t alignment) {
  return alignment <= 256 && alignment != 0 &&
         (alignment & (alignment - 1)) == 0;
}

bool has_row_scales(uint32_t semantic) {
  return semantic == APXINF_QUANTIZATION_SEMANTIC_ROWWISE_E4M3 ||
         semantic == APXINF_QUANTIZATION_SEMANTIC_ROWWISE_I8;
}

uint32_t dtype_bytes(uint32_t dtype) {
  switch (dtype) {
    case APXINF_DTYPE_F32:
      return 4;
    case APXINF_DTYPE_F16:
    case APXINF_DTYPE_BF16:
      return 2;
    case APXINF_DTYPE_E4M3:
    case APXINF_DTYPE_I8:
      return 1;
    default:
      return 0;
  }
}

void validate_recorded_alignment(const void* pointer, uint32_t alignment,
                                 const char* name) {
  if (pointer != nullptr &&
      reinterpret_cast<uintptr_t>(pointer) % alignment != 0) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                  std::string(name) + " violates its alignment class");
  }
}

void validate_spec(const apxinf_quantization_spec_t& spec) {
  if (spec.version != APXINF_QUANTIZATION_SPEC_VERSION ||
      spec.semantic > APXINF_QUANTIZATION_SEMANTIC_ROWWISE_I8 ||
      spec.scale_dtype != APXINF_DTYPE_F32 || spec.rows <= 0 ||
      spec.input_cols <= 0 || spec.output_cols <= 0 ||
      spec.rows > INT32_MAX || spec.input_cols > INT32_MAX ||
      spec.output_cols > INT32_MAX ||
      !valid_alignment(spec.input_alignment) ||
      !valid_alignment(spec.output_alignment) ||
      !valid_alignment(spec.scales_alignment)) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                  "invalid Quantization Spec");
  }
  if (spec.input_alignment < dtype_bytes(spec.input_dtype) ||
      spec.output_alignment < dtype_bytes(spec.output_dtype) ||
      (has_row_scales(spec.semantic) &&
       spec.scales_alignment < sizeof(float))) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                  "Quantization binding violates dtype alignment");
  }
  switch (spec.semantic) {
    case APXINF_QUANTIZATION_SEMANTIC_FIXED_E4M3:
      if ((spec.input_dtype != APXINF_DTYPE_F16 &&
           spec.input_dtype != APXINF_DTYPE_BF16) ||
          spec.output_dtype != APXINF_DTYPE_E4M3 ||
          spec.input_cols != spec.output_cols) {
        throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                      "invalid fixed-scale E4M3 Spec");
      }
      break;
    case APXINF_QUANTIZATION_SEMANTIC_ROWWISE_E4M3:
      if (spec.input_dtype != APXINF_DTYPE_BF16 ||
          spec.output_dtype != APXINF_DTYPE_E4M3 ||
          spec.output_cols < spec.input_cols) {
        throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                      "invalid rowwise E4M3 Spec");
      }
      break;
    case APXINF_QUANTIZATION_SEMANTIC_CAST_F16_BF16:
      if (spec.input_dtype != APXINF_DTYPE_F16 ||
          spec.output_dtype != APXINF_DTYPE_BF16 ||
          spec.input_cols != spec.output_cols) {
        throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                      "invalid F16-to-BF16 cast Spec");
      }
      break;
    case APXINF_QUANTIZATION_SEMANTIC_SLICE_BF16:
      if (spec.input_dtype != APXINF_DTYPE_BF16 ||
          spec.output_dtype != APXINF_DTYPE_BF16 ||
          spec.output_cols > spec.input_cols) {
        throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                      "invalid BF16 slice Spec");
      }
      break;
    case APXINF_QUANTIZATION_SEMANTIC_ROWWISE_I8:
      if (spec.input_dtype != APXINF_DTYPE_BF16 ||
          spec.output_dtype != APXINF_DTYPE_I8 ||
          spec.input_cols != spec.output_cols) {
        throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                      "invalid rowwise INT8 Spec");
      }
      break;
    default:
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                    "unknown Quantization semantic");
  }
}

void validate_bindings(const apxinf_quantization_spec_t& spec,
                       const apxinf_quantization_bindings_t& bindings) {
  if (bindings.input == nullptr || bindings.output == nullptr) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                  "missing Quantization binding");
  }
  if (has_row_scales(spec.semantic) != (bindings.scales != nullptr)) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                  "Quantization scales binding disagrees with semantic");
  }
  if (spec.semantic == APXINF_QUANTIZATION_SEMANTIC_FIXED_E4M3) {
    if (!(bindings.scale > 0.0f) || !std::isfinite(bindings.scale)) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                    "fixed E4M3 scale must be finite and positive");
    }
  } else if (bindings.scale != 1.0f) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                  "unused Quantization scale must be one");
  }
  validate_recorded_alignment(bindings.input, spec.input_alignment,
                              "Quantization input");
  validate_recorded_alignment(bindings.output, spec.output_alignment,
                              "Quantization output");
  validate_recorded_alignment(bindings.scales, spec.scales_alignment,
                              "Quantization scales");
}

cudaError_t launch(const apxinf_quantization_spec_t& spec,
                   const apxinf_quantization_bindings_t& bindings) {
  auto stream = static_cast<cudaStream_t>(bindings.stream);
  const int rows = static_cast<int>(spec.rows);
  const int input_cols = static_cast<int>(spec.input_cols);
  const int output_cols = static_cast<int>(spec.output_cols);
  const int64_t input_count = spec.rows * spec.input_cols;

  switch (spec.semantic) {
    case APXINF_QUANTIZATION_SEMANTIC_FIXED_E4M3:
      return spec.input_dtype == APXINF_DTYPE_F16
                 ? apxinf_static_quantize_f16_e4m3(
                       bindings.input, bindings.output, input_count,
                       bindings.scale, stream)
                 : apxinf_static_quantize_bf16_e4m3(
                       bindings.input, bindings.output, input_count,
                       bindings.scale, stream);
    case APXINF_QUANTIZATION_SEMANTIC_ROWWISE_E4M3:
      return apxinf_dynamic_quantize_rows_bf16_e4m3(
          bindings.input, bindings.output, bindings.scales, rows, input_cols,
          output_cols, stream);
    case APXINF_QUANTIZATION_SEMANTIC_CAST_F16_BF16:
      return apxinf_static_cast_f16_bf16(
          bindings.input, bindings.output, input_count, stream);
    case APXINF_QUANTIZATION_SEMANTIC_SLICE_BF16:
      return apxinf_slice_columns_bf16(bindings.input, bindings.output, rows,
                                       input_cols, output_cols, stream);
    case APXINF_QUANTIZATION_SEMANTIC_ROWWISE_I8:
      return apxinf_static_quantize_rows_bf16_int8(
          bindings.input, bindings.output, bindings.scales, rows, input_cols,
          stream);
    default:
      return cudaErrorInvalidValue;
  }
}

}  // namespace

extern "C" apxinf_status_t apxinf_quantization_launch(
    apxinf_runtime_t runtime, const apxinf_quantization_spec_t* spec,
    const apxinf_quantization_bindings_t* bindings) {
  return apxinf::framework::abi_boundary([&] {
    if (runtime == nullptr || spec == nullptr || bindings == nullptr) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                    "null Quantization argument");
    }
    validate_spec(*spec);
    validate_bindings(*spec, *bindings);
    apxinf::framework::check_cuda(cudaSetDevice(runtime->device));
    apxinf::framework::check_cuda(launch(*spec, *bindings));
  });
}
