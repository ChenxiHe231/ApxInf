#include "../internal.h"

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

namespace apxinf::quantization {

cudaError_t launch_ported(Execution& execution) {
  const auto& spec = execution.spec;
  const auto& bindings = execution.bindings;
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

}  // namespace apxinf::quantization
