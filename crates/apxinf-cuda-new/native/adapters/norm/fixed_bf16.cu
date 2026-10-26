#include "../../kernels/norm/fixed_bf16.cuh"

extern "C" cudaError_t apxinf_norm_adaptive_layer_norm_bf16(
    const void* input, const void* modulation, void* output, uint32_t rows,
    uint32_t cols, float eps, cudaStream_t stream) {
  if (rows == 0 || cols == 0 || !(eps > 0.0f)) return cudaErrorInvalidValue;
  adaptive_layer_norm_bf16_kernel<<<rows, kBlockSize, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(input),
      static_cast<const __nv_bfloat16*>(modulation),
      static_cast<__nv_bfloat16*>(output), rows, cols, eps);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_norm_adaptive_layer_norm_quant_bf16_e4m3(
    const void* input, const void* modulation, void* output, void* quantized,
    uint32_t rows, uint32_t cols, float eps, float scale, cudaStream_t stream) {
  if (rows == 0 || cols == 0 || !(eps > 0.0f) || !(scale > 0.0f))
    return cudaErrorInvalidValue;
  adaptive_layer_norm_quant_bf16_e4m3_kernel<<<rows, kBlockSize, 0,
                                                     stream>>>(
      static_cast<const __nv_bfloat16*>(input),
      static_cast<const __nv_bfloat16*>(modulation),
      static_cast<__nv_bfloat16*>(output),
      static_cast<__nv_fp8_e4m3*>(quantized), rows, cols, eps, 1.0f / scale);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_norm_rms_norm_quant_bf16_e4m3(
    const void* input, const void* weight, void* output, int rows, int cols,
    float eps, float scale, cudaStream_t stream) {
  if (input == nullptr || weight == nullptr || output == nullptr || rows <= 0 ||
      cols <= 0 || !std::isfinite(scale) || scale <= 0.0f)
    return cudaErrorInvalidValue;
  rms_norm_quant_bf16_e4m3_kernel<<<rows, kBlockSize, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(input),
      static_cast<const __nv_bfloat16*>(weight),
      static_cast<__nv_fp8_e4m3*>(output), rows, cols, eps, 1.0f / scale);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_norm_layer_norm_quant_bf16_e4m3(
    const void* input, const void* weight, const void* bias, void* output,
    int rows, int cols, float eps, float scale, cudaStream_t stream) {
  if (input == nullptr || weight == nullptr || bias == nullptr ||
      output == nullptr || rows <= 0 || cols <= 0 || !std::isfinite(scale) ||
      scale <= 0.0f)
    return cudaErrorInvalidValue;
  const size_t shared_bytes = static_cast<size_t>(cols) * sizeof(float);
  layer_norm_quant_bf16_e4m3_kernel<<<rows, kBlockSize,
                                            shared_bytes, stream>>>(
      static_cast<const __nv_bfloat16*>(input),
      static_cast<const __nv_bfloat16*>(weight),
      static_cast<const __nv_bfloat16*>(bias),
      static_cast<__nv_fp8_e4m3*>(output), rows, cols, eps, 1.0f / scale);
  return cudaGetLastError();
}
