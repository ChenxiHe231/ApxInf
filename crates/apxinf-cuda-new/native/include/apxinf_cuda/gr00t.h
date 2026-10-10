#pragma once

#include <cuda_runtime.h>

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* GR00T-family fused activation primitives. Copied verbatim from the legacy
   apxinf-cuda adapters with the apxinf_gr00t_ prefix, because the legacy crate
   links into the same binary and the legacy symbols are already present. */

cudaError_t apxinf_gr00t_silu_mul_separate_bf16(
    const void* gate, const void* up, void* output, uint32_t count,
    cudaStream_t stream);

cudaError_t apxinf_gr00t_silu_mul_quant_bf16_e4m3(
    const void* gate, const void* up, void* output, int64_t count, float scale,
    cudaStream_t stream);

cudaError_t apxinf_gr00t_bias_gelu_quant_bf16_e4m3(
    const void* input, const void* bias, void* output, int rows, int cols,
    float scale, cudaStream_t stream);

cudaError_t apxinf_gr00t_bias_gelu_bf16_packed8(
    const void* input, const void* bias, void* output, int rows, int cols,
    cudaStream_t stream);

/* Elementwise / row-selection primitives. Copied verbatim from the legacy
   static_bf16_adapter.cu with the apxinf_gr00t_ prefix. */

cudaError_t apxinf_gr00t_scatter_rows_bf16(
    const void* source, const void* rows, void* output, int row_count, int cols,
    int add, cudaStream_t stream);

cudaError_t apxinf_gr00t_bias_qkv_in_place_bf16(
    void* query, void* key, void* value, const void* query_bias,
    const void* key_bias, const void* value_bias, int rows, int cols,
    cudaStream_t stream);


cudaError_t apxinf_gr00t_adaptive_layer_norm_bf16(
    const void* input, const void* modulation, void* output, uint32_t rows,
    uint32_t cols, float eps, cudaStream_t stream);

cudaError_t apxinf_gr00t_adaptive_layer_norm_quant_bf16_e4m3(
    const void* input, const void* modulation, void* output, void* quantized,
    uint32_t rows, uint32_t cols, float eps, float scale, cudaStream_t stream);

cudaError_t apxinf_gr00t_rms_norm_quant_bf16_e4m3(
    const void* input, const void* weight, void* output, int rows, int cols,
    float eps, float scale, cudaStream_t stream);

cudaError_t apxinf_gr00t_layer_norm_quant_bf16_e4m3(
    const void* input, const void* weight, const void* bias, void* output,
    int rows, int cols, float eps, float scale, cudaStream_t stream);


cudaError_t apxinf_gr00t_bias_then_residual_bf16(
    const void* projection, const void* bias, const void* residual,
    void* output, int rows, int cols, cudaStream_t stream);

cudaError_t apxinf_gr00t_bias_residual_bf16_packed4(
    const void* projection, const void* bias, const void* residual,
    void* output, int rows, int cols, cudaStream_t stream);

cudaError_t apxinf_gr00t_bias_then_residual_bf16_packed4(
    const void* projection, const void* bias, const void* residual,
    void* output, int rows, int cols, cudaStream_t stream);

cudaError_t apxinf_gr00t_bias_residual_layer_norm_bf16_cached_1024(
    const void* projection, const void* projection_bias, const void* residual,
    const void* norm_weight, const void* norm_bias, void* hidden,
    void* normalized, int rows, int cols, float eps, cudaStream_t stream);

cudaError_t apxinf_gr00t_bias_then_residual_layer_norm_bf16_cached_1536(
    const void* projection, const void* projection_bias, const void* residual,
    const void* norm_weight, const void* norm_bias, void* hidden,
    void* normalized, int rows, int cols, float eps, cudaStream_t stream);

cudaError_t
apxinf_gr00t_bias_then_residual_adaptive_layer_norm_bf16_cached_1536(
    const void* projection, const void* projection_bias, const void* residual,
    const void* modulation, void* hidden, void* normalized, int rows, int cols,
    float eps, cudaStream_t stream);

cudaError_t apxinf_gr00t_bias_residual_layer_norm_quant_bf16_e4m3(
    const void* projection, const void* projection_bias, const void* residual,
    const void* norm_weight, const void* norm_bias, void* hidden,
    void* normalized, int rows, int cols, float eps, float scale,
    cudaStream_t stream);

#ifdef __cplusplus
}
#endif
