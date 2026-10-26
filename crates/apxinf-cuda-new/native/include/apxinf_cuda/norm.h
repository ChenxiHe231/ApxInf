#pragma once

#include "norm_types.h"

#ifdef __cplusplus
extern "C" {
#endif

apxinf_status_t apxinf_norm_head_rms(
    void* data, const void* weight, int64_t rows, int64_t head_dim,
    float epsilon, apxinf_cuda_stream_t stream);

apxinf_status_t apxinf_norm_rms_weight_offset(
    const void* input, const void* weight, void* output, int64_t rows,
    int64_t width, float epsilon, apxinf_cuda_stream_t stream);

apxinf_status_t apxinf_norm_rms_norm_rows_bf16_e4m3(
    const void* input, const void* weight, void* output, void* scales,
    int32_t rows, int32_t input_cols, int32_t output_cols, float eps,
    apxinf_cuda_stream_t stream);

apxinf_status_t apxinf_norm_bias_residual_rms_norm_rows_bf16_e4m3(
    const void* projection, const void* bias, const void* residual,
    const void* weight, void* hidden, void* normalized, void* scales,
    int32_t rows, int32_t cols, int32_t output_cols, float eps,
    apxinf_cuda_stream_t stream);

apxinf_status_t apxinf_norm_launch(
    apxinf_runtime_t runtime, const apxinf_norm_spec_t* spec,
    const apxinf_norm_bindings_t* bindings);

#ifdef __cplusplus
}
#endif
