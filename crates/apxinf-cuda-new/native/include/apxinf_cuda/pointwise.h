#pragma once

#include "pointwise_types.h"

#ifdef __cplusplus
extern "C" {
#endif

apxinf_status_t apxinf_pointwise_swiglu(
    const void* fused_gate_up, void* output, int64_t rows, int64_t width,
    apxinf_cuda_stream_t stream);
apxinf_status_t apxinf_pointwise_add_into(
    const void* addend, void* accumulator, int64_t count,
    apxinf_cuda_stream_t stream);

apxinf_status_t apxinf_pointwise_swiglu_rows_bf16_e4m3(
    const void* gate_up, const void* bias, void* output, void* scales,
    int32_t rows, int32_t input_cols, int32_t inner, int32_t output_cols,
    apxinf_cuda_stream_t stream);

apxinf_status_t apxinf_pointwise_launch(
    apxinf_runtime_t runtime, const apxinf_pointwise_spec_t* spec,
    const apxinf_pointwise_bindings_t* bindings);

#ifdef __cplusplus
}
#endif
