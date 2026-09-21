#pragma once

#include "types.h"
#include "status.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Elementwise and reduction operators used by transformer MLP blocks.
   Each has a single implementation and no tuned selection, so they are plain
   entry points rather than registry-backed operators. */

apxinf_status_t apxinf_rms_norm_bf16(const void* input, const void* weight,
                                     void* output, int64_t rows, int64_t width,
                                     float epsilon,
                                     apxinf_cuda_stream_t stream);

/* SwiGLU over a fused [rows, 2*width] gate/up projection, gate first. */
apxinf_status_t apxinf_swiglu_bf16(const void* fused_gate_up, void* output,
                                   int64_t rows, int64_t width,
                                   apxinf_cuda_stream_t stream);

apxinf_status_t apxinf_add_bf16(const void* addend, void* accumulator,
                                int64_t count, apxinf_cuda_stream_t stream);

/* BF16 -> E4M3 against a single per-tensor scale, for the FP8 projections a
   ModelOpt checkpoint stores with scalar scales. */
apxinf_status_t apxinf_quantize_fp8_per_tensor(const void* input, void* output,
                                               int64_t count,
                                               float input_scale,
                                               apxinf_cuda_stream_t stream);

#ifdef __cplusplus
}
#endif
