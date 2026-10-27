#pragma once

#include "types.h"
#include "status.h"

#ifdef __cplusplus
extern "C" {
#endif
/* Elementwise and reduction operators used by transformer MLP blocks.
   Each has a single implementation and no tuned selection, so they are plain
   entry points rather than registry-backed operators. */

apxinf_status_t apxinf_new_rms_norm_bf16(const void* input, const void* weight,
                                     void* output, int64_t rows, int64_t width,
                                     float epsilon,
                                     apxinf_cuda_stream_t stream);

/* SwiGLU over a fused [rows, 2*width] gate/up projection, gate first. */
apxinf_status_t apxinf_swiglu_bf16(const void* fused_gate_up, void* output,
                                   int64_t rows, int64_t width,
                                   apxinf_cuda_stream_t stream);

/* SwiGLU over separate contiguous [rows, width] gate and up matrices. */
apxinf_status_t apxinf_swiglu_split_bf16(
    const void* gate, const void* up, void* output, int64_t rows,
    int64_t width, apxinf_cuda_stream_t stream);

/* SwiGLU with only the final product rounded to BF16 (Qwen3.5 MTP order). */
apxinf_status_t apxinf_swiglu_bf16_round_product(
    const void* fused_gate_up, void* output, int64_t rows, int64_t width,
    apxinf_cuda_stream_t stream);

apxinf_status_t apxinf_new_add_bf16(const void* addend, void* accumulator,
                                int64_t count, apxinf_cuda_stream_t stream);

/* BF16 -> E4M3 against a single per-tensor scale, for the FP8 projections a
   ModelOpt checkpoint stores with scalar scales. */
apxinf_status_t apxinf_quantize_fp8_per_tensor(const void* input, void* output,
                                               int64_t count,
                                               float input_scale,
                                               apxinf_cuda_stream_t stream);

/* Single-token BF16 projection with canonical [K, N] weights. */
apxinf_status_t apxinf_bf16_gemv(const void* activation, const void* weight,
                                 void* output, int64_t n, int64_t k,
                                 apxinf_cuda_stream_t stream);

/* Single-token FP8 projection over a [N, K] weight -- the checkpoint's own
   orientation, so no transpose is needed. */
apxinf_status_t apxinf_fp8_gemv(const void* weight, const void* activation,
                                void* output, int64_t n, int64_t k,
                                float alpha, apxinf_cuda_stream_t stream);

/* Single-token NVFP4 projection. Scales are plain row-major [rows, K/16] --
   the checkpoint's own layout, no atom relayout needed. */
apxinf_status_t apxinf_nvfp4_gemv(const void* weight, const void* weight_scales,
                                  const void* activation,
                                  const void* activation_scales, void* output,
                                  int64_t n, int64_t k, float alpha,
                                  apxinf_cuda_stream_t stream);

/* BF16 x weight-only NVFP4 GEMV using Edge-LLM's BLACKWELL_N128_K64_V1
   physical layout. */
apxinf_status_t apxinf_nvfp4_a16_blackwell_gemv(
    const void* activation, const void* qweights, const void* block_scales,
    const void* global_scale, void* output, int64_t n, int64_t k, int64_t rows,
    apxinf_cuda_stream_t stream);

#ifdef __cplusplus
}
#endif
