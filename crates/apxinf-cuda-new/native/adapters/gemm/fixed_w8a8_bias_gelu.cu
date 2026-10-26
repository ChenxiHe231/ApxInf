// Copyright 2026 ApxInf contributors.
// Fixed W8A8 bias+GELU launcher; ported without numerical changes.
#ifdef APXINF_GEMM_CUTLASS_SM87_W8A8
#include "../../kernels/cutlass/ops/gemm/gemm_i8_bf16_sm80.h"
#endif

#include <cuda_runtime.h>

extern "C" cudaError_t
apxinf_gemm_cutlass_int8_gemm_bias_gelu_bf16_m41_n6144_k1536(
    const void* activation, const void* weight_output_major,
    const void* row_scales, const void* column_scales, const void* bias,
    void* output, int m, int n, int k, cudaStream_t stream) {
#ifdef APXINF_GEMM_CUTLASS_SM87_W8A8
  return apxinf::cuda_new::cutlass_ops::
      w8a8_gemm_bias_gelu_bf16_m41_n6144_k1536(
          activation, weight_output_major, row_scales, column_scales, bias,
          output, m, n, k, stream);
#else
  return cudaErrorNotSupported;
#endif
}
