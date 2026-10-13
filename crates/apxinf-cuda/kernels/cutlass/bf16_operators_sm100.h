// Copyright 2026 ApxInf contributors.
#pragma once

#include <cuda_runtime_api.h>

namespace apxinf::cuda::cutlass_ops {

int bf16_gemm_geglu(
    const void* activation, const void* packed_weight, const void* gate,
    void* output, int m, int n, int k, int full_n, int tactic,
    cudaStream_t stream);

namespace bf16_dual_geglu_detail {

int production_dual_geglu_bf16(
    const void* activation, const void* interleaved_weight, void* output,
    int m, int n, int k, int full_n, cudaStream_t stream);

// Second specialization of the same paired kernel: Qwen-Drive text SwiGLU.
// `interleaved_weight` is physical `[k, full_n]` with `[gate256, up256]`
// column tiles -- NOT the checkpoint's `[full_n, k]`. The two have the same
// byte count; see Bf16SwigluWeightLayout on the Rust side.
int production_dual_swiglu_bf16(
    const void* activation, const void* interleaved_weight, void* output,
    int m, int n, int k, int full_n, cudaStream_t stream);

}  // namespace bf16_dual_geglu_detail
}  // namespace apxinf::cuda::cutlass_ops
