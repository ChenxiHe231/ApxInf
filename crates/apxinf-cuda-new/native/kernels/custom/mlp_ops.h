// Copyright 2026 ApxInf contributors.
#pragma once

#include <cuda_runtime_api.h>

namespace apxinf::cuda::mlp_ops {

// y[r,c] = x[r,c] / sqrt(mean(x[r,:]^2) + epsilon) * weight[c]
//
// Reduction accumulates in f32: summing thousands of squared BF16 values in
// BF16 drops the small terms outright.
int rms_norm_bf16(const void* input, const void* weight, void* output,
                  int rows, int width, float epsilon, cudaStream_t stream);

// y[r,c] = silu(fused[r,c]) * fused[r,width+c]
//
// `fused_gate_up` is [rows, 2*width] with gate first, which is what one fused
// gate/up GEMM produces. This is SwiGLU, not the GELU-based `gemm_geglu`.
int swiglu_bf16(const void* fused_gate_up, void* output, int rows, int width,
                cudaStream_t stream);

// accumulator += addend, elementwise. Residual connections.
int add_bf16(const void* addend, void* accumulator, long long count,
             cudaStream_t stream);

}  // namespace apxinf::cuda::mlp_ops
