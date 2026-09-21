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

// Quantize BF16 to E4M3 against a single per-tensor scale.
//
// This is the activation side of the FP8 projections in a ModelOpt checkpoint,
// whose weight_scale and input_scale are scalars rather than the [M]/[N]
// vectors the scaled-FP8 GEMM contract expects. Quantizing against
// input_scale lets the projection run as unit-scale FP8 with
// `alpha = weight_scale * input_scale`, so no new quantization contract is
// needed for attention or GDN.
int quantize_fp8_per_tensor(const void* input, void* output, long long count,
                            float input_scale, cudaStream_t stream);

// y[n] = alpha * sum_k weight[n, k] * activation[k], E4M3 operands, BF16 out.
//
// The single-token projection. A general GEMM reaches about half of this
// device's bandwidth at M=1 because its tiling is built for large M; here one
// block owns one output row and reads that row contiguously, which is the
// access pattern the hardware wants.
//
// `weight` is [N, K] -- the checkpoint's own orientation, so this path also
// skips the transpose the [K, N] GEMM contract requires.
int fp8_gemv(const void* weight, const void* activation, void* output, int n,
             int k, float alpha, cudaStream_t stream);

}  // namespace apxinf::cuda::mlp_ops
