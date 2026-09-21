// Copyright 2026 ApxInf contributors.
#pragma once

#include <cuda_runtime_api.h>
#include <stddef.h>

namespace apxinf::cuda::cutlass_ops {

// Number of tile/cluster configurations exposed to the autotuner.
int nvfp4_gemm_tactic_count();

// Whether a tactic can serve this shape at all. Configurations that overrun
// TMEM on the target device are excluded here rather than discovered by a
// kernel abort, because a CUDA context that has faulted cannot be reused.
bool nvfp4_gemm_tactic_supported(int tactic, int m, int n, int k, int sf_vec);

// Upper bound on device scratch the chosen tactic needs.
size_t nvfp4_gemm_workspace_bytes(int m, int n, int k, int sf_vec, int tactic);

// Y[M,N] (bf16, row-major) = alpha * sum_k A[M,K] * B[N,K]
//
//   a     packed E2M1, physically [M, K/2], K contiguous
//   b     packed E2M1, physically [N, K/2], K contiguous
//   a_sf  unsigned E4M3 block scales for A, already in the kernel atom layout
//   b_sf  unsigned E4M3 block scales for B, already in the kernel atom layout
//
// The per-tensor scales a ModelOpt checkpoint carries (weight_scale_2,
// input_scale) multiply the whole projection and belong in `alpha`.
int nvfp4_gemm_bf16(const void* a, const void* a_sf, const void* b,
                    const void* b_sf, void* out, void* workspace,
                    size_t workspace_bytes, int m, int n, int k, int sf_vec,
                    float alpha, int tactic, cudaStream_t stream);

// Bytes required for one operand's block-scale buffer in the kernel's atom
// layout. `rows` is M for the activation and N for the weight.
size_t nvfp4_scale_buffer_bytes(int rows, int k, int sf_vec);

// Rewrite row-major `[rows, k/sf_vec]` E4M3 block scales -- the layout a
// checkpoint stores -- into the atom layout the kernel reads.
//
// This is a load-time transform: the atom layout depends only on sf_vec, not
// on the tile configuration, so one conversion serves every tactic and the
// autotuner may switch tactics without invalidating it.
int nvfp4_scatter_block_scales(const void* src_row_major, void* dst_atom,
                               int rows, int k, int sf_vec,
                               cudaStream_t stream);

}  // namespace apxinf::cuda::cutlass_ops
