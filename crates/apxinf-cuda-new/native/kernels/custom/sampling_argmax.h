// Copyright 2026 ApxInf contributors.
#pragma once

#include <cuda_runtime_api.h>
#include <cstdint>

namespace apxinf::cuda_new::sampling_argmax {

// Index of the largest logit in a single BF16 row.
//
// Reducing on device avoids copying 248320 logits to the host per token,
// which at BF16 is ~486 KiB of PCIe/unified traffic for one integer.
//
// Greedy selection: the degenerate sampler. The stateful samplers live in
// adapters/sampling.cu behind apxinf_cuda_new_sample_token; this one needs no
// state, no workspace and no RNG, so it stays a flat entry point
// (doc/cuda-new-design.md section 5).
int argmax_bf16(const void* logits, int32_t* index, int count,
                cudaStream_t stream);

}  // namespace apxinf::cuda_new::sampling_argmax
