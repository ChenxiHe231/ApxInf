// Copyright 2026 ApxInf contributors.
// Fixed implementation for this operator family; ported without numerical changes.
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>

namespace {
#include "../custom/gdn_policy.h"
#include "../custom/math.cuh"
#include "../custom/reduction.cuh"
#include "../custom/pillow_bicubic_u8.cuh"
#include "../custom/linear_attention.cuh"
#include "../custom/gdn_raw_inverse_f1.cuh"
#include "../custom/gdn_chunk_state_wmma.cuh"
#include "../custom/gdn_chunk_gemm_tri.cuh"
__global__ void sinusoidal_embedding_bf16_kernel(const float* positions,__nv_bfloat16* out,
    int dim,float scale,float frequency_step) {
  const int row=blockIdx.x,half=dim/2;
  for(int i=threadIdx.x;i<half;i+=blockDim.x) {
    const float frequency=expf(__fmul_rn(-frequency_step,static_cast<float>(i)));
    const float angle=__fmul_rn(__fmul_rn(scale,positions[row]),frequency);
    const int64_t base=static_cast<int64_t>(row)*dim;
    out[base+i]=__float2bfloat16(sinf(angle));
    out[base+half+i]=__float2bfloat16(cosf(angle));
  }
}


}  // namespace
