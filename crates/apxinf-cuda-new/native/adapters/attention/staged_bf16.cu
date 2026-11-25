#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <math_constants.h>
#include <cmath>
#include <cfloat>
#include <cstdint>
#include "../../kernels/primitives/bf16_pairwise.cuh"

#include "../../kernels/custom/attention_staged_bf16.cuh"

cudaError_t apxinf_staged_attention_shared_limit(int bytes) {
  // Large segments retain the original zero-static-shared-memory kernel.
  return cudaFuncSetAttribute(attention_kernel<false>,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes);
}
cudaError_t apxinf_staged_attention(const void*q,const void*k,const void*v,const void*offsets,void*y,
    int tokens,int segments,int maximum_count,int heads,int dim,float scale,cudaStream_t stream) {
  const auto grid=dim3(maximum_count,heads,segments);
  const size_t shared_bytes=size_t(maximum_count)*sizeof(float);
  // <=16KiB dynamic plus8bytes static fits the default shared limit. Keeping
  // the original instantiation above this range preserves opt-in boundaries.
  if(maximum_count>=32&&maximum_count<=4096) {
    attention_kernel<true><<<grid,128,shared_bytes,stream>>>(
      (const __nv_bfloat16*)q,(const __nv_bfloat16*)k,(const __nv_bfloat16*)v,(__nv_bfloat16*)y,
      (const uint32_t*)offsets,tokens,segments,maximum_count,heads,dim,scale);
  } else {
    attention_kernel<false><<<grid,128,shared_bytes,stream>>>(
      (const __nv_bfloat16*)q,(const __nv_bfloat16*)k,(const __nv_bfloat16*)v,(__nv_bfloat16*)y,
      (const uint32_t*)offsets,tokens,segments,maximum_count,heads,dim,scale);
  }
  return cudaGetLastError();
}
