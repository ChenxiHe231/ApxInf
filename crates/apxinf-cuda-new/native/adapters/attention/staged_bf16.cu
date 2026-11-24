#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <math_constants.h>
#include <cmath>
#include <cfloat>
#include <cstdint>
#include "../../kernels/primitives/bf16_pairwise.cuh"

#include "../../kernels/custom/attention_staged_bf16.cuh"

cudaError_t apxinf_staged_attention_shared_limit(int bytes) {
  return cudaFuncSetAttribute(attention_kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,bytes);
}
cudaError_t apxinf_staged_attention(const void*q,const void*k,const void*v,const void*offsets,void*y,
    int tokens,int segments,int maximum_count,int heads,int dim,float scale,cudaStream_t stream) {
  attention_kernel<<<dim3(maximum_count,heads,segments),128,maximum_count*sizeof(float),stream>>>(
    (const __nv_bfloat16*)q,(const __nv_bfloat16*)k,(const __nv_bfloat16*)v,(__nv_bfloat16*)y,
    (const uint32_t*)offsets,tokens,segments,maximum_count,heads,dim,scale);
  return cudaGetLastError();
}
