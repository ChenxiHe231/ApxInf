#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <math_constants.h>
#include <cmath>
#include <cfloat>
#include <cstdint>
#include "../../kernels/primitives/bf16_pairwise.cuh"

#include "../../kernels/custom/rope_staged_fp32.cuh"

cudaError_t apxinf_axial_staged_fp32(const void*qkv,const void*bias,const void*positions,void*q,void*k,void*v,
    int tokens,int heads,int dim,float theta,cudaStream_t stream) {
  rope_kernel<<<tokens,128,dim*sizeof(float),stream>>>((const __nv_bfloat16*)qkv,(const int*)positions,
    (__nv_bfloat16*)q,(__nv_bfloat16*)k,(__nv_bfloat16*)v,(const __nv_bfloat16*)bias,heads,dim,theta);
  return cudaGetLastError();
}
