// Four-lane Welford cascade with explicit F32/FMA boundaries.
// Reduction design follows PyTorch moments_utils.h; see vendor/pytorch/LICENSE.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cmath>
#include <cstdint>

#include "../../kernels/custom/norm_welford4.cuh"

cudaError_t apxinf_layer_norm_welford4(const void*x,const void*g,const void*b,void*y,int rows,int width,float eps,cudaStream_t stream) {
  reference_norm_kernel<<<rows,128,0,stream>>>((const __nv_bfloat16*)x,
    (const __nv_bfloat16*)g,(const __nv_bfloat16*)b,(__nv_bfloat16*)y,rows,width,eps);
  return cudaGetLastError();
}
