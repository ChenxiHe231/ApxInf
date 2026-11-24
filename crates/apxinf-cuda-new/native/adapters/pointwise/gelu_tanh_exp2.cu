#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cmath>
#include <cstdint>

#include "../../kernels/custom/gelu_tanh_exp2.cuh"

cudaError_t apxinf_gelu_tanh_exp2(const void*x,void*y,int64_t n,cudaStream_t stream) {
  exp2_gelu_kernel<<<unsigned((n+127)/128),128,0,stream>>>((const __nv_bfloat16*)x,(__nv_bfloat16*)y,n);
  return cudaGetLastError();
}
