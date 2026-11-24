#pragma once

namespace {
#include "../../vendor/sleef/sleefinline_cuda.h"
using apxinf::numerics::add_round_odd;
__global__ void rope_kernel(const __nv_bfloat16*qkv,const int*positions,__nv_bfloat16*q,__nv_bfloat16*k,
    __nv_bfloat16*v,const __nv_bfloat16*bias,int heads,int dim,float theta) {
  const int token=blockIdx.x,half=dim/2,quarter=dim/4;
  const int64_t projection=int64_t(heads)*dim;
  extern __shared__ float trig[];
  float*c=trig,*s=trig+half;
  for(int slot=threadIdx.x;slot<half;slot+=blockDim.x) {
    int pair=slot%quarter,axis=slot/quarter;
    float exponent=__fdiv_rn(float(2*pair),float(half));
    float power=Sleef_powf1_u10cuda(theta,exponent);
    float frequency=__fdiv_rn(1.f,power),angle=__fmul_rn(float(positions[int64_t(token)*2+axis]),frequency);
    c[slot]=Sleef_cosf1_u10cuda(angle);s[slot]=Sleef_sinf1_u10cuda(angle);
  }
  __syncthreads();
  for(int64_t item=threadIdx.x;item<2*int64_t(heads)*half;item+=blockDim.x) {
    int which=item/(int64_t(heads)*half);
    int64_t local=item%(int64_t(heads)*half);int head=local/half,pair=local%half;
    int64_t offset=int64_t(token)*3*projection+which*projection+int64_t(head)*dim;
    float a=__bfloat162float(qkv[offset+pair]),b=__bfloat162float(qkv[offset+half+pair]);
    if(bias){a=__bfloat162float(__float2bfloat16(__fadd_rn(a,__bfloat162float(bias[which*projection+head*dim+pair]))));
      b=__bfloat162float(__float2bfloat16(__fadd_rn(b,__bfloat162float(bias[which*projection+head*dim+half+pair]))));}
    __nv_bfloat16*dst=which?k:q;dst+=(int64_t(token)*heads+head)*dim;
    dst[pair]=__float2bfloat16(__fsub_rn(__fmul_rn(a,c[pair]),__fmul_rn(b,s[pair])));
    dst[half+pair]=__float2bfloat16(__fadd_rn(__fmul_rn(a,s[pair]),__fmul_rn(b,c[pair])));
  }
  for(int64_t i=threadIdx.x;i<projection;i+=blockDim.x){
    const auto a=qkv[token*3*projection+2*projection+i];
    v[token*projection+i]=bias?__float2bfloat16(__fadd_rn(__bfloat162float(a),__bfloat162float(bias[2*projection+i]))):a;
  }
}
} // namespace
