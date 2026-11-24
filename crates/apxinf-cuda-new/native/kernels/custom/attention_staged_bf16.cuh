#pragma once

namespace {
#include "../../vendor/sleef/sleefinline_cuda.h"
using apxinf::numerics::add_round_odd;
__global__ void attention_kernel(const __nv_bfloat16*q,const __nv_bfloat16*k,const __nv_bfloat16*v,
    __nv_bfloat16*y,const uint32_t* offsets,int tokens,int segments,int maximum_count,int heads,int dim,float scale) {
  extern __shared__ float scores[];
  const int segment=blockIdx.z;
  if(segment>=segments)return;
  const uint32_t begin=offsets[segment],end=offsets[segment+1];
  if(end<=begin||end>uint32_t(tokens)||end-begin>uint32_t(maximum_count)||blockIdx.x>=end-begin)return;
  const int count=int(end-begin),row=begin+blockIdx.x,head=blockIdx.y;
  for(int col=threadIdx.x;col<count;col+=blockDim.x) {
    float sum=0.f;
    for(int i=0;i<dim;i+=2) {
      int64_t a=(int64_t(row)*heads+head)*dim+i,b=((int64_t(begin)+col)*heads+head)*dim+i;
      float p0=__fmul_rn(__bfloat162float(q[a]),__bfloat162float(k[b]));
      float p1=(i+1<dim)?__fmul_rn(__bfloat162float(q[a+1]),__bfloat162float(k[b+1])):0.f;
      sum=add_round_odd(sum,add_round_odd(p0,p1));
    }
    scores[col]=__bfloat162float(__float2bfloat16(__fmul_rn(__bfloat162float(__float2bfloat16(sum)),scale)));
  }
  __syncthreads();
  if(threadIdx.x==0) {
    float maximum=-CUDART_INF_F,total=0.f;
    for(int i=0;i<count;++i)maximum=fmaxf(maximum,scores[i]);
    for(int i=0;i<count;++i){
      float z=__fsub_rn(scores[i],maximum);
      scores[i]=Sleef_expf1_u10cuda(z);
    }
    {
      float s[4]={0.f,0.f,0.f,0.f};
      for(int i=0;i<count;++i)s[i%4]=__fadd_rn(s[i%4],scores[i]);
      total=__fadd_rn(__fadd_rn(s[0],s[2]),__fadd_rn(s[1],s[3]));
      total=__fdiv_rn(1.f,total);
    }
    for(int i=0;i<count;++i)scores[i]=__bfloat162float(__float2bfloat16(__fmul_rn(scores[i],total)));
  }
  __syncthreads();
  for(int d=threadIdx.x;d<dim;d+=blockDim.x) {
    float sum=0.f;
    for(int i=0;i<count;i+=2) {
      float p0=__fmul_rn(scores[i],__bfloat162float(v[((int64_t(begin)+i)*heads+head)*dim+d]));
      float p1=(i+1<count)?__fmul_rn(scores[i+1],__bfloat162float(v[((int64_t(begin)+i+1)*heads+head)*dim+d])):0.f;
      sum=add_round_odd(sum,add_round_odd(p0,p1));
    }
    y[(int64_t(row)*heads+head)*dim+d]=__float2bfloat16(sum);
  }
}
} // namespace
