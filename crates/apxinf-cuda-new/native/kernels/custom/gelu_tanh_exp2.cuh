#pragma once

// GELU exp2 polynomial follows oneDNN SVE; see native/vendor/onednn.
namespace {
#include "../primitives/exp2_table.cuh"
__device__ float exp2_exp_estimate(float x) {
  x=fminf(x,__uint_as_float(0x42b17218));x=fmaxf(x,__uint_as_float(0xc2aeac50));
  float y=__fmul_rn(x,__uint_as_float(0x3fb8aa3b)),n=floorf(y);
  float u=__fadd_rn(__fsub_rn(y,n),1.f);unsigned bits=__float_as_uint(u),word=bits>>17;
  float base=__uint_as_float(((word>>6)<<23)|(exp2_fexpa_bits[word&63]&0x7fffff));
  base=scalbnf(base,int(n));float r=__fsub_rn(u,__uint_as_float(bits&~((1u<<17)-1)));
  float poly=fmaf(__uint_as_float(0x3e772df2),r,__uint_as_float(0x3f31721c));
  return __fmul_rn(base,fmaf(poly,r,1.f));
}
__global__ void exp2_gelu_kernel(const __nv_bfloat16*x,__nv_bfloat16*y,int64_t n) {
  int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;if(i>=n)return;float a=__bfloat162float(x[i]);
  float fit=fmaf(__fmul_rn(a,a),__uint_as_float(0x3d372713),1.f);
  float inner=__fmul_rn(__fmul_rn(a,__uint_as_float(0x3f4c422a)),fit),t;
  if(fabsf(inner)<__uint_as_float(0x3d4ccccd))t=__fmul_rn(inner,fmaf(__fmul_rn(inner,inner),__uint_as_float(0xbeaaaaab),1.f));
  else t=__fsub_rn(1.f,__fdiv_rn(2.f,__fadd_rn(1.f,exp2_exp_estimate(__fadd_rn(inner,inner)))));
  y[i]=__float2bfloat16(__fmul_rn(fmaf(t,a,a),.5f));
}
} // namespace
