#pragma once

namespace {
struct Moments { int n; float mean,m2; };
__device__ void combine_vector(Moments &a,const Moments b) {
  const int n=a.n+b.n;
  const float c=n ? __fdiv_rn(float(b.n),float(n)) : 0.f;
  const float delta=__fsub_rn(b.mean,a.mean);
  const float m2=__fadd_rn(a.m2,b.m2);
  const float cd=__fmul_rn(c,delta), nd=__fmul_rn(float(a.n),delta);
  a.mean=__fadd_rn(a.mean,cd);a.m2=fmaf(nd,cd,m2);a.n=n;
}
__global__ void reference_norm_kernel(const __nv_bfloat16* x,const __nv_bfloat16* gamma,
  const __nv_bfloat16* beta,__nv_bfloat16* out,int rows,int width,float eps) {
  const int row=blockIdx.x,lane=threadIdx.x;
  __shared__ float means[4],m2s[4],scale,bias;
  if(lane<4) {
    const int vectors=width/8,chunks=(vectors+15)/16;
    int depth=0;while((1<<depth)<chunks)++depth;
    Moments stack[16];for(int i=0;i<16;++i)stack[i]={0,0.f,0.f};
    for(int chunk=0;chunk<chunks;++chunk) {
      const int count=min(16,vectors-chunk*16);
      Moments lo={count,0.f,0.f},hi={count,0.f,0.f};
      for(int j=0;j<count;++j) {
        const int64_t base=int64_t(row)*width+chunk*128+j*8+lane;
        const float a=__bfloat162float(x[base]),b=__bfloat162float(x[base+4]);
        const float c=__fdiv_rn(1.f,float(j+1));
        const float da=__fsub_rn(a,lo.mean),db=__fsub_rn(b,hi.mean);
        lo.mean=fmaf(da,c,lo.mean);hi.mean=fmaf(db,c,hi.mean);
        lo.m2=fmaf(da,__fsub_rn(a,lo.mean),lo.m2);
        hi.m2=fmaf(db,__fsub_rn(b,hi.mean),hi.m2);
      }
      combine_vector(stack[0],lo);combine_vector(stack[0],hi);
      int mask=chunk+1;
      for(int level=1;level<depth && (mask&1)==0;++level) {
        combine_vector(stack[level],stack[level-1]);stack[level-1]={0,0.f,0.f};mask>>=1;
      }
    }
    for(int level=1;level<depth;++level)combine_vector(stack[0],stack[level]);
    means[lane]=stack[0].mean;m2s[lane]=stack[0].m2;
  }
  __syncthreads();
  if(lane==0) {
    Moments total={0,0.f,0.f};
    for(int i=0;i<4;++i) {
      const int count=width/4,n=total.n+count;
      const float c=__fdiv_rn(float(count),float(n));
      const float delta=__fsub_rn(means[i],total.mean);
      total.mean=fmaf(c,delta,total.mean);
      float cross=__fmul_rn(__fmul_rn(delta,delta),c);
      const float term=fmaf(cross,float(total.n),m2s[i]);
      total.m2=__fadd_rn(total.m2,term);total.n=n;
    }
    const float variance=__fdiv_rn(total.m2,float(width));
    scale=__fdiv_rn(1.f,sqrtf(__fadd_rn(variance,eps)));
    bias=__fmul_rn(-scale,total.mean);
  }
  __syncthreads();
  for(int column=lane;column<width;column+=blockDim.x) {
    const int64_t index=int64_t(row)*width+column;
    const float value=__bfloat162float(x[index]);
    const float normalized=fmaf(value,scale,bias);
    const float g=__bfloat162float(gamma[column]),b=__bfloat162float(beta[column]);
    const float result=fmaf(normalized,g,b);
    out[index]=__float2bfloat16(result);
  }
}


} // namespace
