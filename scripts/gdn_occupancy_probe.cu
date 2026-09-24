// Is the block throughput-bound or latency-bound?
// Same instruction stream, but the shared arrays are ALIASED into overlapping
// pools so the request drops from 209 KB to ~115 KB and TWO blocks co-reside
// per SM. Numerics are meaningless; the instruction count, the smem access
// pattern and the global traffic are byte-for-byte the same. If wall time per
// block-of-work drops, the kernel is latency-bound and warps/SM is the lever.
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#define CHECK(x) do { cudaError_t e=(x); if(e!=cudaSuccess){ printf("CUDA %s @%d: %s\n",#x,__LINE__,cudaGetErrorString(e)); exit(1);} } while(0)
constexpr int kChunk=64, kDim=128;

// ALIAS=0 : production footprint (8 distinct arrays, 209 KB)
// ALIAS=1 : v/nv/kcd share one 32 KB pool, pw/T/M share one 16 KB pool -> 115 KB
template <int ALIAS> struct Pool;
template <> struct Pool<0> {
  float q[kChunk][kDim]; float k[kChunk][kDim]; float v[kChunk][kDim];
  float nv[kChunk][kDim]; float kcd[kChunk][kDim];
  float pw[kChunk][kChunk]; float T[kChunk][kChunk]; float M[kChunk][kChunk];
  float cum[kChunk]; float g[kChunk]; float beta[kChunk];
  __device__ float (&Q())[kChunk][kDim]{return q;} __device__ float (&K())[kChunk][kDim]{return k;}
  __device__ float (&V())[kChunk][kDim]{return v;} __device__ float (&NV())[kChunk][kDim]{return nv;}
  __device__ float (&KCD())[kChunk][kDim]{return kcd;}
  __device__ float (&PW())[kChunk][kChunk]{return pw;} __device__ float (&TT())[kChunk][kChunk]{return T;}
  __device__ float (&MM())[kChunk][kChunk]{return M;}
};
template <> struct Pool<1> {
  float q[kChunk][kDim]; float k[kChunk][kDim];
  float poolA[kChunk][kDim];      // v, nv, kcd
  float poolB[kChunk][kChunk];    // pw, T, M
  float cum[kChunk]; float g[kChunk]; float beta[kChunk];
  __device__ float (&Q())[kChunk][kDim]{return q;} __device__ float (&K())[kChunk][kDim]{return k;}
  __device__ float (&V())[kChunk][kDim]{return poolA;} __device__ float (&NV())[kChunk][kDim]{return poolA;}
  __device__ float (&KCD())[kChunk][kDim]{return poolA;}
  __device__ float (&PW())[kChunk][kChunk]{return poolB;} __device__ float (&TT())[kChunk][kChunk]{return poolB;}
  __device__ float (&MM())[kChunk][kChunk]{return poolB;}
};

template <int ALIAS>
__global__ void scan(const __nv_bfloat16* __restrict__ q_in,const __nv_bfloat16* __restrict__ k_in,
    const __nv_bfloat16* __restrict__ v_in,const float* __restrict__ g_in,const float* __restrict__ b_in,
    __nv_bfloat16* __restrict__ out,int st,float* __restrict__ state,int v_heads,int k_heads,int num_chunks){
  extern __shared__ float smem_raw[];
  Pool<ALIAS>& s=*reinterpret_cast<Pool<ALIAS>*>(smem_raw);
  const int head=blockIdx.x%v_heads; const int k_head=head/(v_heads/k_heads); const int tid=threadIdx.x;
  float* head_state=state+(long long)blockIdx.x*kDim*kDim;
  for(int c=0;c<num_chunks;++c){
    const int base=c*kChunk;
    for(int t=0;t<kChunk;++t){ const int tok=base+t;
      s.Q()[t][tid]=__bfloat162float(q_in[(long long)tok*st+(long long)k_head*kDim+tid]);
      s.K()[t][tid]=__bfloat162float(k_in[(long long)tok*st+(long long)k_head*kDim+tid]);
      s.V()[t][tid]=__bfloat162float(v_in[(long long)tok*st+(long long)head*kDim+tid]); }
    if(tid<kChunk){ s.g[tid]=g_in[(long long)(base+tid)*v_heads+head]; s.beta[tid]=b_in[(long long)(base+tid)*v_heads+head]; }
    __syncthreads();
    const float qs=rsqrtf((float)kDim);
    if(tid<kChunk){ float qn=0,kn=0;
      for(int d=0;d<kDim;++d){ qn+=s.Q()[tid][d]*s.Q()[tid][d]; kn+=s.K()[tid][d]*s.K()[tid][d]; }
      const float qi=rsqrtf(qn+1e-6f),ki=rsqrtf(kn+1e-6f);
      for(int d=0;d<kDim;++d){ s.Q()[tid][d]*=qi*qs; s.K()[tid][d]*=ki; } }
    __syncthreads();
    if(tid==0){ float a=0; for(int t=0;t<kChunk;++t){a+=s.g[t]; s.cum[t]=a;} }
    __syncthreads();
    if(tid<kChunk){ const int j=tid; for(int i=0;i<kChunk;++i) s.PW()[i][j]=(i>=j)?__expf(s.cum[i]-s.cum[j]):0.f; }
    __syncthreads();
    { const int j=tid%kChunk; const int ib=(tid/kChunk)*(kChunk/2);
      for(int i=ib;i<ib+kChunk/2;++i){ float ut=0,at=0; const float bi=s.beta[i];
        for(int d=0;d<kDim;++d){ const float kjd=s.K()[j][d]; ut+=(bi*s.K()[i][d])*kjd; at+=s.Q()[i][d]*kjd; }
        s.MM()[i][j]=ut*s.PW()[i][j]; s.TT()[i][j]=at*s.PW()[i][j]; } }
    __syncthreads();
    const int col=tid;
    if(col<kChunk) for(int i=0;i<kChunk;++i) s.PW()[i][col]=0.f;
    __syncthreads();
    for(int i=1;i<kChunk;++i){ if(col<i){ float acc=-s.MM()[i][col];
      for(int p=col+1;p<i;++p) acc+=(-s.MM()[i][p])*s.PW()[p][col]; s.PW()[i][col]=acc; } __syncthreads(); }
    if(col<kChunk) s.PW()[col][col]+=1.f;
    __syncthreads();
    { const int d=tid; for(int i=0;i<kChunk;++i){ float nvv=0,kc=0;
        for(int p=0;p<=i;++p){ const float t=s.PW()[i][p]; nvv+=t*(s.beta[p]*s.V()[p][d]); kc+=t*(s.beta[p]*s.K()[p][d]*__expf(s.cum[p])); }
        s.NV()[i][d]=nvv; s.KCD()[i][d]=kc; } }
    __syncthreads();
    const float cl=s.cum[kChunk-1]; const float cd=__expf(cl); const int d=tid;
    float sref[kDim];
    for(int kk=0;kk<kDim;++kk) sref[kk]=head_state[(long long)d*kDim+kk];
    for(int i=0;i<kChunk;++i){ float acc=0; for(int kk=0;kk<kDim;++kk) acc+=s.KCD()[i][kk]*sref[kk]; s.NV()[i][d]-=acc; }
    __syncthreads();
    for(int i=0;i<kChunk;++i){ const float e=__expf(s.cum[i]); float inter=0;
      for(int kk=0;kk<kDim;++kk) inter+=(s.Q()[i][kk]*e)*sref[kk];
      float intra=0; for(int j=0;j<=i;++j) intra+=s.TT()[i][j]*s.NV()[j][d];
      out[((long long)(base+i)*v_heads+head)*kDim+d]=__float2bfloat16(inter+intra); }
    __syncthreads();
    for(int kk=0;kk<kDim;++kk){ float upd=0;
      for(int i=0;i<kChunk;++i) upd+=(s.K()[i][kk]*__expf(cl-s.cum[i]))*s.NV()[i][d];
      head_state[(long long)d*kDim+kk]=sref[kk]*cd+upd; }
    __syncthreads();
  }
}

template<int A> void run(const __nv_bfloat16*q,const __nv_bfloat16*k,const __nv_bfloat16*v,const float*g,
    const float*b,__nv_bfloat16*o,float*stt,int st,int vh,int kh,int nc,int iters){
  size_t shared=sizeof(Pool<A>);
  CHECK(cudaFuncSetAttribute(scan<A>,cudaFuncAttributeMaxDynamicSharedMemorySize,(int)shared));
  int mb=0; CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&mb,scan<A>,kDim,shared));
  printf("\nALIAS=%d smem=%zu B (%.1f KB)  -> %d block(s)/SM = %d warps/SM (%.1f%% thread slots)\n",
     A,shared,shared/1024.0,mb,mb*4,100.0*mb*128/1536);
  printf("%6s %10s %12s\n","grid","ms","ms/block");
  int grids[]={20,40,48,60,80,96};
  for(int gi=0;gi<6;++gi){ int G=grids[gi];
    auto L=[&]{ scan<A><<<G,kDim,shared>>>(q,k,v,g,b,o,st,stt,vh,kh,nc); };
    L(); CHECK(cudaDeviceSynchronize()); CHECK(cudaGetLastError());
    cudaEvent_t e0,e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    cudaEventRecord(e0); for(int i=0;i<iters;++i) L(); cudaEventRecord(e1); CHECK(cudaEventSynchronize(e1));
    float ms; cudaEventElapsedTime(&ms,e0,e1); ms/=iters;
    printf("%6d %10.3f %12.4f\n",G,ms,ms/G);
    cudaEventDestroy(e0); cudaEventDestroy(e1); }
}

int main(int argc,char**argv){
  int seq=(argc>1)?atoi(argv[1]):2048; int iters=(argc>2)?atoi(argv[2]):3;
  const int vh=48,kh=16,st=10240; int nc=seq/kChunk;
  std::vector<__nv_bfloat16> hq((size_t)seq*st);
  for(size_t i=0;i<hq.size();++i) hq[i]=__float2bfloat16(((float)((i*1103515245u+12345u)%1000)/1000.f-0.5f)*0.2f);
  std::vector<float> hg((size_t)seq*vh,-0.02f),hb((size_t)seq*vh,0.5f);
  __nv_bfloat16*dq,*dout; float*dg,*db,*dst;
  CHECK(cudaMalloc(&dq,hq.size()*2)); CHECK(cudaMemcpy(dq,hq.data(),hq.size()*2,cudaMemcpyHostToDevice));
  CHECK(cudaMalloc(&dg,hg.size()*4)); CHECK(cudaMemcpy(dg,hg.data(),hg.size()*4,cudaMemcpyHostToDevice));
  CHECK(cudaMalloc(&db,hb.size()*4)); CHECK(cudaMemcpy(db,hb.data(),hb.size()*4,cudaMemcpyHostToDevice));
  CHECK(cudaMalloc(&dout,(size_t)seq*vh*kDim*2));
  CHECK(cudaMalloc(&dst,(size_t)128*kDim*kDim*4)); CHECK(cudaMemset(dst,0,(size_t)128*kDim*kDim*4));
  const __nv_bfloat16 *qp=dq,*kp=dq+(size_t)kh*kDim,*vp=dq+(size_t)2*kh*kDim;
  printf("seq=%d chunks=%d\n",seq,nc);
  run<0>(qp,kp,vp,dg,db,dout,dst,st,vh,kh,nc,iters);
  run<1>(qp,kp,vp,dg,db,dout,dst,st,vh,kh,nc,iters);
  return 0;
}
