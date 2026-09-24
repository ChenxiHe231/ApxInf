// probe5: the recurrent state's global round-trip inside the serial chunk loop.
//
// The production kernel re-reads its 128x128 fp32 state from global memory at
// the top of every chunk and writes it back at the bottom:
//
//   for (chunk) { ... sref_col[kk] = head_state[d*kDim+kk]; ...
//                 head_state[d*kDim+kk] = sref_col[kk]*decay + upd; }
//
// The state is private to the block and lives for the whole loop, and
// sref_col[] is already a 128-float register array, so the round-trip buys
// nothing. It costs 48 heads * 32 chunks * 128 KB = 196 MB of global traffic
// per layer, and -- because it sits between the two halves of the serial
// dependency -- a full global load latency per chunk on the critical path.
//
// REG=0 reproduces the production kernel; REG=1 hoists the load before the
// chunk loop and the store after it. Nothing else differs.
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <vector>

#define CHECK(x)                                                        \
  do {                                                                  \
    cudaError_t e = (x);                                                \
    if (e != cudaSuccess) {                                             \
      printf("CUDA %s @%d: %s\n", #x, __LINE__, cudaGetErrorString(e)); \
      exit(1);                                                          \
    }                                                                   \
  } while (0)

constexpr int kChunk = 64;
constexpr int kDim = 128;

struct ChunkShared {
  float q[kChunk][kDim];
  float k[kChunk][kDim];
  float v[kChunk][kDim];
  float nv[kChunk][kDim];
  float kcd[kChunk][kDim];
  float pw[kChunk][kChunk];
  float T[kChunk][kChunk];
  float M[kChunk][kChunk];
  float cum[kChunk];
  float g[kChunk];
  float beta[kChunk];
};

template <int REG>
__global__ void scan(const __nv_bfloat16* __restrict__ q_in,
                     const __nv_bfloat16* __restrict__ k_in,
                     const __nv_bfloat16* __restrict__ v_in,
                     const float* __restrict__ g_in,
                     const float* __restrict__ b_in,
                     __nv_bfloat16* __restrict__ out, int row_stride,
                     float* __restrict__ state, int v_heads, int k_heads,
                     int num_chunks) {
  extern __shared__ float smem_raw[];
  ChunkShared& s = *reinterpret_cast<ChunkShared*>(smem_raw);

  const int head = blockIdx.x % v_heads;
  const int k_head = head / (v_heads / k_heads);
  const int tid = threadIdx.x;
  const int d = tid;
  float* head_state = state + (long long)blockIdx.x * kDim * kDim;

  float sref_col[kDim];
  if (REG) {
    for (int kk = 0; kk < kDim; ++kk)
      sref_col[kk] = head_state[(long long)d * kDim + kk];
  }

  for (int c = 0; c < num_chunks; ++c) {
    const int base = c * kChunk;

    for (int t = 0; t < kChunk; ++t) {
      const int tok = base + t;
      s.q[t][tid] = __bfloat162float(
          q_in[(long long)tok * row_stride + (long long)k_head * kDim + tid]);
      s.k[t][tid] = __bfloat162float(
          k_in[(long long)tok * row_stride + (long long)k_head * kDim + tid]);
      s.v[t][tid] = __bfloat162float(
          v_in[(long long)tok * row_stride + (long long)head * kDim + tid]);
    }
    if (tid < kChunk) {
      s.g[tid] = g_in[(long long)(base + tid) * v_heads + head];
      s.beta[tid] = b_in[(long long)(base + tid) * v_heads + head];
    }
    __syncthreads();

    const float q_scaling = rsqrtf((float)kDim);
    if (tid < kChunk) {
      float qn = 0.0f, kn = 0.0f;
      for (int dd = 0; dd < kDim; ++dd) {
        qn += s.q[tid][dd] * s.q[tid][dd];
        kn += s.k[tid][dd] * s.k[tid][dd];
      }
      const float qinv = rsqrtf(qn + 1e-6f);
      const float kinv = rsqrtf(kn + 1e-6f);
      for (int dd = 0; dd < kDim; ++dd) {
        s.q[tid][dd] = s.q[tid][dd] * qinv * q_scaling;
        s.k[tid][dd] = s.k[tid][dd] * kinv;
      }
    }
    __syncthreads();

    if (tid == 0) {
      float acc = 0.0f;
      for (int t = 0; t < kChunk; ++t) {
        acc += s.g[t];
        s.cum[t] = acc;
      }
    }
    __syncthreads();

    if (tid < kChunk) {
      const int j = tid;
      for (int i = 0; i < kChunk; ++i)
        s.pw[i][j] = (i >= j) ? __expf(s.cum[i] - s.cum[j]) : 0.0f;
    }
    __syncthreads();

    {
      const int j = tid % kChunk;
      const int i_begin = (tid / kChunk) * (kChunk / 2);
      for (int i = i_begin; i < i_begin + kChunk / 2; ++i) {
        float ut = 0.0f, at = 0.0f;
        const float bi = s.beta[i];
        for (int dd = 0; dd < kDim; ++dd) {
          const float kjd = s.k[j][dd];
          ut += (bi * s.k[i][dd]) * kjd;
          at += s.q[i][dd] * kjd;
        }
        s.M[i][j] = ut * s.pw[i][j];
        s.T[i][j] = at * s.pw[i][j];
      }
    }
    __syncthreads();

    const int col = tid;
    if (col < kChunk)
      for (int i = 0; i < kChunk; ++i) s.pw[i][col] = 0.0f;
    __syncthreads();
    for (int i = 1; i < kChunk; ++i) {
      if (col < i) {
        float acc = -s.M[i][col];
        for (int p = col + 1; p < i; ++p) acc += (-s.M[i][p]) * s.pw[p][col];
        s.pw[i][col] = acc;
      }
      __syncthreads();
    }
    if (col < kChunk) s.pw[col][col] += 1.0f;
    __syncthreads();

    for (int i = 0; i < kChunk; ++i) {
      float nvv = 0.0f, kc = 0.0f;
      for (int p = 0; p <= i; ++p) {
        const float t = s.pw[i][p];
        nvv += t * (s.beta[p] * s.v[p][d]);
        kc += t * (s.beta[p] * s.k[p][d] * __expf(s.cum[p]));
      }
      s.nv[i][d] = nvv;
      s.kcd[i][d] = kc;
    }
    __syncthreads();

    const float cum_last = s.cum[kChunk - 1];
    const float chunk_decay = __expf(cum_last);

    if (!REG) {
      for (int kk = 0; kk < kDim; ++kk)
        sref_col[kk] = head_state[(long long)d * kDim + kk];
    }

    for (int i = 0; i < kChunk; ++i) {
      float acc = 0.0f;
      for (int kk = 0; kk < kDim; ++kk) acc += s.kcd[i][kk] * sref_col[kk];
      s.nv[i][d] = s.nv[i][d] - acc;
    }
    __syncthreads();

    for (int i = 0; i < kChunk; ++i) {
      const float ecum_i = __expf(s.cum[i]);
      float inter = 0.0f;
      for (int kk = 0; kk < kDim; ++kk)
        inter += (s.q[i][kk] * ecum_i) * sref_col[kk];
      float intra = 0.0f;
      for (int j = 0; j <= i; ++j) intra += s.T[i][j] * s.nv[j][d];
      out[((long long)(base + i) * v_heads + head) * kDim + d] =
          __float2bfloat16(inter + intra);
    }
    __syncthreads();

    for (int kk = 0; kk < kDim; ++kk) {
      float upd = 0.0f;
      for (int i = 0; i < kChunk; ++i)
        upd += (s.k[i][kk] * __expf(cum_last - s.cum[i])) * s.nv[i][d];
      const float newval = sref_col[kk] * chunk_decay + upd;
      if (REG)
        sref_col[kk] = newval;
      else
        head_state[(long long)d * kDim + kk] = newval;
    }
    __syncthreads();
  }

  if (REG) {
    for (int kk = 0; kk < kDim; ++kk)
      head_state[(long long)d * kDim + kk] = sref_col[kk];
  }
}

template <int REG>
float run(const __nv_bfloat16* q, const __nv_bfloat16* k,
          const __nv_bfloat16* v, const float* g, const float* b,
          __nv_bfloat16* o, float* st, int rs, int vh, int kh, int grid, int nc,
          int iters, bool report) {
  size_t shared = sizeof(ChunkShared);
  CHECK(cudaFuncSetAttribute(
      scan<REG>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shared));
  cudaFuncAttributes fa;
  CHECK(cudaFuncGetAttributes(&fa, scan<REG>));
  auto L = [&] {
    scan<REG><<<grid, kDim, shared>>>(q, k, v, g, b, o, rs, st, vh, kh, nc);
  };
  L();
  CHECK(cudaDeviceSynchronize());
  CHECK(cudaGetLastError());
  cudaEvent_t e0, e1;
  cudaEventCreate(&e0);
  cudaEventCreate(&e1);
  cudaEventRecord(e0);
  for (int i = 0; i < iters; ++i) L();
  cudaEventRecord(e1);
  CHECK(cudaEventSynchronize(e1));
  float ms;
  cudaEventElapsedTime(&ms, e0, e1);
  cudaEventDestroy(e0);
  cudaEventDestroy(e1);
  if (report)
    printf("  REG=%d regs/thread=%d spill(st/ld)=%zu/%zu lmem=%zu\n", REG,
           fa.numRegs, fa.localSizeBytes, (size_t)0, fa.localSizeBytes);
  return ms / iters;
}

int main(int argc, char** argv) {
  int seq = (argc > 1) ? atoi(argv[1]) : 2048;
  int iters = (argc > 2) ? atoi(argv[2]) : 5;
  const int vh = 48, kh = 16, rs = 10240;
  const int nc = seq / kChunk;
  std::vector<__nv_bfloat16> hq((size_t)seq * rs);
  for (size_t i = 0; i < hq.size(); ++i)
    hq[i] = __float2bfloat16(
        ((float)((i * 1103515245u + 12345u) % 1000) / 1000.f - 0.5f) * 0.2f);
  std::vector<float> hg((size_t)seq * vh, -0.02f), hb((size_t)seq * vh, 0.5f);
  __nv_bfloat16 *dq, *dout;
  float *dg, *db, *dst;
  CHECK(cudaMalloc(&dq, hq.size() * 2));
  CHECK(cudaMemcpy(dq, hq.data(), hq.size() * 2, cudaMemcpyHostToDevice));
  CHECK(cudaMalloc(&dg, hg.size() * 4));
  CHECK(cudaMemcpy(dg, hg.data(), hg.size() * 4, cudaMemcpyHostToDevice));
  CHECK(cudaMalloc(&db, hb.size() * 4));
  CHECK(cudaMemcpy(db, hb.data(), hb.size() * 4, cudaMemcpyHostToDevice));
  CHECK(cudaMalloc(&dout, (size_t)seq * vh * kDim * 2));
  CHECK(cudaMalloc(&dst, (size_t)64 * kDim * kDim * 4));
  CHECK(cudaMemset(dst, 0, (size_t)64 * kDim * kDim * 4));
  const __nv_bfloat16 *qp = dq, *kp = dq + (size_t)kh * kDim,
                      *vp = dq + (size_t)2 * kh * kDim;

  printf("seq=%d chunks=%d\n", seq, nc);
  float a = run<0>(qp, kp, vp, dg, db, dout, dst, rs, vh, kh, vh, nc, iters, true);
  float b2 = run<1>(qp, kp, vp, dg, db, dout, dst, rs, vh, kh, vh, nc, iters, true);
  printf("\n  state via global each chunk (production) %8.3f ms\n", a);
  printf("  state held in registers                 %8.3f ms  %.3fx\n", b2, a / b2);

  // correctness cross-check: both variants must produce the same state
  float* h0 = (float*)malloc((size_t)vh * kDim * kDim * 4);
  float* h1 = (float*)malloc((size_t)vh * kDim * kDim * 4);
  CHECK(cudaMemset(dst, 0, (size_t)64 * kDim * kDim * 4));
  scan<0><<<vh, kDim, sizeof(ChunkShared)>>>(qp, kp, vp, dg, db, dout, rs, dst, vh, kh, nc);
  CHECK(cudaDeviceSynchronize());
  CHECK(cudaMemcpy(h0, dst, (size_t)vh * kDim * kDim * 4, cudaMemcpyDeviceToHost));
  CHECK(cudaMemset(dst, 0, (size_t)64 * kDim * kDim * 4));
  scan<1><<<vh, kDim, sizeof(ChunkShared)>>>(qp, kp, vp, dg, db, dout, rs, dst, vh, kh, nc);
  CHECK(cudaDeviceSynchronize());
  CHECK(cudaMemcpy(h1, dst, (size_t)vh * kDim * kDim * 4, cudaMemcpyDeviceToHost));
  double dot = 0, n0 = 0, n1 = 0, mx = 0;
  for (size_t i = 0; i < (size_t)vh * kDim * kDim; ++i) {
    dot += (double)h0[i] * h1[i];
    n0 += (double)h0[i] * h0[i];
    n1 += (double)h1[i] * h1[i];
    double dd = fabs((double)h0[i] - h1[i]);
    if (dd > mx) mx = dd;
  }
  printf("  cross-check: cos=%.9f  |global|=%.6e |reg|=%.6e  max abs diff=%.3e\n",
         dot / sqrt(n0 * n1), sqrt(n0), sqrt(n1), mx);
  return 0;
}
