// probe4: where does chunk_scan_kernel's time go, and what is the ceiling on a
// chunk-parallel (FlashInfer PR #5109 style) restructure at our 48-head shape?
//
// The kernel body is a faithful copy of the production chunk_scan_kernel
// (gdn_ops.cu @40ec495). Three compile-time flags carve it into the three
// buckets that decide whether the restructure can pay:
//
//   A (bit 0) : everything that does NOT read the incoming state -- l2 norm,
//               cumulative decay, pairwise decay, ut_system/intra_chunk_attn,
//               the forward-substitution inverse, new_values and k_cumdecay.
//               A chunk-parallel split runs all of this in parallel.
//   B (bit 1) : the output term (inter + intra @ v_new). Reads the state, but
//               nothing downstream reads it, so it moves to the parallel
//               "prefill" launch.
//   C (bit 2) : v_new = u - w@S and S = S*decay + k_scaled^T@v_new. This is
//               the serial chain. It is exactly the factored rank-64 form of
//               PR #5109's S_j = M_j S_{j-1} + B_j, so a fixup launch does
//               this same work -- no cheaper, and still only 48 tiles wide.
//   loads     : always on (every variant needs q/k/v resident).
//
// Mask 0 = loads only. 7 = production kernel.
//
// The second experiment (--cp) runs the production mask over grid = 48*split
// with num_chunks/split chunks per block. Reconciliation is simply skipped, so
// the numbers are wrong but the instruction stream and total work are identical
// to the real thing: it is a hard upper bound on the parallel part of any
// chunk-parallel scheme.
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#define CHECK(x)                                                            \
  do {                                                                      \
    cudaError_t e = (x);                                                    \
    if (e != cudaSuccess) {                                                 \
      printf("CUDA %s @%d: %s\n", #x, __LINE__, cudaGetErrorString(e));     \
      exit(1);                                                              \
    }                                                                       \
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

template <int MASK>
__global__ void scan(const __nv_bfloat16* __restrict__ q_in,
                     const __nv_bfloat16* __restrict__ k_in,
                     const __nv_bfloat16* __restrict__ v_in,
                     const float* __restrict__ g_in,
                     const float* __restrict__ b_in,
                     __nv_bfloat16* __restrict__ out, int row_stride,
                     float* __restrict__ state, int v_heads, int k_heads,
                     int num_chunks, int chunk_base) {
  constexpr bool kA = (MASK & 1) != 0;
  constexpr bool kB = (MASK & 2) != 0;
  constexpr bool kC = (MASK & 4) != 0;

  extern __shared__ float smem_raw[];
  ChunkShared& s = *reinterpret_cast<ChunkShared*>(smem_raw);

  const int head = blockIdx.x % v_heads;
  const int group = blockIdx.x / v_heads;
  const int k_head = head / (v_heads / k_heads);
  const int tid = threadIdx.x;
  float* head_state = state + (long long)blockIdx.x * kDim * kDim;

  // With A off the arrays the later stages read are never written. Seed them
  // once with benign finite values so no variant spends its time on inf/nan
  // (fp32 throughput does not depend on the operand, but keep it clean).
  if (!kA) {
    for (int t = 0; t < kChunk; ++t) {
      s.pw[t][tid % kChunk] = 0.5f;
      s.T[t][tid % kChunk] = 0.25f;
      s.M[t][tid % kChunk] = 0.125f;
      s.nv[t][tid] = 0.1f;
      s.kcd[t][tid] = 0.1f;
    }
    if (tid < kChunk) s.cum[tid] = -0.01f * tid;
    __syncthreads();
  }

  const int first = chunk_base + group * num_chunks;
  for (int c = first; c < first + num_chunks; ++c) {
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

    if (kA) {
      const float q_scaling = rsqrtf((float)kDim);
      if (tid < kChunk) {
        float qn = 0.0f, kn = 0.0f;
        for (int d = 0; d < kDim; ++d) {
          qn += s.q[tid][d] * s.q[tid][d];
          kn += s.k[tid][d] * s.k[tid][d];
        }
        const float qinv = rsqrtf(qn + 1e-6f);
        const float kinv = rsqrtf(kn + 1e-6f);
        for (int d = 0; d < kDim; ++d) {
          s.q[tid][d] = s.q[tid][d] * qinv * q_scaling;
          s.k[tid][d] = s.k[tid][d] * kinv;
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
          for (int d = 0; d < kDim; ++d) {
            const float kjd = s.k[j][d];
            ut += (bi * s.k[i][d]) * kjd;
            at += s.q[i][d] * kjd;
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

      {
        const int d = tid;
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
      }
      __syncthreads();
    }

    const float cum_last = s.cum[kChunk - 1];
    const float chunk_decay = __expf(cum_last);
    const int d = tid;

    float sref_col[kDim];
    for (int kk = 0; kk < kDim; ++kk)
      sref_col[kk] = head_state[(long long)d * kDim + kk];

    if (kC) {
      for (int i = 0; i < kChunk; ++i) {
        float acc = 0.0f;
        for (int kk = 0; kk < kDim; ++kk) acc += s.kcd[i][kk] * sref_col[kk];
        s.nv[i][d] = s.nv[i][d] - acc;
      }
      __syncthreads();
    }

    if (kB) {
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
    } else {
      // Keep the output traffic identical so the comparison isolates the math.
      for (int i = 0; i < kChunk; ++i)
        out[((long long)(base + i) * v_heads + head) * kDim + d] =
            __float2bfloat16(s.nv[i][d]);
      __syncthreads();
    }

    if (kC) {
      for (int kk = 0; kk < kDim; ++kk) {
        float upd = 0.0f;
        for (int i = 0; i < kChunk; ++i)
          upd += (s.k[i][kk] * __expf(cum_last - s.cum[i])) * s.nv[i][d];
        head_state[(long long)d * kDim + kk] = sref_col[kk] * chunk_decay + upd;
      }
    } else {
      for (int kk = 0; kk < kDim; ++kk)
        head_state[(long long)d * kDim + kk] = sref_col[kk] + s.nv[kk][d];
    }
    __syncthreads();
  }
}

// ---------------------------------------------------------------------------
// The fixup launch of a chunk-parallel scheme, standalone: 48 tiles, one per
// head, walking `steps` chunks serially. Applies the transfer in the SAME
// factored rank-64 form the production kernel already uses (v_new = u - w@S,
// then S = S*decay + k_scaled^T@v_new) -- materialising a 128x128 M would only
// add work. Per-chunk states are written out, as downstream needs them.
__global__ void fixup(const float* __restrict__ u, const float* __restrict__ w,
                      const float* __restrict__ kdec, float* __restrict__ vnew,
                      float* __restrict__ states, int steps, int heads) {
  extern __shared__ float smem_raw[];
  float(*sw)[kDim] = reinterpret_cast<float(*)[kDim]>(smem_raw);
  float(*sv)[kDim] = sw + kChunk;
  float(*sk)[kDim] = sv + kChunk;

  const int head = blockIdx.x;
  const int d = threadIdx.x;
  float sc[kDim];
  for (int kk = 0; kk < kDim; ++kk) sc[kk] = 0.0f;

  for (int c = 0; c < steps; ++c) {
    const long long off = ((long long)c * heads + head) * kChunk * kDim;
    for (int i = 0; i < kChunk; ++i) {
      sw[i][d] = w[off + (long long)i * kDim + d];
      sv[i][d] = u[off + (long long)i * kDim + d];
      sk[i][d] = kdec[off + (long long)i * kDim + d];
    }
    __syncthreads();
    for (int i = 0; i < kChunk; ++i) {
      float acc = 0.0f;
      for (int kk = 0; kk < kDim; ++kk) acc += sw[i][kk] * sc[kk];
      sv[i][d] -= acc;
    }
    __syncthreads();
    for (int i = 0; i < kChunk; ++i)
      vnew[off + (long long)i * kDim + d] = sv[i][d];
    float* st = states + ((long long)c * heads + head) * kDim * kDim;
    for (int kk = 0; kk < kDim; ++kk) {
      float upd = 0.0f;
      for (int i = 0; i < kChunk; ++i) upd += sk[i][kk] * sv[i][d];
      sc[kk] = sc[kk] * 0.9f + upd;
      st[(long long)d * kDim + kk] = sc[kk];
    }
    __syncthreads();
  }
}

template <int MASK>
float time_mask(const __nv_bfloat16* q, const __nv_bfloat16* k,
                const __nv_bfloat16* v, const float* g, const float* b,
                __nv_bfloat16* o, float* st, int row_stride, int vh, int kh,
                int grid, int nc_per_block, int iters) {
  size_t shared = sizeof(ChunkShared);
  CHECK(cudaFuncSetAttribute(scan<MASK>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             (int)shared));
  auto L = [&] {
    scan<MASK><<<grid, kDim, shared>>>(q, k, v, g, b, o, row_stride, st, vh, kh,
                                       nc_per_block, 0);
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
  return ms / iters;
}

int main(int argc, char** argv) {
  int seq = (argc > 1) ? atoi(argv[1]) : 2048;
  int iters = (argc > 2) ? atoi(argv[2]) : 5;
  const int vh = 48, kh = 16, row_stride = 10240;
  const int nc = seq / kChunk;

  std::vector<__nv_bfloat16> hq((size_t)seq * row_stride);
  for (size_t i = 0; i < hq.size(); ++i)
    hq[i] = __float2bfloat16(((float)((i * 1103515245u + 12345u) % 1000) / 1000.f - 0.5f) * 0.2f);
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
  // one state row per block, so the 48*split variants do not race
  CHECK(cudaMalloc(&dst, (size_t)1536 * kDim * kDim * 4));
  CHECK(cudaMemset(dst, 0, (size_t)1536 * kDim * kDim * 4));
  const __nv_bfloat16 *qp = dq, *kp = dq + (size_t)kh * kDim,
                      *vp = dq + (size_t)2 * kh * kDim;

  printf("seq=%d chunks=%d heads=%d  smem=%.1f KB\n", seq, nc, vh,
         sizeof(ChunkShared) / 1024.0);

  printf("\n== bucket split, production geometry (grid=%d, %d chunks/block) ==\n",
         vh, nc);
  float t0 = time_mask<0>(qp, kp, vp, dg, db, dout, dst, row_stride, vh, kh, vh, nc, iters);
  float t1 = time_mask<1>(qp, kp, vp, dg, db, dout, dst, row_stride, vh, kh, vh, nc, iters);
  float t3 = time_mask<3>(qp, kp, vp, dg, db, dout, dst, row_stride, vh, kh, vh, nc, iters);
  float t5 = time_mask<5>(qp, kp, vp, dg, db, dout, dst, row_stride, vh, kh, vh, nc, iters);
  float t7 = time_mask<7>(qp, kp, vp, dg, db, dout, dst, row_stride, vh, kh, vh, nc, iters);
  printf("  loads only          (mask 0) %8.3f ms\n", t0);
  printf("  loads+A             (mask 1) %8.3f ms\n", t1);
  printf("  loads+A+B           (mask 3) %8.3f ms\n", t3);
  printf("  loads+A+C           (mask 5) %8.3f ms\n", t5);
  printf("  full                (mask 7) %8.3f ms  <- production\n", t7);
  printf("\n  A (state-independent, parallelisable) = %6.3f ms  %5.1f%%\n",
         t1 - t0, 100.0 * (t1 - t0) / t7);
  printf("  B (output term, parallelisable)       = %6.3f ms  %5.1f%%\n",
         t3 - t1, 100.0 * (t3 - t1) / t7);
  printf("  C (SERIAL chain: v_new, state update) = %6.3f ms  %5.1f%%\n",
         t5 - t1, 100.0 * (t5 - t1) / t7);
  printf("  loads + fixed cost                    = %6.3f ms  %5.1f%%\n", t0,
         100.0 * t0 / t7);
  printf("  additivity check: A+B+C+loads = %.3f vs full %.3f (%+.1f%%)\n",
         t0 + (t1 - t0) + (t3 - t1) + (t5 - t1), t7,
         100.0 * ((t0 + (t1 - t0) + (t3 - t1) + (t5 - t1)) / t7 - 1.0));

  printf("\n== chunk-parallel ceiling: same total work, grid=48*split, "
         "reconciliation skipped ==\n");
  printf("%6s %6s %10s %8s\n", "split", "grid", "ms", "vs 1x");
  int splits[] = {1, 2, 4, 8, 16, 32};
  for (int si = 0; si < 6; ++si) {
    int sp = splits[si];
    if (nc % sp || vh * sp > 1536) continue;
    float t = time_mask<7>(qp, kp, vp, dg, db, dout, dst, row_stride, vh, kh,
                           vh * sp, nc / sp, iters);
    printf("%6d %6d %10.3f %7.2fx\n", sp, vh * sp, t, t7 / t);
  }

  printf("\n== the same, for the parallelisable part alone (mask 3 = A+B) ==\n");
  float p1 = time_mask<3>(qp, kp, vp, dg, db, dout, dst, row_stride, vh, kh, vh, nc, iters);
  for (int si = 0; si < 6; ++si) {
    int sp = splits[si];
    if (nc % sp || vh * sp > 1536) continue;
    float t = time_mask<3>(qp, kp, vp, dg, db, dout, dst, row_stride, vh, kh,
                           vh * sp, nc / sp, iters);
    printf("%6d %6d %10.3f %7.2fx\n", sp, vh * sp, t, p1 / t);
  }

  printf("\n== fixup launch, standalone (grid=48, %d serial steps) ==\n", nc);
  {
    float *du, *dw, *dk, *dvn, *dstates;
    size_t n = (size_t)nc * vh * kChunk * kDim;
    CHECK(cudaMalloc(&du, n * 4));
    CHECK(cudaMalloc(&dw, n * 4));
    CHECK(cudaMalloc(&dk, n * 4));
    CHECK(cudaMalloc(&dvn, n * 4));
    CHECK(cudaMalloc(&dstates, (size_t)nc * vh * kDim * kDim * 4));
    CHECK(cudaMemset(du, 0x3c, n * 4));
    CHECK(cudaMemset(dw, 0x3c, n * 4));
    CHECK(cudaMemset(dk, 0x3c, n * 4));
    size_t shared = 3 * kChunk * kDim * sizeof(float);
    CHECK(cudaFuncSetAttribute(fixup, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)shared));
    int mb = 0;
    CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&mb, fixup, kDim, shared));
    auto L = [&] { fixup<<<vh, kDim, shared>>>(du, dw, dk, dvn, dstates, nc, vh); };
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
    ms /= iters;
    printf("  smem=%.1f KB -> %d blocks/SM;  %8.3f ms  (%.1f%% of production "
           "full kernel)\n", shared / 1024.0, mb, ms, 100.0 * ms / t7);
    printf("  workspace traffic: %.0f MB written + %.0f MB read per layer\n",
           4.0 * n * 4 / 1e6, 4.0 * n * 4 / 1e6);
  }
  return 0;
}
