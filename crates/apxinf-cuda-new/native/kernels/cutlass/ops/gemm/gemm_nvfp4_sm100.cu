// Copyright 2026 ApxInf contributors.
//
// NVFP4 (packed E2M1 data, unsigned-E4M3 block scales) GEMM for the SM100
// family. Compiled for sm_110a on Jetson Thor, where CUTLASS enables the same
// tcgen05 block-scaled MMA path via CUTE_ARCH_TCGEN05_MXF4NVF4_MMA_ENABLED.
//
// Tile selection was measured on Thor (20 SMs, 32 MiB L2):
//   * tile N=256 aborts the kernel on this device -- excluded entirely.
//   * cluster 2x1 is worth roughly +30% over 1x1 on large-M shapes.
//   * deep-K shapes lose >2x once the activation operand exceeds L2, which the
//     caller controls by chunking M, not by the tile choice.

#include "gemm_nvfp4_sm100.h"

#include <cuda_bf16.h>

#include <algorithm>

#include "cutlass/cutlass.h"
#include "cutlass/numeric_types.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/util/packed_stride.hpp"

namespace apxinf::cuda::cutlass_ops {
namespace {

using namespace cute;

using ElementA = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using ElementB = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using ElementD = cutlass::bfloat16_t;
using ElementAcc = float;

// Both operands are K-contiguous: A is [M, K] row-major and B is the
// checkpoint's native [N, K], described to CUTLASS as a column-major [K, N].
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutD = cutlass::layout::RowMajor;

constexpr int kAlignA = 32;
constexpr int kAlignB = 32;
constexpr int kAlignD = 8;

template <class MmaTile, class Cluster>
struct Kernel {
  using Epilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassTensorOp, MmaTile, Cluster,
      cutlass::epilogue::collective::EpilogueTileAuto, ElementAcc, float, void,
      LayoutD, kAlignD, ElementD, LayoutD, kAlignD,
      cutlass::epilogue::collective::EpilogueScheduleAuto>::CollectiveOp;

  using Mainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm100, cutlass::arch::OpClassBlockScaledTensorOp, ElementA,
      LayoutA, kAlignA, ElementB, LayoutB, kAlignB, ElementAcc, MmaTile,
      Cluster,
      cutlass::gemm::collective::StageCountAutoCarveout<
          static_cast<int>(sizeof(typename Epilogue::SharedStorage))>,
      cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;

  using Universal = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>, Mainloop, Epilogue, void>;
  using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Universal>;
};

// Ordered best-first for large M, which is where the choice matters; the
// autotuner still times all of them.
using Tactic0 = Kernel<Shape<_256, _128, _256>, Shape<_2, _1, _1>>;
using Tactic1 = Kernel<Shape<_128, _128, _256>, Shape<_2, _1, _1>>;
using Tactic2 = Kernel<Shape<_128, _128, _256>, Shape<_1, _1, _1>>;
using Tactic3 = Kernel<Shape<_128, _128, _128>, Shape<_1, _1, _1>>;

constexpr int kTacticCount = 4;

template <class Cfg>
int launch(const void* a, const void* a_sf, const void* b, const void* b_sf,
           void* out, void* workspace, size_t workspace_bytes, int m, int n,
           int k, float alpha, cudaStream_t stream) {
  using Gemm = typename Cfg::Gemm;
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideD = typename Gemm::GemmKernel::StrideD;
  using SfConfig =
      typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
  using ElementSF = typename ElementA::ScaleFactorType;

  auto problem = cute::make_shape(m, n, k, 1);
  StrideA stride_a = cutlass::make_cute_packed_stride(StrideA{}, {m, k, 1});
  StrideB stride_b = cutlass::make_cute_packed_stride(StrideB{}, {n, k, 1});
  StrideD stride_d = cutlass::make_cute_packed_stride(StrideD{}, {m, n, 1});

  typename Gemm::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGemm,
      problem,
      {reinterpret_cast<typename ElementA::DataType const*>(a), stride_a,
       reinterpret_cast<typename ElementB::DataType const*>(b), stride_b,
       reinterpret_cast<ElementSF const*>(a_sf),
       SfConfig::tile_atom_to_shape_SFA(problem),
       reinterpret_cast<ElementSF const*>(b_sf),
       SfConfig::tile_atom_to_shape_SFB(problem)},
      {{alpha, 0.0f},
       nullptr,
       stride_d,
       reinterpret_cast<ElementD*>(out),
       stride_d}};

  Gemm gemm;
  if (Gemm::get_workspace_size(args) > workspace_bytes) return -3;
  if (gemm.can_implement(args) != cutlass::Status::kSuccess) return -1;
  if (gemm.initialize(args, workspace, stream) != cutlass::Status::kSuccess) {
    return -2;
  }
  return gemm.run(stream) == cutlass::Status::kSuccess ? 0 : -4;
}

template <class Cfg>
size_t workspace_for(int m, int n, int k) {
  using Gemm = typename Cfg::Gemm;
  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideD = typename Gemm::GemmKernel::StrideD;
  using SfConfig =
      typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;
  using ElementSF = typename ElementA::ScaleFactorType;

  auto problem = cute::make_shape(m, n, k, 1);
  ElementSF const* no_scales = nullptr;
  typename Gemm::Arguments args{
      cutlass::gemm::GemmUniversalMode::kGemm,
      problem,
      {nullptr, cutlass::make_cute_packed_stride(StrideA{}, {m, k, 1}), nullptr,
       cutlass::make_cute_packed_stride(StrideB{}, {n, k, 1}),
       no_scales, SfConfig::tile_atom_to_shape_SFA(problem),
       no_scales, SfConfig::tile_atom_to_shape_SFB(problem)},
      {{1.0f, 0.0f}, nullptr,
       cutlass::make_cute_packed_stride(StrideD{}, {m, n, 1}), nullptr,
       cutlass::make_cute_packed_stride(StrideD{}, {m, n, 1})}};
  return Gemm::get_workspace_size(args);
}

// The atom layout depends only on SFVecSize, so tactic 0's view is
// authoritative for every tactic.
using CanonicalSfConfig =
    typename Tactic0::Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

__global__ void scatter_block_scales_kernel(const uint8_t* __restrict__ src,
                                            uint8_t* __restrict__ dst,
                                            int rows, int k, int sf_vec,
                                            int k_blocks,
                                            CanonicalSfConfig::LayoutSF layout) {
  const long long index = blockIdx.x * (long long)blockDim.x + threadIdx.x;
  const long long total = (long long)rows * k_blocks;
  if (index >= total) return;
  const int row = static_cast<int>(index / k_blocks);
  const int block = static_cast<int>(index % k_blocks);
  // The atom layout is addressed in element coordinates along K; every k
  // inside one block aliases to the same scale entry.
  auto tensor = cute::make_tensor(dst, layout);
  tensor(row, block * sf_vec, 0) = src[index];
}

}  // namespace

int nvfp4_gemm_tactic_count() { return kTacticCount; }

bool nvfp4_gemm_tactic_supported(int tactic, int m, int n, int k, int sf_vec) {
  if (tactic < 0 || tactic >= kTacticCount) return false;
  if (sf_vec != 16) return false;
  // The block-scaled MMA consumes 256 elements of K per instruction for the
  // _256 tiles; a shorter K would leave the scale atom partially covered.
  if (k % 128 != 0 || n % 64 != 0) return false;
  if (tactic == 3) return true;  // 128x128x128 has the loosest K requirement
  return k % 256 == 0;
}

size_t nvfp4_gemm_workspace_bytes(int m, int n, int k, int sf_vec, int tactic) {
  (void)sf_vec;
  switch (tactic) {
    case 0: return workspace_for<Tactic0>(m, n, k);
    case 1: return workspace_for<Tactic1>(m, n, k);
    case 2: return workspace_for<Tactic2>(m, n, k);
    case 3: return workspace_for<Tactic3>(m, n, k);
    default: return 0;
  }
}

int nvfp4_gemm_bf16(const void* a, const void* a_sf, const void* b,
                    const void* b_sf, void* out, void* workspace,
                    size_t workspace_bytes, int m, int n, int k, int sf_vec,
                    float alpha, int tactic, cudaStream_t stream) {
  if (sf_vec != 16) return -10;
  switch (tactic) {
    case 0:
      return launch<Tactic0>(a, a_sf, b, b_sf, out, workspace, workspace_bytes,
                             m, n, k, alpha, stream);
    case 1:
      return launch<Tactic1>(a, a_sf, b, b_sf, out, workspace, workspace_bytes,
                             m, n, k, alpha, stream);
    case 2:
      return launch<Tactic2>(a, a_sf, b, b_sf, out, workspace, workspace_bytes,
                             m, n, k, alpha, stream);
    case 3:
      return launch<Tactic3>(a, a_sf, b, b_sf, out, workspace, workspace_bytes,
                             m, n, k, alpha, stream);
    default:
      return -11;
  }
}

size_t nvfp4_scale_buffer_bytes(int rows, int k, int sf_vec) {
  if (sf_vec != 16) return 0;
  // SFB is built from (N, K); SFA from (M, K). Both go through the same atom,
  // so either entry point gives the same size for a given row count.
  auto layout = CanonicalSfConfig::tile_atom_to_shape_SFB(
      cute::make_shape(1, rows, k, 1));
  return cute::cosize(layout);
}

int nvfp4_scatter_block_scales(const void* src_row_major, void* dst_atom,
                               int rows, int k, int sf_vec,
                               cudaStream_t stream) {
  if (sf_vec != 16) return -10;
  const int k_blocks = (k + sf_vec - 1) / sf_vec;
  auto layout = CanonicalSfConfig::tile_atom_to_shape_SFB(
      cute::make_shape(1, rows, k, 1));
  const size_t bytes = cute::cosize(layout);
  cudaError_t status = cudaMemsetAsync(dst_atom, 0, bytes, stream);
  if (status != cudaSuccess) return -20;
  const long long total = (long long)rows * k_blocks;
  const int threads = 256;
  const long long blocks = (total + threads - 1) / threads;
  scatter_block_scales_kernel<<<static_cast<int>(blocks), threads, 0, stream>>>(
      static_cast<const uint8_t*>(src_row_major),
      static_cast<uint8_t*>(dst_atom), rows, k, sf_vec, k_blocks, layout);
  return cudaGetLastError() == cudaSuccess ? 0 : -21;
}

}  // namespace apxinf::cuda::cutlass_ops
