// Copyright 2025 ApxInf contributors.
// SPDX-License-Identifier: Apache-2.0
// Raw-pointer adaptation of an upstream CUTLASS SM100 FP8 rowwise GEMM.

#if defined(__CUDA_ARCH_FEAT_SM101_ALL)
#define CUTLASS_ARCH_MMA_SM100A_ENABLED 1
#endif

#include "fp8_operators_sm100.h"

#include <cuda_runtime.h>
#include <math.h>

#include <cstdint>
#include <type_traits>

#include <cute/tensor.hpp>
#include <cutlass/arch/arch.h>
#include <cutlass/cutlass.h>
#include <cutlass/epilogue/collective/collective_builder.hpp>
#include <cutlass/epilogue/collective/default_epilogue.hpp>
#include <cutlass/epilogue/dispatch_policy.hpp>
#include <cutlass/epilogue/fusion/callbacks.hpp>
#include <cutlass/epilogue/fusion/operations.hpp>
#include <cutlass/epilogue/fusion/sm90_callbacks_tma_warpspecialized.hpp>
#include <cutlass/gemm/collective/collective_builder.hpp>
#include <cutlass/gemm/device/gemm_universal_adapter.h>
#include <cutlass/gemm/kernel/gemm_universal.hpp>
#include <cutlass/layout/matrix.h>
#include <cutlass/numeric_types.h>
#include <cutlass/util/packed_stride.hpp>

using namespace cute;

namespace apxinf_cuda_cutlass_detail {

struct GeGluScaleArguments {
  float alpha = 1.0f;
  float inverse_scale = 1.0f;
};

template <class T>
struct ProductionGeGlu;

template <>
struct ProductionGeGlu<float> {
  using Arguments = GeGluScaleArguments;

  CUTLASS_DEVICE float operator()(
      float const& gate, float const& up,
      Arguments const& arguments) const {
    constexpr float kAlpha = 0.7978845608028654f;
    const float gelu = 0.5f * gate *
        (1.0f + tanhf(kAlpha *
            (gate + 0.044715f * gate * gate * gate)));
    return gelu * (up * arguments.alpha) * arguments.inverse_scale;
  }
};

template <class T, int N>
struct ProductionGeGlu<cutlass::Array<T, N>> {
  using Arguments = GeGluScaleArguments;

  CUTLASS_DEVICE cutlass::Array<T, N> operator()(
      cutlass::Array<T, N> const& gate,
      cutlass::Array<T, N> const& up,
      Arguments const& arguments) const {
    cutlass::Array<T, N> output;
    CUTLASS_PRAGMA_UNROLL
    for (int index = 0; index < N; ++index) {
      constexpr float kAlpha = 0.7978845608028654f;
      const float gate_value = gate[index];
      const float gelu = 0.5f * gate_value *
          (1.0f + tanhf(kAlpha *
              (gate_value + 0.044715f * gate_value * gate_value * gate_value)));
      output[index] = gelu * (up[index] * arguments.alpha) *
          arguments.inverse_scale;
    }
    return output;
  }
};

using ElementSource = cutlass::half_t;
using ElementOutput = cutlass::float_e4m3_t;
using ElementCompute = float;
constexpr auto kRound = cutlass::FloatRoundStyle::round_to_nearest;
using GeGluEVTBase = cutlass::epilogue::fusion::Sm90EVT<
    cutlass::epilogue::fusion::Sm90Compute<
        ProductionGeGlu, ElementOutput, ElementCompute, kRound>,
    cutlass::epilogue::fusion::Sm90SrcFetch<ElementSource>,
    cutlass::epilogue::fusion::Sm90AccFetch>;

struct GeGluEVT : GeGluEVTBase {
  using GeGluEVTBase::GeGluEVTBase;
};

struct GeGluOperation : cutlass::epilogue::fusion::FusionOperation {
  using ElementOutput = apxinf_cuda_cutlass_detail::ElementOutput;
  using ElementCompute = apxinf_cuda_cutlass_detail::ElementCompute;
  using ElementSource = apxinf_cuda_cutlass_detail::ElementSource;
  static constexpr bool IsSourceSupported = true;
};

}  // namespace apxinf_cuda_cutlass_detail

namespace cutlass::epilogue::fusion {
template <>
struct FusionCallbacksTraits<apxinf_cuda_cutlass_detail::GeGluEVT> {
  using DispatchPolicy = void;
  using Callbacks = apxinf_cuda_cutlass_detail::GeGluEVT;
  using Operation = apxinf_cuda_cutlass_detail::GeGluOperation;
  using CtaTile_MNK = void;
  using EpilogueTile_MN = void;
  using ElementCompute = apxinf_cuda_cutlass_detail::ElementCompute;
};
}  // namespace cutlass::epilogue::fusion

namespace apxinf::cuda::cutlass_ops {
template <typename TileShape, typename ClusterShape>
struct Fp8Gemm {
  using ElementInput = cutlass::float_e4m3_t;
  using ElementOutput = cutlass::half_t;
  using ElementAccumulator = float;
  using LayoutA = cutlass::layout::RowMajor;
  // ApxInf stores linear weights physically as contiguous [K, N]. The original
  // wrapper passed a non-contiguous PyTorch transpose whose same
  // logical [K, N] tensor was physically [N, K], so its ColumnMajor tag was
  // correct there but transposed ApxInf weights a second time.  Keep the
  // shared ApxInf/cuBLASLt [K, N] allocation and describe it as RowMajor.
  using LayoutB = cutlass::layout::RowMajor;
  using LayoutD = cutlass::layout::RowMajor;
  static constexpr int AlignmentInput = 16;
  static constexpr int AlignmentOutput = 8;

  using FusionOperation = cutlass::epilogue::fusion::ScaledAcc<
      ElementOutput, float, float>;
  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm100,
      cutlass::arch::OpClassTensorOp,
      TileShape,
      ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator,
      float,
      void,
      LayoutD,
      AlignmentOutput,
      ElementOutput,
      LayoutD,
      AlignmentOutput,
      cutlass::epilogue::collective::EpilogueScheduleAuto,
      FusionOperation>::CollectiveOp;
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm100,
      cutlass::arch::OpClassTensorOp,
      ElementInput,
      LayoutA,
      AlignmentInput,
      ElementInput,
      LayoutB,
      AlignmentInput,
      ElementAccumulator,
      TileShape,
      ClusterShape,
      cutlass::gemm::collective::StageCountAutoCarveout<
          static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::gemm::collective::KernelScheduleAuto>::CollectiveOp;
  using Kernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue, void>;
  using Device = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
};

template <
    typename TileShape, typename ClusterShape, int Stages = 0,
    typename MainloopSchedule = cutlass::gemm::collective::KernelScheduleAuto>
struct Fp8GemmGeGlu {
  using ElementInput = cutlass::float_e4m3_t;
  using ElementSource = cutlass::half_t;
  using ElementOutput = cutlass::float_e4m3_t;
  using ElementAccumulator = float;
  using ElementCompute = float;
  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::RowMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using LayoutD = cutlass::layout::RowMajor;
  static constexpr int AlignmentInput = 16;
  static constexpr int AlignmentC = 8;
  static constexpr int AlignmentD = 16;

  using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
      cutlass::arch::Sm100,
      cutlass::arch::OpClassTensorOp,
      TileShape,
      ClusterShape,
      cutlass::epilogue::collective::EpilogueTileAuto,
      ElementAccumulator,
      ElementCompute,
      ElementSource,
      LayoutC,
      AlignmentC,
      ElementOutput,
      LayoutD,
      AlignmentD,
      cutlass::epilogue::collective::EpilogueScheduleAuto,
      apxinf_cuda_cutlass_detail::GeGluEVT>::CollectiveOp;
  using MainloopStages = std::conditional_t<
      Stages == 0,
      cutlass::gemm::collective::StageCountAutoCarveout<
          static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
      cutlass::gemm::collective::StageCount<Stages>>;
  using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
      cutlass::arch::Sm100,
      cutlass::arch::OpClassTensorOp,
      ElementInput,
      LayoutA,
      AlignmentInput,
      ElementInput,
      LayoutB,
      AlignmentInput,
      ElementAccumulator,
      TileShape,
      ClusterShape,
      MainloopStages,
      MainloopSchedule>::CollectiveOp;
  using Kernel = cutlass::gemm::kernel::GemmUniversal<
      Shape<int, int, int, int>, CollectiveMainloop, CollectiveEpilogue, void>;
  using Device = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;
};

template <typename Gemm>
int launch(
    const void* activation, const void* weight, void* output,
    int m, int n, int k, float alpha, cudaStream_t stream) {
  using Device = typename Gemm::Device;
  using Kernel = typename Gemm::Kernel;
  using ElementInput = typename Gemm::ElementInput;
  using ElementOutput = typename Gemm::ElementOutput;
  using StrideA = typename Kernel::StrideA;
  using StrideB = typename Kernel::StrideB;
  using StrideD = typename Kernel::StrideD;

  StrideA stride_a = cutlass::make_cute_packed_stride(
      StrideA{}, cute::make_shape(m, k, 1));
  StrideB stride_b = cutlass::make_cute_packed_stride(
      StrideB{}, cute::make_shape(n, k, 1));
  StrideD stride_d = cutlass::make_cute_packed_stride(
      StrideD{}, cute::make_shape(m, n, 1));
  typename Kernel::MainloopArguments mainloop{
      static_cast<ElementInput const*>(activation), stride_a,
      static_cast<ElementInput const*>(weight), stride_b};
  typename Kernel::EpilogueArguments epilogue{
      {alpha, 0.0f, nullptr, nullptr}, nullptr, stride_d,
      static_cast<ElementOutput*>(output), stride_d};
  cutlass::KernelHardwareInfo hardware;
  cudaDeviceGetAttribute(&hardware.sm_count, cudaDevAttrMultiProcessorCount, 0);
  typename Kernel::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {m, n, k, 1}, mainloop, epilogue, hardware, {}};

  Device operation;
  if (operation.can_implement(arguments) != cutlass::Status::kSuccess) return -1;
  size_t required = operation.get_workspace_size(arguments);
  // These serial-K static-inference tactics must remain workspace-free so
  // execution is safe inside CUDA Graph capture. Reject any future CUTLASS
  // schedule that silently changes that contract.
  if (required != 0) return -4;
  if (operation.initialize(arguments, nullptr, stream) != cutlass::Status::kSuccess)
    return -2;
  return operation.run(stream) == cutlass::Status::kSuccess ? 0 : -3;
}

template <typename Gemm>
int launch_geglu(
    const void* activation, const void* up_weight, const void* gate,
    void* output, int m, int n, int k, int full_n, float alpha,
    float output_scale, cudaStream_t stream) {
  using Device = typename Gemm::Device;
  using Kernel = typename Gemm::Kernel;
  using ElementInput = typename Gemm::ElementInput;
  using ElementSource = typename Gemm::ElementSource;
  using ElementOutput = typename Gemm::ElementOutput;
  using StrideA = typename Kernel::StrideA;
  using StrideB = typename Kernel::StrideB;
  using StrideC = typename Kernel::StrideC;
  using StrideD = typename Kernel::StrideD;

  StrideA stride_a = cutlass::make_cute_packed_stride(
      StrideA{}, cute::make_shape(m, k, 1));
  StrideB stride_b = cutlass::make_cute_packed_stride(
      StrideB{}, cute::make_shape(full_n, k, 1));
  StrideC stride_c = cutlass::make_cute_packed_stride(
      StrideC{}, cute::make_shape(m, full_n, 1));
  StrideD stride_d = cutlass::make_cute_packed_stride(
      StrideD{}, cute::make_shape(m, n, 1));
  typename Kernel::MainloopArguments mainloop{
      static_cast<ElementInput const*>(activation), stride_a,
      static_cast<ElementInput const*>(up_weight), stride_b};
  typename Kernel::EpilogueArguments epilogue{
      {{}, {}, {alpha, 1.0f / output_scale}},
      static_cast<ElementSource const*>(gate), stride_c,
      static_cast<ElementOutput*>(output), stride_d};
  cutlass::KernelHardwareInfo hardware;
  cudaDeviceGetAttribute(&hardware.sm_count, cudaDevAttrMultiProcessorCount, 0);
  typename Kernel::Arguments arguments{
      cutlass::gemm::GemmUniversalMode::kGemm,
      {m, n, k, 1}, mainloop, epilogue, hardware, {}};

  Device operation;
  if (operation.can_implement(arguments) != cutlass::Status::kSuccess) return -1;
  if (operation.get_workspace_size(arguments) != 0) return -4;
  if (operation.initialize(arguments, nullptr, stream) != cutlass::Status::kSuccess)
    return -2;
  return operation.run(stream) == cutlass::Status::kSuccess ? 0 : -3;
}
int fp8_gemm_f16(
    const void* activation, const void* weight, void* output,
    int m, int n, int k, float alpha, int tactic, cudaStream_t stream) {
  switch (tactic) {
    case 0:
      return launch<Fp8Gemm<Shape<_64, _64, _128>, Shape<_1, _4, _1>>>(
          activation, weight, output, m, n, k, alpha, stream);
    case 1:
      return launch<Fp8Gemm<Shape<_64, _64, _128>, Shape<_1, _1, _1>>>(
          activation, weight, output, m, n, k, alpha, stream);
    case 2:
      return launch<Fp8Gemm<Shape<_128, _128, _128>, Shape<_2, _1, _1>>>(
          activation, weight, output, m, n, k, alpha, stream);
    case 3:
      return launch<Fp8Gemm<Shape<_256, _128, _64>, Shape<_2, _2, _1>>>(
          activation, weight, output, m, n, k, alpha, stream);
    case 4:
      return launch<Fp8Gemm<Shape<_256, _256, _128>, Shape<_2, _2, _1>>>(
          activation, weight, output, m, n, k, alpha, stream);
    case 5:
      return launch<Fp8Gemm<Shape<_256, _128, _128>, Shape<_2, _2, _1>>>(
          activation, weight, output, m, n, k, alpha, stream);
    case 6:
      // Safe tall-N auto-scheduled candidate. Unlike the former
      // tactic 6, this is a regular one-SM schedule and is graph-replay safe.
      return launch<Fp8Gemm<Shape<_128, _256, _128>, Shape<_1, _2, _1>>>(
          activation, weight, output, m, n, k, alpha, stream);
    case 7:
      // Alternate wide auto-scheduled candidate. Keep the 2x1
      // cluster while avoiding the explicit two-SM epilogue that wedges the
      // current Thor-U driver during graph replay.
      return launch<Fp8Gemm<Shape<_256, _128, _128>, Shape<_2, _1, _1>>>(
          activation, weight, output, m, n, k, alpha, stream);
    default:
      return -5;
  }
}

int fp8_gemm_geglu_e4m3(
    const void* activation, const void* packed_weight, const void* gate,
    void* output, int m, int n, int k, int full_n, float alpha,
    float output_scale, int tactic, cudaStream_t stream) {
  if (activation == nullptr || packed_weight == nullptr || gate == nullptr ||
      output == nullptr || m <= 0 || n <= 0 || k <= 0 || full_n != 2 * n ||
      !(alpha > 0.0f) || !(output_scale > 0.0f)) {
    return -6;
  }
  const auto* up_weight = static_cast<const uint8_t*>(packed_weight) + n;
  // Keep this namespace separate from the plain-GEMM tactics and fail closed.
  // 0 preserves the original schedule exactly; the other values select two 2-SM
  // candidates whose 2x2 cluster removes the 1-SM kernel's register spills.
  switch (tactic) {
    case 0:
      return launch_geglu<Fp8GemmGeGlu<
          Shape<_128, _256, _128>, Shape<_1, _2, _1>>>(
          activation, up_weight, gate, output, m, n, k, full_n, alpha,
          output_scale, stream);
    case 1:
      return launch_geglu<Fp8GemmGeGlu<
          Shape<_128, _256, _128>, Shape<_2, _2, _1>>>(
          activation, up_weight, gate, output, m, n, k, full_n, alpha,
          output_scale, stream);
    case 2:
      return launch_geglu<Fp8GemmGeGlu<
          Shape<_128, _256, _128>, Shape<_2, _2, _1>, 3>>(
          activation, up_weight, gate, output, m, n, k, full_n, alpha,
          output_scale, stream);
    case 3:
      // Validated exact-shape M522 configuration. The explicit two-SM mainloop
      // is 5.85% faster than KernelScheduleAuto under NCU while preserving
      // the same 192-register, spill-free resource contract.
      if (m != 522 || n != 16384 || k != 2048 || full_n != 32768) return -7;
      return launch_geglu<Fp8GemmGeGlu<
          Shape<_128, _256, _128>, Shape<_2, _2, _1>, 0,
          cutlass::gemm::KernelTmaWarpSpecialized2SmSm100>>(
          activation, up_weight, gate, output, m, n, k, full_n, alpha,
          output_scale, stream);
    default:
      return -5;
  }
}

}  // namespace apxinf::cuda::cutlass_ops
