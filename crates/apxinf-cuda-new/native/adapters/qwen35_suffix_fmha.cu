// Copyright 2026 ApxInf contributors.
//
// Qwen suffix-causal prefill using D256 2-CTA CUTLASS FMHA.
// Tensor geometry and attention scale are fixed. No autotune or workspace.
#include "../include/apxinf_cuda/qwen35_attention.h"
#include "../framework/runtime_internal.h"

#include <cstdint>
#include <mutex>

#define APXINF_D256_NS apxinf_qwen_suffix_fmha
#define APXINF_D256_BUILD_LANGUAGE 1
#define APXINF_D256_BUILD_ACTION 0
#include "../kernels/cutlass/ops/attention/fmha_d256_sm110.cuh"

namespace APXINF_D256_NS {
namespace {

// The vendored mask handles bottom-right causal coordinates, including the
// exact prefix offset. Distinct mask type keeps this TU's numerics separate.
struct SuffixMask : cutlass::fmha::collective::CausalMask<false> {};
using SuffixTile = Shape<_256, _64, _256>;
using Mainloop = cutlass::fmha::collective::Sm100FmhaFwdMainloopTmaWarpspecialized<
    Element, float, float, SuffixTile, StrideQ, StrideK, StrideV, SuffixMask,
    ThreadShape, ClusterShape>;
using Epilogue = cutlass::fmha::collective::Sm100FmhaFwdEpilogueTmaWarpspecialized<
    Element, float, typename Mainloop::CtaShapePV, StrideO, StrideLSE,
    cute::false_type, 1>;
using Kernel = cutlass::fmha::kernel::Sm100FmhaFwdKernelTmaWarpspecialized<
    ProblemShape, Mainloop, Epilogue, ReverseCluster2TileScheduler,
    cutlass::fmha::kernel::Sm100FmhaCtxKernelWarpspecialized1QSchedule>;
using SuffixFmha = cutlass::fmha::device::FMHA<Kernel>;
using apxinf::framework::Failure;
using apxinf::framework::check_cuda;

bool aligned(const void* ptr) {
  return ptr != nullptr && reinterpret_cast<uintptr_t>(ptr) % 16 == 0;
}

void run(const apxinf_qwen35_suffix_fmha_spec_t& spec,
         const apxinf_qwen35_suffix_fmha_bindings_t& bindings) {
  constexpr int d = APXINF_QWEN35_HEAD_DIM;
  constexpr int qh = APXINF_QWEN35_Q_HEADS;
  constexpr int kvh = APXINF_QWEN35_KV_HEADS;
  constexpr int g = qh / kvh;

  if (spec.version != APXINF_QWEN35_SUFFIX_FMHA_SPEC_VERSION ||
      spec.query_tokens <= 4 || spec.key_tokens < spec.query_tokens ||
      spec.key_tokens > APXINF_QWEN35_SUFFIX_FMHA_MAX_KEY_TOKENS ||
      !aligned(bindings.query) || !aligned(bindings.key) ||
      !aligned(bindings.value) || !aligned(bindings.output)) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT, "invalid Qwen3.5 suffix FMHA arguments");
  }
  const int queries = spec.query_tokens;
  const int keys = spec.key_tokens;
  int device = -1;
  int sms = 0;
  if (device_properties(device, sms) != 0) {
    throw Failure(APXINF_STATUS_CUDA_ERROR, "could not query suffix FMHA device");
  }
  auto problem = make_tuple(queries, keys, d, make_tuple(make_tuple(g, kvh), 1));
  StrideQ q_stride = make_stride(qh * d, _1{}, make_stride(
      make_stride(d, g * d), qh * d * queries));
  StrideK kv_stride = make_stride(kvh * d, _1{}, make_stride(
      make_stride(_0{}, d), kvh * d * keys));
  int lse_rows = ((queries + 127) / 128) * 128;
  StrideLSE lse_stride = make_stride(_1{}, make_stride(
      make_stride(lse_rows, g * lse_rows), qh * lse_rows));

  SuffixFmha::Arguments args{
      problem,
      {{static_cast<const Element*>(bindings.query), q_stride,
        static_cast<const Element*>(bindings.key), kv_stride,
        static_cast<const Element*>(bindings.value), kv_stride},
       1.0f / 16, 1.0f, 1.0f, 1.0f, 1.0f},
      {static_cast<Element*>(bindings.output), q_stride, nullptr, lse_stride},
      {device, sms}};
  SuffixFmha op;
  if (op.can_implement(args) != cutlass::Status::kSuccess ||
      SuffixFmha::get_workspace_size(args) != 0) {
    throw Failure(APXINF_STATUS_UNSUPPORTED, "suffix FMHA cannot implement this shape");
  }
  const auto stream = static_cast<cudaStream_t>(bindings.stream);
  {
    // The device launcher caches its shared-memory attribute with a static
    // Boolean. Serialize initialization between concurrent request threads.
    static std::mutex init_mutex;
    std::lock_guard<std::mutex> lock(init_mutex);
    if (op.initialize(args, nullptr, stream) != cutlass::Status::kSuccess) {
      throw Failure(APXINF_STATUS_CUDA_ERROR, "suffix FMHA initialization failed");
    }
  }
  if (op.run(stream) != cutlass::Status::kSuccess) {
    throw Failure(APXINF_STATUS_CUDA_ERROR, "suffix FMHA launch failed");
  }
}
}  // namespace
}  // namespace APXINF_D256_NS

extern "C" apxinf_status_t apxinf_qwen35_suffix_fmha(
    apxinf_runtime_t runtime, const apxinf_qwen35_suffix_fmha_spec_t* spec,
    const apxinf_qwen35_suffix_fmha_bindings_t* bindings) {
  return apxinf::framework::abi_boundary([&] {
    if (runtime == nullptr || spec == nullptr || bindings == nullptr) {
      throw apxinf::framework::Failure(APXINF_STATUS_INVALID_ARGUMENT,
                                       "missing suffix FMHA runtime or arguments");
    }
    // CUDA12 Thor is a single integrated GPU. The compiled FMHA wrapper's
    // process-wide cached attributes must never be reused across devices.
    if (runtime->device != 0) {
      throw apxinf::framework::Failure(APXINF_STATUS_UNSUPPORTED,
                                       "suffix FMHA requires the Thor GPU at device 0");
    }
    apxinf::framework::check_cuda(cudaSetDevice(runtime->device));
    APXINF_D256_NS::run(*spec, *bindings);
  });
}
