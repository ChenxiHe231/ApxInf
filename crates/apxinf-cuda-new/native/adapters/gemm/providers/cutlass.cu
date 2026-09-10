#include "../internal.h"
#include "../../../kernels/custom/gemm.cuh"

#ifdef APXINF_GEMM_CUTLASS
#include "../../../kernels/cutlass/ops/gemm/gemm_bf16_sm100.h"
#include "../../../kernels/cutlass/ops/gemm/gemm_e4m3_sm100.h"
#endif

namespace apxinf::gemm {
namespace {

struct CutlassGegluState {
  void* packed_weight = nullptr;
  size_t packed_weight_bytes = 0;

  ~CutlassGegluState() { release_resources(); }

  void release_resources() noexcept {
    if (packed_weight != nullptr) cudaFree(packed_weight);
    packed_weight = nullptr;
  }
};

CutlassGegluState& provider(State& state) {
  return *static_cast<CutlassGegluState*>(state.provider_state);
}

const void* pack_geglu_weight(State& state,
                              const apxinf_gemm_bindings_t& bindings) {
  auto& resources = provider(state);
  const auto stream = static_cast<cudaStream_t>(bindings.stream);
  const auto& spec = state.spec;
  const int blocks = static_cast<int>(
      std::min<int64_t>((spec.k * spec.n + 255) / 256, 4096));
  apxinf::cuda::custom::pack_gate_up<<<blocks, 256, 0, stream>>>(
      bindings.b, resources.packed_weight, spec.b_dtype, spec.k, spec.n);
  check_cuda(cudaGetLastError());
  return resources.packed_weight;
}

void check_cutlass_status(int status) {
  if (status != 0) {
    throw Failure(APXINF_STATUS_PROVIDER_ERROR,
                  "CUTLASS launch status " + std::to_string(status));
  }
}

}  // namespace

void prepare_cutlass_fp8_gemm(State&, const State*) {}

void prepare_cutlass_geglu(State& state, const State*) {
  auto resources = std::make_unique<CutlassGegluState>();
  resources->packed_weight_bytes =
      state.spec.k * state.spec.n * dtype_bytes(state.spec.b_dtype);
  check_cuda(cudaMalloc(&resources->packed_weight,
                        resources->packed_weight_bytes));
  state.resource_bytes = resources->packed_weight_bytes;
  state.provider_state = resources.release();
}

void release_cutlass_resources(State& state) noexcept {
  if (state.provider_state != nullptr) provider(state).release_resources();
}

void destroy_cutlass(State& state) noexcept {
  delete static_cast<CutlassGegluState*>(state.provider_state);
  state.provider_state = nullptr;
}

cudaError_t launch_cutlass_fp8_gemm(
    State& state, const apxinf_gemm_bindings_t& bindings) {
#ifdef APXINF_GEMM_CUTLASS
  const auto& spec = state.spec;
  const auto stream = static_cast<cudaStream_t>(bindings.stream);
  using namespace apxinf::cuda::cutlass_ops;
  check_cutlass_status(fp8_gemm_f16(
      bindings.a, bindings.b, bindings.output, spec.m, spec.n, spec.k,
      spec.alpha, state.configuration, stream));
  return cudaSuccess;
#else
  (void)state;
  (void)bindings;
  return cudaErrorNotSupported;
#endif
}

cudaError_t launch_cutlass_fp8_geglu(
    State& state, const apxinf_gemm_bindings_t& bindings) {
#ifdef APXINF_GEMM_CUTLASS
  const auto& spec = state.spec;
  const auto stream = static_cast<cudaStream_t>(bindings.stream);
  const void* weight = pack_geglu_weight(state, bindings);
  check_cutlass_status(
      apxinf::cuda::cutlass_ops::fp8_dual_geglu_detail::production_dual_geglu(
          bindings.a, weight, bindings.output, spec.m, spec.n / 2, spec.k,
          spec.n, spec.alpha, spec.output_scale, stream));
  return cudaSuccess;
#else
  (void)state;
  (void)bindings;
  return cudaErrorNotSupported;
#endif
}

cudaError_t launch_cutlass_bf16_geglu(
    State& state, const apxinf_gemm_bindings_t& bindings) {
#ifdef APXINF_GEMM_CUTLASS
  const auto& spec = state.spec;
  const auto stream = static_cast<cudaStream_t>(bindings.stream);
  const void* weight = pack_geglu_weight(state, bindings);
  check_cutlass_status(apxinf::cuda::cutlass_ops::bf16_dual_geglu_detail::
                           production_dual_geglu_bf16(
                               bindings.a, weight, bindings.output, spec.m,
                               spec.n / 2, spec.k, spec.n, stream));
  return cudaSuccess;
#else
  (void)state;
  (void)bindings;
  return cudaErrorNotSupported;
#endif
}

}  // namespace apxinf::gemm
