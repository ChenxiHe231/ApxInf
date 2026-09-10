#include "../internal.h"
#include "../../../kernels/custom/gemm.cuh"

#ifdef APXINF_GEMM_CUTLASS
#include "../../../kernels/cutlass/ops/gemm/gemm_bf16_sm100.h"
#include "../../../kernels/cutlass/ops/gemm/gemm_e4m3_sm100.h"
#endif

namespace apxinf::gemm {

void prepare_cutlass(State& state) {
  if (state.implementation->implementation_id == 2) {
    const size_t bytes = state.spec.k * state.spec.n *
                         dtype_bytes(state.spec.b_dtype);
    check_cuda(cudaMalloc(&state.unpack_b, bytes));
    state.resource_bytes += bytes;
  }
}

cudaError_t launch_cutlass(State& state,
                           const apxinf_gemm_bindings_t& bindings) {
#ifdef APXINF_GEMM_CUTLASS
  const auto& spec = state.spec;
  const auto stream = static_cast<cudaStream_t>(bindings.stream);
  const void* weight = bindings.b;
  if (state.unpack_b != nullptr) {
    const int blocks = static_cast<int>(
        std::min<int64_t>((spec.k * spec.n + 255) / 256, 4096));
    apxinf::cuda::custom::pack_gate_up<<<blocks, 256, 0, stream>>>(
        bindings.b, state.unpack_b, spec.b_dtype, spec.k, spec.n);
    check_cuda(cudaGetLastError());
    weight = state.unpack_b;
  }

  using namespace apxinf::cuda::cutlass_ops;
  int status = 0;
  if (state.implementation->implementation_id == 1) {
    status = fp8_gemm_f16(bindings.a, bindings.b, bindings.output, spec.m,
                          spec.n, spec.k, spec.alpha, state.configuration,
                          stream);
  } else if (state.implementation->implementation_id == 3) {
    status = fp8_rowwise_gemm_bf16(
        bindings.a, bindings.b, bindings.a_scales, bindings.b_scales,
        bindings.bias, bindings.output, spec.m, spec.n, spec.k,
        state.configuration, stream);
  } else if (spec.a_dtype == APXINF_DTYPE_E4M3) {
    status = fp8_dual_geglu_detail::production_dual_geglu(
        bindings.a, weight, bindings.output, spec.m, spec.n / 2, spec.k,
        spec.n, spec.alpha, spec.output_scale, stream);
  } else {
    status = bf16_dual_geglu_detail::production_dual_geglu_bf16(
        bindings.a, weight, bindings.output, spec.m, spec.n / 2, spec.k,
        spec.n, stream);
  }
  if (status != 0) {
    throw Failure(APXINF_STATUS_PROVIDER_ERROR,
                  "CUTLASS launch status " + std::to_string(status));
  }
  return cudaSuccess;
#else
  (void)state;
  (void)bindings;
  return cudaErrorNotSupported;
#endif
}

}  // namespace apxinf::gemm
