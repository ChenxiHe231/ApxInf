#include "../internal.h"
#include "../../../kernels/custom/gemm_pairwise_bf16.cuh"

namespace apxinf::gemm {

bool supports_pairwise_bf16_bias(const Spec& s) {
  return s.numerics == APXINF_GEMM_NUMERICS_PAIRWISE_BF16_ODD &&
      s.semantic == APXINF_GEMM_SEMANTIC_GEMM_BIAS &&
      s.a_dtype == APXINF_DTYPE_BF16 && s.b_dtype == APXINF_DTYPE_BF16 &&
      s.output_dtype == APXINF_DTYPE_BF16 && s.accumulation_dtype == APXINF_DTYPE_F32 &&
      s.b_layout == APXINF_GEMM_LAYOUT_NK && s.quantization == APXINF_GEMM_QUANT_NONE &&
      s.alpha_is_unit && s.output_scale_is_unit && s.k % 2 == 0 &&
      s.m > 0 && s.n > 0 && s.k > 0 &&
      ((s.m + 7) / 8) * ((s.n + 15) / 16) <= INT32_MAX;
}

size_t pairwise_bf16_resource_requirements(const Spec&) { return 0; }
void prepare_pairwise_bf16(Execution& execution) {
  if (!supports_pairwise_bf16_bias(execution.spec) || execution.configuration != 0)
    throw Failure(APXINF_STATUS_UNSUPPORTED, "unsupported pairwise BF16 GEMM contract");
  execution.resource_bytes = 0;
}
void destroy_pairwise_bf16(Execution&) noexcept {}
cudaError_t launch_pairwise_bf16(Execution& execution) {
  const auto& s = execution.spec;
  const auto& b = execution.bindings;
  const auto stream = static_cast<cudaStream_t>(b.stream);
  const auto* a = static_cast<const __nv_bfloat16*>(b.a);
  const auto* weight = static_cast<const __nv_bfloat16*>(b.b);
  const auto* bias = static_cast<const __nv_bfloat16*>(b.bias);
  auto* output = static_cast<__nv_bfloat16*>(b.output);
  // A larger register tile pays off for projection/MLP-sized matrices while
  // preserving each output's ordered reduction. Small/skinny cases retain
  // their original tiles; no tuning or allocation is performed on enqueue.
  if (s.m >= 128 && s.n >= 1024 && s.k >= 1024) {
    const unsigned blocks = unsigned(((s.m + 31) / 32) * ((s.n + 63) / 64));
    kernels::pairwise_bf16_bias<4, 4><<<blocks, dim3(16, 8), 0, stream>>>(
        a, weight, bias, output, int(s.m), int(s.n), int(s.k));
  } else if (s.m >= 16 && s.n >= 32) {
    const unsigned blocks = unsigned(((s.m + 15) / 16) * ((s.n + 31) / 32));
    kernels::pairwise_bf16_bias<2, 2><<<blocks, dim3(16, 8), 0, stream>>>(
        a, weight, bias, output, int(s.m), int(s.n), int(s.k));
  } else {
    const unsigned blocks = unsigned(((s.m + 7) / 8) * ((s.n + 15) / 16));
    kernels::pairwise_bf16_bias<1, 1><<<blocks, dim3(16, 8), 0, stream>>>(
        a, weight, bias, output, int(s.m), int(s.n), int(s.k));
  }
  return cudaGetLastError();
}

}  // namespace apxinf::gemm
