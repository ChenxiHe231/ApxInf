#include "../../include/apxinf_cuda/pointwise.h"

#include "../../framework/runtime_internal.h"
#include "../../kernels/custom/quant_ops.h"

#include <cstdint>
#include <string>

namespace {

using apxinf::framework::Failure;
using apxinf::framework::abi_boundary;

void check(int status, const char* what) {
  if (status != 0) {
    throw Failure(APXINF_STATUS_PROVIDER_ERROR,
                  std::string(what) + " failed with status " +
                      std::to_string(status));
  }
}

}  // namespace

extern "C" apxinf_status_t apxinf_pointwise_swiglu_rows_bf16_e4m3(
    const void* gate_up, const void* bias, void* output, void* scales,
    int32_t rows, int32_t input_cols, int32_t inner, int32_t output_cols,
    apxinf_cuda_stream_t stream) {
  return abi_boundary([&] {
    if (gate_up == nullptr || output == nullptr || scales == nullptr ||
        rows <= 0 || input_cols <= 0 || inner <= 0 || output_cols < inner) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                    "invalid fused SwiGLU quantization arguments");
    }
    check(apxinf::cuda_new::quant_ops::swiglu_quantize_rows_bf16_e4m3(
              gate_up, bias, output, scales, rows, input_cols, inner,
              output_cols, static_cast<cudaStream_t>(stream)),
          "fused SwiGLU quantization");
  });
}
