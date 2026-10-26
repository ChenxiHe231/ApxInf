#include "../../include/apxinf_cuda/pointwise.h"

#include "../../framework/runtime_internal.h"
#include "../../kernels/custom/pointwise_ops.h"

#include <cstdint>
#include <string>

namespace {
using apxinf::framework::Failure;
using apxinf::framework::abi_boundary;
bool extent(int64_t value) { return value > 0 && value <= INT32_MAX; }
void check(int status, const char* what) {
  if (status != 0)
    throw Failure(APXINF_STATUS_PROVIDER_ERROR,
                  std::string(what) + " failed with status " +
                      std::to_string(status));
}
}  // namespace

extern "C" apxinf_status_t apxinf_pointwise_swiglu(
    const void* fused_gate_up, void* output, int64_t rows, int64_t width,
    apxinf_cuda_stream_t stream) {
  return abi_boundary([&] {
    if (fused_gate_up == nullptr || output == nullptr || !extent(rows) ||
        !extent(width))
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT, "invalid SwiGLU arguments");
    check(apxinf::cuda_new::pointwise::swiglu(
              fused_gate_up, output, static_cast<int>(rows),
              static_cast<int>(width), static_cast<cudaStream_t>(stream)),
          "SwiGLU");
  });
}

extern "C" apxinf_status_t apxinf_pointwise_add_into(
    const void* addend, void* accumulator, int64_t count,
    apxinf_cuda_stream_t stream) {
  return abi_boundary([&] {
    if (addend == nullptr || accumulator == nullptr || count <= 0)
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT, "invalid add arguments");
    check(apxinf::cuda_new::pointwise::add_into(
              addend, accumulator, count, static_cast<cudaStream_t>(stream)),
          "add");
  });
}
