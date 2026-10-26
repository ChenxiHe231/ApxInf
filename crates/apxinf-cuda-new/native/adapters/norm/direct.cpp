#include "../../include/apxinf_cuda/norm.h"

#include "../../framework/runtime_internal.h"
#include "../../kernels/custom/norm_head.h"
#include "../../kernels/custom/norm_weight_offset.h"

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

extern "C" apxinf_status_t apxinf_norm_head_rms(
    void* data, const void* weight, int64_t rows, int64_t head_dim,
    float epsilon, apxinf_cuda_stream_t stream) {
  return abi_boundary([&] {
    if (data == nullptr || weight == nullptr || !extent(rows) ||
        !extent(head_dim))
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                    "invalid head RMSNorm arguments");
    check(apxinf::cuda_new::norm::head_rms_norm(
              data, weight, static_cast<int>(rows), static_cast<int>(head_dim),
              epsilon, static_cast<cudaStream_t>(stream)),
          "head RMSNorm");
  });
}

extern "C" apxinf_status_t apxinf_norm_rms_weight_offset(
    const void* input, const void* weight, void* output, int64_t rows,
    int64_t width, float epsilon, apxinf_cuda_stream_t stream) {
  return abi_boundary([&] {
    if (input == nullptr || weight == nullptr || output == nullptr ||
        !extent(rows) || !extent(width) || !(epsilon >= 0.0F))
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT, "invalid RMSNorm arguments");
    check(apxinf::cuda_new::norm::rms_norm_weight_offset(
              input, weight, output, static_cast<int>(rows),
              static_cast<int>(width), epsilon,
              static_cast<cudaStream_t>(stream)),
          "RMSNorm");
  });
}
