#include "../../include/apxinf_cuda/attention.h"

#include "../../framework/runtime_internal.h"
#include "../../kernels/custom/attention_gates.h"

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

extern "C" apxinf_status_t apxinf_attention_split_query_and_gate(
    const void* fused, void* query, void* gate, int64_t tokens, int64_t heads,
    int64_t head_dim, apxinf_cuda_stream_t stream) {
  return abi_boundary([&] {
    if (fused == nullptr || query == nullptr || gate == nullptr ||
        !extent(tokens) || !extent(heads) || !extent(head_dim))
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                    "invalid query/gate split arguments");
    check(apxinf::cuda_new::attention::split_query_and_gate(
              fused, query, gate, static_cast<int>(tokens),
              static_cast<int>(heads), static_cast<int>(head_dim),
              static_cast<cudaStream_t>(stream)),
          "query/gate split");
  });
}

extern "C" apxinf_status_t apxinf_attention_apply_output_gate(
    void* data, const void* gate, int64_t count,
    apxinf_cuda_stream_t stream) {
  return abi_boundary([&] {
    if (data == nullptr || gate == nullptr || count <= 0)
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                    "invalid output gate arguments");
    check(apxinf::cuda_new::attention::apply_output_gate(
              data, gate, count, static_cast<cudaStream_t>(stream)),
          "attention output gate");
  });
}
