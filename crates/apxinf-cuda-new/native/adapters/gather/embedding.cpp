#include "../../include/apxinf_cuda/gather.h"

#include "../../framework/runtime_internal.h"
#include "../../kernels/custom/gather_embedding.h"

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

extern "C" apxinf_status_t apxinf_gather_embedding(
    const void* table, const void* ids, void* output, int64_t tokens,
    int64_t hidden, int64_t vocab, apxinf_cuda_stream_t stream) {
  return abi_boundary([&] {
    if (table == nullptr || ids == nullptr || output == nullptr ||
        !extent(tokens) || !extent(hidden) || !extent(vocab))
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                    "invalid embedding gather arguments");
    check(apxinf::cuda_new::gather::embedding_gather(
              table, static_cast<const int32_t*>(ids), output,
              static_cast<int>(tokens), static_cast<int>(hidden),
              static_cast<int>(vocab), static_cast<cudaStream_t>(stream)),
          "embedding gather");
  });
}
