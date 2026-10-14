// Reduction / model-head kernels: embedding gather, argmax.
//
// These ops have a single fixed implementation and no persisted selection,
// so per doc/adding-new-kernels.md §6 they carry no candidate registry,
// tuning key, or autotuner; they are direct C-ABI forwarders.

#include "../../include/apxinf_cuda/model.h"

#include "../../framework/runtime_internal.h"
#include "../../kernels/custom/model_ops.h"

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

bool extent(int64_t value) { return value > 0 && value <= INT32_MAX; }

}  // namespace

extern "C" apxinf_status_t apxinf_model_embedding_gather(
    const void* table, const void* ids, void* output, int64_t tokens,
    int64_t hidden, int64_t vocab, apxinf_cuda_stream_t stream) {
  return abi_boundary([&] {
    if (table == nullptr || ids == nullptr || output == nullptr ||
        !extent(tokens) || !extent(hidden) || !extent(vocab)) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                    "invalid embedding gather arguments");
    }
    check(apxinf::cuda_new::model_ops::embedding_gather(
              table, static_cast<const int32_t*>(ids), output,
              static_cast<int>(tokens), static_cast<int>(hidden),
              static_cast<int>(vocab), static_cast<cudaStream_t>(stream)),
          "embedding gather");
  });
}

extern "C" apxinf_status_t apxinf_model_argmax_bf16(
    const void* logits, void* index, int64_t count,
    apxinf_cuda_stream_t stream) {
  return abi_boundary([&] {
    if (logits == nullptr || index == nullptr || !extent(count)) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT, "invalid argmax arguments");
    }
    check(apxinf::cuda_new::model_ops::argmax_bf16(
              logits, static_cast<int32_t*>(index), static_cast<int>(count),
              static_cast<cudaStream_t>(stream)),
          "argmax");
  });
}

extern "C" apxinf_status_t apxinf_model_replace_rows_bf16(
    void* destination, const void* source, const void* source_rows,
    int64_t destination_rows, int64_t source_row_count, int64_t columns,
    apxinf_cuda_stream_t stream) {
  return abi_boundary([&] {
    if (destination == nullptr || source == nullptr || source_rows == nullptr ||
        !extent(destination_rows) || !extent(source_row_count) ||
        !extent(columns)) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                    "invalid row replacement arguments");
    }
    check(apxinf::cuda_new::model_ops::replace_rows_bf16(
              destination, source, static_cast<const uint32_t*>(source_rows),
              static_cast<int>(destination_rows),
              static_cast<int>(source_row_count), static_cast<int>(columns),
              static_cast<cudaStream_t>(stream)),
          "row replacement");
  });
}
