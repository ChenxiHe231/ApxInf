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

extern "C" apxinf_status_t apxinf_model_cast_f32_to_bf16(
    const void* input, void* output, int64_t count,
    apxinf_cuda_stream_t stream) {
  return abi_boundary([&] {
    if (input == nullptr || output == nullptr || count <= 0) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT, "invalid cast arguments");
    }
    check(apxinf::cuda_new::model_ops::cast_f32_to_bf16(
              input, output, count, static_cast<cudaStream_t>(stream)),
          "F32 to BF16 cast");
  });
}

extern "C" apxinf_status_t apxinf_model_gelu_exact_bf16(
    const void* input, void* output, int64_t count,
    apxinf_cuda_stream_t stream) {
  return abi_boundary([&] {
    if (input == nullptr || output == nullptr || count <= 0) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                    "invalid exact GELU arguments");
    }
    check(apxinf::cuda_new::model_ops::gelu_exact_bf16(
              input, output, count, static_cast<cudaStream_t>(stream)),
          "exact GELU");
  });
}

extern "C" apxinf_status_t apxinf_model_replace_rows_bf16(
    const void* base, const void* replacement, const void* row_map,
    void* output, int64_t rows, int64_t columns, int64_t replacement_rows,
    apxinf_cuda_stream_t stream) {
  return abi_boundary([&] {
    if (base == nullptr || replacement == nullptr || row_map == nullptr ||
        output == nullptr || !extent(rows) || !extent(columns) ||
        !extent(replacement_rows)) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                    "invalid row replacement arguments");
    }
    check(apxinf::cuda_new::model_ops::replace_rows_bf16(
              base, replacement, static_cast<const int32_t*>(row_map), output,
              static_cast<int>(rows), static_cast<int>(columns),
              static_cast<int>(replacement_rows),
              static_cast<cudaStream_t>(stream)),
          "row replacement");
  });
}
