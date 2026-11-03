// Linear-attention (GDN) kernels: recurrent step, causal conv, gated norm, L2 norm, decay/beta.
//
// These ops have a single fixed implementation and no persisted selection,
// so per doc/adding-new-kernels.md §6 they carry no candidate registry,
// tuning key, or autotuner; they are direct C-ABI forwarders.

#include "../../include/apxinf_cuda/gdn.h"

#include "../../framework/runtime_internal.h"
#include "../../kernels/custom/gdn_ops.h"
#include "../../kernels/flashinfer_gdn/flashinfer_gdn.h"

#include <cmath>
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

extern "C" apxinf_status_t apxinf_gdn_recurrent_verify(
    void* state, const void* q, const void* k, const void* v,
    const void* decay, const void* beta, void* output, void* checkpoints,
    int64_t tokens, int64_t v_heads, int64_t k_heads, int64_t v_dim,
    int64_t k_dim, int64_t q_row_stride, int64_t k_row_stride,
    int64_t v_row_stride, apxinf_cuda_stream_t stream) {
  return abi_boundary([&] {
    if (state == nullptr || q == nullptr || k == nullptr || v == nullptr ||
        decay == nullptr || beta == nullptr || output == nullptr ||
        checkpoints == nullptr || !extent(tokens) || !extent(v_heads) ||
        !extent(k_heads) || !extent(v_dim) || !extent(k_dim) ||
        !extent(q_row_stride) || !extent(k_row_stride) ||
        !extent(v_row_stride)) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                    "invalid GDN recurrent verify arguments");
    }
    check(apxinf::cuda_new::gdn_ops::gdn_recurrent_verify(
              state, q, k, v, decay, beta, output, checkpoints,
              static_cast<int>(tokens), static_cast<int>(v_heads),
              static_cast<int>(k_heads), static_cast<int>(v_dim),
              static_cast<int>(k_dim), static_cast<int>(q_row_stride),
              static_cast<int>(k_row_stride), static_cast<int>(v_row_stride),
              static_cast<cudaStream_t>(stream)),
          "GDN recurrent verify");
  });
}

extern "C" apxinf_status_t apxinf_gdn_causal_conv_verify(
    void* window, const void* input, const void* weight, void* output,
    void* checkpoints, int64_t tokens, int64_t channels,
    int64_t kernel_width, apxinf_cuda_stream_t stream) {
  return abi_boundary([&] {
    if (window == nullptr || input == nullptr || weight == nullptr ||
        output == nullptr || checkpoints == nullptr || !extent(tokens) ||
        !extent(channels) || !extent(kernel_width)) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                    "invalid GDN conv verify arguments");
    }
    check(apxinf::cuda_new::gdn_ops::gdn_causal_conv_verify(
              window, input, weight, output, checkpoints,
              static_cast<int>(tokens), static_cast<int>(channels),
              static_cast<int>(kernel_width),
              static_cast<cudaStream_t>(stream)),
          "GDN conv verify");
  });
}

extern "C" apxinf_status_t apxinf_gdn_l2_normalize_heads_seq(
    void* data, int64_t tokens, int64_t heads, int64_t head_dim,
    int64_t row_stride, float epsilon, apxinf_cuda_stream_t stream) {
  return abi_boundary([&] {
    if (data == nullptr || !extent(tokens) || !extent(heads) ||
        !extent(head_dim) || !extent(row_stride)) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                    "invalid GDN sequence normalization arguments");
    }
    check(apxinf::cuda_new::gdn_ops::gdn_l2_normalize_heads_seq(
              data, static_cast<int>(tokens), static_cast<int>(heads),
              static_cast<int>(head_dim), static_cast<int>(row_stride), epsilon,
              static_cast<cudaStream_t>(stream)),
          "GDN sequence head normalization");
  });
}
