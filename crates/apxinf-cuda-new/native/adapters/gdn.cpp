#include "../include/apxinf_cuda/gdn.h"

#include "../framework/runtime_internal.h"
#include "../kernels/custom/gdn_ops.h"

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

extern "C" apxinf_status_t apxinf_gdn_recurrent_step(
    void* state, const void* q, const void* k, const void* v,
    const void* decay, const void* beta, void* output, int64_t v_heads,
    int64_t k_heads, int64_t v_dim, int64_t k_dim,
    apxinf_cuda_stream_t stream) {
  return abi_boundary([&] {
    if (state == nullptr || q == nullptr || k == nullptr || v == nullptr ||
        decay == nullptr || beta == nullptr || output == nullptr ||
        !extent(v_heads) || !extent(k_heads) || !extent(v_dim) ||
        !extent(k_dim)) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                    "invalid GDN recurrent step arguments");
    }
    check(apxinf::cuda::gdn_ops::gdn_recurrent_step(
              state, q, k, v, decay, beta, output, static_cast<int>(v_heads),
              static_cast<int>(k_heads), static_cast<int>(v_dim),
              static_cast<int>(k_dim), static_cast<cudaStream_t>(stream)),
          "GDN recurrent step");
  });
}

extern "C" apxinf_status_t apxinf_gdn_gated_norm(
    const void* input, const void* gate, const void* weight, void* output,
    int64_t heads, int64_t head_dim, float epsilon,
    apxinf_cuda_stream_t stream) {
  return abi_boundary([&] {
    if (input == nullptr || gate == nullptr || weight == nullptr ||
        output == nullptr || !extent(heads) || !extent(head_dim)) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                    "invalid GDN gated norm arguments");
    }
    check(apxinf::cuda::gdn_ops::gdn_gated_norm(
              input, gate, weight, output, static_cast<int>(heads),
              static_cast<int>(head_dim), epsilon,
              static_cast<cudaStream_t>(stream)),
          "GDN gated norm");
  });
}

extern "C" apxinf_status_t apxinf_gdn_causal_conv_step(
    void* window, const void* input, const void* weight, void* output,
    int64_t channels, int64_t kernel_width, apxinf_cuda_stream_t stream) {
  return abi_boundary([&] {
    if (window == nullptr || input == nullptr || weight == nullptr ||
        output == nullptr || !extent(channels) || !extent(kernel_width)) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                    "invalid GDN conv step arguments");
    }
    check(apxinf::cuda::gdn_ops::gdn_causal_conv_step(
              window, input, weight, output, static_cast<int>(channels),
              static_cast<int>(kernel_width),
              static_cast<cudaStream_t>(stream)),
          "GDN conv step");
  });
}

extern "C" apxinf_status_t apxinf_gdn_l2_normalize_heads(
    void* data, int64_t heads, int64_t head_dim, float epsilon,
    apxinf_cuda_stream_t stream) {
  return abi_boundary([&] {
    if (data == nullptr || !extent(heads) || !extent(head_dim)) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                    "invalid GDN normalization arguments");
    }
    check(apxinf::cuda::gdn_ops::gdn_l2_normalize_heads(
              data, static_cast<int>(heads), static_cast<int>(head_dim),
              epsilon, static_cast<cudaStream_t>(stream)),
          "GDN head normalization");
  });
}

extern "C" apxinf_status_t apxinf_gdn_decay_and_beta(
    const void* a, const void* b, const void* a_log, const void* dt_bias,
    void* decay, void* beta, int64_t heads, apxinf_cuda_stream_t stream) {
  return abi_boundary([&] {
    if (a == nullptr || b == nullptr || a_log == nullptr ||
        dt_bias == nullptr || decay == nullptr || beta == nullptr ||
        !extent(heads)) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                    "invalid GDN gate arguments");
    }
    check(apxinf::cuda::gdn_ops::gdn_decay_and_beta(
              a, b, a_log, dt_bias, decay, beta, static_cast<int>(heads),
              static_cast<cudaStream_t>(stream)),
          "GDN gates");
  });
}
