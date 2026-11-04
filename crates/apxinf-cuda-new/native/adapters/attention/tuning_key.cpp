#include "internal.h"

#include <algorithm>
#include <cublas_v2.h>

#ifndef APXINF_ATTENTION_BUILD_ID
#define APXINF_ATTENTION_BUILD_ID "attention-development"
#endif

namespace apxinf::attention {
namespace {

int cuda_compat_version(int version) {
  return (version / 1000) * 100 + ((version % 1000) / 10);
}

uint32_t alignment_class(uint32_t alignment, uint32_t maximum) {
  return alignment == 0 ? 0 : std::min(alignment, maximum);
}

}  // namespace

TuningKeys tuning_keys(const Spec& spec,
                       const apxinf_attention_policy_t& policy, int device) {
  int runtime_version = 0;
  int driver_version = 0;
  check_cuda(cudaRuntimeGetVersion(&runtime_version));
  check_cuda(cudaDriverGetVersion(&driver_version));
  int blas_major = 0, blas_minor = 0, blas_patch = 0;
  if (cublasGetProperty(MAJOR_VERSION, &blas_major) != CUBLAS_STATUS_SUCCESS ||
      cublasGetProperty(MINOR_VERSION, &blas_minor) != CUBLAS_STATUS_SUCCESS ||
      cublasGetProperty(PATCH_LEVEL, &blas_patch) != CUBLAS_STATUS_SUCCESS)
    throw Failure(APXINF_STATUS_PROVIDER_ERROR, "cannot query cuBLAS Attention version");
  cudaDeviceProp properties{};
  check_cuda(cudaGetDeviceProperties(&properties, device));

  // One exact key: the namespace prefix captures the candidate/toolchain/GPU
  // identity and the suffix captures the normalized operation and policy.
  // There is no compatible fallback key; a miss runs the full autotuner.
  const bool stable =
      spec.numerics == APXINF_ATTENTION_NUMERICS_STABLE;
  // Single-token causal KV decode keeps an exact execution Spec/binding so FA2
  // remains eligible, but shares its measured recipe within a 4K live-length
  // bucket. The provider/configuration is restored against each exact Spec.
  // This avoids an autotune on every generated token without forcing the
  // capacity-shaped dynamic-decode contract onto providers that cannot read
  // device-side length metadata.
  const bool live_kv_decode =
      !stable && spec.semantic == APXINF_ATTENTION_SEMANTIC_KV_CACHE &&
      spec.dynamic_decode == 0 && spec.batch == 1 &&
      spec.query_tokens == 1 && spec.mask == APXINF_ATTENTION_MASK_CAUSAL &&
      spec.query_start == spec.key_tokens - 1;
  constexpr int64_t kDecodeBucketTokens = 4096;
  const int64_t decode_bucket =
      live_kv_decode
          ? ((spec.key_tokens + kDecodeBucketTokens - 1) /
             kDecodeBucketTokens) *
                kDecodeBucketTokens
          : 0;

  std::ostringstream key;
  key << (stable ? "attention-recipe-v8|ns|" : "attention-recipe-v7|ns|")
      << APXINF_ATTENTION_BUILD_ID << '|'
      << "toolkit=" << CUDART_VERSION << '|'
      << "cc=" << properties.major * 10 + properties.minor
      << "|sms=" << properties.multiProcessorCount
      << "|cuda=" << cuda_compat_version(runtime_version)
      << "|driver=" << cuda_compat_version(driver_version)
      << "|cublas=" << blas_major << '.' << blas_minor << '.' << blas_patch
      // Flexible numerics retains the pre-v5 normalized suffix. The build
      // fingerprint still invalidates recipes when candidate code changes.
      << "|op|" << spec.version << '|' << spec.semantic << '|'
      << spec.qkv_layout << '|' << spec.dtype << '|' << spec.output_dtype << '|' << spec.mask << '|'
      << spec.batch << '|' << spec.query_tokens << '|';
  if (live_kv_decode) {
    key << "decode-bucket-4096|" << decode_bucket << '|' << decode_bucket << '|';
  } else {
    key << spec.key_tokens << '|' << spec.key_capacity << '|';
  }
  key << spec.query_heads << '|' << spec.kv_heads
      << '|' << spec.head_dim << '|';
  key << spec.dynamic_decode << '|';
  if (stable) key << spec.numerics << '|';
  if (live_kv_decode) {
    key << "decode-live|";
  } else if (spec.semantic == APXINF_ATTENTION_SEMANTIC_KV_CACHE &&
      spec.dynamic_decode != 0) {
    key << "decode-meta|";
  } else {
    key << spec.query_start << '|';
  }
  key << spec.segments << '|' << spec.max_segment_tokens << '|'
      << spec.scale_is_default << '|'
      << alignment_class(spec.q_alignment, 16) << '|'
      << alignment_class(spec.k_alignment, 16) << '|'
      << alignment_class(spec.v_alignment, 16) << '|'
      << alignment_class(spec.output_alignment, 16) << '|'
      << alignment_class(spec.offsets_alignment, 4) << '|'
      << policy.workspace_limit << '|' << policy.graph_safe << '|'
      << policy.deterministic;
#if defined(APXINF_ATTENTION_FA2)
  // A new provider/layout must never inherit the ordinary split-KV recipe.
  if (fa2_packed_gqa_requested(spec)) key << "|packed-gqa-q1to4-v1";
#endif
  return {key.str()};
}

}  // namespace apxinf::attention
