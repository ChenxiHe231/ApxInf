#include "internal.h"

#include <algorithm>

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
  cudaDeviceProp properties{};
  check_cuda(cudaGetDeviceProperties(&properties, device));

  // Single-token causal KV decode keeps an exact execution Spec/binding so FA2
  // remains eligible, but shares its measured recipe within a 4K live-length
  // bucket. The provider/configuration is restored against each exact Spec.
  // This avoids an autotune on every generated token without forcing the
  // capacity-shaped dynamic-decode contract onto providers that cannot read
  // device-side length metadata.
  const bool live_kv_decode =
      spec.semantic == APXINF_ATTENTION_SEMANTIC_KV_CACHE &&
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
  key << "attention-recipe-v5|ns|" << APXINF_ATTENTION_BUILD_ID << '|'
      << "toolkit=" << CUDART_VERSION << '|'
      << "cc=" << properties.major * 10 + properties.minor
      << "|sms=" << properties.multiProcessorCount
      << "|cuda=" << cuda_compat_version(runtime_version)
      << "|driver=" << cuda_compat_version(driver_version)
      << "|op|" << spec.version << '|' << spec.semantic << '|'
      << spec.dtype << '|' << spec.output_dtype << '|' << spec.mask << '|'
      << spec.batch << '|' << spec.query_tokens << '|';
  if (live_kv_decode) {
    key << "decode-bucket-4096|" << decode_bucket << '|' << decode_bucket << '|';
  } else {
    key << spec.key_tokens << '|' << spec.key_capacity << '|';
  }
  key << spec.query_heads << '|' << spec.kv_heads
      << '|' << spec.head_dim << '|';
  key << spec.dynamic_decode << '|';
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
  return {key.str()};
}

}  // namespace apxinf::attention
