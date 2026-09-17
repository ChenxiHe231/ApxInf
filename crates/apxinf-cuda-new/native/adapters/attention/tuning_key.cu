#include "internal.h"

#ifndef APXINF_ATTENTION_BUILD_ID
#define APXINF_ATTENTION_BUILD_ID "attention-development"
#endif

namespace apxinf::attention {

TuningKeys tuning_keys(const Spec& spec,
                       const apxinf_attention_policy_t& policy, int device) {
  int runtime_version = 0;
  int driver_version = 0;
  int memory_clock_rate = 0;
  check_cuda(cudaRuntimeGetVersion(&runtime_version));
  check_cuda(cudaDriverGetVersion(&driver_version));
  cudaDeviceProp properties{};
  check_cuda(cudaGetDeviceProperties(&properties, device));
  check_cuda(cudaDeviceGetAttribute(&memory_clock_rate,
                                    cudaDevAttrMemoryClockRate, device));

  std::ostringstream common;
  common << "attention-recipe-v3|" << APXINF_ATTENTION_BUILD_ID << '|'
         << properties.major * 10 + properties.minor << '|'
         << runtime_version << '|' << driver_version << '|'
         << spec.version << '|' << spec.semantic << '|' << spec.dtype << '|'
         << spec.output_dtype << '|' << spec.mask << '|' << spec.batch << '|'
         << spec.query_tokens << '|' << spec.key_tokens << '|'
         << spec.key_capacity << '|' << spec.query_heads << '|'
         << spec.kv_heads << '|' << spec.head_dim << '|' << spec.query_start
         << '|' << spec.segments << '|' << spec.max_segment_tokens << '|'
         << spec.offsets_hash << '|' << spec.scale_is_default << '|'
         << spec.q_alignment << '|' << spec.k_alignment << '|'
         << spec.v_alignment << '|' << spec.output_alignment << '|'
         << spec.offsets_alignment << '|'
         << policy.workspace_limit << '|' << policy.graph_safe << '|'
         << policy.deterministic;

  std::ostringstream compatible;
  compatible << common.str() << "|compat";
  std::ostringstream performance;
  performance << common.str() << "|perf|" << properties.name << '|'
              << properties.multiProcessorCount << '|'
              << memory_clock_rate << '|' << properties.memoryBusWidth;
  return {performance.str(), compatible.str()};
}

}  // namespace apxinf::attention
