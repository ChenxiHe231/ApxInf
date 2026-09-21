#pragma once

#include "../../include/apxinf_cuda/pointwise.h"
#include "../../framework/registry.h"
#include "../../framework/runtime_internal.h"

#include <memory>
#include <string>
#include <vector>

namespace apxinf::pointwise {

using apxinf::framework::Failure;
using apxinf::framework::abi_boundary;
using apxinf::framework::check_cuda;

struct Spec : apxinf_pointwise_spec_t {};

inline size_t dtype_bytes(uint32_t dtype) {
  return dtype == APXINF_DTYPE_E4M3 ? 1 : 2;
}

inline bool reads_secondary(uint32_t semantic) {
  return semantic == APXINF_POINTWISE_SEMANTIC_EULER_UPDATE;
}

inline bool may_have_bias(uint32_t semantic) {
  return semantic == APXINF_POINTWISE_SEMANTIC_BIAS_ACTIVATION;
}

// GeGLU consumes a gate and an up half per output element.
inline int64_t input_cols(const Spec& spec) {
  return spec.semantic == APXINF_POINTWISE_SEMANTIC_GEGLU ? spec.cols * 2
                                                          : spec.cols;
}

struct Execution;
using EnqueueFn = cudaError_t (*)(Execution&);

struct AlignmentRequirements {
  uint32_t input = 1;
  uint32_t secondary = 1;
  uint32_t bias = 1;
  uint32_t output = 1;
};

// See the note in adapters/norm/internal.h: a single memory-bound candidate
// Each semantic currently has one canonical custom implementation.
struct Implementation {
  uint32_t provider_id;
  uint32_t implementation_id;
  uint32_t implementation_version;
  const char* name;
  uint64_t required_device_features;
  bool graph_safe;
  bool deterministic;
  bool fallback;
  bool (*supports)(const Spec&);
  AlignmentRequirements (*alignment_requirements)(const Spec&);
  void (*enumerate_configs)(const Spec&, std::vector<int>&);
  EnqueueFn enqueue;
};

using ImplementationRegistry = apxinf::framework::Registry<Implementation>;

struct Execution {
  Spec spec{};
  apxinf_pointwise_bindings_t bindings{};
  int configuration = 0;
  int device = 0;
  const Implementation* implementation = nullptr;
};

const ImplementationRegistry& registry(uint32_t semantic);
bool supports_device(const Implementation& implementation, int device,
                     std::string* reason = nullptr);
bool supports_alignment(const Implementation& implementation, const Spec& spec);
void initialize(Execution& execution, const Implementation& implementation,
                int configuration, const Spec& spec,
                const apxinf_pointwise_policy_t& policy,
                const apxinf_pointwise_bindings_t& bindings, int device);

cudaError_t launch_custom(Execution& execution);

}  // namespace apxinf::pointwise
