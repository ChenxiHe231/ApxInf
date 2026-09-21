#pragma once

#include "../../include/apxinf_cuda/rope.h"
#include "../../framework/registry.h"
#include "../../framework/runtime_internal.h"

#include <memory>
#include <string>
#include <vector>

namespace apxinf::rope {

using apxinf::framework::Failure;
using apxinf::framework::abi_boundary;
using apxinf::framework::check_cuda;

struct Spec : apxinf_rope_spec_t {};

inline bool applies_rope(uint32_t semantic) {
  return semantic == APXINF_ROPE_SEMANTIC_SPLIT_QKV_ROPE;
}

struct Execution;
using EnqueueFn = cudaError_t (*)(Execution&);

struct AlignmentRequirements {
  uint32_t qkv = 1;
  uint32_t bias = 1;
  uint32_t q = 1;
  uint32_t kv = 1;
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
  apxinf_rope_bindings_t bindings{};
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
                const apxinf_rope_policy_t& policy,
                const apxinf_rope_bindings_t& bindings, int device);

cudaError_t launch_custom(Execution& execution);

}  // namespace apxinf::rope
