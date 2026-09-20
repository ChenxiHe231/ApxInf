#pragma once

#include "../../include/apxinf_cuda/quantization.h"
#include "../../framework/registry.h"
#include "../../framework/runtime_internal.h"

#include <memory>
#include <string>
#include <vector>

namespace apxinf::quantization {

using apxinf::framework::Failure;
using apxinf::framework::abi_boundary;
using apxinf::framework::check_cuda;

struct Spec : apxinf_quantization_spec_t {};

inline bool has_row_scales(uint32_t semantic) {
  return semantic == APXINF_QUANTIZATION_SEMANTIC_ROWWISE_E4M3 ||
         semantic == APXINF_QUANTIZATION_SEMANTIC_ROWWISE_I8;
}

struct Execution;
using PrepareExecutionFn = void (*)(Execution&);
using EnqueueFn = cudaError_t (*)(Execution&);
using DestroyExecutionFn = void (*)(Execution&) noexcept;

struct AlignmentRequirements {
  uint32_t input = 1;
  uint32_t output = 1;
  uint32_t scales = 1;
};

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
  size_t (*resource_requirements)(const Spec&);
  void (*enumerate_configs)(const Spec&, std::vector<int>&);
  PrepareExecutionFn prepare;
  EnqueueFn enqueue;
  DestroyExecutionFn destroy;
};

using ImplementationRegistry = apxinf::framework::Registry<Implementation>;

struct Execution {
  Spec spec{};
  apxinf_quantization_bindings_t bindings{};
  int configuration = 0;
  int device = 0;
  const Implementation* implementation = nullptr;
  void* provider_state = nullptr;
  size_t resource_bytes = 0;
  size_t resource_limit = 0;
  std::string summary;

  ~Execution();
};

const ImplementationRegistry& registry(uint32_t semantic);
bool supports_device(const Implementation& implementation, int device,
                     std::string* reason = nullptr);
bool supports_alignment(const Implementation& implementation, const Spec& spec);
std::unique_ptr<Execution> prepare(
    const Implementation& implementation, int configuration, const Spec& spec,
    const apxinf_quantization_policy_t& policy,
    const apxinf_quantization_bindings_t& bindings, int device);

size_t ported_resource_requirements(const Spec& spec);
void prepare_ported(Execution& execution);
cudaError_t launch_ported(Execution& execution);
void destroy_ported(Execution& execution) noexcept;

}  // namespace apxinf::quantization

struct apxinf_quantization_execution {
  std::unique_ptr<apxinf::quantization::Execution> state;
};
