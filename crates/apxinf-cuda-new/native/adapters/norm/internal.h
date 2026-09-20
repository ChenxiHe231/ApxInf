#pragma once

#include "../../include/apxinf_cuda/norm.h"
#include "../../framework/registry.h"
#include "../../framework/runtime_internal.h"

#include <memory>
#include <string>
#include <vector>

namespace apxinf::norm {

using apxinf::framework::Failure;
using apxinf::framework::abi_boundary;
using apxinf::framework::check_cuda;

struct Spec : apxinf_norm_spec_t {};

inline size_t dtype_bytes(uint32_t dtype) {
  return dtype == APXINF_DTYPE_E4M3 ? 1 : 2;
}

// Which bindings a semantic reads.  Kept as predicates rather than a table so
// that validation, alignment checks and the reference implementation cannot
// drift apart.
inline bool writes_hidden(uint32_t semantic) {
  return semantic != APXINF_NORM_SEMANTIC_RMS &&
         semantic != APXINF_NORM_SEMANTIC_LAYER &&
         semantic != APXINF_NORM_SEMANTIC_ADAPTIVE_RMS;
}

inline bool writes_normalized(uint32_t semantic) {
  return semantic != APXINF_NORM_SEMANTIC_BIAS_RESIDUAL &&
         semantic != APXINF_NORM_SEMANTIC_ADA_GATE_RESIDUAL &&
         semantic != APXINF_NORM_SEMANTIC_BIAS_THEN_RESIDUAL;
}

inline bool reads_residual(uint32_t semantic) {
  return writes_hidden(semantic);
}

inline bool reads_weight(uint32_t semantic) {
  return semantic == APXINF_NORM_SEMANTIC_RMS ||
         semantic == APXINF_NORM_SEMANTIC_LAYER ||
         semantic == APXINF_NORM_SEMANTIC_BIAS_RESIDUAL_RMS ||
         semantic == APXINF_NORM_SEMANTIC_BIAS_RESIDUAL_LAYER;
}

inline bool reads_norm_bias(uint32_t semantic) {
  return semantic == APXINF_NORM_SEMANTIC_LAYER ||
         semantic == APXINF_NORM_SEMANTIC_BIAS_RESIDUAL_LAYER;
}

inline bool reads_norm_style(uint32_t semantic) {
  return semantic == APXINF_NORM_SEMANTIC_ADAPTIVE_RMS ||
         semantic == APXINF_NORM_SEMANTIC_ADA_GATE_RESIDUAL_RMS;
}

inline bool reads_gate_style(uint32_t semantic) {
  return semantic == APXINF_NORM_SEMANTIC_ADA_GATE_RESIDUAL ||
         semantic == APXINF_NORM_SEMANTIC_ADA_GATE_RESIDUAL_RMS;
}

// Only the bias/residual combine semantics carry an optional bias.
inline bool may_have_bias(uint32_t semantic) {
  return semantic == APXINF_NORM_SEMANTIC_BIAS_RESIDUAL ||
         semantic == APXINF_NORM_SEMANTIC_BIAS_RESIDUAL_RMS ||
         semantic == APXINF_NORM_SEMANTIC_BIAS_RESIDUAL_LAYER ||
         semantic == APXINF_NORM_SEMANTIC_BIAS_THEN_RESIDUAL;
}

struct Execution;
using PrepareExecutionFn = void (*)(Execution&);
using EnqueueFn = cudaError_t (*)(Execution&);
using DestroyExecutionFn = void (*)(Execution&) noexcept;

struct AlignmentRequirements {
  uint32_t input = 1;
  uint32_t weight = 1;
  uint32_t bias = 1;
  uint32_t residual = 1;
  uint32_t style = 1;
  uint32_t hidden = 1;
  uint32_t normalized = 1;
};

// Mirrors the gemm and attention descriptor so the families stay readable
// side by side. These kernels are the canonical provider, not a fallback;
// there is one implementation per semantic, so there is nothing to time.
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
  apxinf_norm_bindings_t bindings{};
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
std::unique_ptr<Execution> prepare(const Implementation& implementation,
                                   int configuration, const Spec& spec,
                                   const apxinf_norm_policy_t& policy,
                                   const apxinf_norm_bindings_t& bindings,
                                   int device);

size_t custom_resource_requirements(const Spec& spec);
void prepare_custom(Execution& execution);
cudaError_t launch_custom(Execution& execution);
void destroy_custom(Execution& execution) noexcept;

}  // namespace apxinf::norm

struct apxinf_norm_execution {
  std::unique_ptr<apxinf::norm::Execution> state;
};
