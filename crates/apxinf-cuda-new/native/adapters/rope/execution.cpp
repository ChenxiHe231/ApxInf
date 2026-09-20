#include "internal.h"

#include <cmath>
#include <vector>

namespace {

using apxinf::rope::Execution;
using apxinf::rope::Failure;
using apxinf::rope::Spec;

bool valid_alignment(uint32_t alignment) {
  return alignment <= 256 && alignment != 0 &&
         (alignment & (alignment - 1)) == 0;
}

void validate_spec(const Spec& spec) {
  if (spec.version != APXINF_ROPE_SPEC_VERSION ||
      spec.semantic > APXINF_ROPE_SEMANTIC_SPLIT_QKV_BIAS ||
      (spec.dtype != APXINF_DTYPE_F16 && spec.dtype != APXINF_DTYPE_BF16) ||
      spec.has_bias > 1 || spec.tokens <= 0 || spec.tokens > INT32_MAX ||
      spec.q_heads == 0 || spec.kv_heads == 0 || spec.head_dim == 0 ||
      spec.q_heads > INT32_MAX || spec.kv_heads > INT32_MAX ||
      spec.head_dim > 2048 || spec.head_dim % 2 != 0 ||
      spec.q_heads % spec.kv_heads != 0 ||
      static_cast<uint64_t>(spec.q_heads) + 2ull * spec.kv_heads > 65535 ||
      !valid_alignment(spec.qkv_alignment) ||
      !valid_alignment(spec.bias_alignment) ||
      !valid_alignment(spec.q_alignment) ||
      !valid_alignment(spec.kv_alignment)) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT, "invalid RoPE Spec");
  }
  // The vision split has no grouped-query layout: Q, K and V share a width.
  if (spec.semantic == APXINF_ROPE_SEMANTIC_SPLIT_QKV_BIAS &&
      spec.q_heads != spec.kv_heads) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                  "unrotated QKV split requires q_heads == kv_heads");
  }
}

void validate_bindings(const Spec& spec,
                       const apxinf_rope_bindings_t& bindings) {
  if (bindings.qkv == nullptr || bindings.q == nullptr ||
      bindings.k == nullptr || bindings.v == nullptr) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT, "missing RoPE binding");
  }
  if ((spec.has_bias != 0) != (bindings.bias != nullptr)) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                  "RoPE bias binding disagrees with Spec.has_bias");
  }
  if (bindings.kv_output_offset < 0 || bindings.position_offset < 0) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT, "negative RoPE offset");
  }
  if (apxinf::rope::applies_rope(spec.semantic) &&
      (!std::isfinite(bindings.theta) || bindings.theta <= 0.0f)) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT, "invalid RoPE theta");
  }
  // The unrotated split has no notion of position, so a caller passing one is
  // confused about which semantic it wants.
  if (!apxinf::rope::applies_rope(spec.semantic) &&
      (bindings.position_offset != 0 || bindings.kv_output_offset != 0)) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                  "unrotated QKV split does not take offsets");
  }
}

}  // namespace

namespace apxinf::rope {

Execution::~Execution() {
  if (implementation != nullptr && implementation->destroy != nullptr) {
    implementation->destroy(*this);
  }
}

bool supports_device(const Implementation& implementation, int,
                     std::string* reason) {
  if (implementation.required_device_features == 0) return true;
  if (reason != nullptr) *reason = "missing required device feature";
  return false;
}

bool supports_alignment(const Implementation& implementation,
                        const Spec& spec) {
  const auto required = implementation.alignment_requirements(spec);
  return spec.qkv_alignment >= required.qkv &&
         spec.q_alignment >= required.q && spec.kv_alignment >= required.kv &&
         spec.bias_alignment >= (spec.has_bias != 0 ? required.bias : 0);
}

std::unique_ptr<Execution> prepare(const Implementation& implementation,
                                   int configuration, const Spec& spec,
                                   const apxinf_rope_policy_t& policy,
                                   const apxinf_rope_bindings_t& bindings,
                                   int device) {
  auto execution = std::make_unique<Execution>();
  execution->spec = spec;
  execution->bindings = bindings;
  execution->configuration = configuration;
  execution->device = device;
  execution->implementation = &implementation;
  execution->resource_limit = policy.workspace_limit;
  implementation.prepare(*execution);
  if (execution->resource_bytes > execution->resource_limit &&
      execution->resource_limit != 0) {
    throw Failure(APXINF_STATUS_UNSUPPORTED,
                  "RoPE candidate exceeds the workspace limit");
  }
  return execution;
}

}  // namespace apxinf::rope

extern "C" apxinf_status_t apxinf_rope_prepare(
    apxinf_runtime_t runtime, const apxinf_rope_spec_t* spec,
    const apxinf_rope_policy_t* policy,
    const apxinf_rope_bindings_t* bindings,
    apxinf_rope_execution_t* output) {
  return apxinf::rope::abi_boundary([&] {
    if (runtime == nullptr || spec == nullptr || policy == nullptr ||
        bindings == nullptr || output == nullptr) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT, "null RoPE argument");
    }
    const apxinf::rope::Spec normalized{*spec};
    validate_spec(normalized);
    validate_bindings(normalized, *bindings);

    cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(static_cast<cudaStream_t>(bindings->stream),
                              &capture) == cudaSuccess &&
        capture != cudaStreamCaptureStatusNone) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                    "RoPE preparation is not allowed during capture");
    }

    std::unique_ptr<apxinf::rope::Execution> execution;
    for (const auto& implementation :
         apxinf::rope::registry(normalized.semantic)) {
      if (!implementation.supports(normalized) ||
          !apxinf::rope::supports_device(implementation, runtime->device) ||
          !apxinf::rope::supports_alignment(implementation, normalized)) {
        continue;
      }
      if (policy->graph_safe && !implementation.graph_safe) continue;
      if (policy->deterministic && !implementation.deterministic) continue;
      if (!policy->allow_fallback && implementation.fallback) continue;
      std::vector<int> configurations;
      implementation.enumerate_configs(normalized, configurations);
      if (configurations.empty()) continue;
      execution = apxinf::rope::prepare(implementation, configurations.front(),
                                        normalized, *policy, *bindings,
                                        runtime->device);
      break;
    }
    if (execution == nullptr) {
      throw Failure(APXINF_STATUS_UNSUPPORTED,
                    "no RoPE candidate supports this Spec");
    }
    execution->summary = std::string(execution->implementation->name) +
                         " config=" +
                         std::to_string(execution->configuration) +
                         " source=direct";
    auto wrapper = std::make_unique<apxinf_rope_execution>();
    wrapper->state = std::move(execution);
    *output = wrapper.release();
  });
}

extern "C" apxinf_status_t apxinf_rope_enqueue(
    apxinf_rope_execution_t execution) {
  return apxinf::rope::abi_boundary([&] {
    if (execution == nullptr || execution->state == nullptr) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT, "null RoPE execution");
    }
    apxinf::rope::check_cuda(cudaSetDevice(execution->state->device));
    apxinf::rope::check_cuda(
        execution->state->implementation->enqueue(*execution->state));
  });
}

extern "C" void apxinf_rope_destroy(apxinf_rope_execution_t execution) {
  delete execution;
}

extern "C" const char* apxinf_rope_summary(apxinf_rope_execution_t execution) {
  return execution == nullptr || execution->state == nullptr
             ? ""
             : execution->state->summary.c_str();
}
