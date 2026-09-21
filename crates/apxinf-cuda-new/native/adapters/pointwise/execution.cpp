#include "internal.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cmath>
#include <sstream>
#include <vector>

namespace {

using apxinf::pointwise::Execution;
using apxinf::pointwise::Failure;
using apxinf::pointwise::Spec;

bool valid_alignment(uint32_t alignment) {
  return alignment <= 256 && alignment != 0 &&
         (alignment & (alignment - 1)) == 0;
}

void validate_spec(const Spec& spec) {
  if (spec.version != APXINF_POINTWISE_SPEC_VERSION ||
      spec.semantic > APXINF_POINTWISE_SEMANTIC_EULER_UPDATE ||
      (spec.dtype != APXINF_DTYPE_F16 && spec.dtype != APXINF_DTYPE_BF16) ||
      (spec.output_dtype != spec.dtype &&
       spec.output_dtype != APXINF_DTYPE_E4M3) ||
      spec.activation > APXINF_POINTWISE_ACTIVATION_SILU ||
      spec.has_bias > 1 || spec.rows <= 0 || spec.cols <= 0 ||
      spec.rows > INT32_MAX || spec.cols > INT32_MAX ||
      spec.output_scale_is_unit > 1 ||
      !valid_alignment(spec.input_alignment) ||
      !valid_alignment(spec.secondary_alignment) ||
      !valid_alignment(spec.bias_alignment) ||
      !valid_alignment(spec.output_alignment)) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT, "invalid Pointwise Spec");
  }
  if (spec.has_bias != 0 &&
      !apxinf::pointwise::may_have_bias(spec.semantic)) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                  "Pointwise semantic does not take a bias");
  }
  // An activation only exists on the semantic that applies one; otherwise two
  // different Specs would describe the same kernel.
  if (spec.activation != APXINF_POINTWISE_ACTIVATION_NONE &&
      spec.semantic != APXINF_POINTWISE_SEMANTIC_BIAS_ACTIVATION) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                  "Pointwise semantic does not take an activation");
  }
  // Bias-activation with neither a bias nor an activation is a plain copy;
  // reject it so callers do not pay a kernel launch for nothing.
  if (spec.semantic == APXINF_POINTWISE_SEMANTIC_BIAS_ACTIVATION &&
      spec.has_bias == 0 &&
      spec.activation == APXINF_POINTWISE_ACTIVATION_NONE) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                  "Pointwise bias-activation is a no-op");
  }
  if (spec.output_dtype == APXINF_DTYPE_E4M3 &&
      spec.output_scale_is_unit != 0) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                  "E4M3 Pointwise output requires a non-unit output scale");
  }
  if (spec.output_dtype == spec.dtype && spec.output_scale_is_unit != 1) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                  "unquantized Pointwise output requires a unit output scale");
  }
}

void validate_bindings(const Spec& spec,
                       const apxinf_pointwise_bindings_t& bindings) {
  if (bindings.input == nullptr || bindings.output == nullptr ||
      (apxinf::pointwise::reads_secondary(spec.semantic) &&
       bindings.secondary == nullptr)) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                  "missing Pointwise binding for semantic");
  }
  if ((spec.has_bias != 0) != (bindings.bias != nullptr)) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                  "Pointwise bias binding disagrees with Spec.has_bias");
  }
  if ((spec.output_scale_is_unit != 0) != (bindings.output_scale == 1.0f)) {
    throw Failure(
        APXINF_STATUS_INVALID_ARGUMENT,
        "Pointwise output_scale disagrees with Spec.output_scale_is_unit");
  }
  if (!std::isfinite(bindings.dt) ||
      !std::isfinite(bindings.output_scale) || bindings.output_scale <= 0.0f) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT, "invalid Pointwise scalar");
  }
}

float read_element(const void* buffer, uint32_t dtype, size_t index) {
  if (dtype == APXINF_DTYPE_BF16) {
    return __bfloat162float(static_cast<const __nv_bfloat16*>(buffer)[index]);
  }
  return __half2float(static_cast<const __half*>(buffer)[index]);
}

}  // namespace

namespace apxinf::pointwise {

bool supports_device(const Implementation& implementation, int,
                     std::string* reason) {
  if (implementation.required_device_features == 0) return true;
  if (reason != nullptr) *reason = "missing required device feature";
  return false;
}

bool supports_alignment(const Implementation& implementation,
                        const Spec& spec) {
  const auto required = implementation.alignment_requirements(spec);
  return spec.input_alignment >= required.input &&
         spec.output_alignment >= required.output &&
         spec.secondary_alignment >=
             (reads_secondary(spec.semantic) ? required.secondary : 0) &&
         spec.bias_alignment >= (spec.has_bias != 0 ? required.bias : 0);
}

void initialize(Execution& execution, const Implementation& implementation,
                int configuration, const Spec& spec,
                const apxinf_pointwise_policy_t& policy,
                const apxinf_pointwise_bindings_t& bindings, int device) {
  execution.spec = spec;
  execution.bindings = bindings;
  execution.configuration = configuration;
  execution.device = device;
  execution.implementation = &implementation;
  (void)policy;
}

}  // namespace apxinf::pointwise

extern "C" apxinf_status_t apxinf_pointwise_launch(
    apxinf_runtime_t runtime, const apxinf_pointwise_spec_t* spec,
    const apxinf_pointwise_policy_t* policy,
    const apxinf_pointwise_bindings_t* bindings) {
  return apxinf::pointwise::abi_boundary([&] {
    if (runtime == nullptr || spec == nullptr || policy == nullptr ||
        bindings == nullptr) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT, "null Pointwise argument");
    }
    const apxinf::pointwise::Spec normalized{*spec};
    validate_spec(normalized);
    validate_bindings(normalized, *bindings);

    apxinf::pointwise::Execution execution;
    bool selected = false;
    for (const auto& implementation :
         apxinf::pointwise::registry(normalized.semantic)) {
      if (!implementation.supports(normalized) ||
          !apxinf::pointwise::supports_device(implementation,
                                              runtime->device) ||
          !apxinf::pointwise::supports_alignment(implementation, normalized)) {
        continue;
      }
      if (policy->graph_safe && !implementation.graph_safe) continue;
      if (policy->deterministic && !implementation.deterministic) continue;
      if (!policy->allow_fallback && implementation.fallback) continue;
      std::vector<int> configurations;
      implementation.enumerate_configs(normalized, configurations);
      if (configurations.empty()) continue;
      apxinf::pointwise::initialize(execution, implementation,
                                    configurations.front(), normalized,
                                    *policy, *bindings, runtime->device);
      selected = true;
      break;
    }
    if (!selected) {
      throw Failure(APXINF_STATUS_UNSUPPORTED,
                    "no Pointwise candidate supports this Spec");
    }
    apxinf::pointwise::check_cuda(cudaSetDevice(execution.device));
    apxinf::pointwise::check_cuda(
        execution.implementation->enqueue(execution));
  });
}

extern "C" apxinf_status_t apxinf_pointwise_test_validate_candidates(
    apxinf_runtime_t runtime, const apxinf_pointwise_spec_t* spec,
    const apxinf_pointwise_policy_t* policy,
    const apxinf_pointwise_bindings_t* bindings, const float* expected_output,
    uint64_t expected_output_len) {
  return apxinf::pointwise::abi_boundary([&] {
    if (runtime == nullptr || spec == nullptr || policy == nullptr ||
        bindings == nullptr || expected_output == nullptr) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT, "null Pointwise argument");
    }
    const apxinf::pointwise::Spec normalized{*spec};
    validate_spec(normalized);
    validate_bindings(normalized, *bindings);

    const size_t count = static_cast<size_t>(normalized.rows) *
                         static_cast<size_t>(normalized.cols);
    if (expected_output_len != count) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                    "unexpected Pointwise reference length");
    }
    const size_t bytes =
        count * apxinf::pointwise::dtype_bytes(normalized.dtype);
    std::vector<unsigned char> host(bytes);
    auto stream = static_cast<cudaStream_t>(bindings->stream);

    for (const auto& implementation :
         apxinf::pointwise::registry(normalized.semantic)) {
      if (!implementation.supports(normalized) ||
          !apxinf::pointwise::supports_device(implementation,
                                              runtime->device) ||
          !apxinf::pointwise::supports_alignment(implementation, normalized)) {
        continue;
      }
      if (policy->graph_safe && !implementation.graph_safe) continue;
      if (policy->deterministic && !implementation.deterministic) continue;
      if (!policy->allow_fallback && implementation.fallback) continue;
      std::vector<int> configurations;
      implementation.enumerate_configs(normalized, configurations);
      for (const int configuration : configurations) {
        apxinf::pointwise::check_cuda(
            cudaMemsetAsync(bindings->output, 0xff, bytes, stream));
        apxinf::pointwise::Execution execution;
        apxinf::pointwise::initialize(execution, implementation, configuration,
                                      normalized, *policy, *bindings,
                                      runtime->device);
        apxinf::pointwise::check_cuda(implementation.enqueue(execution));
        apxinf::pointwise::check_cuda(cudaStreamSynchronize(stream));
        apxinf::pointwise::check_cuda(cudaMemcpy(
            host.data(), bindings->output, bytes, cudaMemcpyDeviceToHost));
        for (size_t index = 0; index < count; ++index) {
          const float actual =
              read_element(host.data(), normalized.dtype, index);
          const float reference = expected_output[index];
          const float tolerance = 0.03f + 0.02f * std::fabs(reference);
          if (!std::isfinite(actual) ||
              std::fabs(actual - reference) > tolerance) {
            std::ostringstream message;
            message << implementation.name << " config=" << configuration
                    << " output[" << index << "] = " << actual << ", expected "
                    << reference;
            throw Failure(APXINF_STATUS_INTERNAL_ERROR, message.str());
          }
        }
      }
    }
  });
}
