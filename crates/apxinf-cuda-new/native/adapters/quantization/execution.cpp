#include "internal.h"

#include <cmath>
#include <climits>
#include <vector>

namespace {

using apxinf::quantization::Execution;
using apxinf::quantization::Failure;
using apxinf::quantization::Spec;

bool valid_alignment(uint32_t alignment) {
  return alignment <= 256 && alignment != 0 &&
         (alignment & (alignment - 1)) == 0;
}

uint32_t dtype_bytes(uint32_t dtype) {
  switch (dtype) {
    case APXINF_DTYPE_F32:
      return 4;
    case APXINF_DTYPE_F16:
    case APXINF_DTYPE_BF16:
      return 2;
    case APXINF_DTYPE_E4M3:
    case APXINF_DTYPE_I8:
      return 1;
    default:
      return 0;
  }
}

void validate_recorded_alignment(const void* pointer, uint32_t alignment,
                                 const char* name) {
  if (pointer != nullptr &&
      reinterpret_cast<uintptr_t>(pointer) % alignment != 0) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                  std::string(name) + " violates its alignment class");
  }
}

void validate_spec(const Spec& spec) {
  if (spec.version != APXINF_QUANTIZATION_SPEC_VERSION ||
      spec.semantic > APXINF_QUANTIZATION_SEMANTIC_ROWWISE_I8 ||
      spec.scale_dtype != APXINF_DTYPE_F32 || spec.rows <= 0 ||
      spec.input_cols <= 0 || spec.output_cols <= 0 ||
      spec.rows > INT32_MAX || spec.input_cols > INT32_MAX ||
      spec.output_cols > INT32_MAX ||
      !valid_alignment(spec.input_alignment) ||
      !valid_alignment(spec.output_alignment) ||
      !valid_alignment(spec.scales_alignment)) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                  "invalid Quantization Spec");
  }
  if (spec.input_alignment < dtype_bytes(spec.input_dtype) ||
      spec.output_alignment < dtype_bytes(spec.output_dtype) ||
      (apxinf::quantization::has_row_scales(spec.semantic) &&
       spec.scales_alignment < sizeof(float))) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                  "Quantization binding violates dtype alignment");
  }

  switch (spec.semantic) {
    case APXINF_QUANTIZATION_SEMANTIC_FIXED_E4M3:
      if ((spec.input_dtype != APXINF_DTYPE_F16 &&
           spec.input_dtype != APXINF_DTYPE_BF16) ||
          spec.output_dtype != APXINF_DTYPE_E4M3 ||
          spec.input_cols != spec.output_cols) {
        throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                      "invalid fixed-scale E4M3 Spec");
      }
      break;
    case APXINF_QUANTIZATION_SEMANTIC_ROWWISE_E4M3:
      if (spec.input_dtype != APXINF_DTYPE_BF16 ||
          spec.output_dtype != APXINF_DTYPE_E4M3 ||
          spec.output_cols < spec.input_cols) {
        throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                      "invalid rowwise E4M3 Spec");
      }
      break;
    case APXINF_QUANTIZATION_SEMANTIC_CAST_F16_BF16:
      if (spec.input_dtype != APXINF_DTYPE_F16 ||
          spec.output_dtype != APXINF_DTYPE_BF16 ||
          spec.input_cols != spec.output_cols) {
        throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                      "invalid F16-to-BF16 cast Spec");
      }
      break;
    case APXINF_QUANTIZATION_SEMANTIC_SLICE_BF16:
      if (spec.input_dtype != APXINF_DTYPE_BF16 ||
          spec.output_dtype != APXINF_DTYPE_BF16 ||
          spec.output_cols > spec.input_cols) {
        throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                      "invalid BF16 slice Spec");
      }
      break;
    case APXINF_QUANTIZATION_SEMANTIC_ROWWISE_I8:
      if (spec.input_dtype != APXINF_DTYPE_BF16 ||
          spec.output_dtype != APXINF_DTYPE_I8 ||
          spec.input_cols != spec.output_cols) {
        throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                      "invalid rowwise INT8 Spec");
      }
      break;
    default:
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                    "unknown Quantization semantic");
  }
}

void validate_policy(const apxinf_quantization_policy_t& policy) {
  if (policy.online_tune > 1 || policy.allow_fallback > 1 ||
      policy.graph_safe > 1 || policy.deterministic > 1) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                  "invalid Quantization Policy");
  }
}

void validate_bindings(const Spec& spec,
                       const apxinf_quantization_bindings_t& bindings) {
  if (bindings.input == nullptr || bindings.output == nullptr) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                  "missing Quantization binding");
  }
  const bool needs_scales = apxinf::quantization::has_row_scales(spec.semantic);
  if (needs_scales != (bindings.scales != nullptr)) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                  "Quantization scales binding disagrees with semantic");
  }
  if (spec.semantic == APXINF_QUANTIZATION_SEMANTIC_FIXED_E4M3) {
    if (!(bindings.scale > 0.0f) || !std::isfinite(bindings.scale)) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                    "fixed E4M3 scale must be finite and positive");
    }
  } else if (bindings.scale != 1.0f) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                  "unused Quantization scale must be one");
  }
  validate_recorded_alignment(bindings.input, spec.input_alignment,
                              "Quantization input");
  validate_recorded_alignment(bindings.output, spec.output_alignment,
                              "Quantization output");
  validate_recorded_alignment(bindings.scales, spec.scales_alignment,
                              "Quantization scales");
}

}  // namespace

namespace apxinf::quantization {

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
         spec.scales_alignment >=
             (has_row_scales(spec.semantic) ? required.scales : 0);
}

void initialize(Execution& execution, const Implementation& implementation,
                int configuration, const Spec& spec,
                const apxinf_quantization_policy_t& policy,
                const apxinf_quantization_bindings_t& bindings, int device) {
  execution.spec = spec;
  execution.bindings = bindings;
  execution.configuration = configuration;
  execution.device = device;
  execution.implementation = &implementation;
  (void)policy;
}

}  // namespace apxinf::quantization

extern "C" apxinf_status_t apxinf_quantization_launch(
    apxinf_runtime_t runtime, const apxinf_quantization_spec_t* spec,
    const apxinf_quantization_policy_t* policy,
    const apxinf_quantization_bindings_t* bindings) {
  return apxinf::quantization::abi_boundary([&] {
    if (runtime == nullptr || spec == nullptr || policy == nullptr ||
        bindings == nullptr) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                    "null Quantization argument");
    }
    const apxinf::quantization::Spec normalized{*spec};
    validate_spec(normalized);
    validate_policy(*policy);
    validate_bindings(normalized, *bindings);

    apxinf::quantization::Execution execution;
    bool selected = false;
    for (const auto& implementation :
         apxinf::quantization::registry(normalized.semantic)) {
      if (!implementation.supports(normalized) ||
          !apxinf::quantization::supports_device(implementation,
                                                 runtime->device) ||
          !apxinf::quantization::supports_alignment(implementation,
                                                    normalized)) {
        continue;
      }
      if (policy->graph_safe && !implementation.graph_safe) continue;
      if (policy->deterministic && !implementation.deterministic) continue;
      if (!policy->allow_fallback && implementation.fallback) continue;
      std::vector<int> configurations;
      implementation.enumerate_configs(normalized, configurations);
      if (configurations.empty()) continue;
      apxinf::quantization::initialize(execution, implementation,
                                       configurations.front(), normalized,
                                       *policy, *bindings, runtime->device);
      selected = true;
      break;
    }
    if (!selected) {
      throw Failure(APXINF_STATUS_UNSUPPORTED,
                    "no Quantization candidate supports this Spec");
    }
    apxinf::quantization::check_cuda(cudaSetDevice(execution.device));
    apxinf::quantization::check_cuda(
        execution.implementation->enqueue(execution));
  });
}
