#include "internal.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cmath>
#include <cstring>
#include <sstream>
#include <vector>

namespace {

using apxinf::norm::Execution;
using apxinf::norm::Failure;
using apxinf::norm::Spec;

bool valid_alignment(uint32_t alignment) {
  return alignment <= 256 && alignment != 0 &&
         (alignment & (alignment - 1)) == 0;
}

void validate_spec(const Spec& spec) {
  if (spec.version != APXINF_NORM_SPEC_VERSION ||
      spec.semantic > APXINF_NORM_SEMANTIC_BIAS_THEN_RESIDUAL ||
      (spec.dtype != APXINF_DTYPE_F16 && spec.dtype != APXINF_DTYPE_BF16) ||
      (spec.output_dtype != spec.dtype &&
       spec.output_dtype != APXINF_DTYPE_E4M3) ||
      spec.rows <= 0 || spec.cols <= 0 || spec.rows > INT32_MAX ||
      spec.cols > INT32_MAX || spec.has_bias > 1 ||
      spec.output_scale_is_unit > 1 || !valid_alignment(spec.input_alignment) ||
      !valid_alignment(spec.weight_alignment) ||
      !valid_alignment(spec.bias_alignment) ||
      !valid_alignment(spec.residual_alignment) ||
      !valid_alignment(spec.style_alignment) ||
      !valid_alignment(spec.hidden_alignment) ||
      !valid_alignment(spec.normalized_alignment)) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT, "invalid Norm Spec");
  }
  if (spec.semantic == APXINF_NORM_SEMANTIC_BIAS_THEN_RESIDUAL &&
      (spec.dtype != APXINF_DTYPE_BF16 ||
       spec.output_dtype != APXINF_DTYPE_BF16)) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                  "BiasThenResidual requires BF16 input and output");
  }
  // A bias only exists where the semantic combines one; declaring it anywhere
  // else would make two different Specs describe the same kernel.
  if (spec.has_bias != 0 && !apxinf::norm::may_have_bias(spec.semantic)) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                  "Norm semantic does not take a bias");
  }
  // E4M3 output is the fused-quantization form and always carries a real
  // scale; a unit scale would mean the quantization was forgotten.
  if (spec.output_dtype == APXINF_DTYPE_E4M3 &&
      spec.output_scale_is_unit != 0) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                  "E4M3 Norm output requires a non-unit output scale");
  }
  if (spec.output_dtype == spec.dtype && spec.output_scale_is_unit != 1) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                  "unquantized Norm output requires a unit output scale");
  }
}

void validate_bindings(const Spec& spec,
                       const apxinf_norm_bindings_t& bindings) {
  const uint32_t semantic = spec.semantic;
  const bool missing =
      bindings.input == nullptr ||
      (apxinf::norm::writes_hidden(semantic) && bindings.hidden == nullptr) ||
      (apxinf::norm::writes_normalized(semantic) &&
       bindings.normalized == nullptr) ||
      (apxinf::norm::reads_residual(semantic) &&
       bindings.residual == nullptr) ||
      (apxinf::norm::reads_weight(semantic) && bindings.weight == nullptr) ||
      (apxinf::norm::reads_norm_bias(semantic) &&
       bindings.norm_bias == nullptr) ||
      (apxinf::norm::reads_norm_style(semantic) &&
       bindings.norm_style == nullptr) ||
      (apxinf::norm::reads_gate_style(semantic) &&
       bindings.gate_style == nullptr);
  if (missing) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                  "missing Norm binding for semantic");
  }
  // has_bias is part of the Spec identity, so the pointer must agree with it.
  if ((spec.has_bias != 0) != (bindings.bias != nullptr)) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                  "Norm bias binding disagrees with Spec.has_bias");
  }
  if ((spec.output_scale_is_unit != 0) != (bindings.output_scale == 1.0f)) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                  "Norm output_scale disagrees with Spec.output_scale_is_unit");
  }
  if (!(bindings.eps > 0.0f) || !std::isfinite(bindings.eps) ||
      !std::isfinite(bindings.output_scale) || bindings.output_scale <= 0.0f) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT, "invalid Norm scalar");
  }
}

size_t element_count(const Spec& spec) {
  return static_cast<size_t>(spec.rows) * static_cast<size_t>(spec.cols);
}

float read_element(const void* buffer, uint32_t dtype, size_t index) {
  if (dtype == APXINF_DTYPE_BF16) {
    return __bfloat162float(static_cast<const __nv_bfloat16*>(buffer)[index]);
  }
  return __half2float(static_cast<const __half*>(buffer)[index]);
}

}  // namespace

namespace apxinf::norm {

Execution::~Execution() {
  if (implementation != nullptr && implementation->destroy != nullptr) {
    implementation->destroy(*this);
  }
}

bool supports_device(const Implementation& implementation, int device,
                     std::string* reason) {
  if (implementation.required_device_features == 0) return true;
  cudaDeviceProp properties{};
  if (cudaGetDeviceProperties(&properties, device) != cudaSuccess) {
    if (reason != nullptr) *reason = "device properties unavailable";
    return false;
  }
  if (reason != nullptr) *reason = "missing required device feature";
  return false;
}

bool supports_alignment(const Implementation& implementation,
                        const Spec& spec) {
  const auto required = implementation.alignment_requirements(spec);
  return spec.input_alignment >= required.input &&
         spec.hidden_alignment >=
             (writes_hidden(spec.semantic) ? required.hidden : 0) &&
         spec.normalized_alignment >=
             (writes_normalized(spec.semantic) ? required.normalized : 0) &&
         spec.weight_alignment >=
             (reads_weight(spec.semantic) ? required.weight : 0) &&
         spec.bias_alignment >= (spec.has_bias != 0 ? required.bias : 0) &&
         spec.residual_alignment >=
             (reads_residual(spec.semantic) ? required.residual : 0) &&
         spec.style_alignment >= ((reads_norm_style(spec.semantic) ||
                                   reads_gate_style(spec.semantic))
                                      ? required.style
                                      : 0);
}

std::unique_ptr<Execution> prepare(const Implementation& implementation,
                                   int configuration, const Spec& spec,
                                   const apxinf_norm_policy_t& policy,
                                   const apxinf_norm_bindings_t& bindings,
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
                  "Norm candidate exceeds the workspace limit");
  }
  return execution;
}

}  // namespace apxinf::norm

extern "C" apxinf_status_t apxinf_norm_prepare(
    apxinf_runtime_t runtime, const apxinf_norm_spec_t* spec,
    const apxinf_norm_policy_t* policy,
    const apxinf_norm_bindings_t* bindings,
    apxinf_norm_execution_t* output) {
  return apxinf::norm::abi_boundary([&] {
    if (runtime == nullptr || spec == nullptr || policy == nullptr ||
        bindings == nullptr || output == nullptr) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT, "null Norm argument");
    }
    const apxinf::norm::Spec normalized{*spec};
    validate_spec(normalized);
    validate_bindings(normalized, *bindings);

    // Resource creation may not happen while the stream is capturing; the
    // caller must prepare every operator before it opens a graph capture.
    cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(static_cast<cudaStream_t>(bindings->stream),
                              &capture) == cudaSuccess &&
        capture != cudaStreamCaptureStatusNone) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                    "Norm preparation is not allowed during capture");
    }

    // One memory-bound implementation per semantic: resolve it directly
    // instead of timing a single candidate against itself.
    std::unique_ptr<apxinf::norm::Execution> execution;
    for (const auto& implementation :
         apxinf::norm::registry(normalized.semantic)) {
      if (!implementation.supports(normalized) ||
          !apxinf::norm::supports_device(implementation, runtime->device) ||
          !apxinf::norm::supports_alignment(implementation, normalized)) {
        continue;
      }
      if (policy->graph_safe && !implementation.graph_safe) continue;
      if (policy->deterministic && !implementation.deterministic) continue;
      if (!policy->allow_fallback && implementation.fallback) continue;
      std::vector<int> configurations;
      implementation.enumerate_configs(normalized, configurations);
      if (configurations.empty()) continue;
      execution = apxinf::norm::prepare(implementation, configurations.front(),
                                        normalized, *policy, *bindings,
                                        runtime->device);
      break;
    }
    if (execution == nullptr) {
      throw Failure(APXINF_STATUS_UNSUPPORTED,
                    "no Norm candidate supports this Spec");
    }
    execution->summary = std::string(execution->implementation->name) +
                         " config=" +
                         std::to_string(execution->configuration) +
                         " workspace=" +
                         std::to_string(execution->resource_bytes) +
                         " source=direct";
    auto wrapper = std::make_unique<apxinf_norm_execution>();
    wrapper->state = std::move(execution);
    *output = wrapper.release();
  });
}

extern "C" apxinf_status_t apxinf_norm_enqueue(
    apxinf_norm_execution_t execution) {
  return apxinf::norm::abi_boundary([&] {
    if (execution == nullptr || execution->state == nullptr) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT, "null Norm execution");
    }
    apxinf::norm::check_cuda(cudaSetDevice(execution->state->device));
    apxinf::norm::check_cuda(
        execution->state->implementation->enqueue(*execution->state));
  });
}

extern "C" void apxinf_norm_destroy(apxinf_norm_execution_t execution) {
  delete execution;
}

extern "C" const char* apxinf_norm_summary(apxinf_norm_execution_t execution) {
  return execution == nullptr || execution->state == nullptr
             ? ""
             : execution->state->summary.c_str();
}

extern "C" apxinf_status_t apxinf_norm_test_validate_candidates(
    apxinf_runtime_t runtime, const apxinf_norm_spec_t* spec,
    const apxinf_norm_policy_t* policy,
    const apxinf_norm_bindings_t* bindings, const float* expected_output,
    uint64_t expected_output_len) {
  return apxinf::norm::abi_boundary([&] {
    if (runtime == nullptr || spec == nullptr || policy == nullptr ||
        bindings == nullptr || expected_output == nullptr) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT, "null Norm argument");
    }
    const apxinf::norm::Spec normalized{*spec};
    validate_spec(normalized);
    validate_bindings(normalized, *bindings);

    // Expected values are laid out hidden-then-normalized, skipping whichever
    // output the semantic does not write.
    const size_t count = element_count(normalized);
    const bool has_hidden = apxinf::norm::writes_hidden(normalized.semantic);
    const bool has_normalized =
        apxinf::norm::writes_normalized(normalized.semantic);
    const size_t expected_count =
        count * ((has_hidden ? 1u : 0u) + (has_normalized ? 1u : 0u));
    if (expected_output_len != expected_count) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                    "unexpected Norm reference length");
    }

    const size_t bytes = count * apxinf::norm::dtype_bytes(normalized.dtype);
    std::vector<unsigned char> host(bytes);
    auto stream = static_cast<cudaStream_t>(bindings->stream);

    for (const auto& implementation :
         apxinf::norm::registry(normalized.semantic)) {
      if (!implementation.supports(normalized) ||
          !apxinf::norm::supports_device(implementation, runtime->device) ||
          !apxinf::norm::supports_alignment(implementation, normalized)) {
        continue;
      }
      if (policy->graph_safe && !implementation.graph_safe) continue;
      if (policy->deterministic && !implementation.deterministic) continue;
      if (!policy->allow_fallback && implementation.fallback) continue;
      std::vector<int> configurations;
      implementation.enumerate_configs(normalized, configurations);
      for (const int configuration : configurations) {
        // Poison both outputs so a kernel that never writes is caught.
        if (has_hidden) {
          apxinf::norm::check_cuda(
              cudaMemsetAsync(bindings->hidden, 0xff, bytes, stream));
        }
        if (has_normalized) {
          apxinf::norm::check_cuda(
              cudaMemsetAsync(bindings->normalized, 0xff, bytes, stream));
        }
        auto execution =
            apxinf::norm::prepare(implementation, configuration, normalized,
                                  *policy, *bindings, runtime->device);
        apxinf::norm::check_cuda(implementation.enqueue(*execution));
        apxinf::norm::check_cuda(cudaStreamSynchronize(stream));

        size_t offset = 0;
        for (int output = 0; output < 2; ++output) {
          const void* device_buffer =
              output == 0 ? bindings->hidden : bindings->normalized;
          const bool present = output == 0 ? has_hidden : has_normalized;
          if (!present) continue;
          apxinf::norm::check_cuda(cudaMemcpy(host.data(), device_buffer,
                                              bytes, cudaMemcpyDeviceToHost));
          for (size_t index = 0; index < count; ++index) {
            const float actual =
                read_element(host.data(), normalized.dtype, index);
            const float reference = expected_output[offset + index];
            const float tolerance = 0.03f + 0.02f * std::fabs(reference);
            if (!std::isfinite(actual) ||
                std::fabs(actual - reference) > tolerance) {
              std::ostringstream message;
              message << implementation.name << " config=" << configuration
                      << (output == 0 ? " hidden[" : " normalized[") << index
                      << "] = " << actual << ", expected " << reference;
              throw Failure(APXINF_STATUS_INTERNAL_ERROR, message.str());
            }
          }
          offset += count;
        }
      }
    }
  });
}
