#include "internal.h"

#include <vector>

namespace {

using apxinf::gather::Execution;
using apxinf::gather::Failure;
using apxinf::gather::Spec;

bool valid_alignment(uint32_t alignment) {
  return alignment <= 256 && alignment != 0 &&
         (alignment & (alignment - 1)) == 0;
}

void validate_spec(const Spec& spec) {
  if (spec.version != APXINF_GATHER_SPEC_VERSION ||
      spec.semantic > APXINF_GATHER_SEMANTIC_RGB_TO_PATCHES ||
      (spec.dtype != APXINF_DTYPE_F16 && spec.dtype != APXINF_DTYPE_BF16) ||
      spec.has_bias > 1 || spec.nhwc > 1 || spec.rows <= 0 || spec.cols <= 0 ||
      spec.rows > INT32_MAX || spec.cols > INT32_MAX ||
      spec.vocab_size > INT32_MAX || spec.tokens_per_view > INT32_MAX ||
      spec.views > INT32_MAX || spec.image_size > INT32_MAX ||
      spec.patch_size > INT32_MAX ||
      !valid_alignment(spec.input_alignment) ||
      !valid_alignment(spec.bias_alignment) ||
      !valid_alignment(spec.output_alignment)) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT, "invalid Gather Spec");
  }
  if (spec.has_bias != 0 && !apxinf::gather::may_have_bias(spec.semantic)) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                  "Gather semantic does not take a bias");
  }
  if (spec.semantic == APXINF_GATHER_SEMANTIC_EMBEDDING_LOOKUP &&
      spec.vocab_size == 0) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                  "embedding lookup requires a vocabulary size");
  }
  if (spec.semantic == APXINF_GATHER_SEMANTIC_BIAS_POSITION &&
      (spec.tokens_per_view == 0 ||
       spec.rows % static_cast<int64_t>(spec.tokens_per_view) != 0)) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                  "bias-position rows must be a whole number of views");
  }
  if (spec.semantic == APXINF_GATHER_SEMANTIC_RGB_TO_PATCHES) {
    if (spec.views == 0 || spec.image_size == 0 || spec.patch_size == 0 ||
        spec.image_size % spec.patch_size != 0) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                    "invalid RGB-to-patches geometry");
    }
    // The Spec carries the output shape twice, once as rows/cols and once as
    // the image geometry; disagreeing values would silently truncate.
    const int64_t per_side = spec.image_size / spec.patch_size;
    const int64_t expected_rows =
        static_cast<int64_t>(spec.views) * per_side * per_side;
    const int64_t expected_cols =
        3 * static_cast<int64_t>(spec.patch_size) * spec.patch_size;
    if (spec.rows != expected_rows || spec.cols != expected_cols) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                    "RGB-to-patches shape disagrees with the geometry");
    }
  }
}

void validate_bindings(const Spec& spec,
                       const apxinf_gather_bindings_t& bindings) {
  if (bindings.input == nullptr || bindings.output == nullptr) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT, "missing Gather binding");
  }
  if (spec.semantic == APXINF_GATHER_SEMANTIC_EMBEDDING_LOOKUP &&
      bindings.ids == nullptr) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                  "embedding lookup requires token ids");
  }
  if (spec.semantic == APXINF_GATHER_SEMANTIC_BIAS_POSITION &&
      bindings.position == nullptr) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                  "bias-position requires a position embedding");
  }
  if ((spec.has_bias != 0) != (bindings.bias != nullptr)) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                  "Gather bias binding disagrees with Spec.has_bias");
  }
}

}  // namespace

namespace apxinf::gather {

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
  return spec.input_alignment >= required.input &&
         spec.output_alignment >= required.output &&
         spec.bias_alignment >= (spec.has_bias != 0 ? required.bias : 0);
}

std::unique_ptr<Execution> prepare(const Implementation& implementation,
                                   int configuration, const Spec& spec,
                                   const apxinf_gather_policy_t& policy,
                                   const apxinf_gather_bindings_t& bindings,
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
                  "Gather candidate exceeds the workspace limit");
  }
  return execution;
}

}  // namespace apxinf::gather

extern "C" apxinf_status_t apxinf_gather_prepare(
    apxinf_runtime_t runtime, const apxinf_gather_spec_t* spec,
    const apxinf_gather_policy_t* policy,
    const apxinf_gather_bindings_t* bindings,
    apxinf_gather_execution_t* output) {
  return apxinf::gather::abi_boundary([&] {
    if (runtime == nullptr || spec == nullptr || policy == nullptr ||
        bindings == nullptr || output == nullptr) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT, "null Gather argument");
    }
    const apxinf::gather::Spec normalized{*spec};
    validate_spec(normalized);
    validate_bindings(normalized, *bindings);

    cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing(static_cast<cudaStream_t>(bindings->stream),
                              &capture) == cudaSuccess &&
        capture != cudaStreamCaptureStatusNone) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                    "Gather preparation is not allowed during capture");
    }

    std::unique_ptr<apxinf::gather::Execution> execution;
    for (const auto& implementation :
         apxinf::gather::registry(normalized.semantic)) {
      if (!implementation.supports(normalized) ||
          !apxinf::gather::supports_device(implementation, runtime->device) ||
          !apxinf::gather::supports_alignment(implementation, normalized)) {
        continue;
      }
      if (policy->graph_safe && !implementation.graph_safe) continue;
      if (policy->deterministic && !implementation.deterministic) continue;
      if (!policy->allow_fallback && implementation.fallback) continue;
      std::vector<int> configurations;
      implementation.enumerate_configs(normalized, configurations);
      if (configurations.empty()) continue;
      execution = apxinf::gather::prepare(
          implementation, configurations.front(), normalized, *policy,
          *bindings, runtime->device);
      break;
    }
    if (execution == nullptr) {
      throw Failure(APXINF_STATUS_UNSUPPORTED,
                    "no Gather candidate supports this Spec");
    }
    execution->summary = std::string(execution->implementation->name) +
                         " config=" +
                         std::to_string(execution->configuration) +
                         " source=direct";
    auto wrapper = std::make_unique<apxinf_gather_execution>();
    wrapper->state = std::move(execution);
    *output = wrapper.release();
  });
}

extern "C" apxinf_status_t apxinf_gather_enqueue(
    apxinf_gather_execution_t execution) {
  return apxinf::gather::abi_boundary([&] {
    if (execution == nullptr || execution->state == nullptr) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT, "null Gather execution");
    }
    apxinf::gather::check_cuda(cudaSetDevice(execution->state->device));
    apxinf::gather::check_cuda(
        execution->state->implementation->enqueue(*execution->state));
  });
}

extern "C" void apxinf_gather_destroy(apxinf_gather_execution_t execution) {
  delete execution;
}

extern "C" const char* apxinf_gather_summary(
    apxinf_gather_execution_t execution) {
  return execution == nullptr || execution->state == nullptr
             ? ""
             : execution->state->summary.c_str();
}
