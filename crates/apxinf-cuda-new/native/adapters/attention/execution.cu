#include "internal.h"

#include <cuda_fp16.h>

#include <cmath>
#include <cstring>

namespace {

using apxinf::attention::Execution;
using apxinf::attention::Failure;
using apxinf::attention::Recipe;
using apxinf::attention::Spec;

bool valid_alignment(uint32_t alignment) {
  return alignment <= 256 && alignment != 0 &&
         (alignment & (alignment - 1)) == 0;
}

void validate_spec(const Spec& spec) {
  if (spec.version != 2 ||
      spec.semantic > APXINF_ATTENTION_SEMANTIC_SEGMENTED ||
      (spec.dtype != APXINF_DTYPE_F16 && spec.dtype != APXINF_DTYPE_BF16) ||
      spec.output_dtype != spec.dtype ||
      spec.mask > APXINF_ATTENTION_MASK_CAUSAL || spec.batch <= 0 ||
      spec.query_tokens <= 0 || spec.key_tokens <= 0 ||
      spec.key_capacity < spec.key_tokens ||
      spec.query_heads <= 0 || spec.kv_heads <= 0 || spec.head_dim <= 0 ||
      spec.query_heads % spec.kv_heads != 0 ||
      spec.batch > INT32_MAX || spec.query_tokens > INT32_MAX ||
      spec.key_tokens > INT32_MAX || spec.key_capacity > INT32_MAX ||
      spec.query_heads > INT32_MAX || spec.kv_heads > INT32_MAX ||
      spec.head_dim > INT32_MAX || spec.query_start > INT32_MAX ||
      spec.segments > INT32_MAX || spec.max_segment_tokens > INT32_MAX ||
      spec.scale_is_default > 1) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT, "invalid Attention Spec");
  }
  if (spec.semantic == APXINF_ATTENTION_SEMANTIC_DENSE &&
      (spec.key_capacity != spec.key_tokens ||
       spec.query_start !=
           (spec.mask == APXINF_ATTENTION_MASK_CAUSAL
                ? spec.key_tokens - spec.query_tokens
                : 0) ||
       spec.segments != 0 || spec.max_segment_tokens != 0 ||
       spec.offsets_hash != 0 || spec.offsets_alignment != 0)) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                  "invalid dense Attention semantic fields");
  }
  if (spec.semantic == APXINF_ATTENTION_SEMANTIC_KV_CACHE &&
      (spec.query_start < 0 || spec.query_start > spec.key_tokens ||
       (spec.mask == APXINF_ATTENTION_MASK_NONE && spec.query_start != 0) ||
       spec.segments != 0 || spec.max_segment_tokens != 0 ||
       spec.offsets_hash != 0 || spec.offsets_alignment != 0)) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                  "invalid KV-cache Attention semantic fields");
  }
  if (spec.semantic == APXINF_ATTENTION_SEMANTIC_SEGMENTED &&
      (spec.batch != 1 || spec.query_tokens != spec.key_tokens ||
       spec.key_capacity != spec.key_tokens || spec.query_heads != spec.kv_heads ||
       spec.mask != APXINF_ATTENTION_MASK_NONE || spec.query_start != 0 ||
       spec.segments <= 0 || spec.max_segment_tokens <= 0 ||
       spec.max_segment_tokens > spec.query_tokens)) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                  "invalid segmented Attention semantic fields");
  }
  if (spec.mask == APXINF_ATTENTION_MASK_CAUSAL &&
      (spec.query_start < 0 ||
       spec.query_start + spec.query_tokens > spec.key_tokens)) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                  "causal Attention query positions exceed valid keys");
  }
  for (uint32_t alignment : {spec.q_alignment, spec.k_alignment,
                             spec.v_alignment, spec.output_alignment}) {
    if (!valid_alignment(alignment) ||
        alignment < apxinf::attention::dtype_bytes(spec.dtype)) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                    "invalid Attention binding alignment");
    }
  }
  if (spec.semantic == APXINF_ATTENTION_SEMANTIC_SEGMENTED &&
      (!valid_alignment(spec.offsets_alignment) || spec.offsets_alignment < 4)) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                  "invalid segmented Attention offsets alignment");
  }
}

void validate_policy(const apxinf_attention_policy_t& policy) {
  if (policy.online_tune > 1 || policy.allow_fallback > 1 ||
      policy.graph_safe > 1 || policy.deterministic > 1) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT, "invalid Attention Policy");
  }
}

void validate_bindings(const Spec& spec,
                       const apxinf_attention_bindings_t& bindings) {
  if (bindings.query == nullptr || bindings.key == nullptr ||
      bindings.value == nullptr || bindings.output == nullptr ||
      !std::isfinite(bindings.scale) || bindings.scale <= 0.0F ||
      ((bindings.scale == 1.0F / std::sqrt(static_cast<float>(spec.head_dim))) !=
       (spec.scale_is_default != 0)) ||
      ((spec.semantic == APXINF_ATTENTION_SEMANTIC_SEGMENTED) !=
       (bindings.offsets != nullptr))) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                  "invalid Attention bindings");
  }
  if (bindings.offsets != nullptr &&
      reinterpret_cast<uintptr_t>(bindings.offsets) % spec.offsets_alignment != 0) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                  "Attention offsets pointer contradicts recorded alignment");
  }
  for (const auto [pointer, alignment] : {
           std::pair{bindings.query, spec.q_alignment},
           std::pair{bindings.key, spec.k_alignment},
           std::pair{bindings.value, spec.v_alignment},
           std::pair{static_cast<const void*>(bindings.output),
                     spec.output_alignment}}) {
    if (reinterpret_cast<uintptr_t>(pointer) % alignment != 0) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                    "Attention pointer contradicts recorded alignment");
    }
  }
}

const apxinf::attention::Implementation* find_implementation(
    const Recipe& recipe, const Spec& spec, int device) {
  const auto* implementation = apxinf::attention::registry(spec.semantic).find(
      recipe.provider_id, recipe.implementation_id,
      recipe.implementation_version);
  if (implementation == nullptr ||
      !apxinf::attention::supports_device(*implementation, device) ||
      !implementation->supports(spec)) {
    return nullptr;
  }
  std::vector<int> configurations;
  implementation->enumerate_configs(spec, configurations);
  return std::find(configurations.begin(), configurations.end(),
                   recipe.configuration) != configurations.end()
             ? implementation
             : nullptr;
}

std::unique_ptr<Execution> fallback(
    const Spec& spec, const apxinf_attention_policy_t& policy,
    const apxinf_attention_bindings_t& bindings, int device) {
  for (const auto& implementation :
       apxinf::attention::registry(spec.semantic)) {
    if (!implementation.fallback ||
        !apxinf::attention::supports_device(implementation, device) ||
        !implementation.supports(spec) ||
        (policy.graph_safe && !implementation.graph_safe) ||
        (policy.deterministic && !implementation.deterministic)) {
      continue;
    }
    std::vector<int> configurations;
    implementation.enumerate_configs(spec, configurations);
    for (int configuration : configurations) {
      try {
        return apxinf::attention::prepare(
            implementation, configuration, spec, policy, bindings, device);
      } catch (const Failure&) {
        cudaGetLastError();
      }
    }
  }
  throw Failure(APXINF_STATUS_UNSUPPORTED,
                "no Attention fallback satisfies the Spec and Policy");
}

std::string serialize(const Recipe& recipe) {
  std::ostringstream output;
  output << recipe.provider_id << ' ' << recipe.implementation_id << ' '
         << recipe.implementation_version << ' ' << recipe.configuration;
  return output.str();
}

bool parse(const std::string& value, Recipe& recipe) {
  std::istringstream input(value);
  return static_cast<bool>(input >> recipe.provider_id >>
                           recipe.implementation_id >>
                           recipe.implementation_version >>
                           recipe.configuration);
}

std::vector<float> read_output(const Spec& spec,
                               const apxinf_attention_bindings_t& bindings,
                               size_t count) {
  std::vector<uint16_t> storage(count);
  const auto stream = static_cast<cudaStream_t>(bindings.stream);
  apxinf::attention::check_cuda(cudaMemcpyAsync(
      storage.data(), bindings.output, count * sizeof(uint16_t),
      cudaMemcpyDeviceToHost, stream));
  apxinf::attention::check_cuda(cudaStreamSynchronize(stream));
  std::vector<float> values;
  values.reserve(count);
  for (uint16_t bits : storage) {
    if (spec.dtype == APXINF_DTYPE_BF16) {
      const uint32_t word = static_cast<uint32_t>(bits) << 16;
      float value = 0.0F;
      std::memcpy(&value, &word, sizeof(value));
      values.push_back(value);
    } else {
      __half value{};
      std::memcpy(&value, &bits, sizeof(value));
      values.push_back(__half2float(value));
    }
  }
  return values;
}

void check_output(const std::vector<float>& actual, const float* expected,
                  const std::string& label) {
  for (size_t index = 0; index < actual.size(); ++index) {
    const float tolerance = 0.03F + 0.02F * std::abs(expected[index]);
    if (!std::isfinite(actual[index]) ||
        std::abs(actual[index] - expected[index]) > tolerance) {
      throw Failure(
          APXINF_STATUS_PROVIDER_ERROR,
          label + " failed precision at element " + std::to_string(index) +
              ": actual=" + std::to_string(actual[index]) +
              " expected=" + std::to_string(expected[index]));
    }
  }
}

}  // namespace

extern "C" apxinf_status_t apxinf_attention_prepare(
    apxinf_runtime_t runtime, const apxinf_attention_spec_t* spec,
    const apxinf_attention_policy_t* policy,
    const apxinf_attention_bindings_t* bindings,
    apxinf_attention_execution_t* output) {
  if (output != nullptr) *output = nullptr;
  return apxinf::attention::abi_boundary([&] {
    if (runtime == nullptr || spec == nullptr || policy == nullptr ||
        bindings == nullptr || output == nullptr) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                    "null Attention execution argument");
    }
    Spec normalized{};
    static_cast<apxinf_attention_spec_t&>(normalized) = *spec;
    validate_spec(normalized);
    validate_policy(*policy);
    validate_bindings(normalized, *bindings);
    cudaStreamCaptureStatus capture_status;
    apxinf::attention::check_cuda(cudaStreamIsCapturing(
        static_cast<cudaStream_t>(bindings->stream), &capture_status));
    if (capture_status != cudaStreamCaptureStatusNone) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                    "prepare Attention executions before graph capture");
    }
    apxinf::attention::check_cuda(cudaSetDevice(runtime->device));
    const auto keys =
        apxinf::attention::tuning_keys(normalized, *policy, runtime->device);
    std::lock_guard<std::mutex> lock(runtime->attention_mutex);

    Recipe recipe{};
    bool recipe_found = false;
    Recipe compatible_hint{};
    bool compatible_hint_found = false;
    std::string source = "recipe";
    if (const auto cached = runtime->attention_recipes.find(keys.performance);
        cached != runtime->attention_recipes.end()) {
      recipe = cached->second;
      recipe_found = true;
      source = "memory-recipe";
    } else {
      recipe_found = parse(
          apxinf::framework::read_recipe(
              policy->cache_dir != nullptr ? policy->cache_dir : "",
              keys.performance),
          recipe);
    }
    if (!recipe_found) {
      compatible_hint_found = parse(
          apxinf::framework::read_recipe(
              policy->cache_dir != nullptr ? policy->cache_dir : "",
              keys.compatible_hint),
          compatible_hint);
      if (compatible_hint_found &&
          find_implementation(compatible_hint, normalized, runtime->device) ==
              nullptr) {
        compatible_hint_found = false;
      }
    }

    std::unique_ptr<Execution> execution;
    if (recipe_found) {
      if (const auto* implementation =
              find_implementation(recipe, normalized, runtime->device)) {
        try {
          execution = apxinf::attention::prepare(
              *implementation, recipe.configuration, normalized, *policy,
              *bindings, runtime->device);
        } catch (const Failure&) {
          execution.reset();
          cudaGetLastError();
        }
        if (execution != nullptr) {
          runtime->attention_recipes[keys.performance] = recipe;
        }
      }
      if (execution == nullptr) {
        compatible_hint = recipe;
        compatible_hint_found = true;
      }
    }

    bool persist = false;
    if (execution == nullptr && policy->online_tune) {
      recipe = apxinf::attention::tune(normalized, *policy, *bindings,
                                       runtime->device, source,
                                       compatible_hint_found ? &compatible_hint
                                                             : nullptr);
      const auto* implementation =
          find_implementation(recipe, normalized, runtime->device);
      if (implementation == nullptr) {
        throw Failure(APXINF_STATUS_INTERNAL_ERROR,
                      "tuned Attention Recipe is not restorable");
      }
      try {
        execution = apxinf::attention::prepare(
            *implementation, recipe.configuration, normalized, *policy,
            *bindings, runtime->device);
        persist = true;
      } catch (const Failure&) {
        execution.reset();
        cudaGetLastError();
        if (!policy->allow_fallback) throw;
        execution = fallback(normalized, *policy, *bindings, runtime->device);
        recipe = {execution->implementation->provider_id,
                  execution->implementation->implementation_id,
                  execution->implementation->implementation_version,
                  execution->configuration};
        source = "fallback-after-tune";
      }
    }
    if (execution == nullptr && policy->allow_fallback) {
      execution = fallback(normalized, *policy, *bindings, runtime->device);
      recipe = {execution->implementation->provider_id,
                execution->implementation->implementation_id,
                execution->implementation->implementation_version,
                execution->configuration};
      source = "fallback";
    }
    if (execution == nullptr) {
      throw Failure(APXINF_STATUS_CACHE_MISS,
                    "Attention Recipe miss and fallback is disabled");
    }

    if (persist) {
      runtime->attention_recipes[keys.performance] = recipe;
      const std::string encoded = serialize(recipe);
      const std::string directory =
          policy->cache_dir != nullptr ? policy->cache_dir : "";
      apxinf::framework::write_recipe(directory, keys.performance, encoded);
      apxinf::framework::write_recipe(directory, keys.compatible_hint,
                                      encoded);
    }
    execution->summary =
        std::string(execution->implementation->name) +
        " config=" + std::to_string(recipe.configuration) +
        " workspace=" + std::to_string(execution->resource_bytes) +
        " source=" + source;
    auto wrapper = std::make_unique<apxinf_attention_execution>();
    wrapper->state = std::move(execution);
    *output = wrapper.release();
  });
}

extern "C" apxinf_status_t apxinf_attention_enqueue(
    apxinf_attention_execution_t execution) {
  return apxinf::attention::abi_boundary([&] {
    if (execution == nullptr || execution->state == nullptr) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                    "null Attention execution");
    }
    apxinf::attention::check_cuda(cudaSetDevice(execution->state->device));
    apxinf::attention::check_cuda(
        execution->state->implementation->enqueue(*execution->state));
  });
}

extern "C" void apxinf_attention_destroy(
    apxinf_attention_execution_t execution) {
  delete execution;
}

extern "C" const char* apxinf_attention_summary(
    apxinf_attention_execution_t execution) {
  return execution == nullptr || execution->state == nullptr
             ? ""
             : execution->state->summary.c_str();
}

extern "C" apxinf_status_t apxinf_attention_test_validate_candidates(
    apxinf_runtime_t runtime, const apxinf_attention_spec_t* spec,
    const apxinf_attention_policy_t* policy,
    const apxinf_attention_bindings_t* bindings,
    const float* expected_output, uint64_t expected_output_len) {
  return apxinf::attention::abi_boundary([&] {
    if (runtime == nullptr || spec == nullptr || policy == nullptr ||
        bindings == nullptr || expected_output == nullptr) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                    "invalid Attention candidate validation arguments");
    }
    Spec normalized{};
    static_cast<apxinf_attention_spec_t&>(normalized) = *spec;
    validate_spec(normalized);
    validate_policy(*policy);
    validate_bindings(normalized, *bindings);
    const uint64_t count = static_cast<uint64_t>(normalized.batch) *
                           normalized.query_tokens * normalized.query_heads *
                           normalized.head_dim;
    if (count != expected_output_len || count > SIZE_MAX / sizeof(uint16_t)) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                    "invalid Attention candidate validation output");
    }
    apxinf::attention::check_cuda(cudaSetDevice(runtime->device));
    size_t implementations_checked = 0;
    for (const auto& implementation :
         apxinf::attention::registry(normalized.semantic)) {
      if (!apxinf::attention::supports_device(implementation, runtime->device) ||
          !implementation.supports(normalized) ||
          !apxinf::attention::supports_alignment(implementation, normalized) ||
          (policy->graph_safe && !implementation.graph_safe) ||
          (policy->deterministic && !implementation.deterministic)) {
        continue;
      }
      std::vector<int> configurations;
      implementation.enumerate_configs(normalized, configurations);
      if (configurations.empty()) {
        throw Failure(APXINF_STATUS_PROVIDER_ERROR,
                      std::string(implementation.name) +
                          " has no Attention configuration");
      }
      for (int configuration : configurations) {
        auto candidate = apxinf::attention::prepare(
            implementation, configuration, normalized, *policy, *bindings,
            runtime->device);
        const size_t bytes = static_cast<size_t>(count) * sizeof(uint16_t);
        const auto stream = static_cast<cudaStream_t>(bindings->stream);
        apxinf::attention::check_cuda(
            cudaMemsetAsync(bindings->output, 0xff, bytes, stream));
        apxinf::attention::check_cuda(implementation.enqueue(*candidate));
        check_output(read_output(normalized, *bindings, count), expected_output,
                     std::string(implementation.name) + "#" +
                         std::to_string(configuration));
      }
      ++implementations_checked;
    }
    if (implementations_checked == 0) {
      throw Failure(APXINF_STATUS_UNSUPPORTED,
                    "no registered Attention candidate applies to the validation Spec");
    }
  });
}
