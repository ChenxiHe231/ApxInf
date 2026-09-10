#include "internal.h"

#include <cmath>
#include <limits>

namespace {

using apxinf::gemm::Failure;
using apxinf::gemm::Recipe;
using apxinf::gemm::State;
using apxinf::gemm::APXINF_GEMM_SEMANTIC_GEMM;
using apxinf::gemm::APXINF_GEMM_SEMANTIC_GEMM_BIAS;
using apxinf::gemm::APXINF_GEMM_SEMANTIC_GEMM_GEGLU;
using apxinf::gemm::APXINF_GEMM_SEMANTIC_GEMM_BIAS_GELU;

bool valid_alignment_class(uint32_t alignment) {
  return alignment <= 256 &&
         (alignment == 0 || (alignment & (alignment - 1)) == 0);
}

void validate_recorded_alignment(const void* pointer, uint32_t alignment,
                                 const char* name) {
  if (pointer != nullptr &&
      (alignment == 0 ||
       reinterpret_cast<uintptr_t>(pointer) % alignment != 0)) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                  std::string(name) + " does not satisfy plan alignment");
  }
}

void validate_spec(const apxinf::gemm::Spec& spec) {
  if (spec.version != 3 || spec.semantic > APXINF_GEMM_SEMANTIC_GEMM_BIAS ||
      spec.a_dtype > APXINF_DTYPE_I8 || spec.b_dtype > APXINF_DTYPE_I8 ||
      spec.accumulation_dtype > APXINF_DTYPE_I32 ||
      spec.output_dtype > APXINF_DTYPE_E4M3 ||
      spec.quantization > APXINF_GEMM_QUANT_W8A8_ROW_CHANNEL || spec.m <= 0 ||
      spec.n <= 0 || spec.k <= 0 || spec.m > INT32_MAX ||
      spec.n > INT32_MAX || spec.k > INT32_MAX ||
      spec.alpha_is_unit > 1 || spec.output_scale_is_unit > 1) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT, "invalid GEMM Spec");
  }
  for (uint32_t alignment : {
           spec.a_alignment, spec.b_alignment, spec.bias_alignment,
           spec.a_scales_alignment, spec.b_scales_alignment,
           spec.output_alignment}) {
    if (!valid_alignment_class(alignment)) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                    "invalid GEMM binding alignment class");
    }
  }
  if (spec.a_alignment < apxinf::gemm::dtype_bytes(spec.a_dtype) ||
      spec.b_alignment < apxinf::gemm::dtype_bytes(spec.b_dtype) ||
      spec.output_alignment < apxinf::gemm::dtype_bytes(spec.output_dtype)) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                  "GEMM binding violates dtype alignment");
  }
  if ((spec.a_dtype == APXINF_DTYPE_I8 ||
       spec.b_dtype == APXINF_DTYPE_I8) &&
      (spec.a_dtype != APXINF_DTYPE_I8 ||
       spec.b_dtype != APXINF_DTYPE_I8 ||
       spec.accumulation_dtype != APXINF_DTYPE_I32 ||
       spec.quantization != APXINF_GEMM_QUANT_W8A8_ROW_CHANNEL ||
       spec.output_dtype != APXINF_DTYPE_BF16 || spec.k > 131071 ||
       (spec.semantic != APXINF_GEMM_SEMANTIC_GEMM &&
        spec.semantic != APXINF_GEMM_SEMANTIC_GEMM_BIAS))) {
    throw Failure(APXINF_STATUS_UNSUPPORTED, "invalid INT8 GEMM contract");
  }
  if (spec.a_dtype != APXINF_DTYPE_I8 &&
      spec.accumulation_dtype != APXINF_DTYPE_F32) {
    throw Failure(APXINF_STATUS_UNSUPPORTED,
                  "floating-point GEMM requires F32 accumulation");
  }
  if ((spec.quantization == APXINF_GEMM_QUANT_FP8_UNIT_SCALE ||
       spec.quantization == APXINF_GEMM_QUANT_FP8_ROW_CHANNEL) &&
      (spec.a_dtype != APXINF_DTYPE_E4M3 ||
       spec.b_dtype != APXINF_DTYPE_E4M3)) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                  "FP8 quantization requires two E4M3 inputs");
  }
  if (spec.quantization == APXINF_GEMM_QUANT_NONE &&
      (spec.a_dtype == APXINF_DTYPE_E4M3 ||
       spec.a_dtype == APXINF_DTYPE_I8 ||
       spec.a_dtype != spec.b_dtype)) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                  "plain GEMM requires matching non-quantized inputs");
  }
  if (spec.semantic == APXINF_GEMM_SEMANTIC_GEMM_GEGLU && spec.n % 2 != 0) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                  "GEMM+GeGLU requires an even projection width");
  }
  if (apxinf::gemm::has_row_channel_scales(spec) &&
      spec.semantic == APXINF_GEMM_SEMANTIC_GEMM_GEGLU) {
    throw Failure(APXINF_STATUS_UNSUPPORTED,
                  "rowwise GEMM+GeGLU is not implemented");
  }
  constexpr int64_t kMaximumElementBytes = 4;
  if (spec.m > INT64_MAX / spec.n / kMaximumElementBytes ||
      spec.m > INT64_MAX / spec.k / kMaximumElementBytes ||
      spec.k > INT64_MAX / spec.n / kMaximumElementBytes) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT, "GEMM size overflow");
  }
}

void validate_policy(const apxinf_gemm_policy_t& policy) {
  if (policy.online_tune > 1 || policy.allow_fallback > 1 ||
      policy.graph_safe > 1 || policy.deterministic > 1 ||
      policy.execution_mode > 1) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT, "invalid GEMM Policy");
  }
}

void validate_bindings(const apxinf::gemm::Spec& spec,
                       const apxinf_gemm_bindings_t& bindings,
                       bool require_output) {
  const bool needs_bias =
      spec.semantic == APXINF_GEMM_SEMANTIC_GEMM_BIAS ||
      spec.semantic == APXINF_GEMM_SEMANTIC_GEMM_BIAS_GELU;
  const bool needs_scales =
      apxinf::gemm::has_row_channel_scales(spec);
  if (bindings.a == nullptr || bindings.b == nullptr ||
      (require_output && bindings.output == nullptr) ||
      (needs_bias && bindings.bias == nullptr) ||
      (needs_scales &&
       (bindings.a_scales == nullptr || bindings.b_scales == nullptr))) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                  "missing required GEMM bindings");
  }
  if (!needs_bias && bindings.bias != nullptr) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT, "unexpected GEMM bias");
  }
  if (!std::isfinite(bindings.alpha) ||
      !std::isfinite(bindings.output_scale) ||
      bindings.output_scale <= 0.0F) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT, "invalid GEMM scale binding");
  }
  // The Spec only records whether each scale is unit. A binding that
  // contradicts that predicate would silently execute on a candidate selected
  // for the other case, so it is rejected instead.
  if ((bindings.alpha == 1.0F) != (spec.alpha_is_unit != 0) ||
      (bindings.output_scale == 1.0F) != (spec.output_scale_is_unit != 0)) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                  "GEMM scale binding contradicts the plan scale predicate");
  }
  if (bindings.b_is_immutable > 1 ||
      (bindings.b_is_immutable == 0 && bindings.b_version != 0)) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                  "invalid GEMM immutable-weight identity");
  }
  validate_recorded_alignment(bindings.a, spec.a_alignment, "GEMM A binding");
  validate_recorded_alignment(bindings.b, spec.b_alignment, "GEMM B binding");
  validate_recorded_alignment(bindings.bias, spec.bias_alignment,
                              "GEMM bias binding");
  validate_recorded_alignment(bindings.a_scales, spec.a_scales_alignment,
                              "GEMM A scales binding");
  validate_recorded_alignment(bindings.b_scales, spec.b_scales_alignment,
                              "GEMM B scales binding");
  validate_recorded_alignment(bindings.output, spec.output_alignment,
                              "GEMM output binding");
}

const apxinf::gemm::Implementation* find_implementation(const Recipe& recipe,
                                                        const apxinf::gemm::Spec& spec,
                                                        int device) {
  for (const auto& implementation :
       apxinf::gemm::registry(spec.semantic)) {
    if (implementation.provider_id == recipe.provider_id &&
        implementation.implementation_id == recipe.implementation_id &&
        implementation.implementation_version ==
            recipe.implementation_version &&
        apxinf::gemm::supports_device(implementation, device) &&
        implementation.supports(spec)) {
      std::vector<int> configurations;
      implementation.enumerate_configs(spec, configurations);
      if (std::find(configurations.begin(), configurations.end(),
                    recipe.configuration) != configurations.end()) {
        return &implementation;
      }
    }
  }
  return nullptr;
}

std::shared_ptr<State> fallback(const apxinf::gemm::Spec& spec,
                                const apxinf_gemm_policy_t& policy,
                                int device) {
  for (const auto& implementation :
       apxinf::gemm::registry(spec.semantic)) {
    if (!apxinf::gemm::supports_device(implementation, device) ||
        !implementation.supports(spec) ||
        ((policy.graph_safe || policy.execution_mode == 1) &&
         !implementation.graph_safe) ||
        (policy.deterministic && !implementation.deterministic)) {
      continue;
    }
    std::vector<int> configurations;
    implementation.enumerate_configs(spec, configurations);
    for (int configuration : configurations) {
      try {
        return apxinf::gemm::prepare(implementation, configuration, spec,
                                    policy, device);
      } catch (const Failure&) {
        cudaGetLastError();
      }
    }
  }
  throw Failure(APXINF_STATUS_UNSUPPORTED,
                "no GEMM fallback satisfies the Spec and Policy");
}

void release_instance_resources(State& state) {
  // The framework only requests compaction. Each provider knows which
  // allocations are transient and which recipe data must survive so that a
  // later execution instance can be reconstructed.
  state.implementation->release_resources(state);
}

}  // namespace

const char* plan_summary(apxinf_gemm_plan* plan) {
  return plan != nullptr ? plan->summary.c_str() : "null GEMM plan";
}

apxinf_status_t plan_create(
    apxinf_runtime_t runtime,
    const apxinf_gemm_spec_t* spec,
    const apxinf_gemm_policy_t* policy,
    const apxinf_gemm_tuning_bindings_t* tuning_bindings,
    apxinf::gemm::Semantic semantic,
    apxinf_gemm_plan** output) {
  if (output != nullptr) {
    *output = nullptr;
  }
  return apxinf::gemm::abi_boundary([&] {
    if (runtime == nullptr || spec == nullptr || policy == nullptr ||
        output == nullptr) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                    "null GEMM plan argument");
    }
    apxinf::gemm::Spec normalized_spec{};
    static_cast<apxinf_gemm_spec_t&>(normalized_spec) = *spec;
    normalized_spec.semantic = semantic;
    validate_spec(normalized_spec);
    validate_policy(*policy);
    if (tuning_bindings != nullptr) {
      validate_bindings(normalized_spec, tuning_bindings->execution, false);
      if (tuning_bindings->reference_kind >
          APXINF_GEMM_REFERENCE_ORIGINAL_FP32) {
        throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                      "invalid GEMM reference kind");
      }
      const bool needs_bias =
          semantic == APXINF_GEMM_SEMANTIC_GEMM_BIAS ||
          semantic == APXINF_GEMM_SEMANTIC_GEMM_BIAS_GELU;
      if (tuning_bindings->reference_kind ==
          APXINF_GEMM_REFERENCE_ORIGINAL_FP32) {
        const uint64_t a_count = static_cast<uint64_t>(normalized_spec.m) *
                                 static_cast<uint64_t>(normalized_spec.k);
        const uint64_t b_count = static_cast<uint64_t>(normalized_spec.k) *
                                 static_cast<uint64_t>(normalized_spec.n);
        if (tuning_bindings->original_a == nullptr ||
            tuning_bindings->original_b == nullptr ||
            tuning_bindings->original_a_len != a_count ||
            tuning_bindings->original_b_len != b_count ||
            (needs_bias &&
             (tuning_bindings->original_bias == nullptr ||
              tuning_bindings->original_bias_len !=
                  static_cast<uint64_t>(normalized_spec.n))) ||
            (!needs_bias &&
             (tuning_bindings->original_bias != nullptr ||
              tuning_bindings->original_bias_len != 0))) {
          throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                        "invalid original FP32 GEMM reference operands");
        }
      } else if (tuning_bindings->original_a != nullptr ||
                 tuning_bindings->original_a_len != 0 ||
                 tuning_bindings->original_b != nullptr ||
                 tuning_bindings->original_b_len != 0 ||
                 tuning_bindings->original_bias != nullptr ||
                 tuning_bindings->original_bias_len != 0) {
        throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                      "unexpected original FP32 GEMM reference operands");
      }
      cudaStreamCaptureStatus capture_status;
      apxinf::gemm::check_cuda(cudaStreamIsCapturing(
          static_cast<cudaStream_t>(tuning_bindings->execution.stream),
          &capture_status));
      if (capture_status != cudaStreamCaptureStatusNone) {
        throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                      "create GEMM plans before graph capture");
      }
    }

    apxinf::gemm::check_cuda(cudaSetDevice(runtime->device));
    const auto keys =
        apxinf::gemm::tuning_keys(normalized_spec, *policy, runtime->device);
    const std::string& key = keys.performance;
    std::lock_guard<std::mutex> lock(runtime->gemm_mutex);
    if (const auto cached = runtime->gemm_plans.find(key);
        cached != runtime->gemm_plans.end()) {
      auto plan = std::make_unique<apxinf_gemm_plan>();
      plan->state = cached->second;
      plan->policy = *policy;
      plan->policy.cache_dir = nullptr;
      plan->summary = std::string(plan->state->implementation->name) +
                      " source=memory";
      *output = plan.release();
      return;
    }

    Recipe recipe{};
    bool recipe_found = false;
    Recipe compatible_hint{};
    bool compatible_hint_found = false;
    std::string source = "recipe";
    if (const auto cached = runtime->gemm_recipes.find(key);
        cached != runtime->gemm_recipes.end()) {
      recipe = cached->second;
      recipe_found = true;
      source = "memory-recipe";
    } else {
      const std::string serialized = apxinf::gemm::read_recipe(
          policy->cache_dir != nullptr ? policy->cache_dir : "", key);
      std::istringstream input(serialized);
      recipe_found = static_cast<bool>(
          input >> recipe.provider_id >> recipe.implementation_id >>
          recipe.implementation_version >> recipe.configuration);
    }

    // A compatibility hint may come from a device with a different
    // performance profile. It is never restored as a fully tuned plan: it is
    // only moved to the front of the next complete tuning pass.
    if (!recipe_found) {
      const std::string serialized = apxinf::gemm::read_recipe(
          policy->cache_dir != nullptr ? policy->cache_dir : "",
          keys.compatible_hint);
      std::istringstream input(serialized);
      compatible_hint_found = static_cast<bool>(
          input >> compatible_hint.provider_id >>
          compatible_hint.implementation_id >>
          compatible_hint.implementation_version >>
          compatible_hint.configuration);
      if (compatible_hint_found &&
          find_implementation(compatible_hint, normalized_spec,
                              runtime->device) == nullptr) {
        compatible_hint_found = false;
      }
    }

    std::shared_ptr<State> state;
    if (recipe_found) {
      if (const auto* implementation =
              find_implementation(recipe, normalized_spec, runtime->device)) {
        try {
          state = apxinf::gemm::prepare(*implementation, recipe.configuration,
                                       normalized_spec, *policy, runtime->device);
        } catch (const Failure&) {
          cudaGetLastError();
        }
        if (state == nullptr) {
          compatible_hint = recipe;
          compatible_hint_found = true;
        }
      }
    }

    if (state == nullptr) {
      if (policy->online_tune && tuning_bindings != nullptr) {
        state = apxinf::gemm::tune(normalized_spec, *policy, *tuning_bindings,
                                   runtime->device, source,
                                   compatible_hint_found ? &compatible_hint
                                                         : nullptr);
      } else if (policy->allow_fallback) {
        state = fallback(normalized_spec, *policy, runtime->device);
        source = "fallback";
      } else {
        throw Failure(APXINF_STATUS_CACHE_MISS,
                      "GEMM recipe miss and tuning/fallback are disabled");
      }

      recipe = {state->implementation->provider_id,
                state->implementation->implementation_id,
                state->implementation->implementation_version,
                state->configuration};
      if (source != "fallback") {
        runtime->gemm_recipes[key] = recipe;
        std::ostringstream serialized;
        serialized << recipe.provider_id << ' ' << recipe.implementation_id
                   << ' ' << recipe.implementation_version << ' '
                   << recipe.configuration;
        apxinf::gemm::write_recipe(
            policy->cache_dir != nullptr ? policy->cache_dir : "", key,
            serialized.str());
        apxinf::gemm::write_recipe(
            policy->cache_dir != nullptr ? policy->cache_dir : "",
            keys.compatible_hint, serialized.str());
      }
    }

    release_instance_resources(*state);
    if (source != "fallback") {
      if (runtime->gemm_plans.size() >= 64) {
        runtime->gemm_plans.erase(runtime->gemm_plans.begin());
      }
      runtime->gemm_plans[key] = state;
    }

    auto plan = std::make_unique<apxinf_gemm_plan>();
    plan->state = state;
    plan->policy = *policy;
    plan->policy.cache_dir = nullptr;
    plan->summary =
        std::string(state->implementation->name) +
        " config=" + std::to_string(state->configuration) +
        " workspace=" + std::to_string(state->resource_bytes) +
        " source=" + source;
    *output = plan.release();
  });
}

apxinf_status_t instance_create(
    apxinf_gemm_plan* plan,
    const apxinf_gemm_bindings_t* bindings,
    apxinf_gemm_instance** output) {
  if (output != nullptr) {
    *output = nullptr;
  }
  return apxinf::gemm::abi_boundary([&] {
    if (plan == nullptr || bindings == nullptr || output == nullptr) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                    "null GEMM instance argument");
    }
    validate_bindings(plan->state->spec, *bindings, true);
    cudaStreamCaptureStatus capture_status;
    apxinf::gemm::check_cuda(cudaStreamIsCapturing(
        static_cast<cudaStream_t>(bindings->stream), &capture_status));
    if (capture_status != cudaStreamCaptureStatusNone) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                    "create GEMM instances before graph capture");
    }

    auto instance = std::make_unique<apxinf_gemm_instance>();
    const State& blueprint = *plan->state;
    instance->state = apxinf::gemm::prepare(
        *blueprint.implementation, blueprint.configuration, blueprint.spec,
        plan->policy, blueprint.device, &blueprint);
    instance->bindings = *bindings;
    if (instance->state->implementation->bind_state != nullptr) {
      instance->state->implementation->bind_state(*instance->state,
                                                  instance->bindings);
    }

    void* scratch = nullptr;
    auto warmup_bindings = instance->bindings;
    const int64_t output_width =
        blueprint.spec.semantic == APXINF_GEMM_SEMANTIC_GEMM_GEGLU
            ? blueprint.spec.n / 2
            : blueprint.spec.n;
    const size_t scratch_bytes = blueprint.spec.m * output_width *
                                 apxinf::gemm::dtype_bytes(
                                     blueprint.spec.output_dtype);
    apxinf::gemm::check_cuda(cudaMalloc(&scratch, scratch_bytes));
    warmup_bindings.output = scratch;
    try {
      apxinf::gemm::check_cuda(instance->state->implementation->launch(
          *instance->state, warmup_bindings));
      apxinf::gemm::check_cuda(cudaStreamSynchronize(
          static_cast<cudaStream_t>(bindings->stream)));
    } catch (...) {
      cudaFree(scratch);
      throw;
    }
    apxinf::gemm::check_cuda(cudaFree(scratch));
    *output = instance.release();
  });
}

apxinf_status_t enqueue(apxinf_gemm_instance* instance) {
  return apxinf::gemm::abi_boundary([&] {
    if (instance == nullptr) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT, "null GEMM instance");
    }
    apxinf::gemm::check_cuda(cudaSetDevice(instance->state->device));
    apxinf::gemm::check_cuda(instance->state->implementation->launch(
        *instance->state, instance->bindings));
  });
}

void instance_destroy(apxinf_gemm_instance* instance) {
  delete instance;
}

extern "C" uint64_t apxinf_gemm_instance_weight_prepack_count(
    apxinf_gemm_instance* instance) {
  if (instance == nullptr || instance->state->implementation->provider_id != 3) {
    return 0;
  }
  return apxinf::gemm::cutlass_weight_prepack_count(*instance->state);
}

// Private test hook used to select a provider without running the very large
// provider-independent CPU reference required by production GeGLU shapes.
extern "C" apxinf_status_t apxinf_gemm_test_seed_recipe(
    const apxinf_gemm_spec_t* spec,
    const apxinf_gemm_policy_t* policy,
    uint32_t semantic,
    int device,
    uint32_t provider_id,
    uint32_t implementation_id,
    uint32_t implementation_version,
    int32_t configuration) {
  return apxinf::gemm::abi_boundary([&] {
    if (spec == nullptr || policy == nullptr || policy->cache_dir == nullptr ||
        semantic > static_cast<uint32_t>(APXINF_GEMM_SEMANTIC_GEMM_BIAS)) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                    "invalid test recipe seed arguments");
    }
    apxinf::gemm::Spec normalized{};
    static_cast<apxinf_gemm_spec_t&>(normalized) = *spec;
    normalized.semantic = static_cast<apxinf::gemm::Semantic>(semantic);
    validate_spec(normalized);
    const Recipe recipe{provider_id, implementation_id,
                        implementation_version, configuration};
    if (find_implementation(recipe, normalized, device) == nullptr) {
      throw Failure(APXINF_STATUS_UNSUPPORTED,
                    "test recipe candidate is unavailable");
    }
    const auto keys =
        apxinf::gemm::tuning_keys(normalized, *policy, device);
    std::ostringstream serialized;
    serialized << provider_id << ' ' << implementation_id << ' '
               << implementation_version << ' ' << configuration;
    apxinf::gemm::write_recipe(policy->cache_dir, keys.performance,
                               serialized.str());
  });
}

void plan_destroy(apxinf_gemm_plan* plan) {
  delete plan;
}

#define APXINF_DEFINE_GEMM_PLAN_ABI(prefix, semantic_value, plan_type,          \
                                    instance_type)                             \
  extern "C" apxinf_status_t prefix##_plan_create(                            \
      apxinf_runtime_t runtime, const apxinf_gemm_spec_t* spec,                \
      const apxinf_gemm_policy_t* policy,                                      \
      const apxinf_gemm_tuning_bindings_t* tuning_bindings,                    \
      plan_type* output) {                                                     \
    if (output == nullptr) {                                                   \
      return plan_create(runtime, spec, policy, tuning_bindings, semantic_value,\
                         nullptr);                                             \
    }                                                                          \
    *output = nullptr;                                                         \
    apxinf_gemm_plan* plan = nullptr;                                          \
    const apxinf_status_t status = plan_create(                                \
        runtime, spec, policy, tuning_bindings, semantic_value, &plan);         \
    if (status == APXINF_STATUS_OK) {                                          \
      *output = reinterpret_cast<plan_type>(plan);                             \
    }                                                                          \
    return status;                                                             \
  }                                                                            \
  extern "C" apxinf_status_t prefix##_instance_create(                        \
      plan_type plan, const apxinf_gemm_bindings_t* bindings,                  \
      instance_type* output) {                                                 \
    if (output == nullptr) {                                                   \
      return instance_create(reinterpret_cast<apxinf_gemm_plan*>(plan),        \
                             bindings, nullptr);                               \
    }                                                                          \
    *output = nullptr;                                                         \
    apxinf_gemm_instance* instance = nullptr;                                  \
    const apxinf_status_t status = instance_create(                            \
        reinterpret_cast<apxinf_gemm_plan*>(plan), bindings, &instance);       \
    if (status == APXINF_STATUS_OK) {                                          \
      *output = reinterpret_cast<instance_type>(instance);                     \
    }                                                                          \
    return status;                                                             \
  }                                                                            \
  extern "C" apxinf_status_t prefix##_enqueue(instance_type instance) {       \
    return enqueue(reinterpret_cast<apxinf_gemm_instance*>(instance));         \
  }                                                                            \
  extern "C" void prefix##_instance_destroy(instance_type instance) {         \
    instance_destroy(reinterpret_cast<apxinf_gemm_instance*>(instance));       \
  }                                                                            \
  extern "C" void prefix##_plan_destroy(plan_type plan) {                     \
    plan_destroy(reinterpret_cast<apxinf_gemm_plan*>(plan));                   \
  }                                                                            \
  extern "C" const char* prefix##_plan_summary(plan_type plan) {              \
    return plan_summary(reinterpret_cast<apxinf_gemm_plan*>(plan));            \
  }

APXINF_DEFINE_GEMM_PLAN_ABI(apxinf_gemm, apxinf::gemm::Semantic::kGemm,
                            apxinf_gemm_plan_t, apxinf_gemm_instance_t)
APXINF_DEFINE_GEMM_PLAN_ABI(apxinf_gemm_bias,
                            apxinf::gemm::Semantic::kGemmBias,
                            apxinf_gemm_bias_plan_t,
                            apxinf_gemm_bias_instance_t)
APXINF_DEFINE_GEMM_PLAN_ABI(apxinf_gemm_bias_gelu,
                            apxinf::gemm::Semantic::kGemmBiasGelu,
                            apxinf_gemm_bias_gelu_plan_t,
                            apxinf_gemm_bias_gelu_instance_t)
APXINF_DEFINE_GEMM_PLAN_ABI(apxinf_gemm_geglu,
                            apxinf::gemm::Semantic::kGemmGeglu,
                            apxinf_gemm_geglu_plan_t,
                            apxinf_gemm_geglu_instance_t)

#undef APXINF_DEFINE_GEMM_PLAN_ABI
