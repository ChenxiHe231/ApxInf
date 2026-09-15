#include "../internal.h"
#include "../../../kernels/custom/attention.cuh"

namespace apxinf::attention {
namespace {

struct CustomState {
  float* probabilities = nullptr;
};

}  // namespace

size_t custom_resource_requirements(const Spec& spec) {
  const auto rows = static_cast<uint64_t>(spec.batch) * spec.query_tokens *
                    spec.query_heads;
  const auto elements = rows * spec.key_tokens;
  if (elements > SIZE_MAX / sizeof(float)) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                  "Attention probability workspace size overflow");
  }
  return static_cast<size_t>(elements) * sizeof(float);
}

void prepare_custom(Execution& execution) {
  auto state = std::make_unique<CustomState>();
  const size_t bytes = custom_resource_requirements(execution.spec);
  if (bytes > execution.resource_limit) {
    throw Failure(APXINF_STATUS_UNSUPPORTED,
                  "Attention probability workspace exceeds policy");
  }
  check_cuda(cudaMalloc(&state->probabilities, bytes));
  execution.resource_bytes = bytes;
  execution.provider_state = state.release();
}

cudaError_t launch_custom(Execution& execution) {
  auto* state = static_cast<CustomState*>(execution.provider_state);
  const auto& spec = execution.spec;
  const auto& bindings = execution.bindings;
  const auto stream = static_cast<cudaStream_t>(bindings.stream);
  if (spec.dtype == APXINF_DTYPE_F16) {
    return kernels::launch_attention<__half>(
        bindings.query, bindings.key, bindings.value, bindings.output,
        state->probabilities, static_cast<int>(spec.batch),
        static_cast<int>(spec.query_tokens), static_cast<int>(spec.key_tokens),
        static_cast<int>(spec.query_heads), static_cast<int>(spec.kv_heads),
        static_cast<int>(spec.head_dim),
        spec.mask == APXINF_ATTENTION_MASK_CAUSAL, bindings.scale, stream);
  }
  return kernels::launch_attention<__nv_bfloat16>(
      bindings.query, bindings.key, bindings.value, bindings.output,
      state->probabilities, static_cast<int>(spec.batch),
      static_cast<int>(spec.query_tokens), static_cast<int>(spec.key_tokens),
      static_cast<int>(spec.query_heads), static_cast<int>(spec.kv_heads),
      static_cast<int>(spec.head_dim),
      spec.mask == APXINF_ATTENTION_MASK_CAUSAL, bindings.scale, stream);
}

void destroy_custom(Execution& execution) noexcept {
  auto* state = static_cast<CustomState*>(execution.provider_state);
  if (state == nullptr) return;
  if (state->probabilities != nullptr) cudaFree(state->probabilities);
  delete state;
  execution.provider_state = nullptr;
}

}  // namespace apxinf::attention
