#include "../internal.h"
#include "../../../kernels/custom/attention.cuh"

cudaError_t apxinf_staged_attention_shared_limit(int);
cudaError_t apxinf_staged_attention(const void*,const void*,const void*,const void*,void*,int,int,int,int,int,float,cudaStream_t);
cudaError_t apxinf_staged_attention_tiled(const void*,const void*,const void*,const void*,void*,void*,int,int,int,int,int,int,float,cudaStream_t);

namespace apxinf::attention {
namespace {

struct CustomState {
  float* probabilities = nullptr;
};

}  // namespace

bool supports_staged_tiled(const Spec& spec) {
  return spec.semantic == APXINF_ATTENTION_SEMANTIC_SEGMENTED &&
      spec.numerics == APXINF_ATTENTION_NUMERICS_STAGED_BF16 &&
      spec.dtype == APXINF_DTYPE_BF16 && spec.output_dtype == APXINF_DTYPE_BF16 &&
      spec.batch == 1 && spec.query_heads == spec.kv_heads &&
      spec.query_tokens == spec.key_tokens && spec.mask == APXINF_ATTENTION_MASK_NONE &&
      spec.query_tokens > 0 && spec.query_tokens <= 4096 &&
      spec.max_segment_tokens >= 128 && spec.max_segment_tokens <= 4096 &&
      spec.head_dim > 0 && spec.head_dim <= 256;
}

size_t custom_resource_requirements(const Spec& spec, int configuration) {
  if (spec.semantic == APXINF_ATTENTION_SEMANTIC_SEGMENTED) {
    if (spec.numerics != APXINF_ATTENTION_NUMERICS_STAGED_BF16 || configuration == 1)
      return 0;
    if (configuration != 0 || !supports_staged_tiled(spec))
      throw Failure(APXINF_STATUS_UNSUPPORTED, "unsupported staged tiled geometry");
    size_t bytes = 1;
    for (uint64_t extent : {uint64_t(spec.segments), uint64_t(spec.query_heads),
                           uint64_t(kStagedQueryChunkRows),
                           uint64_t(spec.max_segment_tokens), uint64_t(sizeof(float))}) {
      if (extent == 0 || extent > SIZE_MAX / bytes)
        throw Failure(APXINF_STATUS_UNSUPPORTED, "staged probability workspace overflow");
      bytes *= static_cast<size_t>(extent);
    }
    return bytes;
  }
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
  const size_t bytes =
      custom_resource_requirements(execution.spec, execution.configuration);
  if (bytes > execution.resource_limit) {
    throw Failure(APXINF_STATUS_UNSUPPORTED,
                  "Attention probability workspace exceeds policy");
  }
  if (execution.spec.semantic == APXINF_ATTENTION_SEMANTIC_SEGMENTED) {
    cudaDeviceProp properties{};
    check_cuda(cudaGetDeviceProperties(&properties, execution.device));
    if (execution.spec.query_heads > properties.maxGridSize[1] ||
        execution.spec.segments > properties.maxGridSize[2]) {
      throw Failure(APXINF_STATUS_UNSUPPORTED,
                    "segmented Attention grid exceeds device limits");
    }
    const uint64_t shared_bytes =
        static_cast<uint64_t>(execution.spec.max_segment_tokens) *
        sizeof(float);
    int max_shared_bytes = 0;
    check_cuda(cudaDeviceGetAttribute(
        &max_shared_bytes, cudaDevAttrMaxSharedMemoryPerBlockOptin,
        execution.device));
    if (shared_bytes > static_cast<uint64_t>(max_shared_bytes) ||
        shared_bytes > static_cast<uint64_t>(INT32_MAX)) {
      throw Failure(APXINF_STATUS_UNSUPPORTED,
                    "segmented Attention segment exceeds shared-memory limit");
    }
    if (shared_bytes > static_cast<uint64_t>(properties.sharedMemPerBlock)) {
      if (execution.spec.dtype == APXINF_DTYPE_F16) {
        check_cuda(cudaFuncSetAttribute(
            kernels::segmented_attention_kernel<__half>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(shared_bytes)));
      } else if (execution.spec.numerics == APXINF_ATTENTION_NUMERICS_STAGED_BF16) {
        check_cuda(apxinf_staged_attention_shared_limit(static_cast<int>(shared_bytes)));
      } else {
        check_cuda(cudaFuncSetAttribute(
            kernels::segmented_attention_kernel<__nv_bfloat16>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(shared_bytes)));
      }
    }
  }
  if (bytes != 0) check_cuda(cudaMalloc(&state->probabilities, bytes));
  execution.resource_bytes = bytes;
  execution.provider_state = state.release();
}

cudaError_t launch_custom(Execution& execution) {
  auto* state = static_cast<CustomState*>(execution.provider_state);
  const auto& spec = execution.spec;
  const auto& bindings = execution.bindings;
  const auto stream = static_cast<cudaStream_t>(bindings.stream);
  if (spec.semantic == APXINF_ATTENTION_SEMANTIC_SEGMENTED) {
    if (spec.dtype == APXINF_DTYPE_F16) {
      return kernels::launch_segmented_attention<__half>(
          bindings.query, bindings.key, bindings.value, bindings.offsets,
          bindings.output, static_cast<int>(spec.segments),
          static_cast<int>(spec.max_segment_tokens),
          static_cast<int>(spec.query_tokens),
          static_cast<int>(spec.query_heads), static_cast<int>(spec.head_dim),
          bindings.scale, stream);
    }
    if (spec.numerics == APXINF_ATTENTION_NUMERICS_STAGED_BF16) {
      if (execution.configuration == 0) {
        return apxinf_staged_attention_tiled(
            bindings.query,bindings.key,bindings.value,bindings.offsets,
            bindings.output,state->probabilities,int(spec.query_tokens),int(spec.segments),
            int(spec.max_segment_tokens),int(spec.query_heads),int(spec.head_dim),
            kStagedQueryChunkRows,bindings.scale,stream);
      }
      return apxinf_staged_attention(bindings.query,bindings.key,bindings.value,bindings.offsets,
          bindings.output,int(spec.query_tokens),int(spec.segments),int(spec.max_segment_tokens),
          int(spec.query_heads),int(spec.head_dim),bindings.scale,stream);
    }
    return kernels::launch_segmented_attention<__nv_bfloat16>(
        bindings.query, bindings.key, bindings.value, bindings.offsets,
        bindings.output, static_cast<int>(spec.segments),
        static_cast<int>(spec.max_segment_tokens),
        static_cast<int>(spec.query_tokens),
        static_cast<int>(spec.query_heads), static_cast<int>(spec.head_dim),
        bindings.scale, stream);
  }
  const int key_stride = spec.semantic == APXINF_ATTENTION_SEMANTIC_KV_CACHE
                             ? static_cast<int>(spec.key_capacity)
                             : static_cast<int>(spec.key_tokens);
  if (spec.dtype == APXINF_DTYPE_F16) {
    return kernels::launch_attention<__half>(
        bindings.query, bindings.key, bindings.value, bindings.output,
        state->probabilities, static_cast<int>(spec.batch),
        static_cast<int>(spec.query_tokens), static_cast<int>(spec.key_tokens),
        key_stride,
        static_cast<int>(spec.query_heads), static_cast<int>(spec.kv_heads),
        static_cast<int>(spec.head_dim),
        spec.mask == APXINF_ATTENTION_MASK_CAUSAL,
        static_cast<int>(spec.query_start), bindings.decode_meta,
        bindings.scale, stream);
  }
  return kernels::launch_attention<__nv_bfloat16>(
      bindings.query, bindings.key, bindings.value, bindings.output,
      state->probabilities, static_cast<int>(spec.batch),
      static_cast<int>(spec.query_tokens), static_cast<int>(spec.key_tokens),
      key_stride,
      static_cast<int>(spec.query_heads), static_cast<int>(spec.kv_heads),
      static_cast<int>(spec.head_dim),
      spec.mask == APXINF_ATTENTION_MASK_CAUSAL,
      static_cast<int>(spec.query_start), bindings.decode_meta,
      bindings.scale, stream);
}

void destroy_custom(Execution& execution) noexcept {
  auto* state = static_cast<CustomState*>(execution.provider_state);
  if (state == nullptr) return;
  if (state->probabilities != nullptr) cudaFree(state->probabilities);
  delete state;
  execution.provider_state = nullptr;
}

}  // namespace apxinf::attention

// Private test hooks: check ownership directly rather than inferring it from
// free system memory, which also fluctuates with CPU allocations on Tegra.
extern "C" const void* apxinf_attention_test_custom_workspace(const void* raw) {
  const auto* wrapper = static_cast<const apxinf_attention_execution*>(raw);
  const auto* execution = wrapper == nullptr ? nullptr : wrapper->state.get();
  if (execution == nullptr || execution->implementation->provider_id != 1 ||
      execution->implementation->implementation_id != 4) return nullptr;
  const auto* state = static_cast<const apxinf::attention::CustomState*>(
      execution->provider_state);
  return state == nullptr ? nullptr : state->probabilities;
}

extern "C" int apxinf_attention_test_is_live_device_pointer(const void* pointer) {
  if (pointer == nullptr) return 0;
  cudaPointerAttributes attributes{};
  const auto status = cudaPointerGetAttributes(&attributes, pointer);
  if (status != cudaSuccess) {
    cudaGetLastError();
    return 0;
  }
  return attributes.type == cudaMemoryTypeDevice &&
      attributes.devicePointer == pointer;
}
