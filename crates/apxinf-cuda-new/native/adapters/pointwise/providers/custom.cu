#include "../internal.h"

#include "../../../kernels/custom/pointwise.cuh"

#include <cstdint>

namespace {
#include "../../../kernels/ported/math.cuh"
#include "../../../kernels/ported/activation.cuh"

int ported_blocks_for(int64_t count) {
  return static_cast<int>((count + kThreads - 1) / kThreads);
}

cudaError_t launch_ported_bf16_geglu(
    const __nv_bfloat16* input, __nv_bfloat16* output, int rows, int cols,
    cudaStream_t stream) {
  const int64_t count = static_cast<int64_t>(rows) * cols;
  const bool packed4 =
      cols % 4 == 0 &&
      reinterpret_cast<uintptr_t>(input) % alignof(Bf16x4) == 0 &&
      reinterpret_cast<uintptr_t>(output) % alignof(Bf16x4) == 0;
  const bool packed2 =
      cols % 2 == 0 &&
      reinterpret_cast<uintptr_t>(input) % alignof(__nv_bfloat162) == 0 &&
      reinterpret_cast<uintptr_t>(output) % alignof(__nv_bfloat162) == 0;
  if (packed4) {
    geglu_bf16_packed4_kernel<<<ported_blocks_for(count / 4), kThreads, 0,
                                stream>>>(input, output, rows, cols);
  } else if (packed2) {
    geglu_bf16_packed2_kernel<<<ported_blocks_for(count / 2), kThreads, 0,
                                stream>>>(input, output, rows, cols);
  } else {
    geglu_bf16_kernel<<<ported_blocks_for(count), kThreads, 0, stream>>>(
        input, output, rows, cols);
  }
  return cudaGetLastError();
}

cudaError_t launch_ported_bf16_bias_activation(
    const __nv_bfloat16* input, const __nv_bfloat16* bias,
    __nv_bfloat16* output, int rows, int cols, int activation,
    cudaStream_t stream) {
  const int64_t count = static_cast<int64_t>(rows) * cols;
  const bool packed4 =
      cols % 4 == 0 &&
      reinterpret_cast<uintptr_t>(input) % alignof(Bf16x4) == 0 &&
      reinterpret_cast<uintptr_t>(output) % alignof(Bf16x4) == 0 &&
      (bias == nullptr ||
       reinterpret_cast<uintptr_t>(bias) % alignof(Bf16x4) == 0);
  const bool packed2 =
      cols % 2 == 0 &&
      reinterpret_cast<uintptr_t>(input) % alignof(__nv_bfloat162) == 0 &&
      reinterpret_cast<uintptr_t>(output) % alignof(__nv_bfloat162) == 0 &&
      (bias == nullptr ||
       reinterpret_cast<uintptr_t>(bias) % alignof(__nv_bfloat162) == 0);
  if (packed4) {
    bias_activation_bf16_packed4_kernel<<<
        ported_blocks_for(count / 4), kThreads, 0, stream>>>(
        input, bias, output, count / 4, cols, activation);
  } else if (packed2) {
    bias_activation_bf16_packed2_kernel<<<
        ported_blocks_for(count / 2), kThreads, 0, stream>>>(
        input, bias, output, count / 2, cols, activation);
  } else {
    bias_activation_bf16_kernel<<<ported_blocks_for(count), kThreads, 0,
                                  stream>>>(
        input, bias, output, count, cols, activation);
  }
  return cudaGetLastError();
}
}  // namespace

namespace apxinf::pointwise {
namespace {

namespace k = apxinf::pointwise::kernels;

constexpr int kThreads = 256;
constexpr int kMaxBlocks = 4096;

int blocks_for(int64_t count) {
  const int64_t blocks = (count + kThreads - 1) / kThreads;
  if (blocks < 1) return 1;
  return static_cast<int>(blocks < kMaxBlocks ? blocks : kMaxBlocks);
}

template <class T>
cudaError_t launch(const Spec& spec,
                   const apxinf_pointwise_bindings_t& bindings) {
  const auto* input = static_cast<const T*>(bindings.input);
  const auto* secondary = static_cast<const T*>(bindings.secondary);
  const auto* bias = static_cast<const T*>(bindings.bias);
  auto* output = static_cast<T*>(bindings.output);

  const int rows = static_cast<int>(spec.rows);
  const int cols = static_cast<int>(spec.cols);
  const int64_t count = spec.rows * spec.cols;
  auto stream = static_cast<cudaStream_t>(bindings.stream);
  const int grid = blocks_for(count);

  switch (spec.semantic) {
    case APXINF_POINTWISE_SEMANTIC_GEGLU:
      k::geglu<T><<<grid, kThreads, 0, stream>>>(input, output, rows, cols);
      break;
    case APXINF_POINTWISE_SEMANTIC_BIAS_ACTIVATION:
      k::bias_activation<T><<<grid, kThreads, 0, stream>>>(
          input, bias, output, count, cols,
          static_cast<int>(spec.activation));
      break;
    case APXINF_POINTWISE_SEMANTIC_EULER_UPDATE:
      k::euler_update<T><<<grid, kThreads, 0, stream>>>(
          input, secondary, output, count, bindings.dt);
      break;
    default:
      return cudaErrorInvalidValue;
  }
  return cudaGetLastError();
}

cudaError_t launch_bf16(const Spec& spec,
                        const apxinf_pointwise_bindings_t& bindings) {
  const auto* input = static_cast<const __nv_bfloat16*>(bindings.input);
  const auto* bias = static_cast<const __nv_bfloat16*>(bindings.bias);
  auto* output = static_cast<__nv_bfloat16*>(bindings.output);
  auto stream = static_cast<cudaStream_t>(bindings.stream);
  const int rows = static_cast<int>(spec.rows);
  const int cols = static_cast<int>(spec.cols);
  if (spec.semantic == APXINF_POINTWISE_SEMANTIC_GEGLU) {
    return launch_ported_bf16_geglu(input, output, rows, cols, stream);
  }
  if (spec.semantic == APXINF_POINTWISE_SEMANTIC_BIAS_ACTIVATION) {
    return launch_ported_bf16_bias_activation(
        input, bias, output, rows, cols, static_cast<int>(spec.activation),
        stream);
  }
  return launch<__nv_bfloat16>(spec, bindings);
}

}  // namespace

size_t custom_resource_requirements(const Spec&) { return 0; }

void prepare_custom(Execution& execution) {
  execution.provider_state = nullptr;
  execution.resource_bytes = 0;
}

cudaError_t launch_custom(Execution& execution) {
  return execution.spec.dtype == APXINF_DTYPE_BF16
             ? launch_bf16(execution.spec, execution.bindings)
             : launch<__half>(execution.spec, execution.bindings);
}

void destroy_custom(Execution&) noexcept {}

}  // namespace apxinf::pointwise
