#include "../internal.h"

#include "../../../kernels/custom/pointwise.cuh"

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

}  // namespace

size_t custom_resource_requirements(const Spec&) { return 0; }

void prepare_custom(Execution& execution) {
  execution.provider_state = nullptr;
  execution.resource_bytes = 0;
}

cudaError_t launch_custom(Execution& execution) {
  return execution.spec.dtype == APXINF_DTYPE_BF16
             ? launch<__nv_bfloat16>(execution.spec, execution.bindings)
             : launch<__half>(execution.spec, execution.bindings);
}

void destroy_custom(Execution&) noexcept {}

}  // namespace apxinf::pointwise
