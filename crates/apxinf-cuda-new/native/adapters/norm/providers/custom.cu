#include "../internal.h"

#include "../../../kernels/custom/norm.cuh"

namespace apxinf::norm {
namespace {

namespace k = apxinf::norm::kernels;

constexpr int kThreads = 256;
// Enough blocks to fill the device for the grid-stride element-wise kernels
// without making the launch depend on the shape.
constexpr int kMaxElementwiseBlocks = 4096;

int elementwise_blocks(int64_t count) {
  const int64_t blocks = (count + kThreads - 1) / kThreads;
  if (blocks < 1) return 1;
  return static_cast<int>(blocks < kMaxElementwiseBlocks
                              ? blocks
                              : kMaxElementwiseBlocks);
}

template <class T>
cudaError_t launch(const Spec& spec, const apxinf_norm_bindings_t& bindings) {
  const auto* input = static_cast<const T*>(bindings.input);
  const auto* bias = static_cast<const T*>(bindings.bias);
  const auto* residual = static_cast<const T*>(bindings.residual);
  const auto* weight = static_cast<const T*>(bindings.weight);
  const auto* norm_bias = static_cast<const T*>(bindings.norm_bias);
  const auto* norm_style = static_cast<const T*>(bindings.norm_style);
  const auto* gate_style = static_cast<const T*>(bindings.gate_style);
  auto* hidden = static_cast<T*>(bindings.hidden);
  auto* normalized = static_cast<T*>(bindings.normalized);

  const int rows = static_cast<int>(spec.rows);
  const int cols = static_cast<int>(spec.cols);
  const int64_t count = spec.rows * spec.cols;
  auto stream = static_cast<cudaStream_t>(bindings.stream);

  switch (spec.semantic) {
    case APXINF_NORM_SEMANTIC_RMS:
      k::rms_norm<T><<<rows, kThreads, 0, stream>>>(
          input, weight, normalized, rows, cols, bindings.eps);
      break;
    case APXINF_NORM_SEMANTIC_LAYER:
      k::layer_norm<T><<<rows, kThreads, 0, stream>>>(
          input, weight, norm_bias, normalized, rows, cols, bindings.eps);
      break;
    case APXINF_NORM_SEMANTIC_ADAPTIVE_RMS:
      k::ada_rms_norm<T><<<rows, kThreads, 0, stream>>>(
          input, norm_style, normalized, rows, cols, bindings.eps);
      break;
    case APXINF_NORM_SEMANTIC_BIAS_RESIDUAL:
      k::bias_residual<T><<<elementwise_blocks(count), kThreads, 0, stream>>>(
          input, bias, residual, hidden, count, cols);
      break;
    case APXINF_NORM_SEMANTIC_BIAS_RESIDUAL_RMS:
      k::bias_residual_rms_norm<T><<<rows, kThreads, 0, stream>>>(
          input, bias, residual, weight, hidden, normalized, rows, cols,
          bindings.eps);
      break;
    case APXINF_NORM_SEMANTIC_BIAS_RESIDUAL_LAYER:
      k::bias_residual_layer_norm<T><<<rows, kThreads, 0, stream>>>(
          input, bias, residual, weight, norm_bias, hidden, normalized, rows,
          cols, bindings.eps);
      break;
    case APXINF_NORM_SEMANTIC_ADA_GATE_RESIDUAL:
      k::ada_gate_residual<T>
          <<<elementwise_blocks(count), kThreads, 0, stream>>>(
              input, residual, gate_style, hidden, count, cols);
      break;
    case APXINF_NORM_SEMANTIC_ADA_GATE_RESIDUAL_RMS:
      k::ada_gate_residual_rms_norm<T><<<rows, kThreads, 0, stream>>>(
          input, residual, gate_style, norm_style, hidden, normalized, rows,
          cols, bindings.eps);
      break;
    case APXINF_NORM_SEMANTIC_BIAS_THEN_RESIDUAL:
      k::bias_then_residual<T>
          <<<elementwise_blocks(count), kThreads, 0, stream>>>(
              input, bias, residual, hidden, count, cols);
      break;
    default:
      return cudaErrorInvalidValue;
  }
  return cudaGetLastError();
}

}  // namespace

size_t custom_resource_requirements(const Spec&) { return 0; }

void prepare_custom(Execution& execution) {
  // The kernels own no descriptors, handles or scratch, so preparation only
  // has to record that there is nothing to allocate.  Keeping the hook means
  // the family still obeys the prepare-before-capture discipline that CUDA
  // graph capture requires.
  execution.provider_state = nullptr;
  execution.resource_bytes = 0;
}

cudaError_t launch_custom(Execution& execution) {
  return execution.spec.dtype == APXINF_DTYPE_BF16
             ? launch<__nv_bfloat16>(execution.spec, execution.bindings)
             : launch<__half>(execution.spec, execution.bindings);
}

void destroy_custom(Execution&) noexcept {}

}  // namespace apxinf::norm
