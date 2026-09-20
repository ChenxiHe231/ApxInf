#include "../internal.h"

#include "../../../kernels/custom/gather.cuh"

namespace apxinf::gather {
namespace {

namespace k = apxinf::gather::kernels;

constexpr int kThreads = 256;
constexpr int kMaxBlocks = 4096;

int blocks_for(int64_t count) {
  const int64_t blocks = (count + kThreads - 1) / kThreads;
  if (blocks < 1) return 1;
  return static_cast<int>(blocks < kMaxBlocks ? blocks : kMaxBlocks);
}

template <class T>
cudaError_t launch(const Spec& spec,
                   const apxinf_gather_bindings_t& bindings) {
  const auto* input = static_cast<const T*>(bindings.input);
  const auto* bias = static_cast<const T*>(bindings.bias);
  const auto* position = static_cast<const T*>(bindings.position);
  auto* output = static_cast<T*>(bindings.output);

  const int rows = static_cast<int>(spec.rows);
  const int cols = static_cast<int>(spec.cols);
  const int64_t count = spec.rows * spec.cols;
  auto stream = static_cast<cudaStream_t>(bindings.stream);
  const int grid = blocks_for(count);

  switch (spec.semantic) {
    case APXINF_GATHER_SEMANTIC_EMBEDDING_LOOKUP:
      k::embedding_lookup<T><<<grid, kThreads, 0, stream>>>(
          input, bindings.ids, output, rows, cols,
          static_cast<int>(spec.vocab_size));
      break;
    case APXINF_GATHER_SEMANTIC_BIAS_POSITION:
      k::bias_position<T><<<grid, kThreads, 0, stream>>>(
          input, bias, position, output, count, cols,
          static_cast<int>(spec.tokens_per_view));
      break;
    case APXINF_GATHER_SEMANTIC_RGB_TO_PATCHES: {
      const auto* images = static_cast<const uint8_t*>(bindings.input);
      if (spec.nhwc != 0) {
        k::rgb_u8_to_patches<T, true><<<grid, kThreads, 0, stream>>>(
            images, output, static_cast<int>(spec.views),
            static_cast<int>(spec.image_size),
            static_cast<int>(spec.patch_size));
      } else {
        k::rgb_u8_to_patches<T, false><<<grid, kThreads, 0, stream>>>(
            images, output, static_cast<int>(spec.views),
            static_cast<int>(spec.image_size),
            static_cast<int>(spec.patch_size));
      }
      break;
    }
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

}  // namespace apxinf::gather
