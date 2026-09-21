#include "../internal.h"

#include "../../../kernels/custom/rope.cuh"

namespace apxinf::rope {
namespace {

namespace k = apxinf::rope::kernels;

// The rotation kernel assigns one thread per (element, element + head_dim/2)
// pair, so the block only has to cover half a head.
int rope_threads(int head_dim) {
  const int half = head_dim / 2;
  int threads = 32;
  while (threads < half && threads < 1024) threads *= 2;
  return threads;
}

template <class T>
cudaError_t launch(const Spec& spec, const apxinf_rope_bindings_t& bindings) {
  const auto* qkv = static_cast<const T*>(bindings.qkv);
  const auto* bias = static_cast<const T*>(bindings.bias);
  auto* q = static_cast<T*>(bindings.q);
  auto* kv_k = static_cast<T*>(bindings.k);
  auto* kv_v = static_cast<T*>(bindings.v);

  const int tokens = static_cast<int>(spec.tokens);
  const int q_heads = static_cast<int>(spec.q_heads);
  const int kv_heads = static_cast<int>(spec.kv_heads);
  const int head_dim = static_cast<int>(spec.head_dim);
  auto stream = static_cast<cudaStream_t>(bindings.stream);

  switch (spec.semantic) {
    case APXINF_ROPE_SEMANTIC_SPLIT_QKV_ROPE: {
      const dim3 grid(static_cast<unsigned>(tokens),
                      static_cast<unsigned>(q_heads + 2 * kv_heads));
      k::split_qkv_rope<T><<<grid, rope_threads(head_dim), 0, stream>>>(
          qkv, bias, q, kv_k, kv_v, tokens, q_heads, kv_heads, head_dim,
          bindings.theta, bindings.position_offset, bindings.kv_output_offset);
      break;
    }
    case APXINF_ROPE_SEMANTIC_SPLIT_QKV_BIAS:
      k::split_qkv_bias<T><<<tokens, 256, 0, stream>>>(
          qkv, bias, q, kv_k, kv_v, tokens, q_heads * head_dim);
      break;
    default:
      return cudaErrorInvalidValue;
  }
  return cudaGetLastError();
}

}  // namespace

cudaError_t launch_custom(Execution& execution) {
  return execution.spec.dtype == APXINF_DTYPE_BF16
             ? launch<__nv_bfloat16>(execution.spec, execution.bindings)
             : launch<__half>(execution.spec, execution.bindings);
}

}  // namespace apxinf::rope
