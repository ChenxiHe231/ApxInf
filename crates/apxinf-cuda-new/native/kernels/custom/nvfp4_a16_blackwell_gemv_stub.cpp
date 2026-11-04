#include "mlp_ops.h"

namespace apxinf::cuda_new::mlp_ops {

int nvfp4_a16_blackwell_gemv(const void*, const void*, const void*,
                              const void*, void*, int, int, int, cudaStream_t) {
  return -1;
}

int nvfp4_a16_blackwell_wmma(const void*, const void*, const void*, const void*,
                           void*, int, int, int, int, cudaStream_t) {
  return -1;
}

}  // namespace apxinf::cuda_new::mlp_ops
