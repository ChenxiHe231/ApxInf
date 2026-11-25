#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <math_constants.h>
#include <cmath>
#include <cfloat>
#include <cstdint>

#include "../../kernels/custom/attention_staged_tiled_bf16.cuh"

cudaError_t apxinf_staged_attention_tiled(
    const void* q, const void* k, const void* v, const void* offsets,
    void* output, void* workspace, int tokens, int segments, int maximum,
    int heads, int dim, int chunk, float scale, cudaStream_t stream) {
  if (workspace == nullptr || chunk <= 0 || chunk > 128 || chunk % 16 != 0 ||
      maximum <= 0 || maximum > 4096)
    return cudaErrorInvalidValue;
  using namespace apxinf::attention::kernels::staged_tiled;
  const auto* query = static_cast<const __nv_bfloat16*>(q);
  const auto* key = static_cast<const __nv_bfloat16*>(k);
  const auto* value = static_cast<const __nv_bfloat16*>(v);
  const auto* boundaries = static_cast<const uint32_t*>(offsets);
  auto* out = static_cast<__nv_bfloat16*>(output);
  auto* probabilities = static_cast<float*>(workspace);
  const dim3 qk_grid(((chunk + 15) / 16) * ((maximum + 31) / 32), heads, segments);
  const dim3 pv_grid(((chunk + 15) / 16) * ((dim + 31) / 32), heads, segments);
  const dim3 rows_grid(chunk, heads, segments);
  for (int first = 0; first < maximum; first += chunk) {
    qk<<<qk_grid, dim3(16, 8), 0, stream>>>(
        query, key, boundaries, probabilities, tokens, maximum, heads, dim,
        first, chunk, scale);
    auto status = cudaGetLastError();
    if (status != cudaSuccess) return status;
    softmax<<<rows_grid, 128, size_t(maximum) * sizeof(float), stream>>>(
        probabilities, boundaries, tokens, maximum, heads, first, chunk);
    status = cudaGetLastError();
    if (status != cudaSuccess) return status;
    pv<<<pv_grid, dim3(16, 8), 0, stream>>>(
        probabilities, value, boundaries, out, tokens, maximum, heads, dim,
        first, chunk);
    status = cudaGetLastError();
    if (status != cudaSuccess) return status;
  }
  return cudaSuccess;
}
