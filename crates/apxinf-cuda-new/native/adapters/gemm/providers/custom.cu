#include "../internal.h"
#include "../../../kernels/custom/gemm.cuh"

namespace apxinf::gemm {

State::~State() {
  cudaSetDevice(device);
  if (workspace != nullptr) cudaFree(workspace);
  if (projection != nullptr) cudaFree(projection);
  if (unpack_a != nullptr) cudaFree(unpack_a);
  if (unpack_b != nullptr) cudaFree(unpack_b);
  if (a_layout != nullptr) cublasLtMatrixLayoutDestroy(a_layout);
  if (b_layout != nullptr) cublasLtMatrixLayoutDestroy(b_layout);
  if (output_layout != nullptr) cublasLtMatrixLayoutDestroy(output_layout);
  if (operation != nullptr) cublasLtMatmulDescDestroy(operation);
  if (cublaslt != nullptr) cublasLtDestroy(cublaslt);
  if (cublas != nullptr) cublasDestroy(cublas);
}

void allocate_common_resources(State& state, bool native_fp8) {
  const auto& spec = state.spec;
  const bool unpack =
      ((spec.a_dtype == APXINF_DTYPE_E4M3 ||
        spec.b_dtype == APXINF_DTYPE_E4M3) &&
       !native_fp8) ||
      (has_row_channel_scales(spec) && !native_fp8 &&
       spec.a_dtype != APXINF_DTYPE_I8);

  state.projection_dtype =
      spec.a_dtype == APXINF_DTYPE_I8
          ? APXINF_DTYPE_I32
          : has_row_channel_scales(spec)
                ? APXINF_DTYPE_F32
                : (spec.a_dtype == APXINF_DTYPE_E4M3 ||
                   spec.b_dtype == APXINF_DTYPE_E4M3)
                      ? APXINF_DTYPE_F16
                      : spec.a_dtype;

  if (unpack) {
    const size_t a_bytes = spec.m * spec.k * dtype_bytes(state.projection_dtype);
    const size_t b_bytes = spec.k * spec.n * dtype_bytes(state.projection_dtype);
    check_cuda(cudaMalloc(&state.unpack_a, a_bytes));
    check_cuda(cudaMalloc(&state.unpack_b, b_bytes));
    state.resource_bytes += a_bytes + b_bytes;
  }

  const bool needs_postprocess =
      spec.semantic != APXINF_GEMM_SEMANTIC_GEMM ||
      has_row_channel_scales(spec) ||
      spec.output_dtype != state.projection_dtype || spec.output_scale != 1.0F;
  if (needs_postprocess) {
    const size_t bytes = spec.m * spec.n * dtype_bytes(state.projection_dtype);
    check_cuda(cudaMalloc(&state.projection, bytes));
    state.resource_bytes += bytes;
  }
}

cudaError_t launch_postprocess(State& state,
                               const apxinf_gemm_bindings_t& bindings,
                               void* projection) {
  if (state.projection == nullptr) {
    return cudaSuccess;
  }
  const auto& spec = state.spec;
  const int64_t output_width =
      spec.semantic == APXINF_GEMM_SEMANTIC_GEMM_GEGLU ? spec.n / 2 : spec.n;
  const int64_t count = spec.m * output_width;
  const int blocks = static_cast<int>(
      std::min<int64_t>((count + 255) / 256, 4096));
  apxinf::cuda::custom::finish<<<blocks, 256, 0,
                                 static_cast<cudaStream_t>(bindings.stream)>>>(
      projection, state.projection_dtype, bindings.output, spec.output_dtype,
      bindings.bias,
      has_row_channel_scales(spec)
          ? spec.output_dtype
          : state.projection_dtype,
      bindings.a_scales, bindings.b_scales, spec.m, spec.n,
      static_cast<int>(spec.semantic),
      has_row_channel_scales(spec) ? 1 : 0,
      has_row_channel_scales(spec) ? spec.alpha : 1.0F,
      spec.output_scale);
  return cudaGetLastError();
}

}  // namespace apxinf::gemm
