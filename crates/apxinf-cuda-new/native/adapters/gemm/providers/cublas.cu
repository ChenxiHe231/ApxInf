#include "../internal.h"
#include "../../../kernels/custom/gemm.cuh"

namespace apxinf::gemm {

void prepare_cublas(State& state) {
  allocate_common_resources(state, false);
  check_cublas(cublasCreate(&state.cublas));
  check_cublas(cublasSetMathMode(state.cublas, CUBLAS_PEDANTIC_MATH));
  state.workspace_bytes = 4 * 1024 * 1024;
  check_cuda(cudaMalloc(&state.workspace, state.workspace_bytes));
  state.resource_bytes += state.workspace_bytes;
}

cudaError_t launch_cublas(State& state,
                          const apxinf_gemm_bindings_t& bindings) {
  const auto stream = static_cast<cudaStream_t>(bindings.stream);
  const auto& spec = state.spec;
  const void* activation = bindings.a;
  const void* weight = bindings.b;
  if (state.unpack_a != nullptr) {
    check_cuda(apxinf::cuda::custom::unpack_gemm(
        activation, state.unpack_a, state.projection_dtype, spec.a_dtype,
        spec.m, spec.k, APXINF_GEMM_LAYOUT_KN, stream));
    check_cuda(apxinf::cuda::custom::unpack_gemm(
        weight, state.unpack_b, state.projection_dtype, spec.b_dtype, spec.k,
        spec.n, APXINF_GEMM_LAYOUT_KN, stream));
    activation = state.unpack_a;
    weight = state.unpack_b;
  }

  void* projection =
      state.projection != nullptr ? state.projection : bindings.output;
  const float alpha =
      has_row_channel_scales(spec) ? 1.0F : spec.alpha;
  const float beta = 0.0F;
  const int32_t integer_alpha = 1;
  const int32_t integer_beta = 0;
  const void* alpha_pointer = spec.a_dtype == APXINF_DTYPE_I8
                                  ? static_cast<const void*>(&integer_alpha)
                                  : static_cast<const void*>(&alpha);
  const void* beta_pointer = spec.a_dtype == APXINF_DTYPE_I8
                                 ? static_cast<const void*>(&integer_beta)
                                 : static_cast<const void*>(&beta);
  const cudaDataType_t data_type =
      state.projection_dtype == 5
          ? CUDA_R_32I
          : state.projection_dtype == APXINF_DTYPE_F32
                ? CUDA_R_32F
                : state.projection_dtype == APXINF_DTYPE_F16 ? CUDA_R_16F
                                                             : CUDA_R_16BF;
  check_cublas(cublasSetStream(state.cublas, stream));
  check_cublas(cublasSetWorkspace(state.cublas, state.workspace,
                                  state.workspace_bytes));
  check_cublas(cublasGemmEx(
      state.cublas, CUBLAS_OP_N, CUBLAS_OP_N, spec.n, spec.m, spec.k,
      alpha_pointer, weight,
      spec.a_dtype == APXINF_DTYPE_I8 ? CUDA_R_8I : data_type,
      spec.n, activation,
      spec.a_dtype == APXINF_DTYPE_I8 ? CUDA_R_8I : data_type, spec.k,
      beta_pointer, projection, data_type, spec.n,
      spec.a_dtype == APXINF_DTYPE_I8 ? CUBLAS_COMPUTE_32I
                                      : CUBLAS_COMPUTE_32F,
      CUBLAS_GEMM_DEFAULT));
  return launch_postprocess(state, bindings, projection);
}

}  // namespace apxinf::gemm
