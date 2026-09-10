#include "../internal.h"
#include "../../../kernels/custom/gemm.cuh"

namespace apxinf::gemm {
namespace {

cudaDataType_t projection_cuda_dtype(const State& state) {
  if (state.projection_dtype == APXINF_DTYPE_I32) return CUDA_R_32I;
  if (state.projection_dtype == APXINF_DTYPE_F32) return CUDA_R_32F;
  if (state.projection_dtype == APXINF_DTYPE_F16) return CUDA_R_16F;
  return CUDA_R_16BF;
}

}  // namespace

void prepare_cublaslt(State& state) {
  const auto& spec = state.spec;
  const bool native_fp8 = state.implementation->implementation_id == 2;
  allocate_common_resources(state, native_fp8);
  const cudaDataType_t projection_type = projection_cuda_dtype(state);
  const cudaDataType_t input_type =
      spec.a_dtype == APXINF_DTYPE_I8
          ? CUDA_R_8I
          : native_fp8 ? CUDA_R_8F_E4M3 : projection_type;
  const cublasComputeType_t compute_type =
      spec.a_dtype == APXINF_DTYPE_I8 ? CUBLAS_COMPUTE_32I
                                      : CUBLAS_COMPUTE_32F;
  const cudaDataType_t scale_type =
      spec.a_dtype == APXINF_DTYPE_I8 ? CUDA_R_32I : CUDA_R_32F;
  const cublasOperation_t transpose = CUBLAS_OP_N;

  check_cublas(cublasLtCreate(&state.cublaslt));
  check_cublas(cublasLtMatmulDescCreate(&state.operation, compute_type,
                                        scale_type));
  check_cublas(cublasLtMatmulDescSetAttribute(
      state.operation, CUBLASLT_MATMUL_DESC_TRANSA, &transpose,
      sizeof(transpose)));
  check_cublas(cublasLtMatrixLayoutCreate(
      &state.a_layout, input_type, transpose == CUBLAS_OP_T ? spec.k : spec.n,
      transpose == CUBLAS_OP_T ? spec.n : spec.k,
      transpose == CUBLAS_OP_T ? spec.k : spec.n));
  check_cublas(cublasLtMatrixLayoutCreate(&state.b_layout, input_type, spec.k,
                                          spec.m, spec.k));
  check_cublas(cublasLtMatrixLayoutCreate(&state.output_layout,
                                          projection_type, spec.n, spec.m,
                                          spec.n));

  if (state.has_algorithm) {
    cublasLtMatmulHeuristicResult_t checked{};
    check_cublas(cublasLtMatmulAlgoCheck(
        state.cublaslt, state.operation, state.a_layout, state.b_layout,
        state.output_layout, state.output_layout, &state.algorithm, &checked));
    check_cublas(checked.state);
    state.workspace_bytes = checked.workspaceSize;
  } else {
    cublasLtMatmulPreference_t preference = nullptr;
    check_cublas(cublasLtMatmulPreferenceCreate(&preference));
    const size_t workspace_limit = 32 * 1024 * 1024;
    cublasStatus_t status = cublasLtMatmulPreferenceSetAttribute(
        preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &workspace_limit, sizeof(workspace_limit));
    if (status != CUBLAS_STATUS_SUCCESS) {
      cublasLtMatmulPreferenceDestroy(preference);
      check_cublas(status);
    }
    cublasLtMatmulHeuristicResult_t candidates[8]{};
    int candidate_count = 0;
    status = cublasLtMatmulAlgoGetHeuristic(
        state.cublaslt, state.operation, state.a_layout, state.b_layout,
        state.output_layout, state.output_layout, preference, 8, candidates,
        &candidate_count);
    cublasLtMatmulPreferenceDestroy(preference);
    check_cublas(status);
    if (state.configuration >= candidate_count ||
        candidates[state.configuration].state != CUBLAS_STATUS_SUCCESS) {
      throw Failure(APXINF_STATUS_UNSUPPORTED,
                    "cuBLASLt heuristic is unavailable");
    }
    state.algorithm = candidates[state.configuration].algo;
    state.has_algorithm = true;
    state.workspace_bytes = candidates[state.configuration].workspaceSize;
  }

  if (state.workspace_bytes != 0) {
    check_cuda(cudaMalloc(&state.workspace, state.workspace_bytes));
    state.resource_bytes += state.workspace_bytes;
  }
}

cudaError_t launch_cublaslt(State& state,
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
  check_cublas(cublasLtMatmul(
      state.cublaslt, state.operation, alpha_pointer, weight, state.a_layout,
      activation, state.b_layout, beta_pointer, projection,
      state.output_layout, projection, state.output_layout, &state.algorithm,
      state.workspace, state.workspace_bytes, stream));
  return launch_postprocess(state, bindings, projection);
}

}  // namespace apxinf::gemm
