#pragma once

#include "../../include/apxinf_cuda/gemm.h"
#include "../../include/apxinf_cuda/gemm_bias.h"
#include "../../include/apxinf_cuda/gemm_gelu.h"
#include "../../include/apxinf_cuda/gemm_geglu.h"

#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace apxinf::gemm {

enum class Semantic : uint32_t {
  kGemm = 0,
  kGemmBiasGelu = 1,
  kGemmGeglu = 2,
  kGemmBias = 3,
};

constexpr Semantic APXINF_GEMM_SEMANTIC_GEMM = Semantic::kGemm;
constexpr Semantic APXINF_GEMM_SEMANTIC_GEMM_BIAS_GELU =
    Semantic::kGemmBiasGelu;
constexpr Semantic APXINF_GEMM_SEMANTIC_GEMM_GEGLU = Semantic::kGemmGeglu;
constexpr Semantic APXINF_GEMM_SEMANTIC_GEMM_BIAS = Semantic::kGemmBias;

struct Spec : apxinf_gemm_spec_t {
  Semantic semantic = Semantic::kGemm;
};

struct Failure : std::runtime_error {
  apxinf_status_t status;

  Failure(apxinf_status_t status, const std::string& message)
      : std::runtime_error(message), status(status) {}
};

void set_last_error(const std::string& message);
void clear_last_error();

template <class Function>
apxinf_status_t abi_boundary(Function&& function) {
  try {
    function();
    clear_last_error();
    return APXINF_STATUS_OK;
  } catch (const Failure& failure) {
    set_last_error(failure.what());
    return failure.status;
  } catch (const std::exception& exception) {
    set_last_error(exception.what());
    return APXINF_STATUS_INTERNAL_ERROR;
  } catch (...) {
    set_last_error("unknown native exception");
    return APXINF_STATUS_INTERNAL_ERROR;
  }
}

inline void check_cuda(cudaError_t status) {
  if (status != cudaSuccess) {
    throw Failure(APXINF_STATUS_CUDA_ERROR, cudaGetErrorString(status));
  }
}

inline void check_cublas(cublasStatus_t status) {
  if (status != CUBLAS_STATUS_SUCCESS) {
    throw Failure(APXINF_STATUS_PROVIDER_ERROR,
                  "cuBLAS status " + std::to_string(status));
  }
}

inline size_t dtype_bytes(uint32_t dtype) {
  if (dtype == APXINF_DTYPE_F32 || dtype == APXINF_DTYPE_I32) {
    return 4;
  }
  if (dtype == APXINF_DTYPE_E4M3 || dtype == APXINF_DTYPE_I8) {
    return 1;
  }
  return 2;
}

inline bool has_row_channel_scales(const Spec& spec) {
  return spec.quantization == APXINF_GEMM_QUANT_FP8_ROW_CHANNEL ||
         spec.quantization == APXINF_GEMM_QUANT_W8A8_ROW_CHANNEL;
}

struct State;
using LaunchFn = cudaError_t (*)(State&, const apxinf_gemm_bindings_t&);

struct AlignmentRequirements {
  uint32_t a = 1;
  uint32_t b = 1;
  uint32_t bias = 1;
  uint32_t a_scales = 1;
  uint32_t b_scales = 1;
  uint32_t output = 1;
};

using AlignmentFn = AlignmentRequirements (*)(const Spec&);

struct Implementation {
  uint32_t provider_id;
  uint32_t implementation_id;
  uint32_t implementation_version;
  const char* name;
  bool graph_safe;
  bool deterministic;
  bool (*supports)(const Spec&);
  AlignmentFn alignment_requirements;
  void (*enumerate_configs)(const Spec&, std::vector<int>&);
  void (*create_state)(State&);
  LaunchFn launch;
};

inline bool supports_alignment(const Implementation& implementation,
                               const Spec& spec) {
  const auto required = implementation.alignment_requirements(spec);
  return spec.a_alignment >= required.a &&
         spec.b_alignment >= required.b &&
         spec.bias_alignment >= (spec.semantic == Semantic::kGemmBias ||
                                         spec.semantic == Semantic::kGemmBiasGelu
                                     ? required.bias
                                     : 0) &&
         spec.a_scales_alignment >=
             (has_row_channel_scales(spec) ? required.a_scales : 0) &&
         spec.b_scales_alignment >=
             (has_row_channel_scales(spec) ? required.b_scales : 0) &&
         spec.output_alignment >= required.output;
}

struct Recipe {
  uint32_t provider_id;
  uint32_t implementation_id;
  uint32_t implementation_version;
  int32_t configuration;
};

struct TuningKeys {
  // A recipe under this key is fully tuned for an equivalent performance
  // profile and may be restored directly.
  std::string performance;
  // A recipe under this key only proves execution compatibility. It may be
  // tried first, but all candidates must still be tuned on this device.
  std::string compatible_hint;
};

struct State {
  Spec spec{};
  int configuration = 0;
  int device = 0;
  const Implementation* implementation = nullptr;
  cublasLtHandle_t cublaslt = nullptr;
  cublasHandle_t cublas = nullptr;
  cublasLtMatmulDesc_t operation = nullptr;
  cublasLtMatrixLayout_t a_layout = nullptr;
  cublasLtMatrixLayout_t b_layout = nullptr;
  cublasLtMatrixLayout_t output_layout = nullptr;
  cublasLtMatmulAlgo_t algorithm{};
  bool has_algorithm = false;
  void* workspace = nullptr;
  void* projection = nullptr;
  void* unpack_a = nullptr;
  void* unpack_b = nullptr;
  size_t workspace_bytes = 0;
  size_t resource_bytes = 0;
  uint32_t projection_dtype = APXINF_DTYPE_F16;

  ~State();
};

const std::vector<Implementation>& registry(Semantic semantic);
void prepare_cublas(State& state);
cudaError_t launch_cublas(State& state, const apxinf_gemm_bindings_t& bindings);
void prepare_cublaslt(State& state);
cudaError_t launch_cublaslt(State& state, const apxinf_gemm_bindings_t& bindings);
void prepare_cutlass(State& state);
cudaError_t launch_cutlass(State& state, const apxinf_gemm_bindings_t& bindings);

void allocate_common_resources(State& state, bool native_fp8);
cudaError_t launch_postprocess(State& state,
                               const apxinf_gemm_bindings_t& bindings,
                               void* projection);

TuningKeys tuning_keys(const Spec& spec,
                       const apxinf_gemm_policy_t& policy,
                       int device);
std::string read_recipe(const std::string& directory, const std::string& key);
void write_recipe(const std::string& directory,
                  const std::string& key,
                  const std::string& recipe);
std::shared_ptr<State> prepare(const Implementation& implementation,
                               int configuration,
                               const Spec& spec,
                               const apxinf_gemm_policy_t& policy,
                               int device,
                               const State* blueprint = nullptr);
std::shared_ptr<State> tune(const Spec& spec,
                            const apxinf_gemm_policy_t& policy,
                            const apxinf_gemm_tuning_bindings_t& bindings,
                            int device,
                            std::string& report,
                            const Recipe* preferred = nullptr);

}  // namespace apxinf::gemm

struct apxinf_runtime {
  int device = 0;
  std::mutex gemm_mutex;
  std::map<std::string, apxinf::gemm::Recipe> gemm_recipes;
  std::map<std::string, std::shared_ptr<apxinf::gemm::State>> gemm_plans;
};

struct apxinf_gemm_plan {
  std::shared_ptr<apxinf::gemm::State> state;
  apxinf_gemm_policy_t policy{};
  std::string summary;
};

struct apxinf_gemm_instance {
  std::shared_ptr<apxinf::gemm::State> state;
  apxinf_gemm_bindings_t bindings{};
};
