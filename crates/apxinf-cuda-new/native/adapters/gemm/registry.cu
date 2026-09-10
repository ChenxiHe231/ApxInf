#include "internal.h"

namespace apxinf::gemm {
namespace {

constexpr uint32_t kProviderCublas = 1;
constexpr uint32_t kProviderCublasLt = 2;
constexpr uint32_t kProviderCutlass = 3;

bool supports_vendor(const Spec& spec) {
  return spec.a_dtype == spec.b_dtype &&
         ((spec.a_dtype == APXINF_DTYPE_I8 &&
           spec.accumulation_dtype == APXINF_DTYPE_I32) ||
          (spec.a_dtype != APXINF_DTYPE_I8 &&
           spec.accumulation_dtype == APXINF_DTYPE_F32));
}

bool supports_native_fp8(const Spec& spec) {
  return supports_vendor(spec) && spec.a_dtype == APXINF_DTYPE_E4M3 &&
         spec.k % 16 == 0 && spec.n % 16 == 0;
}

AlignmentRequirements vendor_alignment(const Spec&) {
  return {};
}

AlignmentRequirements cublaslt_alignment(const Spec&) {
  // Algorithms returned by the heuristic API may use vectorized accesses;
  // cuBLASLt does not accept the operand pointers when enumerating them, so a
  // conservative 16-byte contract is required for a reusable recipe.
  AlignmentRequirements requirements{};
  requirements.a = 16;
  requirements.b = 16;
  requirements.output = 16;
  return requirements;
}

#ifdef APXINF_GEMM_CUTLASS
AlignmentRequirements cutlass_fp8_alignment(const Spec&) {
  AlignmentRequirements requirements{};
  requirements.a = 16;
  requirements.b = 16;
  requirements.output = 16;
  return requirements;
}

AlignmentRequirements cutlass_geglu_alignment(const Spec& spec) {
  AlignmentRequirements requirements{};
  if (spec.a_dtype == APXINF_DTYPE_E4M3) {
    requirements.a = 16;
    requirements.output = 8;
  } else {
    requirements.a = 32;
    requirements.output = 16;
  }
  return requirements;
}
#endif

void one_configuration(const Spec&, std::vector<int>& configs) {
  configs.push_back(0);
}

void cublaslt_configurations(const Spec&,
                             std::vector<int>& configs) {
  for (int rank = 0; rank < 8; ++rank) {
    configs.push_back(rank);
  }
}

#ifdef APXINF_GEMM_CUTLASS
bool supports_cutlass_fp8(const Spec& spec) {
  return spec.semantic == APXINF_GEMM_SEMANTIC_GEMM &&
         spec.a_dtype == APXINF_DTYPE_E4M3 &&
         spec.b_dtype == APXINF_DTYPE_E4M3 &&
         spec.output_dtype == APXINF_DTYPE_F16 &&
         spec.quantization == APXINF_GEMM_QUANT_FP8_UNIT_SCALE &&
         spec.n % 16 == 0 &&
         spec.k % 16 == 0 && spec.output_scale == 1.0F;
}

bool supports_cutlass_geglu(const Spec& spec) {
  const bool exact_shape = (spec.m == 522 || spec.m == 533) &&
                           spec.n == 32768 && spec.k == 2048;
  const bool valid_dtype =
      (spec.a_dtype == APXINF_DTYPE_E4M3 &&
       spec.output_dtype == APXINF_DTYPE_E4M3 &&
       spec.quantization == APXINF_GEMM_QUANT_FP8_UNIT_SCALE) ||
      (spec.a_dtype == APXINF_DTYPE_BF16 &&
       spec.output_dtype == APXINF_DTYPE_BF16 && spec.alpha == 1.0F &&
       spec.output_scale == 1.0F &&
       spec.quantization == APXINF_GEMM_QUANT_NONE);
  return exact_shape && valid_dtype &&
         spec.semantic == APXINF_GEMM_SEMANTIC_GEMM_GEGLU &&
         spec.a_dtype == spec.b_dtype;
}

void cutlass_configurations(const Spec&,
                            std::vector<int>& configs) {
  for (int configuration = 0; configuration < 4; ++configuration) {
    configs.push_back(configuration);
  }
}
#endif

}  // namespace

const std::vector<Implementation>& registry(Semantic semantic) {
  static const std::vector<Implementation> vendor_entries = {
      {kProviderCublas, 1, 1, "cublas+custom-epilogue", true, true,
       supports_vendor, vendor_alignment, one_configuration, prepare_cublas, launch_cublas},
      {kProviderCublasLt, 1, 1, "cublasLt+custom-epilogue", true, false,
       supports_vendor, cublaslt_alignment, cublaslt_configurations, prepare_cublaslt,
       launch_cublaslt},
  };
  // Keep GEMM+bias as a separate L3 tuning domain even though its current L1
  // candidates happen to be the same vendor implementations.
  static const std::vector<Implementation> gemm_bias_entries = {
      {kProviderCublas, 1, 1, "cublas+custom-epilogue", true, true,
       supports_vendor, vendor_alignment, one_configuration, prepare_cublas, launch_cublas},
      {kProviderCublasLt, 1, 1, "cublasLt+custom-epilogue", true, false,
       supports_vendor, cublaslt_alignment, cublaslt_configurations, prepare_cublaslt,
       launch_cublaslt},
  };
  static const std::vector<Implementation> gemm_entries = {
      {kProviderCublas, 1, 1, "cublas+custom-epilogue", true, true,
       supports_vendor, vendor_alignment, one_configuration, prepare_cublas, launch_cublas},
      {kProviderCublasLt, 1, 1, "cublasLt+custom-epilogue", true, false,
       supports_vendor, cublaslt_alignment, cublaslt_configurations, prepare_cublaslt,
       launch_cublaslt},
      {kProviderCublasLt, 2, 1, "cublasLt-native-fp8+custom-epilogue",
       true, false, supports_native_fp8, cublaslt_alignment, cublaslt_configurations,
       prepare_cublaslt, launch_cublaslt},
#ifdef APXINF_GEMM_CUTLASS
      {kProviderCutlass, 1, 1, "cutlass-fp8", true, true,
       supports_cutlass_fp8, cutlass_fp8_alignment, cutlass_configurations, prepare_cutlass,
       launch_cutlass},
#endif
  };
  static const std::vector<Implementation> gemm_geglu_entries = {
      {kProviderCublas, 1, 1, "cublas+custom-epilogue", true, true,
       supports_vendor, vendor_alignment, one_configuration, prepare_cublas, launch_cublas},
      {kProviderCublasLt, 1, 1, "cublasLt+custom-epilogue", true, false,
       supports_vendor, cublaslt_alignment, cublaslt_configurations, prepare_cublaslt,
       launch_cublaslt},
#ifdef APXINF_GEMM_CUTLASS
      {kProviderCutlass, 2, 1, "cutlass-dual-geglu", true, true,
       supports_cutlass_geglu, cutlass_geglu_alignment, one_configuration, prepare_cutlass,
       launch_cutlass},
#endif
  };
  switch (semantic) {
    case Semantic::kGemm:
      return gemm_entries;
    case Semantic::kGemmBiasGelu:
      return vendor_entries;
    case Semantic::kGemmGeglu:
      return gemm_geglu_entries;
    case Semantic::kGemmBias:
      return gemm_bias_entries;
  }
  throw Failure(APXINF_STATUS_INTERNAL_ERROR, "unknown GEMM semantic registry");
}

std::shared_ptr<State> prepare(const Implementation& implementation,
                               int configuration,
                               const Spec& spec,
                               const apxinf_gemm_policy_t& policy,
                               int device,
                               const State* blueprint) {
  if (!implementation.supports(spec)) {
    throw Failure(APXINF_STATUS_UNSUPPORTED, "candidate contract mismatch");
  }
  if (!supports_alignment(implementation, spec)) {
    throw Failure(APXINF_STATUS_UNSUPPORTED,
                  "candidate binding alignment mismatch");
  }
  if (policy.graph_safe && !implementation.graph_safe) {
    throw Failure(APXINF_STATUS_UNSUPPORTED,
                  "candidate is not CUDA Graph safe");
  }
  if (policy.deterministic && !implementation.deterministic) {
    throw Failure(APXINF_STATUS_UNSUPPORTED,
                  "candidate is not deterministic");
  }

  auto state = std::make_shared<State>();
  state->spec = spec;
  state->configuration = configuration;
  state->implementation = &implementation;
  state->device = device;
  if (blueprint != nullptr) {
    state->algorithm = blueprint->algorithm;
    state->has_algorithm = blueprint->has_algorithm;
    state->workspace_bytes = blueprint->workspace_bytes;
  }
  check_cuda(cudaSetDevice(device));
  implementation.create_state(*state);
  if (state->resource_bytes > policy.workspace_limit) {
    throw Failure(APXINF_STATUS_UNSUPPORTED, "workspace policy exceeded");
  }
  return state;
}

}  // namespace apxinf::gemm
