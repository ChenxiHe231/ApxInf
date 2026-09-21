#include "internal.h"

namespace apxinf::norm {
namespace {

constexpr uint32_t kProviderCustom = 1;

// The custom kernels convert through float and write element-wise. E4M3 output
// is not implemented, and BiasThenResidual intentionally supports BF16 only.
bool supports_custom(const Spec& spec) {
  const bool supported_dtype =
      spec.semantic == APXINF_NORM_SEMANTIC_BIAS_THEN_RESIDUAL
          ? spec.dtype == APXINF_DTYPE_BF16
          : (spec.dtype == APXINF_DTYPE_F16 ||
             spec.dtype == APXINF_DTYPE_BF16);
  return supported_dtype &&
         spec.output_dtype == spec.dtype && spec.cols > 0 && spec.rows > 0;
}

AlignmentRequirements natural_alignment(const Spec&) { return {}; }

void one_configuration(const Spec&, std::vector<int>& configurations) {
  configurations.push_back(0);
}

}  // namespace

const ImplementationRegistry& registry(uint32_t semantic) {
  static const ImplementationRegistry general = {
      {kProviderCustom, 1, 1, "custom-norm", 0, true, true, false,
       supports_custom, natural_alignment, one_configuration, launch_custom},
  };
  // Keep the two-rounding contract visible in candidate identity and
  // diagnostics. It shares a provider implementation today but is not an
  // alias of the ordinary bias/residual candidate.
  static const ImplementationRegistry bias_then_residual = {
      {kProviderCustom, 2, 1, "custom-bias-then-residual-bf16", 0, true, true,
       false, supports_custom, natural_alignment, one_configuration,
       launch_custom},
  };
  return semantic == APXINF_NORM_SEMANTIC_BIAS_THEN_RESIDUAL
             ? bias_then_residual
             : general;
}

}  // namespace apxinf::norm
