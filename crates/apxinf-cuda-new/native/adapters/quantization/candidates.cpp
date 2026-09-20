#include "internal.h"

namespace apxinf::quantization {
namespace {

constexpr uint32_t kProviderPorted = 1;

bool supports_ported(const Spec& spec) {
  switch (spec.semantic) {
    case APXINF_QUANTIZATION_SEMANTIC_FIXED_E4M3:
      return (spec.input_dtype == APXINF_DTYPE_F16 ||
              spec.input_dtype == APXINF_DTYPE_BF16) &&
             spec.output_dtype == APXINF_DTYPE_E4M3 &&
             spec.scale_dtype == APXINF_DTYPE_F32 &&
             spec.input_cols == spec.output_cols;
    case APXINF_QUANTIZATION_SEMANTIC_ROWWISE_E4M3:
      return spec.input_dtype == APXINF_DTYPE_BF16 &&
             spec.output_dtype == APXINF_DTYPE_E4M3 &&
             spec.scale_dtype == APXINF_DTYPE_F32 &&
             spec.output_cols >= spec.input_cols;
    case APXINF_QUANTIZATION_SEMANTIC_CAST_F16_BF16:
      return spec.input_dtype == APXINF_DTYPE_F16 &&
             spec.output_dtype == APXINF_DTYPE_BF16 &&
             spec.scale_dtype == APXINF_DTYPE_F32 &&
             spec.input_cols == spec.output_cols;
    case APXINF_QUANTIZATION_SEMANTIC_SLICE_BF16:
      return spec.input_dtype == APXINF_DTYPE_BF16 &&
             spec.output_dtype == APXINF_DTYPE_BF16 &&
             spec.scale_dtype == APXINF_DTYPE_F32 &&
             spec.output_cols <= spec.input_cols;
    case APXINF_QUANTIZATION_SEMANTIC_ROWWISE_I8:
      return spec.input_dtype == APXINF_DTYPE_BF16 &&
             spec.output_dtype == APXINF_DTYPE_I8 &&
             spec.scale_dtype == APXINF_DTYPE_F32 &&
             spec.input_cols == spec.output_cols;
    default:
      return false;
  }
}

AlignmentRequirements natural_alignment(const Spec&) { return {}; }

void one_configuration(const Spec&, std::vector<int>& configurations) {
  configurations.push_back(0);
}

}  // namespace

const ImplementationRegistry& registry(uint32_t semantic) {
  static const ImplementationRegistry entries = {
      {kProviderPorted, 1, 1, "ported-quantization", 0, true, true, false,
       supports_ported, natural_alignment, ported_resource_requirements,
       one_configuration, prepare_ported, launch_ported, destroy_ported},
  };
  (void)semantic;
  return entries;
}

}  // namespace apxinf::quantization
