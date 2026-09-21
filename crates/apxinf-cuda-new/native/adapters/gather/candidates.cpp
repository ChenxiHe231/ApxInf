#include "internal.h"

namespace apxinf::gather {
namespace {

constexpr uint32_t kProviderCustom = 1;

bool supports_custom(const Spec& spec) {
  return (spec.dtype == APXINF_DTYPE_F16 || spec.dtype == APXINF_DTYPE_BF16) &&
         spec.rows > 0 && spec.cols > 0;
}

AlignmentRequirements natural_alignment(const Spec&) { return {}; }

void one_configuration(const Spec&, std::vector<int>& configurations) {
  configurations.push_back(0);
}

}  // namespace

const ImplementationRegistry& registry(uint32_t semantic) {
  static const ImplementationRegistry entries = {
      {kProviderCustom, 1, 1, "custom-gather", 0, true, true, false,
       supports_custom, natural_alignment, one_configuration, launch_custom},
  };
  (void)semantic;
  return entries;
}

}  // namespace apxinf::gather
