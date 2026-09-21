#include "internal.h"

namespace apxinf::rope {
namespace {

constexpr uint32_t kProviderCustom = 1;

bool supports_custom(const Spec& spec) {
  // The rotation pairs element i with element i + head_dim/2, so an odd
  // head_dim has no valid pairing.
  return (spec.dtype == APXINF_DTYPE_F16 || spec.dtype == APXINF_DTYPE_BF16) &&
         spec.tokens > 0 && spec.head_dim > 0 && spec.head_dim % 2 == 0 &&
         spec.q_heads > 0 && spec.kv_heads > 0 &&
         spec.q_heads % spec.kv_heads == 0;
}

AlignmentRequirements natural_alignment(const Spec&) { return {}; }

void one_configuration(const Spec&, std::vector<int>& configurations) {
  configurations.push_back(0);
}

}  // namespace

const ImplementationRegistry& registry(uint32_t semantic) {
  static const ImplementationRegistry entries = {
      {kProviderCustom, 1, 1, "custom-rope", 0, true, true, false,
       supports_custom, natural_alignment, one_configuration, launch_custom},
  };
  (void)semantic;
  return entries;
}

}  // namespace apxinf::rope
