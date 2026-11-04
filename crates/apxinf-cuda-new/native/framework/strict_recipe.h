#pragma once

#include <cstdlib>
#include <cstring>

namespace apxinf::framework {

// Opt-in execution constraint used by reproducible benchmarks. Read once per
// process: formal runs must not inherit executions prepared by a tuning run.
// This does not change normal runtime fallback behavior when the flag is off.
inline bool require_recipes() {
  static const bool required = [] {
    const char* value = std::getenv("APXINF_REQUIRE_RECIPES");
    return value != nullptr && value[0] != '\0' &&
           std::strcmp(value, "0") != 0 &&
           std::strcmp(value, "false") != 0 &&
           std::strcmp(value, "off") != 0;
  }();
  return required;
}

}  // namespace apxinf::framework
