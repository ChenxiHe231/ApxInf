// SPDX-License-Identifier: Apache-2.0

#include "../../include/apxinf_cuda/moe.h"
#include "../../framework/runtime_internal.h"

using apxinf::framework::Failure;

namespace {
[[noreturn]] void unsupported() {
  throw Failure(
      APXINF_STATUS_UNSUPPORTED,
      "TensorRT-Edge-LLM NVFP4 MoE is available only in the aarch64 SM110 build");
}
}  // namespace

extern "C" apxinf_status_t apxinf_moe_edge_nvfp4_workspace_size(
    const apxinf_moe_edge_nvfp4_spec_t*, size_t*) {
  return apxinf::framework::abi_boundary([] { unsupported(); });
}

extern "C" apxinf_status_t apxinf_moe_edge_nvfp4(
    apxinf_runtime_t, const apxinf_moe_edge_nvfp4_spec_t*,
    const apxinf_moe_edge_nvfp4_bindings_t*) {
  return apxinf::framework::abi_boundary([] { unsupported(); });
}
