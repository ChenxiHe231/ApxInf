#include "internal.h"

#include <iomanip>

namespace apxinf::gemm {

std::string tuning_key(const Spec& spec,
                       const apxinf_gemm_policy_t& policy,
                       int device) {
  cudaDeviceProp properties{};
  check_cuda(cudaGetDeviceProperties(&properties, device));
  int runtime_version = 0;
  int driver_version = 0;
  check_cuda(cudaRuntimeGetVersion(&runtime_version));
  check_cuda(cudaDriverGetVersion(&driver_version));

  std::ostringstream key;
  key << "gemm-recipe-v3|" << APXINF_GEMM_BUILD_ID << '|'
      << APXINF_GEMM_SM << '|' << runtime_version << '|' << driver_version
      << '|' << cublasLtGetVersion() << '|';
  for (const auto byte : properties.uuid.bytes) {
    key << std::hex << std::setw(2) << std::setfill('0')
        << static_cast<int>(static_cast<unsigned char>(byte));
  }
  key << std::dec << '|' << properties.multiProcessorCount << '|'
      << spec.version << '|' << static_cast<uint32_t>(spec.semantic) << '|' << spec.m << '|'
      << spec.n << '|' << spec.k << '|' << spec.a_dtype << '|'
      << spec.b_dtype << '|' << spec.accumulation_dtype << '|'
      << spec.output_dtype << '|' << spec.quantization << '|'
      << spec.a_alignment << '|' << spec.b_alignment << '|'
      << spec.bias_alignment << '|' << spec.a_scales_alignment << '|'
      << spec.b_scales_alignment << '|' << spec.output_alignment;

  uint32_t alpha_bits = 0;
  uint32_t output_scale_bits = 0;
  std::memcpy(&alpha_bits, &spec.alpha, sizeof(alpha_bits));
  std::memcpy(&output_scale_bits, &spec.output_scale,
              sizeof(output_scale_bits));
  key << '|' << alpha_bits << '|' << output_scale_bits << '|'
      << policy.workspace_limit << '|' << policy.graph_safe << '|'
      << policy.deterministic;
  return key.str();
}

}  // namespace apxinf::gemm
