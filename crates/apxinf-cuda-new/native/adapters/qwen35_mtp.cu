#include "../kernels/qwen35/qwen35_mtp.h"

extern "C" apxinf_status_t apxinf_qwen35_mtp_moe(
    apxinf_runtime_t runtime, const apxinf_qwen35_mtp_moe_spec_t* spec,
    const apxinf_qwen35_mtp_moe_bindings_t* bindings) {
  return apxinf::cuda_new::qwen35_launch::qwen35_mtp_moe(runtime, spec, bindings);
}

extern "C" apxinf_status_t apxinf_qwen35_mtp_accept(
    apxinf_runtime_t runtime, const apxinf_qwen35_mtp_accept_spec_t* spec,
    const apxinf_qwen35_mtp_accept_bindings_t* bindings) {
  return apxinf::cuda_new::qwen35_launch::qwen35_mtp_accept(runtime, spec, bindings);
}
