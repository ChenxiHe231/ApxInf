#pragma once
#include "../../include/apxinf_cuda/qwen35_mtp.h"
namespace apxinf::cuda_new::qwen35_launch {
apxinf_status_t qwen35_mtp_moe(
    apxinf_runtime_t runtime, const apxinf_qwen35_mtp_moe_spec_t* spec,
    const apxinf_qwen35_mtp_moe_bindings_t* bindings);
apxinf_status_t qwen35_mtp_accept(
    apxinf_runtime_t runtime, const apxinf_qwen35_mtp_accept_spec_t* spec,
    const apxinf_qwen35_mtp_accept_bindings_t* bindings);
}
