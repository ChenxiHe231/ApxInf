#pragma once
#include "../../include/apxinf_cuda/moe.h"
namespace apxinf::cuda_new::qwen35_launch {
apxinf_status_t moe_route_topk(
    apxinf_runtime_t runtime, const apxinf_moe_route_spec_t* spec,
    const apxinf_moe_route_bindings_t* bindings);
apxinf_status_t moe_route_topk_only(
    apxinf_runtime_t runtime, const apxinf_moe_route_spec_t* spec,
    const apxinf_moe_route_bindings_t* bindings);
apxinf_status_t moe_combine(
    apxinf_runtime_t runtime, const apxinf_moe_combine_spec_t* spec,
    const apxinf_moe_combine_bindings_t* bindings);
apxinf_status_t moe_add_shared(
    apxinf_runtime_t runtime, const apxinf_moe_combine_spec_t* spec,
    const apxinf_moe_add_shared_bindings_t* bindings);
apxinf_status_t moe_nvfp4_experts(
    apxinf_runtime_t runtime, const apxinf_moe_nvfp4_experts_spec_t* spec,
    const apxinf_moe_nvfp4_experts_bindings_t* bindings);
}
