#pragma once

#include "moe_types.h"
#include "runtime.h"

#ifdef __cplusplus
extern "C" {
#endif

apxinf_status_t apxinf_moe_route_topk(
    apxinf_runtime_t runtime, const apxinf_moe_route_spec_t* spec,
    const apxinf_moe_route_bindings_t* bindings);

apxinf_status_t apxinf_moe_combine(
    apxinf_runtime_t runtime, const apxinf_moe_combine_spec_t* spec,
    const apxinf_moe_combine_bindings_t* bindings);

apxinf_status_t apxinf_moe_nvfp4_experts(
    apxinf_runtime_t runtime, const apxinf_moe_nvfp4_experts_spec_t* spec,
    const apxinf_moe_nvfp4_experts_bindings_t* bindings);

#ifdef __cplusplus
}
#endif

