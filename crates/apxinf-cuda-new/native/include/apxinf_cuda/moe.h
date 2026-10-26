#pragma once

#include "moe_types.h"
#include "runtime.h"

#ifdef __cplusplus
extern "C" {
#endif

apxinf_status_t apxinf_moe_route_topk(
    apxinf_runtime_t runtime, const apxinf_moe_route_spec_t* spec,
    const apxinf_moe_route_bindings_t* bindings);

apxinf_status_t apxinf_moe_route_topk_only(
    apxinf_runtime_t runtime, const apxinf_moe_route_spec_t* spec,
    const apxinf_moe_route_bindings_t* bindings);

apxinf_status_t apxinf_moe_combine(
    apxinf_runtime_t runtime, const apxinf_moe_combine_spec_t* spec,
    const apxinf_moe_combine_bindings_t* bindings);

apxinf_status_t apxinf_moe_add_shared(
    apxinf_runtime_t runtime, const apxinf_moe_combine_spec_t* spec,
    const apxinf_moe_add_shared_bindings_t* bindings);

apxinf_status_t apxinf_moe_nvfp4_experts(
    apxinf_runtime_t runtime, const apxinf_moe_nvfp4_experts_spec_t* spec,
    const apxinf_moe_nvfp4_experts_bindings_t* bindings);

apxinf_status_t apxinf_moe_edge_nvfp4_workspace_size(
    const apxinf_moe_edge_nvfp4_spec_t* spec, size_t* workspace_bytes);

apxinf_status_t apxinf_moe_edge_nvfp4(
    apxinf_runtime_t runtime, const apxinf_moe_edge_nvfp4_spec_t* spec,
    const apxinf_moe_edge_nvfp4_bindings_t* bindings);

apxinf_status_t apxinf_moe_edge_bf16_decode_workspace_size(
    const apxinf_moe_edge_bf16_decode_spec_t* spec, size_t* workspace_bytes);

apxinf_status_t apxinf_moe_edge_bf16_decode(
    apxinf_runtime_t runtime, const apxinf_moe_edge_bf16_decode_spec_t* spec,
    const apxinf_moe_edge_bf16_decode_bindings_t* bindings);

#ifdef __cplusplus
}
#endif

