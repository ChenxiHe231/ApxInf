#include "../kernels/qwen35/moe.h"

extern "C" apxinf_status_t apxinf_moe_route_topk(
    apxinf_runtime_t runtime, const apxinf_moe_route_spec_t* spec,
    const apxinf_moe_route_bindings_t* bindings) {
  return apxinf::cuda_new::qwen35_launch::moe_route_topk(runtime, spec, bindings);
}

extern "C" apxinf_status_t apxinf_moe_route_topk_only(
    apxinf_runtime_t runtime, const apxinf_moe_route_spec_t* spec,
    const apxinf_moe_route_bindings_t* bindings) {
  return apxinf::cuda_new::qwen35_launch::moe_route_topk_only(runtime, spec, bindings);
}

extern "C" apxinf_status_t apxinf_moe_combine(
    apxinf_runtime_t runtime, const apxinf_moe_combine_spec_t* spec,
    const apxinf_moe_combine_bindings_t* bindings) {
  return apxinf::cuda_new::qwen35_launch::moe_combine(runtime, spec, bindings);
}

extern "C" apxinf_status_t apxinf_moe_add_shared(
    apxinf_runtime_t runtime, const apxinf_moe_combine_spec_t* spec,
    const apxinf_moe_add_shared_bindings_t* bindings) {
  return apxinf::cuda_new::qwen35_launch::moe_add_shared(runtime, spec, bindings);
}

extern "C" apxinf_status_t apxinf_moe_nvfp4_experts(
    apxinf_runtime_t runtime, const apxinf_moe_nvfp4_experts_spec_t* spec,
    const apxinf_moe_nvfp4_experts_bindings_t* bindings) {
  return apxinf::cuda_new::qwen35_launch::moe_nvfp4_experts(runtime, spec, bindings);
}
