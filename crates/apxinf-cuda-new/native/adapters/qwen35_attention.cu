#include "../kernels/qwen35/qwen35_attention.h"

extern "C" apxinf_status_t apxinf_qwen35_attention_fp8_draft(
    apxinf_runtime_t runtime, const apxinf_qwen35_attention_spec_t* spec,
    const apxinf_qwen35_attention_bindings_t* bindings, float* workspace,
    size_t workspace_elements) {
  return apxinf::cuda_new::qwen35_launch::qwen35_attention_fp8_draft(runtime, spec, bindings, workspace, workspace_elements);
}

extern "C" apxinf_status_t apxinf_qwen35_attention_fp8(
    apxinf_runtime_t runtime, const apxinf_qwen35_attention_spec_t* spec,
    const apxinf_qwen35_attention_bindings_t* bindings) {
  return apxinf::cuda_new::qwen35_launch::qwen35_attention_fp8(runtime, spec, bindings);
}

extern "C" apxinf_status_t apxinf_qwen35_static_kv_append(
    apxinf_runtime_t runtime, const apxinf_qwen35_attention_spec_t* spec,
    const apxinf_qwen35_static_kv_bindings_t* bindings) {
  return apxinf::cuda_new::qwen35_launch::qwen35_static_kv_append(runtime, spec, bindings);
}

extern "C" apxinf_status_t apxinf_qwen35_query_prepare(
    apxinf_runtime_t runtime, const apxinf_qwen35_query_prepare_spec_t* spec,
    const apxinf_qwen35_query_prepare_bindings_t* bindings) {
  return apxinf::cuda_new::qwen35_launch::qwen35_query_prepare(runtime, spec, bindings);
}

extern "C" apxinf_status_t apxinf_qwen35_dynamic_kv_append(
    apxinf_runtime_t runtime, const apxinf_qwen35_dynamic_kv_spec_t* spec,
    const apxinf_qwen35_dynamic_kv_bindings_t* bindings) {
  return apxinf::cuda_new::qwen35_launch::qwen35_dynamic_kv_append(runtime, spec, bindings);
}

extern "C" apxinf_status_t apxinf_qwen35_dynamic_kv_prepare(
    apxinf_runtime_t runtime, const apxinf_qwen35_dynamic_prepare_spec_t* spec,
    const apxinf_qwen35_dynamic_prepare_bindings_t* bindings) {
  return apxinf::cuda_new::qwen35_launch::qwen35_dynamic_kv_prepare(runtime, spec, bindings);
}

extern "C" apxinf_status_t apxinf_qwen35_dynamic_decode_attention(
    apxinf_runtime_t runtime, const apxinf_qwen35_dynamic_decode_spec_t* spec,
    const apxinf_qwen35_dynamic_decode_bindings_t* bindings) {
  return apxinf::cuda_new::qwen35_launch::qwen35_dynamic_decode_attention(runtime, spec, bindings);
}

extern "C" apxinf_status_t apxinf_qwen35_dynamic_kv_dequant(
    apxinf_runtime_t runtime, const apxinf_qwen35_dynamic_dequant_spec_t* spec,
    const apxinf_qwen35_dynamic_dequant_bindings_t* bindings) {
  return apxinf::cuda_new::qwen35_launch::qwen35_dynamic_kv_dequant(runtime, spec, bindings);
}
