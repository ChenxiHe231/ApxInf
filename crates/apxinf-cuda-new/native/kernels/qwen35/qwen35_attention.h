#pragma once
#include "../../include/apxinf_cuda/qwen35_attention.h"
namespace apxinf::cuda_new::qwen35_launch {
apxinf_status_t qwen35_attention_fp8_draft(
    apxinf_runtime_t runtime, const apxinf_qwen35_attention_spec_t* spec,
    const apxinf_qwen35_attention_bindings_t* bindings, float* workspace,
    size_t workspace_elements);
apxinf_status_t qwen35_attention_fp8(
    apxinf_runtime_t runtime, const apxinf_qwen35_attention_spec_t* spec,
    const apxinf_qwen35_attention_bindings_t* bindings);
apxinf_status_t qwen35_static_kv_append(
    apxinf_runtime_t runtime, const apxinf_qwen35_attention_spec_t* spec,
    const apxinf_qwen35_static_kv_bindings_t* bindings);
apxinf_status_t qwen35_query_prepare(
    apxinf_runtime_t runtime, const apxinf_qwen35_query_prepare_spec_t* spec,
    const apxinf_qwen35_query_prepare_bindings_t* bindings);
apxinf_status_t qwen35_dynamic_kv_append(
    apxinf_runtime_t runtime, const apxinf_qwen35_dynamic_kv_spec_t* spec,
    const apxinf_qwen35_dynamic_kv_bindings_t* bindings);
apxinf_status_t qwen35_dynamic_kv_prepare(
    apxinf_runtime_t runtime, const apxinf_qwen35_dynamic_prepare_spec_t* spec,
    const apxinf_qwen35_dynamic_prepare_bindings_t* bindings);
apxinf_status_t qwen35_dynamic_decode_attention(
    apxinf_runtime_t runtime, const apxinf_qwen35_dynamic_decode_spec_t* spec,
    const apxinf_qwen35_dynamic_decode_bindings_t* bindings);
apxinf_status_t qwen35_dynamic_kv_dequant(
    apxinf_runtime_t runtime, const apxinf_qwen35_dynamic_dequant_spec_t* spec,
    const apxinf_qwen35_dynamic_dequant_bindings_t* bindings);
}
