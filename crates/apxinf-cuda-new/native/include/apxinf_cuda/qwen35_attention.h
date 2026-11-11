#pragma once

#include "qwen35_attention_types.h"
#include "status.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Append projected K/V to a real FP8-E4M3 cache and evaluate causal GQA for
   the appended token range.  Prefill uses cache_offset=0; decode uses
   tokens=1 and cache_offset equal to the committed prefix length. */
apxinf_status_t apxinf_qwen35_attention_fp8(
    apxinf_runtime_t runtime, const apxinf_qwen35_attention_spec_t* spec,
    const apxinf_qwen35_attention_bindings_t* bindings);

/* Split Q/gate, Q RMSNorm and RoPE with intermediate BF16 rounding. */
apxinf_status_t apxinf_qwen35_query_prepare(
    apxinf_runtime_t runtime, const apxinf_qwen35_query_prepare_spec_t* spec,
    const apxinf_qwen35_query_prepare_bindings_t* bindings);

/* Quantize post-RoPE K and projected V with one online scale per token and KV
   head, then append both payloads and scales to their persistent caches. */
apxinf_status_t apxinf_qwen35_dynamic_kv_append(
    apxinf_runtime_t runtime, const apxinf_qwen35_dynamic_kv_spec_t* spec,
    const apxinf_qwen35_dynamic_kv_bindings_t* bindings);

/* Fused K-Norm, RoPE and online quantization/cache write. */
apxinf_status_t apxinf_qwen35_dynamic_kv_prepare(
    apxinf_runtime_t runtime, const apxinf_qwen35_dynamic_prepare_spec_t* spec,
    const apxinf_qwen35_dynamic_prepare_bindings_t* bindings);

/* Decode one BF16 query row directly from dynamic FP8 K/V.  The kernel never
   materializes a BF16 cache; it converts each cache tile in registers and
   combines split-KV partial softmax states in FP32. */
apxinf_status_t apxinf_qwen35_dynamic_decode_attention(
    apxinf_runtime_t runtime, const apxinf_qwen35_dynamic_decode_spec_t* spec,
    const apxinf_qwen35_dynamic_decode_bindings_t* bindings);

/* Expand the valid dynamic FP8 prefix into caller-owned BF16 buffers for
   prefill attention. */
apxinf_status_t apxinf_qwen35_dynamic_kv_dequant(
    apxinf_runtime_t runtime, const apxinf_qwen35_dynamic_dequant_spec_t* spec,
    const apxinf_qwen35_dynamic_dequant_bindings_t* bindings);

/* Exact-live BF16 suffix-causal attention for the Thor prefill path.
   The caller applies the Qwen output gate to the BF16 result separately. */
apxinf_status_t apxinf_qwen35_suffix_fmha(
    apxinf_runtime_t runtime, const apxinf_qwen35_suffix_fmha_spec_t* spec,
    const apxinf_qwen35_suffix_fmha_bindings_t* bindings);

#ifdef __cplusplus
}
#endif

