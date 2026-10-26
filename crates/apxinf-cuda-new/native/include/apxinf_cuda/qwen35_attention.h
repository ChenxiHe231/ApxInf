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

#ifdef __cplusplus
}
#endif

