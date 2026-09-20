#pragma once

#include "quantization_types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct apxinf_quantization_execution*
    apxinf_quantization_execution_t;

apxinf_status_t apxinf_quantization_prepare(
    apxinf_runtime_t runtime, const apxinf_quantization_spec_t* spec,
    const apxinf_quantization_policy_t* policy,
    const apxinf_quantization_bindings_t* bindings,
    apxinf_quantization_execution_t* execution);
apxinf_status_t apxinf_quantization_enqueue(
    apxinf_quantization_execution_t execution);
void apxinf_quantization_destroy(apxinf_quantization_execution_t execution);
const char* apxinf_quantization_summary(
    apxinf_quantization_execution_t execution);

#ifdef __cplusplus
}
#endif
