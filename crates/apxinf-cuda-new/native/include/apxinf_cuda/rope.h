#pragma once

#include "rope_types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct apxinf_rope_execution* apxinf_rope_execution_t;

apxinf_status_t apxinf_rope_prepare(
    apxinf_runtime_t runtime, const apxinf_rope_spec_t* spec,
    const apxinf_rope_policy_t* policy,
    const apxinf_rope_bindings_t* bindings,
    apxinf_rope_execution_t* execution);
apxinf_status_t apxinf_rope_enqueue(apxinf_rope_execution_t execution);
void apxinf_rope_destroy(apxinf_rope_execution_t execution);
const char* apxinf_rope_summary(apxinf_rope_execution_t execution);

#ifdef __cplusplus
}
#endif
