#pragma once

#include "gather_types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct apxinf_gather_execution* apxinf_gather_execution_t;

apxinf_status_t apxinf_gather_prepare(
    apxinf_runtime_t runtime, const apxinf_gather_spec_t* spec,
    const apxinf_gather_policy_t* policy,
    const apxinf_gather_bindings_t* bindings,
    apxinf_gather_execution_t* execution);
apxinf_status_t apxinf_gather_enqueue(apxinf_gather_execution_t execution);
void apxinf_gather_destroy(apxinf_gather_execution_t execution);
const char* apxinf_gather_summary(apxinf_gather_execution_t execution);

#ifdef __cplusplus
}
#endif
