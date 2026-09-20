#pragma once

#include "norm_types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct apxinf_norm_execution* apxinf_norm_execution_t;

apxinf_status_t apxinf_norm_prepare(
    apxinf_runtime_t runtime, const apxinf_norm_spec_t* spec,
    const apxinf_norm_policy_t* policy,
    const apxinf_norm_bindings_t* bindings,
    apxinf_norm_execution_t* execution);
apxinf_status_t apxinf_norm_enqueue(apxinf_norm_execution_t execution);
void apxinf_norm_destroy(apxinf_norm_execution_t execution);
const char* apxinf_norm_summary(apxinf_norm_execution_t execution);
apxinf_status_t apxinf_norm_test_validate_candidates(
    apxinf_runtime_t runtime, const apxinf_norm_spec_t* spec,
    const apxinf_norm_policy_t* policy,
    const apxinf_norm_bindings_t* bindings, const float* expected_output,
    uint64_t expected_output_len);

#ifdef __cplusplus
}
#endif
