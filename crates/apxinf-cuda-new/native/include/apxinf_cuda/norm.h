#pragma once

#include "norm_types.h"

#ifdef __cplusplus
extern "C" {
#endif

apxinf_status_t apxinf_norm_launch(
    apxinf_runtime_t runtime, const apxinf_norm_spec_t* spec,
    const apxinf_norm_policy_t* policy,
    const apxinf_norm_bindings_t* bindings);
apxinf_status_t apxinf_norm_test_validate_candidates(
    apxinf_runtime_t runtime, const apxinf_norm_spec_t* spec,
    const apxinf_norm_policy_t* policy,
    const apxinf_norm_bindings_t* bindings, const float* expected_output,
    uint64_t expected_output_len);

#ifdef __cplusplus
}
#endif
