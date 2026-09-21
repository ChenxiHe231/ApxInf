#pragma once

#include "pointwise_types.h"

#ifdef __cplusplus
extern "C" {
#endif

apxinf_status_t apxinf_pointwise_launch(
    apxinf_runtime_t runtime, const apxinf_pointwise_spec_t* spec,
    const apxinf_pointwise_policy_t* policy,
    const apxinf_pointwise_bindings_t* bindings);
apxinf_status_t apxinf_pointwise_test_validate_candidates(
    apxinf_runtime_t runtime, const apxinf_pointwise_spec_t* spec,
    const apxinf_pointwise_policy_t* policy,
    const apxinf_pointwise_bindings_t* bindings, const float* expected_output,
    uint64_t expected_output_len);

#ifdef __cplusplus
}
#endif
