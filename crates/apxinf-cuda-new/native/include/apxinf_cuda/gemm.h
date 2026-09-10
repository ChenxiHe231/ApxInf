#pragma once

#include "gemm_types.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct apxinf_gemm_plan* apxinf_gemm_plan_t;
typedef struct apxinf_gemm_instance* apxinf_gemm_instance_t;

apxinf_status_t apxinf_gemm_plan_create(
    apxinf_runtime_t runtime, const apxinf_gemm_spec_t* spec,
    const apxinf_gemm_policy_t* policy,
    const apxinf_gemm_tuning_bindings_t* tuning_bindings,
    apxinf_gemm_plan_t* plan);
apxinf_status_t apxinf_gemm_instance_create(
    apxinf_gemm_plan_t plan, const apxinf_gemm_bindings_t* bindings,
    apxinf_gemm_instance_t* instance);
apxinf_status_t apxinf_gemm_enqueue(apxinf_gemm_instance_t instance);
void apxinf_gemm_instance_destroy(apxinf_gemm_instance_t instance);
void apxinf_gemm_plan_destroy(apxinf_gemm_plan_t plan);
const char* apxinf_gemm_plan_summary(apxinf_gemm_plan_t plan);

#ifdef __cplusplus
}
#endif
