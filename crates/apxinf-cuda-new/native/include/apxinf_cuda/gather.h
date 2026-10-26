#pragma once

#include "gather_types.h"

#ifdef __cplusplus
extern "C" {
#endif

apxinf_status_t apxinf_gather_embedding(
    const void* table, const void* ids, void* output, int64_t tokens,
    int64_t hidden, int64_t vocab, apxinf_cuda_stream_t stream);

apxinf_status_t apxinf_gather_launch(
    apxinf_runtime_t runtime, const apxinf_gather_spec_t* spec,
    const apxinf_gather_bindings_t* bindings);

#ifdef __cplusplus
}
#endif
