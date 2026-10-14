#pragma once

#include "types.h"
#include "status.h"

#ifdef __cplusplus
extern "C" {
#endif

apxinf_status_t apxinf_model_embedding_gather(const void* table,
                                              const void* ids, void* output,
                                              int64_t tokens, int64_t hidden,
                                              int64_t vocab,
                                              apxinf_cuda_stream_t stream);

apxinf_status_t apxinf_model_argmax_bf16(const void* logits, void* index,
                                         int64_t count,
                                         apxinf_cuda_stream_t stream);

apxinf_status_t apxinf_model_cast_f32_to_bf16(
    const void* input, void* output, int64_t count,
    apxinf_cuda_stream_t stream);

apxinf_status_t apxinf_model_gelu_exact_bf16(
    const void* input, void* output, int64_t count,
    apxinf_cuda_stream_t stream);

apxinf_status_t apxinf_model_replace_rows_bf16(
    const void* base, const void* replacement, const void* row_map,
    void* output, int64_t rows, int64_t columns, int64_t replacement_rows,
    apxinf_cuda_stream_t stream);

#ifdef __cplusplus
}
#endif
