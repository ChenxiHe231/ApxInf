#pragma once

#include "types.h"
#include "status.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Out-of-place elementwise and activation operators backing the portable
   `apxinf_core::Backend` trait. One fixed implementation each, no tuning:
   plain C-ABI entry points, not registry-backed operators. */

/* output[i] = activation(input[i]); activation is 0 none, 1 gelu_tanh,
   2 silu. `count` is the total element count. */
apxinf_status_t apxinf_elementwise_activation_bf16(const void* input,
                                                   void* output, int64_t count,
                                                   int32_t activation,
                                                   apxinf_cuda_stream_t stream);

/* output[i] = a[i] * b[i]. */
apxinf_status_t apxinf_elementwise_mul_bf16(const void* a, const void* b,
                                            void* output, int64_t count,
                                            apxinf_cuda_stream_t stream);

/* output[i] = a[i] + b[i]. */
apxinf_status_t apxinf_elementwise_add_bf16(const void* a, const void* b,
                                            void* output, int64_t count,
                                            apxinf_cuda_stream_t stream);

/* output[i] = input[i] * factor. */
apxinf_status_t apxinf_elementwise_scale_bf16(const void* input, void* output,
                                              int64_t count, float factor,
                                              apxinf_cuda_stream_t stream);

/* output[r, c] = input[r, c] + bias[c], broadcasting bias over rows. */
apxinf_status_t apxinf_elementwise_add_bias_bf16(const void* input,
                                                 const void* bias, void* output,
                                                 int64_t rows, int64_t cols,
                                                 apxinf_cuda_stream_t stream);

#ifdef __cplusplus
}
#endif
