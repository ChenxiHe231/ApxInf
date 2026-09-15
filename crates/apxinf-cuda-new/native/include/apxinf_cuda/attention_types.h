#pragma once

#include <stdint.h>

#include "gemm_types.h"

typedef enum {
  APXINF_ATTENTION_MASK_NONE = 0,
  APXINF_ATTENTION_MASK_CAUSAL = 1,
} apxinf_attention_mask_t;

typedef struct {
  uint32_t version;
  uint32_t dtype;
  uint32_t output_dtype;
  uint32_t mask;
  uint32_t q_alignment;
  uint32_t k_alignment;
  uint32_t v_alignment;
  uint32_t output_alignment;
  int64_t batch;
  int64_t query_tokens;
  int64_t key_tokens;
  int64_t query_heads;
  int64_t kv_heads;
  int64_t head_dim;
  uint32_t scale_is_default;
} apxinf_attention_spec_t;

typedef apxinf_tuning_policy_t apxinf_attention_policy_t;

typedef struct {
  const void* query;
  const void* key;
  const void* value;
  void* output;
  apxinf_cuda_stream_t stream;
  float scale;
} apxinf_attention_bindings_t;
