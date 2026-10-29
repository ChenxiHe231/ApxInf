#pragma once

#include <stdint.h>

#include "runtime.h"

/* Exact Qwen3.5-35B-A3B full-attention geometry.  This is intentionally not
   a generic attention descriptor: accepting another geometry here would make
   it too easy to run a plausible but wrong cache layout. */
#define APXINF_QWEN35_ATTENTION_SPEC_VERSION 2u
#define APXINF_QWEN35_Q_HEADS 16u
#define APXINF_QWEN35_KV_HEADS 2u
#define APXINF_QWEN35_HEAD_DIM 256u
#define APXINF_QWEN35_ROTARY_DIM 64u
#define APXINF_QWEN35_ATTENTION_MAX_SPLITS 32u
#define APXINF_QWEN35_DYNAMIC_KV_SPEC_VERSION 1u
#define APXINF_QWEN35_DYNAMIC_DECODE_SPEC_VERSION 1u
#define APXINF_QWEN35_DECODE_SPLIT_TOKENS 256u

typedef struct {
  uint32_t version;
  uint32_t tokens;
  uint32_t cache_offset;
  uint32_t cache_capacity;
  float rope_theta;
  float rms_epsilon;
  /* TensorRT Edge-LLM scale_quant_orig values.  Checkpoint amax conversion
     happens once in the model loader: scale = checkpoint_amax / 448. */
  float key_scale;
  float value_scale;
} apxinf_qwen35_attention_spec_t;

typedef struct {
  /* BF16 projections: [T,16,512], [T,2,256], [T,2,256]. */
  const void* projected_query_gate;
  const void* projected_key;
  const void* projected_value;
  /* BF16 zero-centered RMSNorm weights, each [256]. */
  const void* query_norm;
  const void* key_norm;
  /* I32 [3,T], channel-major temporal/height/width positions. */
  const void* positions;

  /* F8E4M3 caches, independently allocated [capacity,2,256]. */
  void* key_cache;
  void* value_cache;

  /* BF16 caller-owned scratch/output, each [T,16,256]. */
  void* query;
  void* gate;
  void* output;
  /* FP32 split attention scores [T,16,capacity]. */
  void* scores;
  /* FP32 split-KV accumulation [T,2,32,8,256]. */
  void* partial_output;
  /* FP32 split maximum/denominator pairs [T,2,32,8,2]. */
  void* partial_stats;
  apxinf_cuda_stream_t stream;
} apxinf_qwen35_attention_bindings_t;

typedef struct {
  uint32_t version;
  uint32_t tokens;
  uint32_t cache_offset;
  uint32_t cache_capacity;
} apxinf_qwen35_dynamic_kv_spec_t;

typedef struct {
  /* Post-K-Norm, post-RoPE BF16 K and projected BF16 V: [T,2,256]. */
  const void* key;
  const void* value;
  /* F8E4M3 [capacity,2,256] plus FP32 per-token-per-head scales
     [capacity,2]. */
  void* key_cache;
  void* value_cache;
  void* key_scales;
  void* value_scales;
  apxinf_cuda_stream_t stream;
} apxinf_qwen35_dynamic_kv_bindings_t;

typedef struct {
  uint32_t version;
  uint32_t valid_tokens;
  uint32_t cache_capacity;
  uint32_t max_splits;
} apxinf_qwen35_dynamic_decode_spec_t;

typedef struct {
  /* BF16 [1,16,256]. */
  const void* query;
  /* F8E4M3 [capacity,2,256] and FP32 [capacity,2]. */
  const void* key_cache;
  const void* value_cache;
  const void* key_scales;
  const void* value_scales;
  /* FP32 workspaces [16,max_splits], [16,max_splits] and
     [16,max_splits,256]. */
  void* partial_max;
  void* partial_sum;
  void* partial_output;
  /* BF16 [1,16,256]. */
  void* output;
  apxinf_cuda_stream_t stream;
} apxinf_qwen35_dynamic_decode_bindings_t;

typedef struct {
  uint32_t version;
  uint32_t valid_tokens;
  uint32_t cache_capacity;
} apxinf_qwen35_dynamic_dequant_spec_t;

typedef struct {
  const void* key_cache;
  const void* value_cache;
  const void* key_scales;
  const void* value_scales;
  /* BF16 [1,valid_tokens,2,256]. */
  void* key;
  void* value;
  apxinf_cuda_stream_t stream;
} apxinf_qwen35_dynamic_dequant_bindings_t;

