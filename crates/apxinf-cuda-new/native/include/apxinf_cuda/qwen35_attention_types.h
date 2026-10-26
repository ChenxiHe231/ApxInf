#pragma once

#include <stdint.h>

#include "runtime.h"

/* Exact Qwen3.5-35B-A3B full-attention geometry.  This is intentionally not
   a generic attention descriptor: accepting another geometry here would make
   it too easy to run a plausible but wrong cache layout. */
#define APXINF_QWEN35_ATTENTION_SPEC_VERSION 1u
#define APXINF_QWEN35_Q_HEADS 16u
#define APXINF_QWEN35_KV_HEADS 2u
#define APXINF_QWEN35_HEAD_DIM 256u
#define APXINF_QWEN35_ROTARY_DIM 64u

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
  apxinf_cuda_stream_t stream;
} apxinf_qwen35_attention_bindings_t;

