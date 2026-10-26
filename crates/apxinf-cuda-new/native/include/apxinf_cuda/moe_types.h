#pragma once

#include <stdint.h>

#include "gemm_types.h"

#define APXINF_MOE_ROUTE_SPEC_VERSION 1u
#define APXINF_MOE_COMBINE_SPEC_VERSION 1u
#define APXINF_MOE_NVFP4_EXPERTS_SPEC_VERSION 1u

typedef struct {
  uint32_t version;
  uint32_t logits_dtype;
  uint32_t tokens;
  uint32_t experts;
  uint32_t top_k;
  uint32_t renormalize;
} apxinf_moe_route_spec_t;

typedef struct {
  const void* logits;
  int32_t* expert_ids;
  float* expert_weights;
  int32_t* assignment_expert_ids;
  int32_t* assignment_token_ids;
  int32_t* assignment_slots;
  float* assignment_weights;
  int32_t* expert_offsets;
  apxinf_cuda_stream_t stream;
} apxinf_moe_route_bindings_t;

typedef struct {
  uint32_t version;
  uint32_t dtype;
  uint32_t tokens;
  uint32_t routes;
  uint32_t hidden;
} apxinf_moe_combine_spec_t;

typedef struct {
  const void* assignment_output;
  const int32_t* assignment_token_ids;
  const float* assignment_weights;
  const void* shared_output;
  const void* shared_gate_logits;
  void* output;
  apxinf_cuda_stream_t stream;
} apxinf_moe_combine_bindings_t;

typedef struct {
  uint32_t version;
  uint32_t dtype;
  uint32_t tokens;
  uint32_t routes;
  uint32_t experts;
  uint32_t hidden;
  uint32_t intermediate;
  uint32_t block_size;
  uint32_t scale_layout;
} apxinf_moe_nvfp4_experts_spec_t;

typedef struct {
  const void* input;
  const int32_t* assignment_expert_ids;
  const int32_t* assignment_token_ids;

  const void* expert_gate_weight;
  const void* expert_gate_scales;
  const float* expert_gate_alpha;
  const void* expert_up_weight;
  const void* expert_up_scales;
  const float* expert_up_alpha;
  const void* expert_down_weight;
  const void* expert_down_scales;
  const float* expert_down_alpha;

  const void* shared_gate_weight;
  const void* shared_gate_scales;
  const float* shared_gate_alpha;
  const void* shared_up_weight;
  const void* shared_up_scales;
  const float* shared_up_alpha;
  const void* shared_down_weight;
  const void* shared_down_scales;
  const float* shared_down_alpha;

  void* route_intermediate;
  void* shared_intermediate;
  void* assignment_output;
  void* shared_output;
  apxinf_cuda_stream_t stream;
} apxinf_moe_nvfp4_experts_bindings_t;

