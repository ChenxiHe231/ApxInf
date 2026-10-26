#pragma once

#include <stddef.h>
#include <stdint.h>

#include "gemm_types.h"

#define APXINF_MOE_ROUTE_SPEC_VERSION 1u
#define APXINF_MOE_COMBINE_SPEC_VERSION 1u
#define APXINF_MOE_NVFP4_EXPERTS_SPEC_VERSION 1u
#define APXINF_MOE_EDGE_NVFP4_SPEC_VERSION 1u
#define APXINF_MOE_EDGE_BF16_DECODE_SPEC_VERSION 1u

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
  int32_t* token_assignment_indices;
  float* assignment_weights;
  int32_t* expert_offsets;
  apxinf_cuda_stream_t stream;
} apxinf_moe_route_bindings_t;

typedef struct {
  uint32_t version;
  uint32_t dtype;
  uint32_t tokens;
  uint32_t routes;
  uint32_t top_k;
  uint32_t hidden;
} apxinf_moe_combine_spec_t;

typedef struct {
  const void* assignment_output;
  const int32_t* assignment_token_ids;
  const float* assignment_weights;
  const int32_t* token_assignment_indices;
  const void* shared_output;
  const void* shared_gate_logits;
  void* output;
  apxinf_cuda_stream_t stream;
} apxinf_moe_combine_bindings_t;

typedef struct {
  const void* routed_output;
  const void* shared_output;
  const void* shared_gate_logits;
  void* output;
  apxinf_cuda_stream_t stream;
} apxinf_moe_add_shared_bindings_t;

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

// TensorRT-Edge-LLM CuTeDSL NVFP4 routed-MoE contract.  Unlike the
// correctness candidate above, FC2 applies the token-major routing weights
// and reduces directly into `output` [tokens, hidden].
typedef struct {
  uint32_t version;
  uint32_t dtype;
  uint32_t tokens;
  uint32_t experts;
  uint32_t top_k;
  uint32_t hidden;
  uint32_t intermediate;
} apxinf_moe_edge_nvfp4_spec_t;

typedef struct {
  const void* input;
  const int32_t* expert_ids;
  const float* expert_weights;

  // FC1 is 64-row [up, gate] interleaved.  Both scale tensors use the CuTe
  // 128x4 MMA layout documented by TensorRT-Edge-LLM.
  const void* fc1_weight;
  const void* fc1_scales;
  const float* fc1_alpha;
  const float* input_global_scale;
  const void* fc2_weight;
  const void* fc2_scales;
  const float* fc2_alpha;
  const float* down_input_scale;

  void* workspace;
  size_t workspace_bytes;
  void* output;
  apxinf_cuda_stream_t stream;
} apxinf_moe_edge_nvfp4_bindings_t;

// TensorRT-Edge-LLM CuTeDSL homogeneous grouped-GEMM contract specialized
// for one-token BF16 decode. FC1 applies SwiGLU and FC2 applies the top-k
// weights, producing one token-major routed contribution.
typedef struct {
  uint32_t version;
  uint32_t dtype;
  uint32_t experts;
  uint32_t top_k;
  uint32_t hidden;
  uint32_t intermediate;
} apxinf_moe_edge_bf16_decode_spec_t;

typedef struct {
  const void* input;
  const int32_t* expert_ids;
  const float* expert_weights;
  // FC1 is [E, 2*I, H] with 64-row [up, gate] interleaving.
  const void* fc1_weight;
  // FC2 is [E, H, I].
  const void* fc2_weight;
  void* workspace;
  size_t workspace_bytes;
  void* output;
  apxinf_cuda_stream_t stream;
} apxinf_moe_edge_bf16_decode_bindings_t;

