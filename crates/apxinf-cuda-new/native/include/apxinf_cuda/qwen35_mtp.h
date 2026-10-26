#pragma once

#include <stdint.h>

#include "gemm_types.h"
#include "runtime.h"

#define APXINF_QWEN35_MTP_MOE_SPEC_VERSION 1u

#ifdef __cplusplus
extern "C" {
#endif

// Exact execution contract for the single BF16 MTP MoE layer in
// txn545/Qwen3.5-35B-A3B-NVFP4.  Base decoder experts use ModelOpt NVFP4 and
// deliberately do not share this ABI.
typedef struct {
  uint32_t version;
  uint32_t dtype;
  uint32_t experts;
  uint32_t top_k;
  uint32_t hidden;
  uint32_t intermediate;
  uint32_t vocab_size;
  uint32_t mtp_layers;
  uint32_t draft_steps;
  uint32_t draft_step;
  uint64_t base_committed_position;
  uint64_t mtp_committed_position;
} apxinf_qwen35_mtp_moe_spec_t;

typedef struct {
  // Activations and every weight are BF16.  Dense matrices use cuda-new's
  // canonical [K,N] layout; expert matrices are [E,K,N].
  const void* input;
  const void* router_weight;
  const void* expert_gate_weight;
  const void* expert_up_weight;
  const void* expert_down_weight;
  const void* shared_gate_weight;
  const void* shared_up_weight;
  const void* shared_down_weight;
  const void* shared_expert_gate_weight;

  // Caller-owned CUDA scratch/output.  Keeping these explicit makes the
  // operation allocation-free and CUDA-graph compatible.
  void* router_logits;
  int32_t* expert_ids;
  float* expert_weights;
  void* route_intermediate;
  void* route_output;
  void* shared_intermediate;
  void* shared_output;
  void* shared_gate_logit;
  void* output;
  apxinf_cuda_stream_t stream;
} apxinf_qwen35_mtp_moe_bindings_t;

apxinf_status_t apxinf_qwen35_mtp_moe(
    apxinf_runtime_t runtime, const apxinf_qwen35_mtp_moe_spec_t* spec,
    const apxinf_qwen35_mtp_moe_bindings_t* bindings);

#ifdef __cplusplus
}
#endif
