#include "../include/apxinf_cuda/qwen35_mtp.h"
#include "../framework/runtime_internal.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cfloat>
#include <climits>
#include <cstddef>
#include <cstdint>

namespace {

using apxinf::framework::Failure;

constexpr int kThreads = 256;
constexpr int kExperts = 256;
constexpr int kTopK = 8;
constexpr int kHidden = 2048;
constexpr int kIntermediate = 512;
constexpr int kVocabSize = 248320;
constexpr int kMtpLayers = 1;
constexpr int kDraftSteps = 3;
static_assert(kExperts == kThreads);
static_assert(kHidden % kThreads == 0);
static_assert(kIntermediate % kThreads == 0);

__device__ __forceinline__ float stable_sigmoid(float value) {
  if (value >= 0.0f) return 1.0f / (1.0f + expf(-value));
  const float exponential = expf(value);
  return exponential / (1.0f + exponential);
}

// One block computes all BF16 router logits. One thread owns an expert and
// adjacent threads read adjacent [hidden, experts] weights at each K step.
__global__ void router_kernel(const __nv_bfloat16* __restrict__ input,
                              const __nv_bfloat16* __restrict__ weight,
                              __nv_bfloat16* __restrict__ logits) {
  const int expert = threadIdx.x;
  if (expert >= kExperts) return;
  float value = 0.0f;
  for (int feature = 0; feature < kHidden; ++feature) {
    value += __bfloat162float(input[feature]) *
             __bfloat162float(weight[static_cast<int64_t>(feature) *
                                      kExperts + expert]);
  }
  logits[expert] = __float2bfloat16_rn(value);
}

// Routing is deliberately serial after the device GEMV.  It is only 256
// values, makes lower-ID tie breaking exact, and never copies logits to host.
__global__ void route_topk_kernel(
    const __nv_bfloat16* __restrict__ logits,
    int32_t* __restrict__ expert_ids, float* __restrict__ expert_weights) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;

  float maximum = -FLT_MAX;
  for (int expert = 0; expert < kExperts; ++expert) {
    maximum = fmaxf(maximum, __bfloat162float(logits[expert]));
  }
  float denominator = 0.0f;
  for (int expert = 0; expert < kExperts; ++expert) {
    denominator += expf(__bfloat162float(logits[expert]) - maximum);
  }

  float selected_values[kTopK];
  int selected_ids[kTopK];
#pragma unroll
  for (int slot = 0; slot < kTopK; ++slot) {
    selected_values[slot] = -FLT_MAX;
    selected_ids[slot] = INT_MAX;
  }
  for (int expert = 0; expert < kExperts; ++expert) {
    const float value = __bfloat162float(logits[expert]);
    int insertion = kTopK;
#pragma unroll
    for (int slot = 0; slot < kTopK; ++slot) {
      if (value > selected_values[slot] ||
          (value == selected_values[slot] && expert < selected_ids[slot])) {
        insertion = slot;
        break;
      }
    }
    if (insertion == kTopK) continue;
    for (int slot = kTopK - 1; slot > insertion; --slot) {
      selected_values[slot] = selected_values[slot - 1];
      selected_ids[slot] = selected_ids[slot - 1];
    }
    selected_values[insertion] = value;
    selected_ids[insertion] = expert;
  }

  float selected_sum = 0.0f;
#pragma unroll
  for (int slot = 0; slot < kTopK; ++slot) {
    expert_weights[slot] =
        expf(selected_values[slot] - maximum) / denominator;
    selected_sum += expert_weights[slot];
  }
#pragma unroll
  for (int slot = 0; slot < kTopK; ++slot) {
    expert_ids[slot] = selected_ids[slot];
    expert_weights[slot] /= selected_sum;
  }
}

__global__ void selected_swiglu_kernel(
    const __nv_bfloat16* __restrict__ input,
    const int32_t* __restrict__ expert_ids,
    const __nv_bfloat16* __restrict__ gate_weight,
    const __nv_bfloat16* __restrict__ up_weight,
    __nv_bfloat16* __restrict__ intermediate) {
  constexpr int kTiles = kIntermediate / kThreads;
  const int route = blockIdx.x / kTiles;
  const int output = (blockIdx.x % kTiles) * kThreads + threadIdx.x;
  const int expert = expert_ids[route];
  float gate = 0.0f;
  float up = 0.0f;
  if (expert >= 0 && expert < kExperts) {
    const int64_t expert_base =
        static_cast<int64_t>(expert) * kHidden * kIntermediate;
    for (int feature = 0; feature < kHidden; ++feature) {
      const float activation = __bfloat162float(input[feature]);
      const int64_t index = expert_base +
                            static_cast<int64_t>(feature) * kIntermediate +
                            output;
      gate += activation * __bfloat162float(gate_weight[index]);
      up += activation * __bfloat162float(up_weight[index]);
    }
  }
  gate = __bfloat162float(__float2bfloat16_rn(gate));
  up = __bfloat162float(__float2bfloat16_rn(up));
  intermediate[static_cast<int64_t>(route) * kIntermediate + output] =
      __float2bfloat16_rn((gate * stable_sigmoid(gate)) * up);
}

__global__ void selected_down_kernel(
    const __nv_bfloat16* __restrict__ intermediate,
    const int32_t* __restrict__ expert_ids,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ output) {
  constexpr int kTiles = kHidden / kThreads;
  const int route = blockIdx.x / kTiles;
  const int output_feature =
      (blockIdx.x % kTiles) * kThreads + threadIdx.x;
  const int expert = expert_ids[route];
  float value = 0.0f;
  if (expert >= 0 && expert < kExperts) {
    const int64_t expert_base =
        static_cast<int64_t>(expert) * kIntermediate * kHidden;
    for (int feature = 0; feature < kIntermediate; ++feature) {
      value += __bfloat162float(
                   intermediate[static_cast<int64_t>(route) * kIntermediate +
                                feature]) *
               __bfloat162float(
                   weight[expert_base +
                          static_cast<int64_t>(feature) * kHidden +
                          output_feature]);
    }
  }
  output[static_cast<int64_t>(route) * kHidden + output_feature] =
      __float2bfloat16_rn(value);
}

__global__ void shared_swiglu_kernel(
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ gate_weight,
    const __nv_bfloat16* __restrict__ up_weight,
    __nv_bfloat16* __restrict__ intermediate) {
  const int output = blockIdx.x * kThreads + threadIdx.x;
  float gate = 0.0f;
  float up = 0.0f;
  for (int feature = 0; feature < kHidden; ++feature) {
    const float activation = __bfloat162float(input[feature]);
    const int64_t index = static_cast<int64_t>(feature) * kIntermediate +
                          output;
    gate += activation * __bfloat162float(gate_weight[index]);
    up += activation * __bfloat162float(up_weight[index]);
  }
  gate = __bfloat162float(__float2bfloat16_rn(gate));
  up = __bfloat162float(__float2bfloat16_rn(up));
  intermediate[output] =
      __float2bfloat16_rn((gate * stable_sigmoid(gate)) * up);
}

__global__ void shared_down_kernel(
    const __nv_bfloat16* __restrict__ intermediate,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ output) {
  const int output_feature = blockIdx.x * kThreads + threadIdx.x;
  float value = 0.0f;
  for (int feature = 0; feature < kIntermediate; ++feature) {
    value += __bfloat162float(intermediate[feature]) *
             __bfloat162float(
                 weight[static_cast<int64_t>(feature) * kHidden +
                        output_feature]);
  }
  output[output_feature] = __float2bfloat16_rn(value);
}

__global__ void shared_gate_kernel(
    const __nv_bfloat16* __restrict__ input,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ output) {
  if (blockIdx.x != 0 || threadIdx.x != 0) return;
  float value = 0.0f;
  for (int feature = 0; feature < kHidden; ++feature) {
    value += __bfloat162float(input[feature]) *
             __bfloat162float(weight[feature]);
  }
  output[0] = __float2bfloat16_rn(value);
}

__global__ void combine_kernel(
    const __nv_bfloat16* __restrict__ route_output,
    const float* __restrict__ expert_weights,
    const __nv_bfloat16* __restrict__ shared_output,
    const __nv_bfloat16* __restrict__ shared_gate_logit,
    __nv_bfloat16* __restrict__ output) {
  for (int feature = blockIdx.x * blockDim.x + threadIdx.x;
       feature < kHidden; feature += blockDim.x * gridDim.x) {
    float value = 0.0f;
#pragma unroll
    for (int route = 0; route < kTopK; ++route) {
      value += expert_weights[route] *
               __bfloat162float(
                   route_output[static_cast<int64_t>(route) * kHidden +
                                feature]);
    }
    value += stable_sigmoid(__bfloat162float(shared_gate_logit[0])) *
             __bfloat162float(shared_output[feature]);
    output[feature] = __float2bfloat16_rn(value);
  }
}

void validate_spec(const apxinf_qwen35_mtp_moe_spec_t& spec) {
  // Steps 0..2 are draft rows. Step 3 is the no-logits d2 cache row used
  // only after full acceptance plus a bonus; 4 and above are invalid.
  if (spec.version != APXINF_QWEN35_MTP_MOE_SPEC_VERSION ||
      spec.dtype != APXINF_DTYPE_BF16 || spec.experts != kExperts ||
      spec.top_k != kTopK || spec.hidden != kHidden ||
      spec.intermediate != kIntermediate ||
      spec.vocab_size != kVocabSize || spec.mtp_layers != kMtpLayers ||
      spec.draft_steps != kDraftSteps || spec.draft_step > kDraftSteps ||
      spec.base_committed_position != spec.mtp_committed_position ||
      spec.base_committed_position > UINT64_MAX - spec.draft_step) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                  "invalid Qwen3.5 MTP BF16 MoE Spec");
  }
}

cudaError_t launch(const apxinf_qwen35_mtp_moe_bindings_t& bindings) {
  auto stream = static_cast<cudaStream_t>(bindings.stream);
  const auto* input = static_cast<const __nv_bfloat16*>(bindings.input);

  router_kernel<<<1, kThreads, 0, stream>>>(
      input, static_cast<const __nv_bfloat16*>(bindings.router_weight),
      static_cast<__nv_bfloat16*>(bindings.router_logits));
  cudaError_t status = cudaGetLastError();
  if (status != cudaSuccess) return status;
  route_topk_kernel<<<1, 1, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(bindings.router_logits),
      bindings.expert_ids, bindings.expert_weights);
  status = cudaGetLastError();
  if (status != cudaSuccess) return status;

  selected_swiglu_kernel<<<kTopK * (kIntermediate / kThreads), kThreads, 0,
                             stream>>>(
      input, bindings.expert_ids,
      static_cast<const __nv_bfloat16*>(bindings.expert_gate_weight),
      static_cast<const __nv_bfloat16*>(bindings.expert_up_weight),
      static_cast<__nv_bfloat16*>(bindings.route_intermediate));
  status = cudaGetLastError();
  if (status != cudaSuccess) return status;
  selected_down_kernel<<<kTopK * (kHidden / kThreads), kThreads, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(bindings.route_intermediate),
      bindings.expert_ids,
      static_cast<const __nv_bfloat16*>(bindings.expert_down_weight),
      static_cast<__nv_bfloat16*>(bindings.route_output));
  status = cudaGetLastError();
  if (status != cudaSuccess) return status;

  shared_swiglu_kernel<<<kIntermediate / kThreads, kThreads, 0, stream>>>(
      input,
      static_cast<const __nv_bfloat16*>(bindings.shared_gate_weight),
      static_cast<const __nv_bfloat16*>(bindings.shared_up_weight),
      static_cast<__nv_bfloat16*>(bindings.shared_intermediate));
  status = cudaGetLastError();
  if (status != cudaSuccess) return status;
  shared_down_kernel<<<kHidden / kThreads, kThreads, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(bindings.shared_intermediate),
      static_cast<const __nv_bfloat16*>(bindings.shared_down_weight),
      static_cast<__nv_bfloat16*>(bindings.shared_output));
  status = cudaGetLastError();
  if (status != cudaSuccess) return status;
  shared_gate_kernel<<<1, 1, 0, stream>>>(
      input,
      static_cast<const __nv_bfloat16*>(bindings.shared_expert_gate_weight),
      static_cast<__nv_bfloat16*>(bindings.shared_gate_logit));
  status = cudaGetLastError();
  if (status != cudaSuccess) return status;

  combine_kernel<<<8, kThreads, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(bindings.route_output),
      bindings.expert_weights,
      static_cast<const __nv_bfloat16*>(bindings.shared_output),
      static_cast<const __nv_bfloat16*>(bindings.shared_gate_logit),
      static_cast<__nv_bfloat16*>(bindings.output));
  return cudaGetLastError();
}

}  // namespace

extern "C" apxinf_status_t apxinf_qwen35_mtp_moe(
    apxinf_runtime_t runtime, const apxinf_qwen35_mtp_moe_spec_t* spec,
    const apxinf_qwen35_mtp_moe_bindings_t* bindings) {
  return apxinf::framework::abi_boundary([&] {
    if (runtime == nullptr || spec == nullptr || bindings == nullptr) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                    "null Qwen3.5 MTP MoE argument");
    }
    validate_spec(*spec);
    if (bindings->input == nullptr || bindings->router_weight == nullptr ||
        bindings->expert_gate_weight == nullptr ||
        bindings->expert_up_weight == nullptr ||
        bindings->expert_down_weight == nullptr ||
        bindings->shared_gate_weight == nullptr ||
        bindings->shared_up_weight == nullptr ||
        bindings->shared_down_weight == nullptr ||
        bindings->shared_expert_gate_weight == nullptr ||
        bindings->router_logits == nullptr ||
        bindings->expert_ids == nullptr ||
        bindings->expert_weights == nullptr ||
        bindings->route_intermediate == nullptr ||
        bindings->route_output == nullptr ||
        bindings->shared_intermediate == nullptr ||
        bindings->shared_output == nullptr ||
        bindings->shared_gate_logit == nullptr || bindings->output == nullptr) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                    "missing Qwen3.5 MTP MoE binding");
    }
    apxinf::framework::check_cuda(cudaSetDevice(runtime->device));
    apxinf::framework::check_cuda(launch(*bindings));
  });
}
