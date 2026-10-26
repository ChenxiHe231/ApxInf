//! Native BF16 expert execution for the single Qwen3.5-35B-A3B MTP layer.
//!
//! The pinned checkpoint deliberately excludes `mtp.layers.0*` from ModelOpt
//! quantization.  Consequently this operator accepts only the checkpoint's
//! BF16 `[E,K,N]` expert layout.  The base decoder's NVFP4 experts are a
//! different semantic and must use the routed-NVFP4 operator.

use std::cmp::Ordering;

use apxinf_core::{DType, Result, Tensor};

use crate::ffi::abi::{qwen35_mtp as abi, status};
use crate::ops::gemm::contracts::{dtype, invalid, tensor_storage};
use crate::{CudaBuffer, CudaContext};

pub const QWEN35_MTP_EXPERTS: usize = 256;
pub const QWEN35_MTP_TOP_K: usize = 8;
pub const QWEN35_MTP_HIDDEN: usize = 2_048;
pub const QWEN35_MTP_INTERMEDIATE: usize = 512;
pub const QWEN35_MTP_VOCAB_SIZE: usize = 248_320;
pub const QWEN35_MTP_LAYERS: usize = 1;
pub const QWEN35_MTP_DRAFT_STEPS: usize = 3;

/// Transactional identity of one MTP proposal execution.
///
/// The expert operator is mathematically stateless, but it must not be
/// launched against divergent base/MTP cache positions.  `draft_step` is
/// zero-based: `0..3` names the proposal chain and `3` is the no-logits row
/// materialized only after full acceptance plus a bonus. Teacher-forced
/// prompt rows use step zero because they are outside a speculative round.
/// Cache shadow commit/rollback remains the schedule/runtime's responsibility.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Qwen35MtpMoeState {
    pub base_committed_position: usize,
    pub mtp_committed_position: usize,
    pub draft_step: usize,
}

impl Qwen35MtpMoeState {
    pub fn validate(self) -> Result<()> {
        if self.base_committed_position != self.mtp_committed_position {
            return Err(invalid(
                "Qwen3.5 MTP base and MTP committed positions must match",
            ));
        }
        if self.draft_step > QWEN35_MTP_DRAFT_STEPS {
            return Err(invalid(
                "Qwen3.5 MTP draft_step must be a proposal depth or final materialization",
            ));
        }
        self.base_committed_position
            .checked_add(self.draft_step)
            .ok_or_else(|| invalid("Qwen3.5 MTP proposal position overflow"))?;
        Ok(())
    }

    pub fn proposal_position(self) -> Result<usize> {
        self.validate()?;
        self.base_committed_position
            .checked_add(self.draft_step)
            .ok_or_else(|| invalid("Qwen3.5 MTP proposal position overflow"))
    }
}

/// CUDA-resident checkpoint weights.  Every tensor is BF16.
pub struct Qwen35MtpMoeWeights<'a> {
    /// `[hidden, experts]`.
    pub router: &'a Tensor,
    /// `[experts, hidden, intermediate]`.
    pub expert_gate: &'a Tensor,
    /// `[experts, hidden, intermediate]`.
    pub expert_up: &'a Tensor,
    /// `[experts, intermediate, hidden]`.
    pub expert_down: &'a Tensor,
    /// `[hidden, intermediate]`.
    pub shared_gate: &'a Tensor,
    /// `[hidden, intermediate]`.
    pub shared_up: &'a Tensor,
    /// `[intermediate, hidden]`.
    pub shared_down: &'a Tensor,
    /// `[hidden, 1]`.
    pub shared_expert_gate: &'a Tensor,
}

/// Explicit allocation-free scratch for a single-token proposal.
pub struct Qwen35MtpMoeScratch<'a> {
    /// BF16 `[1, experts]`.
    pub router_logits: &'a mut Tensor,
    /// I32 `[top_k]`, in descending logit order with lower-ID tie breaks.
    pub expert_ids: &'a mut Tensor,
    /// F32 `[top_k]`, softmax then selected-probability renormalization.
    pub expert_weights: &'a mut Tensor,
    /// BF16 `[top_k, intermediate]`.
    pub route_intermediate: &'a mut Tensor,
    /// BF16 `[top_k, hidden]`.
    pub route_output: &'a mut Tensor,
    /// BF16 `[1, intermediate]`.
    pub shared_intermediate: &'a mut Tensor,
    /// BF16 `[1, hidden]`.
    pub shared_output: &'a mut Tensor,
    /// BF16 `[1]`.
    pub shared_gate_logit: &'a mut Tensor,
}

pub struct Qwen35MtpMoeArgs<'a> {
    pub state: Qwen35MtpMoeState,
    /// One BF16 proposal row `[1, hidden]`.
    pub input: &'a Tensor,
    pub weights: Qwen35MtpMoeWeights<'a>,
    pub scratch: Qwen35MtpMoeScratch<'a>,
    /// BF16 `[1, hidden]`.
    pub output: &'a mut Tensor,
}

/// Execute router, deterministic softmax/top-8, selected BF16 SwiGLU experts,
/// sigmoid-gated shared expert, and weighted combine without a host round trip.
pub fn qwen35_mtp_moe(ctx: &CudaContext, args: Qwen35MtpMoeArgs<'_>) -> Result<()> {
    let normalized = normalize(ctx, args)?;
    let _storage = &normalized.storage;
    crate::workspace::validate_capture_target(ctx.device_id(), normalized.bindings.stream as usize)?;
    unsafe {
        status::check(abi::apxinf_qwen35_mtp_moe(
            ctx.runtime(),
            &normalized.spec,
            &normalized.bindings,
        ))
    }
}

struct NormalizedMtpMoe {
    spec: abi::MtpMoeSpec,
    bindings: abi::MtpMoeBindings,
    storage: Vec<CudaBuffer>,
}

fn normalize(ctx: &CudaContext, args: Qwen35MtpMoeArgs<'_>) -> Result<NormalizedMtpMoe> {
    args.state.validate()?;
    let Qwen35MtpMoeWeights {
        router,
        expert_gate,
        expert_up,
        expert_down,
        shared_gate,
        shared_up,
        shared_down,
        shared_expert_gate,
    } = args.weights;
    let Qwen35MtpMoeScratch {
        router_logits,
        expert_ids,
        expert_weights,
        route_intermediate,
        route_output,
        shared_intermediate,
        shared_output,
        shared_gate_logit,
    } = args.scratch;

    let storage = vec![
        tensor_storage(ctx, args.input, DType::BF16, &[1, QWEN35_MTP_HIDDEN])?,
        tensor_storage(
            ctx,
            router,
            DType::BF16,
            &[QWEN35_MTP_HIDDEN, QWEN35_MTP_EXPERTS],
        )?,
        tensor_storage(
            ctx,
            expert_gate,
            DType::BF16,
            &[
                QWEN35_MTP_EXPERTS,
                QWEN35_MTP_HIDDEN,
                QWEN35_MTP_INTERMEDIATE,
            ],
        )?,
        tensor_storage(
            ctx,
            expert_up,
            DType::BF16,
            &[
                QWEN35_MTP_EXPERTS,
                QWEN35_MTP_HIDDEN,
                QWEN35_MTP_INTERMEDIATE,
            ],
        )?,
        tensor_storage(
            ctx,
            expert_down,
            DType::BF16,
            &[
                QWEN35_MTP_EXPERTS,
                QWEN35_MTP_INTERMEDIATE,
                QWEN35_MTP_HIDDEN,
            ],
        )?,
        tensor_storage(
            ctx,
            shared_gate,
            DType::BF16,
            &[QWEN35_MTP_HIDDEN, QWEN35_MTP_INTERMEDIATE],
        )?,
        tensor_storage(
            ctx,
            shared_up,
            DType::BF16,
            &[QWEN35_MTP_HIDDEN, QWEN35_MTP_INTERMEDIATE],
        )?,
        tensor_storage(
            ctx,
            shared_down,
            DType::BF16,
            &[QWEN35_MTP_INTERMEDIATE, QWEN35_MTP_HIDDEN],
        )?,
        tensor_storage(
            ctx,
            shared_expert_gate,
            DType::BF16,
            &[QWEN35_MTP_HIDDEN, 1],
        )?,
        tensor_storage(
            ctx,
            router_logits,
            DType::BF16,
            &[1, QWEN35_MTP_EXPERTS],
        )?,
        tensor_storage(ctx, expert_ids, DType::I32, &[QWEN35_MTP_TOP_K])?,
        tensor_storage(ctx, expert_weights, DType::F32, &[QWEN35_MTP_TOP_K])?,
        tensor_storage(
            ctx,
            route_intermediate,
            DType::BF16,
            &[QWEN35_MTP_TOP_K, QWEN35_MTP_INTERMEDIATE],
        )?,
        tensor_storage(
            ctx,
            route_output,
            DType::BF16,
            &[QWEN35_MTP_TOP_K, QWEN35_MTP_HIDDEN],
        )?,
        tensor_storage(
            ctx,
            shared_intermediate,
            DType::BF16,
            &[1, QWEN35_MTP_INTERMEDIATE],
        )?,
        tensor_storage(
            ctx,
            shared_output,
            DType::BF16,
            &[1, QWEN35_MTP_HIDDEN],
        )?,
        tensor_storage(ctx, shared_gate_logit, DType::BF16, &[1])?,
        tensor_storage(
            ctx,
            args.output,
            DType::BF16,
            &[1, QWEN35_MTP_HIDDEN],
        )?,
    ];

    let bindings = abi::MtpMoeBindings {
        input: storage[0].ptr(),
        router_weight: storage[1].ptr(),
        expert_gate_weight: storage[2].ptr(),
        expert_up_weight: storage[3].ptr(),
        expert_down_weight: storage[4].ptr(),
        shared_gate_weight: storage[5].ptr(),
        shared_up_weight: storage[6].ptr(),
        shared_down_weight: storage[7].ptr(),
        shared_expert_gate_weight: storage[8].ptr(),
        router_logits: storage[9].ptr(),
        expert_ids: storage[10].ptr().cast(),
        expert_weights: storage[11].ptr().cast(),
        route_intermediate: storage[12].ptr(),
        route_output: storage[13].ptr(),
        shared_intermediate: storage[14].ptr(),
        shared_output: storage[15].ptr(),
        shared_gate_logit: storage[16].ptr(),
        output: storage[17].ptr(),
        stream: ctx.stream().handle(),
    };
    let spec = abi::MtpMoeSpec {
        version: abi::MTP_MOE_SPEC_VERSION,
        dtype: dtype(DType::BF16)?,
        experts: QWEN35_MTP_EXPERTS as u32,
        top_k: QWEN35_MTP_TOP_K as u32,
        hidden: QWEN35_MTP_HIDDEN as u32,
        intermediate: QWEN35_MTP_INTERMEDIATE as u32,
        vocab_size: QWEN35_MTP_VOCAB_SIZE as u32,
        mtp_layers: QWEN35_MTP_LAYERS as u32,
        draft_steps: QWEN35_MTP_DRAFT_STEPS as u32,
        draft_step: args.state.draft_step as u32,
        base_committed_position: u64::try_from(args.state.base_committed_position)
            .map_err(|_| invalid("Qwen3.5 MTP base position exceeds the native range"))?,
        mtp_committed_position: u64::try_from(args.state.mtp_committed_position)
            .map_err(|_| invalid("Qwen3.5 MTP position exceeds the native range"))?,
    };
    Ok(NormalizedMtpMoe {
        spec,
        bindings,
        storage,
    })
}

/// Scalar weight source for the exact CPU reference.
///
/// Implementations may index real flattened BF16 tensors or procedurally
/// generate values for tests; every returned value must be the F32 decoding of
/// one finite BF16 checkpoint element.
pub trait Qwen35MtpReferenceWeights {
    fn router(&self, input: usize, expert: usize) -> f32;
    fn expert_gate(&self, expert: usize, input: usize, output: usize) -> f32;
    fn expert_up(&self, expert: usize, input: usize, output: usize) -> f32;
    fn expert_down(&self, expert: usize, input: usize, output: usize) -> f32;
    fn shared_gate(&self, input: usize, output: usize) -> f32;
    fn shared_up(&self, input: usize, output: usize) -> f32;
    fn shared_down(&self, input: usize, output: usize) -> f32;
    fn shared_expert_gate(&self, input: usize) -> f32;
}

#[derive(Clone, Debug, PartialEq)]
pub struct Qwen35MtpReferenceOutput {
    pub router_logits: Vec<f32>,
    pub expert_ids: [u32; QWEN35_MTP_TOP_K],
    pub expert_weights: [f32; QWEN35_MTP_TOP_K],
    pub output: Vec<f32>,
}

/// CPU oracle for the exact native rounding and routing semantic.
pub fn qwen35_mtp_moe_reference<W: Qwen35MtpReferenceWeights>(
    input: &[f32],
    weights: &W,
) -> Result<Qwen35MtpReferenceOutput> {
    if input.len() != QWEN35_MTP_HIDDEN || input.iter().any(|value| !value.is_finite()) {
        return Err(invalid(
            "Qwen3.5 MTP reference input must be one finite hidden row",
        ));
    }

    let mut router_logits = vec![0.0f32; QWEN35_MTP_EXPERTS];
    for (expert, logit) in router_logits.iter_mut().enumerate() {
        let mut value = 0.0f32;
        for (feature, activation) in input.iter().copied().enumerate() {
            value += activation * finite(weights.router(feature, expert))?;
        }
        *logit = round_bf16(value);
    }
    let (expert_ids, expert_weights) = route_reference(&router_logits);

    let mut route_output = vec![0.0f32; QWEN35_MTP_TOP_K * QWEN35_MTP_HIDDEN];
    for route in 0..QWEN35_MTP_TOP_K {
        let expert = expert_ids[route] as usize;
        let intermediate = swiglu_reference(input, |feature, output| {
            (
                weights.expert_gate(expert, feature, output),
                weights.expert_up(expert, feature, output),
            )
        })?;
        let start = route * QWEN35_MTP_HIDDEN;
        dense_reference(
            &intermediate,
            &mut route_output[start..start + QWEN35_MTP_HIDDEN],
            |feature, output| weights.expert_down(expert, feature, output),
        )?;
    }

    let shared_intermediate = swiglu_reference(input, |feature, output| {
        (
            weights.shared_gate(feature, output),
            weights.shared_up(feature, output),
        )
    })?;
    let mut shared_output = vec![0.0f32; QWEN35_MTP_HIDDEN];
    dense_reference(
        &shared_intermediate,
        &mut shared_output,
        |feature, output| weights.shared_down(feature, output),
    )?;
    let mut shared_gate_logit = 0.0f32;
    for (feature, activation) in input.iter().copied().enumerate() {
        shared_gate_logit += activation * finite(weights.shared_expert_gate(feature))?;
    }
    shared_gate_logit = round_bf16(shared_gate_logit);

    let shared_scale = sigmoid(shared_gate_logit);
    let mut output = vec![0.0f32; QWEN35_MTP_HIDDEN];
    for feature in 0..QWEN35_MTP_HIDDEN {
        let mut value = 0.0f32;
        for route in 0..QWEN35_MTP_TOP_K {
            value += expert_weights[route]
                * route_output[route * QWEN35_MTP_HIDDEN + feature];
        }
        value += shared_scale * shared_output[feature];
        output[feature] = round_bf16(value);
    }

    Ok(Qwen35MtpReferenceOutput {
        router_logits,
        expert_ids,
        expert_weights,
        output,
    })
}

fn route_reference(
    logits: &[f32],
) -> (
    [u32; QWEN35_MTP_TOP_K],
    [f32; QWEN35_MTP_TOP_K],
) {
    let maximum = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let denominator: f32 = logits.iter().map(|value| (*value - maximum).exp()).sum();
    let mut ranking = logits.iter().copied().enumerate().collect::<Vec<_>>();
    ranking.sort_by(|(left_id, left), (right_id, right)| {
        right
            .partial_cmp(left)
            .unwrap_or(Ordering::Equal)
            .then_with(|| left_id.cmp(right_id))
    });
    let mut ids = [0u32; QWEN35_MTP_TOP_K];
    let mut probabilities = [0.0f32; QWEN35_MTP_TOP_K];
    let mut selected_sum = 0.0f32;
    for slot in 0..QWEN35_MTP_TOP_K {
        ids[slot] = ranking[slot].0 as u32;
        probabilities[slot] = (ranking[slot].1 - maximum).exp() / denominator;
        selected_sum += probabilities[slot];
    }
    for probability in &mut probabilities {
        *probability /= selected_sum;
    }
    (ids, probabilities)
}

fn swiglu_reference(
    input: &[f32],
    mut weight: impl FnMut(usize, usize) -> (f32, f32),
) -> Result<Vec<f32>> {
    let mut output = vec![0.0f32; QWEN35_MTP_INTERMEDIATE];
    for (dimension, result) in output.iter_mut().enumerate() {
        let mut gate = 0.0f32;
        let mut up = 0.0f32;
        for (feature, activation) in input.iter().copied().enumerate() {
            let (gate_weight, up_weight) = weight(feature, dimension);
            gate += activation * finite(gate_weight)?;
            up += activation * finite(up_weight)?;
        }
        gate = round_bf16(gate);
        up = round_bf16(up);
        *result = round_bf16((gate * sigmoid(gate)) * up);
    }
    Ok(output)
}

fn dense_reference(
    input: &[f32],
    output: &mut [f32],
    mut weight: impl FnMut(usize, usize) -> f32,
) -> Result<()> {
    for (dimension, result) in output.iter_mut().enumerate() {
        let mut value = 0.0f32;
        for (feature, activation) in input.iter().copied().enumerate() {
            value += activation * finite(weight(feature, dimension))?;
        }
        *result = round_bf16(value);
    }
    Ok(())
}

fn finite(value: f32) -> Result<f32> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(invalid("Qwen3.5 MTP reference weight must be finite"))
    }
}

fn sigmoid(value: f32) -> f32 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exponential = value.exp();
        exponential / (1.0 + exponential)
    }
}

// IEEE round-to-nearest-even conversion to BF16 and back without adding a
// runtime dependency on a host half-precision library.
fn round_bf16(value: f32) -> f32 {
    let bits = value.to_bits();
    let rounded = bits.wrapping_add(0x7fff + ((bits >> 16) & 1));
    f32::from_bits(rounded & 0xffff_0000)
}

#[cfg(test)]
mod tests {
    use super::*;

    struct ZeroWeights;

    impl Qwen35MtpReferenceWeights for ZeroWeights {
        fn router(&self, _: usize, _: usize) -> f32 { 0.0 }
        fn expert_gate(&self, _: usize, _: usize, _: usize) -> f32 { 0.0 }
        fn expert_up(&self, _: usize, _: usize, _: usize) -> f32 { 0.0 }
        fn expert_down(&self, _: usize, _: usize, _: usize) -> f32 { 0.0 }
        fn shared_gate(&self, _: usize, _: usize) -> f32 { 0.0 }
        fn shared_up(&self, _: usize, _: usize) -> f32 { 0.0 }
        fn shared_down(&self, _: usize, _: usize) -> f32 { 0.0 }
        fn shared_expert_gate(&self, _: usize) -> f32 { 0.0 }
    }

    #[test]
    fn state_enforces_three_drafts_one_materialization_and_matching_positions() {
        for draft_step in 0..=QWEN35_MTP_DRAFT_STEPS {
            let state = Qwen35MtpMoeState {
                base_committed_position: 41,
                mtp_committed_position: 41,
                draft_step,
            };
            assert_eq!(state.proposal_position().unwrap(), 41 + draft_step);
        }
        assert!(Qwen35MtpMoeState {
            base_committed_position: 41,
            mtp_committed_position: 40,
            draft_step: 0,
        }
        .validate()
        .is_err());
        assert!(Qwen35MtpMoeState {
            base_committed_position: 41,
            mtp_committed_position: 41,
            draft_step: 4,
        }
        .validate()
        .is_err());
    }

    #[test]
    fn exact_reference_uses_stable_top8_and_bf16_zero_path() {
        let reference = qwen35_mtp_moe_reference(
            &vec![0.0; QWEN35_MTP_HIDDEN],
            &ZeroWeights,
        )
        .unwrap();
        assert_eq!(reference.expert_ids, [0, 1, 2, 3, 4, 5, 6, 7]);
        assert_eq!(reference.expert_weights, [0.125; QWEN35_MTP_TOP_K]);
        assert!(reference.router_logits.iter().all(|value| *value == 0.0));
        assert!(reference.output.iter().all(|value| *value == 0.0));
    }

    #[test]
    fn reference_rejects_wrong_shape_and_non_finite_input() {
        assert!(qwen35_mtp_moe_reference(&[0.0], &ZeroWeights).is_err());
        let mut input = vec![0.0; QWEN35_MTP_HIDDEN];
        input[17] = f32::NAN;
        assert!(qwen35_mtp_moe_reference(&input, &ZeroWeights).is_err());
    }
}
