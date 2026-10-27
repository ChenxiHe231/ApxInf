//! Native Qwen3.5-35B-A3B MTP proposer.
//!
//! This is deliberately not the Qwen3.8 dense-MLP proposer. The checkpoint's
//! one MTP layer is BF16 and contains the same 256-expert/top-8 routed MoE as
//! the target architecture. Its K/V cache is BF16 as well; base-model FP8 KV
//! calibration values never enter this path.

use apxinf_core::{DType, Device, Error, Result, Shape, Tensor};
use apxinf_cuda_new::{ops, transfers, CudaBuffer, CudaContext};

use super::architecture::AttentionGeometry;
use super::config::Qwen35MoeConfig;
use super::decoder::{DecoderOutput, FullAttentionBackend, Qwen35Decoder};
use super::device_weights::{
    DeviceDenseMoe, DeviceDenseRoutedExperts, DeviceEdgeBf16Experts, DeviceMtpWeights,
    Qwen35DeviceWeights,
};
use super::mtp_schedule::{
    MtpCommit, Qwen35MtpSchedule, RoundOutcome, TokenId, QWEN35_MTP_DRAFT_STEPS,
};
use super::session::{MtpBackend, Qwen35Session, Qwen35SessionSnapshot};
use crate::{GenerationOptions, SamplingMode};

const HIDDEN: usize = ops::QWEN35_MTP_HIDDEN;
const VOCAB: usize = ops::QWEN35_MTP_VOCAB_SIZE;
const EXPERTS: usize = ops::QWEN35_MTP_EXPERTS;
const TOP_K: usize = ops::QWEN35_MTP_TOP_K;
const INTERMEDIATE: usize = ops::QWEN35_MTP_INTERMEDIATE;
const MTP_PREFILL_CHUNK: usize = 2_048;
pub const QWEN35_MTP_KV_DTYPE: DType = DType::BF16;

/// Logical/physical MTP position contract.
///
/// The target session exposes the next committed position. After a prompt of
/// length `P` that is `P`. The shifted MTP teacher-forcing pass has consumed
/// only `prompt[1..]`, so it owns `P-1` physical KV rows. Keeping both values
/// explicit prevents the classic one-token drift between target and proposer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MtpPositionContract {
    pub logical_position: usize,
    pub cache_rows: usize,
}
impl MtpPositionContract {
    pub fn after_prompt(prompt_len: usize) -> Result<Self> {
        if prompt_len == 0 {
            return Err(invalid("Qwen3.5 MTP requires a non-empty prompt"));
        }
        Ok(Self {
            logical_position: prompt_len,
            cache_rows: prompt_len - 1,
        })
    }

    pub fn physical_row_for_draft(self, draft_step: usize) -> Result<usize> {
        if draft_step >= QWEN35_MTP_DRAFT_STEPS {
            return Err(invalid("Qwen3.5 MTP draft depth must be in 0..3"));
        }
        self.cache_rows
            .checked_add(draft_step)
            .ok_or_else(|| invalid("Qwen3.5 MTP physical row overflow"))
    }

    pub fn rope_position_for_draft(self, draft_step: usize) -> Result<usize> {
        if draft_step >= QWEN35_MTP_DRAFT_STEPS {
            return Err(invalid("Qwen3.5 MTP draft depth must be in 0..3"));
        }
        self.logical_position
            .checked_add(draft_step)
            .ok_or_else(|| invalid("Qwen3.5 MTP RoPE position overflow"))
    }

    fn after_commit(self, committed_rows: usize) -> Result<Self> {
        Ok(Self {
            logical_position: self
                .logical_position
                .checked_add(committed_rows)
                .ok_or_else(|| invalid("Qwen3.5 MTP logical position overflow"))?,
            cache_rows: self
                .cache_rows
                .checked_add(committed_rows)
                .ok_or_else(|| invalid("Qwen3.5 MTP cache row overflow"))?,
        })
    }

    fn validate(self, capacity: usize) -> Result<()> {
        if self.logical_position == 0
            || self.cache_rows.checked_add(1) != Some(self.logical_position)
        {
            return Err(invalid(
                "Qwen3.5 MTP logical position must equal BF16 KV rows plus one",
            ));
        }
        if self.cache_rows > capacity {
            return Err(invalid("Qwen3.5 MTP BF16 KV cache exceeds capacity"));
        }
        Ok(())
    }
}

/// One transactional BF16 MTP KV bank.
#[derive(Debug)]
pub struct Qwen35MtpState {
    key: Tensor,
    value: Tensor,
    capacity: usize,
    position: Option<MtpPositionContract>,
    generation: u64,
}

impl Qwen35MtpState {
    pub fn position(&self) -> Option<MtpPositionContract> {
        self.position
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PreparedMtpPrefill {
    generation: u64,
    position: MtpPositionContract,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PreparedMtpCommit {
    generation: u64,
    position: MtpPositionContract,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PendingDraft {
    shadow_generation: u64,
    start: MtpPositionContract,
}

/// Complete native proposer backend. All large tensors remain on CUDA. Draft
/// argmax runs on device and only the three compact I32 proposal IDs cross to
/// the host once the sequential chain has finished.
pub struct Qwen35MtpRuntime {
    geometry: AttentionGeometry,
    rms_norm_eps: f32,
    rope_theta: f32,
    scratch: Option<MtpScratch>,
    prefill_scratch: Option<MtpPrefillScratch>,
    pending_draft: Option<PendingDraft>,
}

impl Qwen35MtpRuntime {
    pub fn new(config: &Qwen35MoeConfig) -> Result<Self> {
        config.validate()?;
        let text = &config.text_config;
        let geometry = AttentionGeometry {
            query_heads: text.num_attention_heads,
            kv_heads: text.num_key_value_heads,
            head_dim: text.head_dim,
            rotary_dim: text.rotary_dim(),
            query_projection_width: text.q_projection_width(),
            kv_projection_width: text.kv_projection_width(),
            output_gate: text.attn_output_gate,
        };
        validate_checkpoint_geometry(geometry)?;
        if text.mtp_num_hidden_layers != 1 || text.mtp_use_dedicated_embeddings {
            return Err(invalid(
                "Qwen3.5 runtime requires one MTP layer sharing the base embedding/head",
            ));
        }
        Ok(Self {
            geometry,
            rms_norm_eps: text.rms_norm_eps,
            rope_theta: text.rope_parameters.rope_theta,
            scratch: None,
            prefill_scratch: None,
            pending_draft: None,
        })
    }

    fn ensure_scratch(&mut self, ctx: &CudaContext) -> Result<&mut MtpScratch> {
        if self.scratch.is_none() {
            self.scratch = Some(MtpScratch::new(ctx, self.geometry)?);
        }
        Ok(self.scratch.as_mut().expect("MTP scratch was initialized"))
    }

    fn ensure_prefill_scratch(&mut self, ctx: &CudaContext) -> Result<&mut MtpPrefillScratch> {
        if self.prefill_scratch.is_none() {
            self.prefill_scratch = Some(MtpPrefillScratch::new(
                ctx,
                self.geometry,
                MTP_PREFILL_CHUNK,
            )?);
        }
        Ok(self
            .prefill_scratch
            .as_mut()
            .expect("MTP prefill scratch was initialized"))
    }

    fn validate_weights(&self, ctx: &CudaContext, weights: &Qwen35DeviceWeights) -> Result<()> {
        validate_tensor(
            &weights.embedding,
            ctx,
            DType::BF16,
            &[VOCAB, HIDDEN],
            "embedding",
        )?;
        if weights.lm_head.input_features != HIDDEN || weights.lm_head.output_features != VOCAB {
            return Err(invalid("Qwen3.6 lm_head geometry mismatch"));
        }
        validate_mtp_weights(ctx, &weights.mtp, self.geometry)
    }
}

impl MtpBackend for Qwen35MtpRuntime {
    type State = Qwen35MtpState;
    type PreparedPrefill = PreparedMtpPrefill;
    type PreparedCommit = PreparedMtpCommit;

    fn allocate_state(&mut self, ctx: &CudaContext, capacity: usize) -> Result<Self::State> {
        if capacity == 0 {
            return Err(invalid("Qwen3.5 MTP cache capacity must be non-zero"));
        }
        self.ensure_scratch(ctx)?;
        Ok(Qwen35MtpState {
            key: zeros(
                ctx,
                vec![capacity, self.geometry.kv_heads, self.geometry.head_dim],
                QWEN35_MTP_KV_DTYPE,
            )?,
            value: zeros(
                ctx,
                vec![capacity, self.geometry.kv_heads, self.geometry.head_dim],
                QWEN35_MTP_KV_DTYPE,
            )?,
            capacity,
            position: None,
            generation: 0,
        })
    }

    fn reset_state(&mut self, ctx: &CudaContext, state: &mut Self::State) -> Result<()> {
        validate_state_storage(ctx, state, self.geometry)?;
        CudaBuffer::from_tensor(&state.key)
            .map_err(Error::Cuda)?
            .zero_async(ctx.stream())
            .map_err(Error::Cuda)?;
        CudaBuffer::from_tensor(&state.value)
            .map_err(Error::Cuda)?
            .zero_async(ctx.stream())
            .map_err(Error::Cuda)?;
        state.position = None;
        state.generation = state
            .generation
            .checked_add(1)
            .ok_or_else(|| invalid("Qwen3.5 MTP state generation overflow"))?;
        self.pending_draft = None;
        Ok(())
    }

    fn copy_state(
        &mut self,
        ctx: &CudaContext,
        source: &Self::State,
        destination: &mut Self::State,
    ) -> Result<()> {
        validate_state_storage(ctx, source, self.geometry)?;
        validate_state_storage(ctx, destination, self.geometry)?;
        if let Some(position) = source.position {
            position.validate(source.capacity)?;
            if position.cache_rows > destination.capacity {
                return Err(invalid(
                    "Qwen3.5 MTP destination cache cannot hold the source prefix",
                ));
            }
            if position.cache_rows > 0 {
                let source_key = cache_prefix_view(
                    &source.key,
                    position.cache_rows,
                    self.geometry.kv_heads,
                    self.geometry.head_dim,
                )?;
                let destination_key = cache_prefix_view(
                    &destination.key,
                    position.cache_rows,
                    self.geometry.kv_heads,
                    self.geometry.head_dim,
                )?;
                let source_value = cache_prefix_view(
                    &source.value,
                    position.cache_rows,
                    self.geometry.kv_heads,
                    self.geometry.head_dim,
                )?;
                let destination_value = cache_prefix_view(
                    &destination.value,
                    position.cache_rows,
                    self.geometry.kv_heads,
                    self.geometry.head_dim,
                )?;
                transfers::copy_cuda_to_cuda(ctx, &source_key, &destination_key)?;
                transfers::copy_cuda_to_cuda(ctx, &source_value, &destination_value)?;
            }
        }
        destination.position = source.position;
        destination.generation = source
            .generation
            .checked_add(1)
            .ok_or_else(|| invalid("Qwen3.5 MTP shadow generation overflow"))?;
        self.pending_draft = None;
        Ok(())
    }

    fn prepare_prefill(
        &mut self,
        ctx: &CudaContext,
        weights: &Qwen35DeviceWeights,
        shadow: &mut Self::State,
        prompt_tokens: &[TokenId],
        target_hidden: &Tensor,
    ) -> Result<Self::PreparedPrefill> {
        self.validate_weights(ctx, weights)?;
        validate_state_storage(ctx, shadow, self.geometry)?;
        let position = MtpPositionContract::after_prompt(prompt_tokens.len())?;
        position.validate(shadow.capacity)?;
        validate_tensor(
            target_hidden,
            ctx,
            DType::BF16,
            &[prompt_tokens.len(), HIDDEN],
            "target prompt hidden",
        )?;

        self.reset_state(ctx, shadow)?;
        if position.cache_rows > 0 {
            let shifted_tokens = prompt_tokens[1..]
                .iter()
                .map(|&token| {
                    i32::try_from(token)
                        .map_err(|_| invalid(format!("Qwen3.5 MTP token {token} exceeds i32")))
                })
                .collect::<Result<Vec<_>>>()?;
            let shifted_positions = (1..prompt_tokens.len())
                .map(|position| {
                    i32::try_from(position)
                        .map_err(|_| invalid("Qwen3.5 MTP prompt position exceeds i32"))
                })
                .collect::<Result<Vec<_>>>()?;
            let shifted_tokens = transfers::to_cuda(
                &Tensor::from_i32(vec![position.cache_rows], &shifted_tokens)?,
                ctx.device_id(),
            )?;
            let shifted_positions = transfers::to_cuda(
                &Tensor::from_i32(vec![position.cache_rows], &shifted_positions)?,
                ctx.device_id(),
            )?;
            for start in (0..position.cache_rows).step_by(MTP_PREFILL_CHUNK) {
                let rows = (position.cache_rows - start).min(MTP_PREFILL_CHUNK);
                let base_hidden = rows_view(target_hidden, start, rows, HIDDEN)?;
                let tokens = vector_view(&shifted_tokens, start, rows)?;
                let rope_positions = vector_view(&shifted_positions, start, rows)?;
                let geometry = self.geometry;
                let eps = self.rms_norm_eps;
                let theta = self.rope_theta;
                let scratch = self.ensure_prefill_scratch(ctx)?;
                prefill_kv_chunk(
                    ctx,
                    weights,
                    shadow,
                    scratch,
                    geometry,
                    eps,
                    theta,
                    &base_hidden,
                    &tokens,
                    &rope_positions,
                    start,
                )?;
            }
        }
        ctx.synchronize().map_err(Error::Cuda)?;
        shadow.position = Some(position);
        self.pending_draft = None;
        Ok(PreparedMtpPrefill {
            generation: shadow.generation,
            position,
        })
    }

    fn prepare_append_prefill(
        &mut self,
        ctx: &CudaContext,
        weights: &Qwen35DeviceWeights,
        committed: &Self::State,
        shadow: &mut Self::State,
        chunk_tokens: &[TokenId],
        previous_target_hidden: &Tensor,
        target_hidden: &Tensor,
    ) -> Result<Self::PreparedPrefill> {
        self.validate_weights(ctx, weights)?;
        validate_state_storage(ctx, committed, self.geometry)?;
        validate_state_storage(ctx, shadow, self.geometry)?;
        if chunk_tokens.is_empty() {
            return Err(invalid("Qwen3.5 MTP append prefill requires tokens"));
        }
        validate_tensor(
            previous_target_hidden,
            ctx,
            DType::BF16,
            &[1, HIDDEN],
            "previous target hidden",
        )?;
        validate_tensor(
            target_hidden,
            ctx,
            DType::BF16,
            &[chunk_tokens.len(), HIDDEN],
            "target append hidden",
        )?;

        let start = committed
            .position
            .ok_or_else(|| invalid("Qwen3.5 MTP append prefill requires committed state"))?;
        start.validate(committed.capacity)?;
        let position = start.after_commit(chunk_tokens.len())?;
        position.validate(committed.capacity)?;
        self.copy_state(ctx, committed, shadow)?;

        let token_values = chunk_tokens
            .iter()
            .map(|&token| {
                i32::try_from(token)
                    .map_err(|_| invalid(format!("Qwen3.5 MTP token {token} exceeds i32")))
            })
            .collect::<Result<Vec<_>>>()?;
        let position_values = (start.logical_position..position.logical_position)
            .map(|position| {
                i32::try_from(position)
                    .map_err(|_| invalid("Qwen3.5 MTP append position exceeds i32"))
            })
            .collect::<Result<Vec<_>>>()?;
        let tokens = transfers::to_cuda(
            &Tensor::from_i32(vec![chunk_tokens.len()], &token_values)?,
            ctx.device_id(),
        )?;
        let positions = transfers::to_cuda(
            &Tensor::from_i32(vec![chunk_tokens.len()], &position_values)?,
            ctx.device_id(),
        )?;

        for start_row in (0..chunk_tokens.len()).step_by(MTP_PREFILL_CHUNK) {
            let rows = (chunk_tokens.len() - start_row).min(MTP_PREFILL_CHUNK);
            let tokens = vector_view(&tokens, start_row, rows)?;
            let rope_positions = vector_view(&positions, start_row, rows)?;
            let geometry = self.geometry;
            let eps = self.rms_norm_eps;
            let theta = self.rope_theta;
            let scratch = self.ensure_prefill_scratch(ctx)?;
            let base_hidden = rows_view(&scratch.base_hidden, 0, rows, HIDDEN)?;
            if start_row == 0 {
                let first = rows_view(&scratch.base_hidden, 0, 1, HIDDEN)?;
                transfers::copy_cuda_to_cuda(ctx, previous_target_hidden, &first)?;
                if rows > 1 {
                    let source = rows_view(target_hidden, 0, rows - 1, HIDDEN)?;
                    let destination = rows_view(&scratch.base_hidden, 1, rows - 1, HIDDEN)?;
                    transfers::copy_cuda_to_cuda(ctx, &source, &destination)?;
                }
            } else {
                let source = rows_view(target_hidden, start_row - 1, rows, HIDDEN)?;
                transfers::copy_cuda_to_cuda(ctx, &source, &base_hidden)?;
            }
            prefill_kv_chunk(
                ctx,
                weights,
                shadow,
                scratch,
                geometry,
                eps,
                theta,
                &base_hidden,
                &tokens,
                &rope_positions,
                start.cache_rows + start_row,
            )?;
        }
        ctx.synchronize().map_err(Error::Cuda)?;
        shadow.position = Some(position);
        self.pending_draft = None;
        Ok(PreparedMtpPrefill {
            generation: shadow.generation,
            position,
        })
    }

    fn publish_prefill(
        &mut self,
        committed: &mut Self::State,
        shadow: &mut Self::State,
        prepared: Self::PreparedPrefill,
    ) {
        debug_assert_eq!(shadow.generation, prepared.generation);
        debug_assert_eq!(shadow.position, Some(prepared.position));
        std::mem::swap(committed, shadow);
        committed.position = Some(prepared.position);
        self.pending_draft = None;
    }

    fn draft(
        &mut self,
        ctx: &CudaContext,
        weights: &Qwen35DeviceWeights,
        committed: &Self::State,
        shadow: &mut Self::State,
        pending: TokenId,
        start_position: usize,
        target_hidden: &Tensor,
    ) -> Result<[TokenId; QWEN35_MTP_DRAFT_STEPS]> {
        self.validate_weights(ctx, weights)?;
        validate_tensor(
            target_hidden,
            ctx,
            DType::BF16,
            &[1, HIDDEN],
            "target hidden",
        )?;
        let start = committed
            .position
            .ok_or_else(|| invalid("Qwen3.5 MTP draft requires committed prompt state"))?;
        if start.logical_position != start_position {
            return Err(invalid(format!(
                "Qwen3.5 MTP/base position divergence: MTP {}, base {start_position}",
                start.logical_position
            )));
        }
        start.validate(committed.capacity)?;
        let required_rows = start
            .cache_rows
            .checked_add(QWEN35_MTP_DRAFT_STEPS)
            .ok_or_else(|| invalid("Qwen3.5 MTP draft capacity overflow"))?;
        if required_rows > committed.capacity {
            return Err(invalid("Qwen3.5 MTP draft exceeds BF16 KV capacity"));
        }
        self.copy_state(ctx, committed, shadow)?;

        let geometry = self.geometry;
        let eps = self.rms_norm_eps;
        let theta = self.rope_theta;
        let scratch = self.ensure_scratch(ctx)?;
        write_i32(&scratch.token, pending)?;
        let draft_positions = (0..QWEN35_MTP_DRAFT_STEPS)
            .map(|depth| {
                i32::try_from(start.rope_position_for_draft(depth)?)
                    .map_err(|_| invalid("Qwen3.5 MTP RoPE position exceeds i32"))
            })
            .collect::<Result<Vec<_>>>()?;
        let draft_positions = transfers::to_cuda(
            &Tensor::from_i32(vec![QWEN35_MTP_DRAFT_STEPS], &draft_positions)?,
            ctx.device_id(),
        )?;
        let mut base_hidden = target_hidden.clone();
        for depth in 0..QWEN35_MTP_DRAFT_STEPS {
            let token_input = scratch.token.clone();
            let rope_position = element_view(&draft_positions, depth)?;
            forward_one(
                ctx,
                weights,
                shadow,
                scratch,
                geometry,
                eps,
                theta,
                &base_hidden,
                &token_input,
                &rope_position,
                start.physical_row_for_draft(depth)?,
                depth,
                true,
            )?;
            let proposal_slot = element_view(&scratch.proposals, depth)?;
            transfers::copy_cuda_to_cuda(ctx, &scratch.next_token, &proposal_slot)?;
            if depth + 1 < QWEN35_MTP_DRAFT_STEPS {
                transfers::copy_cuda_to_cuda(ctx, &scratch.next_token, &scratch.token)?;
                base_hidden = scratch.final_hidden.clone();
            }
        }
        ctx.synchronize().map_err(Error::Cuda)?;
        let host = transfers::to_cpu(&scratch.proposals)?;
        let proposals = host
            .as_i32()?
            .iter()
            .map(|&token| {
                u32::try_from(token)
                    .map_err(|_| invalid(format!("Qwen3.5 MTP argmax returned {token}")))
            })
            .collect::<Result<Vec<_>>>()?
            .try_into()
            .map_err(|_| invalid("Qwen3.5 MTP proposal count changed"))?;
        self.pending_draft = Some(PendingDraft {
            shadow_generation: shadow.generation,
            start,
        });
        Ok(proposals)
    }

    fn prepare_commit(
        &mut self,
        ctx: &CudaContext,
        weights: &Qwen35DeviceWeights,
        shadow: &mut Self::State,
        contract: MtpCommit,
    ) -> Result<Self::PreparedCommit> {
        self.validate_weights(ctx, weights)?;
        validate_state_storage(ctx, shadow, self.geometry)?;
        let pending = self
            .pending_draft
            .ok_or_else(|| invalid("Qwen3.5 MTP commit has no prepared draft"))?;
        let expected_verified_rows = if contract.materialize_final_proposal {
            QWEN35_MTP_DRAFT_STEPS + 1
        } else {
            QWEN35_MTP_DRAFT_STEPS
        };
        let expected_verified_end = pending
            .start
            .logical_position
            .checked_add(expected_verified_rows)
            .ok_or_else(|| invalid("Qwen3.5 MTP verified range overflow"))?;
        if pending.shadow_generation != shadow.generation
            || contract.cache_range.start != pending.start.logical_position
            || contract.cache_range.verified_end != expected_verified_end
            || contract.cache_range.committed_end < contract.cache_range.start
            || contract.cache_range.committed_end > contract.cache_range.verified_end
        {
            return Err(invalid("Qwen3.5 MTP commit does not match prepared shadow"));
        }
        let committed_rows = contract.cache_range.committed_rows();
        if committed_rows == 0 || committed_rows > QWEN35_MTP_DRAFT_STEPS + 1 {
            return Err(invalid("Qwen3.5 MTP commit row count is outside 1..=4"));
        }
        if contract.materialize_final_proposal {
            if committed_rows != QWEN35_MTP_DRAFT_STEPS + 1 {
                return Err(invalid(
                    "Qwen3.5 MTP final proposal is materialized only on full accept plus bonus",
                ));
            }
            let materialize_row = pending
                .start
                .cache_rows
                .checked_add(QWEN35_MTP_DRAFT_STEPS)
                .ok_or_else(|| invalid("Qwen3.5 MTP final cache row overflow"))?;
            if materialize_row >= shadow.capacity {
                return Err(invalid("Qwen3.5 MTP final materialization exceeds cache"));
            }
            let geometry = self.geometry;
            let eps = self.rms_norm_eps;
            let theta = self.rope_theta;
            let scratch = self.ensure_scratch(ctx)?;
            transfers::copy_cuda_to_cuda(ctx, &scratch.next_token, &scratch.token)?;
            write_i32(
                &scratch.position,
                u32::try_from(
                    pending
                        .start
                        .logical_position
                        .checked_add(QWEN35_MTP_DRAFT_STEPS)
                        .ok_or_else(|| invalid("Qwen3.5 MTP final RoPE position overflow"))?,
                )
                .map_err(|_| invalid("Qwen3.5 MTP final RoPE position exceeds i32"))?,
            )?;
            let base_hidden = scratch.final_hidden.clone();
            let token_input = scratch.token.clone();
            let rope_position = scratch.position.clone();
            // The last proposal is consumed as an input but no fourth draft
            // logit is needed. This mirrors the tuned Qwen3.8 protocol.
            forward_one(
                ctx,
                weights,
                shadow,
                scratch,
                geometry,
                eps,
                theta,
                &base_hidden,
                &token_input,
                &rope_position,
                materialize_row,
                QWEN35_MTP_DRAFT_STEPS,
                false,
            )?;
        } else if committed_rows > QWEN35_MTP_DRAFT_STEPS {
            return Err(invalid("Qwen3.5 MTP four-row commit needs materialization"));
        }
        ctx.synchronize().map_err(Error::Cuda)?;
        let position = pending.start.after_commit(committed_rows)?;
        position.validate(shadow.capacity)?;
        shadow.position = Some(position);
        Ok(PreparedMtpCommit {
            generation: shadow.generation,
            position,
        })
    }

    fn publish_commit(
        &mut self,
        committed: &mut Self::State,
        shadow: &mut Self::State,
        prepared: Self::PreparedCommit,
    ) {
        debug_assert_eq!(shadow.generation, prepared.generation);
        debug_assert_eq!(shadow.position, Some(prepared.position));
        std::mem::swap(committed, shadow);
        committed.position = Some(prepared.position);
        self.pending_draft = None;
    }
}

/// Resolved serving policy admitted by the exact argmax MTP protocol.
/// Sampling, penalties and log-probability requests must use the base scalar
/// generation path instead. Temperature/top-k/top-p do not alter an explicit
/// greedy selection and therefore do not exclude the exact MTP path.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MtpGreedyPolicy {
    pub max_new_tokens: usize,
    pub eos_token_ids: Vec<TokenId>,
}

impl MtpGreedyPolicy {
    pub fn from_generation_options(options: &GenerationOptions) -> Result<Self> {
        let complete = GenerationOptions::apxinf_defaults().overlay(options);
        let greedy = complete.sampling_mode == Some(SamplingMode::Greedy)
            || complete.temperature == Some(0.0);
        let neutral_penalties = complete.repetition_penalty == Some(1.0)
            && complete.frequency_penalty == Some(0.0)
            && complete.presence_penalty == Some(0.0);
        let no_logprob = complete.return_logprob == Some(false);
        if !greedy || !neutral_penalties || !no_logprob {
            return Err(invalid(
                "Qwen3.5 MTP accepts only greedy generation without penalties or logprobs; use base scalar generation",
            ));
        }
        let max_new_tokens = complete
            .max_new_tokens
            .ok_or_else(|| invalid("Qwen3.5 generation token budget is missing"))?;
        if max_new_tokens == 0 {
            return Err(invalid("Qwen3.5 MTP requires at least one generated token"));
        }
        Ok(Self {
            max_new_tokens,
            eos_token_ids: complete.eos_token_ids.unwrap_or_default(),
        })
    }
}

/// Result of the atomic prompt transaction. `first_token` is already emitted
/// to the caller but is not yet consumed by the target decoder; it seeds the
/// first speculative verification round.
pub struct MtpPrefillOutcome {
    pub target: DecoderOutput,
    pub first_token: TokenId,
}

/// One committed speculative round. `outcome.emitted` excludes the prior
/// pending token and ends with the next pending token.
pub struct MtpRoundOutput {
    pub outcome: RoundOutcome,
}

/// Owns target and MTP committed/shadow banks and publishes them together.
///
/// The only public speculative entry point is [`Self::run_greedy_round`]. A
/// serving layer must route non-greedy requests to scalar target decoding.
pub struct Qwen35MtpCoordinator<B: FullAttentionBackend, M: MtpBackend> {
    target: Qwen35Session<B>,
    mtp: M,
    committed_mtp: M::State,
    shadow_mtp: M::State,
    schedule: Option<Qwen35MtpSchedule>,
    policy: Option<MtpGreedyPolicy>,
    pending: Option<TokenId>,
    target_hidden: Option<Tensor>,
}

/// Device-resident checkpoint for an MTP-ready text prefix.
///
/// Target attention/GDN state and the draft layer's BF16 KV state are kept
/// together so restoring a prefix cannot create target/draft position drift.
pub struct Qwen35ContextSnapshot<C, S> {
    target: Qwen35SessionSnapshot<C>,
    mtp: S,
    target_hidden: Option<Tensor>,
    position: usize,
}

impl<B: FullAttentionBackend, M: MtpBackend> Qwen35MtpCoordinator<B, M> {
    pub fn new(ctx: &CudaContext, target: Qwen35Session<B>, mut mtp: M) -> Result<Self> {
        let capacity = target.capacity();
        let mut committed_mtp = mtp.allocate_state(ctx, capacity)?;
        let mut shadow_mtp = mtp.allocate_state(ctx, capacity)?;
        mtp.reset_state(ctx, &mut committed_mtp)?;
        mtp.reset_state(ctx, &mut shadow_mtp)?;
        ctx.synchronize().map_err(Error::Cuda)?;
        Ok(Self {
            target,
            mtp,
            committed_mtp,
            shadow_mtp,
            schedule: None,
            policy: None,
            pending: None,
            target_hidden: None,
        })
    }

    pub fn target(&self) -> &Qwen35Session<B> {
        &self.target
    }

    /// Reuse the target cache for scalar fallback (sampling, penalties, or
    /// logprobs). Any scalar mutation intentionally makes the speculative
    /// schedule stale; a later MTP request must call [`Self::prefill_greedy`]
    /// again before drafting.
    pub fn target_mut(&mut self) -> &mut Qwen35Session<B> {
        &mut self.target
    }

    /// Allocate one reusable context checkpoint. No prefix is visible until a
    /// successful [`Self::capture_context`] call publishes its position.
    pub fn allocate_context_snapshot(
        &mut self,
        ctx: &CudaContext,
        decoder: &Qwen35Decoder,
        weights: &Qwen35DeviceWeights,
        capacity: usize,
    ) -> Result<Qwen35ContextSnapshot<B::Cache, M::State>> {
        if capacity == 0 || capacity > self.target.capacity() {
            return Err(invalid("Qwen3.5 context snapshot capacity is invalid"));
        }
        let target = self
            .target
            .allocate_snapshot(ctx, decoder, weights, capacity)?;
        let mtp = self.mtp.allocate_state(ctx, capacity)?;
        Ok(Qwen35ContextSnapshot {
            target,
            mtp,
            target_hidden: None,
            position: 0,
        })
    }

    /// Save the currently committed prompt endpoint. Generation may continue
    /// afterwards without mutating the checkpoint.
    pub fn capture_context(
        &mut self,
        ctx: &CudaContext,
        snapshot: &mut Qwen35ContextSnapshot<B::Cache, M::State>,
    ) -> Result<()> {
        let position = self.target.position();
        let target_hidden = self
            .target_hidden
            .as_ref()
            .ok_or_else(|| invalid("Qwen3.5 context capture requires MTP prefill state"))?
            .clone();
        self.target.capture_snapshot(ctx, &mut snapshot.target)?;
        self.mtp
            .copy_state(ctx, &self.committed_mtp, &mut snapshot.mtp)?;
        ctx.synchronize().map_err(Error::Cuda)?;
        snapshot.target_hidden = Some(target_hidden);
        snapshot.position = position;
        Ok(())
    }

    /// Restore a prior prompt endpoint and discard generation bookkeeping.
    /// The caller must append at least one suffix token before starting a new
    /// speculative schedule.
    pub fn restore_context(
        &mut self,
        ctx: &CudaContext,
        snapshot: &Qwen35ContextSnapshot<B::Cache, M::State>,
    ) -> Result<()> {
        if snapshot.position == 0 || snapshot.target_hidden.is_none() {
            return Err(invalid("Qwen3.5 context snapshot is empty"));
        }
        self.target.restore_snapshot(ctx, &snapshot.target)?;
        if self.target.position() != snapshot.position {
            return Err(invalid("Qwen3.5 context snapshot position mismatch"));
        }
        self.mtp
            .copy_state(ctx, &snapshot.mtp, &mut self.committed_mtp)?;
        ctx.synchronize().map_err(Error::Cuda)?;
        self.schedule = None;
        self.policy = None;
        self.pending = None;
        self.target_hidden = snapshot.target_hidden.clone();
        Ok(())
    }

    /// Clear target and both BF16 MTP cache banks and discard all generation
    /// bookkeeping. No old pending token or hidden row survives the reset.
    pub fn reset(&mut self, ctx: &CudaContext) -> Result<()> {
        self.target.reset(ctx)?;
        self.mtp.reset_state(ctx, &mut self.committed_mtp)?;
        self.mtp.reset_state(ctx, &mut self.shadow_mtp)?;
        ctx.synchronize().map_err(Error::Cuda)?;
        self.schedule = None;
        self.policy = None;
        self.pending = None;
        self.target_hidden = None;
        Ok(())
    }

    pub fn schedule(&self) -> Option<&Qwen35MtpSchedule> {
        self.schedule.as_ref()
    }

    pub fn is_finished(&self) -> bool {
        let Some(schedule) = &self.schedule else {
            return false;
        };
        schedule.is_finished()
            || self.pending.is_some_and(|token| {
                self.policy
                    .as_ref()
                    .is_some_and(|policy| policy.eos_token_ids.contains(&token))
            })
    }

    /// Prepare target prompt state, teacher-force the shifted MTP prompt, and
    /// publish both only after every fallible kernel, compact argmax transfer,
    /// schedule construction and stream synchronization have succeeded.
    pub fn prefill_greedy(
        &mut self,
        ctx: &CudaContext,
        decoder: &Qwen35Decoder,
        weights: &Qwen35DeviceWeights,
        prompt_tokens: &[TokenId],
        options: &GenerationOptions,
    ) -> Result<MtpPrefillOutcome> {
        let policy = MtpGreedyPolicy::from_generation_options(options)?;
        let prepared_target =
            self.target
                .prepare_target_prefill(ctx, decoder, weights, prompt_tokens)?;
        let target_hidden_rows = &prepared_target.output().hidden_states;
        let prepared_mtp = self.mtp.prepare_prefill(
            ctx,
            weights,
            &mut self.shadow_mtp,
            prompt_tokens,
            target_hidden_rows,
        )?;
        let target_hidden = prepared_target.output().last_hidden_state()?;
        let first = prepared_target.output().greedy_token_ids(ctx)?;
        let first_token = *first
            .first()
            .ok_or_else(|| invalid("Qwen3.5 target prefill produced no token"))?;
        let schedule = Qwen35MtpSchedule::new(
            prompt_tokens.len(),
            prompt_tokens.len(),
            policy.max_new_tokens,
            1,
            policy.eos_token_ids.clone(),
        )
        .map_err(schedule_error)?;
        ctx.synchronize().map_err(Error::Cuda)?;

        // Target publication validates before it swaps. MTP publication is an
        // infallible state swap, so neither tree can be left half-published.
        let target = self.target.publish_target_prefill(prepared_target)?;
        self.mtp
            .publish_prefill(&mut self.committed_mtp, &mut self.shadow_mtp, prepared_mtp);
        self.schedule = Some(schedule);
        self.policy = Some(policy);
        self.pending = Some(first_token);
        self.target_hidden = Some(target_hidden);
        Ok(MtpPrefillOutcome {
            target,
            first_token,
        })
    }

    /// Extend an already prepared prompt with a true batched prefill chunk,
    /// then restart speculative generation bookkeeping at the new end.
    pub fn append_prefill_greedy(
        &mut self,
        ctx: &CudaContext,
        decoder: &Qwen35Decoder,
        weights: &Qwen35DeviceWeights,
        chunk_tokens: &[TokenId],
        options: &GenerationOptions,
    ) -> Result<MtpPrefillOutcome> {
        let policy = MtpGreedyPolicy::from_generation_options(options)?;
        let previous_target_hidden = self
            .target_hidden
            .as_ref()
            .ok_or_else(|| invalid("Qwen3.5 MTP append prefill requires prepared target state"))?
            .clone();
        let prepared_target =
            self.target
                .prepare_target_append_prefill(ctx, decoder, weights, chunk_tokens)?;
        let target_hidden_rows = &prepared_target.output().hidden_states;
        let prepared_mtp = self.mtp.prepare_append_prefill(
            ctx,
            weights,
            &self.committed_mtp,
            &mut self.shadow_mtp,
            chunk_tokens,
            &previous_target_hidden,
            target_hidden_rows,
        )?;
        let target_hidden = prepared_target.output().last_hidden_state()?;
        let first = prepared_target.output().greedy_token_ids(ctx)?;
        let first_token = *first
            .first()
            .ok_or_else(|| invalid("Qwen3.5 target append prefill produced no token"))?;
        let prompt_len = prepared_target.output().end_position;
        let schedule = Qwen35MtpSchedule::new(
            prompt_len,
            prompt_len,
            policy.max_new_tokens,
            1,
            policy.eos_token_ids.clone(),
        )
        .map_err(schedule_error)?;
        ctx.synchronize().map_err(Error::Cuda)?;

        let target = self.target.publish_target_prefill(prepared_target)?;
        self.mtp
            .publish_prefill(&mut self.committed_mtp, &mut self.shadow_mtp, prepared_mtp);
        self.schedule = Some(schedule);
        self.policy = Some(policy);
        self.pending = Some(first_token);
        self.target_hidden = Some(target_hidden);
        Ok(MtpPrefillOutcome {
            target,
            first_token,
        })
    }

    /// Execute draft(3) -> canonical scalar target verify(4) -> schedule ->
    /// target/MTP prepare -> dual publish. All device work and the schedule's
    /// fallible transition are completed before either visible state changes.
    pub fn run_greedy_round(
        &mut self,
        ctx: &CudaContext,
        decoder: &Qwen35Decoder,
        weights: &Qwen35DeviceWeights,
    ) -> Result<MtpRoundOutput> {
        if self.is_finished() {
            return Err(invalid("Qwen3.5 MTP generation is finished"));
        }
        let pending = self
            .pending
            .ok_or_else(|| invalid("Qwen3.5 MTP generation has no pending token"))?;
        let target_hidden = self
            .target_hidden
            .as_ref()
            .ok_or_else(|| invalid("Qwen3.5 MTP generation has no target hidden state"))?;
        let schedule = self
            .schedule
            .as_ref()
            .ok_or_else(|| invalid("Qwen3.5 MTP generation has no schedule"))?;
        let start = schedule.positions().mtp;
        if self.target.position() != start {
            return Err(invalid(format!(
                "Qwen3.5 target/MTP position divergence: target {}, schedule {start}",
                self.target.position()
            )));
        }
        // APXINF_QWEN35_MTP_TIMING=1 reports where a speculative round spends
        // its wall clock. Each phase is bracketed by a synchronize so the
        // numbers are attributable; that extra serialization means the totals
        // are only valid relative to each other, not as absolute throughput.
        let timing = std::env::var_os("APXINF_QWEN35_MTP_TIMING").is_some();
        let mark = |label: &str, at: std::time::Instant| {
            if timing {
                eprintln!("[mtp-round] {label} {:.3} ms", at.elapsed().as_secs_f64() * 1e3);
            }
        };
        if timing {
            ctx.synchronize().map_err(Error::Cuda)?;
        }
        let phase = std::time::Instant::now();
        let proposals = self.mtp.draft(
            ctx,
            weights,
            &self.committed_mtp,
            &mut self.shadow_mtp,
            pending,
            start,
            target_hidden,
        )?;
        if timing {
            ctx.synchronize().map_err(Error::Cuda)?;
        }
        mark("draft", phase);
        let phase = std::time::Instant::now();
        let ticket = schedule
            .begin_round(pending, &proposals)
            .map_err(schedule_error)?;
        let verification_inputs = *ticket.verification_inputs();
        let verification =
            self.target
                .verify_target_chunk(ctx, decoder, weights, &verification_inputs)?;
        if timing {
            ctx.synchronize().map_err(Error::Cuda)?;
        }
        mark("verify", phase);
        let phase = std::time::Instant::now();
        let plan = schedule
            .finish_round(ticket, verification.predictions())
            .map_err(schedule_error)?;
        let contract = plan.commit_contract();

        // Preflight the host schedule transition on a clone. After this point
        // the final schedule assignment cannot fail.
        let mut next_schedule = schedule.clone();
        let outcome = next_schedule.commit(plan).map_err(schedule_error)?;
        let prepared_target = self.target.prepare_target_commit(
            ctx,
            decoder,
            weights,
            verification,
            contract.target.kv_range.committed_rows(),
        )?;
        let next_target_hidden = prepared_target.output().last_hidden_state()?;
        let prepared_mtp =
            self.mtp
                .prepare_commit(ctx, weights, &mut self.shadow_mtp, contract.mtp)?;
        ctx.synchronize().map_err(Error::Cuda)?;
        mark("commit", phase);

        // The published target output aliases the session's persistent decode
        // scratch. Its final hidden row was captured above for the next MTP
        // round, and no public result may retain that ephemeral logits view.
        self.target.publish_target_commit(prepared_target)?;
        self.mtp
            .publish_commit(&mut self.committed_mtp, &mut self.shadow_mtp, prepared_mtp);
        self.schedule = Some(next_schedule);
        self.pending = outcome.emitted.last().copied();
        self.target_hidden = Some(next_target_hidden);
        Ok(MtpRoundOutput { outcome })
    }
}

/// Reusable bounded workspace for shifted-token teacher forcing.
///
/// Only K/V materialization is required during prompt ingestion: every MTP
/// row receives its base hidden state directly from the target decoder, and
/// later proposal rows depend on the prompt only through this layer's K/V
/// cache. Running attention, routed experts, final norm, and LM head here was
/// therefore pure discarded work.
struct MtpPrefillScratch {
    max_tokens: usize,
    base_hidden: Tensor,
    embedding: Tensor,
    norm_embedding: Tensor,
    norm_hidden: Tensor,
    joined: Tensor,
    hidden: Tensor,
    normalized: Tensor,
}

impl MtpPrefillScratch {
    fn new(ctx: &CudaContext, _geometry: AttentionGeometry, max_tokens: usize) -> Result<Self> {
        if max_tokens == 0 {
            return Err(invalid("Qwen3.5 MTP prefill chunk must be non-zero"));
        }
        Ok(Self {
            max_tokens,
            base_hidden: zeros(ctx, vec![max_tokens, HIDDEN], DType::BF16)?,
            embedding: zeros(ctx, vec![max_tokens, HIDDEN], DType::BF16)?,
            norm_embedding: zeros(ctx, vec![max_tokens, HIDDEN], DType::BF16)?,
            norm_hidden: zeros(ctx, vec![max_tokens, HIDDEN], DType::BF16)?,
            joined: zeros(ctx, vec![max_tokens, 2 * HIDDEN], DType::BF16)?,
            hidden: zeros(ctx, vec![max_tokens, HIDDEN], DType::BF16)?,
            normalized: zeros(ctx, vec![max_tokens, HIDDEN], DType::BF16)?,
        })
    }
}

#[allow(clippy::too_many_arguments)]
fn prefill_kv_chunk(
    ctx: &CudaContext,
    weights: &Qwen35DeviceWeights,
    state: &mut Qwen35MtpState,
    scratch: &mut MtpPrefillScratch,
    geometry: AttentionGeometry,
    eps: f32,
    rope_theta: f32,
    base_hidden: &Tensor,
    tokens: &Tensor,
    positions: &Tensor,
    cache_row: usize,
) -> Result<()> {
    let rows = tokens.shape().dims().first().copied().unwrap_or(0);
    if rows == 0 || rows > scratch.max_tokens {
        return Err(invalid(
            "Qwen3.5 MTP prefill chunk exceeds scratch capacity",
        ));
    }
    if cache_row
        .checked_add(rows)
        .is_none_or(|end| end > state.capacity)
    {
        return Err(invalid("Qwen3.5 MTP prefill chunk exceeds KV capacity"));
    }
    validate_tensor(
        base_hidden,
        ctx,
        DType::BF16,
        &[rows, HIDDEN],
        "MTP prefill base hidden",
    )?;
    validate_tensor(tokens, ctx, DType::I32, &[rows], "MTP prefill tokens")?;
    validate_tensor(positions, ctx, DType::I32, &[rows], "MTP prefill positions")?;

    let embedding = rows_view(&scratch.embedding, 0, rows, HIDDEN)?;
    let norm_embedding = rows_view(&scratch.norm_embedding, 0, rows, HIDDEN)?;
    let norm_hidden = rows_view(&scratch.norm_hidden, 0, rows, HIDDEN)?;
    let joined = rows_view(&scratch.joined, 0, rows, 2 * HIDDEN)?;
    let mut hidden = rows_view(&scratch.hidden, 0, rows, HIDDEN)?;
    let normalized = rows_view(&scratch.normalized, 0, rows, HIDDEN)?;
    let mtp = &weights.mtp;

    ops::embedding_gather(ctx, &weights.embedding, tokens, &embedding)?;
    ops::mlp::rms_norm(
        ctx,
        &embedding,
        &mtp.pre_fc_norm_embedding,
        &norm_embedding,
        eps,
    )?;
    ops::mlp::rms_norm(ctx, base_hidden, &mtp.pre_fc_norm_hidden, &norm_hidden, eps)?;
    transfers::copy_cuda_columns(ctx, &norm_embedding, &joined, 0)?;
    transfers::copy_cuda_columns(ctx, &norm_hidden, &joined, HIDDEN)?;
    ops::gemm(
        ctx,
        ops::GemmArgs::new(&joined, &mtp.fc.weight, &mut hidden),
    )?;
    ops::mlp::rms_norm(ctx, &hidden, &mtp.layer.input_layernorm, &normalized, eps)?;

    let attention = &mtp.layer.attention;
    let key_slot = cache_rows_view(
        &state.key,
        cache_row,
        rows,
        geometry.kv_heads,
        geometry.head_dim,
    )?;
    let value_slot = cache_rows_view(
        &state.value,
        cache_row,
        rows,
        geometry.kv_heads,
        geometry.head_dim,
    )?;
    let mut key_flat = reshape(
        &key_slot,
        vec![rows, geometry.kv_heads * geometry.head_dim],
        DType::BF16,
    )?;
    let mut value_flat = reshape(
        &value_slot,
        vec![rows, geometry.kv_heads * geometry.head_dim],
        DType::BF16,
    )?;
    ops::gemm(
        ctx,
        ops::GemmArgs::new(&normalized, &attention.k_proj.weight, &mut key_flat),
    )?;
    ops::gemm(
        ctx,
        ops::GemmArgs::new(&normalized, &attention.v_proj.weight, &mut value_flat),
    )?;
    let key_heads = reshape(
        &key_slot,
        vec![rows * geometry.kv_heads, geometry.head_dim],
        DType::BF16,
    )?;
    ops::head_rms_norm(ctx, &key_heads, &attention.k_norm, eps)?;
    ops::partial_rope(ctx, &key_slot, positions, geometry.rotary_dim, rope_theta)
}

struct MtpScratch {
    /// Device-resident `[valid_key_tokens, query_start]` for the draft layer
    /// attention, created on first use because the cache capacity is only
    /// known then. Without it the draft attention Spec carries the live
    /// prefix, so every draft step is a fresh tuning key and a full autotune:
    /// measured at 2055 ms for the three draft forwards of one round, against
    /// 20 ms for an entire 40-layer scalar decode step.
    decode_meta: Option<ops::KvCacheDecodeMeta>,
    token: Tensor,
    position: Tensor,
    embedding: Tensor,
    norm_embedding: Tensor,
    norm_hidden: Tensor,
    joined: Tensor,
    hidden: Tensor,
    normalized: Tensor,
    q_fused: Tensor,
    query: Tensor,
    query_gate: Tensor,
    attention_out: Tensor,
    projected: Tensor,
    moe_out: Tensor,
    final_hidden: Tensor,
    logits: Tensor,
    next_token: Tensor,
    proposals: Tensor,
    router_logits: Tensor,
    expert_ids: Tensor,
    expert_weights: Tensor,
    route_intermediate: Tensor,
    route_output: Tensor,
    shared_intermediate: Tensor,
    shared_output: Tensor,
    shared_gate_logit: Tensor,
    edge_workspace: Option<CudaBuffer>,
}

impl MtpScratch {
    fn new(ctx: &CudaContext, geometry: AttentionGeometry) -> Result<Self> {
        Ok(Self {
            decode_meta: None,
            token: zeros(ctx, vec![1], DType::I32)?,
            position: zeros(ctx, vec![1], DType::I32)?,
            embedding: zeros(ctx, vec![1, HIDDEN], DType::BF16)?,
            norm_embedding: zeros(ctx, vec![1, HIDDEN], DType::BF16)?,
            norm_hidden: zeros(ctx, vec![1, HIDDEN], DType::BF16)?,
            joined: zeros(ctx, vec![1, 2 * HIDDEN], DType::BF16)?,
            hidden: zeros(ctx, vec![1, HIDDEN], DType::BF16)?,
            normalized: zeros(ctx, vec![1, HIDDEN], DType::BF16)?,
            q_fused: zeros(ctx, vec![1, geometry.query_projection_width], DType::BF16)?,
            query: zeros(
                ctx,
                vec![1, geometry.query_heads, geometry.head_dim],
                DType::BF16,
            )?,
            query_gate: zeros(
                ctx,
                vec![1, geometry.query_heads, geometry.head_dim],
                DType::BF16,
            )?,
            attention_out: zeros(
                ctx,
                vec![1, geometry.query_heads * geometry.head_dim],
                DType::BF16,
            )?,
            projected: zeros(ctx, vec![1, HIDDEN], DType::BF16)?,
            moe_out: zeros(ctx, vec![1, HIDDEN], DType::BF16)?,
            final_hidden: zeros(ctx, vec![1, HIDDEN], DType::BF16)?,
            logits: zeros(ctx, vec![1, VOCAB], DType::BF16)?,
            next_token: zeros(ctx, vec![1], DType::I32)?,
            proposals: zeros(ctx, vec![QWEN35_MTP_DRAFT_STEPS], DType::I32)?,
            router_logits: zeros(ctx, vec![1, EXPERTS], DType::BF16)?,
            expert_ids: zeros(ctx, vec![TOP_K], DType::I32)?,
            expert_weights: zeros(ctx, vec![TOP_K], DType::F32)?,
            route_intermediate: zeros(ctx, vec![TOP_K, INTERMEDIATE], DType::BF16)?,
            route_output: zeros(ctx, vec![TOP_K, HIDDEN], DType::BF16)?,
            shared_intermediate: zeros(ctx, vec![1, INTERMEDIATE], DType::BF16)?,
            shared_output: zeros(ctx, vec![1, HIDDEN], DType::BF16)?,
            shared_gate_logit: zeros(ctx, vec![1], DType::BF16)?,
            edge_workspace: ops::edge_bf16_decode_moe_available()
                .then(|| {
                    let bytes = ops::edge_bf16_decode_moe_workspace_size(
                        ops::RoutedMoeConfig::new(EXPERTS, TOP_K),
                        HIDDEN,
                        INTERMEDIATE,
                    )?;
                    CudaBuffer::alloc(bytes, ctx.device_id()).map_err(Error::Cuda)
                })
                .transpose()?,
        })
    }
}

#[allow(clippy::too_many_arguments)]
fn forward_one(
    ctx: &CudaContext,
    weights: &Qwen35DeviceWeights,
    state: &mut Qwen35MtpState,
    scratch: &mut MtpScratch,
    geometry: AttentionGeometry,
    eps: f32,
    rope_theta: f32,
    base_hidden: &Tensor,
    token: &Tensor,
    position: &Tensor,
    cache_row: usize,
    draft_step: usize,
    produce_logits: bool,
) -> Result<()> {
    if cache_row >= state.capacity {
        return Err(invalid("Qwen3.5 MTP cache row exceeds capacity"));
    }
    let mtp = &weights.mtp;
    ops::embedding_gather(ctx, &weights.embedding, token, &scratch.embedding)?;
    ops::mlp::rms_norm(
        ctx,
        &scratch.embedding,
        &mtp.pre_fc_norm_embedding,
        &scratch.norm_embedding,
        eps,
    )?;
    ops::mlp::rms_norm(
        ctx,
        base_hidden,
        &mtp.pre_fc_norm_hidden,
        &scratch.norm_hidden,
        eps,
    )?;
    transfers::copy_cuda_columns(ctx, &scratch.norm_embedding, &scratch.joined, 0)?;
    transfers::copy_cuda_columns(ctx, &scratch.norm_hidden, &scratch.joined, HIDDEN)?;
    let mut hidden = scratch.hidden.clone();
    ops::gemm(
        ctx,
        ops::GemmArgs::new(&scratch.joined, &mtp.fc.weight, &mut hidden),
    )?;

    ops::mlp::rms_norm(
        ctx,
        &scratch.hidden,
        &mtp.layer.input_layernorm,
        &scratch.normalized,
        eps,
    )?;
    let attention = &mtp.layer.attention;
    let mut q_fused = scratch.q_fused.clone();
    ops::gemm(
        ctx,
        ops::GemmArgs::new(&scratch.normalized, &attention.q_proj.weight, &mut q_fused),
    )?;
    let q_fused_heads = reshape(
        &scratch.q_fused,
        vec![1, geometry.query_heads, 2 * geometry.head_dim],
        DType::BF16,
    )?;
    ops::split_query_and_gate(ctx, &q_fused_heads, &scratch.query, &scratch.query_gate)?;

    let key_slot = cache_row_view(&state.key, cache_row, geometry.kv_heads, geometry.head_dim)?;
    let value_slot = cache_row_view(
        &state.value,
        cache_row,
        geometry.kv_heads,
        geometry.head_dim,
    )?;
    let key_flat = reshape(
        &key_slot,
        vec![1, geometry.kv_heads * geometry.head_dim],
        DType::BF16,
    )?;
    let value_flat = reshape(
        &value_slot,
        vec![1, geometry.kv_heads * geometry.head_dim],
        DType::BF16,
    )?;
    let mut key_flat_out = key_flat.clone();
    let mut value_flat_out = value_flat.clone();
    ops::gemm(
        ctx,
        ops::GemmArgs::new(
            &scratch.normalized,
            &attention.k_proj.weight,
            &mut key_flat_out,
        ),
    )?;
    ops::gemm(
        ctx,
        ops::GemmArgs::new(
            &scratch.normalized,
            &attention.v_proj.weight,
            &mut value_flat_out,
        ),
    )?;
    let query_heads = reshape(
        &scratch.query,
        vec![geometry.query_heads, geometry.head_dim],
        DType::BF16,
    )?;
    let key_heads = reshape(
        &key_slot,
        vec![geometry.kv_heads, geometry.head_dim],
        DType::BF16,
    )?;
    ops::head_rms_norm(ctx, &query_heads, &attention.q_norm, eps)?;
    ops::head_rms_norm(ctx, &key_heads, &attention.k_norm, eps)?;
    ops::partial_rope(
        ctx,
        &scratch.query,
        position,
        geometry.rotary_dim,
        rope_theta,
    )?;
    ops::partial_rope(ctx, &key_slot, position, geometry.rotary_dim, rope_theta)?;

    let valid = cache_row + 1;
    // Capacity-shaped views plus a device-resident length, so one tuned Recipe
    // serves every draft step. Rows past `valid` are never read: each split
    // clamps to the device length. Needs `valid >= 1`, which holds since
    // `valid = cache_row + 1`.
    if scratch.decode_meta.is_none() {
        scratch.decode_meta = Some(ops::KvCacheDecodeMeta::new(
            ctx.device_id(),
            state.capacity,
        )?);
    }
    let keys = cache_prefix_view(&state.key, state.capacity, geometry.kv_heads, geometry.head_dim)?;
    let values =
        cache_prefix_view(&state.value, state.capacity, geometry.kv_heads, geometry.head_dim)?;
    let query = reshape(
        &scratch.query,
        vec![1, 1, geometry.query_heads, geometry.head_dim],
        DType::BF16,
    )?;
    let mut attention_out = reshape(
        &scratch.attention_out,
        vec![1, 1, geometry.query_heads, geometry.head_dim],
        DType::BF16,
    )?;
    let meta = scratch
        .decode_meta
        .clone()
        .ok_or_else(|| invalid("Qwen3.5 MTP decode metadata was not initialized"))?;
    meta.update(valid, cache_row)?;
    let mut args = ops::KvCacheAttentionArgs::new(&query, &keys, &values, &mut attention_out)
        .with_decode_meta(&meta);
    args.valid_key_tokens = valid;
    args.query_start = cache_row;
    // The draft attention previously fell back silently to the naive custom
    // kernel on any failure; that path timed 47.79 ms against 0.209 ms for the
    // split-KV winner on the decode shape.
    args.policy.allow_fallback = false;
    ops::kv_cache_attention(ctx, args)?;
    let gate = reshape(
        &scratch.query_gate,
        vec![1, geometry.query_heads * geometry.head_dim],
        DType::BF16,
    )?;
    ops::apply_output_gate(ctx, &scratch.attention_out, &gate)?;
    let mut projected = scratch.projected.clone();
    ops::gemm(
        ctx,
        ops::GemmArgs::new(
            &scratch.attention_out,
            &attention.o_proj.weight,
            &mut projected,
        ),
    )?;
    ops::add_into(ctx, &scratch.projected, &scratch.hidden)?;

    ops::mlp::rms_norm(
        ctx,
        &scratch.hidden,
        &mtp.layer.post_attention_layernorm,
        &scratch.normalized,
        eps,
    )?;
    let moe = &mtp.layer.moe;
    let mut moe_out = scratch.moe_out.clone();
    let moe_committed_position = cache_row
        .checked_sub(draft_step)
        .and_then(|row| row.checked_add(1))
        .ok_or_else(|| invalid("Qwen3.5 MTP MoE position underflow"))?;
    match &moe.routed_experts {
        DeviceDenseRoutedExperts::Reference { gate, up, down } => {
            ops::qwen35_mtp_moe(
                ctx,
                ops::Qwen35MtpMoeArgs {
                    state: ops::Qwen35MtpMoeState {
                        base_committed_position: moe_committed_position,
                        mtp_committed_position: moe_committed_position,
                        draft_step,
                    },
                    input: &scratch.normalized,
                    weights: ops::Qwen35MtpMoeWeights {
                        router: &moe.router.weight,
                        expert_gate: &gate.weight,
                        expert_up: &up.weight,
                        expert_down: &down.weight,
                        shared_gate: &moe.shared_gate.weight,
                        shared_up: &moe.shared_up.weight,
                        shared_down: &moe.shared_down.weight,
                        shared_expert_gate: &moe.shared_expert_gate.weight,
                    },
                    scratch: ops::Qwen35MtpMoeScratch {
                        router_logits: &mut scratch.router_logits,
                        expert_ids: &mut scratch.expert_ids,
                        expert_weights: &mut scratch.expert_weights,
                        route_intermediate: &mut scratch.route_intermediate,
                        route_output: &mut scratch.route_output,
                        shared_intermediate: &mut scratch.shared_intermediate,
                        shared_output: &mut scratch.shared_output,
                        shared_gate_logit: &mut scratch.shared_gate_logit,
                    },
                    output: &mut moe_out,
                },
            )?;
        }
        DeviceDenseRoutedExperts::Edge(edge) => {
            run_edge_bf16_mtp_moe(ctx, moe, edge, scratch, &mut moe_out)?;
        }
    }
    ops::add_into(ctx, &scratch.moe_out, &scratch.hidden)?;
    ops::mlp::rms_norm(ctx, &scratch.hidden, &mtp.norm, &scratch.final_hidden, eps)?;
    if produce_logits {
        let mut logits = scratch.logits.clone();
        super::forward::project_lm_head(ctx, weights, &scratch.final_hidden, &mut logits)?;
        ops::argmax(ctx, &scratch.logits, &scratch.next_token)?;
    }
    Ok(())
}

fn run_edge_bf16_mtp_moe(
    ctx: &CudaContext,
    moe: &DeviceDenseMoe,
    experts: &DeviceEdgeBf16Experts,
    scratch: &mut MtpScratch,
    output: &mut Tensor,
) -> Result<()> {
    let workspace = scratch
        .edge_workspace
        .as_ref()
        .ok_or_else(|| invalid("Qwen3.5 Edge BF16 MoE workspace is unavailable"))?;
    let mut router_logits = scratch.router_logits.clone();
    ops::gemm(
        ctx,
        ops::GemmArgs::new(&scratch.normalized, &moe.router.weight, &mut router_logits),
    )?;
    let config = ops::RoutedMoeConfig::new(EXPERTS, TOP_K);
    let mut expert_ids = reshape(&scratch.expert_ids, vec![1, TOP_K], DType::I32)?;
    let mut expert_weights = reshape(&scratch.expert_weights, vec![1, TOP_K], DType::F32)?;
    ops::routed_moe_route_topk(
        ctx,
        ops::RoutedMoeRouteTopkArgs {
            config,
            logits: &scratch.router_logits,
            expert_ids: &mut expert_ids,
            expert_weights: &mut expert_weights,
        },
    )?;
    let mut routed_output = rows_view(&scratch.route_output, 0, 1, HIDDEN)?;
    ops::edge_bf16_decode_moe(
        ctx,
        ops::EdgeBf16DecodeMoeArgs {
            config,
            intermediate: INTERMEDIATE,
            input: &scratch.normalized,
            expert_ids: &expert_ids,
            expert_weights: &expert_weights,
            weights: ops::EdgeBf16DecodeMoeWeights {
                fc1_weight: &experts.fc1_weight,
                fc2_weight: &experts.fc2_weight,
            },
            workspace,
            output: &mut routed_output,
        },
    )?;

    let mut shared_gate = rows_view(&scratch.route_intermediate, 0, 1, INTERMEDIATE)?;
    let mut shared_up = rows_view(&scratch.route_intermediate, 1, 1, INTERMEDIATE)?;
    ops::gemm(
        ctx,
        ops::GemmArgs::new(
            &scratch.normalized,
            &moe.shared_gate.weight,
            &mut shared_gate,
        ),
    )?;
    ops::gemm(
        ctx,
        ops::GemmArgs::new(&scratch.normalized, &moe.shared_up.weight, &mut shared_up),
    )?;
    let shared_fused = reshape(
        &scratch.route_intermediate,
        vec![TOP_K * INTERMEDIATE],
        DType::BF16,
    )?;
    let shared_fused = vector_view(&shared_fused, 0, 2 * INTERMEDIATE)?;
    let shared_fused = reshape(&shared_fused, vec![1, 2 * INTERMEDIATE], DType::BF16)?;
    ops::swiglu_round_product(ctx, &shared_fused, &mut scratch.shared_intermediate)?;
    let mut shared_output = scratch.shared_output.clone();
    ops::gemm(
        ctx,
        ops::GemmArgs::new(
            &scratch.shared_intermediate,
            &moe.shared_down.weight,
            &mut shared_output,
        ),
    )?;
    let mut shared_gate_logit = reshape(&scratch.shared_gate_logit, vec![1, 1], DType::BF16)?;
    ops::gemm(
        ctx,
        ops::GemmArgs::new(
            &scratch.normalized,
            &moe.shared_expert_gate.weight,
            &mut shared_gate_logit,
        ),
    )?;
    ops::routed_moe_add_shared(
        ctx,
        ops::RoutedMoeAddSharedArgs {
            routed_output: &routed_output,
            shared_output: &scratch.shared_output,
            shared_gate_logits: &scratch.shared_gate_logit,
            output,
        },
    )
}

fn validate_checkpoint_geometry(geometry: AttentionGeometry) -> Result<()> {
    if geometry.query_heads != 16
        || geometry.kv_heads != 2
        || geometry.head_dim != 256
        || geometry.rotary_dim != 64
        || geometry.query_projection_width != 8_192
        || geometry.kv_projection_width != 512
        || !geometry.output_gate
    {
        return Err(invalid(
            "Qwen3.5 MTP attention geometry does not match checkpoint",
        ));
    }
    Ok(())
}

fn validate_mtp_weights(
    ctx: &CudaContext,
    mtp: &DeviceMtpWeights,
    geometry: AttentionGeometry,
) -> Result<()> {
    validate_tensor(
        &mtp.pre_fc_norm_embedding,
        ctx,
        DType::BF16,
        &[HIDDEN],
        "MTP embedding norm",
    )?;
    validate_tensor(
        &mtp.pre_fc_norm_hidden,
        ctx,
        DType::BF16,
        &[HIDDEN],
        "MTP hidden norm",
    )?;
    validate_dense(
        &mtp.fc.weight,
        mtp.fc.input_features,
        mtp.fc.output_features,
        ctx,
        2 * HIDDEN,
        HIDDEN,
        "MTP fc",
    )?;
    validate_tensor(&mtp.norm, ctx, DType::BF16, &[HIDDEN], "MTP final norm")?;
    validate_tensor(
        &mtp.layer.input_layernorm,
        ctx,
        DType::BF16,
        &[HIDDEN],
        "MTP input norm",
    )?;
    validate_tensor(
        &mtp.layer.post_attention_layernorm,
        ctx,
        DType::BF16,
        &[HIDDEN],
        "MTP post-attention norm",
    )?;
    let attention = &mtp.layer.attention;
    if attention.kv_cache_scales.is_some() {
        return Err(invalid(
            "Qwen3.5 MTP BF16 KV cache must not carry FP8 calibration",
        ));
    }
    validate_dense(
        &attention.q_proj.weight,
        attention.q_proj.input_features,
        attention.q_proj.output_features,
        ctx,
        HIDDEN,
        geometry.query_projection_width,
        "MTP q_proj",
    )?;
    validate_dense(
        &attention.k_proj.weight,
        attention.k_proj.input_features,
        attention.k_proj.output_features,
        ctx,
        HIDDEN,
        geometry.kv_projection_width,
        "MTP k_proj",
    )?;
    validate_dense(
        &attention.v_proj.weight,
        attention.v_proj.input_features,
        attention.v_proj.output_features,
        ctx,
        HIDDEN,
        geometry.kv_projection_width,
        "MTP v_proj",
    )?;
    validate_dense(
        &attention.o_proj.weight,
        attention.o_proj.input_features,
        attention.o_proj.output_features,
        ctx,
        geometry.query_heads * geometry.head_dim,
        HIDDEN,
        "MTP o_proj",
    )?;
    validate_tensor(
        &attention.q_norm,
        ctx,
        DType::BF16,
        &[geometry.head_dim],
        "MTP q_norm",
    )?;
    validate_tensor(
        &attention.k_norm,
        ctx,
        DType::BF16,
        &[geometry.head_dim],
        "MTP k_norm",
    )?;
    let moe = &mtp.layer.moe;
    validate_dense(
        &moe.router.weight,
        moe.router.input_features,
        moe.router.output_features,
        ctx,
        HIDDEN,
        EXPERTS,
        "MTP router",
    )?;
    match &moe.routed_experts {
        DeviceDenseRoutedExperts::Reference { gate, up, down } => {
            validate_tensor(
                &gate.weight,
                ctx,
                DType::BF16,
                &[EXPERTS, HIDDEN, INTERMEDIATE],
                "MTP expert gate",
            )?;
            validate_tensor(
                &up.weight,
                ctx,
                DType::BF16,
                &[EXPERTS, HIDDEN, INTERMEDIATE],
                "MTP expert up",
            )?;
            validate_tensor(
                &down.weight,
                ctx,
                DType::BF16,
                &[EXPERTS, INTERMEDIATE, HIDDEN],
                "MTP expert down",
            )?;
        }
        DeviceDenseRoutedExperts::Edge(edge) => {
            if edge.experts != EXPERTS || edge.hidden != HIDDEN || edge.intermediate != INTERMEDIATE
            {
                return Err(invalid("MTP Edge BF16 expert metadata mismatch"));
            }
            validate_tensor(
                &edge.fc1_weight,
                ctx,
                DType::BF16,
                &[EXPERTS, 2 * INTERMEDIATE, HIDDEN],
                "MTP Edge BF16 FC1",
            )?;
            validate_tensor(
                &edge.fc2_weight,
                ctx,
                DType::BF16,
                &[EXPERTS, HIDDEN, INTERMEDIATE],
                "MTP Edge BF16 FC2",
            )?;
        }
    }
    validate_dense(
        &moe.shared_gate.weight,
        moe.shared_gate.input_features,
        moe.shared_gate.output_features,
        ctx,
        HIDDEN,
        INTERMEDIATE,
        "MTP shared gate",
    )?;
    validate_dense(
        &moe.shared_up.weight,
        moe.shared_up.input_features,
        moe.shared_up.output_features,
        ctx,
        HIDDEN,
        INTERMEDIATE,
        "MTP shared up",
    )?;
    validate_dense(
        &moe.shared_down.weight,
        moe.shared_down.input_features,
        moe.shared_down.output_features,
        ctx,
        INTERMEDIATE,
        HIDDEN,
        "MTP shared down",
    )?;
    validate_dense(
        &moe.shared_expert_gate.weight,
        moe.shared_expert_gate.input_features,
        moe.shared_expert_gate.output_features,
        ctx,
        HIDDEN,
        1,
        "MTP shared expert gate",
    )
}

#[allow(clippy::too_many_arguments)]
fn validate_dense(
    tensor: &Tensor,
    input_features: usize,
    output_features: usize,
    ctx: &CudaContext,
    expected_input: usize,
    expected_output: usize,
    label: &str,
) -> Result<()> {
    if input_features != expected_input || output_features != expected_output {
        return Err(invalid(format!("Qwen3.5 {label} metadata mismatch")));
    }
    validate_tensor(
        tensor,
        ctx,
        DType::BF16,
        &[expected_input, expected_output],
        label,
    )
}

fn validate_state_storage(
    ctx: &CudaContext,
    state: &Qwen35MtpState,
    geometry: AttentionGeometry,
) -> Result<()> {
    let shape = [state.capacity, geometry.kv_heads, geometry.head_dim];
    validate_tensor(
        &state.key,
        ctx,
        QWEN35_MTP_KV_DTYPE,
        &shape,
        "MTP key cache",
    )?;
    validate_tensor(
        &state.value,
        ctx,
        QWEN35_MTP_KV_DTYPE,
        &shape,
        "MTP value cache",
    )?;
    if let Some(position) = state.position {
        position.validate(state.capacity)?;
    }
    Ok(())
}

fn validate_tensor(
    tensor: &Tensor,
    ctx: &CudaContext,
    dtype: DType,
    shape: &[usize],
    label: &str,
) -> Result<()> {
    if tensor.device() != Device::Cuda(ctx.device_id())
        || tensor.dtype() != dtype
        || tensor.shape().dims() != shape
    {
        return Err(invalid(format!(
            "Qwen3.5 {label} expected CUDA:{} {dtype} {shape:?}, got {:?} {} {:?}",
            ctx.device_id(),
            tensor.device(),
            tensor.dtype(),
            tensor.shape().dims()
        )));
    }
    Ok(())
}

fn write_i32(destination: &Tensor, value: u32) -> Result<()> {
    let value = i32::try_from(value)
        .map_err(|_| invalid(format!("Qwen3.5 MTP value {value} exceeds i32")))?;
    let host = Tensor::from_i32(vec![1], &[value])?;
    transfers::copy_cpu_to_cuda(&host, destination)
}

fn zeros(ctx: &CudaContext, shape: Vec<usize>, dtype: DType) -> Result<Tensor> {
    let bytes = shape
        .iter()
        .try_fold(dtype.size_in_bytes(), |bytes, dimension| {
            bytes.checked_mul(*dimension)
        })
        .ok_or_else(|| invalid("Qwen3.5 MTP CUDA allocation overflow"))?;
    CudaBuffer::alloc_zeros(bytes, ctx.device_id())
        .map_err(Error::Cuda)?
        .as_tensor(Shape::new(shape), dtype)
        .map_err(Error::Cuda)
}

fn reshape(tensor: &Tensor, shape: Vec<usize>, dtype: DType) -> Result<Tensor> {
    CudaBuffer::from_tensor(tensor)
        .map_err(Error::Cuda)?
        .as_tensor(Shape::new(shape), dtype)
        .map_err(Error::Cuda)
}

fn rows_view(tensor: &Tensor, row: usize, rows: usize, width: usize) -> Result<Tensor> {
    if rows == 0 {
        return Err(invalid("Qwen3.5 MTP row view must be non-empty"));
    }
    let row_bytes = width
        .checked_mul(tensor.dtype().size_in_bytes())
        .ok_or_else(|| invalid("Qwen3.5 MTP row size overflow"))?;
    let bytes = rows
        .checked_mul(row_bytes)
        .ok_or_else(|| invalid("Qwen3.5 MTP row range overflow"))?;
    CudaBuffer::from_tensor(tensor)
        .map_err(Error::Cuda)?
        .view(
            row.checked_mul(row_bytes)
                .ok_or_else(|| invalid("Qwen3.5 MTP row offset overflow"))?,
            bytes,
        )
        .map_err(Error::Cuda)?
        .as_tensor(Shape::new(vec![rows, width]), tensor.dtype())
        .map_err(Error::Cuda)
}

fn vector_view(tensor: &Tensor, start: usize, len: usize) -> Result<Tensor> {
    if len == 0 {
        return Err(invalid("Qwen3.5 MTP vector view must be non-empty"));
    }
    let element_bytes = tensor.dtype().size_in_bytes();
    let bytes = len
        .checked_mul(element_bytes)
        .ok_or_else(|| invalid("Qwen3.5 MTP vector range overflow"))?;
    CudaBuffer::from_tensor(tensor)
        .map_err(Error::Cuda)?
        .view(
            start
                .checked_mul(element_bytes)
                .ok_or_else(|| invalid("Qwen3.5 MTP vector offset overflow"))?,
            bytes,
        )
        .map_err(Error::Cuda)?
        .as_tensor(Shape::new(vec![len]), tensor.dtype())
        .map_err(Error::Cuda)
}

fn element_view(tensor: &Tensor, index: usize) -> Result<Tensor> {
    let bytes = tensor.dtype().size_in_bytes();
    CudaBuffer::from_tensor(tensor)
        .map_err(Error::Cuda)?
        .view(
            index
                .checked_mul(bytes)
                .ok_or_else(|| invalid("Qwen3.5 MTP element offset overflow"))?,
            bytes,
        )
        .map_err(Error::Cuda)?
        .as_tensor(Shape::new(vec![1]), tensor.dtype())
        .map_err(Error::Cuda)
}

fn cache_row_view(cache: &Tensor, row: usize, heads: usize, head_dim: usize) -> Result<Tensor> {
    cache_rows_view(cache, row, 1, heads, head_dim)
}

fn cache_rows_view(
    cache: &Tensor,
    row: usize,
    rows: usize,
    heads: usize,
    head_dim: usize,
) -> Result<Tensor> {
    if rows == 0 {
        return Err(invalid("Qwen3.5 MTP cache row view must be non-empty"));
    }
    let row_elements = heads
        .checked_mul(head_dim)
        .ok_or_else(|| invalid("Qwen3.5 MTP cache row overflow"))?;
    let row_bytes = row_elements
        .checked_mul(DType::BF16.size_in_bytes())
        .ok_or_else(|| invalid("Qwen3.5 MTP cache bytes overflow"))?;
    let bytes = rows
        .checked_mul(row_bytes)
        .ok_or_else(|| invalid("Qwen3.5 MTP cache range overflow"))?;
    CudaBuffer::from_tensor(cache)
        .map_err(Error::Cuda)?
        .view(
            row.checked_mul(row_bytes)
                .ok_or_else(|| invalid("Qwen3.5 MTP cache offset overflow"))?,
            bytes,
        )
        .map_err(Error::Cuda)?
        .as_tensor(Shape::new(vec![rows, heads, head_dim]), DType::BF16)
        .map_err(Error::Cuda)
}

fn cache_prefix_view(cache: &Tensor, rows: usize, heads: usize, head_dim: usize) -> Result<Tensor> {
    let bytes = rows
        .checked_mul(heads)
        .and_then(|n| n.checked_mul(head_dim))
        .and_then(|n| n.checked_mul(DType::BF16.size_in_bytes()))
        .ok_or_else(|| invalid("Qwen3.5 MTP cache prefix overflow"))?;
    CudaBuffer::from_tensor(cache)
        .map_err(Error::Cuda)?
        .view(0, bytes)
        .map_err(Error::Cuda)?
        .as_tensor(Shape::new(vec![1, rows, heads, head_dim]), DType::BF16)
        .map_err(Error::Cuda)
}

fn invalid(message: impl Into<String>) -> Error {
    Error::Other(message.into())
}

fn schedule_error(error: impl std::fmt::Display) -> Error {
    invalid(format!("Qwen3.5 MTP schedule: {error}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prompt_shift_keeps_logical_and_physical_positions_distinct() {
        assert_eq!(QWEN35_MTP_KV_DTYPE, DType::BF16);
        let state = MtpPositionContract::after_prompt(4_096).unwrap();
        assert_eq!(state.logical_position, 4_096);
        assert_eq!(state.cache_rows, 4_095);
        assert_eq!(state.physical_row_for_draft(0).unwrap(), 4_095);
        assert_eq!(state.rope_position_for_draft(0).unwrap(), 4_096);
        assert_eq!(state.physical_row_for_draft(2).unwrap(), 4_097);
        assert_eq!(state.rope_position_for_draft(2).unwrap(), 4_098);
    }

    #[test]
    fn prompt_teacher_forcing_pairs_shifted_tokens_with_previous_hidden_rows() {
        let prompt = [11u32, 22, 33, 44];
        let pairs = prompt[1..]
            .iter()
            .copied()
            .zip(0..prompt.len() - 1)
            .enumerate()
            .map(|(cache_row, (token, hidden_row))| (token, hidden_row, cache_row, cache_row + 1))
            .collect::<Vec<_>>();
        assert_eq!(pairs, vec![(22, 0, 0, 1), (33, 1, 1, 2), (44, 2, 2, 3)]);
    }

    #[test]
    fn partial_and_full_commits_preserve_position_invariant() {
        let start = MtpPositionContract::after_prompt(17).unwrap();
        for committed in 1..=4 {
            let next = start.after_commit(committed).unwrap();
            assert_eq!(next.logical_position, 17 + committed);
            assert_eq!(next.cache_rows, 16 + committed);
            assert_eq!(next.logical_position, next.cache_rows + 1);
        }
    }

    #[test]
    fn empty_prompt_and_fourth_draft_are_rejected() {
        assert!(MtpPositionContract::after_prompt(0).is_err());
        let state = MtpPositionContract::after_prompt(1).unwrap();
        assert!(state.physical_row_for_draft(3).is_err());
        assert!(state.rope_position_for_draft(3).is_err());
    }

    #[test]
    fn speculative_policy_admits_greedy_argmax_with_ignored_sampling_filters() {
        let greedy = GenerationOptions::greedy(32, Some(7));
        assert_eq!(
            MtpGreedyPolicy::from_generation_options(&greedy).unwrap(),
            MtpGreedyPolicy {
                max_new_tokens: 32,
                eos_token_ids: vec![7],
            }
        );

        let mut sampled = greedy.clone();
        sampled.sampling_mode = Some(SamplingMode::Random);
        sampled.temperature = Some(0.8);
        assert!(MtpGreedyPolicy::from_generation_options(&sampled).is_err());

        let mut non_neutral_temperature = greedy.clone();
        non_neutral_temperature.temperature = Some(0.7);
        assert!(MtpGreedyPolicy::from_generation_options(&non_neutral_temperature).is_ok());

        let mut filtered = greedy.clone();
        filtered.top_p = Some(0.9);
        filtered.top_k = Some(20);
        assert!(MtpGreedyPolicy::from_generation_options(&filtered).is_ok());

        let mut penalized = greedy.clone();
        penalized.repetition_penalty = Some(1.1);
        assert!(MtpGreedyPolicy::from_generation_options(&penalized).is_err());

        let mut logprobs = greedy;
        logprobs.return_logprob = Some(true);
        assert!(MtpGreedyPolicy::from_generation_options(&logprobs).is_err());
    }
}
