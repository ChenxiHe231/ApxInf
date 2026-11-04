//! Native Qwen3.5-35B-A3B MTP proposer.
//!
//! This is deliberately not the Qwen3.8 dense-MLP proposer. The checkpoint's
//! one MTP layer is BF16 and contains the same 256-expert/top-8 routed MoE as
//! the target architecture. Its persistent K/V cache follows TensorRT Edge's
//! static unit-scale E4M3 contract and the same 128-token paging as the target.

use apxinf_core::{DType, Device, Error, Result, Shape, Tensor};
use apxinf_cuda_new::{ops, transfers, CudaBuffer, CudaContext, ExecutionSession};

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
use super::session::{MtpBackend, MtpDraft, Qwen35Session, Qwen35SessionSnapshot};
use crate::{GenerationOptions, SamplingMode};

const HIDDEN: usize = ops::QWEN35_MTP_HIDDEN;
const VOCAB: usize = ops::QWEN35_MTP_VOCAB_SIZE;
const EXPERTS: usize = ops::QWEN35_MTP_EXPERTS;
const TOP_K: usize = ops::QWEN35_MTP_TOP_K;
const INTERMEDIATE: usize = ops::QWEN35_MTP_INTERMEDIATE;
const MTP_PREFILL_CHUNK: usize = 2_048;
const MTP_DRAFT_WORKSPACE: usize = 4 * 1024 * 1024;
pub const QWEN35_MTP_KV_DTYPE: DType = DType::F8E4M3;
const MTP_KV_SCALE: f32 = 1.0;

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
                "Qwen3.5 MTP logical position must equal FP8 KV rows plus one",
            ));
        }
        if self.cache_rows > capacity {
            return Err(invalid("Qwen3.5 MTP FP8 KV cache exceeds capacity"));
        }
        Ok(())
    }
}

/// One transactional static-FP8 MTP KV bank.
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
    draft_steps: usize,
}

/// Complete native proposer backend. All large tensors remain on CUDA. Draft
/// argmax runs on device and only the compact I32 proposal IDs cross to
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

    fn ensure_scratch(
        &mut self,
        ctx: &CudaContext,
        cache_capacity: usize,
    ) -> Result<&mut MtpScratch> {
        if self.scratch.is_none() {
            self.scratch = Some(MtpScratch::new(ctx, self.geometry, cache_capacity)?);
        } else if self
            .scratch
            .as_ref()
            .is_some_and(|scratch| scratch.cache_capacity != cache_capacity)
        {
            return Err(invalid(
                "Qwen3.5 MTP scratch/cache capacity changed after initialization",
            ));
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
        let capacity = capacity
            .checked_add(ops::QWEN35_KV_CACHE_BLOCK_TOKENS - 1)
            .map(|value| value / ops::QWEN35_KV_CACHE_BLOCK_TOKENS)
            .and_then(|blocks| blocks.checked_mul(ops::QWEN35_KV_CACHE_BLOCK_TOKENS))
            .ok_or_else(|| invalid("Qwen3.5 MTP cache capacity overflow"))?;
        let blocks = capacity / ops::QWEN35_KV_CACHE_BLOCK_TOKENS;
        Ok(Qwen35MtpState {
            key: zeros(
                ctx,
                vec![
                    blocks,
                    ops::QWEN35_KV_CACHE_BLOCK_TOKENS,
                    self.geometry.kv_heads,
                    self.geometry.head_dim,
                ],
                QWEN35_MTP_KV_DTYPE,
            )?,
            value: zeros(
                ctx,
                vec![
                    blocks,
                    ops::QWEN35_KV_CACHE_BLOCK_TOKENS,
                    self.geometry.kv_heads,
                    self.geometry.head_dim,
                ],
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
            let shifted_tokens = transfers::to_cuda_on_stream(
                ctx,
                Tensor::from_i32(vec![position.cache_rows], &shifted_tokens)?,
            )?;
            let shifted_positions = transfers::to_cuda_on_stream(
                ctx,
                Tensor::from_i32(vec![position.cache_rows], &shifted_positions)?,
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
        let tokens = transfers::to_cuda_on_stream(
            ctx,
            Tensor::from_i32(vec![chunk_tokens.len()], &token_values)?,
        )?;
        let positions = transfers::to_cuda_on_stream(
            ctx,
            Tensor::from_i32(vec![chunk_tokens.len()], &position_values)?,
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
        draft_steps: usize,
    ) -> Result<MtpDraft> {
        if !(1..=QWEN35_MTP_DRAFT_STEPS).contains(&draft_steps) {
            return Err(invalid(format!(
                "Qwen3.5 MTP draft depth must be in 1..={QWEN35_MTP_DRAFT_STEPS}, got {draft_steps}"
            )));
        }
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
            .checked_add(draft_steps)
            .ok_or_else(|| invalid("Qwen3.5 MTP draft capacity overflow"))?;
        if required_rows > committed.capacity {
            return Err(invalid("Qwen3.5 MTP draft exceeds FP8 KV capacity"));
        }
        self.copy_state(ctx, committed, shadow)?;

        let geometry = self.geometry;
        let eps = self.rms_norm_eps;
        let theta = self.rope_theta;
        let scratch = self.ensure_scratch(ctx, committed.capacity)?;
        write_i32(ctx, &scratch.token, pending)?;
        let pending_slot = element_view(&scratch.verification_inputs, 0)?;
        transfers::copy_cuda_to_cuda(ctx, &scratch.token, &pending_slot)?;
        let draft_positions = (0..draft_steps)
            .map(|depth| {
                i32::try_from(start.rope_position_for_draft(depth)?)
                    .map_err(|_| invalid("Qwen3.5 MTP RoPE position exceeds i32"))
            })
            .collect::<Result<Vec<_>>>()?;
        let draft_positions = draft_positions
            .into_iter()
            .flat_map(|position| [position; 3])
            .collect::<Vec<_>>();
        let draft_positions = transfers::to_cuda_on_stream(
            ctx,
            Tensor::from_i32(vec![draft_steps, 3], &draft_positions)?,
        )?;
        // GEMM normally synchronizes after every eager execution so temporary
        // native resources cannot be destroyed while still in flight. Draft
        // uses fixed scratch addresses for all three sequential steps, so one
        // persistent execution session can retain those resources and leave
        // the chain asynchronous until the compact proposal readback below.
        let session_index = draft_steps - 1;
        let draft_session = scratch.draft_sessions[session_index].clone();
        if scratch.draft_sessions_prepared[session_index] {
            draft_session.run(|| {
                run_draft_chain(
                    ctx,
                    weights,
                    shadow,
                    scratch,
                    geometry,
                    eps,
                    theta,
                    target_hidden,
                    &draft_positions,
                    start,
                    draft_steps,
                )
            })?;
        } else {
            draft_session.prepare(|| {
                run_draft_chain(
                    ctx,
                    weights,
                    shadow,
                    scratch,
                    geometry,
                    eps,
                    theta,
                    target_hidden,
                    &draft_positions,
                    start,
                    draft_steps,
                )
            })?;
            scratch.draft_sessions_prepared[session_index] = true;
        }
        let draft = MtpDraft::new(
            vector_view(&scratch.proposals, 0, draft_steps)?,
            vector_view(&scratch.verification_inputs, 0, draft_steps + 1)?,
        )?;
        self.pending_draft = Some(PendingDraft {
            shadow_generation: shadow.generation,
            start,
            draft_steps,
        });
        Ok(draft)
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
            pending.draft_steps + 1
        } else {
            pending.draft_steps
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
        if committed_rows == 0 || committed_rows > pending.draft_steps + 1 {
            return Err(invalid(format!(
                "Qwen3.5 MTP commit row count is outside 1..={} for draft depth {}",
                pending.draft_steps + 1,
                pending.draft_steps
            )));
        }
        if contract.materialize_final_proposal {
            if committed_rows != pending.draft_steps + 1 {
                return Err(invalid(
                    "Qwen3.5 MTP final proposal is materialized only on full accept plus bonus",
                ));
            }
            let materialize_row = pending
                .start
                .cache_rows
                .checked_add(pending.draft_steps)
                .ok_or_else(|| invalid("Qwen3.5 MTP final cache row overflow"))?;
            if materialize_row >= shadow.capacity {
                return Err(invalid("Qwen3.5 MTP final materialization exceeds cache"));
            }
            let geometry = self.geometry;
            let eps = self.rms_norm_eps;
            let theta = self.rope_theta;
            let scratch = self.ensure_scratch(ctx, shadow.capacity)?;
            transfers::copy_cuda_to_cuda(ctx, &scratch.next_token, &scratch.token)?;
            write_text_position(
                ctx,
                &scratch.position,
                u32::try_from(
                    pending
                        .start
                        .logical_position
                        .checked_add(pending.draft_steps)
                        .ok_or_else(|| invalid("Qwen3.5 MTP final RoPE position overflow"))?,
                )
                .map_err(|_| invalid("Qwen3.5 MTP final RoPE position exceeds i32"))?,
            )?;
            let base_hidden = scratch.final_hidden.clone();
            let token_input = scratch.token.clone();
            let rope_position = scratch.position.clone();
            // The last proposal is consumed as an input but no fourth draft
            // logit is needed. This mirrors the tuned Qwen3.8 protocol.
            materialize_final_proposal(
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
                pending.draft_steps,
            )?;
        } else if committed_rows > pending.draft_steps {
            return Err(invalid(
                "Qwen3.5 MTP full verifier commit needs materialization",
            ));
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
        if complete.speculative_decoding == Some(false) {
            return Err(invalid(
                "Qwen3.5 speculative decoding was disabled for this request; use base scalar generation",
            ));
        }
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
/// Target attention/GDN state and the draft layer's static-FP8 KV state are kept
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
        // Verifier outputs are views of persistent scratch. A context
        // checkpoint must retain an independent hidden row across later
        // rounds, just as its target and draft cache banks do.
        let frozen_hidden = zeros(
            ctx,
            target_hidden.shape().dims().to_vec(),
            target_hidden.dtype(),
        )?;
        transfers::copy_cuda_to_cuda(ctx, &target_hidden, &frozen_hidden)?;
        self.target.capture_snapshot(ctx, &mut snapshot.target)?;
        self.mtp
            .copy_state(ctx, &self.committed_mtp, &mut snapshot.mtp)?;
        ctx.synchronize().map_err(Error::Cuda)?;
        snapshot.target_hidden = Some(frozen_hidden);
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

    /// Clear target and both FP8 MTP cache banks and discard all generation
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

    /// Execute draft(1..=3) -> target verify(2..=4) -> schedule ->
    /// target/MTP prepare -> dual publish. All device work and the schedule's
    /// fallible transition are completed before either visible state changes.
    pub fn run_greedy_round(
        &mut self,
        ctx: &CudaContext,
        decoder: &Qwen35Decoder,
        weights: &Qwen35DeviceWeights,
        draft_steps: usize,
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
                eprintln!(
                    "[mtp-round] {label} {:.3} ms",
                    at.elapsed().as_secs_f64() * 1e3
                );
            }
        };
        if timing {
            ctx.synchronize().map_err(Error::Cuda)?;
        }
        let phase = std::time::Instant::now();
        let draft = {
            let _range = crate::profiling::trace::range("qwen35_mtp_draft");
            self.mtp.draft(
                ctx,
                weights,
                &self.committed_mtp,
                &mut self.shadow_mtp,
                pending,
                start,
                target_hidden,
                draft_steps,
            )?
        };
        if timing {
            ctx.synchronize().map_err(Error::Cuda)?;
        }
        mark("draft", phase);
        // A/B escape hatch for validating the device-resident draft chain.
        // The disabled path deliberately reads proposals here, recreating the
        // old draft -> host synchronization before target verification.
        let host_proposals = (!mtp_device_chain_enabled())
            .then(|| draft.read_proposals())
            .transpose()?;
        let phase = std::time::Instant::now();
        let verification = {
            let _range = crate::profiling::trace::range("qwen35_mtp_verify");
            self.target.verify_target_chunk_device(
                ctx,
                decoder,
                weights,
                draft.verification_inputs(),
            )?
        };
        if timing {
            ctx.synchronize().map_err(Error::Cuda)?;
        }
        mark("verify", phase);
        let phase = std::time::Instant::now();
        let resolution = verification.resolve(ctx, &draft)?;
        let proposals = host_proposals.unwrap_or_else(|| resolution.proposals.clone());
        debug_assert_eq!(proposals, resolution.proposals);
        debug_assert_eq!(proposals.len(), draft.draft_steps());
        let ticket = schedule
            .begin_round(pending, &proposals)
            .map_err(schedule_error)?;
        let plan = schedule
            .finish_round(ticket, &resolution.predictions)
            .map_err(schedule_error)?;
        debug_assert_eq!(plan.accepted_proposals(), resolution.accepted_proposals);
        let contract = plan.commit_contract();

        // Preflight the host schedule transition on a clone. After this point
        // the final schedule assignment cannot fail.
        let mut next_schedule = schedule.clone();
        let outcome = next_schedule.commit(plan).map_err(schedule_error)?;
        let (prepared_target, next_target_hidden, prepared_mtp) = {
            let _range = crate::profiling::trace::range("qwen35_mtp_commit");
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
            (prepared_target, next_target_hidden, prepared_mtp)
        };
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

    /// Permanently leave speculative execution for the current request and
    /// advance the authoritative target one token at a time. The proposer is
    /// intentionally not maintained after this transition: re-entering MTP
    /// would make scalar fallback more expensive than staying speculative.
    pub fn run_scalar_fallback_step(
        &mut self,
        ctx: &CudaContext,
        decoder: &Qwen35Decoder,
        weights: &Qwen35DeviceWeights,
    ) -> Result<TokenId> {
        let pending = self
            .pending
            .ok_or_else(|| invalid("Qwen3.5 scalar fallback has no pending token"))?;
        self.target.set_decode_in_place(true);
        let output = self.target.decode(ctx, decoder, weights, pending)?;
        let next = output
            .greedy_token_ids(ctx)?
            .into_iter()
            .next()
            .ok_or_else(|| invalid("Qwen3.5 scalar fallback returned no token"))?;
        self.pending = Some(next);
        self.target_hidden = None;
        Ok(next)
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
    projected_key: Tensor,
    projected_value: Tensor,
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
            projected_key: zeros(
                ctx,
                vec![max_tokens, ops::QWEN35_ATTENTION_KV_WIDTH],
                DType::BF16,
            )?,
            projected_value: zeros(
                ctx,
                vec![max_tokens, ops::QWEN35_ATTENTION_KV_WIDTH],
                DType::BF16,
            )?,
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
    let mut key_flat = rows_view(
        &scratch.projected_key,
        0,
        rows,
        geometry.kv_heads * geometry.head_dim,
    )?;
    let mut value_flat = rows_view(
        &scratch.projected_value,
        0,
        rows,
        geometry.kv_heads * geometry.head_dim,
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
        &key_flat,
        vec![rows * geometry.kv_heads, geometry.head_dim],
        DType::BF16,
    )?;
    ops::head_rms_norm(ctx, &key_heads, &attention.k_norm, eps)?;
    let key_heads = reshape(
        &key_flat,
        vec![rows, geometry.kv_heads, geometry.head_dim],
        DType::BF16,
    )?;
    ops::partial_rope(ctx, &key_heads, positions, geometry.rotary_dim, rope_theta)?;

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
    let key_slot = reshape(
        &key_slot,
        vec![rows, geometry.kv_heads * geometry.head_dim],
        QWEN35_MTP_KV_DTYPE,
    )?;
    let value_slot = reshape(
        &value_slot,
        vec![rows, geometry.kv_heads * geometry.head_dim],
        QWEN35_MTP_KV_DTYPE,
    )?;
    ops::quantize_fp8_per_tensor(ctx, &key_flat, &key_slot, MTP_KV_SCALE)?;
    ops::quantize_fp8_per_tensor(ctx, &value_flat, &value_slot, MTP_KV_SCALE)
}

// GEMM keys include stream, fixed A/output addresses, weight addresses and
// policy. All A/output views below belong to this private, non-resizable scratch;
// changing device/stream/weights/policy must prepare a fresh materialization tape.
#[derive(PartialEq, Eq)]
struct MtpMaterializeBinding {
    device: usize,
    stream: usize,
    deterministic: bool,
    gemm_count: usize,
    gemm_weights: [usize; 10],
}

impl MtpMaterializeBinding {
    fn new(ctx: &CudaContext, weights: &Qwen35DeviceWeights) -> Result<Self> {
        let mtp = &weights.mtp;
        let attention = &mtp.layer.attention;
        let moe = &mtp.layer.moe;
        let operands = [&mtp.fc.weight, &attention.q_proj.weight,
            &attention.k_proj.weight, &attention.v_proj.weight, &attention.o_proj.weight,
            &moe.router.weight, &moe.shared_gate.weight, &moe.shared_up.weight,
            &moe.shared_down.weight, &moe.shared_expert_gate.weight];
        // Reference MoE is one stateless raw launch. Edge MoE adds five
        // GEMMs with stable views into the same MtpScratch allocations.
        let gemm_count = if matches!(&moe.routed_experts, DeviceDenseRoutedExperts::Edge(_)) { 10 } else { 5 };
        let mut gemm_weights = [0; 10];
        for (slot, tensor) in gemm_weights[..gemm_count].iter_mut().zip(&operands) {
            *slot = tensor.storage().as_gpu().map(|storage| storage.ptr())
                .ok_or_else(|| invalid("MTP materialization weight has no CUDA storage"))?;
        }
        Ok(Self { device: ctx.device_id(), stream: ctx.stream().handle() as usize,
            deterministic: std::env::var_os("APXINF_GEMM_DETERMINISTIC").is_some(), gemm_count, gemm_weights })
    }
}

struct MtpMaterializeSession {
    binding: MtpMaterializeBinding,
    session: ExecutionSession,
    prepared: bool,
}

struct MtpScratch {
    cache_capacity: usize,
    token: Tensor,
    /// Three identical T/H/W positions for text-only MTP rows.
    position: Tensor,
    embedding: Tensor,
    norm_embedding: Tensor,
    norm_hidden: Tensor,
    joined: Tensor,
    hidden: Tensor,
    normalized: Tensor,
    q_fused: Tensor,
    projected_key: Tensor,
    projected_value: Tensor,
    query: Tensor,
    query_gate: Tensor,
    attention_out: Tensor,
    attention_scores: Tensor,
    attention_partial_output: Tensor,
    attention_partial_stats: Tensor,
    projected: Tensor,
    moe_out: Tensor,
    final_hidden: Tensor,
    logits: Tensor,
    next_token: Tensor,
    proposals: Tensor,
    /// `[pending, d0, d1, d2]`, populated on the draft stream so target
    /// verification can start without a proposal D2H/H2D round trip.
    verification_inputs: Tensor,
    router_logits: Tensor,
    expert_ids: Tensor,
    expert_weights: Tensor,
    route_intermediate: Tensor,
    route_output: Tensor,
    shared_intermediate: Tensor,
    shared_output: Tensor,
    shared_gate_logit: Tensor,
    edge_workspace: Option<CudaBuffer>,
    draft_sessions: Vec<ExecutionSession>,
    draft_sessions_prepared: [bool; QWEN35_MTP_DRAFT_STEPS],
    materialize_session: Option<MtpMaterializeSession>,
}

impl MtpScratch {
    fn new(ctx: &CudaContext, geometry: AttentionGeometry, cache_capacity: usize) -> Result<Self> {
        if cache_capacity == 0 || cache_capacity % ops::QWEN35_KV_CACHE_BLOCK_TOKENS != 0 {
            return Err(invalid("Qwen3.5 MTP scratch capacity is not page aligned"));
        }
        Ok(Self {
            cache_capacity,
            token: zeros(ctx, vec![1], DType::I32)?,
            position: zeros(ctx, vec![3, 1], DType::I32)?,
            embedding: zeros(ctx, vec![1, HIDDEN], DType::BF16)?,
            norm_embedding: zeros(ctx, vec![1, HIDDEN], DType::BF16)?,
            norm_hidden: zeros(ctx, vec![1, HIDDEN], DType::BF16)?,
            joined: zeros(ctx, vec![1, 2 * HIDDEN], DType::BF16)?,
            hidden: zeros(ctx, vec![1, HIDDEN], DType::BF16)?,
            normalized: zeros(ctx, vec![1, HIDDEN], DType::BF16)?,
            q_fused: zeros(ctx, vec![1, geometry.query_projection_width], DType::BF16)?,
            projected_key: zeros(ctx, vec![1, geometry.kv_projection_width], DType::BF16)?,
            projected_value: zeros(ctx, vec![1, geometry.kv_projection_width], DType::BF16)?,
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
            attention_scores: zeros(
                ctx,
                vec![1, geometry.query_heads, cache_capacity],
                DType::F32,
            )?,
            attention_partial_output: zeros(
                ctx,
                vec![
                    1,
                    geometry.kv_heads,
                    ops::QWEN35_ATTENTION_MAX_SPLITS,
                    geometry.query_heads / geometry.kv_heads,
                    geometry.head_dim,
                ],
                DType::F32,
            )?,
            attention_partial_stats: zeros(
                ctx,
                vec![
                    1,
                    geometry.kv_heads,
                    ops::QWEN35_ATTENTION_MAX_SPLITS,
                    geometry.query_heads / geometry.kv_heads,
                    2,
                ],
                DType::F32,
            )?,
            projected: zeros(ctx, vec![1, HIDDEN], DType::BF16)?,
            moe_out: zeros(ctx, vec![1, HIDDEN], DType::BF16)?,
            final_hidden: zeros(ctx, vec![1, HIDDEN], DType::BF16)?,
            logits: zeros(ctx, vec![1, VOCAB], DType::BF16)?,
            next_token: zeros(ctx, vec![1], DType::I32)?,
            proposals: zeros(ctx, vec![QWEN35_MTP_DRAFT_STEPS], DType::I32)?,
            verification_inputs: zeros(ctx, vec![QWEN35_MTP_DRAFT_STEPS + 1], DType::I32)?,
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
            draft_sessions: (0..QWEN35_MTP_DRAFT_STEPS)
                .map(|_| ExecutionSession::with_capacity(MTP_DRAFT_WORKSPACE, ctx.device_id()))
                .collect::<Result<Vec<_>>>()?,
            draft_sessions_prepared: [false; QWEN35_MTP_DRAFT_STEPS],
            materialize_session: None,
        })
    }
}

#[allow(clippy::too_many_arguments)]
fn materialize_final_proposal(
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
) -> Result<()> {
    let enabled = std::env::var("APXINF_QWEN35_MTP_MATERIALIZE_SESSION").as_deref() == Ok("1");
    let session = if enabled {
        let binding = MtpMaterializeBinding::new(ctx, weights)?;
        if scratch.materialize_session.as_ref().is_none_or(|cached| cached.binding != binding) {
            scratch.materialize_session = Some(MtpMaterializeSession {
                binding,
                session: ExecutionSession::with_capacity(MTP_DRAFT_WORKSPACE, ctx.device_id())?,
                prepared: false,
            });
        }
        let cached = scratch.materialize_session.as_ref().expect("initialized materialization session");
        Some((cached.session.clone(), cached.prepared))
    } else {
        None
    };
    // prepare executes this real row once; never prepare and then execute the
    // same materialization twice. Unlike draft, this tape has no LM-head GEMM.
    let mut forward = || forward_one(ctx, weights, state, scratch, geometry, eps,
        rope_theta, base_hidden, token, position, cache_row, draft_step, false);
    let result = match session.as_ref() {
        Some((execution, true)) => execution.run(forward),
        Some((execution, false)) => execution.prepare(forward),
        None => forward(),
    };
    if enabled {
        scratch.materialize_session.as_mut().expect("initialized materialization session").prepared = result.is_ok();
        // Do not retry a partially executed row. Drain before returning an
        // error, leave it unprepared and let the existing transaction abort.
        if result.is_err() { ctx.synchronize().map_err(Error::Cuda)?; }
    }
    // prepare_commit retains its unconditional successful ctx synchronization,
    // position validation and publish boundary after this function returns.
    result
}

#[allow(clippy::too_many_arguments)]
fn run_draft_chain(
    ctx: &CudaContext,
    weights: &Qwen35DeviceWeights,
    state: &mut Qwen35MtpState,
    scratch: &mut MtpScratch,
    geometry: AttentionGeometry,
    eps: f32,
    rope_theta: f32,
    target_hidden: &Tensor,
    draft_positions: &Tensor,
    start: MtpPositionContract,
    draft_steps: usize,
) -> Result<()> {
    let mut base_hidden = target_hidden.clone();
    for depth in 0..draft_steps {
        let token_input = scratch.token.clone();
        let rope_position = rows_view(draft_positions, depth, 1, 3)?;
        let rope_position = reshape(&rope_position, vec![3, 1], DType::I32)?;
        forward_one(
            ctx,
            weights,
            state,
            scratch,
            geometry,
            eps,
            rope_theta,
            &base_hidden,
            &token_input,
            &rope_position,
            start.physical_row_for_draft(depth)?,
            depth,
            true,
        )?;
        let proposal_slot = element_view(&scratch.proposals, depth)?;
        transfers::copy_cuda_to_cuda(ctx, &scratch.next_token, &proposal_slot)?;
        let verification_slot = element_view(&scratch.verification_inputs, depth + 1)?;
        transfers::copy_cuda_to_cuda(ctx, &scratch.next_token, &verification_slot)?;
        if depth + 1 < draft_steps {
            transfers::copy_cuda_to_cuda(ctx, &scratch.next_token, &scratch.token)?;
            base_hidden = scratch.final_hidden.clone();
        }
    }
    Ok(())
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
    let mut projected_key = scratch.projected_key.clone();
    let mut projected_value = scratch.projected_value.clone();
    ops::gemm(
        ctx,
        ops::GemmArgs::new(
            &scratch.normalized,
            &attention.k_proj.weight,
            &mut projected_key,
        ),
    )?;
    ops::gemm(
        ctx,
        ops::GemmArgs::new(
            &scratch.normalized,
            &attention.v_proj.weight,
            &mut projected_value,
        ),
    )?;
    let attention_out = reshape(
        &scratch.attention_out,
        vec![1, geometry.query_heads, geometry.head_dim],
        DType::BF16,
    )?;
    ops::qwen35_attention_fp8(
        ctx,
        1,
        ops::Qwen35Fp8AttentionArgs {
            projected_query_gate: &scratch.q_fused,
            projected_key: &scratch.projected_key,
            projected_value: &scratch.projected_value,
            query_norm: &attention.q_norm,
            key_norm: &attention.k_norm,
            positions: position,
            cache: ops::Qwen35Fp8KvCache {
                key: &state.key,
                value: &state.value,
            },
            scratch: ops::Qwen35AttentionScratch {
                query: &scratch.query,
                gate: &scratch.query_gate,
                output: &attention_out,
                scores: &scratch.attention_scores,
                partial_output: &scratch.attention_partial_output,
                partial_stats: &scratch.attention_partial_stats,
            },
            cache_offset: cache_row,
            scales: ops::Qwen35KvCacheScales {
                key: MTP_KV_SCALE,
                value: MTP_KV_SCALE,
            },
            rope_theta,
        },
    )?;
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
    if state.capacity == 0 || state.capacity % ops::QWEN35_KV_CACHE_BLOCK_TOKENS != 0 {
        return Err(invalid(
            "Qwen3.5 MTP FP8 cache capacity is not page aligned",
        ));
    }
    let shape = [
        state.capacity / ops::QWEN35_KV_CACHE_BLOCK_TOKENS,
        ops::QWEN35_KV_CACHE_BLOCK_TOKENS,
        geometry.kv_heads,
        geometry.head_dim,
    ];
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

fn write_i32(ctx: &CudaContext, destination: &Tensor, value: u32) -> Result<()> {
    let value = i32::try_from(value)
        .map_err(|_| invalid(format!("Qwen3.5 MTP value {value} exceeds i32")))?;
    let host = Tensor::from_i32(vec![1], &[value])?;
    transfers::copy_cpu_to_cuda_on_stream(ctx, host, destination)
}

fn write_text_position(ctx: &CudaContext, destination: &Tensor, value: u32) -> Result<()> {
    let value = i32::try_from(value)
        .map_err(|_| invalid(format!("Qwen3.5 MTP position {value} exceeds i32")))?;
    let host = Tensor::from_i32(vec![3, 1], &[value, value, value])?;
    transfers::copy_cpu_to_cuda_on_stream(ctx, host, destination)
}

fn zeros(ctx: &CudaContext, shape: Vec<usize>, dtype: DType) -> Result<Tensor> {
    let bytes = shape
        .iter()
        .try_fold(dtype.size_in_bytes(), |bytes, dimension| {
            bytes.checked_mul(*dimension)
        })
        .ok_or_else(|| invalid("Qwen3.5 MTP CUDA allocation overflow"))?;
    CudaBuffer::alloc_zeros_async(bytes, ctx.device_id(), ctx.stream())
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
        .checked_mul(cache.dtype().size_in_bytes())
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
        .as_tensor(Shape::new(vec![rows, heads, head_dim]), cache.dtype())
        .map_err(Error::Cuda)
}

fn cache_prefix_view(cache: &Tensor, rows: usize, heads: usize, head_dim: usize) -> Result<Tensor> {
    let bytes = rows
        .checked_mul(heads)
        .and_then(|n| n.checked_mul(head_dim))
        .and_then(|n| n.checked_mul(cache.dtype().size_in_bytes()))
        .ok_or_else(|| invalid("Qwen3.5 MTP cache prefix overflow"))?;
    CudaBuffer::from_tensor(cache)
        .map_err(Error::Cuda)?
        .view(0, bytes)
        .map_err(Error::Cuda)?
        .as_tensor(Shape::new(vec![1, rows, heads, head_dim]), cache.dtype())
        .map_err(Error::Cuda)
}

fn invalid(message: impl Into<String>) -> Error {
    Error::Other(message.into())
}

fn mtp_device_chain_enabled() -> bool {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ENABLED.get_or_init(|| {
        std::env::var("APXINF_QWEN35_MTP_DEVICE_CHAIN")
            .map(|value| !matches!(value.as_str(), "0" | "false" | "off"))
            .unwrap_or(true)
    })
}

fn schedule_error(error: impl std::fmt::Display) -> Error {
    invalid(format!("Qwen3.5 MTP schedule: {error}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prompt_shift_keeps_logical_and_physical_positions_distinct() {
        assert_eq!(QWEN35_MTP_KV_DTYPE, DType::F8E4M3);
        assert_eq!(MTP_KV_SCALE, 1.0);
        let state = MtpPositionContract::after_prompt(4_096).unwrap();
        assert_eq!(state.logical_position, 4_096);
        assert_eq!(state.cache_rows, 4_095);
        assert_eq!(state.physical_row_for_draft(0).unwrap(), 4_095);
        assert_eq!(state.rope_position_for_draft(0).unwrap(), 4_096);
        let last = QWEN35_MTP_DRAFT_STEPS - 1;
        assert_eq!(state.physical_row_for_draft(last).unwrap(), 4_095 + last);
        assert_eq!(state.rope_position_for_draft(last).unwrap(), 4_096 + last);
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
        for committed in 1..=QWEN35_MTP_DRAFT_STEPS + 1 {
            let next = start.after_commit(committed).unwrap();
            assert_eq!(next.logical_position, 17 + committed);
            assert_eq!(next.cache_rows, 16 + committed);
            assert_eq!(next.logical_position, next.cache_rows + 1);
        }
    }

    #[test]
    fn empty_prompt_and_out_of_range_draft_are_rejected() {
        assert!(MtpPositionContract::after_prompt(0).is_err());
        let state = MtpPositionContract::after_prompt(1).unwrap();
        assert!(state
            .physical_row_for_draft(QWEN35_MTP_DRAFT_STEPS)
            .is_err());
        assert!(state
            .rope_position_for_draft(QWEN35_MTP_DRAFT_STEPS)
            .is_err());
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

        let mut scalar = greedy.clone();
        scalar.speculative_decoding = Some(false);
        assert!(MtpGreedyPolicy::from_generation_options(&scalar).is_err());

        let mut penalized = greedy.clone();
        penalized.repetition_penalty = Some(1.1);
        assert!(MtpGreedyPolicy::from_generation_options(&penalized).is_err());

        let mut logprobs = greedy;
        logprobs.return_logprob = Some(true);
        assert!(MtpGreedyPolicy::from_generation_options(&logprobs).is_err());
    }
}

#[cfg(test)]
pub(super) mod gpu_raw_protocol {
    use super::*;
    use std::path::PathBuf;
    use crate::qwen35moe::{Qwen35MoeActivation, Qwen35KvCacheDType, Qwen35ExecutionOptions, Qwen35Fp8AttentionBackend};
    use crate::qwen35moe::device_weights::load_qwen35_device_weights_from_path_with_options;
    use crate::qwen35moe::mtp_schedule::ShadowCommitRange;

    fn parse_protocol_graphs(value: Option<&std::ffi::OsStr>) -> std::result::Result<bool, &'static str> {
        match value {
            None => Ok(false),
            Some(value) => match value.to_str() {
                Some("0") => Ok(false),
                Some("1") => Ok(true),
                _ => Err("APXINF_QWEN35_PROTOCOL_GRAPHS must be unset, 0 or 1"),
            },
        }
    }

    pub(in crate::qwen35moe) struct Fixture {
        pub(in crate::qwen35moe) ctx: CudaContext,
        pub(in crate::qwen35moe) config: Qwen35MoeConfig,
        pub(in crate::qwen35moe) decoder: Qwen35Decoder,
        pub(in crate::qwen35moe) weights: Qwen35DeviceWeights,
        pub(in crate::qwen35moe) prompt: Vec<u32>,
        pub(in crate::qwen35moe) capacity: usize,
        graphs: bool,
    }

    impl Fixture {
        fn load() -> std::result::Result<Self, Box<dyn std::error::Error>> {
            Self::load_with_autotune(false)
        }
        // Only the separately named ignored recipe-preparation entry calls true.
        // Strict GPU gates always enter through load(), regardless of environment.
        fn load_with_autotune(allow_online_tune: bool) -> std::result::Result<Self, Box<dyn std::error::Error>> {
            let protocol_graphs_env = std::env::var_os("APXINF_QWEN35_PROTOCOL_GRAPHS");
            let graphs = parse_protocol_graphs(protocol_graphs_env.as_deref())?;
            std::env::set_var("APXINF_REQUIRE_RECIPES", if allow_online_tune { "0" } else { "1" });
            std::env::set_var("APXINF_EDGE_NVFP4_MOE_ORDERED_REDUCE", "1");
            std::env::set_var("APXINF_QWEN35_CUDA_GRAPHS", if graphs { "1" } else { "0" });
            std::env::set_var("APXINF_QWEN35_VERIFY_GRAPHS", "0");
            let checkpoint = PathBuf::from(std::env::var_os("APXINF_QWEN35_CHECKPOINT").expect("checkpoint env"));
            let tuning = PathBuf::from(std::env::var_os("APXINF_QWEN35_PARITY_TUNING_DIR").expect("pre-tuned recipe env"));
            assert!(checkpoint.is_absolute() && tuning.is_absolute());
            if allow_online_tune { std::fs::create_dir_all(&tuning)?; }
            else {
                assert!(std::fs::read_dir(&tuning)?.filter_map(std::result::Result::ok)
                    .any(|x| x.path().extension().is_some_and(|x| x == "recipe")));
            }
            let ctx = CudaContext::new_with_options(0, allow_online_tune, Some(&tuning))?;
            let config = Qwen35MoeConfig::from_json_file(&checkpoint.join("config.json"))?;
            let policy = Qwen35ExecutionOptions { moe_activation: Qwen35MoeActivation::Nvfp4, kv_cache: Qwen35KvCacheDType::Bf16 };
            let weights = load_qwen35_device_weights_from_path_with_options(&ctx, &config, &checkpoint, policy)?;
            let decoder = Qwen35Decoder::new(&config)?;
            // Short fixed scope. Existing1024-token and long-context gates are separate.
            let prompt = vec![151644,8948,198,9707,16357,220,19687,25,198,151645,198,151644,77091,198,35064,16933];
            Ok(Self { ctx, config, decoder, weights, prompt, capacity:64, graphs })
        }
        pub(in crate::qwen35moe) fn target(&self) -> Result<Qwen35Session<Qwen35Fp8AttentionBackend>> {
            let backend = Qwen35Fp8AttentionBackend::with_kv_cache(&self.config, Qwen35KvCacheDType::Bf16, None)?;
            Qwen35Session::new(&self.ctx, &self.decoder, &self.weights, backend, self.capacity)
        }
    }

    pub(in crate::qwen35moe) fn raw(tensor: &Tensor) -> Result<Vec<u8>> {
        let cpu = transfers::to_cpu(tensor)?;
        if cpu.dtype() == DType::F32 { assert!(cpu.as_f32()?.iter().all(|x| x.is_finite())); }
        if cpu.dtype() == DType::BF16 { assert!(cpu.as_bf16()?.iter().all(|x| x.to_f32().is_finite())); }
        let bytes=cpu.storage().as_cpu().unwrap();
        if cpu.dtype()==DType::F8E4M3 { assert!(bytes.iter().all(|x|(*x & 0x7f)!=0x7f),"nonfinite E4M3 observation"); }
        Ok(bytes.to_vec())
    }
    pub(in crate::qwen35moe) fn digest_images(images: &[(Vec<u8>,Vec<u8>)]) -> String {
        let hash=images.iter().flat_map(|x|x.0.iter().chain(&x.1))
            .fold(0xcbf29ce484222325u64,|h,b|(h^u64::from(*b)).wrapping_mul(0x100000001b3));
        format!("{hash:016x}")
    }
    pub(in crate::qwen35moe) fn exact(label: &str, actual: &[u8], expected: &[u8]) {
        assert_eq!(actual.len(),expected.len(),"{label}: length");
        if let Some(i)=actual.iter().zip(expected).position(|(a,b)|a!=b) {
            panic!("{label}: first byte mismatch {i}: {} != {}",actual[i],expected[i]);
        }
    }
    pub(in crate::qwen35moe) fn pointer(t: &Tensor) -> usize { t.storage().as_gpu().unwrap().ptr() }
    pub(in crate::qwen35moe) fn fill_rows(ctx: &CudaContext, t: &Tensor, start: usize, end: usize, rows: usize, byte: u8) -> Result<()> {
        let b=CudaBuffer::from_tensor(t).map_err(Error::Cuda)?;
        assert_eq!(b.len()%rows,0);assert!(start<=end && end<=rows);
        b.view(start*(b.len()/rows),(end-start)*(b.len()/rows)).map_err(Error::Cuda)?
            .memset_async(byte,ctx.stream()).map_err(Error::Cuda)
    }
    // CPU copy ledger, independent of CUDA cache view/copy implementations.
    pub(in crate::qwen35moe) fn prefix_image(destination: &[u8], source: &[u8], row_bytes: usize, count: usize) -> Vec<u8> {
        let end=row_bytes.checked_mul(count).unwrap();assert!(end<=destination.len() && end<=source.len());
        let mut expected=destination.to_vec();expected[..end].copy_from_slice(&source[..end]);expected
    }
    #[test]
    fn cpu_draft_copy_ledger_counts_physical_p_minus_one_rows() {
        let source:Vec<u8>=(0..40u8).collect();let destination=vec![0x5au8;64];
        let expected=prefix_image(&destination,&source,2,15);
        assert_eq!(&expected[..30],&source[..30]);assert_eq!(&expected[30..],&destination[30..]);
        assert_eq!(prefix_image(&destination,&source,2,0),destination);
        assert_ne!(expected,prefix_image(&destination,&source,2,16),"logicalP16 must not copy physical16rows");
    }
    fn image(state: &Qwen35MtpState) -> Result<(Vec<u8>,Vec<u8>)> { Ok((raw(&state.key)?,raw(&state.value)?)) }
    fn same(label: &str, state: &Qwen35MtpState, bytes: &(Vec<u8>,Vec<u8>)) -> Result<()> {
        exact(&format!("{label} K"),&raw(&state.key)?,&bytes.0);exact(&format!("{label} V"),&raw(&state.value)?,&bytes.1);Ok(())
    }
    fn ids(state: &Qwen35MtpState) -> (usize,usize) { (pointer(&state.key),pointer(&state.value)) }
    fn check_copy(f: &Fixture, runtime: &mut Qwen35MtpRuntime, source: &Qwen35MtpState, destination: &mut Qwen35MtpState) -> Result<()> {
        let before=image(destination)?;let source_before=image(source)?;
        let count=source.position.map(|p|p.cache_rows).unwrap_or(0);
        let row_bytes=source_before.0.len()/source.capacity;
        let expected=(prefix_image(&before.0,&source_before.0,row_bytes,count),prefix_image(&before.1,&source_before.1,row_bytes,count));
        runtime.copy_state(&f.ctx,source,destination)?;f.ctx.synchronize().map_err(Error::Cuda)?;
        same("copy destination independent CPU prefix ledger",destination,&expected)?;
        same("copy source readonly",source,&source_before)?;
        assert_eq!(destination.position,source.position);assert_eq!(destination.generation,source.generation+1);
        assert!(runtime.pending_draft.is_none());Ok(())
    }

    fn draft_gate(f: &Fixture) -> Result<()> {
        let mut target=f.target()?;
        let prepared_target=target.prepare_target_prefill(&f.ctx,&f.decoder,&f.weights,&f.prompt)?;
        let hidden_rows=prepared_target.output().hidden_states.clone();
        let last_hidden=prepared_target.output().last_hidden_state()?;
        let pending=prepared_target.output().greedy_token_ids(&f.ctx)?[0];
        target.publish_target_prefill(prepared_target)?;
        let mut runtime=Qwen35MtpRuntime::new(&f.config)?;
        let mut committed=runtime.allocate_state(&f.ctx,f.capacity)?;
        let mut shadow=runtime.allocate_state(&f.ctx,f.capacity)?;
        let mut snapshot=runtime.allocate_state(&f.ctx,f.capacity+128)?;
        assert_eq!(committed.capacity,128);assert_eq!(shadow.capacity,128);assert_eq!(snapshot.capacity,256);
        assert_ne!(ids(&committed),ids(&shadow));assert_ne!(ids(&committed),ids(&snapshot));assert_ne!(ids(&shadow),ids(&snapshot));
        for state in [&committed,&shadow,&snapshot] {
            assert_eq!(state.key.shape().dims(),[state.capacity/128,128,runtime.geometry.kv_heads,runtime.geometry.head_dim]);
            assert_eq!(state.key.dtype(),DType::F8E4M3);assert_eq!(state.value.shape(),state.key.shape());
        }
        runtime.reset_state(&f.ctx,&mut committed)?;runtime.reset_state(&f.ctx,&mut shadow)?;
        f.ctx.synchronize().map_err(Error::Cuda)?;
        let empty=image(&committed)?;assert!(empty.0.iter().chain(&empty.1).all(|x|*x==0));
        fill_rows(&f.ctx,&snapshot.key,0,snapshot.capacity,snapshot.capacity,0x4a)?;
        fill_rows(&f.ctx,&snapshot.value,0,snapshot.capacity,snapshot.capacity,0x4a)?;
        check_copy(f,&mut runtime,&committed,&mut snapshot)?; // None means no bytes copied.
        let committed_before=image(&committed)?;let committed_meta=(committed.position,committed.generation);
        let hidden_before=raw(&hidden_rows)?;
        let prepared=runtime.prepare_prefill(&f.ctx,&f.weights,&mut shadow,&f.prompt,&hidden_rows)?;
        same("prefill must not mutate committed",&committed,&committed_before)?;
        assert_eq!((committed.position,committed.generation),committed_meta);
        exact("teacher forcing hidden input readonly",&raw(&hidden_rows)?,&hidden_before);
        assert_eq!(prepared.position,MtpPositionContract::after_prompt(f.prompt.len())?);
        assert_eq!(shadow.position,Some(prepared.position));assert_eq!(shadow.generation,prepared.generation);
        let rows=f.prompt.len()-1;let row_bytes=raw(&shadow.key)?.len()/shadow.capacity;
        // prepare_prefill declares a full reset, so its unused tail must be zero.
        for tensor in [&shadow.key,&shadow.value] { assert!(raw(tensor)?[rows*row_bytes..].iter().all(|x|*x==0)); }
        let staged=image(&shadow)?;let staged_ids=ids(&shadow);let old_ids=ids(&committed);
        runtime.publish_prefill(&mut committed,&mut shadow,prepared);
        assert_eq!(ids(&committed),staged_ids);assert_eq!(ids(&shadow),old_ids);same("prefill publication exact swap",&committed,&staged)?;
        check_copy(f,&mut runtime,&committed,&mut snapshot)?;
        let frozen=image(&snapshot)?;let frozen_meta=(snapshot.position,snapshot.generation);
        for depth in 1..=3usize {
            for keep in 1..=depth+1 {
                check_copy(f,&mut runtime,&snapshot,&mut committed)?;
                assert_eq!(committed.position,Some(MtpPositionContract::after_prompt(f.prompt.len())?));
                for tensor in [&committed.key,&committed.value] { fill_rows(&f.ctx,tensor,rows,committed.capacity,committed.capacity,0x3a)?; }
                for tensor in [&shadow.key,&shadow.value] { fill_rows(&f.ctx,tensor,0,shadow.capacity,shadow.capacity,0x5a)?; }
                let old=image(&committed)?;let old_meta=(committed.position,committed.generation);let old_ids=ids(&committed);
                let _draft=runtime.draft(&f.ctx,&f.weights,&committed,&mut shadow,pending,f.prompt.len(),&last_hidden,depth)?;
                f.ctx.synchronize().map_err(Error::Cuda)?;
                same("draft committed full physical bank readonly",&committed,&old)?;assert_eq!((committed.position,committed.generation),old_meta);
                let staged_metadata=runtime.pending_draft.expect("real draft transaction must be pending");
                assert_eq!(staged_metadata.draft_steps,depth);assert_eq!(staged_metadata.shadow_generation,shadow.generation);
                assert_eq!(staged_metadata.start,MtpPositionContract::after_prompt(f.prompt.len())?);
                let after_draft=image(&shadow)?;
                exact("draft old K prefix",&after_draft.0[..rows*row_bytes],&old.0[..rows*row_bytes]);
                exact("draft old V prefix",&after_draft.1[..rows*row_bytes],&old.1[..rows*row_bytes]);
                assert!(after_draft.0[(rows+depth)*row_bytes..].iter().chain(&after_draft.1[(rows+depth)*row_bytes..]).all(|x|*x==0x5a));
                let full=keep==depth+1;
                let contract=MtpCommit { cache_range:ShadowCommitRange { start:f.prompt.len(),committed_end:f.prompt.len()+keep,
                    verified_end:f.prompt.len()+if full {depth+1}else{depth} },materialize_final_proposal:full };
                let prepared=runtime.prepare_commit(&f.ctx,&f.weights,&mut shadow,contract)?;
                let after_prepare=image(&shadow)?;
                // Full commit may materialize exactly one final physical row;
                // partial commit only changes metadata, never clears its tail.
                for (name,a,b) in [("K",&after_prepare.0,&after_draft.0),("V",&after_prepare.1,&after_draft.1)] {
                    if full {
                        exact(&format!("full materialization {name} old prefix"),&a[..(rows+depth)*row_bytes],&b[..(rows+depth)*row_bytes]);
                        exact(&format!("full materialization {name} future canary"),&a[(rows+depth+1)*row_bytes..],&b[(rows+depth+1)*row_bytes..]);
                    } else { exact(&format!("partial commit {name} all bytes unchanged"),a,b); }
                }
                same("prepare commit committed bank readonly",&committed,&old)?;assert_eq!((committed.position,committed.generation),old_meta);
                let expected_position=MtpPositionContract {logical_position:f.prompt.len()+keep,cache_rows:rows+keep};
                assert_eq!(shadow.position,Some(expected_position));assert_eq!(prepared.position,expected_position);
                let staged_ids=ids(&shadow);let generation=shadow.generation;
                assert_eq!(prepared.generation,generation);
                runtime.publish_commit(&mut committed,&mut shadow,prepared);
                assert!(runtime.pending_draft.is_none());
                assert_eq!(ids(&committed),staged_ids);assert_eq!(ids(&shadow),old_ids);
                same("commit publish exact physical swap",&committed,&after_prepare)?;same("old physical bank unchanged",&shadow,&old)?;
                assert_eq!(committed.position,Some(expected_position));assert_eq!(committed.generation,generation);
                same("MTP snapshot deepfreeze",&snapshot,&frozen)?;assert_eq!((snapshot.position,snapshot.generation),frozen_meta);
                check_copy(f,&mut runtime,&snapshot,&mut committed)?; // Short restore must preserve unreachable tails.
                println!("{}",serde_json::json!({"schema":"apxinf.qwen35.draft_raw_bank.v1","depth":depth,"keep":keep,
                    "logical_position":expected_position.logical_position,"physical_cache_rows":expected_position.cache_rows,
                    "full_materialization":full,"copy_protocol_exact":true,
                    "published_source_digest":digest_images(std::slice::from_ref(&after_prepare)),
                    "snapshot_digest":digest_images(std::slice::from_ref(&frozen)),
                    "independent_gdn_math_oracle":false,"performance_accepted":false}));
            }
        }
        // Append genuine target hidden rows after the restored short prefix.
        let suffix=[1234u32,5678];
        let target_append=target.prepare_target_append_prefill(&f.ctx,&f.decoder,&f.weights,&suffix)?;
        for tensor in [&shadow.key,&shadow.value] { fill_rows(&f.ctx,tensor,0,shadow.capacity,shadow.capacity,0x5a)?; }
        let before=image(&shadow)?;let old=image(&committed)?;
        let prepared=runtime.prepare_append_prefill(&f.ctx,&f.weights,&committed,&mut shadow,&suffix,&last_hidden,&target_append.output().hidden_states)?;
        let after=image(&shadow)?;
        for (a,b) in [(&after.0,&before.0),(&after.1,&before.1)] { exact("append future tail unchanged",&a[(rows+suffix.len())*row_bytes..],&b[(rows+suffix.len())*row_bytes..]); }
        exact("append copied old K prefix",&after.0[..rows*row_bytes],&old.0[..rows*row_bytes]);
        exact("append copied old V prefix",&after.1[..rows*row_bytes],&old.1[..rows*row_bytes]);
        same("append committed readonly",&committed,&old)?;
        runtime.publish_prefill(&mut committed,&mut shadow,prepared);
        assert_eq!(committed.position,Some(MtpPositionContract {logical_position:f.prompt.len()+suffix.len(),cache_rows:rows+suffix.len()}));
        same("append publication bytes",&committed,&after)?;same("snapshot still deepfrozen",&snapshot,&frozen)?;
        check_copy(f,&mut runtime,&snapshot,&mut committed)?;
        Ok(())
    }

    #[test]
    #[ignore = "real checkpoint raw target/draft GPU bank gate; single weights load; exclusive lock and pre-tuned strict recipes"]
    fn real_checkpoint_target_and_draft_raw_bank_protocol() -> std::result::Result<(),Box<dyn std::error::Error>> {
        let f=Fixture::load()?;
        println!("{}",serde_json::json!({"schema":"apxinf.qwen35.raw_bank_config.v1","prompt_tokens":16,
            "capacity":64,"block_tokens":128,"target_kv":"bf16","draft_kv":"fp8","draft_scale":1.0,
            "depths":[1,2,3],"target_commit_cases":9,"draft_commit_cases":9,"weights_loads":1,
            "strict_recipe_required":true,"autotune":false,"ordered_reduce":true,"graphs":f.graphs,"graph_gdn":std::env::var("APXINF_QWEN35_GRAPH_GDN").ok(),
            "copy_protocol_only":true,"independent_gdn_math_oracle":false,"performance_accepted":false}));
        super::super::session::gpu_raw_protocol::target_gate(&f)?;
        draft_gate(&f)?;
        Ok(())
    }
    // Recipe preparation is an explicit different ignored test and process.
    // It executes actual target/MTP operations; it is not a protocol/quality gate.
    fn prep_command(program: &str, args: &[&str]) -> std::result::Result<Vec<u8>,Box<dyn std::error::Error>> {
        let output=std::process::Command::new(program).args(args).output()?;
        if !output.status.success() { return Err(format!("{program} failed: {}",String::from_utf8_lossy(&output.stderr)).into()); }
        Ok(output.stdout)
    }
    fn prep_sha(bytes: &[u8]) -> std::result::Result<String,Box<dyn std::error::Error>> {
        use std::io::Write;
        let mut child=std::process::Command::new("sha256sum").stdin(std::process::Stdio::piped()).stdout(std::process::Stdio::piped()).spawn()?;
        child.stdin.take().expect("sha256 stdin").write_all(bytes)?;
        let output=child.wait_with_output()?;
        if !output.status.success() { return Err("sha256sum failed".into()); }
        let text=String::from_utf8(output.stdout)?;
        let hash=text.split_whitespace().next().ok_or("empty sha256sum")?;
        if hash.len()!=64 || !hash.bytes().all(|x|x.is_ascii_hexdigit()) { return Err("invalid sha256sum".into()); }
        Ok(hash.to_owned())
    }
    fn prep_file_sha(path: &std::path::Path) -> std::result::Result<String,Box<dyn std::error::Error>> {
        let output=std::process::Command::new("sha256sum").arg("--").arg(path).output()?;
        if !output.status.success() { return Err(format!("sha256sum failed for {}",path.display()).into()); }
        let text=String::from_utf8(output.stdout)?;
        let hash=text.split_whitespace().next().ok_or("empty file sha256sum")?;
        if hash.len()!=64 || !hash.bytes().all(|x|x.is_ascii_hexdigit()) { return Err("invalid file sha256sum".into()); }
        Ok(hash.to_owned())
    }
    fn prep_files(root: &std::path::Path, files: &mut Vec<PathBuf>) -> std::io::Result<()> {
        if root.exists() {
            for entry in std::fs::read_dir(root)? {
                let path=entry?.path();
                if path.is_dir() { prep_files(&path,files)?; }
                else if path.is_file() { files.push(path); }
            }
        }
        Ok(())
    }
    fn prep_inventory(root: &std::path::Path) -> std::result::Result<serde_json::Value,Box<dyn std::error::Error>> {
        let mut paths=Vec::new();prep_files(root,&mut paths)?;paths.sort();
        let mut files=Vec::new();let mut recipe_count=0usize;
        for path in paths {
            if path.extension().is_some_and(|x|x=="recipe") { recipe_count+=1; }
            files.push(serde_json::json!({"path":path.strip_prefix(root)?,"bytes":path.metadata()?.len(),"sha256":prep_file_sha(&path)?}));
        }
        Ok(serde_json::json!({"root":root,"recipe_count":recipe_count,"files":files,"files_json_sha256":prep_sha(&serde_json::to_vec(&files)?)?}))
    }

    #[test]
    #[ignore = "RECIPE PREPARATION ONLY: real weights once, online tune enabled, writes explicit tuning dir; not quality or performance acceptance"]
    fn prepare_real_checkpoint_protocol_recipes() -> std::result::Result<(),Box<dyn std::error::Error>> {
        let arguments:Vec<String>=std::env::args().collect();
        let entry="qwen35moe::mtp_runtime::gpu_raw_protocol::prepare_real_checkpoint_protocol_recipes";
        assert!(arguments.iter().any(|x|x=="--ignored") && arguments.iter().any(|x|x=="--exact")
            && arguments.iter().filter(|x|x.as_str()==entry).count()==1
            && arguments.iter().any(|x|x=="--test-threads=1"),
            "recipe preparation requires --ignored --exact <full prepare name> --test-threads=1; never run all ignored tests");
        let checkpoint=PathBuf::from(std::env::var_os("APXINF_QWEN35_CHECKPOINT").expect("checkpoint env"));
        let tuning=PathBuf::from(std::env::var_os("APXINF_QWEN35_PARITY_TUNING_DIR").expect("tuning env"));
        assert!(checkpoint.is_absolute() && tuning.is_absolute());
        let mut model_files=Vec::new();
        let mut paths:Vec<_>=std::fs::read_dir(&checkpoint)?.map(|x|x.map(|y|y.path())).collect::<std::io::Result<_>>()?;paths.sort();
        for path in paths {
            if path.is_file() && (path.extension().is_some_and(|x|x=="safetensors") || path.file_name().is_some_and(|x|x=="config.json" || x=="hf_quant_config.json")) {
                model_files.push(serde_json::json!({"path":path,"bytes":path.metadata()?.len(),"sha256":prep_file_sha(&path)?}));
            }
        }
        assert!(!model_files.is_empty(),"actual model files required");
        let head=String::from_utf8(prep_command("git",&["rev-parse","HEAD"])?)?;
        let status=String::from_utf8(prep_command("git",&["status","--porcelain","--untracked-files=all"])?)?;
        let patch=prep_command("git",&["diff","HEAD","--binary"])?;
        let mut untracked=Vec::new();
        for name in prep_command("git",&["ls-files","--others","--exclude-standard","-z"])?.split(|x|*x==0).filter(|x|!x.is_empty()) {
            let path=PathBuf::from(std::str::from_utf8(name)?);
            if path.is_file() { untracked.push(serde_json::json!({"path":path,"sha256":prep_file_sha(&path)?})); }
        }
        let protocol_graphs_env = std::env::var_os("APXINF_QWEN35_PROTOCOL_GRAPHS");
        let graphs = parse_protocol_graphs(protocol_graphs_env.as_deref())?;
        let before=prep_inventory(&tuning)?;
        println!("{}",serde_json::json!({"schema":"apxinf.qwen35.protocol_recipe_prepare_identity.v2",
            "phase":"prepare_only","source_head":head.trim(),"source_status":status,"source_patch_sha256":prep_sha(&patch)?,"source_untracked_sha256":untracked,
            "executable":std::env::current_exe()?,"executable_sha256":prep_file_sha(&std::env::current_exe()?)?,"model_files":model_files,
            "recipes_before":before,"autotune_constructor_argument":true,"strict_recipe_required":false,"apxinf_require_recipes":"0",
            "target_kv":"bf16","draft_kv":"fp8","draft_scale":1.0,"moe":"nvfp4","ordered_reduce":true,"graphs":graphs,"graph_gdn":std::env::var("APXINF_QWEN35_GRAPH_GDN").ok(),
            "weights_loads":1,"quality_accepted":false,"performance_accepted":false}));
        let f=Fixture::load_with_autotune(true)?;
        assert_eq!(f.graphs, graphs);
        assert_eq!(std::env::var("APXINF_REQUIRE_RECIPES").unwrap(),"0");
        // Raw gate logical capacity64; the public protocol test defaults to
        // prompt16+64=80. Both map to the same128-row physical KV block, but
        // exercise both constructors rather than assuming recipe equivalence.
        for logical_capacity in [64usize,80] {
        let backend=Qwen35Fp8AttentionBackend::with_kv_cache(&f.config,Qwen35KvCacheDType::Bf16,None)?;
        let mut target=Qwen35Session::new(&f.ctx,&f.decoder,&f.weights,backend,logical_capacity)?;
        let mut runtime=Qwen35MtpRuntime::new(&f.config)?;
        let mut committed=runtime.allocate_state(&f.ctx,logical_capacity)?;
        let mut shadow=runtime.allocate_state(&f.ctx,logical_capacity)?;
        let verify_ids=[42i32,1729,19,777];
        for p in 16..=23usize {
            // Original16-token prefix is unchanged; additional rows are an
            // explicitly recorded repetition of that same protocol pattern.
            let prompt:Vec<u32>=f.prompt.iter().copied().cycle().take(p).collect();
            for width in 2..=4usize {
                let input=transfers::to_cuda_on_stream(&f.ctx,Tensor::from_i32(vec![width],&verify_ids[..width])?)?;
                for keep in 1..=width {
                    target.reset(&f.ctx)?;
                    target.prefill(&f.ctx,&f.decoder,&f.weights,&prompt)?;
                    let verified=target.verify_target_chunk_device(&f.ctx,&f.decoder,&f.weights,&input)?;
                    let staged=target.prepare_target_commit(&f.ctx,&f.decoder,&f.weights,verified,keep)?;
                    target.publish_target_commit(staged)?;
                }
            }
            // Scalar one-row kernels/attention at each required short position;
            // four steps also cover the later public protocol continuation tail.
            target.reset(&f.ctx)?;target.prefill(&f.ctx,&f.decoder,&f.weights,&prompt)?;
            for token in [991u32,998,654,991] { target.decode(&f.ctx,&f.decoder,&f.weights,token)?; }
            for suffix in [&[1234u32,5678][..],&[3456u32,7890][..],&[2345u32,6789][..],&[1234u32][..]] {
                target.reset(&f.ctx)?;
                let base=target.prepare_target_prefill(&f.ctx,&f.decoder,&f.weights,&prompt)?;
                let hidden=base.output().hidden_states.clone();
                let last=base.output().last_hidden_state()?;
                let pending=base.output().greedy_token_ids(&f.ctx)?[0];
                target.publish_target_prefill(base)?;
                if suffix==&[1234u32,5678][..] {
                for depth in 1..=3usize {
                    for keep in 1..=depth+1 {
                        runtime.reset_state(&f.ctx,&mut committed)?;runtime.reset_state(&f.ctx,&mut shadow)?;
                        let prepared=runtime.prepare_prefill(&f.ctx,&f.weights,&mut shadow,&prompt,&hidden)?;
                        runtime.publish_prefill(&mut committed,&mut shadow,prepared);
                        runtime.draft(&f.ctx,&f.weights,&committed,&mut shadow,pending,p,&last,depth)?;
                        let full=keep==depth+1;
                        let contract=MtpCommit {cache_range:ShadowCommitRange {start:p,committed_end:p+keep,verified_end:p+if full {depth+1}else{depth}},materialize_final_proposal:full};
                        let prepared=runtime.prepare_commit(&f.ctx,&f.weights,&mut shadow,contract)?;
                        runtime.publish_commit(&mut committed,&mut shadow,prepared);
                    }
                }
                }
                // Restore the original target-aligned draft prefix before
                // actual one/two-row target+shifted draft teacher forcing.
                runtime.reset_state(&f.ctx,&mut committed)?;runtime.reset_state(&f.ctx,&mut shadow)?;
                let prepared=runtime.prepare_prefill(&f.ctx,&f.weights,&mut shadow,&prompt,&hidden)?;
                runtime.publish_prefill(&mut committed,&mut shadow,prepared);
                let appended=target.prepare_target_append_prefill(&f.ctx,&f.decoder,&f.weights,suffix)?;
                let prepared=runtime.prepare_append_prefill(&f.ctx,&f.weights,&committed,&mut shadow,suffix,&last,&appended.output().hidden_states)?;
                runtime.publish_prefill(&mut committed,&mut shadow,prepared);
                target.publish_target_prefill(appended)?;
            }
            f.ctx.synchronize().map_err(Error::Cuda)?;
            println!("{}",serde_json::json!({"schema":"apxinf.qwen35.protocol_recipe_prepare_shape.v2",
                "prompt_tokens":p,"prompt_token_ids":prompt,"target_capacity":logical_capacity,"physical_block_tokens":128,
                "target_verify_rows":[2,3,4],"target_commit_keeps":"1..=verified_rows","scalar_steps":4,
                "append_tokens":[[1234,5678],[3456,7890],[2345,6789],[1234]],"draft_prefill_rows":p-1,
                "draft_depths":[1,2,3],"draft_commit_keeps":"1..=depth+1","actual_gpu_operations_completed":true,
                "quality_accepted":false,"performance_accepted":false}));
        }
        }
        f.ctx.synchronize().map_err(Error::Cuda)?;
        let after=prep_inventory(&tuning)?;
        assert!(after["recipe_count"].as_u64().unwrap_or(0)>0,"real persistent recipes required");
        println!("{}",serde_json::json!({"schema":"apxinf.qwen35.protocol_recipe_prepare_complete.v2",
            "recipes_after":after,"phase":"prepare_only","shape_iterations_completed":16,
            "native_numeric_validation":"unchanged; operator errors propagate; no override installed",
            "quality_accepted":false,"performance_accepted":false,"strict_followup_requires_fresh_process":true}));
        Ok(())
    }
}
