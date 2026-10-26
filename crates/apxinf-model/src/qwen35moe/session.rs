//! Transactional sequence ownership for the Qwen3.5 target decoder.
//!
//! Every mutable recurrent/KV state has committed and shadow storage. A call
//! prepares the shadow, runs all 40 layers, synchronizes the CUDA stream and
//! only then swaps storage and advances the visible position. This gives
//! prefill/decode a strong error guarantee without host snapshots.

use apxinf_core::{Error, Result, Tensor};
use apxinf_cuda_new::{transfers, CudaContext};

use super::architecture::MixerKind;
use super::decoder::{
    DecoderOutput, DecoderPhase, DecoderScratch, FullAttentionBackend, HiddenSelection,
    LogitsSelection, Qwen35Decoder,
};
use super::device_weights::{DeviceMixerWeights, Qwen35DeviceWeights};
use super::forward::GdnState;
use super::mtp_schedule::{MtpCommit, TokenId, QWEN35_MTP_DRAFT_STEPS, QWEN35_MTP_VERIFY_ROWS};

/// Device MTP execution boundary. The schedule already decides which rows are
/// accepted; this trait owns the complete BF16 fusion, attention/cache, and
/// MoE executor around the native primitives. A conforming implementation
/// must stage draft work
/// without changing committed visibility, finish every fallible operation in
/// `prepare_commit`, and make `publish_commit` an allocation-free state swap.
/// The coordinator must synchronize `ctx` after each successful `prepare_*`
/// call and before either target or MTP state is published.
pub trait MtpBackend {
    type State;
    type PreparedPrefill;
    type PreparedCommit;

    fn allocate_state(&mut self, ctx: &CudaContext, capacity: usize) -> Result<Self::State>;

    fn reset_state(&mut self, ctx: &CudaContext, state: &mut Self::State) -> Result<()>;

    fn copy_state(
        &mut self,
        ctx: &CudaContext,
        source: &Self::State,
        destination: &mut Self::State,
    ) -> Result<()>;

    /// Fill the MTP layer's own KV cache from a base prompt. `weights` includes
    /// the shared token embedding/lm_head and `weights.mtp`; `target_hidden`
    /// contains all post-final-norm target rows returned by
    /// [`Qwen35Session::prepare_target_prefill`]. The canonical alignment is
    /// embeddings from `prompt_tokens[1..]` with target hidden rows
    /// `[..prompt_tokens.len()-1]` at MTP cache positions starting at zero.
    fn prepare_prefill(
        &mut self,
        ctx: &CudaContext,
        weights: &Qwen35DeviceWeights,
        shadow: &mut Self::State,
        prompt_tokens: &[TokenId],
        target_hidden: &apxinf_core::Tensor,
    ) -> Result<Self::PreparedPrefill>;

    /// Extend an already committed MTP prompt state with a new target-prefill
    /// chunk. `previous_target_hidden` is the final target row before the
    /// chunk; `target_hidden` contains every target row produced by the new
    /// chunk. Together they preserve the shifted teacher-forcing alignment:
    /// chunk token `i` consumes the target hidden row immediately before it.
    fn prepare_append_prefill(
        &mut self,
        ctx: &CudaContext,
        weights: &Qwen35DeviceWeights,
        committed: &Self::State,
        shadow: &mut Self::State,
        chunk_tokens: &[TokenId],
        previous_target_hidden: &apxinf_core::Tensor,
        target_hidden: &apxinf_core::Tensor,
    ) -> Result<Self::PreparedPrefill>;

    /// Allocation-free publication after both target prefill and MTP prefill
    /// have succeeded. The resulting logical MTP position equals the base
    /// prompt length `P`; physical MTP KV rows are `P-1` because teacher
    /// forcing aligns `prompt[1..]` with target hidden `[..P-1]`.
    fn publish_prefill(
        &mut self,
        committed: &mut Self::State,
        shadow: &mut Self::State,
        prepared: Self::PreparedPrefill,
    );

    /// Produce the fixed three-token chain. `weights` includes the shared token
    /// embedding/lm_head and `weights.mtp`; `target_hidden` is the last
    /// committed post-final-norm target hidden row with shape `[1, hidden]`.
    fn draft(
        &mut self,
        ctx: &CudaContext,
        weights: &Qwen35DeviceWeights,
        committed: &Self::State,
        shadow: &mut Self::State,
        pending: TokenId,
        start_position: usize,
        target_hidden: &apxinf_core::Tensor,
    ) -> Result<[TokenId; QWEN35_MTP_DRAFT_STEPS]>;

    /// Finish the MTP half of a speculative transaction. `weights` is needed
    /// when full acceptance plus a bonus requires materializing the final
    /// proposal's BF16 KV row without another lm_head projection.
    fn prepare_commit(
        &mut self,
        ctx: &CudaContext,
        weights: &Qwen35DeviceWeights,
        shadow: &mut Self::State,
        contract: MtpCommit,
    ) -> Result<Self::PreparedCommit>;

    fn publish_commit(
        &mut self,
        committed: &mut Self::State,
        shadow: &mut Self::State,
        prepared: Self::PreparedCommit,
    );
}

/// Opaque evidence that the four target rows were
/// evaluated against a particular committed session epoch. Dropping it aborts
/// the target half of the speculative transaction.
pub struct TargetVerification {
    epoch: u64,
    shadow_generation: u64,
    start_position: usize,
    inputs: Vec<TokenId>,
    predictions: Vec<TokenId>,
    final_output: DecoderOutput,
}

impl TargetVerification {
    pub fn inputs(&self) -> &[TokenId] {
        &self.inputs
    }

    pub fn predictions(&self) -> &[TokenId] {
        &self.predictions
    }
}

/// Prepared target state whose fallible replay/synchronize work is complete.
/// A coordinator can now prepare the MTP commit and publish both shadows.
pub struct PreparedTargetCommit {
    epoch: u64,
    shadow_generation: u64,
    end_position: usize,
    output: DecoderOutput,
}

impl PreparedTargetCommit {
    pub fn output(&self) -> &DecoderOutput {
        &self.output
    }
}

/// Prepared target prompt whose kernels and synchronize have succeeded but
/// whose GDN/KV shadow is not visible yet. This lets an MTP coordinator prepare
/// its teacher-forcing prefill before atomically publishing both state trees.
pub struct PreparedTargetPrefill {
    epoch: u64,
    shadow_generation: u64,
    start_position: usize,
    end_position: usize,
    output: DecoderOutput,
}

impl PreparedTargetPrefill {
    pub fn output(&self) -> &DecoderOutput {
        &self.output
    }
}

/// One independent text sequence. Sessions must not be shared by concurrent
/// requests; serving should allocate or pool one session per active sequence.
pub struct Qwen35Session<B: FullAttentionBackend> {
    attention: B,
    committed_gdn: Vec<GdnState>,
    shadow_gdn: Vec<GdnState>,
    committed_attention: Vec<B::Cache>,
    shadow_attention: Vec<B::Cache>,
    /// Prefix of the inactive cache bank known to equal the committed bank.
    shadow_attention_valid: usize,
    /// Persistent one-row workspace shared by ordinary decode, scalar target
    /// verification, and partial-commit replay. Variable-size prefill keeps
    /// using call-sized temporary scratch.
    decode_scratch: DecoderScratch,
    /// Persistent four-row candidate workspace. It is enabled only for parity
    /// and performance validation until target-state equivalence is proven on
    /// the real checkpoint.
    verify_scratch: DecoderScratch,
    batched_verify: bool,
    /// Four staged verification inputs and their device argmax outputs. These
    /// eliminate per-row H2D traffic, allocation and host synchronization.
    verify_token_ids: Tensor,
    verify_predictions: Tensor,
    capacity: usize,
    position: usize,
    epoch: u64,
    shadow_generation: u64,
}

/// Device-resident checkpoint for one committed target prefix.
///
/// The checkpoint owns independent storage: later decode transactions may
/// freely swap and overwrite the session's committed/shadow banks without
/// invalidating a reusable prefix. Only rows before `position` are defined in
/// the attention caches.
pub struct Qwen35SessionSnapshot<C> {
    gdn: Vec<GdnState>,
    attention: Vec<C>,
    capacity: usize,
    position: usize,
}

impl<B: FullAttentionBackend> Qwen35Session<B> {
    pub fn new(
        ctx: &CudaContext,
        decoder: &Qwen35Decoder,
        weights: &Qwen35DeviceWeights,
        mut attention: B,
        capacity: usize,
    ) -> Result<Self> {
        decoder.validate_weights(weights)?;
        let architecture = decoder.architecture();
        if capacity == 0 || capacity > architecture.max_position_embeddings {
            return Err(invalid(format!(
                "Qwen3.5 session capacity must be in 1..={}, got {capacity}",
                architecture.max_position_embeddings
            )));
        }
        let mut committed_gdn = Vec::with_capacity(architecture.gdn_layers().count());
        let mut shadow_gdn = Vec::with_capacity(architecture.gdn_layers().count());
        let mut committed_attention =
            Vec::with_capacity(architecture.full_attention_layers().count());
        let mut shadow_attention = Vec::with_capacity(architecture.full_attention_layers().count());

        for (layer, resident) in architecture.layers.iter().zip(&weights.layers) {
            match (&layer.mixer, &resident.mixer) {
                (MixerKind::Gdn(_), DeviceMixerWeights::Gdn(_)) => {
                    committed_gdn.push(GdnState::new(ctx)?);
                    shadow_gdn.push(GdnState::new(ctx)?);
                }
                (
                    MixerKind::FullAttention(geometry),
                    DeviceMixerWeights::FullAttention(full_attention),
                ) => {
                    let scales = full_attention.kv_cache_scales.ok_or_else(|| {
                        invalid(format!(
                            "Qwen3.5 base attention layer {} is missing FP8 KV scales",
                            layer.index
                        ))
                    })?;
                    let committed =
                        attention.allocate_cache(ctx, layer.index, *geometry, capacity, scales)?;
                    let shadow =
                        attention.allocate_cache(ctx, layer.index, *geometry, capacity, scales)?;
                    committed_attention.push(committed);
                    shadow_attention.push(shadow);
                }
                _ => {
                    return Err(invalid(format!(
                        "Qwen3.5 layer {} mixer changed after validation",
                        layer.index
                    )));
                }
            }
        }

        let verify_zeros = vec![0i32; QWEN35_MTP_VERIFY_ROWS];
        let verify_zeros = Tensor::from_i32(vec![QWEN35_MTP_VERIFY_ROWS], &verify_zeros)?;
        let verify_token_ids = transfers::to_cuda(&verify_zeros, ctx.device_id())?;
        let verify_predictions = transfers::to_cuda(&verify_zeros, ctx.device_id())?;
        let mut session = Self {
            attention,
            committed_gdn,
            shadow_gdn,
            committed_attention,
            shadow_attention,
            shadow_attention_valid: 0,
            decode_scratch: decoder.allocate_decode_scratch(ctx)?,
            verify_scratch: decoder.allocate_verify_scratch(ctx)?,
            batched_verify: std::env::var("APXINF_QWEN35_MTP_BATCH_VERIFY")
                .is_ok_and(|value| value == "1"),
            verify_token_ids,
            verify_predictions,
            capacity,
            position: 0,
            epoch: 0,
            shadow_generation: 0,
        };
        session.reset_all_storage(ctx)?;
        ctx.synchronize().map_err(Error::Cuda)?;
        Ok(session)
    }

    pub fn position(&self) -> usize {
        self.position
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn epoch(&self) -> u64 {
        self.epoch
    }

    pub fn attention_backend(&self) -> &B {
        &self.attention
    }

    /// Allocate one independent prefix checkpoint with the same geometry as
    /// this session. Allocation happens once when context reuse is enabled.
    pub fn allocate_snapshot(
        &mut self,
        ctx: &CudaContext,
        decoder: &Qwen35Decoder,
        weights: &Qwen35DeviceWeights,
        capacity: usize,
    ) -> Result<Qwen35SessionSnapshot<B::Cache>> {
        if capacity == 0 || capacity > self.capacity {
            return Err(invalid("Qwen3.5 target snapshot capacity is invalid"));
        }
        let architecture = decoder.architecture();
        let mut gdn = Vec::with_capacity(architecture.gdn_layers().count());
        let mut attention = Vec::with_capacity(architecture.full_attention_layers().count());
        for (layer, resident) in architecture.layers.iter().zip(&weights.layers) {
            match (&layer.mixer, &resident.mixer) {
                (MixerKind::Gdn(_), DeviceMixerWeights::Gdn(_)) => {
                    gdn.push(GdnState::new(ctx)?);
                }
                (
                    MixerKind::FullAttention(geometry),
                    DeviceMixerWeights::FullAttention(full_attention),
                ) => {
                    let scales = full_attention.kv_cache_scales.ok_or_else(|| {
                        invalid(format!(
                            "Qwen3.5 base attention layer {} is missing FP8 KV scales",
                            layer.index
                        ))
                    })?;
                    attention.push(self.attention.allocate_cache(
                        ctx,
                        layer.index,
                        *geometry,
                        capacity,
                        scales,
                    )?);
                }
                _ => {
                    return Err(invalid(format!(
                        "Qwen3.5 layer {} mixer changed after validation",
                        layer.index
                    )));
                }
            }
        }
        Ok(Qwen35SessionSnapshot {
            gdn,
            attention,
            capacity,
            position: 0,
        })
    }

    /// Copy the currently committed target prefix into an independent
    /// checkpoint. The copy is ordered on `ctx`; callers synchronize before
    /// publishing the checkpoint's token identity.
    pub fn capture_snapshot(
        &mut self,
        ctx: &CudaContext,
        snapshot: &mut Qwen35SessionSnapshot<B::Cache>,
    ) -> Result<()> {
        if self.position > snapshot.capacity
            || snapshot.gdn.len() != self.committed_gdn.len()
            || snapshot.attention.len() != self.committed_attention.len()
        {
            return Err(invalid("Qwen3.5 target snapshot geometry mismatch"));
        }
        for (destination, source) in snapshot.gdn.iter().zip(&self.committed_gdn) {
            destination.copy_from(ctx, source)?;
        }
        for (source, destination) in self.committed_attention.iter().zip(&mut snapshot.attention) {
            self.attention
                .copy_cache_range(ctx, source, destination, 0..self.position)?;
        }
        ctx.synchronize().map_err(Error::Cuda)?;
        snapshot.position = self.position;
        Ok(())
    }

    /// Restore a committed target prefix from an independent checkpoint.
    /// Stale rows beyond the checkpoint position remain unreachable and are
    /// overwritten if the sequence later grows into them.
    pub fn restore_snapshot(
        &mut self,
        ctx: &CudaContext,
        snapshot: &Qwen35SessionSnapshot<B::Cache>,
    ) -> Result<()> {
        if snapshot.position == 0
            || snapshot.position > self.capacity
            || snapshot.position > snapshot.capacity
            || snapshot.gdn.len() != self.committed_gdn.len()
            || snapshot.attention.len() != self.committed_attention.len()
        {
            return Err(invalid("Qwen3.5 target snapshot is empty or incompatible"));
        }
        for (destination, source) in self.committed_gdn.iter().zip(&snapshot.gdn) {
            destination.copy_from(ctx, source)?;
        }
        for (source, destination) in snapshot.attention.iter().zip(&mut self.committed_attention) {
            self.attention
                .copy_cache_range(ctx, source, destination, 0..snapshot.position)?;
        }
        ctx.synchronize().map_err(Error::Cuda)?;
        self.position = snapshot.position;
        self.epoch = self
            .epoch
            .checked_add(1)
            .ok_or_else(|| invalid("Qwen3.5 session epoch overflow"))?;
        // The inactive bank contains unrelated history. The next append will
        // copy the complete restored prefix before executing new rows.
        self.shadow_attention_valid = 0;
        Ok(())
    }

    /// Replace the current sequence with `token_ids`. Existing committed state
    /// remains visible if any allocation, kernel, backend call or synchronize
    /// fails.
    pub fn prefill(
        &mut self,
        ctx: &CudaContext,
        decoder: &Qwen35Decoder,
        weights: &Qwen35DeviceWeights,
        token_ids: &[u32],
    ) -> Result<DecoderOutput> {
        self.validate_capacity(0, token_ids.len())?;
        self.bump_shadow_generation()?;
        self.prepare_empty_shadow(ctx)?;
        let output = self.run_shadow(
            ctx,
            decoder,
            weights,
            token_ids,
            0,
            DecoderPhase::Prefill,
            LogitsSelection::Last,
            HiddenSelection::Last,
        )?;
        self.publish_shadow(output.end_position, 0)?;
        Ok(output)
    }

    /// Append a multi-token prefill chunk to the committed sequence. Unlike
    /// `forward(start>0)`, this retains the batched prefill kernels instead of
    /// degrading the chunk into a scalar decode loop.
    pub fn append_prefill(
        &mut self,
        ctx: &CudaContext,
        decoder: &Qwen35Decoder,
        weights: &Qwen35DeviceWeights,
        token_ids: &[u32],
    ) -> Result<DecoderOutput> {
        if self.position == 0 {
            return Err(invalid(
                "Qwen3.5 append prefill requires a committed prefix",
            ));
        }
        let start_position = self.position;
        self.validate_capacity(start_position, token_ids.len())?;
        self.bump_shadow_generation()?;
        self.prepare_shadow_from_committed(ctx)?;
        let output = self.run_shadow(
            ctx,
            decoder,
            weights,
            token_ids,
            start_position,
            DecoderPhase::Prefill,
            LogitsSelection::Last,
            HiddenSelection::Last,
        )?;
        self.publish_shadow(output.end_position, start_position)?;
        Ok(output)
    }

    /// Base `LlmTrait::forward(start=0)` compatibility path. Unlike serving
    /// prefill, this materializes one logit and hidden-state row per prompt
    /// token while keeping the same transactional state publication
    /// semantics. The decoder scratch contract requires hidden coverage for
    /// every selected logit row; callers may discard the extra hidden rows.
    pub fn prefill_all_logits(
        &mut self,
        ctx: &CudaContext,
        decoder: &Qwen35Decoder,
        weights: &Qwen35DeviceWeights,
        token_ids: &[u32],
    ) -> Result<DecoderOutput> {
        self.validate_capacity(0, token_ids.len())?;
        self.bump_shadow_generation()?;
        self.prepare_empty_shadow(ctx)?;
        let output = self.run_shadow(
            ctx,
            decoder,
            weights,
            token_ids,
            0,
            DecoderPhase::Prefill,
            LogitsSelection::All,
            HiddenSelection::All,
        )?;
        self.publish_shadow(output.end_position, 0)?;
        Ok(output)
    }

    /// Prepare the target half of an MTP-enabled prompt transaction. It
    /// materializes every normalized target hidden row (needed to prefill the
    /// draft layer's own KV cache) while projecting only the final logit row.
    /// No target state or position becomes visible until
    /// [`Self::publish_target_prefill`] is called.
    pub fn prepare_target_prefill(
        &mut self,
        ctx: &CudaContext,
        decoder: &Qwen35Decoder,
        weights: &Qwen35DeviceWeights,
        token_ids: &[u32],
    ) -> Result<PreparedTargetPrefill> {
        self.validate_capacity(0, token_ids.len())?;
        self.bump_shadow_generation()?;
        self.prepare_empty_shadow(ctx)?;
        let output = self.run_shadow(
            ctx,
            decoder,
            weights,
            token_ids,
            0,
            DecoderPhase::Prefill,
            LogitsSelection::Last,
            HiddenSelection::All,
        )?;
        self.epoch
            .checked_add(1)
            .ok_or_else(|| invalid("Qwen3.5 session epoch overflow"))?;
        Ok(PreparedTargetPrefill {
            epoch: self.epoch,
            shadow_generation: self.shadow_generation,
            start_position: 0,
            end_position: output.end_position,
            output,
        })
    }

    /// Prepare a target chunk on top of the committed prefix while retaining
    /// all new hidden rows needed to extend the MTP teacher-forcing cache.
    pub fn prepare_target_append_prefill(
        &mut self,
        ctx: &CudaContext,
        decoder: &Qwen35Decoder,
        weights: &Qwen35DeviceWeights,
        token_ids: &[u32],
    ) -> Result<PreparedTargetPrefill> {
        if self.position == 0 {
            return Err(invalid(
                "Qwen3.5 target append prefill requires a committed prefix",
            ));
        }
        let start_position = self.position;
        self.validate_capacity(start_position, token_ids.len())?;
        self.bump_shadow_generation()?;
        self.prepare_shadow_from_committed(ctx)?;
        let output = self.run_shadow(
            ctx,
            decoder,
            weights,
            token_ids,
            start_position,
            DecoderPhase::Prefill,
            LogitsSelection::Last,
            HiddenSelection::All,
        )?;
        self.epoch
            .checked_add(1)
            .ok_or_else(|| invalid("Qwen3.5 session epoch overflow"))?;
        Ok(PreparedTargetPrefill {
            epoch: self.epoch,
            shadow_generation: self.shadow_generation,
            start_position,
            end_position: output.end_position,
            output,
        })
    }

    /// Publish a prepared target prompt after MTP teacher-forcing prefill has
    /// also prepared successfully. Only state-vector swaps and scalar writes
    /// occur here.
    pub fn publish_target_prefill(
        &mut self,
        prepared: PreparedTargetPrefill,
    ) -> Result<DecoderOutput> {
        if prepared.epoch != self.epoch
            || prepared.shadow_generation != self.shadow_generation
            || prepared.end_position <= prepared.start_position
            || prepared.output.start_position != prepared.start_position
            || prepared.output.end_position != prepared.end_position
            || prepared.output.hidden_rows != prepared.end_position - prepared.start_position
        {
            return Err(invalid("stale Qwen3.5 prepared target prefill"));
        }
        self.publish_shadow(prepared.end_position, prepared.start_position)?;
        Ok(prepared.output)
    }

    /// Append exactly one autoregressive input token.
    pub fn decode(
        &mut self,
        ctx: &CudaContext,
        decoder: &Qwen35Decoder,
        weights: &Qwen35DeviceWeights,
        token_id: u32,
    ) -> Result<DecoderOutput> {
        if self.position == 0 {
            return Err(invalid("Qwen3.5 decode requires a committed prompt"));
        }
        let start_position = self.position;
        self.validate_capacity(start_position, 1)?;
        self.bump_shadow_generation()?;
        self.prepare_shadow_from_committed(ctx)?;
        let output = self.run_shadow(
            ctx,
            decoder,
            weights,
            &[token_id],
            start_position,
            DecoderPhase::Decode,
            LogitsSelection::Last,
            HiddenSelection::Last,
        )?;
        self.publish_shadow(output.end_position, start_position)?;
        Ok(output)
    }

    /// Evaluate a target-verification chunk without changing committed state
    /// or position.
    ///
    /// This is the exact base-model half required by `mtp_schedule`: an MTP
    /// executor can draft three tokens, call this method with
    /// `[pending,d0,d1,d2]`, and derive four target predictions. It is
    /// The default is the validated scalar path. Setting
    /// `APXINF_QWEN35_MTP_BATCH_VERIFY=1` at session construction selects the
    /// persistent four-row prefill candidate for parity/profiling. Pass the
    /// resulting ticket to [`Self::prepare_target_commit`] with the schedule's
    /// committed row count; partial acceptance always replays the accepted
    /// prefix through the scalar path.
    pub fn verify_target_chunk(
        &mut self,
        ctx: &CudaContext,
        decoder: &Qwen35Decoder,
        weights: &Qwen35DeviceWeights,
        token_ids: &[u32],
    ) -> Result<TargetVerification> {
        if self.position == 0 {
            return Err(invalid(
                "Qwen3.5 target verification requires a committed prompt",
            ));
        }
        if token_ids.len() != QWEN35_MTP_VERIFY_ROWS {
            return Err(invalid(format!(
                "Qwen3.5 MTP verification requires {QWEN35_MTP_VERIFY_ROWS} rows, got {}",
                token_ids.len()
            )));
        }
        let start_position = self.position;
        self.validate_capacity(start_position, token_ids.len())?;
        self.bump_shadow_generation()?;
        self.prepare_shadow_from_committed(ctx)?;
        self.stage_verify_tokens(token_ids)?;
        let final_output = if self.batched_verify {
            let output = decoder.run_verify_reusing_from_block(
                ctx,
                weights,
                &mut self.attention,
                &mut self.shadow_attention,
                &self.shadow_gdn,
                token_ids,
                &self.verify_token_ids,
                start_position,
                &mut self.verify_scratch,
            )?;
            output.enqueue_greedy_token_ids(ctx, &self.verify_predictions, 0)?;
            output.last_row()?
        } else {
            let mut final_output = None;
            for (offset, (&token, (position, phase))) in token_ids
                .iter()
                .zip(exact_decode_plan(start_position, token_ids.len())?)
                .enumerate()
            {
                debug_assert_eq!(phase, DecoderPhase::Decode);
                let output = self.run_decode_shadow_from_block_enqueued(
                    ctx, decoder, weights, token, offset, position,
                )?;
                // Argmax is enqueued before the next scalar row overwrites the
                // persistent logits buffer. All work shares ctx.stream, so one
                // round-level synchronization preserves exact kernel ordering.
                output.enqueue_greedy_token_ids(ctx, &self.verify_predictions, offset)?;
                final_output = Some(output);
            }
            final_output.ok_or_else(|| invalid("Qwen3.5 target verification produced no rows"))?
        };
        ctx.synchronize().map_err(Error::Cuda)?;
        let host_predictions = transfers::to_cpu(&self.verify_predictions)?;
        let predictions = host_predictions
            .as_i32()?
            .iter()
            .take(token_ids.len())
            .map(|&token| {
                u32::try_from(token).map_err(|_| {
                    invalid(format!(
                        "Qwen3.5 target argmax returned negative ID {token}"
                    ))
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(TargetVerification {
            epoch: self.epoch,
            shadow_generation: self.shadow_generation,
            start_position,
            inputs: token_ids.to_vec(),
            predictions,
            final_output,
        })
    }

    /// Complete all fallible target-side work for a speculative commit.
    /// Partial acceptance rebuilds the shadow from committed state and replays
    /// exactly `committed_rows`; full acceptance keeps the already verified
    /// shadow. No committed state is changed here.
    pub fn prepare_target_commit(
        &mut self,
        ctx: &CudaContext,
        decoder: &Qwen35Decoder,
        weights: &Qwen35DeviceWeights,
        verification: TargetVerification,
        committed_rows: usize,
    ) -> Result<PreparedTargetCommit> {
        self.validate_verification(&verification)?;
        if committed_rows == 0 || committed_rows > verification.inputs.len() {
            return Err(invalid(format!(
                "Qwen3.5 target commit rows must be in 1..={}, got {committed_rows}",
                verification.inputs.len()
            )));
        }
        let output = if committed_rows < verification.inputs.len() {
            self.prepare_shadow_from_committed(ctx)?;
            self.run_exact_decode_rows(
                ctx,
                decoder,
                weights,
                &verification.inputs[..committed_rows],
                verification.start_position,
            )?
        } else {
            verification.final_output
        };
        let end_position = verification
            .start_position
            .checked_add(committed_rows)
            .ok_or_else(|| invalid("Qwen3.5 target commit position overflow"))?;
        self.epoch
            .checked_add(1)
            .ok_or_else(|| invalid("Qwen3.5 session epoch overflow"))?;
        Ok(PreparedTargetCommit {
            epoch: verification.epoch,
            shadow_generation: verification.shadow_generation,
            end_position,
            output,
        })
    }

    /// Publish a target shadow after the coordinator has also prepared the MTP
    /// side. This performs only vector swaps and scalar assignments.
    pub fn publish_target_commit(
        &mut self,
        prepared: PreparedTargetCommit,
    ) -> Result<DecoderOutput> {
        let output_end = prepared.output.start_position.checked_add(1);
        if prepared.epoch != self.epoch
            || prepared.shadow_generation != self.shadow_generation
            || prepared.end_position <= self.position
            || prepared.output.rows != 1
            || prepared.output.hidden_rows != 1
            || prepared.output.end_position != prepared.end_position
            || output_end != Some(prepared.end_position)
        {
            return Err(invalid("stale Qwen3.5 prepared target commit"));
        }
        let preserved_prefix = self.position;
        self.publish_shadow(prepared.end_position, preserved_prefix)?;
        Ok(prepared.output)
    }

    /// Clear the sequence. Reset is also transactional with respect to the
    /// visible position: only shadow storage is cleared, and it becomes
    /// committed after every reset and the stream synchronize succeeds.
    pub fn reset(&mut self, ctx: &CudaContext) -> Result<()> {
        self.bump_shadow_generation()?;
        self.prepare_empty_shadow(ctx)?;
        ctx.synchronize().map_err(Error::Cuda)?;
        self.publish_shadow(0, 0)
    }

    #[allow(clippy::too_many_arguments)]
    fn run_shadow(
        &mut self,
        ctx: &CudaContext,
        decoder: &Qwen35Decoder,
        weights: &Qwen35DeviceWeights,
        token_ids: &[u32],
        start_position: usize,
        phase: DecoderPhase,
        logits_selection: LogitsSelection,
        hidden_selection: HiddenSelection,
    ) -> Result<DecoderOutput> {
        let output = self.run_shadow_enqueued(
            ctx,
            decoder,
            weights,
            token_ids,
            start_position,
            phase,
            logits_selection,
            hidden_selection,
        )?;
        ctx.synchronize().map_err(Error::Cuda)?;
        Ok(output)
    }

    #[allow(clippy::too_many_arguments)]
    fn run_shadow_enqueued(
        &mut self,
        ctx: &CudaContext,
        decoder: &Qwen35Decoder,
        weights: &Qwen35DeviceWeights,
        token_ids: &[u32],
        start_position: usize,
        phase: DecoderPhase,
        logits_selection: LogitsSelection,
        hidden_selection: HiddenSelection,
    ) -> Result<DecoderOutput> {
        let output = match phase {
            DecoderPhase::Decode => {
                if logits_selection != LogitsSelection::Last
                    || hidden_selection != HiddenSelection::Last
                {
                    return Err(invalid(
                        "Qwen3.5 reusable decode supports only last-row logits and hidden state",
                    ));
                }
                decoder.run_decode_reusing(
                    ctx,
                    weights,
                    &mut self.attention,
                    &mut self.shadow_attention,
                    &self.shadow_gdn,
                    token_ids,
                    start_position,
                    &mut self.decode_scratch,
                )?
            }
            DecoderPhase::Prefill => decoder.run(
                ctx,
                weights,
                &mut self.attention,
                &mut self.shadow_attention,
                &self.shadow_gdn,
                token_ids,
                start_position,
                phase,
                logits_selection,
                hidden_selection,
            )?,
        };
        Ok(output)
    }

    fn run_decode_shadow_from_block_enqueued(
        &mut self,
        ctx: &CudaContext,
        decoder: &Qwen35Decoder,
        weights: &Qwen35DeviceWeights,
        token_id: TokenId,
        token_offset: usize,
        start_position: usize,
    ) -> Result<DecoderOutput> {
        decoder.run_decode_reusing_from_block(
            ctx,
            weights,
            &mut self.attention,
            &mut self.shadow_attention,
            &self.shadow_gdn,
            token_id,
            &self.verify_token_ids,
            token_offset,
            start_position,
            &mut self.decode_scratch,
        )
    }

    /// Canonical target verification/replay path. Keeping this as an explicit
    /// scalar loop prevents a future optimization from substituting batched
    /// prefill and changing reductions or attention-provider selection.
    fn run_exact_decode_rows(
        &mut self,
        ctx: &CudaContext,
        decoder: &Qwen35Decoder,
        weights: &Qwen35DeviceWeights,
        token_ids: &[TokenId],
        start_position: usize,
    ) -> Result<DecoderOutput> {
        self.stage_verify_tokens(token_ids)?;
        let mut output = None;
        for (offset, (&token, (position, phase))) in token_ids
            .iter()
            .zip(exact_decode_plan(start_position, token_ids.len())?)
            .enumerate()
        {
            debug_assert_eq!(phase, DecoderPhase::Decode);
            output = Some(self.run_decode_shadow_from_block_enqueued(
                ctx, decoder, weights, token, offset, position,
            )?);
        }
        ctx.synchronize().map_err(Error::Cuda)?;
        output.ok_or_else(|| invalid("Qwen3.5 exact decode produced no rows"))
    }

    fn stage_verify_tokens(&self, token_ids: &[TokenId]) -> Result<()> {
        let staged = staged_verify_values(token_ids)?;
        let host = Tensor::from_i32(vec![QWEN35_MTP_VERIFY_ROWS], &staged)?;
        transfers::copy_cpu_to_cuda(&host, &self.verify_token_ids)
    }

    fn prepare_empty_shadow(&mut self, ctx: &CudaContext) -> Result<()> {
        for state in &self.shadow_gdn {
            state.reset(ctx)?;
        }
        for cache in &mut self.shadow_attention {
            self.attention.reset_cache(ctx, cache)?;
        }
        self.shadow_attention_valid = 0;
        Ok(())
    }

    fn prepare_shadow_from_committed(&mut self, ctx: &CudaContext) -> Result<()> {
        for (committed, shadow) in self.committed_gdn.iter().zip(&self.shadow_gdn) {
            shadow.copy_from(ctx, committed)?;
        }
        if self.shadow_attention_valid > self.position {
            return Err(invalid(
                "Qwen3.5 shadow KV prefix is ahead of committed position",
            ));
        }
        if self.shadow_attention_valid < self.position {
            let positions = self.shadow_attention_valid..self.position;
            for (committed, shadow) in self
                .committed_attention
                .iter()
                .zip(&mut self.shadow_attention)
            {
                self.attention
                    .copy_cache_range(ctx, committed, shadow, positions.clone())?;
            }
            self.shadow_attention_valid = self.position;
        }
        Ok(())
    }

    fn publish_shadow(&mut self, position: usize, preserved_prefix: usize) -> Result<()> {
        let next_epoch = self
            .epoch
            .checked_add(1)
            .ok_or_else(|| invalid("Qwen3.5 session epoch overflow"))?;
        std::mem::swap(&mut self.committed_gdn, &mut self.shadow_gdn);
        std::mem::swap(&mut self.committed_attention, &mut self.shadow_attention);
        self.position = position;
        self.epoch = next_epoch;
        self.shadow_attention_valid = preserved_prefix;
        Ok(())
    }

    fn reset_all_storage(&mut self, ctx: &CudaContext) -> Result<()> {
        for state in &self.committed_gdn {
            state.reset(ctx)?;
        }
        for state in &self.shadow_gdn {
            state.reset(ctx)?;
        }
        for cache in &mut self.committed_attention {
            self.attention.reset_cache(ctx, cache)?;
        }
        for cache in &mut self.shadow_attention {
            self.attention.reset_cache(ctx, cache)?;
        }
        Ok(())
    }

    fn bump_shadow_generation(&mut self) -> Result<()> {
        self.shadow_generation = self
            .shadow_generation
            .checked_add(1)
            .ok_or_else(|| invalid("Qwen3.5 shadow generation overflow"))?;
        Ok(())
    }

    fn validate_verification(&self, verification: &TargetVerification) -> Result<()> {
        if verification.epoch != self.epoch
            || verification.shadow_generation != self.shadow_generation
            || verification.start_position != self.position
            || verification.predictions.len() != verification.inputs.len()
            || verification.final_output.rows != 1
            || verification.final_output.hidden_rows != 1
            || verification.final_output.start_position
                != self
                    .position
                    .checked_add(verification.inputs.len().saturating_sub(1))
                    .unwrap_or(usize::MAX)
            || verification.final_output.end_position
                != self
                    .position
                    .checked_add(verification.inputs.len())
                    .unwrap_or(usize::MAX)
        {
            return Err(invalid("stale or malformed Qwen3.5 target verification"));
        }
        Ok(())
    }

    fn validate_capacity(&self, start: usize, rows: usize) -> Result<()> {
        let end = start
            .checked_add(rows)
            .ok_or_else(|| invalid("Qwen3.5 session position overflow"))?;
        if rows == 0 || end > self.capacity {
            return Err(invalid(format!(
                "Qwen3.5 session range [{start}, {end}) exceeds capacity {}",
                self.capacity
            )));
        }
        Ok(())
    }
}

fn invalid(message: impl Into<String>) -> Error {
    Error::Other(message.into())
}

fn exact_decode_plan(start: usize, rows: usize) -> Result<Vec<(usize, DecoderPhase)>> {
    (0..rows)
        .map(|offset| {
            start
                .checked_add(offset)
                .map(|position| (position, DecoderPhase::Decode))
                .ok_or_else(|| invalid("Qwen3.5 exact-verify position overflow"))
        })
        .collect()
}

fn staged_verify_values(token_ids: &[TokenId]) -> Result<[i32; QWEN35_MTP_VERIFY_ROWS]> {
    if token_ids.is_empty() || token_ids.len() > QWEN35_MTP_VERIFY_ROWS {
        return Err(invalid(format!(
            "Qwen3.5 verification staging requires 1..={QWEN35_MTP_VERIFY_ROWS} tokens"
        )));
    }
    let mut staged = [0i32; QWEN35_MTP_VERIFY_ROWS];
    for (destination, &token) in staged.iter_mut().zip(token_ids) {
        *destination = i32::try_from(token)
            .map_err(|_| invalid(format!("Qwen3.5 token ID {token} exceeds i32")))?;
    }
    Ok(staged)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mtp_verify_positions_are_scalar_and_contiguous() {
        assert_eq!(
            exact_decode_plan(4_096, QWEN35_MTP_VERIFY_ROWS).unwrap(),
            vec![
                (4_096, DecoderPhase::Decode),
                (4_097, DecoderPhase::Decode),
                (4_098, DecoderPhase::Decode),
                (4_099, DecoderPhase::Decode),
            ]
        );
    }

    #[test]
    fn exact_decode_plan_rejects_overflow() {
        assert!(exact_decode_plan(usize::MAX, 2).is_err());
    }

    #[test]
    fn verify_tokens_are_staged_once_with_zeroed_unused_tail() {
        assert_eq!(staged_verify_values(&[11, 22]).unwrap(), [11, 22, 0, 0]);
        assert!(staged_verify_values(&[]).is_err());
        assert!(staged_verify_values(&[1, 2, 3, 4, 5]).is_err());
    }
}
