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
    LogitsSelection, Qwen35DecodePhase, Qwen35Decoder,
};
use super::device_weights::{DeviceMixerWeights, Qwen35DeviceWeights};
use super::forward::{GdnState, GdnStateCheckpoints};
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
    /// State after each row of the fixed four-token target verification.
    verify_gdn_checkpoints: Vec<GdnStateCheckpoints>,
    committed_attention: Vec<B::Cache>,
    shadow_attention: Vec<B::Cache>,
    /// Prefix of the inactive cache bank known to equal the committed bank.
    shadow_attention_valid: usize,
    /// Skip the per-step transactional shadow copy for non-speculative decode.
    /// Set by the coordinator: off while a speculative round can be rejected,
    /// on for plain scalar decode. See `decode`.
    decode_in_place: bool,
    /// Persistent one-row workspace for ordinary decode. Variable-size prefill
    /// deliberately keeps using call-sized temporary scratch.
    decode_scratch: DecoderScratch,
    /// Persistent four-row workspace for the batched target verifier.
    verify_scratch: DecoderScratch,
    /// Graph sets follow their physical GDN state banks when transactions
    /// publish by swapping committed and shadow storage.
    committed_decode_phases: Option<Vec<Qwen35DecodePhase>>,
    shadow_decode_phases: Option<Vec<Qwen35DecodePhase>>,
    /// Device argmax outputs for the four-row verifier.
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
        let mut verify_gdn_checkpoints = Vec::with_capacity(architecture.gdn_layers().count());
        let mut committed_attention =
            Vec::with_capacity(architecture.full_attention_layers().count());
        let mut shadow_attention = Vec::with_capacity(architecture.full_attention_layers().count());

        for (layer, resident) in architecture.layers.iter().zip(&weights.layers) {
            match (&layer.mixer, &resident.mixer) {
                (MixerKind::Gdn(_), DeviceMixerWeights::Gdn(_)) => {
                    committed_gdn.push(GdnState::new(ctx)?);
                    shadow_gdn.push(GdnState::new(ctx)?);
                    verify_gdn_checkpoints
                        .push(GdnStateCheckpoints::new(ctx, QWEN35_MTP_VERIFY_ROWS)?);
                }
                (
                    MixerKind::FullAttention(geometry),
                    DeviceMixerWeights::FullAttention(full_attention),
                ) => {
                    let scales = full_attention.kv_cache_scales;
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
        let verify_predictions = transfers::to_cuda(&verify_zeros, ctx.device_id())?;
        let mut session = Self {
            attention,
            committed_gdn,
            shadow_gdn,
            verify_gdn_checkpoints,
            committed_attention,
            shadow_attention,
            shadow_attention_valid: 0,
            decode_in_place: false,
            decode_scratch: decoder.allocate_decode_scratch(ctx)?,
            verify_scratch: decoder.allocate_verify_scratch(ctx)?,
            committed_decode_phases: None,
            shadow_decode_phases: None,
            verify_predictions,
            capacity,
            position: 0,
            epoch: 0,
            shadow_generation: 0,
        };
        if std::env::var("APXINF_QWEN35_CUDA_GRAPHS").as_deref() == Ok("1") {
            session.committed_decode_phases = Some(decoder.capture_decode_phases(
                ctx,
                weights,
                &session.committed_gdn,
                &mut session.decode_scratch,
            )?);
            session.shadow_decode_phases = Some(decoder.capture_decode_phases(
                ctx,
                weights,
                &session.shadow_gdn,
                &mut session.decode_scratch,
            )?);
            if std::env::var("APXINF_QWEN35_VERIFY_GRAPHS").as_deref() == Ok("1") {
                decoder.verify_decode_phases(
                    ctx,
                    weights,
                    &session.committed_gdn,
                    session
                        .committed_decode_phases
                        .as_deref()
                        .expect("committed graphs were captured"),
                    &session.shadow_gdn,
                    session
                        .shadow_decode_phases
                        .as_deref()
                        .expect("shadow graphs were captured"),
                    &mut session.decode_scratch,
                )?;
            }
        }
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
                    let scales = full_attention.kv_cache_scales;
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
        // APXINF_QWEN35_DECODE_TIMING=1 splits a decode step into the
        // transactional shadow copy and the forward itself. Each half is
        // bracketed by a synchronize, so the figures are attributable but the
        // total is inflated relative to real throughput.
        let timing = std::env::var_os("APXINF_QWEN35_DECODE_TIMING").is_some();
        if timing {
            ctx.synchronize().map_err(Error::Cuda)?;
        }
        let at = std::time::Instant::now();
        // Scalar decode commits unconditionally: there is no rejected-token
        // path, so the shadow bank it runs into is published every time. The
        // copy that seeds it therefore duplicates all thirty GDN recurrent
        // states on every token to produce a bank that is immediately made
        // committed. Measured at 0.844 ms of a 19.42 ms step, which is 4.3%
        // spent reproducing state the step already owns.
        //
        // `decode_in_place` instead runs on the committed banks and only
        // advances the position. The success path is identical. What it gives
        // up is the rollback guarantee: a failure mid-forward leaves the
        // committed GDN state partially advanced, so the session is finished
        // rather than merely erroring. The caller records that through
        // `reset_error`, which already exists for exactly this. Speculation
        // keeps the transactional path, because there a verified round can be
        // rejected and the pre-round state must survive.
        if self.decode_in_place {
            let output = self.run_committed_decode(ctx, decoder, weights, token_id, start_position)?;
            if timing {
                ctx.synchronize().map_err(Error::Cuda)?;
                eprintln!("[decode] shadow 0.000 ms (in place)");
                eprintln!("[decode] forward {:.3} ms", at.elapsed().as_secs_f64() * 1e3);
            }
            self.position = output.end_position;
            self.shadow_attention_valid = self.shadow_attention_valid.min(self.position);
            return Ok(output);
        }
        self.prepare_shadow_from_committed(ctx)?;
        if timing {
            ctx.synchronize().map_err(Error::Cuda)?;
            eprintln!(
                "[decode] shadow {:.3} ms",
                at.elapsed().as_secs_f64() * 1e3
            );
        }
        let at = std::time::Instant::now();
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
        if timing {
            ctx.synchronize().map_err(Error::Cuda)?;
            eprintln!("[decode] forward {:.3} ms", at.elapsed().as_secs_f64() * 1e3);
        }
        self.publish_shadow(output.end_position, start_position)?;
        Ok(output)
    }

    /// Evaluate all four target-verification rows in one fixed-shape forward
    /// without changing committed state or position.
    ///
    /// This is the exact base-model half required by `mtp_schedule`: an MTP
    /// executor can draft three tokens, call this method with
    /// `[pending,d0,d1,d2]`, and derive four target predictions. It is
    /// Each GDN layer records the state after every row, allowing partial
    /// acceptance to select the correct recurrent boundary without replaying
    /// the target model.
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
        let final_output = decoder.run_verify_reusing(
            ctx,
            weights,
            &mut self.attention,
            &mut self.shadow_attention,
            &self.shadow_gdn,
            &self.verify_gdn_checkpoints,
            token_ids,
            start_position,
            &mut self.verify_scratch,
        )?;
        final_output.enqueue_greedy_token_ids(ctx, &self.verify_predictions, 0)?;
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
    /// Partial acceptance selects the saved GDN state after `committed_rows`
    /// and the matching hidden row. No target-model replay or committed-state
    /// mutation occurs here.
    pub fn prepare_target_commit(
        &mut self,
        ctx: &CudaContext,
        _decoder: &Qwen35Decoder,
        _weights: &Qwen35DeviceWeights,
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
        if committed_rows < verification.inputs.len() {
            let row = committed_rows - 1;
            for (checkpoint, shadow) in self.verify_gdn_checkpoints.iter().zip(&self.shadow_gdn) {
                checkpoint.copy_row_to(ctx, row, shadow)?;
            }
        }
        let output = verification.final_output.into_row(committed_rows - 1)?;
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

    /// Choose between in-place and transactional decode. Speculation must keep
    /// the transactional path so a rejected round can be rolled back.
    pub fn set_decode_in_place(&mut self, enabled: bool) {
        self.decode_in_place = enabled;
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

    /// Run one decode row directly on the committed banks. See `decode`.
    fn run_committed_decode(
        &mut self,
        ctx: &CudaContext,
        decoder: &Qwen35Decoder,
        weights: &Qwen35DeviceWeights,
        token_id: u32,
        start_position: usize,
    ) -> Result<DecoderOutput> {
        let output = decoder.run_decode_reusing(
            ctx,
            weights,
            &mut self.attention,
            &mut self.committed_attention,
            &self.committed_gdn,
            &[token_id],
            start_position,
            &mut self.decode_scratch,
            self.committed_decode_phases.as_deref(),
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
                    self.shadow_decode_phases.as_deref(),
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
        std::mem::swap(
            &mut self.committed_decode_phases,
            &mut self.shadow_decode_phases,
        );
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
            // The verification forward runs every row with
            // LogitsSelection::All / HiddenSelection::All, so it carries one
            // logit and hidden row per verified token. It is
            // `prepare_target_commit` that narrows it to the accepted row via
            // `into_row`, and `publish_target_commit` that then requires
            // rows == 1. Demanding a single row here described the old
            // pre-`into_row` shape and rejected every well-formed four-row
            // verification, which made --mode mtp unusable.
            || verification.final_output.rows != verification.inputs.len()
            || verification.final_output.hidden_rows != verification.inputs.len()
            || verification.final_output.start_position != self.position
            || verification.final_output.end_position
                != self
                    .position
                    .checked_add(verification.inputs.len())
                    .unwrap_or(usize::MAX)
        {
            // One opaque message for eight different invariants made this
            // impossible to act on, so name the one that actually tripped.
            let expected_end = self
                .position
                .checked_add(verification.inputs.len())
                .unwrap_or(usize::MAX);
            let which = if verification.epoch != self.epoch {
                format!("epoch {} != session {}", verification.epoch, self.epoch)
            } else if verification.shadow_generation != self.shadow_generation {
                format!(
                    "shadow_generation {} != session {}",
                    verification.shadow_generation, self.shadow_generation
                )
            } else if verification.start_position != self.position {
                format!(
                    "start_position {} != session position {}",
                    verification.start_position, self.position
                )
            } else if verification.predictions.len() != verification.inputs.len() {
                format!(
                    "predictions {} != inputs {}",
                    verification.predictions.len(),
                    verification.inputs.len()
                )
            } else if verification.final_output.rows != verification.inputs.len() {
                format!(
                    "final_output.rows {} != inputs {}",
                    verification.final_output.rows,
                    verification.inputs.len()
                )
            } else if verification.final_output.hidden_rows != verification.inputs.len() {
                format!(
                    "final_output.hidden_rows {} != inputs {}",
                    verification.final_output.hidden_rows,
                    verification.inputs.len()
                )
            } else if verification.final_output.start_position != self.position {
                format!(
                    "final_output.start_position {} != session position {}",
                    verification.final_output.start_position, self.position
                )
            } else {
                format!(
                    "final_output.end_position {} != expected {expected_end}",
                    verification.final_output.end_position
                )
            };
            return Err(invalid(format!(
                "stale or malformed Qwen3.5 target verification: {which}"
            )));
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
