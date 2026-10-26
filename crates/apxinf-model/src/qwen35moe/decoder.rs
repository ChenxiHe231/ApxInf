//! Qwen3.5 target-decoder dataflow.
//!
//! This module owns the exact residual/norm/layer order and calls the real GDN
//! and routed-NVFP4 MoE implementations.  Full attention is deliberately an
//! explicit backend boundary: the target checkpoint requires partial mRoPE,
//! gated queries and an FP8 KV cache, and silently falling back to a different
//! attention/cache semantic would invalidate both accuracy and MTP rollback.

use apxinf_core::{DType, Device, Error, Result, Shape, Tensor};
use apxinf_cuda_new::{ops, transfers, CudaBuffer, CudaContext};
use std::ops::Range;

use super::architecture::{AttentionGeometry, MixerKind, Qwen35MoeArchitecture};
use super::config::Qwen35MoeConfig;
use super::device_weights::{DeviceAttentionWeights, DeviceMixerWeights, Qwen35DeviceWeights};
use super::forward::{BaseMoeScratch, GdnScratch, GdnState};
use super::runtime_weights::KvCacheScales;

/// Whether a decoder call processes a prompt/chunk or one autoregressive row.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DecoderPhase {
    Prefill,
    Decode,
}

/// Which logits a call materializes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LogitsSelection {
    /// Only the final row, used by ordinary prefill and decode.
    Last,
    /// Every row, for callers that explicitly need a dense logits matrix.
    /// The opt-in batched MTP verification candidate uses this mode; the
    /// canonical path invokes four one-row decode calls instead.
    All,
}

/// Which post-final-norm hidden rows are retained for the MTP head.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HiddenSelection {
    Last,
    All,
}

/// Immutable call contract passed to the full-attention implementation.
pub struct FullAttentionRequest<'a> {
    pub layer_index: usize,
    pub phase: DecoderPhase,
    pub start_position: usize,
    pub geometry: AttentionGeometry,
    pub weights: &'a DeviceAttentionWeights,
    pub normalized_hidden: &'a Tensor,
    pub output: &'a Tensor,
}

/// Missing target-specific FP8-attention boundary.
///
/// Implementations own the cache representation and must append exactly the
/// request rows at `start_position`. `copy_cache_range` and `reset_cache`
/// enqueue on `ctx`'s stream; [`super::session::Qwen35Session`] uses them to
/// prepare a shadow cache before invoking `run`, and never exposes that shadow
/// on error.
pub trait FullAttentionBackend {
    type Cache;

    fn allocate_cache(
        &mut self,
        ctx: &CudaContext,
        layer_index: usize,
        geometry: AttentionGeometry,
        capacity: usize,
        scales: KvCacheScales,
    ) -> Result<Self::Cache>;

    fn reset_cache(&mut self, ctx: &CudaContext, cache: &mut Self::Cache) -> Result<()>;

    /// Copy only cache rows in `positions`. Sessions track how much of the
    /// inactive bank already matches the committed prefix, avoiding an O(T)
    /// full-cache copy before every decode token.
    fn copy_cache_range(
        &mut self,
        ctx: &CudaContext,
        source: &Self::Cache,
        destination: &mut Self::Cache,
        positions: Range<usize>,
    ) -> Result<()>;

    fn run(
        &mut self,
        ctx: &CudaContext,
        request: FullAttentionRequest<'_>,
        cache: &mut Self::Cache,
    ) -> Result<()>;
}

/// Result of one target-decoder transaction. Variable-size prefill owns its
/// returned CUDA storage. Session decode results are views of persistent
/// one-row scratch and remain stable only until the next decode on that same
/// session.
#[derive(Debug)]
pub struct DecoderOutput {
    pub logits: Tensor,
    pub rows: usize,
    /// Target hidden states after the base model's final RMSNorm. These are the
    /// `hidden_states_input` consumed by the checkpoint's MTP fusion layer.
    pub hidden_states: Tensor,
    pub hidden_rows: usize,
    pub start_position: usize,
    pub end_position: usize,
}

impl DecoderOutput {
    pub fn last_hidden_state(&self) -> Result<Tensor> {
        let dims = self.hidden_states.shape().dims();
        if dims.len() != 2 || dims[0] != self.hidden_rows || dims[1] == 0 || self.hidden_rows == 0 {
            return Err(invalid(
                "Qwen3.5 final hidden states must be non-empty rank-2",
            ));
        }
        row_view(&self.hidden_states, self.hidden_rows - 1, dims[1])
    }

    /// View the final row as a scalar-decode-shaped output without copying.
    pub(super) fn last_row(&self) -> Result<Self> {
        let logit_dims = self.logits.shape().dims();
        let hidden_dims = self.hidden_states.shape().dims();
        if logit_dims.len() != 2
            || hidden_dims.len() != 2
            || logit_dims[0] != self.rows
            || hidden_dims[0] != self.hidden_rows
            || self.rows == 0
            || self.hidden_rows == 0
            || self.end_position == 0
        {
            return Err(invalid(
                "Qwen3.5 decoder output cannot expose an empty final row",
            ));
        }
        Ok(Self {
            logits: row_view(&self.logits, self.rows - 1, logit_dims[1])?,
            rows: 1,
            hidden_states: row_view(&self.hidden_states, self.hidden_rows - 1, hidden_dims[1])?,
            hidden_rows: 1,
            start_position: self.end_position - 1,
            end_position: self.end_position,
        })
    }

    /// Device argmax for each materialized logit row, followed by one compact
    /// host transfer. For `LogitsSelection::All` this produces the target
    /// predictions consumed by `Qwen35MtpSchedule::finish_round`.
    pub fn greedy_token_ids(&self, ctx: &CudaContext) -> Result<Vec<u32>> {
        let indices = zeros(ctx, vec![self.rows], DType::I32)?;
        self.enqueue_greedy_token_ids(ctx, &indices, 0)?;
        ctx.synchronize().map_err(Error::Cuda)?;
        let host = transfers::to_cpu(&indices)?;
        host.as_i32()?
            .iter()
            .map(|&token| {
                u32::try_from(token)
                    .map_err(|_| invalid(format!("Qwen3.5 argmax returned negative ID {token}")))
            })
            .collect()
    }

    /// Enqueue argmax for every logit row into `indices[offset..]` without a
    /// host synchronization. This lets canonical scalar verification retain
    /// the exact per-row kernel path while paying for one round-level sync and
    /// compact D2H transfer.
    pub fn enqueue_greedy_token_ids(
        &self,
        ctx: &CudaContext,
        indices: &Tensor,
        offset: usize,
    ) -> Result<()> {
        let dims = self.logits.shape().dims();
        if dims.len() != 2 || dims[0] != self.rows {
            return Err(invalid("Qwen3.5 logits must be rank-2"));
        }
        let vocab = dims[1];
        if vocab == 0 || self.rows == 0 {
            return Err(invalid("Qwen3.5 logits dimensions must be non-zero"));
        }
        let index_dims = indices.shape().dims();
        if indices.device() != Device::Cuda(ctx.device_id())
            || indices.dtype() != DType::I32
            || index_dims.len() != 1
            || offset
                .checked_add(self.rows)
                .is_none_or(|end| end > index_dims[0])
        {
            return Err(invalid(
                "Qwen3.5 argmax destination must be a sufficiently large CUDA I32 vector",
            ));
        }
        for row in 0..self.rows {
            let logits = row_view(&self.logits, row, vocab)?;
            let index = element_view(indices, offset + row)?;
            ops::argmax(ctx, &logits, &index)?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct Qwen35Decoder {
    architecture: Qwen35MoeArchitecture,
    rms_norm_eps: f32,
}

impl Qwen35Decoder {
    pub fn new(config: &Qwen35MoeConfig) -> Result<Self> {
        Ok(Self {
            architecture: Qwen35MoeArchitecture::from_config(config)?,
            rms_norm_eps: config.text_config.rms_norm_eps,
        })
    }

    pub fn architecture(&self) -> &Qwen35MoeArchitecture {
        &self.architecture
    }

    pub fn validate_weights(&self, weights: &Qwen35DeviceWeights) -> Result<()> {
        if weights.layers.len() != self.architecture.layers.len() {
            return Err(invalid(format!(
                "Qwen3.5 decoder expected {} layers, got {}",
                self.architecture.layers.len(),
                weights.layers.len()
            )));
        }
        for (layer, resident) in self.architecture.layers.iter().zip(&weights.layers) {
            let matches = matches!(
                (&layer.mixer, &resident.mixer),
                (MixerKind::Gdn(_), DeviceMixerWeights::Gdn(_))
                    | (
                        MixerKind::FullAttention(_),
                        DeviceMixerWeights::FullAttention(_)
                    )
            );
            if !matches {
                return Err(invalid(format!(
                    "Qwen3.5 layer {} resident mixer does not match architecture",
                    layer.index
                )));
            }
        }
        Ok(())
    }

    /// Execute a prompt/chunk or one decode row against caller-provided shadow
    /// state. The caller owns commit/rollback and must synchronize before
    /// publishing those states. This general entry point allocates scratch
    /// sized to the call, which is required for variable-length prefill.
    #[allow(clippy::too_many_arguments)]
    pub fn run<B: FullAttentionBackend>(
        &self,
        ctx: &CudaContext,
        weights: &Qwen35DeviceWeights,
        attention: &mut B,
        attention_caches: &mut [B::Cache],
        gdn_states: &[GdnState],
        token_ids: &[u32],
        start_position: usize,
        phase: DecoderPhase,
        logits_selection: LogitsSelection,
        hidden_selection: HiddenSelection,
    ) -> Result<DecoderOutput> {
        self.validate_weights(weights)?;
        self.validate_call(
            attention_caches.len(),
            gdn_states.len(),
            token_ids,
            start_position,
            phase,
        )?;
        let tokens = token_ids.len();
        let hidden = self.architecture.hidden_size;
        let moe = self.architecture.layers[0].moe;
        let mut scratch = DecoderScratch::new(
            ctx,
            tokens,
            hidden,
            self.architecture.vocab_size,
            moe.experts,
            moe.experts_per_token,
            moe.expert_intermediate_size,
            logits_selection,
            hidden_selection,
        )?;
        self.run_validated(
            ctx,
            weights,
            attention,
            attention_caches,
            gdn_states,
            token_ids,
            start_position,
            phase,
            logits_selection,
            hidden_selection,
            &mut scratch,
        )
    }

    /// Allocate the one-row scratch bank reused by session decode and exact
    /// scalar MTP verification. It is intentionally separate from variable
    /// prompt scratch so a large prefill does not permanently pin its buffers.
    pub(super) fn allocate_decode_scratch(&self, ctx: &CudaContext) -> Result<DecoderScratch> {
        let moe = self.architecture.layers[0].moe;
        DecoderScratch::new(
            ctx,
            1,
            self.architecture.hidden_size,
            self.architecture.vocab_size,
            moe.experts,
            moe.experts_per_token,
            moe.expert_intermediate_size,
            LogitsSelection::Last,
            HiddenSelection::Last,
        )
    }

    /// Persistent four-row scratch for the speculative target candidate.
    pub(super) fn allocate_verify_scratch(&self, ctx: &CudaContext) -> Result<DecoderScratch> {
        let moe = self.architecture.layers[0].moe;
        DecoderScratch::new(
            ctx,
            super::mtp_schedule::QWEN35_MTP_VERIFY_ROWS,
            self.architecture.hidden_size,
            self.architecture.vocab_size,
            moe.experts,
            moe.experts_per_token,
            moe.expert_intermediate_size,
            LogitsSelection::All,
            HiddenSelection::All,
        )
    }

    /// Execute one scalar decode row into caller-owned persistent scratch.
    /// Returned tensors alias that scratch and remain valid until the next
    /// decode on the same session. Callers that need a value across another
    /// row must consume it first or retain only a derived device copy.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn run_decode_reusing<B: FullAttentionBackend>(
        &self,
        ctx: &CudaContext,
        weights: &Qwen35DeviceWeights,
        attention: &mut B,
        attention_caches: &mut [B::Cache],
        gdn_states: &[GdnState],
        token_ids: &[u32],
        start_position: usize,
        scratch: &mut DecoderScratch,
    ) -> Result<DecoderOutput> {
        self.validate_weights(weights)?;
        self.validate_call(
            attention_caches.len(),
            gdn_states.len(),
            token_ids,
            start_position,
            DecoderPhase::Decode,
        )?;
        scratch.validate_contract(
            token_ids.len(),
            self.architecture.hidden_size,
            self.architecture.vocab_size,
            LogitsSelection::Last,
            HiddenSelection::Last,
        )?;
        self.run_validated(
            ctx,
            weights,
            attention,
            attention_caches,
            gdn_states,
            token_ids,
            start_position,
            DecoderPhase::Decode,
            LogitsSelection::Last,
            HiddenSelection::Last,
            scratch,
        )
    }

    /// Same exact scalar decode path as [`Self::run_decode_reusing`], but the
    /// token is selected from a device-resident I32 block. Verification stages
    /// its four host IDs once, then uses ordered D2D copies so a nonblocking
    /// CUDA stream never races a default-stream H2D overwrite.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn run_decode_reusing_from_block<B: FullAttentionBackend>(
        &self,
        ctx: &CudaContext,
        weights: &Qwen35DeviceWeights,
        attention: &mut B,
        attention_caches: &mut [B::Cache],
        gdn_states: &[GdnState],
        token_id: u32,
        token_block: &Tensor,
        token_offset: usize,
        start_position: usize,
        scratch: &mut DecoderScratch,
    ) -> Result<DecoderOutput> {
        self.validate_weights(weights)?;
        self.validate_call(
            attention_caches.len(),
            gdn_states.len(),
            &[token_id],
            start_position,
            DecoderPhase::Decode,
        )?;
        scratch.validate_contract(
            1,
            self.architecture.hidden_size,
            self.architecture.vocab_size,
            LogitsSelection::Last,
            HiddenSelection::Last,
        )?;
        scratch.stage_device_token(ctx, token_block, token_offset)?;
        self.run_after_token_ids(
            ctx,
            weights,
            attention,
            attention_caches,
            gdn_states,
            start_position,
            DecoderPhase::Decode,
            LogitsSelection::Last,
            HiddenSelection::Last,
            scratch,
        )
    }

    /// Execute the fixed four-row speculative verification candidate through
    /// the batched prefill kernels while appending to an existing prefix.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn run_verify_reusing_from_block<B: FullAttentionBackend>(
        &self,
        ctx: &CudaContext,
        weights: &Qwen35DeviceWeights,
        attention: &mut B,
        attention_caches: &mut [B::Cache],
        gdn_states: &[GdnState],
        token_ids: &[u32],
        token_block: &Tensor,
        start_position: usize,
        scratch: &mut DecoderScratch,
    ) -> Result<DecoderOutput> {
        self.validate_weights(weights)?;
        self.validate_call(
            attention_caches.len(),
            gdn_states.len(),
            token_ids,
            start_position,
            DecoderPhase::Prefill,
        )?;
        scratch.validate_contract(
            token_ids.len(),
            self.architecture.hidden_size,
            self.architecture.vocab_size,
            LogitsSelection::All,
            HiddenSelection::All,
        )?;
        scratch.stage_device_tokens(ctx, token_block)?;
        self.run_after_token_ids(
            ctx,
            weights,
            attention,
            attention_caches,
            gdn_states,
            start_position,
            DecoderPhase::Prefill,
            LogitsSelection::All,
            HiddenSelection::All,
            scratch,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn run_validated<B: FullAttentionBackend>(
        &self,
        ctx: &CudaContext,
        weights: &Qwen35DeviceWeights,
        attention: &mut B,
        attention_caches: &mut [B::Cache],
        gdn_states: &[GdnState],
        token_ids: &[u32],
        start_position: usize,
        phase: DecoderPhase,
        logits_selection: LogitsSelection,
        hidden_selection: HiddenSelection,
        scratch: &mut DecoderScratch,
    ) -> Result<DecoderOutput> {
        scratch.write_token_ids(token_ids)?;
        self.run_after_token_ids(
            ctx,
            weights,
            attention,
            attention_caches,
            gdn_states,
            start_position,
            phase,
            logits_selection,
            hidden_selection,
            scratch,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn run_after_token_ids<B: FullAttentionBackend>(
        &self,
        ctx: &CudaContext,
        weights: &Qwen35DeviceWeights,
        attention: &mut B,
        attention_caches: &mut [B::Cache],
        gdn_states: &[GdnState],
        start_position: usize,
        phase: DecoderPhase,
        logits_selection: LogitsSelection,
        hidden_selection: HiddenSelection,
        scratch: &mut DecoderScratch,
    ) -> Result<DecoderOutput> {
        let tokens = scratch.tokens;
        scratch.embed(ctx, &weights.embedding)?;

        let mut gdn_index = 0usize;
        let mut attention_index = 0usize;
        for (layer, resident) in self.architecture.layers.iter().zip(&weights.layers) {
            qwen35_rms_norm(
                ctx,
                &scratch.hidden,
                &resident.input_layernorm,
                &mut scratch.normalized,
                self.rms_norm_eps,
            )
            .map_err(|error| layer_error(layer.index, "input RMSNorm", error))?;

            match (&layer.mixer, &resident.mixer) {
                (MixerKind::Gdn(_), DeviceMixerWeights::Gdn(gdn)) => {
                    let state = &gdn_states[gdn_index];
                    match phase {
                        DecoderPhase::Prefill => scratch
                            .gdn
                            .prefill(ctx, gdn, state, &scratch.normalized)
                            .map_err(|error| layer_error(layer.index, "GDN prefill", error))?,
                        DecoderPhase::Decode => scratch
                            .gdn
                            .decode(ctx, gdn, state, &scratch.normalized)
                            .map_err(|error| layer_error(layer.index, "GDN decode", error))?,
                    }
                    ops::add_into(ctx, scratch.gdn.output(), &scratch.hidden)
                        .map_err(|error| layer_error(layer.index, "GDN residual", error))?;
                    gdn_index += 1;
                }
                (
                    MixerKind::FullAttention(geometry),
                    DeviceMixerWeights::FullAttention(full_attention),
                ) => {
                    attention
                        .run(
                            ctx,
                            FullAttentionRequest {
                                layer_index: layer.index,
                                phase,
                                start_position,
                                geometry: *geometry,
                                weights: full_attention,
                                normalized_hidden: &scratch.normalized,
                                output: &scratch.mixer_output,
                            },
                            &mut attention_caches[attention_index],
                        )
                        .map_err(|error| layer_error(layer.index, "attention", error))?;
                    ops::add_into(ctx, &scratch.mixer_output, &scratch.hidden)
                        .map_err(|error| layer_error(layer.index, "attention residual", error))?;
                    attention_index += 1;
                }
                _ => {
                    return Err(invalid(format!(
                        "Qwen3.5 layer {} mixer changed after validation",
                        layer.index
                    )));
                }
            }

            qwen35_rms_norm(
                ctx,
                &scratch.hidden,
                &resident.post_attention_layernorm,
                &mut scratch.normalized,
                self.rms_norm_eps,
            )
            .map_err(|error| layer_error(layer.index, "post-attention RMSNorm", error))?;
            scratch
                .moe
                .run(ctx, &resident.moe, &scratch.normalized)
                .map_err(|error| layer_error(layer.index, "MoE", error))?;
            ops::add_into(ctx, scratch.moe.output(), &scratch.hidden)
                .map_err(|error| layer_error(layer.index, "MoE residual", error))?;
        }

        scratch.finish(
            ctx,
            &weights.final_norm,
            &weights.lm_head,
            self.rms_norm_eps,
            logits_selection,
            hidden_selection,
        )?;
        let end_position = start_position
            .checked_add(tokens)
            .ok_or_else(|| invalid("Qwen3.5 decoder position overflow"))?;
        Ok(DecoderOutput {
            logits: scratch.logits.clone(),
            rows: scratch.logit_rows,
            hidden_states: scratch.final_normalized.clone(),
            hidden_rows: scratch.hidden_rows,
            start_position,
            end_position,
        })
    }

    fn validate_call(
        &self,
        attention_caches: usize,
        gdn_states: usize,
        token_ids: &[u32],
        start_position: usize,
        phase: DecoderPhase,
    ) -> Result<()> {
        if token_ids.is_empty() {
            return Err(invalid("Qwen3.5 decoder input must not be empty"));
        }
        if token_ids
            .iter()
            .any(|&token| token as usize >= self.architecture.vocab_size)
        {
            return Err(invalid(
                "Qwen3.5 decoder token ID is outside the vocabulary",
            ));
        }
        if phase == DecoderPhase::Decode && token_ids.len() != 1 {
            return Err(invalid("Qwen3.5 decode requires exactly one token"));
        }
        let end = start_position
            .checked_add(token_ids.len())
            .ok_or_else(|| invalid("Qwen3.5 decoder position overflow"))?;
        if end > self.architecture.max_position_embeddings {
            return Err(invalid(format!(
                "Qwen3.5 decoder end position {end} exceeds context capacity {}",
                self.architecture.max_position_embeddings
            )));
        }
        let expected_attention = self.architecture.full_attention_layers().count();
        let expected_gdn = self.architecture.gdn_layers().count();
        if attention_caches != expected_attention || gdn_states != expected_gdn {
            return Err(invalid(format!(
                "Qwen3.5 state count mismatch: expected {expected_attention} attention/{expected_gdn} GDN, got {attention_caches}/{gdn_states}"
            )));
        }
        Ok(())
    }
}

pub(super) struct DecoderScratch {
    token_ids: Tensor,
    hidden: Tensor,
    normalized: Tensor,
    mixer_output: Tensor,
    final_normalized: Tensor,
    logits: Tensor,
    gdn: GdnScratch,
    moe: BaseMoeScratch,
    tokens: usize,
    hidden_size: usize,
    vocab_size: usize,
    logit_rows: usize,
    hidden_rows: usize,
}

impl DecoderScratch {
    #[allow(clippy::too_many_arguments)]
    fn new(
        ctx: &CudaContext,
        tokens: usize,
        hidden_size: usize,
        vocab_size: usize,
        experts: usize,
        top_k: usize,
        intermediate: usize,
        logits_selection: LogitsSelection,
        hidden_selection: HiddenSelection,
    ) -> Result<Self> {
        let (logit_rows, hidden_rows) =
            selected_row_contract(tokens, logits_selection, hidden_selection)?;
        Ok(Self {
            token_ids: zeros(ctx, vec![tokens], DType::I32)?,
            hidden: zeros(ctx, vec![tokens, hidden_size], DType::BF16)?,
            normalized: zeros(ctx, vec![tokens, hidden_size], DType::BF16)?,
            mixer_output: zeros(ctx, vec![tokens, hidden_size], DType::BF16)?,
            final_normalized: zeros(ctx, vec![hidden_rows, hidden_size], DType::BF16)?,
            logits: zeros(ctx, vec![logit_rows, vocab_size], DType::BF16)?,
            gdn: GdnScratch::new(ctx, tokens, hidden_size)?,
            moe: BaseMoeScratch::new(ctx, tokens, hidden_size, experts, top_k, intermediate)?,
            tokens,
            hidden_size,
            vocab_size,
            logit_rows,
            hidden_rows,
        })
    }

    fn validate_contract(
        &self,
        tokens: usize,
        hidden_size: usize,
        vocab_size: usize,
        logits_selection: LogitsSelection,
        hidden_selection: HiddenSelection,
    ) -> Result<()> {
        let logit_rows = selected_rows(tokens, logits_selection);
        let hidden_rows = selected_hidden_rows(tokens, hidden_selection);
        if self.tokens != tokens
            || self.hidden_size != hidden_size
            || self.vocab_size != vocab_size
            || self.logit_rows != logit_rows
            || self.hidden_rows != hidden_rows
        {
            return Err(invalid(
                "Qwen3.5 reusable decoder scratch does not match scalar decode contract",
            ));
        }
        Ok(())
    }

    fn write_token_ids(&self, token_ids: &[u32]) -> Result<()> {
        let values = token_ids
            .iter()
            .map(|&token| {
                i32::try_from(token)
                    .map_err(|_| invalid(format!("Qwen3.5 token ID {token} exceeds i32")))
            })
            .collect::<Result<Vec<_>>>()?;
        let host = Tensor::from_i32(vec![self.tokens], &values)?;
        transfers::copy_cpu_to_cuda(&host, &self.token_ids)
    }

    fn stage_device_token(
        &self,
        ctx: &CudaContext,
        token_block: &Tensor,
        offset: usize,
    ) -> Result<()> {
        let dims = token_block.shape().dims();
        if token_block.device() != Device::Cuda(ctx.device_id())
            || token_block.dtype() != DType::I32
            || dims.len() != 1
            || offset >= dims[0]
        {
            return Err(invalid(
                "Qwen3.5 verification tokens must be a non-empty CUDA I32 vector",
            ));
        }
        let source = element_view(token_block, offset)?;
        transfers::copy_cuda_to_cuda(ctx, &source, &self.token_ids)
    }

    fn stage_device_tokens(&self, ctx: &CudaContext, token_block: &Tensor) -> Result<()> {
        let dims = token_block.shape().dims();
        if token_block.device() != Device::Cuda(ctx.device_id())
            || token_block.dtype() != DType::I32
            || dims != [self.tokens]
        {
            return Err(invalid(
                "Qwen3.5 verification token block has the wrong CUDA I32 shape",
            ));
        }
        transfers::copy_cuda_to_cuda(ctx, token_block, &self.token_ids)
    }

    fn embed(&self, ctx: &CudaContext, embedding: &Tensor) -> Result<()> {
        validate_tensor(
            embedding,
            ctx,
            DType::BF16,
            &[self.vocab_size, self.hidden_size],
            "embedding",
        )?;
        ops::embedding_gather(ctx, embedding, &self.token_ids, &self.hidden)
    }

    fn finish(
        &mut self,
        ctx: &CudaContext,
        final_norm: &Tensor,
        lm_head: &Tensor,
        eps: f32,
        logits_selection: LogitsSelection,
        hidden_selection: HiddenSelection,
    ) -> Result<()> {
        validate_tensor(
            final_norm,
            ctx,
            DType::BF16,
            &[self.hidden_size],
            "final norm",
        )?;
        validate_tensor(
            lm_head,
            ctx,
            DType::BF16,
            &[self.hidden_size, self.vocab_size],
            "lm_head",
        )?;
        let source = match hidden_selection {
            HiddenSelection::All => self.hidden.clone(),
            HiddenSelection::Last => row_view(&self.hidden, self.tokens - 1, self.hidden_size)?,
        };
        qwen35_rms_norm(ctx, &source, final_norm, &mut self.final_normalized, eps)?;
        let head_input = match logits_selection {
            LogitsSelection::All => self.final_normalized.clone(),
            LogitsSelection::Last => row_view(
                &self.final_normalized,
                self.hidden_rows - 1,
                self.hidden_size,
            )?,
        };
        ops::gemm(
            ctx,
            ops::GemmArgs::new(&head_input, lm_head, &mut self.logits),
        )
    }
}

fn selected_rows(tokens: usize, selection: LogitsSelection) -> usize {
    match selection {
        LogitsSelection::Last => 1,
        LogitsSelection::All => tokens,
    }
}

fn selected_hidden_rows(tokens: usize, selection: HiddenSelection) -> usize {
    match selection {
        HiddenSelection::Last => 1,
        HiddenSelection::All => tokens,
    }
}

fn selected_row_contract(
    tokens: usize,
    logits: LogitsSelection,
    hidden: HiddenSelection,
) -> Result<(usize, usize)> {
    let logit_rows = selected_rows(tokens, logits);
    let hidden_rows = selected_hidden_rows(tokens, hidden);
    if logit_rows > hidden_rows {
        return Err(invalid(
            "Qwen3.5 all-row logits require all-row final hidden states",
        ));
    }
    Ok((logit_rows, hidden_rows))
}

fn qwen35_rms_norm(
    ctx: &CudaContext,
    input: &Tensor,
    checkpoint_weight: &Tensor,
    output: &mut Tensor,
    eps: f32,
) -> Result<()> {
    // Qwen3.5 stores the RMSNorm delta. cuda-new's MLP semantic applies the
    // required `(1 + checkpoint_weight)` affine in F32.
    ops::mlp::rms_norm(ctx, input, checkpoint_weight, output, eps)
}

fn row_view(tensor: &Tensor, row: usize, width: usize) -> Result<Tensor> {
    let byte_width = width
        .checked_mul(tensor.dtype().size_in_bytes())
        .ok_or_else(|| invalid("Qwen3.5 row byte width overflow"))?;
    let offset = row
        .checked_mul(byte_width)
        .ok_or_else(|| invalid("Qwen3.5 row offset overflow"))?;
    CudaBuffer::from_tensor(tensor)
        .map_err(Error::Cuda)?
        .view(offset, byte_width)
        .map_err(Error::Cuda)?
        .as_tensor(Shape::new(vec![1, width]), tensor.dtype())
        .map_err(Error::Cuda)
}

fn element_view(tensor: &Tensor, index: usize) -> Result<Tensor> {
    let byte_width = tensor.dtype().size_in_bytes();
    let offset = index
        .checked_mul(byte_width)
        .ok_or_else(|| invalid("Qwen3.5 element offset overflow"))?;
    CudaBuffer::from_tensor(tensor)
        .map_err(Error::Cuda)?
        .view(offset, byte_width)
        .map_err(Error::Cuda)?
        .as_tensor(Shape::new(vec![1]), tensor.dtype())
        .map_err(Error::Cuda)
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

fn zeros(ctx: &CudaContext, shape: Vec<usize>, dtype: DType) -> Result<Tensor> {
    let bytes = shape
        .iter()
        .try_fold(dtype.size_in_bytes(), |bytes, dimension| {
            bytes.checked_mul(*dimension)
        })
        .ok_or_else(|| invalid("Qwen3.5 CUDA allocation size overflow"))?;
    Ok(CudaBuffer::alloc_zeros(bytes, ctx.device_id())
        .map_err(Error::Cuda)?
        .as_tensor(Shape::new(shape), dtype)
        .map_err(Error::Cuda)?)
}

fn invalid(message: impl Into<String>) -> Error {
    Error::Other(message.into())
}

fn layer_error(layer: usize, stage: &str, error: Error) -> Error {
    Error::Other(format!("Qwen3.5 layer {layer} {stage}: {error}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn persistent_decode_scratch_is_exactly_one_row() {
        assert_eq!(
            selected_row_contract(1, LogitsSelection::Last, HiddenSelection::Last).unwrap(),
            (1, 1)
        );
    }

    #[test]
    fn variable_prefill_row_contract_preserves_all_rows() {
        assert_eq!(
            selected_row_contract(17, LogitsSelection::All, HiddenSelection::All).unwrap(),
            (17, 17)
        );
        assert!(selected_row_contract(17, LogitsSelection::All, HiddenSelection::Last).is_err());
    }
}
