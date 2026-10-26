//! Qwen3.5 target-decoder dataflow.
//!
//! This module owns the exact residual/norm/layer order and calls the real GDN
//! and routed-NVFP4 MoE implementations.  Full attention is deliberately an
//! explicit backend boundary: the target checkpoint requires partial mRoPE,
//! gated queries and an FP8 KV cache, and silently falling back to a different
//! attention/cache semantic would invalidate both accuracy and MTP rollback.

use apxinf_core::{DType, Device, Error, Result, Shape, Tensor};
use apxinf_cuda_new::{
    ops, transfers, CudaBuffer, CudaContext, CudaEvent, ExecutionSession, PreparedPhase,
};
use std::cell::Cell;
use std::ops::Range;
use std::rc::Rc;

use super::architecture::{AttentionGeometry, MixerKind, Qwen35MoeArchitecture};
use super::config::Qwen35MoeConfig;
use super::device_weights::{
    DeviceAttentionWeights, DeviceLayerWeights, DeviceMixerWeights, DeviceStaticFp8Linear,
    Qwen35DeviceWeights,
};
use super::forward::{project_lm_head, BaseMoeScratch, GdnScratch, GdnState, GdnStateCheckpoints};
use super::runtime_weights::KvCacheScales;

const DECODE_PHASE_WORKSPACE: usize = 4 * 1024 * 1024;
const GDN_PARALLEL_EVENT_SLOTS: usize = 4;

/// Shared auxiliary stream and reusable event pair for sequential GDN layers.
/// Projection phases own their prepared executions separately because every
/// layer has different weight bindings.
struct GdnParallelStreams {
    auxiliary: CudaContext,
}
struct GdnProjectionEvents {
    normalized_ready: CudaEvent,
    projections_ready: CudaEvent,
}

struct GdnProjectionPhase {
    // Sessions must drop before `streams`: their native executions were
    // prepared by the runtimes owned by the primary/auxiliary contexts.
    primary_session: ExecutionSession,
    auxiliary_session: ExecutionSession,
    streams: Rc<GdnParallelStreams>,
    events: Vec<GdnProjectionEvents>,
    next_event: Cell<usize>,
}

impl GdnProjectionPhase {
    fn prepare(
        ctx: &CudaContext,
        streams: Rc<GdnParallelStreams>,
        weights: &super::device_weights::DeviceGdnWeights,
        state: &GdnState,
        scratch: &mut GdnScratch,
        input: &Tensor,
    ) -> Result<Self> {
        scratch.validate_parallel_decode_inputs(ctx, weights, state, input)?;
        let primary_session =
            ExecutionSession::with_capacity(DECODE_PHASE_WORKSPACE, ctx.device_id())?;
        primary_session.prepare(|| scratch.project_decode_qkv(ctx, weights, input))?;
        let auxiliary_session =
            ExecutionSession::with_capacity(DECODE_PHASE_WORKSPACE, ctx.device_id())?;
        auxiliary_session
            .prepare(|| scratch.project_decode_zab(&streams.auxiliary, weights, input))?;
        let events = (0..GDN_PARALLEL_EVENT_SLOTS)
            .map(|_| {
                Ok(GdnProjectionEvents {
                    normalized_ready: CudaEvent::new(ctx.device_id()).map_err(Error::Cuda)?,
                    projections_ready: CudaEvent::new(ctx.device_id()).map_err(Error::Cuda)?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            primary_session,
            auxiliary_session,
            streams,
            events,
            next_event: Cell::new(0),
        })
    }

    fn run(
        &self,
        ctx: &CudaContext,
        weights: &super::device_weights::DeviceGdnWeights,
        state: &GdnState,
        scratch: &mut GdnScratch,
        input: &Tensor,
    ) -> Result<()> {
        scratch.validate_parallel_decode_inputs(ctx, weights, state, input)?;
        let event_index = self.next_event.get();
        self.next_event
            .set((event_index + 1) % GDN_PARALLEL_EVENT_SLOTS);
        let events = &self.events[event_index];

        events
            .normalized_ready
            .record(ctx.stream())
            .map_err(Error::Cuda)?;
        self.streams
            .auxiliary
            .stream()
            .wait_event(&events.normalized_ready)
            .map_err(Error::Cuda)?;

        self.primary_session
            .run(|| scratch.project_decode_qkv(ctx, weights, input))?;
        self.auxiliary_session
            .run(|| scratch.project_decode_zab(&self.streams.auxiliary, weights, input))?;

        events
            .projections_ready
            .record(self.streams.auxiliary.stream())
            .map_err(Error::Cuda)?;
        ctx.stream()
            .wait_event(&events.projections_ready)
            .map_err(Error::Cuda)
    }
}

/// One layer's prepared/captured body plus the optional exact multi-stream
/// projection front-end used by `APXINF_QWEN35_GRAPH_GDN=parallel`.
pub(super) struct Qwen35DecodePhase {
    prepared: PreparedPhase,
    projections: Option<GdnProjectionPhase>,
}

impl Qwen35DecodePhase {
    fn run(&self, operation: impl FnOnce() -> Result<()>) -> Result<()> {
        self.prepared.run(operation)
    }

    fn replay(&self) -> Result<()> {
        self.prepared.replay()
    }
}

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
    /// Every row, used by the fixed four-row MTP target verifier.
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
    pub weights: &'a DeviceAttentionWeights<DeviceStaticFp8Linear>,
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
        scales: Option<KvCacheScales>,
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

    /// Retain only the final logits/hidden row as stable views of this
    /// output's owned storage. Batched speculative verification uses this to
    /// publish the state after its final row through the ordinary scalar
    /// commit contract.
    pub fn into_last_row(self) -> Result<Self> {
        let row = self
            .rows
            .checked_sub(1)
            .ok_or_else(|| invalid("Qwen3.5 decoder output has no final row"))?;
        self.into_row(row)
    }

    /// Retain one logits/hidden row as stable views of this output's owned
    /// storage. The selected row is also the state boundary committed by MTP.
    pub fn into_row(self, row: usize) -> Result<Self> {
        if self.rows == 0 || self.hidden_rows == 0 {
            return Err(invalid("Qwen3.5 decoder output has no rows"));
        }
        if row >= self.rows || row >= self.hidden_rows {
            return Err(invalid("Qwen3.5 decoder output row is out of range"));
        }
        let vocab = self
            .logits
            .shape()
            .dims()
            .get(1)
            .copied()
            .ok_or_else(|| invalid("Qwen3.5 logits must be rank-2"))?;
        let hidden = self
            .hidden_states
            .shape()
            .dims()
            .get(1)
            .copied()
            .ok_or_else(|| invalid("Qwen3.5 hidden states must be rank-2"))?;
        let logits = row_view(&self.logits, row, vocab)?;
        let hidden_states = row_view(&self.hidden_states, row, hidden)?;
        let start_position = self
            .start_position
            .checked_add(row)
            .ok_or_else(|| invalid("Qwen3.5 output row position overflow"))?;
        let end_position = start_position
            .checked_add(1)
            .ok_or_else(|| invalid("Qwen3.5 output row end position overflow"))?;
        Ok(Self {
            logits,
            rows: 1,
            hidden_states,
            hidden_rows: 1,
            start_position,
            end_position,
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
            None,
            token_ids,
            start_position,
            phase,
            logits_selection,
            hidden_selection,
            &mut scratch,
            None,
        )
    }

    /// Allocate the one-row scratch bank reused by ordinary session decode.
    /// It is intentionally separate from variable prompt and MTP verification
    /// scratch so a large call does not permanently pin its buffers.
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

    #[allow(clippy::too_many_arguments)]
    pub(super) fn run_verify_reusing<B: FullAttentionBackend>(
        &self,
        ctx: &CudaContext,
        weights: &Qwen35DeviceWeights,
        attention: &mut B,
        attention_caches: &mut [B::Cache],
        gdn_states: &[GdnState],
        gdn_checkpoints: &[GdnStateCheckpoints],
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
            DecoderPhase::Prefill,
        )?;
        if gdn_checkpoints.len() != gdn_states.len() {
            return Err(invalid("Qwen3.5 GDN checkpoint count mismatch"));
        }
        scratch.validate_contract(
            token_ids.len(),
            self.architecture.hidden_size,
            self.architecture.vocab_size,
            LogitsSelection::All,
            HiddenSelection::All,
        )?;
        self.run_validated(
            ctx,
            weights,
            attention,
            attention_caches,
            gdn_states,
            Some(gdn_checkpoints),
            token_ids,
            start_position,
            DecoderPhase::Prefill,
            LogitsSelection::All,
            HiddenSelection::All,
            scratch,
            None,
        )
    }

    /// Capture one fixed-shape decode graph per transformer layer. Full
    /// attention remains eager because its live KV prefix changes every row;
    /// its position-independent MoE tail is captured. GDN projections remain
    /// eager to preserve the exact GEMM result while the fixed-shape recurrent
    /// state update is captured.
    pub(super) fn capture_decode_phases(
        &self,
        ctx: &CudaContext,
        weights: &Qwen35DeviceWeights,
        gdn_states: &[GdnState],
        scratch: &mut DecoderScratch,
    ) -> Result<Vec<Qwen35DecodePhase>> {
        let graph_gdn_mode = std::env::var("APXINF_QWEN35_GRAPH_GDN").unwrap_or_default();
        let graph_gdn_full = matches!(graph_gdn_mode.as_str(), "full" | "prepared");
        let graph_gdn_moe = graph_gdn_mode == "moe";
        let parallel_gdn = graph_gdn_mode == "parallel";
        let parallel_streams = parallel_gdn
            .then(|| -> Result<_> {
                Ok(Rc::new(GdnParallelStreams {
                    auxiliary: CudaContext::new_with_autotune(ctx.device_id(), false)
                        .map_err(Error::Cuda)?,
                }))
            })
            .transpose()?;
        let mut phases = Vec::with_capacity(self.architecture.layers.len());
        let mut gdn_index = 0usize;
        for (layer, resident) in self.architecture.layers.iter().zip(&weights.layers) {
            let session = ExecutionSession::with_capacity(DECODE_PHASE_WORKSPACE, ctx.device_id())?;
            let (prepared, projections) = match (&layer.mixer, &resident.mixer) {
                (MixerKind::Gdn(_), DeviceMixerWeights::Gdn(gdn)) => {
                    let state = &gdn_states[gdn_index];
                    gdn_index += 1;
                    let projections = parallel_streams
                        .as_ref()
                        .map(|streams| {
                            GdnProjectionPhase::prepare(
                                ctx,
                                Rc::clone(streams),
                                gdn,
                                state,
                                &mut scratch.gdn,
                                &scratch.normalized,
                            )
                        })
                        .transpose()?;
                    let prepared = PreparedPhase::prepare_and_capture(ctx, session, || {
                        if graph_gdn_full {
                            qwen35_rms_norm(
                                ctx,
                                &scratch.hidden,
                                &resident.input_layernorm,
                                &mut scratch.normalized,
                                self.rms_norm_eps,
                            )?;
                            scratch
                                .gdn
                                .decode_core(ctx, gdn, state, &scratch.normalized)?;
                            scratch.gdn.finish_decode_projection(ctx, gdn)?;
                            ops::add_into(ctx, scratch.gdn.output(), &scratch.hidden)?;
                            run_decode_moe_tail(ctx, resident, scratch, self.rms_norm_eps)
                        } else if parallel_gdn {
                            scratch.gdn.decode_prepared_core(ctx, gdn, state)?;
                            scratch.gdn.finish_decode_projection(ctx, gdn)?;
                            ops::add_into(ctx, scratch.gdn.output(), &scratch.hidden)?;
                            run_decode_moe_tail(ctx, resident, scratch, self.rms_norm_eps)
                        } else if graph_gdn_moe {
                            run_decode_moe_tail(ctx, resident, scratch, self.rms_norm_eps)
                        } else {
                            scratch.gdn.decode_prepared_core(ctx, gdn, state)
                        }
                    })?;
                    (prepared, projections)
                }
                (MixerKind::FullAttention(_), DeviceMixerWeights::FullAttention(_)) => (
                    PreparedPhase::prepare_and_capture(ctx, session, || {
                        run_decode_moe_tail(ctx, resident, scratch, self.rms_norm_eps)
                    })?,
                    None,
                ),
                _ => {
                    return Err(invalid(format!(
                        "Qwen3.5 layer {} mixer changed during graph capture",
                        layer.index
                    )))
                }
            };
            phases.push(Qwen35DecodePhase {
                prepared,
                projections,
            });
        }
        Ok(phases)
    }

    /// Diagnostic used while bringing up graph replay on a new CUDA target.
    /// It advances independent eager/captured GDN banks with identical hidden
    /// rows and reports the first layer whose BF16 output differs.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn verify_decode_phases(
        &self,
        ctx: &CudaContext,
        weights: &Qwen35DeviceWeights,
        eager_states: &[GdnState],
        eager_phases: &[Qwen35DecodePhase],
        graph_states: &[GdnState],
        graph_phases: &[Qwen35DecodePhase],
        scratch: &mut DecoderScratch,
    ) -> Result<()> {
        let graph_gdn_full = std::env::var("APXINF_QWEN35_GRAPH_GDN").as_deref() == Ok("full");
        if eager_phases.len() != self.architecture.layers.len()
            || graph_phases.len() != self.architecture.layers.len()
        {
            return Err(invalid("Qwen3.5 graph verification phase count mismatch"));
        }
        for state in eager_states.iter().chain(graph_states) {
            state.reset(ctx)?;
        }
        ctx.synchronize().map_err(Error::Cuda)?;
        let mut gdn_index = 0usize;
        for (layer_offset, (layer, resident)) in self
            .architecture
            .layers
            .iter()
            .zip(&weights.layers)
            .enumerate()
        {
            for step in 0..3usize {
                let values = (0..self.architecture.hidden_size)
                    .map(|column| {
                        half::bf16::from_f32(
                            ((column * 17 + layer.index * 23 + step * 31) % 257) as f32 / 128.0
                                - 1.0,
                        )
                    })
                    .collect::<Vec<_>>();
                let input = Tensor::from_bf16(vec![1, self.architecture.hidden_size], &values)?;
                transfers::copy_cpu_to_cuda(&input, &scratch.hidden)?;
                if !graph_gdn_full {
                    if let (MixerKind::Gdn(_), DeviceMixerWeights::Gdn(gdn)) =
                        (&layer.mixer, &resident.mixer)
                    {
                        qwen35_rms_norm(
                            ctx,
                            &scratch.hidden,
                            &resident.input_layernorm,
                            &mut scratch.normalized,
                            self.rms_norm_eps,
                        )?;
                        scratch.gdn.prepare_decode_inputs(
                            ctx,
                            gdn,
                            &eager_states[gdn_index],
                            &scratch.normalized,
                        )?;
                    }
                }
                eager_phases[layer_offset].run(|| match (&layer.mixer, &resident.mixer) {
                    (MixerKind::Gdn(_), DeviceMixerWeights::Gdn(gdn)) => {
                        if graph_gdn_full {
                            qwen35_rms_norm(
                                ctx,
                                &scratch.hidden,
                                &resident.input_layernorm,
                                &mut scratch.normalized,
                                self.rms_norm_eps,
                            )?;
                            scratch.gdn.decode_core(
                                ctx,
                                gdn,
                                &eager_states[gdn_index],
                                &scratch.normalized,
                            )?;
                            scratch.gdn.finish_decode_projection(ctx, gdn)?;
                            ops::add_into(ctx, scratch.gdn.output(), &scratch.hidden)?;
                            run_decode_moe_tail(ctx, resident, scratch, self.rms_norm_eps)
                        } else {
                            scratch
                                .gdn
                                .decode_prepared_core(ctx, gdn, &eager_states[gdn_index])
                        }
                    }
                    (MixerKind::FullAttention(_), DeviceMixerWeights::FullAttention(_)) => {
                        run_decode_moe_tail(ctx, resident, scratch, self.rms_norm_eps)
                    }
                    _ => Err(invalid("Qwen3.5 graph verification mixer mismatch")),
                })?;
                ctx.synchronize().map_err(Error::Cuda)?;
                let expected = transfers::to_cpu(&scratch.hidden)?.as_bf16()?.to_vec();

                transfers::copy_cpu_to_cuda(&input, &scratch.hidden)?;
                graph_phases[layer_offset].replay()?;
                ctx.synchronize().map_err(Error::Cuda)?;
                if matches!(&layer.mixer, MixerKind::Gdn(_)) {
                    if let Some((state, element, eager, graph)) =
                        eager_states[gdn_index].first_mismatch(&graph_states[gdn_index])?
                    {
                        return Err(invalid(format!(
                            "Qwen3.5 decode graph state mismatch at layer {} step {step} {state}[{element}]: eager={eager:#010x}, graph={graph:#010x}",
                            layer.index,
                        )));
                    }
                }
                let actual = transfers::to_cpu(&scratch.hidden)?.as_bf16()?.to_vec();
                if expected != actual {
                    let element = expected
                        .iter()
                        .zip(&actual)
                        .position(|(expected, actual)| expected != actual)
                        .unwrap_or(0);
                    return Err(invalid(format!(
                        "Qwen3.5 decode graph mismatch at layer {} step {step} element {element}: eager={:#06x}, graph={:#06x}",
                        layer.index,
                        expected[element].to_bits(),
                        actual[element].to_bits(),
                    )));
                }
            }
            if matches!(&layer.mixer, MixerKind::Gdn(_)) {
                gdn_index += 1;
            }
        }
        Ok(())
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
        decode_phases: Option<&[Qwen35DecodePhase]>,
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
            None,
            token_ids,
            start_position,
            DecoderPhase::Decode,
            LogitsSelection::Last,
            HiddenSelection::Last,
            scratch,
            decode_phases,
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
        gdn_checkpoints: Option<&[GdnStateCheckpoints]>,
        token_ids: &[u32],
        start_position: usize,
        phase: DecoderPhase,
        logits_selection: LogitsSelection,
        hidden_selection: HiddenSelection,
        scratch: &mut DecoderScratch,
        decode_phases: Option<&[Qwen35DecodePhase]>,
    ) -> Result<DecoderOutput> {
        scratch.write_token_ids(token_ids)?;
        self.run_after_token_ids(
            ctx,
            weights,
            attention,
            attention_caches,
            gdn_states,
            gdn_checkpoints,
            start_position,
            phase,
            logits_selection,
            hidden_selection,
            scratch,
            decode_phases,
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
        gdn_checkpoints: Option<&[GdnStateCheckpoints]>,
        start_position: usize,
        phase: DecoderPhase,
        logits_selection: LogitsSelection,
        hidden_selection: HiddenSelection,
        scratch: &mut DecoderScratch,
        decode_phases: Option<&[Qwen35DecodePhase]>,
    ) -> Result<DecoderOutput> {
        let tokens = scratch.tokens;
        scratch.embed(ctx, &weights.embedding)?;

        let mut gdn_index = 0usize;
        let mut attention_index = 0usize;
        if let Some(phases) = decode_phases {
            if phase != DecoderPhase::Decode || phases.len() != self.architecture.layers.len() {
                return Err(invalid(
                    "Qwen3.5 decode graph set does not match the architecture",
                ));
            }
        }
        let graph_gdn_mode = std::env::var("APXINF_QWEN35_GRAPH_GDN").unwrap_or_default();
        let graph_gdn = matches!(
            graph_gdn_mode.as_str(),
            "1" | "moe" | "full" | "prepared" | "parallel"
        );
        let graph_gdn_moe = graph_gdn_mode == "moe";
        let graph_gdn_full = graph_gdn_mode == "full";
        let prepared_gdn_full = graph_gdn_mode == "prepared";
        let parallel_gdn = graph_gdn_mode == "parallel";
        let graph_attention_moe =
            std::env::var("APXINF_QWEN35_GRAPH_ATTN_MOE").as_deref() != Ok("0");
        for (layer_offset, (layer, resident)) in self
            .architecture
            .layers
            .iter()
            .zip(&weights.layers)
            .enumerate()
        {
            trace_hash("pre", layer.index, &scratch.hidden);
            match (&layer.mixer, &resident.mixer) {
                (MixerKind::Gdn(_), DeviceMixerWeights::Gdn(gdn)) => {
                    let state = &gdn_states[gdn_index];
                    let checkpoints = gdn_checkpoints.map(|all| &all[gdn_index]);
                    if let Some(phases) = decode_phases.filter(|_| graph_gdn) {
                        if parallel_gdn {
                            qwen35_rms_norm(
                                ctx,
                                &scratch.hidden,
                                &resident.input_layernorm,
                                &mut scratch.normalized,
                                self.rms_norm_eps,
                            )
                            .map_err(|error| layer_error(layer.index, "input RMSNorm", error))?;
                            phases[layer_offset]
                                .projections
                                .as_ref()
                                .ok_or_else(|| {
                                    invalid(format!(
                                        "Qwen3.5 layer {} is missing parallel GDN projections",
                                        layer.index
                                    ))
                                })?
                                .run(ctx, gdn, state, &mut scratch.gdn, &scratch.normalized)
                                .map_err(|error| {
                                    layer_error(layer.index, "parallel GDN projections", error)
                                })?;
                            phases[layer_offset]
                                .run(|| {
                                    scratch.gdn.decode_prepared_core(ctx, gdn, state)?;
                                    scratch.gdn.finish_decode_projection(ctx, gdn)?;
                                    ops::add_into(ctx, scratch.gdn.output(), &scratch.hidden)?;
                                    run_decode_moe_tail(ctx, resident, scratch, self.rms_norm_eps)
                                })
                                .map_err(|error| {
                                    layer_error(layer.index, "parallel prepared GDN layer", error)
                                })?;
                        } else if prepared_gdn_full {
                            phases[layer_offset]
                                .run(|| {
                                    qwen35_rms_norm(
                                        ctx,
                                        &scratch.hidden,
                                        &resident.input_layernorm,
                                        &mut scratch.normalized,
                                        self.rms_norm_eps,
                                    )?;
                                    scratch.gdn.decode_core(
                                        ctx,
                                        gdn,
                                        state,
                                        &scratch.normalized,
                                    )?;
                                    scratch.gdn.finish_decode_projection(ctx, gdn)?;
                                    ops::add_into(ctx, scratch.gdn.output(), &scratch.hidden)?;
                                    run_decode_moe_tail(ctx, resident, scratch, self.rms_norm_eps)
                                })
                                .map_err(|error| {
                                    layer_error(layer.index, "prepared GDN layer", error)
                                })?;
                        } else if graph_gdn_moe {
                            qwen35_rms_norm(
                                ctx,
                                &scratch.hidden,
                                &resident.input_layernorm,
                                &mut scratch.normalized,
                                self.rms_norm_eps,
                            )
                            .map_err(|error| layer_error(layer.index, "input RMSNorm", error))?;
                            scratch
                                .gdn
                                .decode(ctx, gdn, state, &scratch.normalized)
                                .map_err(|error| layer_error(layer.index, "GDN decode", error))?;
                            ops::add_into(ctx, scratch.gdn.output(), &scratch.hidden)
                                .map_err(|error| layer_error(layer.index, "GDN residual", error))?;
                        } else if !graph_gdn_full {
                            qwen35_rms_norm(
                                ctx,
                                &scratch.hidden,
                                &resident.input_layernorm,
                                &mut scratch.normalized,
                                self.rms_norm_eps,
                            )
                            .map_err(|error| layer_error(layer.index, "input RMSNorm", error))?;
                            scratch
                                .gdn
                                .prepare_decode_inputs(ctx, gdn, state, &scratch.normalized)
                                .map_err(|error| {
                                    layer_error(layer.index, "GDN input projections", error)
                                })?;
                        }
                        if !prepared_gdn_full && !parallel_gdn {
                            phases[layer_offset].replay().map_err(|error| {
                                layer_error(layer.index, "GDN graph replay", error)
                            })?;
                        }
                        if !graph_gdn_full && !graph_gdn_moe && !prepared_gdn_full && !parallel_gdn
                        {
                            scratch
                                .gdn
                                .finish_decode_projection(ctx, gdn)
                                .map_err(|error| {
                                    layer_error(layer.index, "GDN output projection", error)
                                })?;
                            ops::add_into(ctx, scratch.gdn.output(), &scratch.hidden)
                                .map_err(|error| layer_error(layer.index, "GDN residual", error))?;
                            run_decode_moe_tail(ctx, resident, scratch, self.rms_norm_eps)
                                .map_err(|error| layer_error(layer.index, "MoE tail", error))?;
                        }
                    } else {
                        qwen35_rms_norm(
                            ctx,
                            &scratch.hidden,
                            &resident.input_layernorm,
                            &mut scratch.normalized,
                            self.rms_norm_eps,
                        )
                        .map_err(|error| layer_error(layer.index, "input RMSNorm", error))?;
                        match phase {
                            DecoderPhase::Prefill => match checkpoints {
                                Some(checkpoints) => scratch
                                    .gdn
                                    .prefill_with_checkpoints(
                                        ctx,
                                        gdn,
                                        state,
                                        &scratch.normalized,
                                        checkpoints,
                                    )
                                    .map_err(|error| {
                                        layer_error(layer.index, "GDN checkpointed prefill", error)
                                    })?,
                                None => scratch
                                    .gdn
                                    .prefill(ctx, gdn, state, &scratch.normalized)
                                    .map_err(|error| {
                                    layer_error(layer.index, "GDN prefill", error)
                                })?,
                            },
                            DecoderPhase::Decode => scratch
                                .gdn
                                .decode(ctx, gdn, state, &scratch.normalized)
                                .map_err(|error| layer_error(layer.index, "GDN decode", error))?,
                        }
                        ops::add_into(ctx, scratch.gdn.output(), &scratch.hidden)
                            .map_err(|error| layer_error(layer.index, "GDN residual", error))?;
                        run_decode_moe_tail(ctx, resident, scratch, self.rms_norm_eps)
                            .map_err(|error| layer_error(layer.index, "MoE tail", error))?;
                    }
                    gdn_index += 1;
                }
                (
                    MixerKind::FullAttention(geometry),
                    DeviceMixerWeights::FullAttention(full_attention),
                ) => {
                    qwen35_rms_norm(
                        ctx,
                        &scratch.hidden,
                        &resident.input_layernorm,
                        &mut scratch.normalized,
                        self.rms_norm_eps,
                    )
                    .map_err(|error| layer_error(layer.index, "input RMSNorm", error))?;
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
                    if let Some(phases) = decode_phases.filter(|_| graph_attention_moe) {
                        phases[layer_offset].replay().map_err(|error| {
                            layer_error(layer.index, "attention MoE graph replay", error)
                        })?;
                    } else {
                        run_decode_moe_tail(ctx, resident, scratch, self.rms_norm_eps)
                            .map_err(|error| layer_error(layer.index, "MoE tail", error))?;
                    }
                    attention_index += 1;
                }
                _ => {
                    return Err(invalid(format!(
                        "Qwen3.5 layer {} mixer changed after validation",
                        layer.index
                    )));
                }
            }
            trace_hash("post", layer.index, &scratch.hidden);
        }

        scratch.finish(
            ctx,
            weights,
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
        weights: &Qwen35DeviceWeights,
        eps: f32,
        logits_selection: LogitsSelection,
        hidden_selection: HiddenSelection,
    ) -> Result<()> {
        validate_tensor(
            &weights.final_norm,
            ctx,
            DType::BF16,
            &[self.hidden_size],
            "final norm",
        )?;
        if weights.lm_head.input_features != self.hidden_size
            || weights.lm_head.output_features != self.vocab_size
        {
            return Err(invalid("Qwen3.6 lm_head geometry mismatch"));
        }
        let source = match hidden_selection {
            HiddenSelection::All => self.hidden.clone(),
            HiddenSelection::Last => row_view(&self.hidden, self.tokens - 1, self.hidden_size)?,
        };
        trace_hash("hidden_in_finish", 0, &self.hidden);
        trace_hash("source", 0, &source);
        if std::env::var("APXINF_QWEN35_HASH_TRACE").is_ok() {
            eprintln!(
                "[qwen35-hash] counters tokens={} hidden_rows={} hidden={:?} final_norm_buf={:?}",
                self.tokens,
                self.hidden_rows,
                self.hidden.shape().dims(),
                self.final_normalized.shape().dims(),
            );
        }
        qwen35_rms_norm(
            ctx,
            &source,
            &weights.final_norm,
            &mut self.final_normalized,
            eps,
        )?;
        trace_hash("final_normalized", 0, &self.final_normalized);
        let head_input = match logits_selection {
            LogitsSelection::All => self.final_normalized.clone(),
            LogitsSelection::Last => row_view(
                &self.final_normalized,
                self.hidden_rows - 1,
                self.hidden_size,
            )?,
        };
        trace_hash("head_input", 0, &head_input);
        project_lm_head(ctx, weights, &head_input, &mut self.logits)?;
        trace_hash("logits", 0, &self.logits);
        Ok(())
    }
}

/// Hash a device tensor and print it, for localizing run-to-run divergence.
///
/// Default off. `APXINF_QWEN35_HASH_TRACE=1` prints one FNV-1a-64 per layer so
/// two generations in the same process can be diffed to find the first op
/// whose output moves. Synchronizes and copies to host, so it is far too slow
/// for anything but debugging.
pub(super) fn trace_hash(label: &str, index: usize, tensor: &Tensor) {
    static ENABLED: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if !*ENABLED.get_or_init(|| std::env::var("APXINF_QWEN35_HASH_TRACE").is_ok()) {
        return;
    }
    let host = match apxinf_cuda_new::transfers::to_cpu(tensor) {
        Ok(value) => value,
        Err(error) => {
            eprintln!("[qwen35-hash] {label}[{index}] to_cpu failed: {error}");
            return;
        }
    };
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    let mut feed = |bytes: [u8; 4]| {
        for byte in bytes {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x100_0000_01b3);
        }
    };
    match host.dtype() {
        DType::BF16 => match host.as_bf16() {
            Ok(values) => values
                .iter()
                .for_each(|value| feed((value.to_bits() as u32).to_le_bytes())),
            Err(error) => {
                eprintln!("[qwen35-hash] {label}[{index}] as_bf16 failed: {error}");
                return;
            }
        },
        DType::F32 => match host.as_f32() {
            Ok(values) => values.iter().for_each(|value| feed(value.to_bits().to_le_bytes())),
            Err(error) => {
                eprintln!("[qwen35-hash] {label}[{index}] as_f32 failed: {error}");
                return;
            }
        },
        other => {
            eprintln!("[qwen35-hash] {label}[{index}] unsupported dtype {other:?}");
            return;
        }
    }
    eprintln!("[qwen35-hash] {label}[{index}] shape={:?} = {hash:016x}", tensor.shape().dims());
}

fn run_decode_moe_tail(
    ctx: &CudaContext,
    resident: &DeviceLayerWeights,
    scratch: &mut DecoderScratch,
    rms_norm_eps: f32,
) -> Result<()> {
    qwen35_rms_norm(
        ctx,
        &scratch.hidden,
        &resident.post_attention_layernorm,
        &mut scratch.normalized,
        rms_norm_eps,
    )?;
    scratch.moe.run(ctx, &resident.moe, &scratch.normalized)?;
    ops::add_into(ctx, scratch.moe.output(), &scratch.hidden)
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
