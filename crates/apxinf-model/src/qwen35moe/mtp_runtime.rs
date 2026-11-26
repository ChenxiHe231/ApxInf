//! Native Qwen3.5-35B-A3B MTP proposer.
//!
//! This is deliberately not the Qwen3.8 dense-MLP proposer. The checkpoint's
//! one MTP layer is BF16 and contains the same 256-expert/top-8 routed MoE as
//! the target architecture. Its persistent K/V cache follows TensorRT Edge's
//! static unit-scale E4M3 contract and the same 128-token paging as the target.

use super::vision_input::ImagePositions;
use std::cell::Cell;
use std::rc::Rc;
use std::sync::atomic::{AtomicU64, Ordering};

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
use super::session::{
    MtpBackend, MtpDraft, PreparedTargetPrefill, Qwen35Session, Qwen35SessionSnapshot,
};
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

// Keep legacy reads lazy: non-qualified constructors retain their existing
// read point, while a qualified static profile cannot be disabled by ENV0.
fn profile_or_legacy(
    qualified: bool,
    read_legacy: impl FnOnce() -> Option<String>,
) -> bool {
    qualified || read_legacy().as_deref() == Some("1")
}

fn split_profile_or_legacy(
    qualified: bool,
    sm: u32,
    read_legacy: impl FnOnce() -> Option<String>,
) -> bool {
    profile_or_legacy(qualified, read_legacy) && matches!(sm, 101 | 110)
}

// Draft-only candidate vocabulary. Main verification always projects VOCAB.
const MTP_SHORTLIST_PREFIX: usize = 131_072;
const MTP_SHORTLIST_TAIL: usize = 512;
const MTP_HOT_TILE_ROWS: usize = 128;

fn mtp_shortlist_contains(token: TokenId) -> bool {
    (token as usize) < MTP_SHORTLIST_PREFIX
        || ((token as usize) >= VOCAB - MTP_SHORTLIST_TAIL && (token as usize) < VOCAB)
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct MtpHeadBinding(usize, usize, usize, usize);

impl MtpHeadBinding {
    fn new(ctx: &CudaContext, weights: &Qwen35DeviceWeights) -> Result<Self> {
        let head = weights.lm_head.a16.as_ref()
            .ok_or_else(|| invalid("MTP shortlist needs the checkpoint-native W4A16 head"))?;
        let address = |tensor: &Tensor| -> Result<usize> {
            tensor.storage().as_gpu().map(|storage| storage.ptr())
                .ok_or_else(|| invalid("MTP shortlist head weight has no CUDA storage"))
        };
        Ok(Self(ctx.device_id(), address(&head.qweight)?, address(&head.block_scales)?, address(&head.global_scale)?))
    }
}

struct MtpShortlistHead {
    binding: MtpHeadBinding,
    qprefix: Tensor,
    sprefix: Tensor,
    qtail: Tensor,
    stail: Tensor,
    qhot: Tensor,
    shot: Tensor,
    global_scale: Tensor,
    logits: Tensor,
    prefix_output: Tensor,
    tail_output: Tensor,
    hot_output: Tensor,
    remap: Tensor,
    remap_hot: Tensor,
    staged_hot_tile: Cell<Option<usize>>,
}

impl MtpShortlistHead {
    fn slice(tensor: &Tensor, start: usize, rows: usize, width: usize) -> Result<Tensor> {
        let tile_bytes = HIDDEN / 64 * 128 * width * tensor.dtype().size_in_bytes();
        CudaBuffer::from_tensor(tensor).map_err(Error::Cuda)?
            .view(start / 128 * tile_bytes, rows / 128 * tile_bytes).map_err(Error::Cuda)?
            .as_tensor(Shape::new(vec![rows / 128, HIDDEN / 64, 128, width]), tensor.dtype())
            .map_err(Error::Cuda)
    }

    fn new(ctx: &CudaContext, weights: &Qwen35DeviceWeights) -> Result<Self> {
        let binding = MtpHeadBinding::new(ctx, weights)?;
        let head = weights.lm_head.a16.as_ref().expect("validated head");
        if head.qweight.shape().dims() != [VOCAB / 128, HIDDEN / 64, 128, 32]
            || head.block_scales.shape().dims() != [VOCAB / 128, HIDDEN / 64, 128, 4] {
            return Err(invalid("MTP shortlist head does not match fixed checkpoint geometry"));
        }
        let logits = zeros(ctx, vec![1, MTP_SHORTLIST_PREFIX + MTP_SHORTLIST_TAIL + MTP_HOT_TILE_ROWS], DType::BF16)?;
        let prefix_output = rows_view(&logits, 0, 1, MTP_SHORTLIST_PREFIX)?;
        let tail_output = reshape(&vector_view(&logits, MTP_SHORTLIST_PREFIX, MTP_SHORTLIST_TAIL)?,
            vec![1, MTP_SHORTLIST_TAIL], DType::BF16)?;
        let hot_output = reshape(&vector_view(&logits, MTP_SHORTLIST_PREFIX + MTP_SHORTLIST_TAIL,
            MTP_HOT_TILE_ROWS)?, vec![1, MTP_HOT_TILE_ROWS], DType::BF16)?;
        let map: Vec<i32> = (0..MTP_SHORTLIST_PREFIX)
            .chain(VOCAB - MTP_SHORTLIST_TAIL..VOCAB)
            .chain(0..MTP_HOT_TILE_ROWS).map(|x| x as i32).collect();
        let remap = transfers::to_cuda_on_stream(ctx, Tensor::from_i32(vec![map.len()], &map)?)?;
        let remap_hot = vector_view(&remap, MTP_SHORTLIST_PREFIX + MTP_SHORTLIST_TAIL,
            MTP_HOT_TILE_ROWS)?;
        Ok(Self {
            binding,
            qprefix: Self::slice(&head.qweight, 0, MTP_SHORTLIST_PREFIX, 32)?,
            sprefix: Self::slice(&head.block_scales, 0, MTP_SHORTLIST_PREFIX, 4)?,
            qtail: Self::slice(&head.qweight, VOCAB - MTP_SHORTLIST_TAIL, MTP_SHORTLIST_TAIL, 32)?,
            stail: Self::slice(&head.block_scales, VOCAB - MTP_SHORTLIST_TAIL, MTP_SHORTLIST_TAIL, 4)?,
            qhot: zeros(ctx, vec![1, HIDDEN / 64, MTP_HOT_TILE_ROWS, 32], DType::E2M1Pair)?,
            shot: zeros(ctx, vec![1, HIDDEN / 64, MTP_HOT_TILE_ROWS, 4], DType::F8E4M3)?,
            global_scale: head.global_scale.clone(), logits, prefix_output, tail_output,
            hot_output, remap, remap_hot, staged_hot_tile: Cell::new(None),
        })
    }

    fn stage_hot_tile(&self, ctx: &CudaContext, weights: &Qwen35DeviceWeights,
        tile: usize) -> Result<()> {
        if tile >= VOCAB / MTP_HOT_TILE_ROWS {
            return Err(invalid("MTP hot tile exceeds vocabulary"));
        }
        if self.staged_hot_tile.get() == Some(tile) {
            return Ok(());
        }
        if MtpHeadBinding::new(ctx, weights)? != self.binding {
            return Err(invalid("MTP hot tile head binding changed"));
        }
        let head = weights.lm_head.a16.as_ref().expect("validated head binding");
        let start = tile * MTP_HOT_TILE_ROWS;
        let qsource = Self::slice(&head.qweight, start, MTP_HOT_TILE_ROWS, 32)?;
        let ssource = Self::slice(&head.block_scales, start, MTP_HOT_TILE_ROWS, 4)?;
        transfers::copy_cuda_to_cuda(ctx, &qsource, &self.qhot)?;
        transfers::copy_cuda_to_cuda(ctx, &ssource, &self.shot)?;
        let map: Vec<i32> = (start..start + MTP_HOT_TILE_ROWS).map(|x| x as i32).collect();
        // The owned upload waits for preceding stream copies before publishing
        // the tile identity. Graph replay only reads fixed CUDA addresses.
        transfers::copy_cpu_to_cuda_on_stream(ctx,
            Tensor::from_i32(vec![MTP_HOT_TILE_ROWS], &map)?, &self.remap_hot)?;
        self.staged_hot_tile.set(Some(tile));
        Ok(())
    }

    fn project(&self, ctx: &CudaContext, hidden: &Tensor, next: &Tensor) -> Result<()> {
        ops::block_quantized_weight_gemv(ctx, hidden, &self.qprefix, &self.sprefix,
            &self.global_scale, &self.prefix_output)?;
        ops::block_quantized_weight_gemv(ctx, hidden, &self.qtail, &self.stail,
            &self.global_scale, &self.tail_output)?;
        ops::block_quantized_weight_gemv(ctx, hidden, &self.qhot, &self.shot,
            &self.global_scale, &self.hot_output)?;
        // This existing remapped argmax has its documented legacy tie rule;
        // the main verifier retains the canonical lowest-index full argmax.
        ops::argmax(ctx, ops::ArgmaxArgs::new(&self.logits, next).with_remap(&self.remap))
    }
}



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
    image_positions: Option<Rc<ImagePositions>>,
    // Text/native-head capability, independent of past token membership.
    // The pending token chooses full versus shortlist for one complete round.
    shortlist_allowed: bool,
    // Last out-of-shortlist pending token's aligned W4A16 output tile.
    // This is transactional state: a failed draft must not publish a hint.
    hot_tile: Option<usize>,
    generation: u64,
    // Distinguishes allocations and writes even when snapshot restore reuses
    // the same position and transaction generation.
    cache_revision: u64,
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
    committed_cache_revision: u64,
    shadow_cache_revision: u64,
    preserved_rows: usize,
}

#[derive(Clone, Copy)]
struct DraftKvPrefix {
    source_revision: u64,
    destination_revision: u64,
    rows: usize,
}

impl DraftKvPrefix {
    fn reusable_rows(self, source_revision: u64, destination_revision: u64, rows: usize) -> usize {
        if self.source_revision == source_revision
            && self.destination_revision == destination_revision
            && self.rows <= rows
        {
            self.rows
        } else {
            0
        }
    }
}

fn next_cache_revision() -> Result<u64> {
    static NEXT: AtomicU64 = AtomicU64::new(1);
    NEXT.fetch_update(Ordering::Relaxed, Ordering::Relaxed, |revision| {
        revision.checked_add(1)
    })
    .map_err(|_| invalid("Qwen3.5 MTP cache revision overflow"))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct PendingDraft {
    shadow_generation: u64,
    committed_cache_revision: u64,
    shadow_cache_revision: u64,
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
    kv_only_materialization: bool,
    // Immutable execution policy; never copied into transactional state.
    auto_split_kv: bool,
    prepared_shortlist_head: Option<Rc<MtpShortlistHead>>,
    scratch: Option<MtpScratch>,
    prefill_scratch: Option<MtpPrefillScratch>,
    pending_draft: Option<PendingDraft>,
    draft_kv_suffix_sync: bool,
    synced_draft_prefix: Option<DraftKvPrefix>,
}

impl Qwen35MtpRuntime {
    pub fn new(config: &Qwen35MoeConfig) -> Result<Self> {
        Self::with_execution(config, false)
    }

    pub(super) fn with_execution(
        config: &Qwen35MoeConfig,
        thor_dynamic_fp8_decode: bool,
    ) -> Result<Self> {
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
            kv_only_materialization: profile_or_legacy(thor_dynamic_fp8_decode, || {
                std::env::var("APXINF_QWEN35_MTP_KV_ONLY_MATERIALIZE").ok()
            }),
            auto_split_kv: thor_dynamic_fp8_decode,
            prepared_shortlist_head: None,
            scratch: None,
            prefill_scratch: None,
            pending_draft: None,
            draft_kv_suffix_sync: profile_or_legacy(thor_dynamic_fp8_decode, || {
                std::env::var("APXINF_QWEN35_MTP_DRAFT_KV_SUFFIX_SYNC").ok()
            }),
            synced_draft_prefix: None,
        })
    }

    fn prepare_prompt(
        &mut self,
        ctx: &CudaContext,
        weights: &Qwen35DeviceWeights,
        shadow: &mut Qwen35MtpState,
        prompt_tokens: &[TokenId],
        target_hidden: &Tensor,
        image: Option<(&Tensor, Rc<ImagePositions>)>,
    ) -> Result<PreparedMtpPrefill> {
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

        if let Some((embeddings, positions)) = image.as_ref() {
            validate_tensor(
                embeddings,
                ctx,
                DType::BF16,
                &[prompt_tokens.len(), HIDDEN],
                "image MTP embeddings",
            )?;
            // Family prefill may stop at a checkpoint in a validated full image plan.
            if positions.prompt.len() < prompt_tokens.len() {
                return Err(invalid("MTP image position length mismatch"));
            }
        }
        if image.is_none() && weights.lm_head.a16.is_some() {
            self.prepare_shortlist_head(ctx, weights)?;
        }
        self.reset_state(ctx, shadow)?;
        shadow.image_positions = image.as_ref().map(|(_, plan)| plan.clone());
        if position.cache_rows > 0 {
            let shifted_tokens = prompt_tokens[1..]
                .iter()
                .map(|&token| {
                    i32::try_from(token)
                        .map_err(|_| invalid(format!("Qwen3.5 MTP token {token} exceeds i32")))
                })
                .collect::<Result<Vec<_>>>()?;
            let shifted_positions = match image.as_ref() {
                Some((_, plan)) => plan.token_major_values(1, prompt_tokens.len() - 1)?,
                None => (1..prompt_tokens.len())
                    .map(|value| {
                        i32::try_from(value).map_err(|_| invalid("MTP prompt position exceeds i32"))
                    })
                    .collect::<Result<Vec<_>>>()?,
            };
            let shifted_tokens = transfers::to_cuda_on_stream(
                ctx,
                Tensor::from_i32(vec![position.cache_rows], &shifted_tokens)?,
            )?;
            let shifted_positions = transfers::to_cuda_on_stream(
                ctx,
                Tensor::from_i32(
                    if image.is_some() {
                        vec![position.cache_rows, 3]
                    } else {
                        vec![position.cache_rows]
                    },
                    &shifted_positions,
                )?,
            )?;
            for start in (0..position.cache_rows).step_by(MTP_PREFILL_CHUNK) {
                let rows = (position.cache_rows - start).min(MTP_PREFILL_CHUNK);
                let base_hidden = rows_view(target_hidden, start, rows, HIDDEN)?;
                let tokens = vector_view(&shifted_tokens, start, rows)?;
                let rope_positions = if image.is_some() {
                    rows_view(&shifted_positions, start, rows, 3)?
                } else {
                    vector_view(&shifted_positions, start, rows)?
                };
                let shifted_embeddings = image
                    .as_ref()
                    .map(|(embeddings, _)| rows_view(embeddings, start + 1, rows, HIDDEN))
                    .transpose()?;
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
                    shifted_embeddings.as_ref(),
                )?;
            }
        }
        ctx.synchronize().map_err(Error::Cuda)?;
        shadow.position = Some(position);
        shadow.shortlist_allowed = shadow.image_positions.is_none()
            && weights.lm_head.a16.is_some();
        self.pending_draft = None;
        Ok(PreparedMtpPrefill {
            generation: shadow.generation,
            position,
        })
    }

    fn prepare_append_prompt(
        &mut self,
        ctx: &CudaContext,
        weights: &Qwen35DeviceWeights,
        committed: &Qwen35MtpState,
        shadow: &mut Qwen35MtpState,
        chunk_tokens: &[TokenId],
        previous_target_hidden: &Tensor,
        target_hidden: &Tensor,
        image: Option<(&Tensor, Rc<ImagePositions>)>,
    ) -> Result<PreparedMtpPrefill> {
        if committed.image_positions.is_some() && image.is_none() {
            return Err(invalid(
                "image MTP append requires the original media input",
            ));
        }
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
        if let Some((embeddings, plan)) = image.as_ref() {
            validate_tensor(
                embeddings,
                ctx,
                DType::BF16,
                &[chunk_tokens.len(), HIDDEN],
                "image append embeddings",
            )?;
            if position.logical_position > plan.prompt.len() {
                return Err(invalid("image append exceeds position plan"));
            }
        }
        if image.is_none() && weights.lm_head.a16.is_some() {
            self.prepare_shortlist_head(ctx, weights)?;
        }
        self.copy_state(ctx, committed, shadow)?;
        shadow.image_positions = image.as_ref().map(|(_, plan)| plan.clone());
        if image.is_some() {
            shadow.shortlist_allowed = false;
        }

        let token_values = chunk_tokens
            .iter()
            .map(|&token| {
                i32::try_from(token)
                    .map_err(|_| invalid(format!("Qwen3.5 MTP token {token} exceeds i32")))
            })
            .collect::<Result<Vec<_>>>()?;
        let position_values = match image.as_ref() {
            Some((_, plan)) => {
                plan.token_major_values(start.logical_position, chunk_tokens.len())?
            }
            None => (start.logical_position..position.logical_position)
                .map(|p| i32::try_from(p).map_err(|_| invalid("MTP append position exceeds i32")))
                .collect::<Result<Vec<_>>>()?,
        };
        let tokens = transfers::to_cuda_on_stream(
            ctx,
            Tensor::from_i32(vec![chunk_tokens.len()], &token_values)?,
        )?;
        let positions = transfers::to_cuda_on_stream(
            ctx,
            Tensor::from_i32(
                if image.is_some() {
                    vec![chunk_tokens.len(), 3]
                } else {
                    vec![chunk_tokens.len()]
                },
                &position_values,
            )?,
        )?;

        for start_row in (0..chunk_tokens.len()).step_by(MTP_PREFILL_CHUNK) {
            let rows = (chunk_tokens.len() - start_row).min(MTP_PREFILL_CHUNK);
            let tokens = vector_view(&tokens, start_row, rows)?;
            let rope_positions = if image.is_some() {
                rows_view(&positions, start_row, rows, 3)?
            } else {
                vector_view(&positions, start_row, rows)?
            };
            let shifted_embeddings = image
                .as_ref()
                .map(|(embeddings, _)| rows_view(embeddings, start_row, rows, HIDDEN))
                .transpose()?;
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
                shifted_embeddings.as_ref(),
            )?;
        }
        ctx.synchronize().map_err(Error::Cuda)?;
        shadow.position = Some(position);
        shadow.shortlist_allowed = shadow.image_positions.is_none()
            && weights.lm_head.a16.is_some();
        self.pending_draft = None;
        Ok(PreparedMtpPrefill {
            generation: shadow.generation,
            position,
        })
    }

    fn prepare_shortlist_head(
        &mut self,
        ctx: &CudaContext,
        weights: &Qwen35DeviceWeights,
    ) -> Result<()> {
        let binding = MtpHeadBinding::new(ctx, weights)?;
        if self.prepared_shortlist_head.as_ref().is_none_or(|head| head.binding != binding) {
            // Allocate only the existing logits/remap and non-copying views,
            // before normal Decode; do not prepare or traverse model state.
            let head = Rc::new(MtpShortlistHead::new(ctx, weights)?);
            self.prepared_shortlist_head = Some(head);
        }
        Ok(())
    }

    fn ensure_scratch(
        &mut self,
        ctx: &CudaContext,
        cache_capacity: usize,
    ) -> Result<&mut MtpScratch> {
        if self.scratch.is_none() {
            self.scratch = Some(MtpScratch::new(
                ctx,
                self.geometry,
                cache_capacity,
                self.auto_split_kv,
            )?);
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

    fn copy_state_rows(
        &mut self,
        ctx: &CudaContext,
        source: &Qwen35MtpState,
        destination: &mut Qwen35MtpState,
        first_row: usize,
    ) -> Result<()> {
        validate_state_storage(ctx, source, self.geometry)?;
        validate_state_storage(ctx, destination, self.geometry)?;
        let rows = source.position.map_or(0, |position| position.cache_rows);
        if rows > destination.capacity || first_row > rows {
            return Err(invalid(
                "Qwen3.5 MTP destination cache cannot hold the source prefix",
            ));
        }
        let generation = source
            .generation
            .checked_add(1)
            .ok_or_else(|| invalid("Qwen3.5 MTP shadow generation overflow"))?;
        destination.cache_revision = next_cache_revision()?;
        self.pending_draft = None;
        if first_row < rows {
            let view = |cache| {
                cache_rows_view(
                    cache,
                    first_row,
                    rows - first_row,
                    self.geometry.kv_heads,
                    self.geometry.head_dim,
                )
            };
            let source_key = view(&source.key)?;
            let destination_key = view(&destination.key)?;
            let source_value = view(&source.value)?;
            let destination_value = view(&destination.value)?;
            destination.position = None;
            destination.image_positions = None;
            transfers::copy_cuda_to_cuda(ctx, &source_key, &destination_key)?;
            transfers::copy_cuda_to_cuda(ctx, &source_value, &destination_value)?;
        }
        destination.position = source.position;
        destination.image_positions = source.image_positions.clone();
        destination.shortlist_allowed = source.shortlist_allowed;
        destination.hot_tile = source.hot_tile;
        destination.generation = generation;
        Ok(())
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
            image_positions: None,
            shortlist_allowed: false,
            hot_tile: None,
            generation: 0,
            cache_revision: next_cache_revision()?,
        })
    }

    fn reset_state(&mut self, ctx: &CudaContext, state: &mut Self::State) -> Result<()> {
        validate_state_storage(ctx, state, self.geometry)?;
        let generation = state
            .generation
            .checked_add(1)
            .ok_or_else(|| invalid("Qwen3.5 MTP state generation overflow"))?;
        state.cache_revision = next_cache_revision()?;
        state.position = None;
        state.image_positions = None;
        state.shortlist_allowed = false;
        state.hot_tile = None;
        self.pending_draft = None;
        self.synced_draft_prefix = None;
        CudaBuffer::from_tensor(&state.key)
            .map_err(Error::Cuda)?
            .zero_async(ctx.stream())
            .map_err(Error::Cuda)?;
        CudaBuffer::from_tensor(&state.value)
            .map_err(Error::Cuda)?
            .zero_async(ctx.stream())
            .map_err(Error::Cuda)?;
        state.generation = generation;
        Ok(())
    }

    fn copy_state(
        &mut self,
        ctx: &CudaContext,
        source: &Self::State,
        destination: &mut Self::State,
    ) -> Result<()> {
        self.synced_draft_prefix = None;
        self.copy_state_rows(ctx, source, destination, 0)
    }

    fn prepare_prefill(
        &mut self,
        ctx: &CudaContext,
        weights: &Qwen35DeviceWeights,
        shadow: &mut Self::State,
        prompt_tokens: &[TokenId],
        target_hidden: &Tensor,
    ) -> Result<Self::PreparedPrefill> {
        self.prepare_prompt(ctx, weights, shadow, prompt_tokens, target_hidden, None)
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
        self.prepare_append_prompt(
            ctx,
            weights,
            committed,
            shadow,
            chunk_tokens,
            previous_target_hidden,
            target_hidden,
            None,
        )
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
        self.synced_draft_prefix = None;
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
        // Reestablish the watermark only after a successful dual publication.
        let synced_prefix = self.synced_draft_prefix.take();
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
        if required_rows > committed.capacity || required_rows > shadow.capacity {
            return Err(invalid("Qwen3.5 MTP draft exceeds FP8 KV capacity"));
        }
        let first_row = synced_prefix.map_or(0, |prefix| {
            prefix.reusable_rows(
                committed.cache_revision,
                shadow.cache_revision,
                start.cache_rows,
            )
        });
        self.copy_state_rows(ctx, committed, shadow, first_row)?;
        // One out-of-set token selects the full head for this round. It does
        // not disable an eligible later round or change policy within depth3.
        let shortlist_active = shadow.shortlist_allowed && mtp_shortlist_contains(pending);
        if !mtp_shortlist_contains(pending) && (pending as usize) < VOCAB {
            shadow.hot_tile = Some((pending as usize) / MTP_HOT_TILE_ROWS);
        }

        if shortlist_active || self.prepared_shortlist_head.is_some() {
            // Normal prefill/append already owns these resources. Retain the
            // existing cold direct-draft and weight-rebinding preparation
            // behavior before entering any prepared execution session.
            self.prepare_shortlist_head(ctx, weights)?;
        }
        let prepared_shortlist_head = self.prepared_shortlist_head.clone();
        if shortlist_active {
            // Always stage a fixed 128-row tile before running the prepared
            // draft Graph. Tile zero duplicates prefix IDs until an outside
            // pending token has supplied a hot tile; the Graph node set never
            // changes between cold prepare and later replay.
            prepared_shortlist_head.as_ref().expect("prepared shortlist")
                .stage_hot_tile(ctx, weights, shadow.hot_tile.unwrap_or(0))?;
        }

        let geometry = self.geometry;
        let eps = self.rms_norm_eps;
        let theta = self.rope_theta;
        let scratch = self.ensure_scratch(ctx, committed.capacity)?;
        scratch.shortlist_head = prepared_shortlist_head;
        scratch.shortlist_active = shortlist_active;
        write_i32(ctx, &scratch.token, pending)?;
        let pending_slot = element_view(&scratch.verification_inputs, 0)?;
        transfers::copy_cuda_to_cuda(ctx, &scratch.token, &pending_slot)?;
        let draft_positions = if let Some(plan) = committed.image_positions.as_ref() {
            plan.token_major_values(start.logical_position, draft_steps)?
        } else {
            (0..draft_steps)
                .map(|depth| {
                    i32::try_from(start.rope_position_for_draft(depth)?)
                        .map_err(|_| invalid("MTP RoPE position exceeds i32"))
                })
                .collect::<Result<Vec<_>>>()?
                .into_iter()
                .flat_map(|position| [position; 3])
                .collect::<Vec<_>>()
        };
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
            committed_cache_revision: committed.cache_revision,
            shadow_cache_revision: shadow.cache_revision,
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
            || pending.shadow_cache_revision != shadow.cache_revision
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
            shadow.cache_revision = next_cache_revision()?;
            let logical = pending
                .start
                .logical_position
                .checked_add(pending.draft_steps)
                .ok_or_else(|| invalid("MTP final RoPE position overflow"))?;
            let actual_position = match shadow.image_positions.as_ref() {
                Some(plan) => u32::try_from(plan.values(logical, 1)?[0])
                    .map_err(|_| invalid("MTP final image position overflow"))?,
                None => {
                    u32::try_from(logical).map_err(|_| invalid("MTP final position exceeds u32"))?
                }
            };
            let geometry = self.geometry;
            let eps = self.rms_norm_eps;
            let theta = self.rope_theta;
            let kv_only = self.kv_only_materialization;
            let scratch = self.ensure_scratch(ctx, shadow.capacity)?;
            transfers::copy_cuda_to_cuda(ctx, &scratch.next_token, &scratch.token)?;
            write_text_position(ctx, &scratch.position, actual_position)?;
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
                kv_only,
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
            committed_cache_revision: pending.committed_cache_revision,
            shadow_cache_revision: shadow.cache_revision,
            preserved_rows: pending.start.cache_rows,
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
        // Draft and final materialization write only at or beyond the old
        // physical endpoint. After swapping, that old prefix still matches.
        self.synced_draft_prefix = (self.draft_kv_suffix_sync
            && committed.cache_revision == prepared.committed_cache_revision
            && shadow.cache_revision == prepared.shadow_cache_revision
            && committed.position.map(|position| position.cache_rows)
                == Some(prepared.preserved_rows)
            && shadow.position == Some(prepared.position))
        .then_some(DraftKvPrefix {
            source_revision: shadow.cache_revision,
            destination_revision: committed.cache_revision,
            rows: prepared.preserved_rows,
        });
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

    /// A complete prompt checkpoint already owns the final target hidden row
    /// and proposer state. Restart only request-local scheduling, without
    /// replaying a token or changing either committed cache.
    pub(super) fn restart_prefill_greedy(
        &mut self,
        first_token: TokenId,
        options: &GenerationOptions,
    ) -> Result<()> {
        let policy = MtpGreedyPolicy::from_generation_options(options)?;
        let position = self.target.position();
        if position == 0 || self.target_hidden.is_none() {
            return Err(invalid("complete prompt restart requires restored state"));
        }
        let schedule = Qwen35MtpSchedule::new(
            position, position, policy.max_new_tokens, 1, policy.eos_token_ids.clone(),
        ).map_err(schedule_error)?;
        self.schedule = Some(schedule);
        self.policy = Some(policy);
        self.pending = Some(first_token);
        Ok(())
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
        self.finish_prefill(
            ctx,
            prepared_target,
            prepared_mtp,
            prompt_tokens.len(),
            policy,
        )
    }

    fn finish_prefill(
        &mut self,
        ctx: &CudaContext,
        prepared_target: PreparedTargetPrefill,
        prepared_mtp: M::PreparedPrefill,
        prompt_len: usize,
        policy: MtpGreedyPolicy,
    ) -> Result<MtpPrefillOutcome> {
        let target_hidden = prepared_target.output().last_hidden_state()?;
        let first = prepared_target.output().greedy_token_ids(ctx)?;
        let first_token = *first
            .first()
            .ok_or_else(|| invalid("Qwen3.5 target prefill produced no token"))?;
        let schedule = Qwen35MtpSchedule::new(
            prompt_len,
            prompt_len,
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
        let proposals = &resolution.proposals;
        debug_assert_eq!(proposals.len(), draft.draft_steps());
        let ticket = schedule
            .begin_round(pending, proposals)
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
impl<B: FullAttentionBackend> Qwen35MtpCoordinator<B, Qwen35MtpRuntime> {
    pub(super) fn append_image_prefill_greedy(
        &mut self,
        ctx: &CudaContext,
        decoder: &Qwen35Decoder,
        weights: &Qwen35DeviceWeights,
        tokens: &[TokenId],
        embeddings: &Tensor,
        positions: Rc<ImagePositions>,
        options: &GenerationOptions,
    ) -> Result<MtpPrefillOutcome> {
        let policy = MtpGreedyPolicy::from_generation_options(options)?;
        let previous = self
            .target_hidden
            .as_ref()
            .ok_or_else(|| invalid("image append requires prepared target state"))?
            .clone();
        let start = self.target.position();
        if start == 0 {
            return Err(invalid("image append requires a nonempty prefix"));
        }
        let target = self
            .target
            .prepare_target_image_prefill(ctx, decoder, weights, tokens, embeddings, start)?;
        let mtp = self.mtp.prepare_append_prompt(
            ctx,
            weights,
            &self.committed_mtp,
            &mut self.shadow_mtp,
            tokens,
            &previous,
            &target.output().hidden_states,
            Some((embeddings, positions)),
        )?;
        let end = target.output().end_position;
        self.finish_prefill(ctx, target, mtp, end, policy)
    }

    pub(super) fn prefill_image_greedy(
        &mut self,
        ctx: &CudaContext,
        decoder: &Qwen35Decoder,
        weights: &Qwen35DeviceWeights,
        tokens: &[TokenId],
        embeddings: &Tensor,
        positions: Rc<ImagePositions>,
        options: &GenerationOptions,
    ) -> Result<MtpPrefillOutcome> {
        let policy = MtpGreedyPolicy::from_generation_options(options)?;
        let target = self
            .target
            .prepare_target_image_prefill(ctx, decoder, weights, tokens, embeddings, 0)?;
        let mtp = self.mtp.prepare_prompt(
            ctx,
            weights,
            &mut self.shadow_mtp,
            tokens,
            &target.output().hidden_states,
            Some((embeddings, positions)),
        )?;
        self.finish_prefill(ctx, target, mtp, tokens.len(), policy)
    }
}

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
    embedding_override: Option<&Tensor>,
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
    let multimodal = positions.shape().dims() == [rows, 3];
    let position_shape = if multimodal {
        vec![rows, 3]
    } else {
        vec![rows]
    };
    validate_tensor(
        positions,
        ctx,
        DType::I32,
        &position_shape,
        "MTP prefill positions",
    )?;

    let embedding = rows_view(&scratch.embedding, 0, rows, HIDDEN)?;
    let norm_embedding = rows_view(&scratch.norm_embedding, 0, rows, HIDDEN)?;
    let norm_hidden = rows_view(&scratch.norm_hidden, 0, rows, HIDDEN)?;
    let joined = rows_view(&scratch.joined, 0, rows, 2 * HIDDEN)?;
    let mut hidden = rows_view(&scratch.hidden, 0, rows, HIDDEN)?;
    let normalized = rows_view(&scratch.normalized, 0, rows, HIDDEN)?;
    let mtp = &weights.mtp;

    if let Some(source) = embedding_override {
        validate_tensor(
            source,
            ctx,
            DType::BF16,
            &[rows, HIDDEN],
            "MTP shifted image embeddings",
        )?;
        ops::copy_tensor(ctx, source, &embedding)?;
    } else {
        ops::embedding_gather(ctx, &weights.embedding, tokens, &embedding)?;
    }
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
    if multimodal {
        let mut key = key_heads.clone();
        ops::multimodal_rope(
            ctx,
            ops::MultimodalRopeArgs {
                data: &mut key,
                positions,
                rotary_dim: geometry.rotary_dim,
                sections: [11, 11, 10],
                theta: rope_theta,
            },
        )?;
    } else {
        ops::partial_rope(ctx, &key_heads, positions, geometry.rotary_dim, rope_theta)?;
    }

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
    ops::quantize_activation_per_tensor(ctx, &key_flat, &key_slot, MTP_KV_SCALE)?;
    ops::quantize_activation_per_tensor(ctx, &value_flat, &value_slot, MTP_KV_SCALE)
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
    fn new(ctx: &CudaContext, weights: &Qwen35DeviceWeights, kv_only: bool) -> Result<Self> {
        let mtp = &weights.mtp;
        let attention = &mtp.layer.attention;
        let moe = &mtp.layer.moe;
        let operands = [
            &mtp.fc.weight,
            &attention.k_proj.weight,
            &attention.v_proj.weight,
            &attention.q_proj.weight,
            &attention.o_proj.weight,
            &moe.router.weight,
            &moe.shared_gate.weight,
            &moe.shared_up.weight,
            &moe.shared_down.weight,
            &moe.shared_expert_gate.weight,
        ];
        // Reference MoE is one stateless raw launch. Edge MoE adds five
        // GEMMs with stable views into the same MtpScratch allocations.
        let gemm_count = if kv_only {
            3
        } else if matches!(&moe.routed_experts, DeviceDenseRoutedExperts::Edge(_)) {
            10
        } else {
            5
        };
        let mut gemm_weights = [0; 10];
        for (slot, tensor) in gemm_weights[..gemm_count].iter_mut().zip(&operands) {
            *slot = tensor
                .storage()
                .as_gpu()
                .map(|storage| storage.ptr())
                .ok_or_else(|| invalid("MTP materialization weight has no CUDA storage"))?;
        }
        Ok(Self {
            device: ctx.device_id(),
            stream: ctx.stream().handle() as usize,
            deterministic: std::env::var_os("APXINF_GEMM_DETERMINISTIC").is_some(),
            gemm_count,
            gemm_weights,
        })
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
    draft_attention_workspace: Option<Tensor>,
    projected: Tensor,
    moe_out: Tensor,
    final_hidden: Tensor,
    shortlist_head: Option<Rc<MtpShortlistHead>>,
    shortlist_active: bool,
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
    fn new(
        ctx: &CudaContext,
        geometry: AttentionGeometry,
        cache_capacity: usize,
        auto_split_kv: bool,
    ) -> Result<Self> {
        if cache_capacity == 0 || cache_capacity % ops::QWEN35_KV_CACHE_BLOCK_TOKENS != 0 {
            return Err(invalid("Qwen3.5 MTP scratch capacity is not page aligned"));
        }
        let draft_attention_workspace = if split_profile_or_legacy(
            auto_split_kv,
            ctx.caps().sm,
            || std::env::var("APXINF_QWEN35_MTP_SPLIT_KV").ok(),
        ) {
            let elements = ops::qwen35_draft_attention_workspace_elements(cache_capacity)?;
            Some(zeros(ctx, vec![elements], DType::F32)?)
        } else {
            None
        };
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
            draft_attention_workspace,
            projected: zeros(ctx, vec![1, HIDDEN], DType::BF16)?,
            moe_out: zeros(ctx, vec![1, HIDDEN], DType::BF16)?,
            final_hidden: zeros(ctx, vec![1, HIDDEN], DType::BF16)?,
            shortlist_head: None,
            shortlist_active: false,
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
            edge_workspace: ops::small_routed_moe_available()
                .then(|| {
                    let bytes = ops::small_routed_moe_workspace_size(
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
    kv_only: bool,
) -> Result<()> {
    let enabled = std::env::var("APXINF_QWEN35_MTP_MATERIALIZE_SESSION").as_deref() == Ok("1");
    let session = if enabled {
        let binding = MtpMaterializeBinding::new(ctx, weights, kv_only)?;
        if scratch
            .materialize_session
            .as_ref()
            .is_none_or(|cached| cached.binding != binding)
        {
            scratch.materialize_session = Some(MtpMaterializeSession {
                binding,
                session: ExecutionSession::with_capacity(MTP_DRAFT_WORKSPACE, ctx.device_id())?,
                prepared: false,
            });
        }
        let cached = scratch
            .materialize_session
            .as_ref()
            .expect("initialized materialization session");
        Some((cached.session.clone(), cached.prepared))
    } else {
        None
    };
    // prepare executes this real row once; never prepare and then execute the
    // same materialization twice. Unlike draft, this tape has no LM-head GEMM.
    let mut forward = || {
        if !kv_only {
            return forward_one(
                ctx,
                weights,
                state,
                scratch,
                geometry,
                eps,
                rope_theta,
                base_hidden,
                token,
                position,
                cache_row,
                draft_step,
                false,
            );
        }

        // Only this proposal's K/V and cache position survive publication.
        // Use the same M=1 FC and K/V projections as the normal draft step.
        project_mtp_input(ctx, weights, scratch, eps, base_hidden, token)?;
        project_mtp_kv(ctx, &weights.mtp, scratch)?;
        ops::qwen35_static_kv_append(
            ctx,
            1,
            ops::Qwen35StaticFp8KvAppendArgs {
                projected_key: &scratch.projected_key,
                projected_value: &scratch.projected_value,
                key_norm: &weights.mtp.layer.attention.k_norm,
                positions: position,
                cache: ops::Qwen35Fp8KvCache {
                    key: &state.key,
                    value: &state.value,
                },
                cache_offset: cache_row,
                scales: ops::Qwen35KvCacheScales {
                    key: MTP_KV_SCALE,
                    value: MTP_KV_SCALE,
                },
                rope_theta,
            },
        )
    };
    let result = match session.as_ref() {
        Some((execution, true)) => execution.run(forward),
        Some((execution, false)) => execution.prepare(forward),
        None => forward(),
    };
    if enabled {
        scratch
            .materialize_session
            .as_mut()
            .expect("initialized materialization session")
            .prepared = result.is_ok();
    }
    if (enabled || kv_only) && result.is_err() {
        // A failed submission may have left writes to the shadow in flight.
        // Drain them before returning without publishing the transaction.
        ctx.synchronize().map_err(Error::Cuda)?;
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

fn project_mtp_input(
    ctx: &CudaContext,
    weights: &Qwen35DeviceWeights,
    scratch: &MtpScratch,
    eps: f32,
    base_hidden: &Tensor,
    token: &Tensor,
) -> Result<()> {
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
    )
}

fn project_mtp_kv(ctx: &CudaContext, mtp: &DeviceMtpWeights, scratch: &MtpScratch) -> Result<()> {
    let attention = &mtp.layer.attention;
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
    )
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
    project_mtp_input(ctx, weights, scratch, eps, base_hidden, token)?;
    let mtp = &weights.mtp;
    let attention = &mtp.layer.attention;
    let mut q_fused = scratch.q_fused.clone();
    ops::gemm(
        ctx,
        ops::GemmArgs::new(&scratch.normalized, &attention.q_proj.weight, &mut q_fused),
    )?;
    project_mtp_kv(ctx, mtp, scratch)?;
    let attention_out = reshape(
        &scratch.attention_out,
        vec![1, geometry.query_heads, geometry.head_dim],
        DType::BF16,
    )?;
    let attention_args = ops::Qwen35Fp8AttentionArgs {
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
    };
    if let Some(workspace) = &scratch.draft_attention_workspace {
        ops::qwen35_attention_draft_cache(ctx, attention_args, workspace)?;
    } else {
        ops::qwen35_attention_static_cache(ctx, 1, attention_args)?;
    }
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
    if produce_logits && scratch.shortlist_active {
        scratch.shortlist_head.as_ref().expect("prepared shortlist")
            .project(ctx, &scratch.final_hidden, &scratch.next_token)?;
    } else if produce_logits {
        let mut logits = scratch.logits.clone();
        super::forward::project_lm_head(ctx, weights, &scratch.final_hidden, &mut logits)?;
        ops::argmax(
            ctx,
            ops::ArgmaxArgs::new(&scratch.logits, &scratch.next_token),
        )?;
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
    ops::small_routed_moe(
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
    fn qualified_mtp_policy_cannot_be_disabled_by_legacy_zero() {
        let reads = std::cell::Cell::new(0);
        assert!(profile_or_legacy(true, || {
            reads.set(reads.get() + 1);
            Some("0".into())
        }));
        assert_eq!(reads.get(), 0, "qualified policy must not consult legacy ENV");
    }

    #[test]
    fn unqualified_mtp_policy_reads_legacy_value_once_without_env_mutation() {
        for (value, expected) in [
            (None, false), (Some("0"), false), (Some("1"), true),
            (Some(""), false), (Some("true"), false), (Some("invalid"), false),
        ] {
            let reads = std::cell::Cell::new(0);
            assert_eq!(profile_or_legacy(false, || {
                reads.set(reads.get() + 1);
                value.map(str::to_owned)
            }), expected);
            assert_eq!(reads.get(), 1);
        }
    }

    #[test]
    fn mtp_split_profile_retains_sm_guard_and_deferred_legacy_read() {
        for sm in [80u32, 89, 100, 101, 110, 120] {
            let supported = matches!(sm, 101 | 110);
            assert_eq!(split_profile_or_legacy(true, sm, || {
                panic!("qualified split policy must not consult legacy ENV")
            }), supported);
            let reads = std::cell::Cell::new(0);
            assert_eq!(split_profile_or_legacy(false, sm, || {
                reads.set(reads.get() + 1);
                Some("1".into())
            }), supported);
            assert_eq!(reads.get(), 1, "legacy read still precedes the SM guard");
            assert!(!split_profile_or_legacy(false, sm, || Some("0".into())));
        }
    }

    #[test]
    fn draft_kv_prefix_requires_both_bank_revisions_and_a_valid_row_count() {
        let prefix = DraftKvPrefix {
            source_revision: next_cache_revision().unwrap(),
            destination_revision: next_cache_revision().unwrap(),
            rows: 127,
        };
        let source = prefix.source_revision;
        let destination = prefix.destination_revision;
        assert_eq!(prefix.reusable_rows(source, destination, 131), 127);
        assert_eq!(prefix.reusable_rows(source, destination, 126), 0);
        assert_eq!(prefix.reusable_rows(destination, source, 131), 0);
        assert_eq!(
            prefix.reusable_rows(next_cache_revision().unwrap(), destination, 131),
            0
        );
        assert_eq!(
            prefix.reusable_rows(source, next_cache_revision().unwrap(), 131),
            0
        );
    }

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
    use crate::qwen35moe::device_weights::load_qwen35_device_weights_from_path_with_options;
    use crate::qwen35moe::mtp_schedule::ShadowCommitRange;
    use crate::qwen35moe::{
        Qwen35ExecutionOptions, Qwen35Fp8AttentionBackend, Qwen35KvCacheDType, Qwen35MoeActivation,
    };
    use std::path::PathBuf;

    fn parse_protocol_graphs(
        value: Option<&std::ffi::OsStr>,
    ) -> std::result::Result<bool, &'static str> {
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
        fn load_with_autotune(
            allow_online_tune: bool,
        ) -> std::result::Result<Self, Box<dyn std::error::Error>> {
            let protocol_graphs_env = std::env::var_os("APXINF_QWEN35_PROTOCOL_GRAPHS");
            let graphs = parse_protocol_graphs(protocol_graphs_env.as_deref())?;
            std::env::set_var(
                "APXINF_REQUIRE_RECIPES",
                if allow_online_tune { "0" } else { "1" },
            );
            std::env::set_var("APXINF_QWEN35_CUDA_GRAPHS", if graphs { "1" } else { "0" });
            std::env::set_var("APXINF_QWEN35_VERIFY_GRAPHS", "0");
            let checkpoint = PathBuf::from(
                std::env::var_os("APXINF_QWEN35_CHECKPOINT").expect("checkpoint env"),
            );
            let tuning = PathBuf::from(
                std::env::var_os("APXINF_QWEN35_PARITY_TUNING_DIR").expect("pre-tuned recipe env"),
            );
            assert!(checkpoint.is_absolute() && tuning.is_absolute());
            if allow_online_tune {
                std::fs::create_dir_all(&tuning)?;
            } else {
                assert!(std::fs::read_dir(&tuning)?
                    .filter_map(std::result::Result::ok)
                    .any(|x| x.path().extension().is_some_and(|x| x == "recipe")));
            }
            let ctx = CudaContext::new_with_options(0, allow_online_tune, Some(&tuning))?;
            let config = Qwen35MoeConfig::from_json_file(&checkpoint.join("config.json"))?;
            let policy = Qwen35ExecutionOptions {
                moe_activation: Qwen35MoeActivation::Nvfp4,
                kv_cache: Qwen35KvCacheDType::Bf16,
            };
            let weights = load_qwen35_device_weights_from_path_with_options(
                &ctx,
                &config,
                &checkpoint,
                policy,
            )?;
            let decoder = Qwen35Decoder::new(&config)?;
            // Short fixed scope. Existing1024-token and long-context gates are separate.
            let prompt = vec![
                151644, 8948, 198, 9707, 16357, 220, 19687, 25, 198, 151645, 198, 151644, 77091,
                198, 35064, 16933,
            ];
            Ok(Self {
                ctx,
                config,
                decoder,
                weights,
                prompt,
                capacity: 64,
                graphs,
            })
        }
        pub(in crate::qwen35moe) fn target(
            &self,
        ) -> Result<Qwen35Session<Qwen35Fp8AttentionBackend>> {
            let backend = Qwen35Fp8AttentionBackend::with_kv_cache(
                &self.config,
                Qwen35KvCacheDType::Bf16,
                None,
            )?;
            Qwen35Session::new(
                &self.ctx,
                &self.decoder,
                &self.weights,
                backend,
                self.capacity,
            )
        }
    }

    pub(in crate::qwen35moe) fn raw(tensor: &Tensor) -> Result<Vec<u8>> {
        let cpu = transfers::to_cpu(tensor)?;
        if cpu.dtype() == DType::F32 {
            assert!(cpu.as_f32()?.iter().all(|x| x.is_finite()));
        }
        if cpu.dtype() == DType::BF16 {
            assert!(cpu.as_bf16()?.iter().all(|x| x.to_f32().is_finite()));
        }
        let bytes = cpu.storage().as_cpu().unwrap();
        if cpu.dtype() == DType::F8E4M3 {
            assert!(
                bytes.iter().all(|x| (*x & 0x7f) != 0x7f),
                "nonfinite E4M3 observation"
            );
        }
        Ok(bytes.to_vec())
    }
    pub(in crate::qwen35moe) fn digest_images(images: &[(Vec<u8>, Vec<u8>)]) -> String {
        let hash = images
            .iter()
            .flat_map(|x| x.0.iter().chain(&x.1))
            .fold(0xcbf29ce484222325u64, |h, b| {
                (h ^ u64::from(*b)).wrapping_mul(0x100000001b3)
            });
        format!("{hash:016x}")
    }
    pub(in crate::qwen35moe) fn exact(label: &str, actual: &[u8], expected: &[u8]) {
        assert_eq!(actual.len(), expected.len(), "{label}: length");
        if let Some(i) = actual.iter().zip(expected).position(|(a, b)| a != b) {
            panic!(
                "{label}: first byte mismatch {i}: {} != {}",
                actual[i], expected[i]
            );
        }
    }
    pub(in crate::qwen35moe) fn pointer(t: &Tensor) -> usize {
        t.storage().as_gpu().unwrap().ptr()
    }
    pub(in crate::qwen35moe) fn fill_rows(
        ctx: &CudaContext,
        t: &Tensor,
        start: usize,
        end: usize,
        rows: usize,
        byte: u8,
    ) -> Result<()> {
        let b = CudaBuffer::from_tensor(t).map_err(Error::Cuda)?;
        assert_eq!(b.len() % rows, 0);
        assert!(start <= end && end <= rows);
        b.view(start * (b.len() / rows), (end - start) * (b.len() / rows))
            .map_err(Error::Cuda)?
            .fill_async(byte, ctx.stream())
            .map_err(Error::Cuda)
    }
    // CPU copy ledger, independent of CUDA cache view/copy implementations.
    pub(in crate::qwen35moe) fn prefix_image(
        destination: &[u8],
        source: &[u8],
        row_bytes: usize,
        count: usize,
    ) -> Vec<u8> {
        let end = row_bytes.checked_mul(count).unwrap();
        assert!(end <= destination.len() && end <= source.len());
        let mut expected = destination.to_vec();
        expected[..end].copy_from_slice(&source[..end]);
        expected
    }
    #[test]
    fn cpu_draft_copy_ledger_counts_physical_p_minus_one_rows() {
        let source: Vec<u8> = (0..40u8).collect();
        let destination = vec![0x5au8; 64];
        let expected = prefix_image(&destination, &source, 2, 15);
        assert_eq!(&expected[..30], &source[..30]);
        assert_eq!(&expected[30..], &destination[30..]);
        assert_eq!(prefix_image(&destination, &source, 2, 0), destination);
        assert_ne!(
            expected,
            prefix_image(&destination, &source, 2, 16),
            "logicalP16 must not copy physical16rows"
        );
    }
    fn image(state: &Qwen35MtpState) -> Result<(Vec<u8>, Vec<u8>)> {
        Ok((raw(&state.key)?, raw(&state.value)?))
    }
    fn same(label: &str, state: &Qwen35MtpState, bytes: &(Vec<u8>, Vec<u8>)) -> Result<()> {
        exact(&format!("{label} K"), &raw(&state.key)?, &bytes.0);
        exact(&format!("{label} V"), &raw(&state.value)?, &bytes.1);
        Ok(())
    }
    fn ids(state: &Qwen35MtpState) -> (usize, usize) {
        (pointer(&state.key), pointer(&state.value))
    }
    fn check_copy(
        f: &Fixture,
        runtime: &mut Qwen35MtpRuntime,
        source: &Qwen35MtpState,
        destination: &mut Qwen35MtpState,
    ) -> Result<()> {
        let before = image(destination)?;
        let source_before = image(source)?;
        let count = source.position.map(|p| p.cache_rows).unwrap_or(0);
        let row_bytes = source_before.0.len() / source.capacity;
        let expected = (
            prefix_image(&before.0, &source_before.0, row_bytes, count),
            prefix_image(&before.1, &source_before.1, row_bytes, count),
        );
        runtime.copy_state(&f.ctx, source, destination)?;
        f.ctx.synchronize().map_err(Error::Cuda)?;
        same(
            "copy destination independent CPU prefix ledger",
            destination,
            &expected,
        )?;
        same("copy source readonly", source, &source_before)?;
        assert_eq!(destination.position, source.position);
        assert_eq!(destination.shortlist_allowed,source.shortlist_allowed);
        assert_eq!(destination.hot_tile,source.hot_tile);
        assert_eq!(destination.generation, source.generation + 1);
        assert!(runtime.pending_draft.is_none());
        Ok(())
    }

    fn draft_gate(f: &Fixture, pending_override: Option<u32>) -> Result<()> {
        let mut target = f.target()?;
        let prepared_target =
            target.prepare_target_prefill(&f.ctx, &f.decoder, &f.weights, &f.prompt)?;
        let hidden_rows = prepared_target.output().hidden_states.clone();
        let last_hidden = prepared_target.output().last_hidden_state()?;
        let pending = pending_override.unwrap_or(prepared_target.output().greedy_token_ids(&f.ctx)?[0]);
        target.publish_target_prefill(prepared_target)?;
        let mut runtime = Qwen35MtpRuntime::new(&f.config)?;
        let mut committed = runtime.allocate_state(&f.ctx, f.capacity)?;
        let mut shadow = runtime.allocate_state(&f.ctx, f.capacity)?;
        let mut snapshot = runtime.allocate_state(&f.ctx, f.capacity + 128)?;
        assert_eq!(committed.capacity, 128);
        assert_eq!(shadow.capacity, 128);
        assert_eq!(snapshot.capacity, 256);
        assert_ne!(ids(&committed), ids(&shadow));
        assert_ne!(ids(&committed), ids(&snapshot));
        assert_ne!(ids(&shadow), ids(&snapshot));
        for state in [&committed, &shadow, &snapshot] {
            assert_eq!(
                state.key.shape().dims(),
                [
                    state.capacity / 128,
                    128,
                    runtime.geometry.kv_heads,
                    runtime.geometry.head_dim
                ]
            );
            assert_eq!(state.key.dtype(), DType::F8E4M3);
            assert_eq!(state.value.shape(), state.key.shape());
        }
        runtime.reset_state(&f.ctx, &mut committed)?;
        runtime.reset_state(&f.ctx, &mut shadow)?;
        f.ctx.synchronize().map_err(Error::Cuda)?;
        let empty = image(&committed)?;
        assert!(empty.0.iter().chain(&empty.1).all(|x| *x == 0));
        fill_rows(
            &f.ctx,
            &snapshot.key,
            0,
            snapshot.capacity,
            snapshot.capacity,
            0x4a,
        )?;
        fill_rows(
            &f.ctx,
            &snapshot.value,
            0,
            snapshot.capacity,
            snapshot.capacity,
            0x4a,
        )?;
        check_copy(f, &mut runtime, &committed, &mut snapshot)?; // None means no bytes copied.
        let committed_before = image(&committed)?;
        let committed_meta = (committed.position, committed.generation);
        let hidden_before = raw(&hidden_rows)?;
        let prepared =
            runtime.prepare_prefill(&f.ctx, &f.weights, &mut shadow, &f.prompt, &hidden_rows)?;
        same(
            "prefill must not mutate committed",
            &committed,
            &committed_before,
        )?;
        assert_eq!((committed.position, committed.generation), committed_meta);
        exact(
            "teacher forcing hidden input readonly",
            &raw(&hidden_rows)?,
            &hidden_before,
        );
        assert_eq!(
            prepared.position,
            MtpPositionContract::after_prompt(f.prompt.len())?
        );
        assert_eq!(shadow.position, Some(prepared.position));
        assert_eq!(shadow.generation, prepared.generation);
        let rows = f.prompt.len() - 1;
        let row_bytes = raw(&shadow.key)?.len() / shadow.capacity;
        // prepare_prefill declares a full reset, so its unused tail must be zero.
        for tensor in [&shadow.key, &shadow.value] {
            assert!(raw(tensor)?[rows * row_bytes..].iter().all(|x| *x == 0));
        }
        let staged = image(&shadow)?;
        let staged_ids = ids(&shadow);
        let old_ids = ids(&committed);
        runtime.publish_prefill(&mut committed, &mut shadow, prepared);
        assert_eq!(ids(&committed), staged_ids);
        assert_eq!(ids(&shadow), old_ids);
        same("prefill publication exact swap", &committed, &staged)?;
        check_copy(f, &mut runtime, &committed, &mut snapshot)?;
        let frozen = image(&snapshot)?;
        let frozen_meta = (snapshot.position, snapshot.generation, snapshot.shortlist_allowed, snapshot.hot_tile);
        for depth in 1..=3usize {
            for keep in 1..=depth + 1 {
                check_copy(f, &mut runtime, &snapshot, &mut committed)?;
                assert_eq!(
                    committed.position,
                    Some(MtpPositionContract::after_prompt(f.prompt.len())?)
                );
                for tensor in [&committed.key, &committed.value] {
                    fill_rows(
                        &f.ctx,
                        tensor,
                        rows,
                        committed.capacity,
                        committed.capacity,
                        0x3a,
                    )?;
                }
                for tensor in [&shadow.key, &shadow.value] {
                    fill_rows(&f.ctx, tensor, 0, shadow.capacity, shadow.capacity, 0x5a)?;
                }
                let old = image(&committed)?;
                let old_meta = (committed.position, committed.generation);
                let old_ids = ids(&committed);
                let _draft = runtime.draft(
                    &f.ctx,
                    &f.weights,
                    &committed,
                    &mut shadow,
                    pending,
                    f.prompt.len(),
                    &last_hidden,
                    depth,
                )?;
                f.ctx.synchronize().map_err(Error::Cuda)?;
                if pending_override.is_some() {
                    assert!(runtime.scratch.as_ref().unwrap().shortlist_active);
                    assert!(committed.shortlist_allowed && shadow.shortlist_allowed);
                }
                same(
                    "draft committed full physical bank readonly",
                    &committed,
                    &old,
                )?;
                assert_eq!((committed.position, committed.generation), old_meta);
                let staged_metadata = runtime
                    .pending_draft
                    .expect("real draft transaction must be pending");
                assert_eq!(staged_metadata.draft_steps, depth);
                assert_eq!(staged_metadata.shadow_generation, shadow.generation);
                assert_eq!(
                    staged_metadata.start,
                    MtpPositionContract::after_prompt(f.prompt.len())?
                );
                let after_draft = image(&shadow)?;
                exact(
                    "draft old K prefix",
                    &after_draft.0[..rows * row_bytes],
                    &old.0[..rows * row_bytes],
                );
                exact(
                    "draft old V prefix",
                    &after_draft.1[..rows * row_bytes],
                    &old.1[..rows * row_bytes],
                );
                assert!(after_draft.0[(rows + depth) * row_bytes..]
                    .iter()
                    .chain(&after_draft.1[(rows + depth) * row_bytes..])
                    .all(|x| *x == 0x5a));
                let full = keep == depth + 1;
                let contract = MtpCommit {
                    cache_range: ShadowCommitRange {
                        start: f.prompt.len(),
                        committed_end: f.prompt.len() + keep,
                        verified_end: f.prompt.len() + if full { depth + 1 } else { depth },
                    },
                    materialize_final_proposal: full,
                };
                let prepared = runtime.prepare_commit(&f.ctx, &f.weights, &mut shadow, contract)?;
                let after_prepare = image(&shadow)?;
                // Full commit may materialize exactly one final physical row;
                // partial commit only changes metadata, never clears its tail.
                for (name, a, b) in [
                    ("K", &after_prepare.0, &after_draft.0),
                    ("V", &after_prepare.1, &after_draft.1),
                ] {
                    if full {
                        exact(
                            &format!("full materialization {name} old prefix"),
                            &a[..(rows + depth) * row_bytes],
                            &b[..(rows + depth) * row_bytes],
                        );
                        exact(
                            &format!("full materialization {name} future canary"),
                            &a[(rows + depth + 1) * row_bytes..],
                            &b[(rows + depth + 1) * row_bytes..],
                        );
                    } else {
                        exact(&format!("partial commit {name} all bytes unchanged"), a, b);
                    }
                }
                same("prepare commit committed bank readonly", &committed, &old)?;
                assert_eq!((committed.position, committed.generation), old_meta);
                let expected_position = MtpPositionContract {
                    logical_position: f.prompt.len() + keep,
                    cache_rows: rows + keep,
                };
                assert_eq!(shadow.position, Some(expected_position));
                assert_eq!(prepared.position, expected_position);
                let staged_ids = ids(&shadow);
                let generation = shadow.generation;
                assert_eq!(prepared.generation, generation);
                runtime.publish_commit(&mut committed, &mut shadow, prepared);
                assert!(runtime.pending_draft.is_none());
                assert_eq!(ids(&committed), staged_ids);
                assert_eq!(ids(&shadow), old_ids);
                same(
                    "commit publish exact physical swap",
                    &committed,
                    &after_prepare,
                )?;
                same("old physical bank unchanged", &shadow, &old)?;
                assert_eq!(committed.position, Some(expected_position));
                assert_eq!(committed.generation, generation);
                same("MTP snapshot deepfreeze", &snapshot, &frozen)?;
                assert_eq!((snapshot.position, snapshot.generation, snapshot.shortlist_allowed, snapshot.hot_tile), frozen_meta);
                if runtime.draft_kv_suffix_sync && depth == 3 && full {
                    let prefix = runtime.synced_draft_prefix.expect("published draft prefix");
                    let new_rows = expected_position.cache_rows;
                    assert_eq!(
                        prefix.reusable_rows(
                            committed.cache_revision,
                            shadow.cache_revision,
                            new_rows,
                        ),
                        rows,
                    );
                    for tensor in [&shadow.key, &shadow.value] {
                        fill_rows(&f.ctx, tensor, rows, shadow.capacity, shadow.capacity, 0x5a)?;
                    }
                    // This fails after the suffix copy and before a proposal is
                    // written. The pending transaction and reuse watermark must
                    // both be absent, so the next attempt copies the full prefix.
                    assert!(runtime
                        .draft(
                            &f.ctx,
                            &f.weights,
                            &committed,
                            &mut shadow,
                            u32::MAX,
                            expected_position.logical_position,
                            &last_hidden,
                            depth,
                        )
                        .is_err());
                    f.ctx.synchronize().map_err(Error::Cuda)?;
                    assert!(runtime.synced_draft_prefix.is_none());
                    assert!(runtime.pending_draft.is_none());
                    let failed = image(&shadow)?;
                    for (actual, expected) in
                        [(&failed.0, &after_prepare.0), (&failed.1, &after_prepare.1)]
                    {
                        exact(
                            "draft suffix copy",
                            &actual[..new_rows * row_bytes],
                            &expected[..new_rows * row_bytes],
                        );
                        assert!(actual[new_rows * row_bytes..]
                            .iter()
                            .all(|byte| *byte == 0x5a));
                    }

                    for tensor in [&shadow.key, &shadow.value] {
                        fill_rows(&f.ctx, tensor, 0, 1, shadow.capacity, 0x5a)?;
                    }
                    runtime.draft(
                        &f.ctx,
                        &f.weights,
                        &committed,
                        &mut shadow,
                        pending,
                        expected_position.logical_position,
                        &last_hidden,
                        depth,
                    )?;
                    f.ctx.synchronize().map_err(Error::Cuda)?;
                    let retry = image(&shadow)?;
                    for (actual, expected) in
                        [(&retry.0, &after_prepare.0), (&retry.1, &after_prepare.1)]
                    {
                        exact(
                            "draft retry copies full prefix",
                            &actual[..new_rows * row_bytes],
                            &expected[..new_rows * row_bytes],
                        );
                    }
                    same(
                        "draft retry committed bank readonly",
                        &committed,
                        &after_prepare,
                    )?;
                }
                check_copy(f, &mut runtime, &snapshot, &mut committed)?; // Short restore must preserve unreachable tails.
                println!(
                    "{}",
                    serde_json::json!({"schema":"apxinf.qwen35.draft_raw_bank.v1","depth":depth,"keep":keep,
                    "logical_position":expected_position.logical_position,"physical_cache_rows":expected_position.cache_rows,
                    "full_materialization":full,"copy_protocol_exact":true,
                    "published_source_digest":digest_images(std::slice::from_ref(&after_prepare)),
                    "snapshot_digest":digest_images(std::slice::from_ref(&frozen)),
                    "independent_gdn_math_oracle":false,"performance_accepted":false})
                );
            }
        }
        // Append genuine target hidden rows after the restored short prefix.
        let suffix = [1234u32, 5678];
        let target_append =
            target.prepare_target_append_prefill(&f.ctx, &f.decoder, &f.weights, &suffix)?;
        for tensor in [&shadow.key, &shadow.value] {
            fill_rows(&f.ctx, tensor, 0, shadow.capacity, shadow.capacity, 0x5a)?;
        }
        let before = image(&shadow)?;
        let old = image(&committed)?;
        let prepared = runtime.prepare_append_prefill(
            &f.ctx,
            &f.weights,
            &committed,
            &mut shadow,
            &suffix,
            &last_hidden,
            &target_append.output().hidden_states,
        )?;
        let after = image(&shadow)?;
        for (a, b) in [(&after.0, &before.0), (&after.1, &before.1)] {
            exact(
                "append future tail unchanged",
                &a[(rows + suffix.len()) * row_bytes..],
                &b[(rows + suffix.len()) * row_bytes..],
            );
        }
        exact(
            "append copied old K prefix",
            &after.0[..rows * row_bytes],
            &old.0[..rows * row_bytes],
        );
        exact(
            "append copied old V prefix",
            &after.1[..rows * row_bytes],
            &old.1[..rows * row_bytes],
        );
        same("append committed readonly", &committed, &old)?;
        runtime.publish_prefill(&mut committed, &mut shadow, prepared);
        assert_eq!(
            committed.position,
            Some(MtpPositionContract {
                logical_position: f.prompt.len() + suffix.len(),
                cache_rows: rows + suffix.len()
            })
        );
        same("append publication bytes", &committed, &after)?;
        same("snapshot still deepfrozen", &snapshot, &frozen)?;
        check_copy(f, &mut runtime, &snapshot, &mut committed)?;
        Ok(())
    }

    #[test]
    #[ignore = "real checkpoint raw target/draft GPU bank gate; single weights load; exclusive lock and pre-tuned strict recipes"]
    fn real_checkpoint_target_and_draft_raw_bank_protocol(
    ) -> std::result::Result<(), Box<dyn std::error::Error>> {
        let f = Fixture::load()?;
        println!(
            "{}",
            serde_json::json!({"schema":"apxinf.qwen35.raw_bank_config.v1","prompt_tokens":16,
            "capacity":64,"block_tokens":128,"target_kv":"bf16","draft_kv":"fp8","draft_scale":1.0,
            "depths":[1,2,3],"target_commit_cases":9,"draft_commit_cases":9,"weights_loads":1,
            "strict_recipe_required":true,"autotune":false,"ordered_reduce":true,"graphs":f.graphs,"graph_gdn":std::env::var("APXINF_QWEN35_GRAPH_GDN").ok(),
            "copy_protocol_only":true,"independent_gdn_math_oracle":false,"performance_accepted":false})
        );
        super::super::session::gpu_raw_protocol::target_gate(&f)?;
        draft_gate(&f,None)?;
        Ok(())
    }
    #[test]
    #[ignore = "exclusive GPU; target-only GDN storage/partial-full commit/snapshot/bank gate"]
    fn real_checkpoint_target_gdn_storage_bank_protocol(
    ) -> std::result::Result<(), Box<dyn std::error::Error>> {
        let f = Fixture::load()?;
        super::super::session::gpu_raw_protocol::target_gate(&f)?;
        Ok(())
    }
    fn select_prefill_lifecycle_decoder(f: &mut Fixture) -> Result<()> {
        let execution = Qwen35ExecutionOptions {
            moe_activation: Qwen35MoeActivation::Nvfp4,
            kv_cache: Qwen35KvCacheDType::Fp8Dynamic,
        };
        let thor_profile = execution.uses_thor_dynamic_fp8_decode(f.ctx.device_id(), f.capacity);
        f.decoder = Qwen35Decoder::with_execution(&f.config, thor_profile)?;
        Ok(())
    }

    #[test]
    #[ignore = "exclusive GPU; strict recipes; real8+8 dynamic-FP8 prefill session lifecycle gate"]
    fn real_checkpoint_prefill_session_lifecycle(
    ) -> std::result::Result<(), Box<dyn std::error::Error>> {
        let mut f = Fixture::load()?;
        select_prefill_lifecycle_decoder(&mut f)?;
        super::super::session::gpu_raw_protocol::prefill_session_lifecycle_gate(&f)?;
        Ok(())
    }

    #[test]
    #[ignore = "RECIPE PREPARATION ONLY: explicit8+8 dynamic-FP8 lifecycle recipes; not correctness or performance acceptance"]
    fn prepare_prefill_session_lifecycle_recipes(
    ) -> std::result::Result<(), Box<dyn std::error::Error>> {
        let argv: Vec<String> = std::env::args().collect();
        assert!(argv.iter().any(|s| s == "--exact") && argv.iter().any(|s| s == "--ignored")
            && argv.iter().any(|s| s == "qwen35moe::mtp_runtime::gpu_raw_protocol::prepare_prefill_session_lifecycle_recipes")
            && argv.iter().any(|s| s == "--test-threads=1"));
        let mut f = Fixture::load_with_autotune(true)?;
        select_prefill_lifecycle_decoder(&mut f)?;
        super::super::session::gpu_raw_protocol::prefill_session_lifecycle_gate(&f)?;
        println!("{}", serde_json::json!({
            "schema": "apxinf.qwen35.prefill_session_recipe_preparation.v1",
            "preparation_only": true,
            "correctness_accepted": false,
            "performance_accepted": false
        }));
        Ok(())
    }

    fn long_prefill_lifecycle_case(
        prefix_rows: usize,
        preparation_only: bool,
    ) -> std::result::Result<(), Box<dyn std::error::Error>> {
        assert!(matches!(prefix_rows, 4096 | 32768));
        let append_rows = 4096;
        let total_rows = prefix_rows + append_rows;
        // Exact frozen context-benchmark input algorithm; one continuous stream
        // across prefix and append, not repetition of the short fixture prompt.
        let mut state = 0x35A3_BF04_4096_0001u64;
        let tokens: Vec<u32> = (0..total_rows).map(|_| {
            state = state.wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((state >> 17) as usize % (248_320 - 1_000)) as u32
        }).collect();
        let digest = |ids: &[u32]| ids.iter().fold(0xcbf29ce484222325u64, |h, id| {
            id.to_le_bytes().into_iter().fold(h, |h, byte| {
                (h ^ u64::from(byte)).wrapping_mul(0x100000001b3)
            })
        });
        let mut f = if preparation_only {
            Fixture::load_with_autotune(true)?
        } else {
            Fixture::load()?
        };
        f.prompt = tokens;
        // Same minimum as context bench max-new1 + fixed MTP lookahead3;
        // four subsequent target decodes fit without raising operator limits.
        f.capacity = total_rows + 4;
        let execution = Qwen35ExecutionOptions {
            moe_activation: Qwen35MoeActivation::Nvfp4,
            kv_cache: Qwen35KvCacheDType::Fp8Dynamic,
        };
        let thor_profile = execution.uses_thor_dynamic_fp8_decode(f.ctx.device_id(), f.capacity);
        f.decoder = Qwen35Decoder::with_execution(&f.config, thor_profile)?;
        let (prefix, append) = f.prompt.split_at(prefix_rows);
        println!("{}", serde_json::json!({
            "schema": "apxinf.qwen35.prefill_session_long_gate_config.v1",
            "prefix_rows": prefix_rows, "append_rows": append_rows,
            "capacity": f.capacity, "target_kv": "fp8_dynamic",
            "thor_dynamic_fp8_decode_profile": thor_profile,
            "input_seed_hex": "35a3bf0440960001",
            "input_digest_fnv1a64": format!("{:016x}", digest(&f.prompt)),
            "prefix_digest_fnv1a64": format!("{:016x}", digest(prefix)),
            "chunk_digest_fnv1a64": format!("{:016x}", digest(append)),
            "preparation_only": preparation_only,
            "performance_accepted": false
        }));
        super::super::session::gpu_raw_protocol::prefill_session_lifecycle_gate_case(
            &f, prefix, append, false)?;
        if preparation_only {
            println!("{}", serde_json::json!({
                "schema": "apxinf.qwen35.prefill_session_recipe_preparation.v1",
                "prefix_rows": prefix_rows, "append_rows": append_rows,
                "preparation_only": true,
                "correctness_accepted": false, "performance_accepted": false
            }));
        }
        Ok(())
    }

    #[test]
    #[ignore = "exclusive GPU; strict frozen recipes; real LCG4K+4K raw-state/readonly/retry/decode gate"]
    fn real_checkpoint_prefill_session_lifecycle_4k_4k(
    ) -> std::result::Result<(), Box<dyn std::error::Error>> {
        long_prefill_lifecycle_case(4096, false)
    }

    #[test]
    #[ignore = "exclusive GPU; strict frozen recipes; real LCG32K+4K raw-state/readonly/retry/decode gate"]
    fn real_checkpoint_prefill_session_lifecycle_32k_4k(
    ) -> std::result::Result<(), Box<dyn std::error::Error>> {
        long_prefill_lifecycle_case(32768, false)
    }

    #[test]
    #[ignore = "RECIPE PREPARATION ONLY: explicit LCG4K+4K gate; never correctness/performance acceptance"]
    fn prepare_prefill_session_lifecycle_4k_4k_recipes(
    ) -> std::result::Result<(), Box<dyn std::error::Error>> {
        let argv: Vec<String> = std::env::args().collect();
        assert!(argv.iter().any(|s| s == "--exact") && argv.iter().any(|s| s == "--ignored")
            && argv.iter().any(|s| s == "qwen35moe::mtp_runtime::gpu_raw_protocol::prepare_prefill_session_lifecycle_4k_4k_recipes")
            && argv.iter().any(|s| s == "--test-threads=1"));
        long_prefill_lifecycle_case(4096, true)
    }

    #[test]
    #[ignore = "RECIPE PREPARATION ONLY: explicit LCG32K+4K gate; never correctness/performance acceptance"]
    fn prepare_prefill_session_lifecycle_32k_4k_recipes(
    ) -> std::result::Result<(), Box<dyn std::error::Error>> {
        let argv: Vec<String> = std::env::args().collect();
        assert!(argv.iter().any(|s| s == "--exact") && argv.iter().any(|s| s == "--ignored")
            && argv.iter().any(|s| s == "qwen35moe::mtp_runtime::gpu_raw_protocol::prepare_prefill_session_lifecycle_32k_4k_recipes")
            && argv.iter().any(|s| s == "--test-threads=1"));
        long_prefill_lifecycle_case(32768, true)
    }

    #[test]
    #[ignore = "preparation only: fills target-only storage gate recipe cache; never a performance result"]
    fn prepare_target_gdn_storage_recipes() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let argv: Vec<String> = std::env::args().collect();
        assert!(argv.iter().any(|s| s == "--exact") && argv.iter().any(|s| s == "--ignored")
            && argv.iter().any(|s| s == "qwen35moe::mtp_runtime::gpu_raw_protocol::prepare_target_gdn_storage_recipes")
            && argv.iter().any(|s| s == "--test-threads=1"));
        let f = Fixture::load_with_autotune(true)?;
        super::super::session::gpu_raw_protocol::target_gate(&f)?;
        Ok(())
    }
    fn shortlist_head_and_context_gate(f: &Fixture) -> Result<f64> {
        let decoder=Qwen35Decoder::with_execution(&f.config,true)?;
        let backend=Qwen35Fp8AttentionBackend::with_kv_cache(
            &f.config,Qwen35KvCacheDType::Fp8Dynamic,None)?;
        let target=Qwen35Session::new(&f.ctx,&decoder,&f.weights,backend,64)?;
        let mut coordinator=Qwen35MtpCoordinator::new(&f.ctx,target,Qwen35MtpRuntime::new(&f.config)?)?;
        let options=GenerationOptions::greedy(32,None);
        coordinator.prefill_greedy(&f.ctx,&decoder,&f.weights,&f.prompt,&options)?;
        assert!(coordinator.committed_mtp.shortlist_allowed);
        let prepared_head=coordinator.mtp.prepared_shortlist_head.as_ref().unwrap().clone();
        assert!(coordinator.mtp.scratch.is_none(),"head preparation must not allocate full decode scratch");
        // A known in-range pending token makes this a protocol test rather
        // than an assumption about the model's chosen text continuation.
        coordinator.pending=Some(42);
        coordinator.run_greedy_round(&f.ctx,&decoder,&f.weights,3)?;
        assert!(coordinator.mtp.scratch.as_ref().unwrap().shortlist_active);
        assert!(Rc::ptr_eq(&prepared_head,
            coordinator.mtp.scratch.as_ref().unwrap().shortlist_head.as_ref().unwrap()));
        let weights=f.weights.lm_head.a16.as_ref().unwrap();
        let before=(raw(&weights.qweight)?,raw(&weights.block_scales)?,raw(&weights.global_scale)?);
        let minimum={
            let scratch=coordinator.mtp.scratch.as_ref().unwrap();
            let shortlist=scratch.shortlist_head.as_ref().unwrap();
            let host=transfers::to_cpu(&scratch.final_hidden)?;
            let values=host.as_bf16()?.to_vec();
            let input=transfers::to_cuda_on_stream(&f.ctx,Tensor::from_bf16(vec![1,HIDDEN],&values)?)?;
            let mut full=zeros(&f.ctx,vec![1,VOCAB],DType::BF16)?;
            let next_storage=transfers::to_cuda_on_stream(&f.ctx,Tensor::from_i32(vec![3],&[0x5a5a5a5a;3])?)?;
            let next=element_view(&next_storage,1)?;
            shortlist.project(&f.ctx,&input,&next)?;
            f.ctx.synchronize().map_err(Error::Cuda)?;
            let graph=apxinf_cuda_new::capture(&f.ctx,||shortlist.project(&f.ctx,&input,&next))?;
            let mut minimum=1f64;
            for data in [values.clone(),values.iter().rev().copied().collect()] {
                transfers::copy_cpu_to_cuda_on_stream(&f.ctx,Tensor::from_bf16(vec![1,HIDDEN],&data)?,&input)?;
                graph.replay()?;
                super::super::forward::project_lm_head(&f.ctx,&f.weights,&input,&mut full)?;
                f.ctx.synchronize().map_err(Error::Cuda)?;
                let a=transfers::to_cpu(&full)?;let b=transfers::to_cpu(&shortlist.logits)?;
                let map=transfers::to_cpu(&shortlist.remap)?;
                let(mut dot,mut aa,mut bb)=(0f64,0f64,0f64);
                let mut maximum=f32::NEG_INFINITY;
                for(local,&id)in map.as_i32()?.iter().enumerate(){
                    let(a,b)=(a.as_bf16()?[id as usize].to_f32(),b.as_bf16()?[local].to_f32());
                    assert!(a.is_finite()&&b.is_finite());maximum=maximum.max(b);
                    let(a,b)=(f64::from(a),f64::from(b));dot+=a*b;aa+=a*a;bb+=b*b;
                }
                let cosine=dot/(aa*bb).sqrt();assert!(cosine>=0.9999);minimum=minimum.min(cosine);
                let id=transfers::to_cpu(&next)?.as_i32()?[0] as u32;
                assert!(mtp_shortlist_contains(id));assert_eq!(a.as_bf16()?[id as usize].to_f32(),maximum);
                assert_eq!(transfers::to_cpu(&input)?.as_bf16()?,&data);
                let guard=transfers::to_cpu(&next_storage)?;
                assert_eq!(guard.as_i32()?[0],0x5a5a5a5a);assert_eq!(guard.as_i32()?[2],0x5a5a5a5a);
            }
            // Change the fixed hot buffers after Graph capture and verify
            // that the replay reads the new tile without changing its nodes.
            let tile = 200000usize / MTP_HOT_TILE_ROWS;
            shortlist.stage_hot_tile(&f.ctx, &f.weights, tile)?;
            graph.replay()?;
            super::super::forward::project_lm_head(&f.ctx, &f.weights, &input, &mut full)?;
            f.ctx.synchronize().map_err(Error::Cuda)?;
            let full_bits = transfers::to_cpu(&full)?;
            let hot_bits = transfers::to_cpu(&shortlist.logits)?;
            let map = transfers::to_cpu(&shortlist.remap)?;
            let hot_start = MTP_SHORTLIST_PREFIX + MTP_SHORTLIST_TAIL;
            for local in 0..MTP_HOT_TILE_ROWS {
                let id = tile * MTP_HOT_TILE_ROWS + local;
                assert_eq!(map.as_i32()?[hot_start + local], id as i32);
                assert_eq!(hot_bits.as_bf16()?[hot_start + local].to_bits(),
                    full_bits.as_bf16()?[id].to_bits());
            }
            // Replay the same captured nodes with both a changed tile and
            // changed hidden row. The staged weight copies must be bit-exact.
            transfers::copy_cpu_to_cuda_on_stream(&f.ctx,
                Tensor::from_bf16(vec![1,HIDDEN],&values)?,&input)?;
            graph.replay()?;
            super::super::forward::project_lm_head(&f.ctx,&f.weights,&input,&mut full)?;
            f.ctx.synchronize().map_err(Error::Cuda)?;
            let full_changed=transfers::to_cpu(&full)?;
            let hot_changed=transfers::to_cpu(&shortlist.logits)?;
            for local in 0..MTP_HOT_TILE_ROWS {
                let id=tile*MTP_HOT_TILE_ROWS+local;
                assert_eq!(hot_changed.as_bf16()?[hot_start+local].to_bits(),
                    full_changed.as_bf16()?[id].to_bits());
            }
            let source_q=MtpShortlistHead::slice(&weights.qweight,
                tile*MTP_HOT_TILE_ROWS,MTP_HOT_TILE_ROWS,32)?;
            let source_s=MtpShortlistHead::slice(&weights.block_scales,
                tile*MTP_HOT_TILE_ROWS,MTP_HOT_TILE_ROWS,4)?;
            assert_eq!(raw(&shortlist.qhot)?,raw(&source_q)?);
            assert_eq!(raw(&shortlist.shot)?,raw(&source_s)?);
            minimum
        };
        assert_eq!(before,(raw(&weights.qweight)?,raw(&weights.block_scales)?,raw(&weights.global_scale)?));
        let mut snapshot=coordinator.allocate_context_snapshot(&f.ctx,&decoder,&f.weights,64)?;
        coordinator.capture_context(&f.ctx,&mut snapshot)?;
        assert!(snapshot.mtp.shortlist_allowed);
        assert_eq!(snapshot.mtp.hot_tile, None);
        coordinator.pending=Some(200000);
        coordinator.run_greedy_round(&f.ctx,&decoder,&f.weights,3)?;
        assert!(coordinator.committed_mtp.shortlist_allowed);
        assert_eq!(coordinator.committed_mtp.hot_tile, Some(200000 / MTP_HOT_TILE_ROWS));
        assert!(!coordinator.mtp.scratch.as_ref().unwrap().shortlist_active);
        assert!(snapshot.mtp.shortlist_allowed);
        let prepared_sessions=coordinator.mtp.scratch.as_ref().unwrap().draft_sessions_prepared;
        coordinator.pending=Some(42);
        coordinator.run_greedy_round(&f.ctx,&decoder,&f.weights,3)?;
        let scratch=coordinator.mtp.scratch.as_ref().unwrap();
        assert!(scratch.shortlist_active,"an eligible pending token recovers next round");
        assert_eq!(coordinator.committed_mtp.hot_tile, Some(200000 / MTP_HOT_TILE_ROWS));
        assert_eq!(scratch.draft_sessions_prepared,prepared_sessions);
        assert!(Rc::ptr_eq(&prepared_head,scratch.shortlist_head.as_ref().unwrap()),
            "recovery must reuse prefill-prepared head buffers and views");
        assert_eq!(scratch.shortlist_head.as_ref().unwrap().staged_hot_tile.get(),
            Some(200000/MTP_HOT_TILE_ROWS));
        // Snapshot a live hot hint, mutate it, restore, append, and replay the
        // already prepared Graph. A snapshot must restore the hint with KV.
        let mut hot_snapshot=coordinator.allocate_context_snapshot(&f.ctx,&decoder,&f.weights,64)?;
        coordinator.capture_context(&f.ctx,&mut hot_snapshot)?;
        assert_eq!(hot_snapshot.mtp.hot_tile,Some(200000/MTP_HOT_TILE_ROWS));
        coordinator.pending=Some(180000);
        coordinator.run_greedy_round(&f.ctx,&decoder,&f.weights,3)?;
        assert_eq!(coordinator.committed_mtp.hot_tile,Some(180000/MTP_HOT_TILE_ROWS));
        coordinator.restore_context(&f.ctx,&hot_snapshot)?;
        assert_eq!(coordinator.committed_mtp.hot_tile,Some(200000/MTP_HOT_TILE_ROWS));
        coordinator.append_prefill_greedy(&f.ctx,&decoder,&f.weights,&[42],&options)?;
        assert_eq!(coordinator.committed_mtp.hot_tile,Some(200000/MTP_HOT_TILE_ROWS));
        coordinator.pending=Some(42);
        coordinator.run_greedy_round(&f.ctx,&decoder,&f.weights,3)?;
        assert!(coordinator.mtp.scratch.as_ref().unwrap().shortlist_active);
        assert_eq!(coordinator.mtp.scratch.as_ref().unwrap().shortlist_head
            .as_ref().unwrap().staged_hot_tile.get(),Some(200000/MTP_HOT_TILE_ROWS));
        coordinator.reset(&f.ctx)?;
        assert!(!coordinator.committed_mtp.shortlist_allowed && !coordinator.shadow_mtp.shortlist_allowed);
        coordinator.restore_context(&f.ctx,&snapshot)?;
        assert!(coordinator.committed_mtp.shortlist_allowed);
        assert_eq!(coordinator.committed_mtp.hot_tile, None);
        coordinator.append_prefill_greedy(&f.ctx,&decoder,&f.weights,&[42],&options)?;
        assert!(coordinator.committed_mtp.shortlist_allowed);
        coordinator.append_prefill_greedy(&f.ctx,&decoder,&f.weights,&[200000],&options)?;
        assert!(coordinator.committed_mtp.shortlist_allowed);
        coordinator.reset(&f.ctx)?;
        let mut outside=f.prompt.clone();outside[4]=200000;
        coordinator.prefill_greedy(&f.ctx,&decoder,&f.weights,&outside,&options)?;
        assert!(coordinator.committed_mtp.shortlist_allowed);
        coordinator.pending=Some(42);
        coordinator.run_greedy_round(&f.ctx,&decoder,&f.weights,3)?;
        assert!(coordinator.mtp.scratch.as_ref().unwrap().shortlist_active);
        assert!(Rc::ptr_eq(&prepared_head,
            coordinator.mtp.scratch.as_ref().unwrap().shortlist_head.as_ref().unwrap()));
        Ok(minimum)
    }

    fn shortlist_weight_rebind_gate(f: &mut Fixture) -> Result<()> {
        let mut runtime=Qwen35MtpRuntime::new(&f.config)?;
        let mut committed=runtime.allocate_state(&f.ctx,64)?;
        let mut shadow=runtime.allocate_state(&f.ctx,64)?;
        committed.position=Some(MtpPositionContract::after_prompt(16)?);
        committed.shortlist_allowed=true;
        committed.hot_tile=Some(200000/MTP_HOT_TILE_ROWS);
        let data=vec![half::bf16::from_f32(0.5);HIDDEN];
        let hidden=transfers::to_cuda_on_stream(&f.ctx,Tensor::from_bf16(vec![1,HIDDEN],&data)?)?;
        runtime.draft(&f.ctx,&f.weights,&committed,&mut shadow,42,16,&hidden,3)?;
        f.ctx.synchronize().map_err(Error::Cuda)?;
        let old_binding=runtime.scratch.as_ref().unwrap().shortlist_head.as_ref().unwrap().binding;
        let original=f.weights.lm_head.a16.as_ref().unwrap().global_scale.clone();
        let replacement=zeros(&f.ctx,vec![1],DType::F32)?;
        transfers::copy_cuda_to_cuda(&f.ctx,&original,&replacement)?;
        f.ctx.synchronize().map_err(Error::Cuda)?;
        assert_eq!(raw(&original)?,raw(&replacement)?);
        f.weights.lm_head.a16.as_mut().unwrap().global_scale=replacement;
        runtime.draft(&f.ctx,&f.weights,&committed,&mut shadow,42,16,&hidden,3)?;
        f.ctx.synchronize().map_err(Error::Cuda)?;
        let head=runtime.scratch.as_ref().unwrap().shortlist_head.as_ref().unwrap();
        assert!(head.binding!=old_binding);
        assert!(head.binding==MtpHeadBinding::new(&f.ctx,&f.weights)?);
        assert_eq!(head.staged_hot_tile.get(),Some(200000/MTP_HOT_TILE_ROWS));
        let source=f.weights.lm_head.a16.as_ref().unwrap();
        let source_q=MtpShortlistHead::slice(&source.qweight,
            (200000/MTP_HOT_TILE_ROWS)*MTP_HOT_TILE_ROWS,MTP_HOT_TILE_ROWS,32)?;
        let source_s=MtpShortlistHead::slice(&source.block_scales,
            (200000/MTP_HOT_TILE_ROWS)*MTP_HOT_TILE_ROWS,MTP_HOT_TILE_ROWS,4)?;
        assert_eq!(raw(&head.qhot)?,raw(&source_q)?);
        assert_eq!(raw(&head.shot)?,raw(&source_s)?);
        f.weights.lm_head.a16.as_mut().unwrap().global_scale=original;
        Ok(())
    }

    #[test]
    #[ignore="exclusive real-checkpoint shortlist head,Graph and transactional state contract"]
    fn real_checkpoint_shortlist_head_and_state_contract()->std::result::Result<(),Box<dyn std::error::Error>> {
        let mut f=Fixture::load()?;
        // Fixed valid tokens; no dependency on private benchmark files or traces.
        f.prompt=vec![248045,846,198,68111,449,21870,5640,4849,1675,883,264,3110,8151,310,26629,11];
        assert!(f.prompt.iter().copied().all(mtp_shortlist_contains));
        for t in [0u32,131071,247808,248319]{assert!(mtp_shortlist_contains(t));}
        for t in [131072u32,200000,247807,248320,u32::MAX]{assert!(!mtp_shortlist_contains(t));}
        let cosine=shortlist_head_and_context_gate(&f)?;
        draft_gate(&f,Some(42))?;
        shortlist_weight_rebind_gate(&mut f)?;
        println!("{}",serde_json::json!({"schema":"apxinf.mtp.shortlist_contract.v1",
            "same_input_min_cosine":cosine,"head_weights_and_input_readonly":true,
            "graph_rebinding":true,"partial_commit_rows":[1,2,3,4],"failed_retry_and_snapshot_restore":true,
            "eligibility_reset_copy_append_fallback":true,"head_weight_binding_reprepared":true,
            "main_verification_full_vocabulary":true,"performance_result":false}));
        Ok(())
    }


    #[test]
    #[ignore = "real checkpoint BF16 image rowwise verifier; exclusive GPU locks and prepared strict recipes"]
    fn real_checkpoint_rowwise_target_raw_bank_protocol(
    ) -> std::result::Result<(), Box<dyn std::error::Error>> {
        let f = Fixture::load()?;
        super::super::session::gpu_raw_protocol::rowwise_target_gate(&f)?;
        Ok(())
    }

    #[test]
    #[ignore = "RECIPE PREPARATION ONLY: rowwise target protocol, explicit tuning directory and exclusive GPU locks"]
    fn prepare_rowwise_target_protocol_recipes(
    ) -> std::result::Result<(), Box<dyn std::error::Error>> {
        let arguments: Vec<String> = std::env::args().collect();
        let name =
            "qwen35moe::mtp_runtime::gpu_raw_protocol::prepare_rowwise_target_protocol_recipes";
        assert!(
            arguments.iter().any(|x| x == "--ignored")
                && arguments.iter().any(|x| x == "--exact")
                && arguments.iter().any(|x| x == name),
            "run this preparation entry explicitly by exact test name"
        );
        let f = Fixture::load_with_autotune(true)?;
        super::super::session::gpu_raw_protocol::rowwise_target_gate(&f)?;
        Ok(())
    }


    #[test]
    #[ignore = "RECIPE PREPARATION ONLY: explicit shortlist contract recipes; not a correctness or performance result"]
    fn prepare_shortlist_candidate_recipes() -> std::result::Result<(), Box<dyn std::error::Error>> {
        let arguments: Vec<String> = std::env::args().collect();
        let name = "qwen35moe::mtp_runtime::gpu_raw_protocol::prepare_shortlist_candidate_recipes";
        assert!(arguments.iter().any(|x| x == "--ignored")
            && arguments.iter().any(|x| x == "--exact")
            && arguments.iter().any(|x| x == name)
            && arguments.iter().any(|x| x == "--test-threads=1"));
        let mut f = Fixture::load_with_autotune(true)?;
        f.prompt = vec![248045,846,198,68111,449,21870,5640,4849,1675,883,264,3110,8151,310,26629,11];
        shortlist_head_and_context_gate(&f)?;
        draft_gate(&f, Some(42))?;
        shortlist_weight_rebind_gate(&mut f)?;
        Ok(())
    }

    // Recipe preparation is an explicit different ignored test and process.
    // It executes actual target/MTP operations; it is not a protocol/quality gate.
    fn prep_command(
        program: &str,
        args: &[&str],
    ) -> std::result::Result<Vec<u8>, Box<dyn std::error::Error>> {
        let output = std::process::Command::new(program).args(args).output()?;
        if !output.status.success() {
            return Err(format!(
                "{program} failed: {}",
                String::from_utf8_lossy(&output.stderr)
            )
            .into());
        }
        Ok(output.stdout)
    }
    fn prep_sha(bytes: &[u8]) -> std::result::Result<String, Box<dyn std::error::Error>> {
        use std::io::Write;
        let mut child = std::process::Command::new("sha256sum")
            .stdin(std::process::Stdio::piped())
            .stdout(std::process::Stdio::piped())
            .spawn()?;
        child.stdin.take().expect("sha256 stdin").write_all(bytes)?;
        let output = child.wait_with_output()?;
        if !output.status.success() {
            return Err("sha256sum failed".into());
        }
        let text = String::from_utf8(output.stdout)?;
        let hash = text.split_whitespace().next().ok_or("empty sha256sum")?;
        if hash.len() != 64 || !hash.bytes().all(|x| x.is_ascii_hexdigit()) {
            return Err("invalid sha256sum".into());
        }
        Ok(hash.to_owned())
    }
    fn prep_file_sha(
        path: &std::path::Path,
    ) -> std::result::Result<String, Box<dyn std::error::Error>> {
        let output = std::process::Command::new("sha256sum")
            .arg("--")
            .arg(path)
            .output()?;
        if !output.status.success() {
            return Err(format!("sha256sum failed for {}", path.display()).into());
        }
        let text = String::from_utf8(output.stdout)?;
        let hash = text
            .split_whitespace()
            .next()
            .ok_or("empty file sha256sum")?;
        if hash.len() != 64 || !hash.bytes().all(|x| x.is_ascii_hexdigit()) {
            return Err("invalid file sha256sum".into());
        }
        Ok(hash.to_owned())
    }
    fn prep_files(root: &std::path::Path, files: &mut Vec<PathBuf>) -> std::io::Result<()> {
        if root.exists() {
            for entry in std::fs::read_dir(root)? {
                let path = entry?.path();
                if path.is_dir() {
                    prep_files(&path, files)?;
                } else if path.is_file() {
                    files.push(path);
                }
            }
        }
        Ok(())
    }
    fn prep_inventory(
        root: &std::path::Path,
    ) -> std::result::Result<serde_json::Value, Box<dyn std::error::Error>> {
        let mut paths = Vec::new();
        prep_files(root, &mut paths)?;
        paths.sort();
        let mut files = Vec::new();
        let mut recipe_count = 0usize;
        for path in paths {
            if path.extension().is_some_and(|x| x == "recipe") {
                recipe_count += 1;
            }
            files.push(serde_json::json!({"path":path.strip_prefix(root)?,"bytes":path.metadata()?.len(),"sha256":prep_file_sha(&path)?}));
        }
        Ok(
            serde_json::json!({"root":root,"recipe_count":recipe_count,"files":files,"files_json_sha256":prep_sha(&serde_json::to_vec(&files)?)?}),
        )
    }

    #[test]
    #[ignore = "RECIPE PREPARATION ONLY: real weights once, online tune enabled, writes explicit tuning dir; not quality or performance acceptance"]
    fn prepare_real_checkpoint_protocol_recipes(
    ) -> std::result::Result<(), Box<dyn std::error::Error>> {
        let arguments: Vec<String> = std::env::args().collect();
        let entry =
            "qwen35moe::mtp_runtime::gpu_raw_protocol::prepare_real_checkpoint_protocol_recipes";
        assert!(arguments.iter().any(|x|x=="--ignored") && arguments.iter().any(|x|x=="--exact")
            && arguments.iter().filter(|x|x.as_str()==entry).count()==1
            && arguments.iter().any(|x|x=="--test-threads=1"),
            "recipe preparation requires --ignored --exact <full prepare name> --test-threads=1; never run all ignored tests");
        let checkpoint =
            PathBuf::from(std::env::var_os("APXINF_QWEN35_CHECKPOINT").expect("checkpoint env"));
        let tuning =
            PathBuf::from(std::env::var_os("APXINF_QWEN35_PARITY_TUNING_DIR").expect("tuning env"));
        assert!(checkpoint.is_absolute() && tuning.is_absolute());
        let mut model_files = Vec::new();
        let mut paths: Vec<_> = std::fs::read_dir(&checkpoint)?
            .map(|x| x.map(|y| y.path()))
            .collect::<std::io::Result<_>>()?;
        paths.sort();
        for path in paths {
            if path.is_file()
                && (path.extension().is_some_and(|x| x == "safetensors")
                    || path
                        .file_name()
                        .is_some_and(|x| x == "config.json" || x == "hf_quant_config.json"))
            {
                model_files.push(serde_json::json!({"path":path,"bytes":path.metadata()?.len(),"sha256":prep_file_sha(&path)?}));
            }
        }
        assert!(!model_files.is_empty(), "actual model files required");
        let head = String::from_utf8(prep_command("git", &["rev-parse", "HEAD"])?)?;
        let status = String::from_utf8(prep_command(
            "git",
            &["status", "--porcelain", "--untracked-files=all"],
        )?)?;
        let patch = prep_command("git", &["diff", "HEAD", "--binary"])?;
        let mut untracked = Vec::new();
        for name in prep_command("git", &["ls-files", "--others", "--exclude-standard", "-z"])?
            .split(|x| *x == 0)
            .filter(|x| !x.is_empty())
        {
            let path = PathBuf::from(std::str::from_utf8(name)?);
            if path.is_file() {
                untracked.push(serde_json::json!({"path":path,"sha256":prep_file_sha(&path)?}));
            }
        }
        let protocol_graphs_env = std::env::var_os("APXINF_QWEN35_PROTOCOL_GRAPHS");
        let graphs = parse_protocol_graphs(protocol_graphs_env.as_deref())?;
        let before = prep_inventory(&tuning)?;
        println!(
            "{}",
            serde_json::json!({"schema":"apxinf.qwen35.protocol_recipe_prepare_identity.v2",
            "phase":"prepare_only","source_head":head.trim(),"source_status":status,"source_patch_sha256":prep_sha(&patch)?,"source_untracked_sha256":untracked,
            "executable":std::env::current_exe()?,"executable_sha256":prep_file_sha(&std::env::current_exe()?)?,"model_files":model_files,
            "recipes_before":before,"autotune_constructor_argument":true,"strict_recipe_required":false,"apxinf_require_recipes":"0",
            "target_kv":"bf16","draft_kv":"fp8","draft_scale":1.0,"moe":"nvfp4","ordered_reduce":true,"graphs":graphs,"graph_gdn":std::env::var("APXINF_QWEN35_GRAPH_GDN").ok(),
            "weights_loads":1,"quality_accepted":false,"performance_accepted":false})
        );
        let f = Fixture::load_with_autotune(true)?;
        assert_eq!(f.graphs, graphs);
        assert_eq!(std::env::var("APXINF_REQUIRE_RECIPES").unwrap(), "0");
        // Raw gate logical capacity64; the public protocol test defaults to
        // prompt16+64=80. Both map to the same128-row physical KV block, but
        // exercise both constructors rather than assuming recipe equivalence.
        for logical_capacity in [64usize, 80] {
            let backend = Qwen35Fp8AttentionBackend::with_kv_cache(
                &f.config,
                Qwen35KvCacheDType::Bf16,
                None,
            )?;
            let mut target =
                Qwen35Session::new(&f.ctx, &f.decoder, &f.weights, backend, logical_capacity)?;
            let mut runtime = Qwen35MtpRuntime::new(&f.config)?;
            let mut committed = runtime.allocate_state(&f.ctx, logical_capacity)?;
            let mut shadow = runtime.allocate_state(&f.ctx, logical_capacity)?;
            let verify_ids = [42i32, 1729, 19, 777];
            for p in 16..=23usize {
                // Original16-token prefix is unchanged; additional rows are an
                // explicitly recorded repetition of that same protocol pattern.
                let prompt: Vec<u32> = f.prompt.iter().copied().cycle().take(p).collect();
                for width in 2..=4usize {
                    let input = transfers::to_cuda_on_stream(
                        &f.ctx,
                        Tensor::from_i32(vec![width], &verify_ids[..width])?,
                    )?;
                    for keep in 1..=width {
                        target.reset(&f.ctx)?;
                        target.prefill(&f.ctx, &f.decoder, &f.weights, &prompt)?;
                        let verified = target
                            .verify_target_chunk_device(&f.ctx, &f.decoder, &f.weights, &input)?;
                        let staged = target.prepare_target_commit(
                            &f.ctx, &f.decoder, &f.weights, verified, keep,
                        )?;
                        target.publish_target_commit(staged)?;
                    }
                }
                // Scalar one-row kernels/attention at each required short position;
                // four steps also cover the later public protocol continuation tail.
                target.reset(&f.ctx)?;
                target.prefill(&f.ctx, &f.decoder, &f.weights, &prompt)?;
                for token in [991u32, 998, 654, 991] {
                    target.decode(&f.ctx, &f.decoder, &f.weights, token)?;
                }
                for suffix in [
                    &[1234u32, 5678][..],
                    &[3456u32, 7890][..],
                    &[2345u32, 6789][..],
                    &[1234u32][..],
                ] {
                    target.reset(&f.ctx)?;
                    let base =
                        target.prepare_target_prefill(&f.ctx, &f.decoder, &f.weights, &prompt)?;
                    let hidden = base.output().hidden_states.clone();
                    let last = base.output().last_hidden_state()?;
                    let pending = base.output().greedy_token_ids(&f.ctx)?[0];
                    target.publish_target_prefill(base)?;
                    if suffix == &[1234u32, 5678][..] {
                        for depth in 1..=3usize {
                            for keep in 1..=depth + 1 {
                                runtime.reset_state(&f.ctx, &mut committed)?;
                                runtime.reset_state(&f.ctx, &mut shadow)?;
                                let prepared = runtime.prepare_prefill(
                                    &f.ctx,
                                    &f.weights,
                                    &mut shadow,
                                    &prompt,
                                    &hidden,
                                )?;
                                runtime.publish_prefill(&mut committed, &mut shadow, prepared);
                                runtime.draft(
                                    &f.ctx,
                                    &f.weights,
                                    &committed,
                                    &mut shadow,
                                    pending,
                                    p,
                                    &last,
                                    depth,
                                )?;
                                let full = keep == depth + 1;
                                let contract = MtpCommit {
                                    cache_range: ShadowCommitRange {
                                        start: p,
                                        committed_end: p + keep,
                                        verified_end: p + if full { depth + 1 } else { depth },
                                    },
                                    materialize_final_proposal: full,
                                };
                                let prepared = runtime.prepare_commit(
                                    &f.ctx,
                                    &f.weights,
                                    &mut shadow,
                                    contract,
                                )?;
                                runtime.publish_commit(&mut committed, &mut shadow, prepared);
                            }
                        }
                    }
                    // Restore the original target-aligned draft prefix before
                    // actual one/two-row target+shifted draft teacher forcing.
                    runtime.reset_state(&f.ctx, &mut committed)?;
                    runtime.reset_state(&f.ctx, &mut shadow)?;
                    let prepared = runtime.prepare_prefill(
                        &f.ctx,
                        &f.weights,
                        &mut shadow,
                        &prompt,
                        &hidden,
                    )?;
                    runtime.publish_prefill(&mut committed, &mut shadow, prepared);
                    let appended = target
                        .prepare_target_append_prefill(&f.ctx, &f.decoder, &f.weights, suffix)?;
                    let prepared = runtime.prepare_append_prefill(
                        &f.ctx,
                        &f.weights,
                        &committed,
                        &mut shadow,
                        suffix,
                        &last,
                        &appended.output().hidden_states,
                    )?;
                    runtime.publish_prefill(&mut committed, &mut shadow, prepared);
                    target.publish_target_prefill(appended)?;
                }
                f.ctx.synchronize().map_err(Error::Cuda)?;
                println!(
                    "{}",
                    serde_json::json!({"schema":"apxinf.qwen35.protocol_recipe_prepare_shape.v2",
                "prompt_tokens":p,"prompt_token_ids":prompt,"target_capacity":logical_capacity,"physical_block_tokens":128,
                "target_verify_rows":[2,3,4],"target_commit_keeps":"1..=verified_rows","scalar_steps":4,
                "append_tokens":[[1234,5678],[3456,7890],[2345,6789],[1234]],"draft_prefill_rows":p-1,
                "draft_depths":[1,2,3],"draft_commit_keeps":"1..=depth+1","actual_gpu_operations_completed":true,
                "quality_accepted":false,"performance_accepted":false})
                );
            }
        }
        f.ctx.synchronize().map_err(Error::Cuda)?;
        let after = prep_inventory(&tuning)?;
        assert!(
            after["recipe_count"].as_u64().unwrap_or(0) > 0,
            "real persistent recipes required"
        );
        println!(
            "{}",
            serde_json::json!({"schema":"apxinf.qwen35.protocol_recipe_prepare_complete.v2",
            "recipes_after":after,"phase":"prepare_only","shape_iterations_completed":16,
            "native_numeric_validation":"unchanged; operator errors propagate; no override installed",
            "quality_accepted":false,"performance_accepted":false,"strict_followup_requires_fresh_process":true})
        );
        Ok(())
    }
}
