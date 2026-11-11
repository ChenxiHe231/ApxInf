//! Concrete base-model full-attention backend for Qwen3.5-35B-A3B.
//!
//! The base checkpoint's ten full-attention layers use BF16 projections. Their
//! persistent K/V cache is selected at load time. BF16 uses FA2, fixed-scale
//! FP8 retains the checkpoint-calibrated path, and dynamic FP8 computes one
//! scale per token and KV head and reads the cache directly in decode.
//! MTP attention has a separate cache contract and is not handled here.

use std::cell::Cell;
use std::collections::HashMap;
use std::ops::Range;
use std::rc::Rc;

use apxinf_core::{DType, Device, Error, Result, Shape, Tensor};
use apxinf_cuda_new::{ops, transfers, CudaBuffer, CudaContext, ExecutionSession};

use super::architecture::AttentionGeometry;
use super::config::Qwen35MoeConfig;
use super::decoder::{DecoderPhase, FullAttentionBackend, FullAttentionRequest};
use super::device_weights::{DeviceAttentionWeights, DeviceStaticFp8Linear};
use super::execution::Qwen35KvCacheDType;
use super::forward::{project_static_fp8_prequantized, quantize_static_fp8};
use super::runtime_weights::KvCacheScales;

const TEXT_POSITION_CHANNELS: usize = 3;
const TARGET_HIDDEN_SIZE: usize = 2_048;
const TARGET_LAYERS: usize = 40;
const FULL_ATTENTION_INTERVAL: usize = 4;
const DECODE_PROJECTION_WORKSPACE: usize = 4 * 1024 * 1024;
/// Match TensorRT-LLM's engine configuration (`tokens_per_block=128`).
/// This constant is physical cache page granularity, not a scale group size.
pub const QWEN35_KV_CACHE_BLOCK_TOKENS: usize = 128;

/// Persistent cache for one base-model full-attention layer.
#[derive(Debug)]
pub struct Qwen35AttentionCache {
    layer_index: usize,
    geometry: AttentionGeometry,
    /// Logical token capacity admitted by the session.
    capacity: usize,
    /// Physical pages backing the logical token range.
    blocks: usize,
    scales: Option<KvCacheScales>,
    dtype: Qwen35KvCacheDType,
    key: Tensor,
    value: Tensor,
    dynamic_key_scales: Option<Tensor>,
    dynamic_value_scales: Option<Tensor>,
}

impl Qwen35AttentionCache {
    pub fn layer_index(&self) -> usize {
        self.layer_index
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn block_tokens(&self) -> usize {
        QWEN35_KV_CACHE_BLOCK_TOKENS
    }

    pub fn blocks(&self) -> usize {
        self.blocks
    }

    pub fn scales(&self) -> Option<KvCacheScales> {
        self.scales
    }

    pub fn dtype(&self) -> Qwen35KvCacheDType {
        self.dtype
    }

    pub fn key(&self) -> &Tensor {
        &self.key
    }

    pub fn value(&self) -> &Tensor {
        &self.value
    }

    pub fn dynamic_key_scales(&self) -> Option<&Tensor> {
        self.dynamic_key_scales.as_ref()
    }

    pub fn dynamic_value_scales(&self) -> Option<&Tensor> {
        self.dynamic_value_scales.as_ref()
    }

    fn dynamic_fp8_cache(&self) -> Result<ops::Qwen35DynamicFp8KvCache<'_>> {
        if self.dtype != Qwen35KvCacheDType::Fp8Dynamic {
            return Err(invalid("dynamic FP8 cache requested from another KV mode"));
        }
        Ok(ops::Qwen35DynamicFp8KvCache {
            key: &self.key,
            value: &self.value,
            key_scales: self
                .dynamic_key_scales
                .as_ref()
                .ok_or_else(|| invalid("missing dynamic K scales"))?,
            value_scales: self
                .dynamic_value_scales
                .as_ref()
                .ok_or_else(|| invalid("missing dynamic V scales"))?,
        })
    }
}

/// CUDA implementation of the decoder's target-specific attention boundary.
///
/// Scratch profiles are retained by token count, so ordinary one-token decode
/// does not allocate projection or attention buffers on every step.
pub struct Qwen35Fp8AttentionBackend {
    hidden_size: usize,
    rope_theta: f32,
    kv_cache_dtype: Qwen35KvCacheDType,
    scratch: HashMap<(usize, usize, usize, bool), AttentionScratch>,
    pre_projection_sessions: HashMap<(usize, usize, usize), Rc<PreparedExecution>>,
    post_projection_sessions: HashMap<(usize, usize, usize), Rc<PreparedExecution>>,
}

struct PreparedExecution {
    session: ExecutionSession,
    prepared: Cell<bool>,
}

impl PreparedExecution {
    fn new(device: usize) -> Result<Self> {
        Ok(Self {
            session: ExecutionSession::with_capacity(DECODE_PROJECTION_WORKSPACE, device)?,
            prepared: Cell::new(false),
        })
    }

    fn run<T>(&self, operation: impl FnOnce() -> Result<T>) -> Result<T> {
        if self.prepared.get() {
            self.session.run(operation)
        } else {
            let output = self.session.prepare(operation)?;
            self.prepared.set(true);
            Ok(output)
        }
    }
}

impl Qwen35Fp8AttentionBackend {
    pub fn new(config: &Qwen35MoeConfig) -> Result<Self> {
        Self::with_kv_cache(config, Qwen35KvCacheDType::Bf16, None)
    }

    pub fn with_kv_cache(
        config: &Qwen35MoeConfig,
        kv_cache_dtype: Qwen35KvCacheDType,
        scales: Option<KvCacheScales>,
    ) -> Result<Self> {
        config.validate()?;
        match (kv_cache_dtype, scales) {
            (Qwen35KvCacheDType::Bf16, None) => {}
            (Qwen35KvCacheDType::Fp8Dynamic, None) => {}
            (Qwen35KvCacheDType::Fp8, Some(scales)) => validate_scales(scales)?,
            (Qwen35KvCacheDType::Fp8, None) => {
                return Err(invalid(
                    "Qwen3.6 FP8 KV cache requires TensorRT-compatible static scales",
                ));
            }
            (Qwen35KvCacheDType::Bf16, Some(_)) => {
                return Err(invalid("Qwen3.6 BF16 KV cache must not carry FP8 scales"));
            }
            (Qwen35KvCacheDType::Fp8Dynamic, Some(_)) => {
                return Err(invalid("dynamic FP8 KV cache must not carry fixed scales"));
            }
        }
        Ok(Self {
            hidden_size: TARGET_HIDDEN_SIZE,
            rope_theta: config.text_config.rope_parameters.rope_theta,
            kv_cache_dtype,
            scratch: HashMap::new(),
            pre_projection_sessions: HashMap::new(),
            post_projection_sessions: HashMap::new(),
        })
    }

    pub fn retained_profiles(&self) -> usize {
        self.scratch.len()
    }

    fn scratch(
        &mut self,
        ctx: &CudaContext,
        tokens: usize,
        cache_capacity: usize,
        phase: DecoderPhase,
    ) -> Result<&mut AttentionScratch> {
        let dynamic_decode = self.kv_cache_dtype == Qwen35KvCacheDType::Fp8Dynamic
            && tokens <= ops::QWEN35_MTP_DRAFT_STEPS + 1;
        let key = (ctx.device_id(), tokens, cache_capacity, dynamic_decode);
        if !self.scratch.contains_key(&key) {
            // Prefill lengths differ on almost every AIOS request. Each profile
            // owns full-capacity dequant buffers, so retaining one per length
            // grew device memory by ~140 MB per request until CUDA OOM. Keep
            // the fixed decode/verify profiles and at most one prefill profile.
            if tokens > ops::QWEN35_MTP_DRAFT_STEPS + 1 {
                self.scratch
                    .retain(|existing, _| existing.1 <= ops::QWEN35_MTP_DRAFT_STEPS + 1);
            }
            let profile =
                AttentionScratch::new(ctx, tokens, cache_capacity, self.kv_cache_dtype, phase)?;
            self.scratch.insert(key, profile);
        }
        Ok(self
            .scratch
            .get_mut(&key)
            .expect("Qwen3.5 attention scratch profile was just inserted"))
    }

    fn projection_session(
        sessions: &mut HashMap<(usize, usize, usize), Rc<PreparedExecution>>,
        ctx: &CudaContext,
        layer_index: usize,
        phase: DecoderPhase,
        tokens: usize,
    ) -> Result<Option<Rc<PreparedExecution>>> {
        let fixed_profile = phase == DecoderPhase::Decode
            || (phase == DecoderPhase::Prefill && tokens == ops::QWEN35_MTP_DRAFT_STEPS + 1);
        if !fixed_profile {
            return Ok(None);
        }
        let key = (ctx.device_id(), layer_index, tokens);
        if !sessions.contains_key(&key) {
            sessions.insert(key, Rc::new(PreparedExecution::new(ctx.device_id())?));
        }
        Ok(sessions.get(&key).cloned())
    }
}

impl FullAttentionBackend for Qwen35Fp8AttentionBackend {
    type Cache = Qwen35AttentionCache;

    fn allocate_cache(
        &mut self,
        ctx: &CudaContext,
        layer_index: usize,
        geometry: AttentionGeometry,
        capacity: usize,
        scales: Option<KvCacheScales>,
    ) -> Result<Self::Cache> {
        validate_full_attention_layer(layer_index)?;
        validate_cache_contract(geometry, capacity, scales, self.kv_cache_dtype)?;
        let (blocks, _physical_capacity) = cache_block_geometry(capacity)?;
        let shape = vec![
            blocks,
            QWEN35_KV_CACHE_BLOCK_TOKENS,
            ops::QWEN35_ATTENTION_KV_HEADS,
            ops::QWEN35_ATTENTION_HEAD_DIM,
        ];
        let dtype = self.kv_cache_dtype.tensor_dtype();
        let (dynamic_key_scales, dynamic_value_scales) =
            if self.kv_cache_dtype == Qwen35KvCacheDType::Fp8Dynamic {
                let scale_shape = vec![
                    blocks,
                    QWEN35_KV_CACHE_BLOCK_TOKENS,
                    ops::QWEN35_ATTENTION_KV_HEADS,
                ];
                (
                    Some(zeros(ctx, scale_shape.clone(), DType::F32)?),
                    Some(zeros(ctx, scale_shape, DType::F32)?),
                )
            } else {
                (None, None)
            };
        Ok(Qwen35AttentionCache {
            layer_index,
            geometry,
            capacity,
            blocks,
            scales,
            dtype: self.kv_cache_dtype,
            key: zeros(ctx, shape.clone(), dtype)?,
            value: zeros(ctx, shape, dtype)?,
            dynamic_key_scales,
            dynamic_value_scales,
        })
    }

    fn reset_cache(&mut self, ctx: &CudaContext, cache: &mut Self::Cache) -> Result<()> {
        validate_cache_device(ctx, cache)?;
        for tensor in [&cache.key, &cache.value]
            .into_iter()
            .chain(cache.dynamic_key_scales.iter())
            .chain(cache.dynamic_value_scales.iter())
        {
            CudaBuffer::from_tensor(tensor)
                .map_err(Error::Cuda)?
                .zero_async(ctx.stream())
                .map_err(Error::Cuda)?;
        }
        Ok(())
    }

    fn copy_cache_range(
        &mut self,
        ctx: &CudaContext,
        source: &Self::Cache,
        destination: &mut Self::Cache,
        positions: Range<usize>,
    ) -> Result<()> {
        validate_cache_pair(ctx, source, destination, &positions)?;
        if positions.is_empty() {
            return Ok(());
        }
        let rows = positions.end - positions.start;
        let (byte_offset, byte_len) = cache_byte_range(positions.clone(), source.dtype)?;
        let shape = Shape::new(vec![
            rows,
            ops::QWEN35_ATTENTION_KV_HEADS,
            ops::QWEN35_ATTENTION_HEAD_DIM,
        ]);
        for (source_tensor, destination_tensor) in [
            (&source.key, &destination.key),
            (&source.value, &destination.value),
        ] {
            let source_view = CudaBuffer::from_tensor(source_tensor)
                .map_err(Error::Cuda)?
                .view(byte_offset, byte_len)
                .map_err(Error::Cuda)?
                .as_tensor(shape.clone(), source.dtype.tensor_dtype())
                .map_err(Error::Cuda)?;
            let destination_view = CudaBuffer::from_tensor(destination_tensor)
                .map_err(Error::Cuda)?
                .view(byte_offset, byte_len)
                .map_err(Error::Cuda)?
                .as_tensor(shape.clone(), source.dtype.tensor_dtype())
                .map_err(Error::Cuda)?;
            transfers::copy_cuda_to_cuda(ctx, &source_view, &destination_view)?;
        }
        if source.dtype == Qwen35KvCacheDType::Fp8Dynamic {
            let row_bytes = ops::QWEN35_ATTENTION_KV_HEADS * DType::F32.size_in_bytes();
            let offset = positions.start * row_bytes;
            let len = rows * row_bytes;
            let shape = Shape::new(vec![rows, ops::QWEN35_ATTENTION_KV_HEADS]);
            for (src, dst) in [
                (
                    source.dynamic_key_scales.as_ref(),
                    destination.dynamic_key_scales.as_ref(),
                ),
                (
                    source.dynamic_value_scales.as_ref(),
                    destination.dynamic_value_scales.as_ref(),
                ),
            ] {
                let src = src.ok_or_else(|| invalid("missing source dynamic FP8 scale cache"))?;
                let dst =
                    dst.ok_or_else(|| invalid("missing destination dynamic FP8 scale cache"))?;
                let src_view = CudaBuffer::from_tensor(src)
                    .map_err(Error::Cuda)?
                    .view(offset, len)
                    .map_err(Error::Cuda)?
                    .as_tensor(shape.clone(), DType::F32)
                    .map_err(Error::Cuda)?;
                let dst_view = CudaBuffer::from_tensor(dst)
                    .map_err(Error::Cuda)?
                    .view(offset, len)
                    .map_err(Error::Cuda)?
                    .as_tensor(shape.clone(), DType::F32)
                    .map_err(Error::Cuda)?;
                transfers::copy_cuda_to_cuda(ctx, &src_view, &dst_view)?;
            }
        }
        Ok(())
    }

    fn run(
        &mut self,
        ctx: &CudaContext,
        request: FullAttentionRequest<'_>,
        cache: &mut Self::Cache,
    ) -> Result<()> {
        let tokens = validate_request(ctx, self.hidden_size, &request, cache)?;
        let rope_theta = self.rope_theta;
        let pre_projection = Self::projection_session(
            &mut self.pre_projection_sessions,
            ctx,
            request.layer_index,
            request.phase,
            tokens,
        )?;
        let post_projection = Self::projection_session(
            &mut self.post_projection_sessions,
            ctx,
            request.layer_index,
            request.phase,
            tokens,
        )?;
        let scratch = self.scratch(ctx, tokens, cache.capacity, request.phase)?;
        scratch.ensure_text_positions(ctx, request.start_position)?;
        let profile_attention = tokens == ops::QWEN35_MTP_DRAFT_STEPS + 1
            && std::env::var_os("APXINF_QWEN35_ATTENTION_TIMING").is_some();
        if profile_attention {
            ctx.synchronize().map_err(Error::Cuda)?;
        }
        let attention_started = std::time::Instant::now();

        // ---- q/k/v projection GEMMs ----
        let mut project_qkv = || {
            quantize_static_fp8(
                ctx,
                request.normalized_hidden,
                &scratch.fp8_projection_input,
                request.weights.q_proj.input_scale,
            )?;
            for (linear, output) in [
                (&request.weights.q_proj, &mut scratch.projected_query_gate),
                (&request.weights.k_proj, &mut scratch.projected_key),
                (&request.weights.v_proj, &mut scratch.projected_value),
            ] {
                project_static_fp8_prequantized(
                    ctx,
                    linear,
                    &scratch.fp8_projection_input,
                    output,
                    tokens,
                )?;
            }
            Ok(())
        };
        if let Some(session) = pre_projection {
            session.run(project_qkv)?;
        } else {
            project_qkv()?;
        }
        if profile_attention {
            ctx.synchronize().map_err(Error::Cuda)?;
            eprintln!(
                "[verify-attention] layer={} qkv_ms={:.3}",
                request.layer_index,
                attention_started.elapsed().as_secs_f64() * 1000.0
            );
        }
        let attention_core_started = std::time::Instant::now();

        if cache.dtype == Qwen35KvCacheDType::Fp8Dynamic {
            return run_dynamic_fp8_attention(
                ctx,
                &request,
                cache,
                scratch,
                tokens,
                rope_theta,
                post_projection,
            );
        }

        // Decode and fixed-row MTP verification consume the FP8 cache directly.
        // The native path fuses head RMSNorm, mRoPE, cache append, parallel
        // score/value reduction, and sigmoid gating. Larger prompt chunks keep
        // using FA2 below; launching one score grid per prompt row would be
        // wasteful for prefill.
        if cache.dtype == Qwen35KvCacheDType::Fp8
            && tokens <= ops::QWEN35_MTP_DRAFT_STEPS + 1
            && std::env::var("APXINF_QWEN35_FP8_DIRECT_ATTENTION").as_deref() != Ok("0")
        {
            let scales = cache.scales.ok_or_else(|| {
                invalid("Qwen3.6 FP8 KV cache requested without static dequant scales")
            })?;
            ops::qwen35_attention_fp8(
                ctx,
                tokens,
                ops::Qwen35Fp8AttentionArgs {
                    projected_query_gate: &scratch.projected_query_gate,
                    projected_key: &scratch.projected_key,
                    projected_value: &scratch.projected_value,
                    query_norm: &request.weights.q_norm,
                    key_norm: &request.weights.k_norm,
                    positions: &scratch.positions,
                    cache: ops::Qwen35Fp8KvCache {
                        key: &cache.key,
                        value: &cache.value,
                    },
                    scratch: ops::Qwen35AttentionScratch {
                        query: &scratch.query,
                        gate: &scratch.gate,
                        output: &scratch.attention_output,
                        scores: &scratch.fp8_scores,
                        partial_output: &scratch.fp8_partial_output,
                        partial_stats: &scratch.fp8_partial_stats,
                    },
                    cache_offset: request.start_position,
                    scales: ops::Qwen35KvCacheScales {
                        key: scales.key,
                        value: scales.value,
                    },
                    rope_theta,
                },
            )?;
            if profile_attention {
                ctx.synchronize().map_err(Error::Cuda)?;
                eprintln!(
                    "[verify-attention] layer={} core_ms={:.3}",
                    request.layer_index,
                    attention_core_started.elapsed().as_secs_f64() * 1000.0
                );
            }
            let output_projection_started = std::time::Instant::now();

            let attention_output = view(
                &scratch.attention_output,
                vec![
                    tokens,
                    ops::QWEN35_ATTENTION_Q_HEADS * ops::QWEN35_ATTENTION_HEAD_DIM,
                ],
                DType::BF16,
            )?;
            let mut output = request.output.clone();
            let mut project_output = || {
                quantize_static_fp8(
                    ctx,
                    &attention_output,
                    &scratch.fp8_output_input,
                    request.weights.o_proj.input_scale,
                )?;
                project_static_fp8_prequantized(
                    ctx,
                    &request.weights.o_proj,
                    &scratch.fp8_output_input,
                    &mut output,
                    tokens,
                )
            };
            let result = if let Some(session) = post_projection {
                session.run(project_output)
            } else {
                project_output()
            };
            if profile_attention {
                ctx.synchronize().map_err(Error::Cuda)?;
                eprintln!(
                    "[verify-attention] layer={} output_ms={:.3}",
                    request.layer_index,
                    output_projection_started.elapsed().as_secs_f64() * 1000.0
                );
            }
            return result;
        }

        // ---- Decomposed pre/post processing (shared with old scalar path) ----
        // Split fused query+gate → query / gate [tokens, Q_HEADS, HEAD_DIM]
        let q_fused = view(
            &scratch.projected_query_gate,
            vec![
                tokens,
                ops::QWEN35_ATTENTION_Q_HEADS,
                2 * ops::QWEN35_ATTENTION_HEAD_DIM,
            ],
            DType::BF16,
        )?;
        ops::split_query_and_gate(ctx, &q_fused, &scratch.query, &scratch.gate)?;

        // Head RMSNorm: the kernel wants [rows, head_dim], so flatten the head
        // dimension before calling, then use the 3D views for the RoPE that
        // follows.
        let q_2d = view(
            &scratch.query,
            vec![
                tokens * ops::QWEN35_ATTENTION_Q_HEADS,
                ops::QWEN35_ATTENTION_HEAD_DIM,
            ],
            DType::BF16,
        )?;
        ops::head_rms_norm(
            ctx,
            &q_2d,
            &request.weights.q_norm,
            ops::QWEN35_ATTENTION_RMS_EPSILON,
        )?;

        let key_heads = view(
            &scratch.projected_key,
            vec![
                tokens,
                ops::QWEN35_ATTENTION_KV_HEADS,
                ops::QWEN35_ATTENTION_HEAD_DIM,
            ],
            DType::BF16,
        )?;
        let key_2d = view(
            &scratch.projected_key,
            vec![
                tokens * ops::QWEN35_ATTENTION_KV_HEADS,
                ops::QWEN35_ATTENTION_HEAD_DIM,
            ],
            DType::BF16,
        )?;
        ops::head_rms_norm(
            ctx,
            &key_2d,
            &request.weights.k_norm,
            ops::QWEN35_ATTENTION_RMS_EPSILON,
        )?;

        // Partial RoPE (mRoPE) on query and key.
        // `partial_rope` expects 1-D positions [tokens]; `ensure_text_positions`
        // fills [3, tokens] for mRoPE, whose three channels are identical in
        // text mode.  Take the first channel as a view.
        let pos_bytes = tokens * DType::I32.size_in_bytes();
        let positions_1d = CudaBuffer::from_tensor(&scratch.positions)
            .map_err(Error::Cuda)?
            .view(0, pos_bytes)
            .map_err(Error::Cuda)?
            .as_tensor(Shape::new(vec![tokens]), DType::I32)
            .map_err(Error::Cuda)?;
        ops::partial_rope(
            ctx,
            &scratch.query,
            &positions_1d,
            ops::QWEN35_ATTENTION_ROTARY_DIM,
            rope_theta,
        )?;
        ops::partial_rope(
            ctx,
            &key_heads,
            &positions_1d,
            ops::QWEN35_ATTENTION_ROTARY_DIM,
            rope_theta,
        )?;

        // ---- Append K/V using the selected runtime cache representation ----
        let cache_dtype = cache.dtype.tensor_dtype();
        let kv_elements = tokens * ops::QWEN35_ATTENTION_KV_WIDTH;
        let kv_bytes = kv_elements * cache_dtype.size_in_bytes();
        let cache_offset_bytes =
            request.start_position * ops::QWEN35_ATTENTION_KV_WIDTH * cache_dtype.size_in_bytes();
        let scales = cache.scales;
        for (src, dst_tensor, key) in [
            (&scratch.projected_key, &cache.key, true),
            (&scratch.projected_value, &cache.value, false),
        ] {
            let src_view = CudaBuffer::from_tensor(src)
                .map_err(Error::Cuda)?
                .as_tensor(
                    Shape::new(vec![tokens, ops::QWEN35_ATTENTION_KV_WIDTH]),
                    DType::BF16,
                )
                .map_err(Error::Cuda)?;
            let mut dst_view = CudaBuffer::from_tensor(dst_tensor)
                .map_err(Error::Cuda)?
                .view(cache_offset_bytes, kv_bytes)
                .map_err(Error::Cuda)?
                .as_tensor(
                    Shape::new(vec![tokens, ops::QWEN35_ATTENTION_KV_WIDTH]),
                    cache_dtype,
                )
                .map_err(Error::Cuda)?;
            match cache.dtype {
                Qwen35KvCacheDType::Bf16 => {
                    transfers::copy_cuda_to_cuda(ctx, &src_view, &dst_view)?;
                }
                Qwen35KvCacheDType::Fp8 => {
                    let scales = scales.ok_or_else(|| {
                        invalid(
                            "Qwen3.6 FP8 KV cache requested without checkpoint calibration scales",
                        )
                    })?;
                    let scale = if key { scales.key } else { scales.value };
                    let mut args = ops::QuantizationArgs::new(
                        ops::QuantizationSemantic::FixedScaleE4m3,
                        &src_view,
                        &mut dst_view,
                    );
                    args.scale = scale;
                    ops::quantization(ctx, args)?;
                }
                Qwen35KvCacheDType::Fp8Dynamic => unreachable!("dynamic path returned above"),
            }
        }

        // ---- FA2 attention over the full prefix ----
        let valid = request.start_position + tokens;
        let (key_prefix, value_prefix) = match cache.dtype {
            Qwen35KvCacheDType::Bf16 => (
                cache_prefix_view(
                    &cache.key,
                    valid,
                    ops::QWEN35_ATTENTION_KV_HEADS,
                    ops::QWEN35_ATTENTION_HEAD_DIM,
                    DType::BF16,
                )?,
                cache_prefix_view(
                    &cache.value,
                    valid,
                    ops::QWEN35_ATTENTION_KV_HEADS,
                    ops::QWEN35_ATTENTION_HEAD_DIM,
                    DType::BF16,
                )?,
            ),
            Qwen35KvCacheDType::Fp8 => {
                let key_fp8 = cache_prefix_view(
                    &cache.key,
                    valid,
                    ops::QWEN35_ATTENTION_KV_HEADS,
                    ops::QWEN35_ATTENTION_HEAD_DIM,
                    DType::F8E4M3,
                )?;
                let value_fp8 = cache_prefix_view(
                    &cache.value,
                    valid,
                    ops::QWEN35_ATTENTION_KV_HEADS,
                    ops::QWEN35_ATTENTION_HEAD_DIM,
                    DType::F8E4M3,
                )?;
                let mut key_bf16 = scratch.fp8_prefix_view(true, valid)?;
                let mut value_bf16 = scratch.fp8_prefix_view(false, valid)?;
                let scales = scales.ok_or_else(|| {
                    invalid("Qwen3.6 FP8 KV cache requested without checkpoint calibration scales")
                })?;
                for (input, output, scale) in [
                    (&key_fp8, &mut key_bf16, scales.key),
                    (&value_fp8, &mut value_bf16, scales.value),
                ] {
                    let mut args = ops::QuantizationArgs::new(
                        ops::QuantizationSemantic::FixedScaleE4m3ToBf16,
                        input,
                        output,
                    );
                    args.scale = scale;
                    ops::quantization(ctx, args)?;
                }
                (key_bf16, value_bf16)
            }
            Qwen35KvCacheDType::Fp8Dynamic => unreachable!("dynamic path returned above"),
        };
        let query_fa2 = view(
            &scratch.query,
            vec![
                1,
                tokens,
                ops::QWEN35_ATTENTION_Q_HEADS,
                ops::QWEN35_ATTENTION_HEAD_DIM,
            ],
            DType::BF16,
        )?;
        let mut attn_out = view(
            &scratch.attention_output,
            vec![
                1,
                tokens,
                ops::QWEN35_ATTENTION_Q_HEADS,
                ops::QWEN35_ATTENTION_HEAD_DIM,
            ],
            DType::BF16,
        )?;
        let scale = 1.0 / (ops::QWEN35_ATTENTION_HEAD_DIM as f32).sqrt();
        let mut fa2_args =
            ops::KvCacheAttentionArgs::new(&query_fa2, &key_prefix, &value_prefix, &mut attn_out);
        fa2_args.valid_key_tokens = valid;
        fa2_args.query_start = request.start_position;
        fa2_args.scale = scale;
        // An unseen append length can use fixed-config FA2 without tuning.
        // The native adapter selects FA2 before its scalar correctness fallback.
        fa2_args.policy.allow_fallback = true;
        // The live KV view changes on every decode step. Caching this execution
        // would require a capacity-shaped dynamic binding, which excludes FA2
        // and falls back to the scalar correctness kernel (seconds/token at
        // AIOS context lengths). Prefix snapshots do not require that execution
        // cache: keep the stable projection sessions, but launch attention from
        // the exact live view so the optimized FA2 provider remains eligible.
        ops::kv_cache_attention(ctx, fa2_args)?;

        // ---- Output gate (sigmoid, not silu) ----
        let gate_2d = view(
            &scratch.gate,
            vec![
                tokens * ops::QWEN35_ATTENTION_Q_HEADS,
                ops::QWEN35_ATTENTION_HEAD_DIM,
            ],
            DType::BF16,
        )?;
        let output_2d = view(
            &scratch.attention_output,
            vec![
                tokens * ops::QWEN35_ATTENTION_Q_HEADS,
                ops::QWEN35_ATTENTION_HEAD_DIM,
            ],
            DType::BF16,
        )?;
        ops::apply_output_gate(ctx, &output_2d, &gate_2d)?;

        // ---- o_proj GEMM ----
        let attention_output = view(
            &scratch.attention_output,
            vec![
                tokens,
                ops::QWEN35_ATTENTION_Q_HEADS * ops::QWEN35_ATTENTION_HEAD_DIM,
            ],
            DType::BF16,
        )?;
        let mut output = request.output.clone();
        let mut project_output = || {
            quantize_static_fp8(
                ctx,
                &attention_output,
                &scratch.fp8_output_input,
                request.weights.o_proj.input_scale,
            )?;
            project_static_fp8_prequantized(
                ctx,
                &request.weights.o_proj,
                &scratch.fp8_output_input,
                &mut output,
                tokens,
            )
        };
        if let Some(session) = post_projection {
            session.run(project_output)
        } else {
            project_output()
        }
    }
}

/// The dynamic mode owns a separate path, leaving BF16 and calibrated FP8
/// dispatch identical to the 36d0d0f baseline.
fn run_dynamic_fp8_attention(
    ctx: &CudaContext,
    request: &FullAttentionRequest<'_>,
    cache: &Qwen35AttentionCache,
    scratch: &AttentionScratch,
    tokens: usize,
    rope_theta: f32,
    post_projection: Option<Rc<PreparedExecution>>,
) -> Result<()> {
    let query_gate = view(
        &scratch.projected_query_gate,
        vec![
            tokens,
            ops::QWEN35_ATTENTION_Q_HEADS,
            2 * ops::QWEN35_ATTENTION_HEAD_DIM,
        ],
        DType::BF16,
    )?;
    ops::qwen35_query_prepare(
        ctx,
        tokens,
        &query_gate,
        &request.weights.q_norm,
        &scratch.positions,
        rope_theta,
        &scratch.query,
        &scratch.gate,
    )?;
    let key_heads = view(
        &scratch.projected_key,
        vec![
            tokens,
            ops::QWEN35_ATTENTION_KV_HEADS,
            ops::QWEN35_ATTENTION_HEAD_DIM,
        ],
        DType::BF16,
    )?;
    let value_heads = view(
        &scratch.projected_value,
        vec![
            tokens,
            ops::QWEN35_ATTENTION_KV_HEADS,
            ops::QWEN35_ATTENTION_HEAD_DIM,
        ],
        DType::BF16,
    )?;
    ops::qwen35_dynamic_kv_prepare(
        ctx,
        tokens,
        request.start_position,
        &key_heads,
        &value_heads,
        &request.weights.k_norm,
        &scratch.positions,
        rope_theta,
        cache.dynamic_fp8_cache()?,
    )?;

    let valid = request.start_position + tokens;
    if tokens <= ops::QWEN35_MTP_DRAFT_STEPS + 1 {
        ops::qwen35_dynamic_decode_attention_gated(
            ctx,
            valid,
            &scratch.query,
            &scratch.gate,
            cache.dynamic_fp8_cache()?,
            scratch.dynamic_decode_scratch()?,
            &scratch.attention_output,
        )?;
    } else {
        let key_bf16 = scratch.fp8_prefix_view(true, valid)?;
        let value_bf16 = scratch.fp8_prefix_view(false, valid)?;
        ops::qwen35_dynamic_kv_dequant(
            ctx,
            valid,
            cache.dynamic_fp8_cache()?,
            &key_bf16,
            &value_bf16,
        )?;
        let query = view(
            &scratch.query,
            vec![
                1,
                tokens,
                ops::QWEN35_ATTENTION_Q_HEADS,
                ops::QWEN35_ATTENTION_HEAD_DIM,
            ],
            DType::BF16,
        )?;
        let mut output = view(
            &scratch.attention_output,
            vec![
                1,
                tokens,
                ops::QWEN35_ATTENTION_Q_HEADS,
                ops::QWEN35_ATTENTION_HEAD_DIM,
            ],
            DType::BF16,
        )?;
        if request.phase == DecoderPhase::Prefill
            && ops::QWEN35_SUFFIX_FMHA_AVAILABLE
            && ctx.device_id() == 0
        {
            let _range = crate::profiling::trace::range("qwen35_suffix_fmha");
            ops::qwen35_suffix_fmha(ctx, &query, &key_bf16, &value_bf16, &mut output)?;
        } else {
            let mut args =
                ops::KvCacheAttentionArgs::new(&query, &key_bf16, &value_bf16, &mut output);
            args.valid_key_tokens = valid;
            args.query_start = request.start_position;
            args.scale = 1.0 / (ops::QWEN35_ATTENTION_HEAD_DIM as f32).sqrt();
            args.policy.allow_fallback = true;
            ops::kv_cache_attention(ctx, args)?;
        }
        let gate = view(
            &scratch.gate,
            vec![
                tokens * ops::QWEN35_ATTENTION_Q_HEADS,
                ops::QWEN35_ATTENTION_HEAD_DIM,
            ],
            DType::BF16,
        )?;
        let output = view(
            &scratch.attention_output,
            vec![
                tokens * ops::QWEN35_ATTENTION_Q_HEADS,
                ops::QWEN35_ATTENTION_HEAD_DIM,
            ],
            DType::BF16,
        )?;
        ops::apply_output_gate(ctx, &output, &gate)?;
    }

    let attention_output = view(
        &scratch.attention_output,
        vec![
            tokens,
            ops::QWEN35_ATTENTION_Q_HEADS * ops::QWEN35_ATTENTION_HEAD_DIM,
        ],
        DType::BF16,
    )?;
    let mut output = request.output.clone();
    let mut project_output = || {
        quantize_static_fp8(
            ctx,
            &attention_output,
            &scratch.fp8_output_input,
            request.weights.o_proj.input_scale,
        )?;
        project_static_fp8_prequantized(
            ctx,
            &request.weights.o_proj,
            &scratch.fp8_output_input,
            &mut output,
            tokens,
        )
    };
    if let Some(session) = post_projection {
        session.run(project_output)
    } else {
        project_output()
    }
}

struct AttentionScratch {
    fp8_projection_input: Tensor,
    fp8_output_input: Tensor,
    projected_query_gate: Tensor,
    projected_key: Tensor,
    projected_value: Tensor,
    positions: Tensor,
    query: Tensor,
    gate: Tensor,
    attention_output: Tensor,
    fp8_scores: Tensor,
    fp8_partial_output: Tensor,
    fp8_partial_stats: Tensor,
    fp8_key_bf16: Option<Tensor>,
    fp8_value_bf16: Option<Tensor>,
    dynamic_partial_max: Option<Tensor>,
    dynamic_partial_sum: Option<Tensor>,
    dynamic_partial_output: Option<Tensor>,
    tokens: usize,
    positions_start: Option<usize>,
}

impl AttentionScratch {
    fn new(
        ctx: &CudaContext,
        tokens: usize,
        cache_capacity: usize,
        cache_dtype: Qwen35KvCacheDType,
        phase: DecoderPhase,
    ) -> Result<Self> {
        if tokens == 0
            || cache_capacity == 0
            || i32::try_from(tokens).is_err()
            || i32::try_from(cache_capacity).is_err()
        {
            return Err(invalid("Qwen3.5 attention profile token count is invalid"));
        }
        let attention_shape = vec![
            tokens,
            ops::QWEN35_ATTENTION_Q_HEADS,
            ops::QWEN35_ATTENTION_HEAD_DIM,
        ];
        let (_, physical_cache_capacity) = cache_block_geometry(cache_capacity)?;
        // The custom static-FP8 attention kernel is only selected for scalar
        // decode and the fixed MTP verifier. Prompt prefill uses FA2 and never
        // touches its score/split scratch. Do not reserve O(tokens * capacity)
        // dead storage for a 4K+ prefill profile (over 1 GiB at 4K/5K).
        let fp8_direct_tokens = if cache_dtype == Qwen35KvCacheDType::Fp8
            && tokens <= ops::QWEN35_MTP_DRAFT_STEPS + 1
        {
            tokens
        } else {
            1
        };
        let fp8_cache_shape = vec![
            cache_capacity,
            ops::QWEN35_ATTENTION_KV_HEADS,
            ops::QWEN35_ATTENTION_HEAD_DIM,
        ];
        let (fp8_key_bf16, fp8_value_bf16) = if cache_dtype == Qwen35KvCacheDType::Fp8
            || (cache_dtype == Qwen35KvCacheDType::Fp8Dynamic
                && tokens > ops::QWEN35_MTP_DRAFT_STEPS + 1)
        {
            (
                Some(zeros(ctx, fp8_cache_shape.clone(), DType::BF16)?),
                Some(zeros(ctx, fp8_cache_shape, DType::BF16)?),
            )
        } else {
            (None, None)
        };
        let (dynamic_partial_max, dynamic_partial_sum, dynamic_partial_output) =
            if cache_dtype == Qwen35KvCacheDType::Fp8Dynamic
                && tokens <= ops::QWEN35_MTP_DRAFT_STEPS + 1
            {
                let max_splits = ops::qwen35_dynamic_decode_max_splits(physical_cache_capacity)?;
                let stats_shape = vec![tokens * ops::QWEN35_ATTENTION_Q_HEADS, max_splits];
                (
                    Some(zeros(ctx, stats_shape.clone(), DType::F32)?),
                    Some(zeros(ctx, stats_shape, DType::F32)?),
                    Some(zeros(
                        ctx,
                        vec![
                            tokens * ops::QWEN35_ATTENTION_Q_HEADS,
                            max_splits,
                            ops::QWEN35_ATTENTION_HEAD_DIM,
                        ],
                        DType::F32,
                    )?),
                )
            } else {
                (None, None, None)
            };
        let static_fp8_capacity = if cache_dtype == Qwen35KvCacheDType::Fp8Dynamic {
            1
        } else {
            physical_cache_capacity
        };
        let static_fp8_splits = if cache_dtype == Qwen35KvCacheDType::Fp8Dynamic {
            1
        } else {
            ops::QWEN35_ATTENTION_MAX_SPLITS
        };
        Ok(Self {
            fp8_projection_input: zeros(ctx, vec![tokens, TARGET_HIDDEN_SIZE], DType::F8E4M3)?,
            fp8_output_input: zeros(
                ctx,
                vec![
                    tokens,
                    ops::QWEN35_ATTENTION_Q_HEADS * ops::QWEN35_ATTENTION_HEAD_DIM,
                ],
                DType::F8E4M3,
            )?,
            projected_query_gate: zeros(
                ctx,
                vec![tokens, ops::QWEN35_ATTENTION_Q_PROJECTION_WIDTH],
                DType::BF16,
            )?,
            projected_key: zeros(
                ctx,
                vec![tokens, ops::QWEN35_ATTENTION_KV_WIDTH],
                DType::BF16,
            )?,
            projected_value: zeros(
                ctx,
                vec![tokens, ops::QWEN35_ATTENTION_KV_WIDTH],
                DType::BF16,
            )?,
            positions: zeros(ctx, vec![TEXT_POSITION_CHANNELS, tokens], DType::I32)?,
            query: zeros(ctx, attention_shape.clone(), DType::BF16)?,
            gate: zeros(ctx, attention_shape.clone(), DType::BF16)?,
            attention_output: zeros(ctx, attention_shape, DType::BF16)?,
            fp8_scores: zeros(
                ctx,
                vec![
                    fp8_direct_tokens,
                    ops::QWEN35_ATTENTION_Q_HEADS,
                    static_fp8_capacity,
                ],
                DType::F32,
            )?,
            fp8_partial_output: zeros(
                ctx,
                vec![
                    fp8_direct_tokens,
                    ops::QWEN35_ATTENTION_KV_HEADS,
                    static_fp8_splits,
                    ops::QWEN35_ATTENTION_Q_HEADS / ops::QWEN35_ATTENTION_KV_HEADS,
                    ops::QWEN35_ATTENTION_HEAD_DIM,
                ],
                DType::F32,
            )?,
            fp8_partial_stats: zeros(
                ctx,
                vec![
                    fp8_direct_tokens,
                    ops::QWEN35_ATTENTION_KV_HEADS,
                    static_fp8_splits,
                    ops::QWEN35_ATTENTION_Q_HEADS / ops::QWEN35_ATTENTION_KV_HEADS,
                    2,
                ],
                DType::F32,
            )?,
            fp8_key_bf16,
            fp8_value_bf16,
            dynamic_partial_max,
            dynamic_partial_sum,
            dynamic_partial_output,
            tokens,
            positions_start: None,
        })
    }

    fn ensure_text_positions(&mut self, ctx: &CudaContext, start_position: usize) -> Result<()> {
        if self.positions_start == Some(start_position) {
            return Ok(());
        }
        let values = text_position_values(start_position, self.tokens)?;
        let host = Tensor::from_i32(vec![TEXT_POSITION_CHANNELS, self.tokens], &values)?;
        transfers::copy_cpu_to_cuda_on_stream(ctx, host, &self.positions)?;
        self.positions_start = Some(start_position);
        Ok(())
    }

    fn fp8_prefix_view(&self, key: bool, valid: usize) -> Result<Tensor> {
        let tensor = if key {
            self.fp8_key_bf16.as_ref()
        } else {
            self.fp8_value_bf16.as_ref()
        }
        .ok_or_else(|| invalid("Qwen3.5 FP8 cache dequantization scratch is unavailable"))?;
        cache_prefix_view(
            tensor,
            valid,
            ops::QWEN35_ATTENTION_KV_HEADS,
            ops::QWEN35_ATTENTION_HEAD_DIM,
            DType::BF16,
        )
    }

    fn dynamic_decode_scratch(&self) -> Result<ops::Qwen35DynamicDecodeScratch<'_>> {
        Ok(ops::Qwen35DynamicDecodeScratch {
            partial_max: self
                .dynamic_partial_max
                .as_ref()
                .ok_or_else(|| invalid("missing dynamic decode max scratch"))?,
            partial_sum: self
                .dynamic_partial_sum
                .as_ref()
                .ok_or_else(|| invalid("missing dynamic decode sum scratch"))?,
            partial_output: self
                .dynamic_partial_output
                .as_ref()
                .ok_or_else(|| invalid("missing dynamic decode output scratch"))?,
        })
    }
}

fn validate_request(
    ctx: &CudaContext,
    hidden_size: usize,
    request: &FullAttentionRequest<'_>,
    cache: &Qwen35AttentionCache,
) -> Result<usize> {
    validate_geometry(request.geometry)?;
    validate_cache_device(ctx, cache)?;
    if cache.layer_index != request.layer_index || cache.geometry != request.geometry {
        return Err(invalid(
            "Qwen3.5 attention request does not match its layer cache",
        ));
    }
    let input_shape = request.normalized_hidden.shape().dims();
    if input_shape.len() != 2 || input_shape[1] != hidden_size {
        return Err(invalid(format!(
            "Qwen3.5 normalized hidden expected [tokens, {hidden_size}], got {input_shape:?}"
        )));
    }
    let tokens = input_shape[0];
    validate_phase_tokens(request.phase, tokens)?;
    let end = request
        .start_position
        .checked_add(tokens)
        .ok_or_else(|| invalid("Qwen3.5 attention position range overflow"))?;
    if end > cache.capacity || i32::try_from(end).is_err() {
        return Err(invalid("Qwen3.5 attention append exceeds cache capacity"));
    }
    validate_tensor(
        request.normalized_hidden,
        ctx,
        DType::BF16,
        &[tokens, hidden_size],
        "normalized hidden",
    )?;
    validate_tensor(
        request.output,
        ctx,
        DType::BF16,
        &[tokens, hidden_size],
        "mixer output",
    )?;
    validate_attention_weights(ctx, request.weights, hidden_size)?;
    if request.weights.kv_cache_scales != cache.scales {
        return Err(invalid(
            "Qwen3.5 attention weights and cache metadata use different K/V scales",
        ));
    }
    Ok(tokens)
}

fn validate_attention_weights(
    ctx: &CudaContext,
    weights: &DeviceAttentionWeights<DeviceStaticFp8Linear>,
    hidden_size: usize,
) -> Result<()> {
    for (linear, input, output, label) in [
        (
            &weights.q_proj,
            hidden_size,
            ops::QWEN35_ATTENTION_Q_PROJECTION_WIDTH,
            "q+gate projection",
        ),
        (
            &weights.k_proj,
            hidden_size,
            ops::QWEN35_ATTENTION_KV_WIDTH,
            "key projection",
        ),
        (
            &weights.v_proj,
            hidden_size,
            ops::QWEN35_ATTENTION_KV_WIDTH,
            "value projection",
        ),
        (
            &weights.o_proj,
            ops::QWEN35_ATTENTION_Q_HEADS * ops::QWEN35_ATTENTION_HEAD_DIM,
            hidden_size,
            "output projection",
        ),
    ] {
        validate_fp8_linear(ctx, linear, input, output, label)?;
    }
    validate_tensor(
        &weights.q_norm,
        ctx,
        DType::BF16,
        &[ops::QWEN35_ATTENTION_HEAD_DIM],
        "zero-centered query norm",
    )?;
    validate_tensor(
        &weights.k_norm,
        ctx,
        DType::BF16,
        &[ops::QWEN35_ATTENTION_HEAD_DIM],
        "zero-centered key norm",
    )
}

fn validate_fp8_linear(
    ctx: &CudaContext,
    linear: &DeviceStaticFp8Linear,
    input_features: usize,
    output_features: usize,
    label: &str,
) -> Result<()> {
    if linear.input_features != input_features || linear.output_features != output_features {
        return Err(invalid(format!(
            "Qwen3.5 {label} expected {input_features}->{output_features}, got {}->{}",
            linear.input_features, linear.output_features
        )));
    }
    validate_tensor(
        &linear.weight,
        ctx,
        DType::F8E4M3,
        &[output_features, input_features],
        label,
    )?;
    validate_tensor(
        &linear.transposed,
        ctx,
        DType::F8E4M3,
        &[input_features, output_features],
        label,
    )
}

fn validate_cache_contract(
    geometry: AttentionGeometry,
    capacity: usize,
    scales: Option<KvCacheScales>,
    dtype: Qwen35KvCacheDType,
) -> Result<()> {
    validate_geometry(geometry)?;
    if capacity == 0 || i32::try_from(capacity).is_err() {
        return Err(invalid("Qwen3.5 BF16 cache capacity is invalid"));
    }
    match (dtype, scales) {
        (Qwen35KvCacheDType::Bf16, None) => Ok(()),
        (Qwen35KvCacheDType::Fp8Dynamic, None) => Ok(()),
        (Qwen35KvCacheDType::Fp8, Some(scales)) => validate_scales(scales),
        (Qwen35KvCacheDType::Fp8, None) => Err(invalid(
            "Qwen3.6 checkpoint has no FP8 KV calibration scales; use BF16 KV cache",
        )),
        (Qwen35KvCacheDType::Bf16, Some(_)) => Err(invalid(
            "Qwen3.6 BF16 KV cache must not carry fabricated FP8 scales",
        )),
        (Qwen35KvCacheDType::Fp8Dynamic, Some(_)) => Err(invalid(
            "Qwen3.6 dynamic FP8 KV cache must not carry fixed scales",
        )),
    }
}

fn validate_full_attention_layer(layer_index: usize) -> Result<()> {
    if layer_index >= TARGET_LAYERS || (layer_index + 1) % FULL_ATTENTION_INTERVAL != 0 {
        return Err(invalid(format!(
            "Qwen3.5 layer {layer_index} is not one of the ten base full-attention layers"
        )));
    }
    Ok(())
}

fn validate_geometry(geometry: AttentionGeometry) -> Result<()> {
    let exact = geometry.query_heads == ops::QWEN35_ATTENTION_Q_HEADS
        && geometry.kv_heads == ops::QWEN35_ATTENTION_KV_HEADS
        && geometry.head_dim == ops::QWEN35_ATTENTION_HEAD_DIM
        && geometry.rotary_dim == ops::QWEN35_ATTENTION_ROTARY_DIM
        && geometry.query_projection_width == ops::QWEN35_ATTENTION_Q_PROJECTION_WIDTH
        && geometry.kv_projection_width == ops::QWEN35_ATTENTION_KV_WIDTH
        && geometry.output_gate;
    if !exact {
        return Err(invalid("unsupported Qwen3.5 full-attention geometry"));
    }
    Ok(())
}

fn validate_scales(scales: KvCacheScales) -> Result<()> {
    // These are already dequantization scales. Do not divide by 448 here.
    ops::Qwen35KvCacheScales {
        key: scales.key,
        value: scales.value,
    }
    .validate()
}

fn validate_phase_tokens(phase: DecoderPhase, tokens: usize) -> Result<()> {
    if tokens == 0 || i32::try_from(tokens).is_err() {
        return Err(invalid("Qwen3.5 attention input must contain tokens"));
    }
    if phase == DecoderPhase::Decode && tokens != 1 {
        return Err(invalid(
            "Qwen3.5 full-attention decode requires exactly one token",
        ));
    }
    Ok(())
}

fn validate_cache_device(ctx: &CudaContext, cache: &Qwen35AttentionCache) -> Result<()> {
    let dtype = cache.dtype.tensor_dtype();
    let (blocks, _) = cache_block_geometry(cache.capacity)?;
    if cache.blocks != blocks {
        return Err(invalid("Qwen3.5 cache block count does not match capacity"));
    }
    validate_tensor(
        &cache.key,
        ctx,
        dtype,
        &[
            cache.blocks,
            QWEN35_KV_CACHE_BLOCK_TOKENS,
            ops::QWEN35_ATTENTION_KV_HEADS,
            ops::QWEN35_ATTENTION_HEAD_DIM,
        ],
        "key cache",
    )?;
    validate_tensor(
        &cache.value,
        ctx,
        dtype,
        &[
            cache.blocks,
            QWEN35_KV_CACHE_BLOCK_TOKENS,
            ops::QWEN35_ATTENTION_KV_HEADS,
            ops::QWEN35_ATTENTION_HEAD_DIM,
        ],
        "value cache",
    )?;
    if cache.dtype == Qwen35KvCacheDType::Fp8Dynamic {
        let scale_shape = [
            cache.blocks,
            QWEN35_KV_CACHE_BLOCK_TOKENS,
            ops::QWEN35_ATTENTION_KV_HEADS,
        ];
        validate_tensor(
            cache
                .dynamic_key_scales
                .as_ref()
                .ok_or_else(|| invalid("missing dynamic K scales"))?,
            ctx,
            DType::F32,
            &scale_shape,
            "dynamic K scales",
        )?;
        validate_tensor(
            cache
                .dynamic_value_scales
                .as_ref()
                .ok_or_else(|| invalid("missing dynamic V scales"))?,
            ctx,
            DType::F32,
            &scale_shape,
            "dynamic V scales",
        )?;
    } else if cache.dynamic_key_scales.is_some() || cache.dynamic_value_scales.is_some() {
        return Err(invalid("non-dynamic cache must not carry per-token scales"));
    }
    Ok(())
}

fn validate_cache_pair(
    ctx: &CudaContext,
    source: &Qwen35AttentionCache,
    destination: &Qwen35AttentionCache,
    positions: &Range<usize>,
) -> Result<()> {
    validate_cache_device(ctx, source)?;
    validate_cache_device(ctx, destination)?;
    if source.layer_index != destination.layer_index
        || source.geometry != destination.geometry
        || source.scales != destination.scales
        || source.dtype != destination.dtype
    {
        return Err(invalid(
            "Qwen3.5 cache copy requires matching dtype, layer, geometry and metadata",
        ));
    }
    // Prefix snapshots are intentionally sized to the reusable prefix while
    // the live session is sized to prompt + output admission.  Their physical
    // block counts therefore need not match; each token row has the same
    // contiguous layout and the range check below proves that both buffers
    // contain the copied rows.
    if positions.start > positions.end
        || positions.end > source.capacity
        || positions.end > destination.capacity
    {
        return Err(invalid("Qwen3.5 cache copy range is out of bounds"));
    }
    Ok(())
}

fn cache_block_geometry(capacity: usize) -> Result<(usize, usize)> {
    if capacity == 0 {
        return Err(invalid("Qwen3.5 cache capacity must be non-zero"));
    }
    let blocks = capacity
        .checked_add(QWEN35_KV_CACHE_BLOCK_TOKENS - 1)
        .ok_or_else(|| invalid("Qwen3.5 cache block rounding overflow"))?
        / QWEN35_KV_CACHE_BLOCK_TOKENS;
    let physical_capacity = blocks
        .checked_mul(QWEN35_KV_CACHE_BLOCK_TOKENS)
        .ok_or_else(|| invalid("Qwen3.5 physical cache capacity overflow"))?;
    Ok((blocks, physical_capacity))
}

fn cache_byte_range(positions: Range<usize>, dtype: Qwen35KvCacheDType) -> Result<(usize, usize)> {
    if positions.start > positions.end {
        return Err(invalid("Qwen3.5 cache copy range is reversed"));
    }
    let row_bytes = ops::QWEN35_ATTENTION_KV_HEADS
        .checked_mul(ops::QWEN35_ATTENTION_HEAD_DIM)
        .and_then(|elements| elements.checked_mul(dtype.tensor_dtype().size_in_bytes()))
        .ok_or_else(|| invalid("Qwen3.5 cache row size overflow"))?;
    let offset = positions
        .start
        .checked_mul(row_bytes)
        .ok_or_else(|| invalid("Qwen3.5 cache copy offset overflow"))?;
    let len = (positions.end - positions.start)
        .checked_mul(row_bytes)
        .ok_or_else(|| invalid("Qwen3.5 cache copy length overflow"))?;
    Ok((offset, len))
}

fn text_position_values(start_position: usize, tokens: usize) -> Result<Vec<i32>> {
    validate_phase_tokens(DecoderPhase::Prefill, tokens)?;
    let end = start_position
        .checked_add(tokens)
        .ok_or_else(|| invalid("Qwen3.5 text position range overflow"))?;
    if end > i32::MAX as usize + 1 {
        return Err(invalid("Qwen3.5 text positions exceed i32"));
    }
    let row = (start_position..end)
        .map(|position| {
            i32::try_from(position)
                .map_err(|_| invalid("Qwen3.5 text position does not fit in i32"))
        })
        .collect::<Result<Vec<_>>>()?;
    let mut values = Vec::with_capacity(TEXT_POSITION_CHANNELS * tokens);
    for _ in 0..TEXT_POSITION_CHANNELS {
        values.extend_from_slice(&row);
    }
    Ok(values)
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
        .ok_or_else(|| invalid("Qwen3.5 attention CUDA allocation size overflow"))?;
    Ok(CudaBuffer::alloc_zeros_async(bytes, ctx.device_id(), ctx.stream())
        .map_err(Error::Cuda)?
        .as_tensor(Shape::new(shape), dtype)
        .map_err(Error::Cuda)?)
}

fn view(tensor: &Tensor, shape: Vec<usize>, dtype: DType) -> Result<Tensor> {
    CudaBuffer::from_tensor(tensor)
        .map_err(Error::Cuda)?
        .as_tensor(Shape::new(shape), dtype)
        .map_err(Error::Cuda)
}

/// View of the first `valid` rows of a physically contiguous
/// `[blocks, 128, kv_heads, head_dim]` cache, flattened to
/// `[1, valid, kv_heads, head_dim]` for `ops::kv_cache_attention`.
fn cache_prefix_view(
    cache: &Tensor,
    valid: usize,
    kv_heads: usize,
    head_dim: usize,
    dtype: DType,
) -> Result<Tensor> {
    let bytes = valid
        .checked_mul(kv_heads)
        .and_then(|elements| elements.checked_mul(head_dim))
        .and_then(|elements| elements.checked_mul(dtype.size_in_bytes()))
        .ok_or_else(|| invalid("Qwen3.5 cache prefix size overflow"))?;
    CudaBuffer::from_tensor(cache)
        .map_err(Error::Cuda)?
        .view(0, bytes)
        .map_err(Error::Cuda)?
        .as_tensor(Shape::new(vec![1, valid, kv_heads, head_dim]), dtype)
        .map_err(Error::Cuda)
}

fn invalid(message: impl Into<String>) -> Error {
    Error::Other(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn target_geometry() -> AttentionGeometry {
        AttentionGeometry {
            query_heads: 16,
            kv_heads: 2,
            head_dim: 256,
            rotary_dim: 64,
            query_projection_width: 8_192,
            kv_projection_width: 512,
            output_gate: true,
        }
    }

    #[test]
    fn text_positions_repeat_absolute_indices_for_all_mrope_channels() {
        assert_eq!(
            text_position_values(7, 3).unwrap(),
            vec![7, 8, 9, 7, 8, 9, 7, 8, 9]
        );
    }

    #[test]
    fn cache_range_uses_bf16_rows() {
        let row_bytes = 512 * DType::BF16.size_in_bytes();
        assert_eq!(
            cache_byte_range(4..9, Qwen35KvCacheDType::Bf16).unwrap(),
            (4 * row_bytes, 5 * row_bytes)
        );
        let fp8_row_bytes = 512 * DType::F8E4M3.size_in_bytes();
        assert_eq!(
            cache_byte_range(4..9, Qwen35KvCacheDType::Fp8).unwrap(),
            (4 * fp8_row_bytes, 5 * fp8_row_bytes)
        );
        assert_eq!(
            cache_byte_range(4..9, Qwen35KvCacheDType::Fp8Dynamic).unwrap(),
            (4 * fp8_row_bytes, 5 * fp8_row_bytes)
        );
    }

    #[test]
    fn cache_capacity_rounds_to_128_token_physical_blocks() {
        assert_eq!(QWEN35_KV_CACHE_BLOCK_TOKENS, 128);
        assert_eq!(cache_block_geometry(1).unwrap(), (1, 128));
        assert_eq!(cache_block_geometry(128).unwrap(), (1, 128));
        assert_eq!(cache_block_geometry(129).unwrap(), (2, 256));
        assert_eq!(cache_block_geometry(20_003).unwrap(), (157, 20_096));
        assert!(cache_block_geometry(0).is_err());
    }

    #[test]
    fn contract_rejects_non_target_geometry_and_checkpoint_amax_reuse() {
        let scales = KvCacheScales {
            key: 1.0 / 448.0,
            value: 224.0 / 448.0,
        };
        validate_cache_contract(
            target_geometry(),
            4_096,
            Some(scales),
            Qwen35KvCacheDType::Fp8,
        )
        .unwrap();
        let mut wrong = target_geometry();
        wrong.output_gate = false;
        assert!(
            validate_cache_contract(wrong, 4_096, Some(scales), Qwen35KvCacheDType::Fp8,).is_err()
        );
        assert!(
            validate_cache_contract(target_geometry(), 4_096, None, Qwen35KvCacheDType::Bf16,)
                .is_ok()
        );
        assert!(
            validate_cache_contract(target_geometry(), 4_096, None, Qwen35KvCacheDType::Fp8,)
                .is_err()
        );
        assert!(validate_cache_contract(
            target_geometry(),
            4_096,
            None,
            Qwen35KvCacheDType::Fp8Dynamic,
        )
        .is_ok());
        assert!(validate_cache_contract(
            target_geometry(),
            4_096,
            Some(scales),
            Qwen35KvCacheDType::Fp8Dynamic,
        )
        .is_err());

        // Backend scales are dequant multipliers and are forwarded unchanged.
        let cuda_scales = ops::Qwen35KvCacheScales {
            key: scales.key,
            value: scales.value,
        };
        assert_eq!(cuda_scales.key, 1.0 / 448.0);
        assert_eq!(cuda_scales.value, 0.5);
    }

    #[test]
    fn decode_contract_is_single_row_but_prefill_can_be_chunked() {
        validate_phase_tokens(DecoderPhase::Decode, 1).unwrap();
        assert!(validate_phase_tokens(DecoderPhase::Decode, 2).is_err());
        validate_phase_tokens(DecoderPhase::Prefill, 17).unwrap();
    }

    #[test]
    fn only_the_ten_hybrid_full_attention_layers_own_caches() {
        let layers = (0..TARGET_LAYERS)
            .filter(|&layer| validate_full_attention_layer(layer).is_ok())
            .collect::<Vec<_>>();
        assert_eq!(layers, vec![3, 7, 11, 15, 19, 23, 27, 31, 35, 39]);
        assert!(validate_full_attention_layer(40).is_err());
    }

    #[test]
    fn target_hidden_size_matches_checkpoint_contract() {
        assert_eq!(TARGET_HIDDEN_SIZE, 2_048);
    }
    /// GPU integration smoke covers real projection/backend shapes, prefill,
    /// single-token decode and scale-preserving shadow-cache copy/reset.
    #[test]
    #[ignore = "requires exclusive CUDA device"]
    fn dynamic_fp8_model_attention_wiring_and_shadow_cache() {
        let ctx = CudaContext::new(0).unwrap();
        let config = super::super::config::tests::valid_config();
        let mut backend =
            Qwen35Fp8AttentionBackend::with_kv_cache(&config, Qwen35KvCacheDType::Fp8Dynamic, None)
                .unwrap();
        let geom = target_geometry();
        let mut cache = backend.allocate_cache(&ctx, 3, geom, 32, None).unwrap();
        let mut shadow = backend.allocate_cache(&ctx, 3, geom, 32, None).unwrap();
        let linear = |input, output| {
            let weight = zeros(&ctx, vec![output, input], DType::F8E4M3).unwrap();
            let transposed = zeros(&ctx, vec![input, output], DType::F8E4M3).unwrap();
            for t in [&weight, &transposed] {
                CudaBuffer::from_tensor(t)
                    .unwrap()
                    .memset_async(0x38, ctx.stream())
                    .unwrap();
            }
            DeviceStaticFp8Linear {
                weight,
                transposed,
                input_features: input,
                output_features: output,
                input_scale: 1.0,
                alpha: 0.001,
            }
        };
        let weights = DeviceAttentionWeights {
            q_proj: linear(2048, 8192),
            k_proj: linear(2048, 512),
            v_proj: linear(2048, 512),
            o_proj: linear(4096, 2048),
            q_norm: zeros(&ctx, vec![256], DType::BF16).unwrap(),
            k_norm: zeros(&ctx, vec![256], DType::BF16).unwrap(),
            kv_cache_scales: None,
        };
        for (phase, start, tokens) in [
            (DecoderPhase::Prefill, 0, 2),
            (DecoderPhase::Decode, 2, 1),
            (DecoderPhase::Decode, 3, 1),
        ] {
            let input = zeros(&ctx, vec![tokens, 2048], DType::BF16).unwrap();
            CudaBuffer::from_tensor(&input)
                .unwrap()
                .memset_async(0x3b, ctx.stream())
                .unwrap();
            let output = zeros(&ctx, vec![tokens, 2048], DType::BF16).unwrap();
            backend
                .run(
                    &ctx,
                    FullAttentionRequest {
                        layer_index: 3,
                        phase,
                        start_position: start,
                        geometry: geom,
                        weights: &weights,
                        normalized_hidden: &input,
                        output: &output,
                    },
                    &mut cache,
                )
                .unwrap();
            ctx.synchronize().unwrap();
            let host = transfers::to_cpu(&output).unwrap();
            assert!(host
                .as_bf16()
                .unwrap()
                .iter()
                .all(|v| v.to_f32().is_finite()));
            assert!(host.as_bf16().unwrap().iter().any(|v| v.to_f32() != 0.0));
        }
        backend
            .copy_cache_range(&ctx, &cache, &mut shadow, 1..4)
            .unwrap();
        ctx.synchronize().unwrap();
        for (a, b) in [(&cache.key, &shadow.key), (&cache.value, &shadow.value)] {
            let a = transfers::to_cpu(a).unwrap();
            let b = transfers::to_cpu(b).unwrap();
            assert_eq!(
                &a.as_f8_e4m3().unwrap()[512..2048],
                &b.as_f8_e4m3().unwrap()[512..2048]
            );
            assert!(b.as_f8_e4m3().unwrap()[..512].iter().all(|&v| v == 0));
        }
        for (a, b) in [
            (cache.dynamic_key_scales(), shadow.dynamic_key_scales()),
            (cache.dynamic_value_scales(), shadow.dynamic_value_scales()),
        ] {
            let a = transfers::to_cpu(a.unwrap()).unwrap();
            let b = transfers::to_cpu(b.unwrap()).unwrap();
            assert_eq!(&a.as_f32().unwrap()[2..8], &b.as_f32().unwrap()[2..8]);
            assert!(b.as_f32().unwrap()[..2].iter().all(|&v| v == 0.0));
        }
        backend.reset_cache(&ctx, &mut shadow).unwrap();
        ctx.synchronize().unwrap();
        for tensor in [shadow.dynamic_key_scales(), shadow.dynamic_value_scales()] {
            let host = transfers::to_cpu(tensor.unwrap()).unwrap();
            assert!(host.as_f32().unwrap().iter().all(|&v| v == 0.0));
        }
    }
}
