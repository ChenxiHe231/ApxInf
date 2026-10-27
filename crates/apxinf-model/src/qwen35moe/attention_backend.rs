//! Concrete base-model full-attention backend for Qwen3.5-35B-A3B.
//!
//! The base checkpoint's ten full-attention layers use BF16 projections. Their
//! persistent K/V cache is selected at load time: BF16 is consumed directly by
//! FA2, while FP8-E4M3 is appended with the checkpoint calibration scales and
//! dequantized into retained BF16 scratch before the same FA2 implementation.
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

/// Persistent cache for one base-model full-attention layer.
#[derive(Debug)]
pub struct Qwen35AttentionCache {
    layer_index: usize,
    geometry: AttentionGeometry,
    capacity: usize,
    scales: Option<KvCacheScales>,
    dtype: Qwen35KvCacheDType,
    key: Tensor,
    value: Tensor,
}

impl Qwen35AttentionCache {
    pub fn layer_index(&self) -> usize {
        self.layer_index
    }

    pub fn capacity(&self) -> usize {
        self.capacity
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
}

/// CUDA implementation of the decoder's target-specific attention boundary.
///
/// Scratch profiles are retained by token count, so ordinary one-token decode
/// does not allocate projection or attention buffers on every step.
pub struct Qwen35Fp8AttentionBackend {
    hidden_size: usize,
    rope_theta: f32,
    kv_cache_dtype: Qwen35KvCacheDType,
    scratch: HashMap<(usize, usize, usize), AttentionScratch>,
    pre_projection_sessions: HashMap<(usize, usize), Rc<PreparedExecution>>,
    post_projection_sessions: HashMap<(usize, usize), Rc<PreparedExecution>>,
    /// The attention call sits between the two projection sessions, so it ran
    /// outside every session and `execute` then issued a full ctx.synchronize()
    /// after each enqueue, plus another when the prepared Execution dropped.
    /// That is two device synchronizes per attention layer per token.
    ///
    /// Caching it needs a stable ExecutionKey, which only became true after
    /// two other changes: the tuning key stopped moving with the live prefix,
    /// and in-place decode stopped alternating KV banks every step. With
    /// either of those missing this misses on every call and fails capture.
    attention_sessions: HashMap<(usize, usize), Rc<PreparedExecution>>,
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
        Self::with_kv_cache(config, Qwen35KvCacheDType::Bf16)
    }

    pub fn with_kv_cache(
        config: &Qwen35MoeConfig,
        kv_cache_dtype: Qwen35KvCacheDType,
    ) -> Result<Self> {
        config.validate()?;
        if kv_cache_dtype == Qwen35KvCacheDType::Fp8
            && config.quantization_config.kv_cache_scheme.is_none()
        {
            return Err(invalid(
                "Qwen3.6 checkpoint has no FP8 KV calibration scales; use BF16 KV cache",
            ));
        }
        Ok(Self {
            hidden_size: TARGET_HIDDEN_SIZE,
            rope_theta: config.text_config.rope_parameters.rope_theta,
            kv_cache_dtype,
            scratch: HashMap::new(),
            pre_projection_sessions: HashMap::new(),
            post_projection_sessions: HashMap::new(),
            attention_sessions: HashMap::new(),
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
    ) -> Result<&mut AttentionScratch> {
        let key = (ctx.device_id(), tokens, cache_capacity);
        if !self.scratch.contains_key(&key) {
            let profile = AttentionScratch::new(ctx, tokens, cache_capacity, self.kv_cache_dtype)?;
            self.scratch.insert(key, profile);
        }
        Ok(self
            .scratch
            .get_mut(&key)
            .expect("Qwen3.5 attention scratch profile was just inserted"))
    }

    fn projection_session(
        sessions: &mut HashMap<(usize, usize), Rc<PreparedExecution>>,
        ctx: &CudaContext,
        layer_index: usize,
        phase: DecoderPhase,
    ) -> Result<Option<Rc<PreparedExecution>>> {
        if phase != DecoderPhase::Decode
            || std::env::var("APXINF_QWEN35_ATTENTION_SESSIONS").as_deref() == Ok("0")
        {
            return Ok(None);
        }
        let key = (ctx.device_id(), layer_index);
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
        let shape = vec![
            capacity,
            ops::QWEN35_ATTENTION_KV_HEADS,
            ops::QWEN35_ATTENTION_HEAD_DIM,
        ];
        let dtype = self.kv_cache_dtype.tensor_dtype();
        Ok(Qwen35AttentionCache {
            layer_index,
            geometry,
            capacity,
            scales,
            dtype: self.kv_cache_dtype,
            key: zeros(ctx, shape.clone(), dtype)?,
            value: zeros(ctx, shape, dtype)?,
        })
    }

    fn reset_cache(&mut self, ctx: &CudaContext, cache: &mut Self::Cache) -> Result<()> {
        validate_cache_device(ctx, cache)?;
        for tensor in [&cache.key, &cache.value] {
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
        )?;
        let post_projection = Self::projection_session(
            &mut self.post_projection_sessions,
            ctx,
            request.layer_index,
            request.phase,
        )?;
        let attention_session = Self::projection_session(
            &mut self.attention_sessions,
            ctx,
            request.layer_index,
            request.phase,
        )?;
        let scratch = self.scratch(ctx, tokens, cache.capacity)?;
        scratch.ensure_text_positions(request.start_position)?;

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
        fa2_args.policy.allow_fallback = false;
        match attention_session {
            Some(session) => session.run(|| ops::kv_cache_attention(ctx, fa2_args))?,
            None => ops::kv_cache_attention(ctx, fa2_args)?,
        }

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

#[derive(Debug)]
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
    fp8_key_bf16: Option<Tensor>,
    fp8_value_bf16: Option<Tensor>,
    tokens: usize,
    positions_start: Option<usize>,
}

impl AttentionScratch {
    fn new(
        ctx: &CudaContext,
        tokens: usize,
        cache_capacity: usize,
        cache_dtype: Qwen35KvCacheDType,
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
        let fp8_cache_shape = vec![
            cache_capacity,
            ops::QWEN35_ATTENTION_KV_HEADS,
            ops::QWEN35_ATTENTION_HEAD_DIM,
        ];
        let (fp8_key_bf16, fp8_value_bf16) = if cache_dtype == Qwen35KvCacheDType::Fp8 {
            (
                Some(zeros(ctx, fp8_cache_shape.clone(), DType::BF16)?),
                Some(zeros(ctx, fp8_cache_shape, DType::BF16)?),
            )
        } else {
            (None, None)
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
            fp8_key_bf16,
            fp8_value_bf16,
            tokens,
            positions_start: None,
        })
    }

    fn ensure_text_positions(&mut self, start_position: usize) -> Result<()> {
        if self.positions_start == Some(start_position) {
            return Ok(());
        }
        let values = text_position_values(start_position, self.tokens)?;
        let host = Tensor::from_i32(vec![TEXT_POSITION_CHANNELS, self.tokens], &values)?;
        transfers::copy_cpu_to_cuda(&host, &self.positions)?;
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
        (Qwen35KvCacheDType::Fp8, Some(scales)) => validate_scales(scales),
        (Qwen35KvCacheDType::Fp8, None) => Err(invalid(
            "Qwen3.6 checkpoint has no FP8 KV calibration scales; use BF16 KV cache",
        )),
        (Qwen35KvCacheDType::Bf16, Some(_)) => Err(invalid(
            "Qwen3.6 BF16 KV cache must not carry fabricated FP8 scales",
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
    validate_tensor(
        &cache.key,
        ctx,
        dtype,
        &[
            cache.capacity,
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
            cache.capacity,
            ops::QWEN35_ATTENTION_KV_HEADS,
            ops::QWEN35_ATTENTION_HEAD_DIM,
        ],
        "value cache",
    )
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
    if positions.start > positions.end
        || positions.end > source.capacity
        || positions.end > destination.capacity
    {
        return Err(invalid("Qwen3.5 cache copy range is out of bounds"));
    }
    Ok(())
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
    Ok(CudaBuffer::alloc_zeros(bytes, ctx.device_id())
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

/// View of the first `valid` rows of a `[capacity, kv_heads, head_dim]` cache
/// tensor, shaped `[1, valid, kv_heads, head_dim]` for `ops::kv_cache_attention`.
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
}
