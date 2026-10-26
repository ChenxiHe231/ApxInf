//! Concrete base-model full-attention backend for Qwen3.5-35B-A3B.
//!
//! The base checkpoint's ten full-attention layers use BF16 projections and
//! static FP8-E4M3 K/V storage.  MTP attention is intentionally not handled by
//! this backend: its checkpoint subtree has BF16 K/V and a different cache
//! contract.

use std::collections::HashMap;
use std::ops::Range;

use apxinf_core::{DType, Device, Error, Result, Shape, Tensor};
use apxinf_cuda_new::{ops, transfers, CudaBuffer, CudaContext};

use super::architecture::AttentionGeometry;
use super::config::Qwen35MoeConfig;
use super::decoder::{DecoderPhase, FullAttentionBackend, FullAttentionRequest};
use super::device_weights::{DeviceAttentionWeights, DeviceDenseLinear};
use super::runtime_weights::KvCacheScales;

const TEXT_POSITION_CHANNELS: usize = 3;
const TARGET_HIDDEN_SIZE: usize = 2_048;
const TARGET_LAYERS: usize = 40;
const FULL_ATTENTION_INTERVAL: usize = 4;

/// Persistent BF16 cache for one base-model full-attention layer.
///
/// Formerly FP8-E4M3; now BF16 so `ops::kv_cache_attention` dispatches to FA2.
/// The per-tensor FP8 scales that the old scalar kernel consumed are no longer
/// needed and are kept only for the session contract — `run` does not read them.
#[derive(Debug)]
pub struct Qwen35AttentionCache {
    layer_index: usize,
    geometry: AttentionGeometry,
    capacity: usize,
    scales: KvCacheScales,
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

    pub fn scales(&self) -> KvCacheScales {
        self.scales
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
#[derive(Debug)]
pub struct Qwen35Fp8AttentionBackend {
    hidden_size: usize,
    rope_theta: f32,
    scratch: HashMap<(usize, usize), AttentionScratch>,
}

impl Qwen35Fp8AttentionBackend {
    pub fn new(config: &Qwen35MoeConfig) -> Result<Self> {
        config.validate()?;
        Ok(Self {
            hidden_size: TARGET_HIDDEN_SIZE,
            rope_theta: config.text_config.rope_parameters.rope_theta,
            scratch: HashMap::new(),
        })
    }

    pub fn retained_profiles(&self) -> usize {
        self.scratch.len()
    }

    fn scratch(&mut self, ctx: &CudaContext, tokens: usize) -> Result<&mut AttentionScratch> {
        let key = (ctx.device_id(), tokens);
        if !self.scratch.contains_key(&key) {
            let profile = AttentionScratch::new(ctx, tokens)?;
            self.scratch.insert(key, profile);
        }
        Ok(self
            .scratch
            .get_mut(&key)
            .expect("Qwen3.5 attention scratch profile was just inserted"))
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
        scales: KvCacheScales,
    ) -> Result<Self::Cache> {
        validate_full_attention_layer(layer_index)?;
        validate_cache_contract(geometry, capacity, scales)?;
        let shape = vec![
            capacity,
            ops::QWEN35_ATTENTION_KV_HEADS,
            ops::QWEN35_ATTENTION_HEAD_DIM,
        ];
        Ok(Qwen35AttentionCache {
            layer_index,
            geometry,
            capacity,
            scales,
            key: zeros(ctx, shape.clone(), DType::BF16)?,
            value: zeros(ctx, shape, DType::BF16)?,
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
        let (byte_offset, byte_len) = cache_byte_range(positions.clone())?;
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
                .as_tensor(shape.clone(), DType::BF16)
                .map_err(Error::Cuda)?;
            let destination_view = CudaBuffer::from_tensor(destination_tensor)
                .map_err(Error::Cuda)?
                .view(byte_offset, byte_len)
                .map_err(Error::Cuda)?
                .as_tensor(shape.clone(), DType::BF16)
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
        let scratch = self.scratch(ctx, tokens)?;
        scratch.ensure_text_positions(request.start_position)?;

        // ---- q/k/v projection GEMMs ----
        for (linear, output) in [
            (&request.weights.q_proj, &mut scratch.projected_query_gate),
            (&request.weights.k_proj, &mut scratch.projected_key),
            (&request.weights.v_proj, &mut scratch.projected_value),
        ] {
            ops::gemm(
                ctx,
                ops::GemmArgs::new(request.normalized_hidden, &linear.weight, output),
            )?;
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

        // ---- Write K/V to BF16 cache at cache_offset ----
        let kv_bytes = tokens * ops::QWEN35_ATTENTION_KV_WIDTH * DType::BF16.size_in_bytes();
        let cache_offset_bytes =
            request.start_position * ops::QWEN35_ATTENTION_KV_WIDTH * DType::BF16.size_in_bytes();
        for (src, dst_tensor) in [
            (&scratch.projected_key, &cache.key),
            (&scratch.projected_value, &cache.value),
        ] {
            let src_view = CudaBuffer::from_tensor(src)
                .map_err(Error::Cuda)?
                .as_tensor(
                    Shape::new(vec![tokens, ops::QWEN35_ATTENTION_KV_WIDTH]),
                    DType::BF16,
                )
                .map_err(Error::Cuda)?;
            let dst_view = CudaBuffer::from_tensor(dst_tensor)
                .map_err(Error::Cuda)?
                .view(cache_offset_bytes, kv_bytes)
                .map_err(Error::Cuda)?
                .as_tensor(
                    Shape::new(vec![tokens, ops::QWEN35_ATTENTION_KV_WIDTH]),
                    DType::BF16,
                )
                .map_err(Error::Cuda)?;
            transfers::copy_cuda_to_cuda(ctx, &src_view, &dst_view)?;
        }

        // ---- FA2 attention over the full prefix ----
        let valid = request.start_position + tokens;
        let key_prefix = cache_prefix_view(
            &cache.key,
            valid,
            ops::QWEN35_ATTENTION_KV_HEADS,
            ops::QWEN35_ATTENTION_HEAD_DIM,
        )?;
        let value_prefix = cache_prefix_view(
            &cache.value,
            valid,
            ops::QWEN35_ATTENTION_KV_HEADS,
            ops::QWEN35_ATTENTION_HEAD_DIM,
        )?;
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
        ops::gemm(
            ctx,
            ops::GemmArgs::new(
                &attention_output,
                &request.weights.o_proj.weight,
                &mut output,
            ),
        )
    }
}

#[derive(Debug)]
struct AttentionScratch {
    projected_query_gate: Tensor,
    projected_key: Tensor,
    projected_value: Tensor,
    positions: Tensor,
    query: Tensor,
    gate: Tensor,
    attention_output: Tensor,
    tokens: usize,
    positions_start: Option<usize>,
}

impl AttentionScratch {
    fn new(ctx: &CudaContext, tokens: usize) -> Result<Self> {
        if tokens == 0 || i32::try_from(tokens).is_err() {
            return Err(invalid("Qwen3.5 attention profile token count is invalid"));
        }
        let attention_shape = vec![
            tokens,
            ops::QWEN35_ATTENTION_Q_HEADS,
            ops::QWEN35_ATTENTION_HEAD_DIM,
        ];
        Ok(Self {
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
        return Err(invalid(
            "Qwen3.5 attention append exceeds BF16 cache capacity",
        ));
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
    let scales = request
        .weights
        .kv_cache_scales
        .ok_or_else(|| invalid("Qwen3.5 base full attention is missing FP8 K/V scales"))?;
    validate_scales(scales)?;
    if scales != cache.scales {
        return Err(invalid(
            "Qwen3.5 attention weights and cache metadata use different K/V scales",
        ));
    }
    Ok(tokens)
}

fn validate_attention_weights(
    ctx: &CudaContext,
    weights: &DeviceAttentionWeights,
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
        validate_linear(ctx, linear, input, output, label)?;
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

fn validate_linear(
    ctx: &CudaContext,
    linear: &DeviceDenseLinear,
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
        DType::BF16,
        &[input_features, output_features],
        label,
    )
}

fn validate_cache_contract(
    geometry: AttentionGeometry,
    capacity: usize,
    scales: KvCacheScales,
) -> Result<()> {
    validate_geometry(geometry)?;
    if capacity == 0 || i32::try_from(capacity).is_err() {
        return Err(invalid("Qwen3.5 BF16 cache capacity is invalid"));
    }
    validate_scales(scales)
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
        return Err(invalid(
            "unsupported Qwen3.5 full-attention geometry for BF16 K/V cache",
        ));
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
    validate_tensor(
        &cache.key,
        ctx,
        DType::BF16,
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
        DType::BF16,
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
    {
        return Err(invalid(
            "Qwen3.5 BF16 cache copy requires matching layer, geometry and metadata",
        ));
    }
    if positions.start > positions.end
        || positions.end > source.capacity
        || positions.end > destination.capacity
    {
        return Err(invalid("Qwen3.5 BF16 cache copy range is out of bounds"));
    }
    Ok(())
}

fn cache_byte_range(positions: Range<usize>) -> Result<(usize, usize)> {
    if positions.start > positions.end {
        return Err(invalid("Qwen3.5 BF16 cache copy range is reversed"));
    }
    let row_bytes = ops::QWEN35_ATTENTION_KV_HEADS
        .checked_mul(ops::QWEN35_ATTENTION_HEAD_DIM)
        .and_then(|elements| elements.checked_mul(DType::BF16.size_in_bytes()))
        .ok_or_else(|| invalid("Qwen3.5 BF16 cache row size overflow"))?;
    let offset = positions
        .start
        .checked_mul(row_bytes)
        .ok_or_else(|| invalid("Qwen3.5 BF16 cache copy offset overflow"))?;
    let len = (positions.end - positions.start)
        .checked_mul(row_bytes)
        .ok_or_else(|| invalid("Qwen3.5 BF16 cache copy length overflow"))?;
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
) -> Result<Tensor> {
    let bytes = valid * kv_heads * head_dim * DType::BF16.size_in_bytes();
    CudaBuffer::from_tensor(cache)
        .map_err(Error::Cuda)?
        .view(0, bytes)
        .map_err(Error::Cuda)?
        .as_tensor(Shape::new(vec![1, valid, kv_heads, head_dim]), DType::BF16)
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
            cache_byte_range(4..9).unwrap(),
            (4 * row_bytes, 5 * row_bytes)
        );
    }

    #[test]
    fn contract_rejects_non_target_geometry_and_checkpoint_amax_reuse() {
        let scales = KvCacheScales {
            key: 1.0 / 448.0,
            value: 224.0 / 448.0,
        };
        validate_cache_contract(target_geometry(), 4_096, scales).unwrap();
        let mut wrong = target_geometry();
        wrong.output_gate = false;
        assert!(validate_cache_contract(wrong, 4_096, scales).is_err());

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
