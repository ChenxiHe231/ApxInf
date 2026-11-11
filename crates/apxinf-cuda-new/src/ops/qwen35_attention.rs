//! Batch-one full-attention core for Qwen3.5-35B-A3B.
//!
//! This module contains the legacy static-scale FP8 reference path plus the
//! dynamic cache path used by the Qwen3.5 backend. Dynamic K/V scales cover one
//! token and one KV head; decode consumes FP8 directly, while prefill may expand
//! the valid prefix for the existing FA2 kernel. MTP attention remains separate.
//! On Ampere and newer, prefixes of at least 256 tokens use BF16 Tensor Cores
//! on exactly converted raw E4M3 tiles. FP32 probability*V-scale is decomposed
//! into three BF16 components; unsafe components/products retain FP32 SIMT PV.
//! This preserves the operands, but FP32 accumulation order can differ from
//! the SIMT route. Short prefixes and older devices use the SIMT route.

use apxinf_core::{DType, Error, Result, Tensor};

use crate::ffi::abi::{qwen35_attention as abi, status};
use crate::ops::gemm::contracts::tensor_storage;
use crate::{CudaBuffer, CudaContext};

pub const QWEN35_ATTENTION_Q_HEADS: usize = 16;
pub const QWEN35_ATTENTION_KV_HEADS: usize = 2;
pub const QWEN35_ATTENTION_HEAD_DIM: usize = 256;
pub const QWEN35_ATTENTION_ROTARY_DIM: usize = 64;
pub const QWEN35_ATTENTION_Q_PROJECTION_WIDTH: usize =
    2 * QWEN35_ATTENTION_Q_HEADS * QWEN35_ATTENTION_HEAD_DIM;
pub const QWEN35_ATTENTION_KV_WIDTH: usize = QWEN35_ATTENTION_KV_HEADS * QWEN35_ATTENTION_HEAD_DIM;
pub const QWEN35_ATTENTION_RMS_EPSILON: f32 = 1.0e-6;
pub const FP8_E4M3_MAX_FINITE: f32 = 448.0;
pub const QWEN35_KV_CACHE_BLOCK_TOKENS: usize = 128;
pub const QWEN35_ATTENTION_MAX_SPLITS: usize = 32;

fn invalid(message: impl Into<String>) -> Error {
    Error::Other(message.into())
}
/// TensorRT Edge-LLM `scale_quant_orig` values used to dequantize the cache.
///
/// The model loader converts ModelOpt checkpoint amax once with
/// `scale = checkpoint_amax / 448`; this operator consumes that scale directly.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Qwen35KvCacheScales {
    pub key: f32,
    pub value: f32,
}

impl Qwen35KvCacheScales {
    pub fn validate(self) -> Result<()> {
        if !self.key.is_finite() || self.key <= 0.0 || !self.value.is_finite() || self.value <= 0.0
        {
            return Err(invalid(
                "Qwen3.5 K/V dequant scales must be finite and positive",
            ));
        }
        Ok(())
    }
}

/// Persistent state for one of the ten base full-attention layers.
pub struct Qwen35Fp8KvCache<'a> {
    /// F8E4M3 `[blocks, 128, 2, 256]`, token-major within each page.
    pub key: &'a Tensor,
    /// F8E4M3 `[blocks, 128, 2, 256]`, token-major within each page.
    pub value: &'a Tensor,
}

/// Allocation-free per-call scratch and output.
pub struct Qwen35AttentionScratch<'a> {
    /// BF16 `[tokens, 16, 256]` normalized and mRoPE-rotated queries.
    pub query: &'a Tensor,
    /// BF16 `[tokens, 16, 256]` unnormalized sigmoid gate logits.
    pub gate: &'a Tensor,
    /// BF16 `[tokens, 16, 256]` gated attention result for `o_proj`.
    pub output: &'a Tensor,
    /// FP32 `[tokens, 16, cache_capacity]` parallel attention scores.
    pub scores: &'a Tensor,
    /// FP32 `[tokens, 2, 32, 8, 256]` split-KV value accumulators.
    pub partial_output: &'a Tensor,
    /// FP32 `[tokens, 2, 32, 8, 2]` split maximum/denominator pairs.
    pub partial_stats: &'a Tensor,
}

pub struct Qwen35Fp8AttentionArgs<'a> {
    /// BF16 `[tokens, 8192]`; each head stores `[query(256), gate(256)]`.
    pub projected_query_gate: &'a Tensor,
    /// BF16 `[tokens, 512]`.
    pub projected_key: &'a Tensor,
    /// BF16 `[tokens, 512]`.
    pub projected_value: &'a Tensor,
    /// BF16 `[256]`, zero-centered (`effective_weight = 1 + weight`).
    pub query_norm: &'a Tensor,
    /// BF16 `[256]`, zero-centered (`effective_weight = 1 + weight`).
    pub key_norm: &'a Tensor,
    /// I32 `[3, tokens]`, temporal/height/width positions.
    pub positions: &'a Tensor,
    pub cache: Qwen35Fp8KvCache<'a>,
    pub scratch: Qwen35AttentionScratch<'a>,
    /// Number of already committed cache rows.  Prefill starts at zero;
    /// single-token decode appends at the prior sequence length.
    pub cache_offset: usize,
    pub scales: Qwen35KvCacheScales,
    pub rope_theta: f32,
}

/// Append K/V and evaluate causal grouped-query attention.
///
/// All device/dtype/shape/range checks happen before the cache is mutated.
/// `tokens > 1` is prompt prefill or chunk verification; `tokens == 1` is
/// decode.  Within a chunk, token `i` can attend through
/// `cache_offset + i`, so prefill and decode have identical update semantics.
pub fn qwen35_attention_fp8(
    ctx: &CudaContext,
    tokens: usize,
    args: Qwen35Fp8AttentionArgs<'_>,
) -> Result<()> {
    if tokens == 0 || i32::try_from(tokens).is_err() {
        return Err(invalid("Qwen3.5 attention token count is invalid"));
    }
    if !args.rope_theta.is_finite() || args.rope_theta <= 0.0 {
        return Err(invalid("Qwen3.5 attention RoPE theta is invalid"));
    }
    args.scales.validate()?;
    let cache_shape = args.cache.key.shape().dims();
    if cache_shape.len() != 4
        || cache_shape[1] != QWEN35_KV_CACHE_BLOCK_TOKENS
        || cache_shape[2..] != [QWEN35_ATTENTION_KV_HEADS, QWEN35_ATTENTION_HEAD_DIM]
        || cache_shape[0] == 0
        || cache_shape != args.cache.value.shape().dims()
    {
        return Err(invalid(
            "Qwen3.5 FP8 K/V caches must match [blocks, 128, 2, 256]",
        ));
    }
    let capacity = cache_shape[0]
        .checked_mul(QWEN35_KV_CACHE_BLOCK_TOKENS)
        .ok_or_else(|| invalid("Qwen3.5 attention cache capacity overflow"))?;
    let end = args
        .cache_offset
        .checked_add(tokens)
        .ok_or_else(|| invalid("Qwen3.5 attention cache range overflow"))?;
    if end > capacity
        || i32::try_from(args.cache_offset).is_err()
        || i32::try_from(capacity).is_err()
    {
        return Err(invalid("Qwen3.5 attention cache append exceeds capacity"));
    }

    let projected_query_gate = tensor_storage(
        ctx,
        args.projected_query_gate,
        DType::BF16,
        &[tokens, QWEN35_ATTENTION_Q_PROJECTION_WIDTH],
    )?;
    let projected_key = tensor_storage(
        ctx,
        args.projected_key,
        DType::BF16,
        &[tokens, QWEN35_ATTENTION_KV_WIDTH],
    )?;
    let projected_value = tensor_storage(
        ctx,
        args.projected_value,
        DType::BF16,
        &[tokens, QWEN35_ATTENTION_KV_WIDTH],
    )?;
    let query_norm = tensor_storage(
        ctx,
        args.query_norm,
        DType::BF16,
        &[QWEN35_ATTENTION_HEAD_DIM],
    )?;
    let key_norm = tensor_storage(
        ctx,
        args.key_norm,
        DType::BF16,
        &[QWEN35_ATTENTION_HEAD_DIM],
    )?;
    let positions = tensor_storage(ctx, args.positions, DType::I32, &[3, tokens])?;
    let key_cache = tensor_storage(
        ctx,
        args.cache.key,
        DType::F8E4M3,
        &[
            cache_shape[0],
            QWEN35_KV_CACHE_BLOCK_TOKENS,
            QWEN35_ATTENTION_KV_HEADS,
            QWEN35_ATTENTION_HEAD_DIM,
        ],
    )?;
    let value_cache = tensor_storage(
        ctx,
        args.cache.value,
        DType::F8E4M3,
        &[
            cache_shape[0],
            QWEN35_KV_CACHE_BLOCK_TOKENS,
            QWEN35_ATTENTION_KV_HEADS,
            QWEN35_ATTENTION_HEAD_DIM,
        ],
    )?;
    let attention_shape = [tokens, QWEN35_ATTENTION_Q_HEADS, QWEN35_ATTENTION_HEAD_DIM];
    let query = tensor_storage(ctx, args.scratch.query, DType::BF16, &attention_shape)?;
    let gate = tensor_storage(ctx, args.scratch.gate, DType::BF16, &attention_shape)?;
    let output = tensor_storage(ctx, args.scratch.output, DType::BF16, &attention_shape)?;
    let scores = tensor_storage(
        ctx,
        args.scratch.scores,
        DType::F32,
        &[tokens, QWEN35_ATTENTION_Q_HEADS, capacity],
    )?;
    let partial_output = tensor_storage(
        ctx,
        args.scratch.partial_output,
        DType::F32,
        &[
            tokens,
            QWEN35_ATTENTION_KV_HEADS,
            QWEN35_ATTENTION_MAX_SPLITS,
            QWEN35_ATTENTION_Q_HEADS / QWEN35_ATTENTION_KV_HEADS,
            QWEN35_ATTENTION_HEAD_DIM,
        ],
    )?;
    let partial_stats = tensor_storage(
        ctx,
        args.scratch.partial_stats,
        DType::F32,
        &[
            tokens,
            QWEN35_ATTENTION_KV_HEADS,
            QWEN35_ATTENTION_MAX_SPLITS,
            QWEN35_ATTENTION_Q_HEADS / QWEN35_ATTENTION_KV_HEADS,
            2,
        ],
    )?;

    let read_only = [
        &projected_query_gate,
        &projected_key,
        &projected_value,
        &query_norm,
        &key_norm,
        &positions,
    ];
    let written = [
        &key_cache,
        &value_cache,
        &query,
        &gate,
        &output,
        &scores,
        &partial_output,
        &partial_stats,
    ];
    for (index, left) in written.iter().enumerate() {
        for right in written.iter().skip(index + 1).chain(read_only.iter()) {
            if storage_overlaps(left, right) {
                return Err(invalid(
                    "Qwen3.5 attention writable tensors must not overlap any binding",
                ));
            }
        }
    }

    let spec = abi::Spec {
        version: abi::SPEC_VERSION,
        tokens: tokens as u32,
        cache_offset: args.cache_offset as u32,
        cache_capacity: capacity as u32,
        rope_theta: args.rope_theta,
        rms_epsilon: QWEN35_ATTENTION_RMS_EPSILON,
        key_scale: args.scales.key,
        value_scale: args.scales.value,
    };
    let bindings = abi::Bindings {
        projected_query_gate: projected_query_gate.ptr(),
        projected_key: projected_key.ptr(),
        projected_value: projected_value.ptr(),
        query_norm: query_norm.ptr(),
        key_norm: key_norm.ptr(),
        positions: positions.ptr(),
        key_cache: key_cache.ptr(),
        value_cache: value_cache.ptr(),
        query: query.ptr(),
        gate: gate.ptr(),
        output: output.ptr(),
        scores: scores.ptr(),
        partial_output: partial_output.ptr(),
        partial_stats: partial_stats.ptr(),
        stream: ctx.stream().handle(),
    };
    crate::workspace::validate_capture_target(ctx.device_id(), bindings.stream as usize)?;
    unsafe {
        status::check(abi::apxinf_qwen35_attention_fp8(
            ctx.runtime(),
            &spec,
            &bindings,
        ))
    }
}

/// Dynamic FP8 payload and scale storage for one full-attention layer.
///
/// Each FP32 scale covers exactly one token, one KV head and all 256 head
/// dimensions. K and V use independent scales.
pub struct Qwen35DynamicFp8KvCache<'a> {
    pub key: &'a Tensor,
    pub value: &'a Tensor,
    pub key_scales: &'a Tensor,
    pub value_scales: &'a Tensor,
}

pub struct Qwen35DynamicDecodeScratch<'a> {
    pub partial_max: &'a Tensor,
    pub partial_sum: &'a Tensor,
    pub partial_output: &'a Tensor,
}

pub const QWEN35_DECODE_SPLIT_TOKENS: usize = abi::DECODE_SPLIT_TOKENS;

/// Workspace bound for the minimum split size, including shorter live prefixes.
pub fn qwen35_dynamic_decode_max_splits(capacity: usize) -> Result<usize> {
    if capacity == 0 || i32::try_from(capacity).is_err() {
        return Err(invalid("Qwen3.5 dynamic decode capacity is invalid"));
    }
    Ok(capacity.div_ceil(abi::MIN_DECODE_SPLIT_TOKENS))
}

fn dynamic_cache_storage(
    ctx: &CudaContext,
    cache: Qwen35DynamicFp8KvCache<'_>,
) -> Result<(usize, CudaBuffer, CudaBuffer, CudaBuffer, CudaBuffer)> {
    let shape = cache.key.shape().dims();
    let (capacity, scale_shape) = match shape {
        [capacity, QWEN35_ATTENTION_KV_HEADS, QWEN35_ATTENTION_HEAD_DIM] if *capacity > 0 => {
            (*capacity, vec![*capacity, QWEN35_ATTENTION_KV_HEADS])
        }
        [blocks, QWEN35_KV_CACHE_BLOCK_TOKENS, QWEN35_ATTENTION_KV_HEADS, QWEN35_ATTENTION_HEAD_DIM]
            if *blocks > 0 =>
        {
            let capacity = blocks
                .checked_mul(QWEN35_KV_CACHE_BLOCK_TOKENS)
                .ok_or_else(|| invalid("dynamic cache physical capacity overflow"))?;
            (
                capacity,
                vec![
                    *blocks,
                    QWEN35_KV_CACHE_BLOCK_TOKENS,
                    QWEN35_ATTENTION_KV_HEADS,
                ],
            )
        }
        _ => {
            return Err(invalid(
                "Qwen3.5 dynamic FP8 K/V caches must be [capacity,2,256] or [blocks,128,2,256]",
            ));
        }
    };
    if cache.value.shape().dims() != shape {
        return Err(invalid("Qwen3.5 dynamic FP8 K/V cache shapes differ"));
    }
    if i32::try_from(capacity).is_err() {
        return Err(invalid("dynamic cache capacity exceeds native range"));
    }
    let key = tensor_storage(ctx, cache.key, DType::F8E4M3, shape)?;
    let value = tensor_storage(ctx, cache.value, DType::F8E4M3, shape)?;
    let key_scales = tensor_storage(ctx, cache.key_scales, DType::F32, &scale_shape)?;
    let value_scales = tensor_storage(ctx, cache.value_scales, DType::F32, &scale_shape)?;
    validate_disjoint(&[], &[&key, &value, &key_scales, &value_scales])?;
    Ok((capacity, key, value, key_scales, value_scales))
}

/// Compute one FP32 absmax scale per token and KV head, quantize post-RoPE K
/// and projected V to E4M3, and append payloads plus scales to the cache.
pub fn qwen35_dynamic_kv_append(
    ctx: &CudaContext,
    tokens: usize,
    cache_offset: usize,
    key: &Tensor,
    value: &Tensor,
    cache: Qwen35DynamicFp8KvCache<'_>,
) -> Result<()> {
    if tokens == 0 || i32::try_from(tokens).is_err() {
        return Err(invalid("Qwen3.5 dynamic KV append token count is invalid"));
    }
    let (capacity, key_cache, value_cache, key_scales, value_scales) =
        dynamic_cache_storage(ctx, cache)?;
    let end = cache_offset
        .checked_add(tokens)
        .ok_or_else(|| invalid("Qwen3.5 dynamic KV append range overflow"))?;
    if end > capacity || i32::try_from(cache_offset).is_err() {
        return Err(invalid("Qwen3.5 dynamic KV append exceeds capacity"));
    }
    let input_shape = [tokens, QWEN35_ATTENTION_KV_HEADS, QWEN35_ATTENTION_HEAD_DIM];
    let key = tensor_storage(ctx, key, DType::BF16, &input_shape)?;
    let value = tensor_storage(ctx, value, DType::BF16, &input_shape)?;
    validate_disjoint(
        &[&key, &value],
        &[&key_cache, &value_cache, &key_scales, &value_scales],
    )?;
    let spec = abi::DynamicKvSpec {
        version: abi::DYNAMIC_KV_SPEC_VERSION,
        tokens: tokens as u32,
        cache_offset: cache_offset as u32,
        cache_capacity: capacity as u32,
    };
    let bindings = abi::DynamicKvBindings {
        key: key.ptr(),
        value: value.ptr(),
        key_cache: key_cache.ptr(),
        value_cache: value_cache.ptr(),
        key_scales: key_scales.ptr(),
        value_scales: value_scales.ptr(),
        stream: ctx.stream().handle(),
    };
    crate::workspace::validate_capture_target(ctx.device_id(), bindings.stream as usize)?;
    unsafe {
        status::check(abi::apxinf_qwen35_dynamic_kv_append(
            ctx.runtime(),
            &spec,
            &bindings,
        ))
    }
}

/// Split projected Q/gate, apply Q RMSNorm and RoPE in one kernel.
/// Intermediate normalization and rotation results retain BF16 rounding.
#[allow(clippy::too_many_arguments)]
pub fn qwen35_query_prepare(
    ctx: &CudaContext,
    tokens: usize,
    projected: &Tensor,
    norm_weight: &Tensor,
    positions: &Tensor,
    rope_theta: f32,
    query: &Tensor,
    gate: &Tensor,
) -> Result<()> {
    if tokens == 0 || i32::try_from(tokens).is_err() || !rope_theta.is_finite() || rope_theta <= 0.0
    {
        return Err(invalid("invalid fused query preparation parameters"));
    }
    let projected = tensor_storage(
        ctx,
        projected,
        DType::BF16,
        &[
            tokens,
            QWEN35_ATTENTION_Q_HEADS,
            2 * QWEN35_ATTENTION_HEAD_DIM,
        ],
    )?;
    let norm_weight = tensor_storage(ctx, norm_weight, DType::BF16, &[QWEN35_ATTENTION_HEAD_DIM])?;
    let positions = tensor_storage(ctx, positions, DType::I32, &[3, tokens])?;
    let shape = [tokens, QWEN35_ATTENTION_Q_HEADS, QWEN35_ATTENTION_HEAD_DIM];
    let query = tensor_storage(ctx, query, DType::BF16, &shape)?;
    let gate = tensor_storage(ctx, gate, DType::BF16, &shape)?;
    validate_disjoint(&[&projected, &norm_weight, &positions], &[&query, &gate])?;
    let spec = abi::QueryPrepareSpec {
        version: abi::QUERY_PREPARE_SPEC_VERSION,
        tokens: tokens as u32,
        rope_theta,
        rms_epsilon: QWEN35_ATTENTION_RMS_EPSILON,
    };
    let bindings = abi::QueryPrepareBindings {
        projected_query_gate: projected.ptr(),
        norm_weight: norm_weight.ptr(),
        positions: positions.ptr(),
        query: query.ptr(),
        gate: gate.ptr(),
        stream: ctx.stream().handle(),
    };
    crate::workspace::validate_capture_target(ctx.device_id(), bindings.stream as usize)?;
    unsafe {
        status::check(abi::apxinf_qwen35_query_prepare(
            ctx.runtime(),
            &spec,
            &bindings,
        ))
    }
}

/// Fused K-Norm -> RoPE -> online per-token-per-head FP8 cache write.
/// BF16 rounding after normalization and RoPE is preserved in registers.
#[allow(clippy::too_many_arguments)]
pub fn qwen35_dynamic_kv_prepare(
    ctx: &CudaContext,
    tokens: usize,
    cache_offset: usize,
    key: &Tensor,
    value: &Tensor,
    norm_weight: &Tensor,
    positions: &Tensor,
    rope_theta: f32,
    cache: Qwen35DynamicFp8KvCache<'_>,
) -> Result<()> {
    if tokens == 0 || i32::try_from(tokens).is_err() || !rope_theta.is_finite() || rope_theta <= 0.0
    {
        return Err(invalid("invalid fused KV preparation parameters"));
    }
    let (capacity, key_cache, value_cache, key_scales, value_scales) =
        dynamic_cache_storage(ctx, cache)?;
    if cache_offset
        .checked_add(tokens)
        .is_none_or(|end| end > capacity)
    {
        return Err(invalid("fused KV preparation exceeds cache capacity"));
    }
    let shape = [tokens, QWEN35_ATTENTION_KV_HEADS, QWEN35_ATTENTION_HEAD_DIM];
    let key = tensor_storage(ctx, key, DType::BF16, &shape)?;
    let value = tensor_storage(ctx, value, DType::BF16, &shape)?;
    let norm_weight = tensor_storage(ctx, norm_weight, DType::BF16, &[QWEN35_ATTENTION_HEAD_DIM])?;
    let positions = tensor_storage(ctx, positions, DType::I32, &[3, tokens])?;
    validate_disjoint(
        &[&key, &value, &norm_weight, &positions],
        &[&key_cache, &value_cache, &key_scales, &value_scales],
    )?;
    let spec = abi::DynamicPrepareSpec {
        kv: abi::DynamicKvSpec {
            version: abi::DYNAMIC_KV_SPEC_VERSION,
            tokens: tokens as u32,
            cache_offset: cache_offset as u32,
            cache_capacity: capacity as u32,
        },
        rope_theta,
        rms_epsilon: QWEN35_ATTENTION_RMS_EPSILON,
    };
    let bindings = abi::DynamicPrepareBindings {
        kv: abi::DynamicKvBindings {
            key: key.ptr(),
            value: value.ptr(),
            key_cache: key_cache.ptr(),
            value_cache: value_cache.ptr(),
            key_scales: key_scales.ptr(),
            value_scales: value_scales.ptr(),
            stream: ctx.stream().handle(),
        },
        norm_weight: norm_weight.ptr(),
        positions: positions.ptr(),
    };
    crate::workspace::validate_capture_target(ctx.device_id(), bindings.kv.stream as usize)?;
    unsafe {
        status::check(abi::apxinf_qwen35_dynamic_kv_prepare(
            ctx.runtime(),
            &spec,
            &bindings,
        ))
    }
}

/// Decode 1..=4 causal query rows ending at `valid_tokens` directly from dynamic E4M3 K/V. Cache values are
/// converted only in registers/shared memory; no BF16 cache is materialized.
pub fn qwen35_dynamic_decode_attention(
    ctx: &CudaContext,
    valid_tokens: usize,
    query: &Tensor,
    cache: Qwen35DynamicFp8KvCache<'_>,
    scratch: Qwen35DynamicDecodeScratch<'_>,
    output: &Tensor,
) -> Result<()> {
    dynamic_decode_attention(ctx, valid_tokens, query, None, cache, scratch, output)
}

/// Direct dynamic FP8 decode with sigmoid gate fused into split reduction.
/// The attention result rounds to BF16 before gating, as in separate ops.
#[allow(clippy::too_many_arguments)]
pub fn qwen35_dynamic_decode_attention_gated(
    ctx: &CudaContext,
    valid_tokens: usize,
    query: &Tensor,
    gate: &Tensor,
    cache: Qwen35DynamicFp8KvCache<'_>,
    scratch: Qwen35DynamicDecodeScratch<'_>,
    output: &Tensor,
) -> Result<()> {
    dynamic_decode_attention(ctx, valid_tokens, query, Some(gate), cache, scratch, output)
}

#[allow(clippy::too_many_arguments)]
fn dynamic_decode_attention(
    ctx: &CudaContext,
    valid_tokens: usize,
    query: &Tensor,
    gate: Option<&Tensor>,
    cache: Qwen35DynamicFp8KvCache<'_>,
    scratch: Qwen35DynamicDecodeScratch<'_>,
    output: &Tensor,
) -> Result<()> {
    let (capacity, key_cache, value_cache, key_scales, value_scales) =
        dynamic_cache_storage(ctx, cache)?;
    if valid_tokens == 0 || valid_tokens > capacity || i32::try_from(valid_tokens).is_err() {
        return Err(invalid("Qwen3.5 dynamic decode valid length is invalid"));
    }
    let rows = query.shape().dims().first().copied().unwrap_or(0);
    if rows == 0 || rows > abi::DYNAMIC_DECODE_MAX_ROWS || rows > valid_tokens {
        return Err(invalid("Qwen3.5 dynamic decode row count is invalid"));
    }
    let query_shape = [rows, QWEN35_ATTENTION_Q_HEADS, QWEN35_ATTENTION_HEAD_DIM];
    let query = tensor_storage(ctx, query, DType::BF16, &query_shape)?;
    let output = tensor_storage(ctx, output, DType::BF16, &query_shape)?;
    let gate = gate
        .map(|g| tensor_storage(ctx, g, DType::BF16, &query_shape))
        .transpose()?;

    let partial_shape = scratch.partial_max.shape().dims();
    if partial_shape.len() != 2
        || partial_shape[0] != rows * QWEN35_ATTENTION_Q_HEADS
        || partial_shape[1] == 0
        || scratch.partial_sum.shape().dims() != partial_shape
    {
        return Err(invalid(
            "Qwen3.5 dynamic decode partial statistics must match [rows*16,max_splits]",
        ));
    }
    let max_splits = partial_shape[1];
    if qwen35_dynamic_decode_max_splits(capacity)? > max_splits
        || i32::try_from(max_splits).is_err()
    {
        return Err(invalid(
            "Qwen3.5 dynamic decode workspace does not cover cache capacity",
        ));
    }
    let partial_max = tensor_storage(ctx, scratch.partial_max, DType::F32, partial_shape)?;
    let partial_sum = tensor_storage(ctx, scratch.partial_sum, DType::F32, partial_shape)?;
    let partial_output = tensor_storage(
        ctx,
        scratch.partial_output,
        DType::F32,
        &[
            rows * QWEN35_ATTENTION_Q_HEADS,
            max_splits,
            QWEN35_ATTENTION_HEAD_DIM,
        ],
    )?;
    validate_disjoint(
        &[&query, &key_cache, &value_cache, &key_scales, &value_scales],
        &[&partial_max, &partial_sum, &partial_output, &output],
    )?;
    if let Some(gate) = &gate {
        validate_disjoint(
            &[gate],
            &[&partial_max, &partial_sum, &partial_output, &output],
        )?;
    }
    let spec = abi::DynamicDecodeSpec {
        version: abi::DYNAMIC_DECODE_SPEC_VERSION,
        valid_tokens: valid_tokens as u32,
        cache_capacity: capacity as u32,
        max_splits: max_splits as u32,
        rows: rows as u32,
    };
    let bindings = abi::DynamicDecodeBindings {
        query: query.ptr(),
        key_cache: key_cache.ptr(),
        value_cache: value_cache.ptr(),
        key_scales: key_scales.ptr(),
        value_scales: value_scales.ptr(),
        partial_max: partial_max.ptr(),
        partial_sum: partial_sum.ptr(),
        partial_output: partial_output.ptr(),
        gate: gate
            .as_ref()
            .map_or(std::ptr::null(), |g| g.ptr().cast_const()),
        output: output.ptr(),
        stream: ctx.stream().handle(),
    };
    crate::workspace::validate_capture_target(ctx.device_id(), bindings.stream as usize)?;
    unsafe {
        status::check(abi::apxinf_qwen35_dynamic_decode_attention(
            ctx.runtime(),
            &spec,
            &bindings,
        ))
    }
}

/// Expand the valid dynamic FP8 prefix into BF16 buffers for prefill FA2.
pub fn qwen35_dynamic_kv_dequant(
    ctx: &CudaContext,
    valid_tokens: usize,
    cache: Qwen35DynamicFp8KvCache<'_>,
    key: &Tensor,
    value: &Tensor,
) -> Result<()> {
    let (capacity, key_cache, value_cache, key_scales, value_scales) =
        dynamic_cache_storage(ctx, cache)?;
    if valid_tokens == 0
        || valid_tokens > capacity
        || valid_tokens
            .checked_mul(QWEN35_ATTENTION_KV_WIDTH)
            .is_none_or(|n| i32::try_from(n).is_err())
    {
        return Err(invalid("Qwen3.5 dynamic dequant valid length is invalid"));
    }
    let output_shape = [
        1,
        valid_tokens,
        QWEN35_ATTENTION_KV_HEADS,
        QWEN35_ATTENTION_HEAD_DIM,
    ];
    let key = tensor_storage(ctx, key, DType::BF16, &output_shape)?;
    let value = tensor_storage(ctx, value, DType::BF16, &output_shape)?;
    validate_disjoint(
        &[&key_cache, &value_cache, &key_scales, &value_scales],
        &[&key, &value],
    )?;
    let spec = abi::DynamicDequantSpec {
        version: abi::DYNAMIC_KV_SPEC_VERSION,
        valid_tokens: valid_tokens as u32,
        cache_capacity: capacity as u32,
    };
    let bindings = abi::DynamicDequantBindings {
        key_cache: key_cache.ptr(),
        value_cache: value_cache.ptr(),
        key_scales: key_scales.ptr(),
        value_scales: value_scales.ptr(),
        key: key.ptr(),
        value: value.ptr(),
        stream: ctx.stream().handle(),
    };
    crate::workspace::validate_capture_target(ctx.device_id(), bindings.stream as usize)?;
    unsafe {
        status::check(abi::apxinf_qwen35_dynamic_kv_dequant(
            ctx.runtime(),
            &spec,
            &bindings,
        ))
    }
}

fn validate_disjoint(inputs: &[&CudaBuffer], outputs: &[&CudaBuffer]) -> Result<()> {
    for (i, output) in outputs.iter().enumerate() {
        if inputs.iter().any(|input| storage_overlaps(input, output))
            || outputs[..i]
                .iter()
                .any(|other| storage_overlaps(other, output))
        {
            return Err(invalid("dynamic attention writable buffers overlap"));
        }
    }
    Ok(())
}

fn storage_overlaps(left: &CudaBuffer, right: &CudaBuffer) -> bool {
    let left_start = left.ptr() as usize;
    let right_start = right.ptr() as usize;
    let left_end = left_start.saturating_add(left.len());
    let right_end = right_start.saturating_add(right.len());
    left_start < right_end && right_start < left_end
}

/// CPU oracle input.  Shapes are the flattened equivalents of the CUDA API.
pub struct Qwen35AttentionReferenceArgs<'a> {
    pub tokens: usize,
    pub cache_offset: usize,
    pub cache_capacity: usize,
    pub projected_query_gate: &'a [f32],
    pub projected_key: &'a [f32],
    pub projected_value: &'a [f32],
    pub query_norm: &'a [f32],
    pub key_norm: &'a [f32],
    pub positions: &'a [i32],
    pub key_cache: &'a mut [u8],
    pub value_cache: &'a mut [u8],
    pub scales: Qwen35KvCacheScales,
    pub rope_theta: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct Qwen35AttentionReferenceOutput {
    pub query: Vec<f32>,
    pub gate: Vec<f32>,
    pub output: Vec<f32>,
}

/// Exact scalar oracle for normalization, interleaved partial mRoPE, static
/// E4M3 cache update, grouped-query causal attention, and sigmoid output gate.
pub fn qwen35_attention_reference(
    args: Qwen35AttentionReferenceArgs<'_>,
) -> Result<Qwen35AttentionReferenceOutput> {
    validate_reference(&args)?;
    args.scales.validate()?;
    let (key_scale, value_scale) = (args.scales.key, args.scales.value);
    let attention_elements = args.tokens * QWEN35_ATTENTION_Q_HEADS * QWEN35_ATTENTION_HEAD_DIM;
    let mut query = vec![0.0f32; attention_elements];
    let mut gate = vec![0.0f32; attention_elements];

    for token in 0..args.tokens {
        for head in 0..QWEN35_ATTENTION_Q_HEADS {
            let source =
                (token * QWEN35_ATTENTION_Q_HEADS + head) * (2 * QWEN35_ATTENTION_HEAD_DIM);
            let destination = (token * QWEN35_ATTENTION_Q_HEADS + head) * QWEN35_ATTENTION_HEAD_DIM;
            normalize_zero_centered(
                &args.projected_query_gate[source..source + QWEN35_ATTENTION_HEAD_DIM],
                args.query_norm,
                &mut query[destination..destination + QWEN35_ATTENTION_HEAD_DIM],
            );
            gate[destination..destination + QWEN35_ATTENTION_HEAD_DIM].copy_from_slice(
                &args.projected_query_gate
                    [source + QWEN35_ATTENTION_HEAD_DIM..source + 2 * QWEN35_ATTENTION_HEAD_DIM],
            );
            apply_mrope(
                &mut query[destination..destination + QWEN35_ATTENTION_HEAD_DIM],
                args.positions,
                args.tokens,
                token,
                args.rope_theta,
            );
        }
        for head in 0..QWEN35_ATTENTION_KV_HEADS {
            let source = (token * QWEN35_ATTENTION_KV_HEADS + head) * QWEN35_ATTENTION_HEAD_DIM;
            let cache = ((args.cache_offset + token) * QWEN35_ATTENTION_KV_HEADS + head)
                * QWEN35_ATTENTION_HEAD_DIM;
            let mut normalized = vec![0.0f32; QWEN35_ATTENTION_HEAD_DIM];
            normalize_zero_centered(
                &args.projected_key[source..source + QWEN35_ATTENTION_HEAD_DIM],
                args.key_norm,
                &mut normalized,
            );
            apply_mrope(
                &mut normalized,
                args.positions,
                args.tokens,
                token,
                args.rope_theta,
            );
            for dim in 0..QWEN35_ATTENTION_HEAD_DIM {
                args.key_cache[cache + dim] = encode_e4m3(normalized[dim] / key_scale);
                args.value_cache[cache + dim] =
                    encode_e4m3(args.projected_value[source + dim] / value_scale);
            }
        }
    }

    let mut output = vec![0.0f32; attention_elements];
    for token in 0..args.tokens {
        for q_head in 0..QWEN35_ATTENTION_Q_HEADS {
            let q_base = (token * QWEN35_ATTENTION_Q_HEADS + q_head) * QWEN35_ATTENTION_HEAD_DIM;
            let kv_head = q_head / (QWEN35_ATTENTION_Q_HEADS / QWEN35_ATTENTION_KV_HEADS);
            let length = args.cache_offset + token + 1;
            let mut scores = Vec::with_capacity(length);
            for position in 0..length {
                let cache =
                    (position * QWEN35_ATTENTION_KV_HEADS + kv_head) * QWEN35_ATTENTION_HEAD_DIM;
                let dot = (0..QWEN35_ATTENTION_HEAD_DIM)
                    .map(|dim| {
                        query[q_base + dim] * decode_e4m3(args.key_cache[cache + dim]) * key_scale
                    })
                    .sum::<f32>();
                scores.push(dot / (QWEN35_ATTENTION_HEAD_DIM as f32).sqrt());
            }
            let maximum = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let denominator = scores
                .iter()
                .map(|score| (*score - maximum).exp())
                .sum::<f32>();
            for dim in 0..QWEN35_ATTENTION_HEAD_DIM {
                let attended = scores
                    .iter()
                    .enumerate()
                    .map(|(position, score)| {
                        let cache = (position * QWEN35_ATTENTION_KV_HEADS + kv_head)
                            * QWEN35_ATTENTION_HEAD_DIM;
                        ((*score - maximum).exp() / denominator)
                            * decode_e4m3(args.value_cache[cache + dim])
                            * value_scale
                    })
                    .sum::<f32>();
                output[q_base + dim] = round_bf16(attended * sigmoid(gate[q_base + dim]));
            }
        }
    }

    Ok(Qwen35AttentionReferenceOutput {
        query,
        gate,
        output,
    })
}

fn validate_reference(args: &Qwen35AttentionReferenceArgs<'_>) -> Result<()> {
    if args.tokens == 0
        || args.cache_capacity == 0
        || args
            .cache_offset
            .checked_add(args.tokens)
            .filter(|end| *end <= args.cache_capacity)
            .is_none()
        || !args.rope_theta.is_finite()
        || args.rope_theta <= 0.0
    {
        return Err(invalid("invalid Qwen3.5 attention reference geometry"));
    }
    args.scales.validate()?;
    let expected = [
        (
            args.projected_query_gate.len(),
            args.tokens * QWEN35_ATTENTION_Q_PROJECTION_WIDTH,
        ),
        (
            args.projected_key.len(),
            args.tokens * QWEN35_ATTENTION_KV_WIDTH,
        ),
        (
            args.projected_value.len(),
            args.tokens * QWEN35_ATTENTION_KV_WIDTH,
        ),
        (args.query_norm.len(), QWEN35_ATTENTION_HEAD_DIM),
        (args.key_norm.len(), QWEN35_ATTENTION_HEAD_DIM),
        (args.positions.len(), 3 * args.tokens),
        (
            args.key_cache.len(),
            args.cache_capacity * QWEN35_ATTENTION_KV_WIDTH,
        ),
        (
            args.value_cache.len(),
            args.cache_capacity * QWEN35_ATTENTION_KV_WIDTH,
        ),
    ];
    if expected.iter().any(|(actual, wanted)| actual != wanted) {
        return Err(invalid("Qwen3.5 attention reference shape mismatch"));
    }
    if args
        .projected_query_gate
        .iter()
        .chain(args.projected_key)
        .chain(args.projected_value)
        .chain(args.query_norm)
        .chain(args.key_norm)
        .any(|value| !value.is_finite())
    {
        return Err(invalid(
            "Qwen3.5 attention reference requires finite inputs",
        ));
    }
    Ok(())
}

fn normalize_zero_centered(input: &[f32], weight: &[f32], output: &mut [f32]) {
    let variance =
        input.iter().map(|value| value * value).sum::<f32>() / QWEN35_ATTENTION_HEAD_DIM as f32;
    let scale = (variance + QWEN35_ATTENTION_RMS_EPSILON).sqrt().recip();
    for ((output, input), weight) in output.iter_mut().zip(input).zip(weight) {
        *output = round_bf16(*input * scale * (1.0 + *weight));
    }
}

fn apply_mrope(data: &mut [f32], positions: &[i32], tokens: usize, token: usize, theta: f32) {
    let original = data[..QWEN35_ATTENTION_ROTARY_DIM].to_vec();
    let pairs = QWEN35_ATTENTION_ROTARY_DIM / 2;
    for pair in 0..pairs {
        let channel = pair % 3;
        let position = positions[channel * tokens + token] as f32;
        let frequency = theta.powf(-2.0 * pair as f32 / QWEN35_ATTENTION_ROTARY_DIM as f32);
        let (sine, cosine) = (position * frequency).sin_cos();
        let low = original[pair];
        let high = original[pairs + pair];
        data[pair] = round_bf16(low * cosine - high * sine);
        data[pairs + pair] = round_bf16(high * cosine + low * sine);
    }
}

fn sigmoid(value: f32) -> f32 {
    1.0 / (1.0 + (-value).exp())
}

fn round_bf16(value: f32) -> f32 {
    let bits = value.to_bits();
    let rounded = bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) & 0xffff_0000;
    f32::from_bits(rounded)
}

/// Decode the CUDA-compatible finite E4M3 byte representation.
pub fn decode_e4m3(code: u8) -> f32 {
    let sign = if code & 0x80 == 0 { 1.0 } else { -1.0 };
    let magnitude = code & 0x7f;
    let exponent = (magnitude >> 3) & 0x0f;
    let mantissa = magnitude & 0x07;
    if exponent == 0 {
        sign * mantissa as f32 * 2.0f32.powi(-9)
    } else if exponent == 0x0f && mantissa == 0x07 {
        f32::NAN
    } else {
        sign * (1.0 + mantissa as f32 / 8.0) * 2.0f32.powi(exponent as i32 - 7)
    }
}

/// Round to nearest finite E4M3, ties to an even code, with saturation.
pub fn encode_e4m3(value: f32) -> u8 {
    if value.is_nan() {
        return 0x7f;
    }
    let sign = if value.is_sign_negative() { 0x80 } else { 0 };
    let magnitude = value.abs().min(FP8_E4M3_MAX_FINITE);
    let mut best_code = 0u8;
    let mut best_error = f32::INFINITY;
    for code in 0u8..=0x7e {
        let error = (decode_e4m3(code) - magnitude).abs();
        if error < best_error || (error == best_error && code & 1 == 0 && best_code & 1 != 0) {
            best_error = error;
            best_code = code;
        }
    }
    sign | best_code
}

#[cfg(test)]
mod tests {
    use super::*;

    fn reference_fixture(tokens: usize, capacity: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        let query = (0..tokens * QWEN35_ATTENTION_Q_PROJECTION_WIDTH)
            .map(|index| ((index % 97) as f32 - 48.0) / 29.0)
            .collect();
        let key = (0..tokens * QWEN35_ATTENTION_KV_WIDTH)
            .map(|index| ((index % 53) as f32 - 26.0) / 17.0)
            .collect();
        let value = (0..tokens * QWEN35_ATTENTION_KV_WIDTH)
            .map(|index| ((index % 41) as f32 - 20.0) / 13.0)
            .collect();
        assert!(capacity >= tokens);
        (query, key, value)
    }

    #[test]
    fn e4m3_contract_has_cuda_range_and_ties_to_even() {
        assert_eq!(decode_e4m3(0x38), 1.0);
        assert_eq!(decode_e4m3(0x7e), 448.0);
        assert_eq!(encode_e4m3(1.0), 0x38);
        assert_eq!(encode_e4m3(1000.0), 0x7e);
        assert_eq!(encode_e4m3(-1000.0), 0xfe);
        // Midpoint between 1.0 (0x38, even) and 1.125 (0x39, odd).
        assert_eq!(encode_e4m3(1.0625), 0x38);
    }

    #[test]
    fn loader_converted_dequant_scale_is_used_directly() {
        let scales = Qwen35KvCacheScales {
            key: 0.5,
            value: 0.25,
        };
        assert_eq!((scales.key, scales.value), (0.5, 0.25));
        assert_eq!(decode_e4m3(encode_e4m3(2.0 / 0.5)) * 0.5, 2.0);
    }

    #[test]
    fn reference_prefill_and_decode_have_identical_cache_semantics() {
        let tokens = 2;
        let capacity = 3;
        let (query, key, value) = reference_fixture(tokens, capacity);
        let norm = vec![0.0; QWEN35_ATTENTION_HEAD_DIM];
        let positions = vec![0, 1, 0, 1, 0, 1];
        let scales = Qwen35KvCacheScales {
            key: 8.0 / FP8_E4M3_MAX_FINITE,
            value: 8.0 / FP8_E4M3_MAX_FINITE,
        };
        let mut prefill_k = vec![0; capacity * QWEN35_ATTENTION_KV_WIDTH];
        let mut prefill_v = prefill_k.clone();
        let prefill = qwen35_attention_reference(Qwen35AttentionReferenceArgs {
            tokens,
            cache_offset: 0,
            cache_capacity: capacity,
            projected_query_gate: &query,
            projected_key: &key,
            projected_value: &value,
            query_norm: &norm,
            key_norm: &norm,
            positions: &positions,
            key_cache: &mut prefill_k,
            value_cache: &mut prefill_v,
            scales,
            rope_theta: 10_000_000.0,
        })
        .unwrap();

        let mut decode_k = vec![0; capacity * QWEN35_ATTENTION_KV_WIDTH];
        let mut decode_v = decode_k.clone();
        for token in 0..tokens {
            let q_start = token * QWEN35_ATTENTION_Q_PROJECTION_WIDTH;
            let kv_start = token * QWEN35_ATTENTION_KV_WIDTH;
            let one_positions = [
                positions[token],
                positions[tokens + token],
                positions[2 * tokens + token],
            ];
            let decode = qwen35_attention_reference(Qwen35AttentionReferenceArgs {
                tokens: 1,
                cache_offset: token,
                cache_capacity: capacity,
                projected_query_gate: &query
                    [q_start..q_start + QWEN35_ATTENTION_Q_PROJECTION_WIDTH],
                projected_key: &key[kv_start..kv_start + QWEN35_ATTENTION_KV_WIDTH],
                projected_value: &value[kv_start..kv_start + QWEN35_ATTENTION_KV_WIDTH],
                query_norm: &norm,
                key_norm: &norm,
                positions: &one_positions,
                key_cache: &mut decode_k,
                value_cache: &mut decode_v,
                scales,
                rope_theta: 10_000_000.0,
            })
            .unwrap();
            let output_start = token * QWEN35_ATTENTION_Q_HEADS * QWEN35_ATTENTION_HEAD_DIM;
            for (actual, expected) in decode.output.iter().zip(
                &prefill.output[output_start
                    ..output_start + QWEN35_ATTENTION_Q_HEADS * QWEN35_ATTENTION_HEAD_DIM],
            ) {
                assert!((actual - expected).abs() < 1e-5, "{actual} != {expected}");
            }
        }
        assert_eq!(prefill_k, decode_k);
        assert_eq!(prefill_v, decode_v);
    }

    #[test]
    fn mrope_uses_three_interleaved_position_channels() {
        let mut data = vec![0.0; QWEN35_ATTENTION_HEAD_DIM];
        for pair in 0..QWEN35_ATTENTION_ROTARY_DIM / 2 {
            data[pair] = 1.0;
        }
        apply_mrope(&mut data, &[1, 2, 3], 1, 0, 10_000.0);
        // With unit low halves and zero high halves, each pair becomes
        // cos(position[channel] * inv_freq); adjacent pairs use T/H/W.
        assert_eq!(data[0], round_bf16(1.0f32.cos()));
        let freq1 = 10_000.0f32.powf(-2.0 / 64.0);
        assert_eq!(data[1], round_bf16((2.0 * freq1).cos()));
        let freq2 = 10_000.0f32.powf(-4.0 / 64.0);
        assert_eq!(data[2], round_bf16((3.0 * freq2).cos()));
        // mrope_section=[11,11,10] leaves the final two pairs assigned to
        // temporal and height respectively: 10 complete THW groups, then T,H.
        let freq30 = 10_000.0f32.powf(-60.0 / 64.0);
        let freq31 = 10_000.0f32.powf(-62.0 / 64.0);
        assert_eq!(data[30], round_bf16((1.0 * freq30).cos()));
        assert_eq!(data[31], round_bf16((2.0 * freq31).cos()));
        assert_eq!(data[QWEN35_ATTENTION_ROTARY_DIM], 0.0);
    }
}
