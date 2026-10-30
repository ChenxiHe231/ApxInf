//! Exact batch-one full-attention core for Qwen3.5-35B-A3B.
//!
//! This module starts at the three BF16 projection outputs and ends at the
//! gated BF16 input to `o_proj`.  It owns the base model's static FP8-E4M3
//! K/V cache contract.  MTP attention is deliberately separate because the
//! checkpoint's MTP subtree has BF16 K/V and no K/V calibration tensors.

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
pub const QWEN35_ATTENTION_KV_WIDTH: usize =
    QWEN35_ATTENTION_KV_HEADS * QWEN35_ATTENTION_HEAD_DIM;
pub const QWEN35_ATTENTION_RMS_EPSILON: f32 = 1.0e-6;
pub const FP8_E4M3_MAX_FINITE: f32 = 448.0;
pub const QWEN35_KV_CACHE_BLOCK_TOKENS: usize = 128;

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
        if !self.key.is_finite()
            || self.key <= 0.0
            || !self.value.is_finite()
            || self.value <= 0.0
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

    let read_only = [
        &projected_query_gate,
        &projected_key,
        &projected_value,
        &query_norm,
        &key_norm,
        &positions,
    ];
    let written = [&key_cache, &value_cache, &query, &gate, &output];
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
        stream: ctx.stream().handle(),
    };
    crate::workspace::validate_capture_target(ctx.device_id(), bindings.stream as usize)?;
    unsafe { status::check(abi::apxinf_qwen35_attention_fp8(ctx.runtime(), &spec, &bindings)) }
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
            let source = (token * QWEN35_ATTENTION_Q_HEADS + head)
                * (2 * QWEN35_ATTENTION_HEAD_DIM);
            let destination =
                (token * QWEN35_ATTENTION_Q_HEADS + head) * QWEN35_ATTENTION_HEAD_DIM;
            normalize_zero_centered(
                &args.projected_query_gate[source..source + QWEN35_ATTENTION_HEAD_DIM],
                args.query_norm,
                &mut query[destination..destination + QWEN35_ATTENTION_HEAD_DIM],
            );
            gate[destination..destination + QWEN35_ATTENTION_HEAD_DIM].copy_from_slice(
                &args.projected_query_gate[source + QWEN35_ATTENTION_HEAD_DIM
                    ..source + 2 * QWEN35_ATTENTION_HEAD_DIM],
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
            let source =
                (token * QWEN35_ATTENTION_KV_HEADS + head) * QWEN35_ATTENTION_HEAD_DIM;
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
            let q_base =
                (token * QWEN35_ATTENTION_Q_HEADS + q_head) * QWEN35_ATTENTION_HEAD_DIM;
            let kv_head = q_head / (QWEN35_ATTENTION_Q_HEADS / QWEN35_ATTENTION_KV_HEADS);
            let length = args.cache_offset + token + 1;
            let mut scores = Vec::with_capacity(length);
            for position in 0..length {
                let cache = (position * QWEN35_ATTENTION_KV_HEADS + kv_head)
                    * QWEN35_ATTENTION_HEAD_DIM;
                let dot = (0..QWEN35_ATTENTION_HEAD_DIM)
                    .map(|dim| {
                        query[q_base + dim]
                            * decode_e4m3(args.key_cache[cache + dim])
                            * key_scale
                    })
                    .sum::<f32>();
                scores.push(dot / (QWEN35_ATTENTION_HEAD_DIM as f32).sqrt());
            }
            let maximum = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let denominator = scores.iter().map(|score| (*score - maximum).exp()).sum::<f32>();
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

    Ok(Qwen35AttentionReferenceOutput { query, gate, output })
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
        return Err(invalid("Qwen3.5 attention reference requires finite inputs"));
    }
    Ok(())
}

fn normalize_zero_centered(input: &[f32], weight: &[f32], output: &mut [f32]) {
    let variance = input.iter().map(|value| value * value).sum::<f32>()
        / QWEN35_ATTENTION_HEAD_DIM as f32;
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
            let one_positions = [positions[token], positions[tokens + token], positions[2 * tokens + token]];
            let decode = qwen35_attention_reference(Qwen35AttentionReferenceArgs {
                tokens: 1,
                cache_offset: token,
                cache_capacity: capacity,
                projected_query_gate: &query[q_start..q_start + QWEN35_ATTENTION_Q_PROJECTION_WIDTH],
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
