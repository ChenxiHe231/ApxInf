//! Device-vs-scalar contract for Qwen3.5 full attention with a real E4M3 K/V
//! cache.  This intentionally exercises exact 16/2/256 geometry rather than a
//! reduced toy shape.

use apxinf_core::{DType, Shape, Tensor};
use apxinf_cuda_new::{ops, transfers, CudaBuffer, CudaContext};
use half::bf16;

fn zeros(ctx: &CudaContext, shape: Vec<usize>, dtype: DType) -> Tensor {
    CudaBuffer::alloc_zeros(shape.iter().product::<usize>() * dtype.size_in_bytes(), ctx.device_id())
        .unwrap()
        .as_tensor(Shape::new(shape), dtype)
        .unwrap()
}

fn fp8_cache_shape(capacity: usize) -> Vec<usize> {
    let blocks = capacity.div_ceil(ops::QWEN35_KV_CACHE_BLOCK_TOKENS);
    vec![
        blocks,
        ops::QWEN35_KV_CACHE_BLOCK_TOKENS,
        ops::QWEN35_ATTENTION_KV_HEADS,
        ops::QWEN35_ATTENTION_HEAD_DIM,
    ]
}
fn bf16_fixture(shape: Vec<usize>, period: usize, divisor: f32) -> (Tensor, Vec<f32>) {
    let values = (0..shape.iter().product::<usize>())
        .map(|index| bf16::from_f32(((index % period) as f32 - (period / 2) as f32) / divisor))
        .collect::<Vec<_>>();
    let reference = values.iter().map(|value| value.to_f32()).collect();
    (Tensor::from_bf16(shape, &values).unwrap(), reference)
}

fn patterned_bf16(length: usize, multiplier: usize, period: usize, divisor: f32) -> Vec<f32> {
    (0..length)
        .map(|index| {
            bf16::from_f32(
                ((index * multiplier % period) as f32 - (period / 2) as f32) / divisor,
            )
            .to_f32()
        })
        .collect()
}

fn positions(start: usize, tokens: usize) -> Vec<i32> {
    [1usize, 3, 5]
        .into_iter()
        .flat_map(|stride| {
            (start..start + tokens).map(move |position| {
                i32::try_from(position * stride).expect("test position must fit i32")
            })
        })
        .collect()
}

fn bf16_tensor(values: &[f32], shape: Vec<usize>) -> Tensor {
    let values = values.iter().map(|value| bf16::from_f32(*value)).collect::<Vec<_>>();
    Tensor::from_bf16(shape, &values).unwrap()
}

#[allow(clippy::too_many_arguments)]
fn run_attention(
    ctx: &CudaContext,
    tokens: usize,
    cache_offset: usize,
    projected_query_gate: &[f32],
    projected_key: &[f32],
    projected_value: &[f32],
    query_norm_values: &[f32],
    key_norm_values: &[f32],
    position_values: &[i32],
    key_cache: &Tensor,
    value_cache: &Tensor,
    scales: ops::Qwen35KvCacheScales,
) -> Tensor {
    let projected_query_gate = transfers::to_cuda(
        &bf16_tensor(
            projected_query_gate,
            vec![tokens, ops::QWEN35_ATTENTION_Q_PROJECTION_WIDTH],
        ),
        0,
    )
    .unwrap();
    let projected_key = transfers::to_cuda(
        &bf16_tensor(
            projected_key,
            vec![tokens, ops::QWEN35_ATTENTION_KV_WIDTH],
        ),
        0,
    )
    .unwrap();
    let projected_value = transfers::to_cuda(
        &bf16_tensor(
            projected_value,
            vec![tokens, ops::QWEN35_ATTENTION_KV_WIDTH],
        ),
        0,
    )
    .unwrap();
    let query_norm = transfers::to_cuda(
        &bf16_tensor(query_norm_values, vec![ops::QWEN35_ATTENTION_HEAD_DIM]),
        0,
    )
    .unwrap();
    let key_norm = transfers::to_cuda(
        &bf16_tensor(key_norm_values, vec![ops::QWEN35_ATTENTION_HEAD_DIM]),
        0,
    )
    .unwrap();
    let positions = transfers::to_cuda(
        &Tensor::from_i32(vec![3, tokens], position_values).unwrap(),
        0,
    )
    .unwrap();
    let attention_shape = vec![
        tokens,
        ops::QWEN35_ATTENTION_Q_HEADS,
        ops::QWEN35_ATTENTION_HEAD_DIM,
    ];
    let query = zeros(ctx, attention_shape.clone(), DType::BF16);
    let gate = zeros(ctx, attention_shape.clone(), DType::BF16);
    let output = zeros(ctx, attention_shape, DType::BF16);
    ops::qwen35_attention_fp8(
        ctx,
        tokens,
        ops::Qwen35Fp8AttentionArgs {
            projected_query_gate: &projected_query_gate,
            projected_key: &projected_key,
            projected_value: &projected_value,
            query_norm: &query_norm,
            key_norm: &key_norm,
            positions: &positions,
            cache: ops::Qwen35Fp8KvCache {
                key: key_cache,
                value: value_cache,
            },
            scratch: ops::Qwen35AttentionScratch {
                query: &query,
                gate: &gate,
                output: &output,
            },
            cache_offset,
            scales,
            rope_theta: 10_000_000.0,
        },
    )
    .unwrap();
    // All input and scratch tensors except output are local to this helper.
    // Synchronize before they are dropped so this remains safe independently
    // of allocator stream-ordering behavior.
    ctx.synchronize().unwrap();
    output
}

fn fp8_bytes(tensor: &Tensor) -> Vec<u8> {
    transfers::to_cpu(tensor)
        .unwrap()
        .as_f8_e4m3()
        .unwrap()
        .to_vec()
}

fn bf16_bits(tensor: &Tensor) -> Vec<u16> {
    transfers::to_cpu(tensor)
        .unwrap()
        .as_bf16()
        .unwrap()
        .iter()
        .map(|value| value.to_bits())
        .collect()
}

#[test]
fn qwen35_prefill_writes_fp8_cache_and_matches_scalar_attention() {
    let ctx = CudaContext::new(0).unwrap();
    let tokens = 2usize;
    let capacity = 4usize;
    let (projected_query_gate_cpu, projected_query_gate_reference) = bf16_fixture(
        vec![tokens, ops::QWEN35_ATTENTION_Q_PROJECTION_WIDTH],
        97,
        29.0,
    );
    let (projected_key_cpu, projected_key_reference) =
        bf16_fixture(vec![tokens, ops::QWEN35_ATTENTION_KV_WIDTH], 53, 17.0);
    let (projected_value_cpu, projected_value_reference) =
        bf16_fixture(vec![tokens, ops::QWEN35_ATTENTION_KV_WIDTH], 41, 13.0);
    let norm_values = (0..ops::QWEN35_ATTENTION_HEAD_DIM)
        .map(|index| bf16::from_f32(((index % 11) as f32 - 5.0) / 64.0))
        .collect::<Vec<_>>();
    let norm_reference = norm_values.iter().map(|value| value.to_f32()).collect::<Vec<_>>();
    let query_norm_cpu = Tensor::from_bf16(vec![ops::QWEN35_ATTENTION_HEAD_DIM], &norm_values).unwrap();
    let key_norm_cpu = Tensor::from_bf16(vec![ops::QWEN35_ATTENTION_HEAD_DIM], &norm_values).unwrap();
    let positions_values = [0i32, 1, 0, 3, 0, 5];
    let positions_cpu = Tensor::from_i32(vec![3, tokens], &positions_values).unwrap();

    let projected_query_gate = transfers::to_cuda(&projected_query_gate_cpu, 0).unwrap();
    let projected_key = transfers::to_cuda(&projected_key_cpu, 0).unwrap();
    let projected_value = transfers::to_cuda(&projected_value_cpu, 0).unwrap();
    let query_norm = transfers::to_cuda(&query_norm_cpu, 0).unwrap();
    let key_norm = transfers::to_cuda(&key_norm_cpu, 0).unwrap();
    let positions = transfers::to_cuda(&positions_cpu, 0).unwrap();
    let key_cache = zeros(
        &ctx,
        fp8_cache_shape(capacity),
        DType::F8E4M3,
    );
    let value_cache = zeros(
        &ctx,
        fp8_cache_shape(capacity),
        DType::F8E4M3,
    );
    let attention_shape = vec![
        tokens,
        ops::QWEN35_ATTENTION_Q_HEADS,
        ops::QWEN35_ATTENTION_HEAD_DIM,
    ];
    let query = zeros(&ctx, attention_shape.clone(), DType::BF16);
    let gate = zeros(&ctx, attention_shape.clone(), DType::BF16);
    let output = zeros(&ctx, attention_shape, DType::BF16);
    let scales = ops::Qwen35KvCacheScales {
        key: 8.0 / ops::FP8_E4M3_MAX_FINITE,
        value: 8.0 / ops::FP8_E4M3_MAX_FINITE,
    };

    ops::qwen35_attention_fp8(
        &ctx,
        tokens,
        ops::Qwen35Fp8AttentionArgs {
            projected_query_gate: &projected_query_gate,
            projected_key: &projected_key,
            projected_value: &projected_value,
            query_norm: &query_norm,
            key_norm: &key_norm,
            positions: &positions,
            cache: ops::Qwen35Fp8KvCache {
                key: &key_cache,
                value: &value_cache,
            },
            scratch: ops::Qwen35AttentionScratch {
                query: &query,
                gate: &gate,
                output: &output,
            },
            cache_offset: 0,
            scales,
            rope_theta: 10_000_000.0,
        },
    )
    .unwrap();
    ctx.synchronize().unwrap();

    let mut reference_key_cache = vec![0u8; capacity * ops::QWEN35_ATTENTION_KV_WIDTH];
    let mut reference_value_cache = reference_key_cache.clone();
    let reference = ops::qwen35_attention_reference(ops::Qwen35AttentionReferenceArgs {
        tokens,
        cache_offset: 0,
        cache_capacity: capacity,
        projected_query_gate: &projected_query_gate_reference,
        projected_key: &projected_key_reference,
        projected_value: &projected_value_reference,
        query_norm: &norm_reference,
        key_norm: &norm_reference,
        positions: &positions_values,
        key_cache: &mut reference_key_cache,
        value_cache: &mut reference_value_cache,
        scales,
        rope_theta: 10_000_000.0,
    })
    .unwrap();

    let logical_elements = capacity * ops::QWEN35_ATTENTION_KV_WIDTH;
    assert_eq!(
        &transfers::to_cpu(&key_cache)
            .unwrap()
            .as_f8_e4m3()
            .unwrap()[..logical_elements],
        reference_key_cache
    );
    assert_eq!(
        &transfers::to_cpu(&value_cache)
            .unwrap()
            .as_f8_e4m3()
            .unwrap()[..logical_elements],
        reference_value_cache
    );
    let actual = transfers::to_cpu(&output)
        .unwrap()
        .as_bf16()
        .unwrap()
        .iter()
        .map(|value| value.to_f32())
        .collect::<Vec<_>>();
    for (actual, expected) in actual.iter().zip(reference.output) {
        assert!((actual - expected).abs() <= 0.02, "{actual} != {expected}");
    }

    // A BF16 buffer with the same logical shape is not a fallback cache.  The
    // public contract must reject it rather than silently doubling memory and
    // bypassing the checkpoint's FP8 calibration.
    let fake_bf16_key_cache = zeros(
        &ctx,
        fp8_cache_shape(capacity),
        DType::BF16,
    );
    assert!(ops::qwen35_attention_fp8(
        &ctx,
        tokens,
        ops::Qwen35Fp8AttentionArgs {
            projected_query_gate: &projected_query_gate,
            projected_key: &projected_key,
            projected_value: &projected_value,
            query_norm: &query_norm,
            key_norm: &key_norm,
            positions: &positions,
            cache: ops::Qwen35Fp8KvCache {
                key: &fake_bf16_key_cache,
                value: &value_cache,
            },
            scratch: ops::Qwen35AttentionScratch {
                query: &query,
                gate: &gate,
                output: &output,
            },
            cache_offset: 0,
            scales,
            rope_theta: 10_000_000.0,
        },
    )
    .is_err());
}

/// Decode must remain on the same numerical path immediately below, at, and
/// above the 128-row provider boundary used by optimized attention backends.
/// The current correctness backend is direct attention at every length; this
/// gate prevents a future split-KV/FA provider switch from changing cache bytes
/// or the BF16 output relative to one-shot causal execution.
#[test]
fn qwen35_decode_is_bit_exact_across_critical_kv_lengths() {
    let ctx = CudaContext::new(0).unwrap();
    let query_norm = patterned_bf16(ops::QWEN35_ATTENTION_HEAD_DIM, 7, 23, 96.0);
    let key_norm = patterned_bf16(ops::QWEN35_ATTENTION_HEAD_DIM, 11, 29, 112.0);
    let scales = ops::Qwen35KvCacheScales {
        key: 8.0 / ops::FP8_E4M3_MAX_FINITE,
        value: 8.0 / ops::FP8_E4M3_MAX_FINITE,
    };

    for prefix_tokens in [1023usize] {
        let total_tokens = prefix_tokens + 1;
        let capacity = total_tokens + 1;
        let projected_query_gate = patterned_bf16(
            total_tokens * ops::QWEN35_ATTENTION_Q_PROJECTION_WIDTH,
            17,
            101,
            37.0,
        );
        let projected_key = patterned_bf16(
            total_tokens * ops::QWEN35_ATTENTION_KV_WIDTH,
            13,
            83,
            31.0,
        );
        let projected_value = patterned_bf16(
            total_tokens * ops::QWEN35_ATTENTION_KV_WIDTH,
            19,
            79,
            29.0,
        );

        let monolithic_key_cache = zeros(
            &ctx,
            fp8_cache_shape(capacity),
            DType::F8E4M3,
        );
        let monolithic_value_cache = zeros(
            &ctx,
            fp8_cache_shape(capacity),
            DType::F8E4M3,
        );
        let monolithic_output = run_attention(
            &ctx,
            total_tokens,
            0,
            &projected_query_gate,
            &projected_key,
            &projected_value,
            &query_norm,
            &key_norm,
            &positions(0, total_tokens),
            &monolithic_key_cache,
            &monolithic_value_cache,
            scales,
        );

        let split_key_cache = zeros(
            &ctx,
            fp8_cache_shape(capacity),
            DType::F8E4M3,
        );
        let split_value_cache = zeros(
            &ctx,
            fp8_cache_shape(capacity),
            DType::F8E4M3,
        );
        let q_prefix = prefix_tokens * ops::QWEN35_ATTENTION_Q_PROJECTION_WIDTH;
        let kv_prefix = prefix_tokens * ops::QWEN35_ATTENTION_KV_WIDTH;
        let _prefix_output = run_attention(
            &ctx,
            prefix_tokens,
            0,
            &projected_query_gate[..q_prefix],
            &projected_key[..kv_prefix],
            &projected_value[..kv_prefix],
            &query_norm,
            &key_norm,
            &positions(0, prefix_tokens),
            &split_key_cache,
            &split_value_cache,
            scales,
        );
        let decode_output = run_attention(
            &ctx,
            1,
            prefix_tokens,
            &projected_query_gate[q_prefix..],
            &projected_key[kv_prefix..],
            &projected_value[kv_prefix..],
            &query_norm,
            &key_norm,
            &positions(prefix_tokens, 1),
            &split_key_cache,
            &split_value_cache,
            scales,
        );

        let monolithic_key = fp8_bytes(&monolithic_key_cache);
        let monolithic_value = fp8_bytes(&monolithic_value_cache);
        let split_key = fp8_bytes(&split_key_cache);
        let split_value = fp8_bytes(&split_value_cache);
        assert_eq!(
            split_key, monolithic_key,
            "FP8 key cache changed across decode boundary at prefix {prefix_tokens}"
        );
        assert_eq!(
            split_value, monolithic_value,
            "FP8 value cache changed across decode boundary at prefix {prefix_tokens}"
        );

        let output_row = ops::QWEN35_ATTENTION_Q_HEADS * ops::QWEN35_ATTENTION_HEAD_DIM;
        let monolithic_bits = bf16_bits(&monolithic_output);
        let decode_bits = bf16_bits(&decode_output);
        assert_eq!(
            decode_bits,
            monolithic_bits[prefix_tokens * output_row..],
            "BF16 decode output changed across provider boundary at prefix {prefix_tokens}"
        );

        // Also retain the independent scalar check for the appended row. The
        // already verified device prefix is used as the oracle's initial FP8
        // cache, avoiding an O(T^2) CPU prefill inside this boundary test.
        let mut reference_key = split_key.clone();
        let mut reference_value = split_value.clone();
        let reference = ops::qwen35_attention_reference(ops::Qwen35AttentionReferenceArgs {
            tokens: 1,
            cache_offset: prefix_tokens,
            cache_capacity: capacity,
            projected_query_gate: &projected_query_gate[q_prefix..],
            projected_key: &projected_key[kv_prefix..],
            projected_value: &projected_value[kv_prefix..],
            query_norm: &query_norm,
            key_norm: &key_norm,
            positions: &positions(prefix_tokens, 1),
            key_cache: &mut reference_key,
            value_cache: &mut reference_value,
            scales,
            rope_theta: 10_000_000.0,
        })
        .unwrap();
        assert_eq!(reference_key, split_key);
        assert_eq!(reference_value, split_value);
        let actual = transfers::to_cpu(&decode_output)
            .unwrap()
            .as_bf16()
            .unwrap()
            .iter()
            .map(|value| value.to_f32())
            .collect::<Vec<_>>();
        for (actual, expected) in actual.iter().zip(reference.output) {
            assert!(
                (actual - expected).abs() <= 0.02,
                "prefix {prefix_tokens}: {actual} != {expected}"
            );
        }
    }
}
