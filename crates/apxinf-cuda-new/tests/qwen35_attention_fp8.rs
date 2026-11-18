//! Device-vs-scalar contract for Qwen3.5 full attention with a real E4M3 K/V
//! cache.  This intentionally exercises exact 16/2/256 geometry rather than a
//! reduced toy shape.

use apxinf_core::{DType, Shape, Tensor};
use apxinf_cuda_new::{ops, transfers, CudaBuffer, CudaContext};
use half::bf16;

fn zeros(ctx: &CudaContext, shape: Vec<usize>, dtype: DType) -> Tensor {
    CudaBuffer::alloc_zeros(
        shape.iter().product::<usize>() * dtype.size_in_bytes(),
        ctx.device_id(),
    )
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
            bf16::from_f32(((index * multiplier % period) as f32 - (period / 2) as f32) / divisor)
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
    let values = values
        .iter()
        .map(|value| bf16::from_f32(*value))
        .collect::<Vec<_>>();
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
        &bf16_tensor(projected_key, vec![tokens, ops::QWEN35_ATTENTION_KV_WIDTH]),
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
    let capacity = key_cache.shape().dims()[0] * ops::QWEN35_KV_CACHE_BLOCK_TOKENS;
    let scores = zeros(
        ctx,
        vec![tokens, ops::QWEN35_ATTENTION_Q_HEADS, capacity],
        DType::F32,
    );
    let partial_output = zeros(
        ctx,
        vec![
            tokens,
            ops::QWEN35_ATTENTION_KV_HEADS,
            ops::QWEN35_ATTENTION_MAX_SPLITS,
            ops::QWEN35_ATTENTION_Q_HEADS / ops::QWEN35_ATTENTION_KV_HEADS,
            ops::QWEN35_ATTENTION_HEAD_DIM,
        ],
        DType::F32,
    );
    let partial_stats = zeros(
        ctx,
        vec![
            tokens,
            ops::QWEN35_ATTENTION_KV_HEADS,
            ops::QWEN35_ATTENTION_MAX_SPLITS,
            ops::QWEN35_ATTENTION_Q_HEADS / ops::QWEN35_ATTENTION_KV_HEADS,
            2,
        ],
        DType::F32,
    );
    ops::qwen35_attention_static_cache(
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
                scores: &scores,
                partial_output: &partial_output,
                partial_stats: &partial_stats,
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
    let norm_reference = norm_values
        .iter()
        .map(|value| value.to_f32())
        .collect::<Vec<_>>();
    let query_norm_cpu =
        Tensor::from_bf16(vec![ops::QWEN35_ATTENTION_HEAD_DIM], &norm_values).unwrap();
    let key_norm_cpu =
        Tensor::from_bf16(vec![ops::QWEN35_ATTENTION_HEAD_DIM], &norm_values).unwrap();
    let positions_values = [0i32, 1, 0, 3, 0, 5];
    let positions_cpu = Tensor::from_i32(vec![3, tokens], &positions_values).unwrap();

    let projected_query_gate = transfers::to_cuda(&projected_query_gate_cpu, 0).unwrap();
    let projected_key = transfers::to_cuda(&projected_key_cpu, 0).unwrap();
    let projected_value = transfers::to_cuda(&projected_value_cpu, 0).unwrap();
    let query_norm = transfers::to_cuda(&query_norm_cpu, 0).unwrap();
    let key_norm = transfers::to_cuda(&key_norm_cpu, 0).unwrap();
    let positions = transfers::to_cuda(&positions_cpu, 0).unwrap();
    let key_cache = zeros(&ctx, fp8_cache_shape(capacity), DType::F8E4M3);
    let value_cache = zeros(&ctx, fp8_cache_shape(capacity), DType::F8E4M3);
    let attention_shape = vec![
        tokens,
        ops::QWEN35_ATTENTION_Q_HEADS,
        ops::QWEN35_ATTENTION_HEAD_DIM,
    ];
    let query = zeros(&ctx, attention_shape.clone(), DType::BF16);
    let gate = zeros(&ctx, attention_shape.clone(), DType::BF16);
    let output = zeros(&ctx, attention_shape, DType::BF16);
    let physical_capacity = fp8_cache_shape(capacity)[0] * ops::QWEN35_KV_CACHE_BLOCK_TOKENS;
    let scores = zeros(
        &ctx,
        vec![tokens, ops::QWEN35_ATTENTION_Q_HEADS, physical_capacity],
        DType::F32,
    );
    let partial_output = zeros(
        &ctx,
        vec![
            tokens,
            ops::QWEN35_ATTENTION_KV_HEADS,
            ops::QWEN35_ATTENTION_MAX_SPLITS,
            ops::QWEN35_ATTENTION_Q_HEADS / ops::QWEN35_ATTENTION_KV_HEADS,
            ops::QWEN35_ATTENTION_HEAD_DIM,
        ],
        DType::F32,
    );
    let partial_stats = zeros(
        &ctx,
        vec![
            tokens,
            ops::QWEN35_ATTENTION_KV_HEADS,
            ops::QWEN35_ATTENTION_MAX_SPLITS,
            ops::QWEN35_ATTENTION_Q_HEADS / ops::QWEN35_ATTENTION_KV_HEADS,
            2,
        ],
        DType::F32,
    );
    let scales = ops::Qwen35KvCacheScales {
        key: 8.0 / ops::FP8_E4M3_MAX_FINITE,
        value: 8.0 / ops::FP8_E4M3_MAX_FINITE,
    };

    ops::qwen35_attention_static_cache(
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
                scores: &scores,
                partial_output: &partial_output,
                partial_stats: &partial_stats,
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
        &transfers::to_cpu(&key_cache).unwrap().as_f8_e4m3().unwrap()[..logical_elements],
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
    let fake_bf16_key_cache = zeros(&ctx, fp8_cache_shape(capacity), DType::BF16);
    assert!(ops::qwen35_attention_static_cache(
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
                scores: &scores,
                partial_output: &partial_output,
                partial_stats: &partial_stats,
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
        let physical_capacity = fp8_cache_shape(capacity)[0] * ops::QWEN35_KV_CACHE_BLOCK_TOKENS;
        let projected_query_gate = patterned_bf16(
            total_tokens * ops::QWEN35_ATTENTION_Q_PROJECTION_WIDTH,
            17,
            101,
            37.0,
        );
        let projected_key =
            patterned_bf16(total_tokens * ops::QWEN35_ATTENTION_KV_WIDTH, 13, 83, 31.0);
        let projected_value =
            patterned_bf16(total_tokens * ops::QWEN35_ATTENTION_KV_WIDTH, 19, 79, 29.0);

        let monolithic_key_cache = zeros(&ctx, fp8_cache_shape(capacity), DType::F8E4M3);
        let monolithic_value_cache = zeros(&ctx, fp8_cache_shape(capacity), DType::F8E4M3);
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

        let split_key_cache = zeros(&ctx, fp8_cache_shape(capacity), DType::F8E4M3);
        let split_value_cache = zeros(&ctx, fp8_cache_shape(capacity), DType::F8E4M3);
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
            cache_capacity: physical_capacity,
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
// Append to crates/apxinf-cuda-new/tests/qwen35_attention_static_cache.rs.
// Uses that test module's bf16_fixture, positions, fp8_bytes and bf16_bits helpers.
#[test]
#[ignore = "Thor GPU correctness gate; run under the shared GPU lock"]
fn qwen35_draft_attention_append_and_graph_contract() {
    let ctx = CudaContext::new_with_autotune(0, false).unwrap();
    assert!(matches!(ctx.caps().sm, 101 | 110), "requires Jetson Thor");

    let q_heads = ops::QWEN35_ATTENTION_Q_HEADS;
    let kv_heads = ops::QWEN35_ATTENTION_KV_HEADS;
    let head_dim = ops::QWEN35_ATTENTION_HEAD_DIM;
    let kv_width = ops::QWEN35_ATTENTION_KV_WIDTH;
    let page = ops::QWEN35_KV_CACHE_BLOCK_TOKENS;
    let unit = ops::Qwen35KvCacheScales {
        key: 1.0,
        value: 1.0,
    };

    let upload = |host| transfers::to_cuda_on_stream(&ctx, host).unwrap();
    let zero = |shape: Vec<usize>, dtype: DType| {
        let bytes = shape.iter().product::<usize>() * dtype.size_in_bytes();
        CudaBuffer::alloc_zeros_async(bytes, ctx.device_id(), ctx.stream())
            .unwrap()
            .as_tensor(Shape::new(shape), dtype)
            .unwrap()
    };
    let projected_query_gate =
        upload(bf16_fixture(vec![1, ops::QWEN35_ATTENTION_Q_PROJECTION_WIDTH], 97, 29.0).0);
    let projected_key = upload(bf16_fixture(vec![1, kv_width], 53, 17.0).0);
    let projected_value = upload(bf16_fixture(vec![1, kv_width], 41, 31.0).0);
    let norm = upload(bf16_fixture(vec![head_dim], 11, 64.0).0);

    // Cover the page boundary, the 128/256 split transition, a partial 48K split,
    // and a split reduction spanning more than 256 partials.
    for offset in [127, 128, 8191, 8192, 49_152, 65_536] {
        let shape = fp8_cache_shape(offset + 2);
        let capacity = shape[0] * page;
        let row = offset * kv_width;
        let codes = [0x01, 0x81, 0x08, 0x88, 0x38, 0xb8, 0x40, 0xc0];
        let seed = |shift: usize| {
            let mut bytes = vec![0x7e; capacity * kv_width];
            for (index, byte) in bytes[..row].iter_mut().enumerate() {
                let kv_head = index / head_dim % kv_heads;
                *byte = codes[(index * 5 + index / kv_width + 3 * kv_head + shift) % codes.len()];
            }
            bytes
        };
        let key_seed = seed(0);
        let value_seed = seed(3);
        let key_cache: [_; 2] = std::array::from_fn(|_| {
            upload(Tensor::from_f8_e4m3(shape.clone(), &key_seed).unwrap())
        });
        let value_cache: [_; 2] = std::array::from_fn(|_| {
            upload(Tensor::from_f8_e4m3(shape.clone(), &value_seed).unwrap())
        });

        let positions = upload(Tensor::from_i32(vec![3, 1], &positions(offset, 1)).unwrap());
        let query = zero(vec![1, q_heads, head_dim], DType::BF16);
        let gate = zero(vec![1, q_heads, head_dim], DType::BF16);
        let output = zero(vec![1, q_heads, head_dim], DType::BF16);
        let scores = zero(vec![1, q_heads, capacity], DType::F32);
        let splits = ops::QWEN35_ATTENTION_MAX_SPLITS;
        let partial_output = zero(
            vec![1, kv_heads, splits, q_heads / kv_heads, head_dim],
            DType::F32,
        );
        let partial_stats = zero(vec![1, kv_heads, splits, q_heads / kv_heads, 2], DType::F32);
        let workspace_elements = ops::qwen35_draft_attention_workspace_elements(capacity).unwrap();
        let workspace = zero(vec![workspace_elements], DType::F32);

        let args = |arm: usize, cache_offset, scales| ops::Qwen35Fp8AttentionArgs {
            projected_query_gate: &projected_query_gate,
            projected_key: &projected_key,
            projected_value: &projected_value,
            query_norm: &norm,
            key_norm: &norm,
            positions: &positions,
            cache: ops::Qwen35Fp8KvCache {
                key: &key_cache[arm],
                value: &value_cache[arm],
            },
            scratch: ops::Qwen35AttentionScratch {
                query: &query,
                gate: &gate,
                output: &output,
                scores: &scores,
                partial_output: &partial_output,
                partial_stats: &partial_stats,
            },
            cache_offset,
            scales,
            rope_theta: 10_000_000.0,
        };

        if offset == 128 {
            let undersized = zero(vec![workspace_elements - 1], DType::F32);
            let non_unit = ops::Qwen35KvCacheScales {
                key: 0.5,
                value: 1.0,
            };
            for (invalid_offset, invalid_scale, invalid_workspace) in [
                (offset, unit, &undersized),
                (capacity, unit, &workspace),
                (offset, non_unit, &workspace),
            ] {
                assert!(ops::qwen35_attention_draft_cache(
                    &ctx,
                    args(1, invalid_offset, invalid_scale),
                    invalid_workspace,
                )
                .is_err());
                ctx.synchronize().unwrap();
                assert_eq!(
                    fp8_bytes(&key_cache[1]),
                    key_seed,
                    "rejected call changed K"
                );
                assert_eq!(
                    fp8_bytes(&value_cache[1]),
                    value_seed,
                    "rejected call changed V"
                );
            }

            let misaligned_key = CudaBuffer::alloc(key_seed.len() + 1, ctx.device_id())
                .unwrap()
                .view(1, key_seed.len())
                .unwrap()
                .as_tensor(Shape::new(shape.clone()), DType::F8E4M3)
                .unwrap();
            transfers::copy_cpu_to_cuda_on_stream(
                &ctx,
                Tensor::from_f8_e4m3(shape.clone(), &key_seed).unwrap(),
                &misaligned_key,
            )
            .unwrap();
            let invalid = ops::Qwen35Fp8AttentionArgs {
                cache: ops::Qwen35Fp8KvCache {
                    key: &misaligned_key,
                    value: &value_cache[1],
                },
                ..args(1, offset, unit)
            };
            assert!(ops::qwen35_attention_draft_cache(&ctx, invalid, &workspace).is_err());
            ctx.synchronize().unwrap();
            assert_eq!(fp8_bytes(&misaligned_key), key_seed);
            assert_eq!(fp8_bytes(&value_cache[1]), value_seed);
        }

        ops::qwen35_attention_static_cache(&ctx, 1, args(0, offset, unit)).unwrap();
        ctx.synchronize().unwrap();
        let expected_key = fp8_bytes(&key_cache[0]);
        let expected_value = fp8_bytes(&value_cache[0]);
        let expected_query = bf16_bits(&query);
        let expected_gate = bf16_bits(&gate);
        let expected_output = bf16_bits(&output);

        let launch = || ops::qwen35_attention_draft_cache(&ctx, args(1, offset, unit), &workspace);
        launch().unwrap();
        ctx.synchronize().unwrap();
        let actual_output = bf16_bits(&output);
        for (&actual, expected) in actual_output.iter().zip(expected_output) {
            let actual = bf16::from_bits(actual).to_f32();
            let expected = bf16::from_bits(expected).to_f32();
            assert!(
                actual.is_finite() && (actual - expected).abs() <= 0.004 * expected.abs() + 5e-4,
                "offset {offset}: {actual} vs {expected}"
            );
        }
        assert_eq!(bf16_bits(&query), expected_query);
        assert_eq!(bf16_bits(&gate), expected_gate);

        let graph = apxinf_cuda_new::capture(&ctx, launch).unwrap();
        for cache in [&key_cache[1], &value_cache[1]] {
            CudaBuffer::from_tensor(cache)
                .unwrap()
                .view(row, kv_width)
                .unwrap()
                .fill_async(0x7e, ctx.stream())
                .unwrap();
        }
        CudaBuffer::from_tensor(&output)
            .unwrap()
            .fill_async(0xff, ctx.stream())
            .unwrap();
        graph.replay().unwrap();
        ctx.synchronize().unwrap();

        assert_eq!(
            bf16_bits(&output),
            actual_output,
            "offset {offset}: graph/eager"
        );
        for (cache, baseline, original) in [
            (&key_cache[1], &expected_key, &key_seed),
            (&value_cache[1], &expected_value, &value_seed),
        ] {
            let actual = fp8_bytes(cache);
            assert_eq!(
                &actual[..row],
                &original[..row],
                "offset {offset}: prefix changed"
            );
            assert_eq!(
                &actual[row..row + kv_width],
                &baseline[row..row + kv_width],
                "offset {offset}: appended FP8 bytes"
            );
            assert_eq!(
                &actual[row + kv_width..],
                &original[row + kv_width..],
                "offset {offset}: tail changed"
            );
        }
    }
}

#[test]
fn qwen35_static_kv_append_matches_full_attention_cache_bytes() {
    let ctx = CudaContext::new_with_autotune(0, false).unwrap();
    let width = ops::QWEN35_ATTENTION_KV_WIDTH;
    let head_dim = ops::QWEN35_ATTENTION_HEAD_DIM;
    let qkv = patterned_bf16(ops::QWEN35_ATTENTION_Q_PROJECTION_WIDTH, 17, 101, 37.0);
    let key = patterned_bf16(width, 13, 83, 31.0);
    let value = patterned_bf16(width, 19, 79, 29.0);
    let norm = patterned_bf16(head_dim, 11, 29, 112.0);
    let key_gpu = transfers::to_cuda_on_stream(&ctx, bf16_tensor(&key, vec![1, width])).unwrap();
    let value_gpu =
        transfers::to_cuda_on_stream(&ctx, bf16_tensor(&value, vec![1, width])).unwrap();
    let norm_gpu = transfers::to_cuda_on_stream(&ctx, bf16_tensor(&norm, vec![head_dim])).unwrap();
    let unit = ops::Qwen35KvCacheScales {
        key: 1.0,
        value: 1.0,
    };

    for offset in [127, 128] {
        let shape = fp8_cache_shape(offset + 2);
        let row = offset * width;
        let capacity = shape[0] * ops::QWEN35_KV_CACHE_BLOCK_TOKENS;
        let mut seed = vec![0x7e; capacity * width];
        seed[..row].fill(0xb8);
        let upload = || {
            let host = Tensor::from_f8_e4m3(shape.clone(), &seed).unwrap();
            transfers::to_cuda_on_stream(&ctx, host).unwrap()
        };
        let legacy_key = upload();
        let legacy_value = upload();
        let append_key = upload();
        let append_value = upload();
        let position_values = positions(offset + 1, 1);
        let position = transfers::to_cuda_on_stream(
            &ctx,
            Tensor::from_i32(vec![3, 1], &position_values).unwrap(),
        )
        .unwrap();

        let args = |cache_offset| ops::Qwen35StaticFp8KvAppendArgs {
            projected_key: &key_gpu,
            projected_value: &value_gpu,
            key_norm: &norm_gpu,
            positions: &position,
            cache: ops::Qwen35Fp8KvCache {
                key: &append_key,
                value: &append_value,
            },
            cache_offset,
            scales: unit,
            rope_theta: 10_000_000.0,
        };
        let overlap = ops::Qwen35StaticFp8KvAppendArgs {
            cache: ops::Qwen35Fp8KvCache {
                key: &append_key,
                value: &append_key,
            },
            ..args(offset)
        };
        assert!(ops::qwen35_static_kv_append(&ctx, 1, overlap).is_err());
        assert!(ops::qwen35_static_kv_append(&ctx, 1, args(capacity)).is_err());
        ctx.synchronize().unwrap();
        assert_eq!(fp8_bytes(&append_key), seed);
        assert_eq!(fp8_bytes(&append_value), seed);

        run_attention(
            &ctx,
            1,
            offset,
            &qkv,
            &key,
            &value,
            &norm,
            &norm,
            &position_values,
            &legacy_key,
            &legacy_value,
            unit,
        );
        let expected_key = fp8_bytes(&legacy_key);
        let expected_value = fp8_bytes(&legacy_value);
        assert_ne!(&expected_key[row..row + width], &seed[row..row + width]);
        let value_bytes: Vec<_> = value
            .iter()
            .copied()
            .map(ops::encode_cache_element)
            .collect();
        assert_eq!(&expected_value[row..row + width], value_bytes.as_slice());
        for cache in [&expected_key, &expected_value] {
            assert_eq!(&cache[..row], &seed[..row]);
            assert_eq!(&cache[row + width..], &seed[row + width..]);
        }

        let append = || ops::qwen35_static_kv_append(&ctx, 1, args(offset));
        append().unwrap();
        ctx.synchronize().unwrap();
        assert_eq!(fp8_bytes(&append_key), expected_key);
        assert_eq!(fp8_bytes(&append_value), expected_value);
        let graph = apxinf_cuda_new::capture(&ctx, append).unwrap();
        for cache in [&append_key, &append_value] {
            CudaBuffer::from_tensor(cache)
                .unwrap()
                .view(row, width)
                .unwrap()
                .fill_async(0x7e, ctx.stream())
                .unwrap();
        }
        graph.replay().unwrap();
        ctx.synchronize().unwrap();
        assert_eq!(fp8_bytes(&append_key), expected_key);
        assert_eq!(fp8_bytes(&append_value), expected_value);
    }
}
