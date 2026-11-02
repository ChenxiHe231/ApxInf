use apxinf_core::{DType, Shape, Tensor};
use apxinf_cuda_new::{ops, transfers, CudaBuffer, CudaContext};
use half::bf16;

fn zeros(ctx: &CudaContext, shape: Vec<usize>, dtype: DType) -> Tensor {
    CudaBuffer::alloc_zeros_async(
        shape.iter().product::<usize>() * dtype.size_in_bytes(),
        ctx.device_id(),
        ctx.stream(),
    )
    .unwrap()
    .as_tensor(Shape::new(shape), dtype)
    .unwrap()
}

fn patterned_bf16(length: usize, multiplier: usize, period: usize, divisor: f32) -> Vec<bf16> {
    (0..length)
        .map(|index| {
            bf16::from_f32(((index * multiplier % period) as f32 - (period / 2) as f32) / divisor)
        })
        .collect()
}

fn dynamic_quantized(values: &[bf16]) -> (f32, Vec<f32>) {
    let absmax = values
        .iter()
        .map(|value| value.to_f32().abs())
        .fold(0.0f32, f32::max);
    let scale = (absmax / ops::FP8_E4M3_MAX_FINITE).max(1.0e-12);
    let decoded = values
        .iter()
        .map(|value| ops::decode_e4m3(ops::encode_e4m3(value.to_f32() / scale)) * scale)
        .collect();
    (scale, decoded)
}

#[test]
fn dynamic_fp8_decode_matches_quantized_reference_across_splits() {
    let ctx = CudaContext::new(0).unwrap();
    let tokens = ops::QWEN35_DECODE_SPLIT_TOKENS + 17;
    let capacity = tokens + 19;
    let kv_elements = tokens * ops::QWEN35_ATTENTION_KV_WIDTH;
    let query_elements = ops::QWEN35_ATTENTION_Q_HEADS * ops::QWEN35_ATTENTION_HEAD_DIM;
    let key_host = patterned_bf16(kv_elements, 17, 257, 91.0);
    let value_host = patterned_bf16(kv_elements, 29, 251, 83.0);
    let query_host = patterned_bf16(query_elements, 31, 263, 97.0);
    let key = transfers::to_cuda(
        &Tensor::from_bf16(
            vec![
                tokens,
                ops::QWEN35_ATTENTION_KV_HEADS,
                ops::QWEN35_ATTENTION_HEAD_DIM,
            ],
            &key_host,
        )
        .unwrap(),
        0,
    )
    .unwrap();
    let value = transfers::to_cuda(
        &Tensor::from_bf16(
            vec![
                tokens,
                ops::QWEN35_ATTENTION_KV_HEADS,
                ops::QWEN35_ATTENTION_HEAD_DIM,
            ],
            &value_host,
        )
        .unwrap(),
        0,
    )
    .unwrap();
    let query = transfers::to_cuda(
        &Tensor::from_bf16(
            vec![
                1,
                ops::QWEN35_ATTENTION_Q_HEADS,
                ops::QWEN35_ATTENTION_HEAD_DIM,
            ],
            &query_host,
        )
        .unwrap(),
        0,
    )
    .unwrap();
    let key_cache = zeros(
        &ctx,
        vec![
            capacity,
            ops::QWEN35_ATTENTION_KV_HEADS,
            ops::QWEN35_ATTENTION_HEAD_DIM,
        ],
        DType::F8E4M3,
    );
    let value_cache = zeros(
        &ctx,
        vec![
            capacity,
            ops::QWEN35_ATTENTION_KV_HEADS,
            ops::QWEN35_ATTENTION_HEAD_DIM,
        ],
        DType::F8E4M3,
    );
    let key_scales = zeros(
        &ctx,
        vec![capacity, ops::QWEN35_ATTENTION_KV_HEADS],
        DType::F32,
    );
    let value_scales = zeros(
        &ctx,
        vec![capacity, ops::QWEN35_ATTENTION_KV_HEADS],
        DType::F32,
    );
    let cache = || ops::Qwen35DynamicFp8KvCache {
        key: &key_cache,
        value: &value_cache,
        key_scales: &key_scales,
        value_scales: &value_scales,
    };
    ops::qwen35_dynamic_kv_append(&ctx, tokens, 0, &key, &value, cache()).unwrap();

    let max_splits = ops::qwen35_dynamic_decode_max_splits(capacity).unwrap();
    let partial_max = zeros(
        &ctx,
        vec![ops::QWEN35_ATTENTION_Q_HEADS, max_splits],
        DType::F32,
    );
    let partial_sum = zeros(
        &ctx,
        vec![ops::QWEN35_ATTENTION_Q_HEADS, max_splits],
        DType::F32,
    );
    let partial_output = zeros(
        &ctx,
        vec![
            ops::QWEN35_ATTENTION_Q_HEADS,
            max_splits,
            ops::QWEN35_ATTENTION_HEAD_DIM,
        ],
        DType::F32,
    );
    let output = zeros(
        &ctx,
        vec![
            1,
            ops::QWEN35_ATTENTION_Q_HEADS,
            ops::QWEN35_ATTENTION_HEAD_DIM,
        ],
        DType::BF16,
    );
    ops::qwen35_dynamic_decode_attention(
        &ctx,
        tokens,
        &query,
        cache(),
        ops::Qwen35DynamicDecodeScratch {
            partial_max: &partial_max,
            partial_sum: &partial_sum,
            partial_output: &partial_output,
        },
        &output,
    )
    .unwrap();
    ctx.synchronize().unwrap();

    let actual_key_scales = transfers::to_cpu(&key_scales)
        .unwrap()
        .as_f32()
        .unwrap()
        .to_vec();
    let actual_value_scales = transfers::to_cpu(&value_scales)
        .unwrap()
        .as_f32()
        .unwrap()
        .to_vec();
    let mut decoded_key = vec![0.0f32; kv_elements];
    let mut decoded_value = vec![0.0f32; kv_elements];
    for position in 0..tokens {
        for head in 0..ops::QWEN35_ATTENTION_KV_HEADS {
            let row = position * ops::QWEN35_ATTENTION_KV_HEADS + head;
            let begin = row * ops::QWEN35_ATTENTION_HEAD_DIM;
            let end = begin + ops::QWEN35_ATTENTION_HEAD_DIM;
            let (key_scale, key_values) = dynamic_quantized(&key_host[begin..end]);
            let (value_scale, value_values) = dynamic_quantized(&value_host[begin..end]);
            assert!((actual_key_scales[row] - key_scale).abs() <= 1.0e-7);
            assert!((actual_value_scales[row] - value_scale).abs() <= 1.0e-7);
            decoded_key[begin..end].copy_from_slice(&key_values);
            decoded_value[begin..end].copy_from_slice(&value_values);
        }
    }

    let actual = transfers::to_cpu(&output)
        .unwrap()
        .as_bf16()
        .unwrap()
        .iter()
        .map(|value| value.to_f32())
        .collect::<Vec<_>>();
    for query_head in 0..ops::QWEN35_ATTENTION_Q_HEADS {
        let kv_head = query_head / (ops::QWEN35_ATTENTION_Q_HEADS / ops::QWEN35_ATTENTION_KV_HEADS);
        let query_begin = query_head * ops::QWEN35_ATTENTION_HEAD_DIM;
        let query_values = &query_host[query_begin..query_begin + ops::QWEN35_ATTENTION_HEAD_DIM];
        let mut scores = Vec::with_capacity(tokens);
        for position in 0..tokens {
            let key_begin = (position * ops::QWEN35_ATTENTION_KV_HEADS + kv_head)
                * ops::QWEN35_ATTENTION_HEAD_DIM;
            let score = query_values
                .iter()
                .zip(&decoded_key[key_begin..key_begin + ops::QWEN35_ATTENTION_HEAD_DIM])
                .map(|(query, key)| query.to_f32() * key)
                .sum::<f32>()
                / (ops::QWEN35_ATTENTION_HEAD_DIM as f32).sqrt();
            scores.push(score);
        }
        let maximum = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let weights = scores
            .iter()
            .map(|score| (*score - maximum).exp())
            .collect::<Vec<_>>();
        let denominator = weights.iter().sum::<f32>();
        for dim in 0..ops::QWEN35_ATTENTION_HEAD_DIM {
            let expected = weights
                .iter()
                .enumerate()
                .map(|(position, weight)| {
                    let value_index = (position * ops::QWEN35_ATTENTION_KV_HEADS + kv_head)
                        * ops::QWEN35_ATTENTION_HEAD_DIM
                        + dim;
                    weight * decoded_value[value_index]
                })
                .sum::<f32>()
                / denominator;
            let actual_value = actual[query_begin + dim];
            assert!(
                (actual_value - expected).abs() <= 0.005,
                "head={query_head} dim={dim}: {actual_value} != {expected}"
            );
        }
    }
}

#[test]
fn fused_prepare_matches_separate_bf16_norm_rope_and_quantization() {
    let ctx = CudaContext::new(0).unwrap();
    for (tokens, offset, position) in [(1, 0, 0), (17, 3, 49151)] {
        let capacity = tokens + offset + 7;
        let mut kh = patterned_bf16(tokens * 512, 23, 331, 39.0);
        let mut vh = patterned_bf16(tokens * 512, 31, 257, 19.0);
        // A zero head exercises the minimum scale, while an outlier tests
        // scale selection independent of K normalization and RoPE.
        kh[..256].fill(bf16::ZERO);
        vh[..256].fill(bf16::ZERO);
        vh[256] = bf16::from_f32(8192.0);
        let key =
            transfers::to_cuda(&Tensor::from_bf16(vec![tokens, 2, 256], &kh).unwrap(), 0).unwrap();
        let value =
            transfers::to_cuda(&Tensor::from_bf16(vec![tokens, 2, 256], &vh).unwrap(), 0).unwrap();
        let weight = transfers::to_cuda(
            &Tensor::from_bf16(vec![256], &patterned_bf16(256, 7, 97, 173.0)).unwrap(),
            0,
        )
        .unwrap();
        let pos: Vec<i32> = (0..tokens).map(|i| position + i as i32).collect();
        let pos1 = transfers::to_cuda(&Tensor::from_i32(vec![tokens], &pos).unwrap(), 0).unwrap();
        let pos3 = transfers::to_cuda(
            &Tensor::from_i32(
                vec![3, tokens],
                &pos.iter()
                    .copied()
                    .cycle()
                    .take(tokens * 3)
                    .collect::<Vec<_>>(),
            )
            .unwrap(),
            0,
        )
        .unwrap();
        let payloads: Vec<Tensor> = (0..4)
            .map(|_| zeros(&ctx, vec![capacity, 2, 256], DType::F8E4M3))
            .collect();
        let scales: Vec<Tensor> = (0..4)
            .map(|_| zeros(&ctx, vec![capacity, 2], DType::F32))
            .collect();
        let cache = |i: usize| ops::Qwen35DynamicFp8KvCache {
            key: &payloads[i],
            value: &payloads[i + 1],
            key_scales: &scales[i],
            value_scales: &scales[i + 1],
        };
        ops::qwen35_dynamic_kv_prepare(
            &ctx,
            tokens,
            offset,
            &key,
            &value,
            &weight,
            &pos3,
            10_000_000.0,
            cache(0),
        )
        .unwrap();
        let key2d = CudaBuffer::from_tensor(&key)
            .unwrap()
            .as_tensor(Shape::new(vec![tokens * 2, 256]), DType::BF16)
            .unwrap();
        ops::head_rms_norm(&ctx, &key2d, &weight, ops::QWEN35_ATTENTION_RMS_EPSILON).unwrap();
        ops::partial_rope(&ctx, &key, &pos1, 64, 10_000_000.0).unwrap();
        ops::qwen35_dynamic_kv_append(&ctx, tokens, offset, &key, &value, cache(2)).unwrap();
        ctx.synchronize().unwrap();
        for i in 0..2 {
            let lhs = transfers::to_cpu(&payloads[i]).unwrap();
            let rhs = transfers::to_cpu(&payloads[i + 2]).unwrap();
            // Exact bytes also verify untouched prefix and tail rows.
            assert_eq!(lhs.as_f8_e4m3().unwrap(), rhs.as_f8_e4m3().unwrap());
            let lhs = transfers::to_cpu(&scales[i]).unwrap();
            let rhs = transfers::to_cpu(&scales[i + 2]).unwrap();
            assert_eq!(lhs.as_f32().unwrap(), rhs.as_f32().unwrap());
        }
    }
}

// Explicit opt-in attention-only timing: dequant+FA2 and direct FP8 use the
// same dynamic cache, queries and valid prefix. Run under the device GPU lock.
#[test]
#[ignore]
fn benchmark_dynamic_fp8_decode_against_dequant_fa2() {
    use std::time::Instant;
    let ctx = CudaContext::new(0).unwrap();
    for tokens in [4096, 8192, 16384, 32768, 49152] {
        let key = transfers::to_cuda(
            &Tensor::from_bf16(
                vec![tokens, 2, 256],
                &patterned_bf16(tokens * 512, 17, 257, 91.0),
            )
            .unwrap(),
            0,
        )
        .unwrap();
        let value = transfers::to_cuda(
            &Tensor::from_bf16(
                vec![tokens, 2, 256],
                &patterned_bf16(tokens * 512, 29, 251, 83.0),
            )
            .unwrap(),
            0,
        )
        .unwrap();
        let query = transfers::to_cuda(
            &Tensor::from_bf16(vec![1, 16, 256], &patterned_bf16(4096, 31, 263, 97.0)).unwrap(),
            0,
        )
        .unwrap();
        let kc = zeros(&ctx, vec![tokens, 2, 256], DType::F8E4M3);
        let vc = zeros(&ctx, vec![tokens, 2, 256], DType::F8E4M3);
        let ks = zeros(&ctx, vec![tokens, 2], DType::F32);
        let vs = zeros(&ctx, vec![tokens, 2], DType::F32);
        let cache = || ops::Qwen35DynamicFp8KvCache {
            key: &kc,
            value: &vc,
            key_scales: &ks,
            value_scales: &vs,
        };
        ops::qwen35_dynamic_kv_append(&ctx, tokens, 0, &key, &value, cache()).unwrap();
        let splits = ops::qwen35_dynamic_decode_max_splits(tokens).unwrap();
        let pm = zeros(&ctx, vec![16, splits], DType::F32);
        let ps = zeros(&ctx, vec![16, splits], DType::F32);
        let po = zeros(&ctx, vec![16, splits, 256], DType::F32);
        let out = zeros(&ctx, vec![1, 16, 256], DType::BF16);
        let mut reference = zeros(&ctx, vec![1, 1, 16, 256], DType::BF16);
        let kb = zeros(&ctx, vec![1, tokens, 2, 256], DType::BF16);
        let vb = zeros(&ctx, vec![1, tokens, 2, 256], DType::BF16);
        let q4 = CudaBuffer::from_tensor(&query)
            .unwrap()
            .as_tensor(Shape::new(vec![1, 1, 16, 256]), DType::BF16)
            .unwrap();
        let mut samples = [Vec::new(), Vec::new()];
        // Interleave both routes and synchronize around each timed sample.
        // Timing includes Rust dispatch and launch overhead for both routes.
        for round in 0..12 {
            for which in [round % 2, 1 - round % 2] {
                ctx.synchronize().unwrap();
                let start = Instant::now();
                if which == 0 {
                    ops::qwen35_dynamic_decode_attention(
                        &ctx,
                        tokens,
                        &query,
                        cache(),
                        ops::Qwen35DynamicDecodeScratch {
                            partial_max: &pm,
                            partial_sum: &ps,
                            partial_output: &po,
                        },
                        &out,
                    )
                    .unwrap();
                } else {
                    ops::qwen35_dynamic_kv_dequant(&ctx, tokens, cache(), &kb, &vb).unwrap();
                    let mut args = ops::KvCacheAttentionArgs::new(&q4, &kb, &vb, &mut reference);
                    args.query_start = tokens - 1;
                    args.valid_key_tokens = tokens;
                    args.policy.allow_fallback = false;
                    ops::kv_cache_attention(&ctx, args).unwrap();
                }
                ctx.synchronize().unwrap();
                if round >= 2 {
                    samples[which].push(start.elapsed().as_secs_f64() * 1e6);
                }
            }
        }
        for sample in &mut samples {
            sample.sort_by(f64::total_cmp);
        }
        let direct_us = samples[0][samples[0].len() / 2];
        let dequant_fa2_us = samples[1][samples[1].len() / 2];
        let actual = transfers::to_cpu(&out).unwrap();
        let expected = transfers::to_cpu(&reference).unwrap();
        let max_abs_error = actual
            .as_bf16()
            .unwrap()
            .iter()
            .zip(expected.as_bf16().unwrap())
            .map(|(a, b)| (a.to_f32() - b.to_f32()).abs())
            .fold(0.0f32, f32::max);
        assert!(
            max_abs_error <= 0.005,
            "tokens={tokens} max_abs_error={max_abs_error}"
        );
        println!("{{\"tokens\":{tokens},\"direct_us\":{direct_us:.3},\"dequant_fa2_us\":{dequant_fa2_us:.3},\"speedup\":{:.3},\"max_abs_error\":{max_abs_error}}}",dequant_fa2_us/direct_us);
    }
}
