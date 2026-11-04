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

// Pageable cudaMemcpy may return before its DMA reaches device memory, while
// kernels run on the context's non-blocking stream. The device-synchronizing
// readback orders the upload before any later stream work and verifies it.
fn ordered_to_cuda(tensor: &Tensor, device: usize) -> apxinf_core::Result<Tensor> {
    let uploaded = transfers::to_cuda(tensor, device)?;
    let readback = transfers::to_cpu(&uploaded)?;
    assert_eq!(readback.storage().as_cpu(), tensor.storage().as_cpu());
    Ok(uploaded)
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
    let mut key_host = patterned_bf16(kv_elements, 17, 257, 91.0);
    let mut value_host = patterned_bf16(kv_elements, 29, 251, 83.0);
    let mut query_host = patterned_bf16(query_elements, 31, 263, 97.0);
    for row in 0..tokens * 2 {
        let ka = if row % 13 == 0 {
            0.0
        } else {
            0.125 * (1 << (row % 7)) as f32
        };
        let va = if row % 17 == 0 {
            0.0
        } else {
            0.25 * (1 << ((row * 3) % 7)) as f32
        };
        for dim in 0..256 {
            key_host[row * 256 + dim] = bf16::from_f32(key_host[row * 256 + dim].to_f32() * ka);
            value_host[row * 256 + dim] = bf16::from_f32(value_host[row * 256 + dim].to_f32() * va);
        }
    }
    query_host[..256].fill(bf16::ZERO);
    let key = ordered_to_cuda(
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
    let value = ordered_to_cuda(
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
    let query = ordered_to_cuda(
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
    for valid in [1, 7, 16, 17, 255, 256, 257, tokens] {
        ops::qwen35_dynamic_decode_attention(
            &ctx,
            valid,
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
                assert!(
                    (actual_key_scales[row] - key_scale).abs()
                        <= key_scale.abs() * 1.0e-6 + 1.0e-12
                );
                assert!(
                    (actual_value_scales[row] - value_scale).abs()
                        <= value_scale.abs() * 1.0e-6 + 1.0e-12
                );
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
            let kv_head =
                query_head / (ops::QWEN35_ATTENTION_Q_HEADS / ops::QWEN35_ATTENTION_KV_HEADS);
            let query_begin = query_head * ops::QWEN35_ATTENTION_HEAD_DIM;
            let query_values =
                &query_host[query_begin..query_begin + ops::QWEN35_ATTENTION_HEAD_DIM];
            let mut scores = Vec::with_capacity(valid);
            for position in 0..valid {
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
                    (actual_value - expected).abs() <= 0.004 * expected.abs() + 0.002,
                    "head={query_head} dim={dim}: {actual_value} != {expected}"
                );
            }
        }
    }
    // The fused gate must exactly preserve the separate BF16->gate rounding.
    let gh: Vec<_> = (0..query_elements)
        .map(|i| bf16::from_f32([-80.0, -8.0, -1.0, 0.0, 1.0, 8.0, 80.0][i % 7]))
        .collect();
    let gate = ordered_to_cuda(&Tensor::from_bf16(vec![1, 16, 256], &gh).unwrap(), 0).unwrap();
    let gated = zeros(&ctx, vec![1, 16, 256], DType::BF16);
    ops::qwen35_dynamic_decode_attention_gated(
        &ctx,
        tokens,
        &query,
        &gate,
        cache(),
        ops::Qwen35DynamicDecodeScratch {
            partial_max: &partial_max,
            partial_sum: &partial_sum,
            partial_output: &partial_output,
        },
        &gated,
    )
    .unwrap();
    let out2d = CudaBuffer::from_tensor(&output)
        .unwrap()
        .as_tensor(Shape::new(vec![16, 256]), DType::BF16)
        .unwrap();
    let gate2d = CudaBuffer::from_tensor(&gate)
        .unwrap()
        .as_tensor(Shape::new(vec![16, 256]), DType::BF16)
        .unwrap();
    ops::apply_output_gate(&ctx, &out2d, &gate2d).unwrap();
    ctx.synchronize().unwrap();
    assert_eq!(
        transfers::to_cpu(&gated).unwrap().as_bf16().unwrap(),
        transfers::to_cpu(&output).unwrap().as_bf16().unwrap()
    );
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
            ordered_to_cuda(&Tensor::from_bf16(vec![tokens, 2, 256], &kh).unwrap(), 0).unwrap();
        let value =
            ordered_to_cuda(&Tensor::from_bf16(vec![tokens, 2, 256], &vh).unwrap(), 0).unwrap();
        let weight = ordered_to_cuda(
            &Tensor::from_bf16(vec![256], &patterned_bf16(256, 7, 97, 173.0)).unwrap(),
            0,
        )
        .unwrap();
        let pos: Vec<i32> = (0..tokens).map(|i| position + i as i32).collect();
        let pos1 = ordered_to_cuda(&Tensor::from_i32(vec![tokens], &pos).unwrap(), 0).unwrap();
        let pos3 = ordered_to_cuda(
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

#[test]
fn fused_query_prepare_matches_separate_norm_rope_exactly() {
    let ctx = CudaContext::new(0).unwrap();
    for (tokens, start) in [(1, 0), (1, 49151), (17, 524287)] {
        let mut host = patterned_bf16(tokens * 8192, 23, 331, 39.0);
        host[..256].fill(bf16::ZERO);
        host[1024] = bf16::from_f32(8192.0);
        let projected =
            ordered_to_cuda(&Tensor::from_bf16(vec![tokens, 16, 512], &host).unwrap(), 0)
                .unwrap();
        let norm = ordered_to_cuda(
            &Tensor::from_bf16(vec![256], &patterned_bf16(256, 7, 97, 173.0)).unwrap(),
            0,
        )
        .unwrap();
        let p: Vec<i32> = (0..tokens).map(|i| start + i as i32).collect();
        let pos1 = ordered_to_cuda(&Tensor::from_i32(vec![tokens], &p).unwrap(), 0).unwrap();
        let p3: Vec<i32> = p.iter().copied().cycle().take(tokens * 3).collect();
        let pos3 = ordered_to_cuda(&Tensor::from_i32(vec![3, tokens], &p3).unwrap(), 0).unwrap();
        let q1 = zeros(&ctx, vec![tokens, 16, 256], DType::BF16);
        let g1 = zeros(&ctx, vec![tokens, 16, 256], DType::BF16);
        let q2 = zeros(&ctx, vec![tokens, 16, 256], DType::BF16);
        let g2 = zeros(&ctx, vec![tokens, 16, 256], DType::BF16);
        ops::qwen35_query_prepare(
            &ctx,
            tokens,
            &projected,
            &norm,
            &pos3,
            10_000_000.0,
            &q1,
            &g1,
        )
        .unwrap();
        ops::split_query_and_gate(&ctx, &projected, &q2, &g2).unwrap();
        let q2d = CudaBuffer::from_tensor(&q2)
            .unwrap()
            .as_tensor(Shape::new(vec![tokens * 16, 256]), DType::BF16)
            .unwrap();
        ops::head_rms_norm(&ctx, &q2d, &norm, ops::QWEN35_ATTENTION_RMS_EPSILON).unwrap();
        ops::partial_rope(&ctx, &q2, &pos1, 64, 10_000_000.0).unwrap();
        ctx.synchronize().unwrap();
        assert_eq!(
            transfers::to_cpu(&q1).unwrap().as_bf16().unwrap(),
            transfers::to_cpu(&q2).unwrap().as_bf16().unwrap()
        );
        assert_eq!(
            transfers::to_cpu(&g1).unwrap().as_bf16().unwrap(),
            transfers::to_cpu(&g2).unwrap().as_bf16().unwrap()
        );
        assert!(ops::qwen35_query_prepare(
            &ctx,
            tokens,
            &projected,
            &norm,
            &pos3,
            f32::NAN,
            &q1,
            &g1
        )
        .is_err());
        assert!(ops::qwen35_query_prepare(
            &ctx,
            tokens,
            &projected,
            &norm,
            &pos3,
            10_000_000.0,
            &q1,
            &q1
        )
        .is_err());
    }
}

#[test]
fn dynamic_decode_negative_scores_and_large_merge() {
    let ctx = CudaContext::new(0).unwrap();
    let tokens = 65537; // 257 splits exercises shared-weight chunking.
    let kh = vec![bf16::from_f32(1.0); tokens * 512];
    let vh: Vec<_> = (0..tokens * 512)
        .map(|i| bf16::from_f32(if (i / 256) % 2 == 0 { 2.0 } else { -3.0 }))
        .collect();
    let key =
        ordered_to_cuda(&Tensor::from_bf16(vec![tokens, 2, 256], &kh).unwrap(), 0).unwrap();
    let value =
        ordered_to_cuda(&Tensor::from_bf16(vec![tokens, 2, 256], &vh).unwrap(), 0).unwrap();
    let query = ordered_to_cuda(
        &Tensor::from_bf16(vec![1, 16, 256], &vec![bf16::from_f32(-10.0); 4096]).unwrap(),
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
    ctx.synchronize().unwrap();
    let out = transfers::to_cpu(&out).unwrap();
    for (i, v) in out.as_bf16().unwrap().iter().enumerate() {
        assert_eq!(v.to_f32(), if i / 256 < 8 { 2.0 } else { -3.0 });
    }
}

// An attention oracle independent of the GPU quantizer: every finite E4M3
// magnitude/sign occurs in the payload, with nonuniform FP32 token/head scales.
// FP64 evaluates raw dot*scale and weighted V; BF16 is rounded only at output.
#[test]
fn dynamic_decode_long_tail_matches_f64_with_tiny_query_and_weights() {
    let ctx = CudaContext::new(0).unwrap();
    // 8193 crosses the short-prefix split boundary with a 1-token tail. The
    // override lets the stress matrix also cover 16K+ so that the 256/512/1024
    // split schedules are checked against the f64 oracle well past the
    // boundary, where the tile count per split actually differs.
    let tokens = std::env::var("APXINF_QWEN35_TEST_LONG_TOKENS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|value| *value > 0)
        .unwrap_or(8193);
    let mut state = 0x31415926u32;
    let mut bytes = || {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        ((state % 127) as u8) | (((state >> 16) as u8) & 0x80)
    };
    let kh: Vec<_> = (0..tokens * 512).map(|_| bytes()).collect();
    let vh: Vec<_> = (0..tokens * 512).map(|_| bytes()).collect();
    let ksh: Vec<_> = (0..tokens * 2)
        .map(|i| 0.001 * (1u32 << (i % 8)) as f32)
        .collect();
    let vsh: Vec<_> = (0..tokens * 2)
        .map(|i| 0.015625 * (1u32 << ((i * 3) % 7)) as f32)
        .collect();
    let kc =
        ordered_to_cuda(&Tensor::from_f8_e4m3(vec![tokens, 2, 256], &kh).unwrap(), 0).unwrap();
    let vc =
        ordered_to_cuda(&Tensor::from_f8_e4m3(vec![tokens, 2, 256], &vh).unwrap(), 0).unwrap();
    let ks = ordered_to_cuda(&Tensor::from_f32(vec![tokens, 2], &ksh).unwrap(), 0).unwrap();
    let vs = ordered_to_cuda(&Tensor::from_f32(vec![tokens, 2], &vsh).unwrap(), 0).unwrap();
    let cache = || ops::Qwen35DynamicFp8KvCache {
        key: &kc,
        value: &vc,
        key_scales: &ks,
        value_scales: &vs,
    };
    let splits = ops::qwen35_dynamic_decode_max_splits(tokens).unwrap();
    let pm = zeros(&ctx, vec![16, splits], DType::F32);
    let ps = zeros(&ctx, vec![16, splits], DType::F32);
    let po = zeros(&ctx, vec![16, splits, 256], DType::F32);
    let output = zeros(&ctx, vec![1, 16, 256], DType::BF16);
    let mut qh = patterned_bf16(4096, 31, 263, 127.0);
    qh[..256].fill(bf16::ZERO);
    for q in &mut qh[512..768] {
        *q = bf16::from_f32(q.to_f32() * 8.0);
    }
    for (subnormal, valid) in [(false, 273), (false, tokens), (true, tokens)] {
        if subnormal {
            qh[256..512].fill(bf16::from_f32(1.0e-39));
        }
        let query =
            ordered_to_cuda(&Tensor::from_bf16(vec![1, 16, 256], &qh).unwrap(), 0).unwrap();
        ops::qwen35_dynamic_decode_attention(
            &ctx,
            valid,
            &query,
            cache(),
            ops::Qwen35DynamicDecodeScratch {
                partial_max: &pm,
                partial_sum: &ps,
                partial_output: &po,
            },
            &output,
        )
        .unwrap();
        ctx.synchronize().unwrap();
        let actual = transfers::to_cpu(&output).unwrap();
        let mut squared_error = 0.0f64;
        let mut squared_reference = 0.0f64;
        let mut max_abs_error = 0.0f64;
        let mut tiny_weights = 0usize;
        for head in 0..16 {
            let kv_head = head / 8;
            let mut scores = vec![0.0f64; valid];
            for (t, score) in scores.iter_mut().enumerate() {
                let row = t * 2 + kv_head;
                let dot: f64 = (0..256)
                    .map(|d| {
                        f64::from(qh[head * 256 + d].to_f32())
                            * f64::from(ops::decode_e4m3(kh[row * 256 + d]))
                    })
                    .sum();
                *score = dot * f64::from(ksh[row]) * 0.0625;
            }
            let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let weights: Vec<_> = scores.iter().map(|v| (v - maximum).exp()).collect();
            tiny_weights += weights
                .iter()
                .filter(|&&w| w > 0.0 && w < f64::from(f32::MIN_POSITIVE))
                .count();
            let denominator: f64 = weights.iter().sum();
            for d in 0..256 {
                let reference: f64 = weights
                    .iter()
                    .enumerate()
                    .map(|(t, w)| {
                        let row = t * 2 + kv_head;
                        w * f64::from(vsh[row]) * f64::from(ops::decode_e4m3(vh[row * 256 + d]))
                    })
                    .sum::<f64>()
                    / denominator;
                let got = f64::from(actual.as_bf16().unwrap()[head * 256 + d].to_f32());
                let error = (got - reference).abs();
                // Covers final BF16 half-ULP, plus a small absolute allowance
                // for FP32 dot/reduction order in cancellation-heavy outputs.
                assert!(error <= 0.004 * reference.abs() + 5.0e-4,
                    "valid={valid} subnormal={subnormal} head={head} d={d} got={got} f64={reference}");
                squared_error += error * error;
                squared_reference += reference * reference;
                max_abs_error = max_abs_error.max(error);
            }
        }
        println!("{{\"oracle\":\"f64_raw_fp8\",\"tokens\":{valid},\"subnormal_query\":{subnormal},\"tiny_weights\":{tiny_weights},\"max_abs_error\":{max_abs_error},\"relative_rms_error\":{}}}", (squared_error / squared_reference).sqrt());
        assert!(
            tiny_weights > 0,
            "fixture must exercise probabilities below FP32's normal range"
        );
    }
}

// This tiny positive V scale needs a subnormal residual in the three-part
// decomposition. Tensor PV must take its FP32 tile fallback instead of FTZ.
#[test]
fn dynamic_decode_preserves_tiny_value_scale() {
    let ctx = CudaContext::new(0).unwrap();
    let tokens = 273;
    for (value_code, scale) in [(0x38, 1.234567e-33f32), (0x01, 2.0f32.powi(-118))] {
        let q = ordered_to_cuda(
            &Tensor::from_bf16(vec![1, 16, 256], &vec![bf16::ZERO; 4096]).unwrap(),
            0,
        )
        .unwrap();
        let kc = ordered_to_cuda(
            &Tensor::from_f8_e4m3(vec![tokens, 2, 256], &vec![0x38; tokens * 512]).unwrap(),
            0,
        )
        .unwrap();
        let vc = ordered_to_cuda(
            &Tensor::from_f8_e4m3(vec![tokens, 2, 256], &vec![value_code; tokens * 512]).unwrap(),
            0,
        )
        .unwrap();
        let ks = ordered_to_cuda(
            &Tensor::from_f32(vec![tokens, 2], &vec![1.0; tokens * 2]).unwrap(),
            0,
        )
        .unwrap();
        let vs = ordered_to_cuda(
            &Tensor::from_f32(vec![tokens, 2], &vec![scale; tokens * 2]).unwrap(),
            0,
        )
        .unwrap();
        let splits = ops::qwen35_dynamic_decode_max_splits(tokens).unwrap();
        let pm = zeros(&ctx, vec![16, splits], DType::F32);
        let ps = zeros(&ctx, vec![16, splits], DType::F32);
        let po = zeros(&ctx, vec![16, splits, 256], DType::F32);
        let out = zeros(&ctx, vec![1, 16, 256], DType::BF16);
        ops::qwen35_dynamic_decode_attention(
            &ctx,
            tokens,
            &q,
            ops::Qwen35DynamicFp8KvCache {
                key: &kc,
                value: &vc,
                key_scales: &ks,
                value_scales: &vs,
            },
            ops::Qwen35DynamicDecodeScratch {
                partial_max: &pm,
                partial_sum: &ps,
                partial_output: &po,
            },
            &out,
        )
        .unwrap();
        ctx.synchronize().unwrap();
        for got in transfers::to_cpu(&out).unwrap().as_bf16().unwrap() {
            assert!((got.to_f32() / (scale * ops::decode_e4m3(value_code)) - 1.0).abs() < 0.004);
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
    for tokens in [4096, 8192, 16384, 24576, 32768, 49152] {
        let key = ordered_to_cuda(
            &Tensor::from_bf16(
                vec![tokens, 2, 256],
                &patterned_bf16(tokens * 512, 17, 257, 91.0),
            )
            .unwrap(),
            0,
        )
        .unwrap();
        let value = ordered_to_cuda(
            &Tensor::from_bf16(
                vec![tokens, 2, 256],
                &patterned_bf16(tokens * 512, 29, 251, 83.0),
            )
            .unwrap(),
            0,
        )
        .unwrap();
        let query = ordered_to_cuda(
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
        println!(
            "{{\"tokens\":{tokens},\"direct_samples_us\":{:?},\"dequant_fa2_samples_us\":{:?}}}",
            samples[0], samples[1]
        );
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
