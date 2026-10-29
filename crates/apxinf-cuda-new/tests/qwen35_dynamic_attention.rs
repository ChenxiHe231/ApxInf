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
                (actual_value - expected).abs() <= 0.025,
                "head={query_head} dim={dim}: {actual_value} != {expected}"
            );
        }
    }
}
