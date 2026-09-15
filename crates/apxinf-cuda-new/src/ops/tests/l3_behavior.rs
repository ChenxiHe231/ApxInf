//! Per-operator L3 semantic and conditional Graph behavior tests.
//!
//! Every new L3 operator must add a public semantic-contract test here. Add a
//! Graph replay test only when the operator has an independent execution path,
//! resource lifetime, binding rule, or capture behavior not already covered by
//! the shared framework tests.

use super::framework::{tensor, values};
use super::*;
use crate::CudaContext;
use half::bf16;

#[allow(clippy::too_many_arguments)]
pub(super) fn attention_reference(
    query: &[f32],
    key: &[f32],
    value: &[f32],
    batch: usize,
    query_tokens: usize,
    key_tokens: usize,
    query_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    scale: f32,
    causal: bool,
) -> Vec<f32> {
    let mut output = vec![0.0; batch * query_tokens * query_heads * head_dim];
    for batch_index in 0..batch {
        for query_token in 0..query_tokens {
            for query_head in 0..query_heads {
                let kv_head = query_head / (query_heads / kv_heads);
                let valid_keys = if causal {
                    (query_token + key_tokens - query_tokens + 1).min(key_tokens)
                } else {
                    key_tokens
                };
                let mut scores = Vec::with_capacity(valid_keys);
                for key_token in 0..valid_keys {
                    let mut score = 0.0;
                    for dimension in 0..head_dim {
                        let q = (((batch_index * query_tokens + query_token) * query_heads
                            + query_head)
                            * head_dim)
                            + dimension;
                        let k = (((batch_index * key_tokens + key_token) * kv_heads + kv_head)
                            * head_dim)
                            + dimension;
                        score += query[q] * key[k];
                    }
                    scores.push(score * scale);
                }
                let maximum = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                let denominator: f32 = scores.iter().map(|score| (score - maximum).exp()).sum();
                for dimension in 0..head_dim {
                    let result = scores
                        .iter()
                        .enumerate()
                        .map(|(key_token, score)| {
                            let v = (((batch_index * key_tokens + key_token) * kv_heads + kv_head)
                                * head_dim)
                                + dimension;
                            ((*score - maximum).exp() / denominator) * value[v]
                        })
                        .sum();
                    let output_index = (((batch_index * query_tokens + query_token) * query_heads
                        + query_head)
                        * head_dim)
                        + dimension;
                    output[output_index] = result;
                }
            }
        }
    }
    output
}

#[test]
fn gemm_has_its_own_semantic_api() {
    let ctx = CudaContext::new(0).unwrap();
    let a = tensor(0, vec![2, 3], &[1.0; 6]);
    let b = tensor(0, vec![3, 4], &[1.0; 12]);
    let mut out = tensor(0, vec![2, 4], &[0.0; 8]);
    let mut args = GemmArgs::new(&a, &b, &mut out);
    args.policy.online_tune = false;
    gemm(&ctx, args).unwrap();
    assert!(values(&out).iter().all(|&value| value == 3.0));
}

#[test]
fn gemm_geglu_is_a_separate_semantic_domain() {
    let ctx = CudaContext::new(0).unwrap();
    let (m, k, n) = (3, 5, 1024);
    let a_values = vec![0.25; m * k];
    let b_values: Vec<_> = (0..k * n).map(|i| ((i % 11) as f32 - 5.0) / 16.0).collect();
    let a = tensor(0, vec![m, k], &a_values);
    let b = tensor(0, vec![k, n], &b_values);
    let mut out = tensor(0, vec![m, n / 2], &vec![0.0; m * n / 2]);
    let mut args = GemmArgs::new(&a, &b, &mut out);
    args.policy.online_tune = false;
    gemm_geglu(&ctx, GemmGegluArgs { gemm: args }).unwrap();
    let actual = values(&out);
    assert_eq!(actual.len(), m * n / 2);
    for row in 0..m {
        for column in 0..n / 2 {
            let dot = |target| {
                bf16::from_f32(
                    (0..k)
                        .map(|inner| a_values[row * k + inner] * b_values[inner * n + target])
                        .sum(),
                )
                .to_f32()
            };
            let gate = dot(column);
            let expected = 0.5
                * gate
                * (1.0 + (0.79788456 * (gate + 0.044715 * gate.powi(3))).tanh())
                * dot(column + n / 2);
            assert!((actual[row * n / 2 + column] - expected).abs() < 0.01);
        }
    }
}

#[test]
fn gemm_bias_gelu_has_its_own_semantic_api() {
    let ctx = CudaContext::new(0).unwrap();
    let a = tensor(0, vec![2, 3], &[1.0; 6]);
    let b = tensor(0, vec![3, 4], &[1.0; 12]);
    let bias = tensor(0, vec![4], &[0.5; 4]);
    let mut out = tensor(0, vec![2, 4], &[0.0; 8]);
    let mut args = GemmArgs::new(&a, &b, &mut out);
    args.policy.online_tune = false;
    gemm_bias_gelu(
        &ctx,
        GemmBiasGeluArgs {
            gemm: args,
            bias: &bias,
        },
    )
    .unwrap();
    let output = values(&out);
    let x: f32 = 3.5;
    let expected = 0.5 * x * (1.0 + (0.79788456 * (x + 0.044715 * x.powi(3))).tanh());
    assert!(output.iter().all(|value| (*value - expected).abs() < 0.02));
}

#[test]
fn gemm_bias_has_its_own_semantic_api() {
    let ctx = CudaContext::new(0).unwrap();
    let a = tensor(0, vec![2, 3], &[1.0; 6]);
    let b = tensor(0, vec![3, 4], &[1.0; 12]);
    let bias = tensor(0, vec![4], &[0.5, -0.5, 1.0, -1.0]);
    let mut out = tensor(0, vec![2, 4], &[0.0; 8]);
    let mut args = GemmArgs::new(&a, &b, &mut out);
    args.policy.online_tune = false;
    gemm_bias(
        &ctx,
        GemmBiasArgs {
            gemm: args,
            bias: &bias,
        },
    )
    .unwrap();
    let expected = [3.5, 2.5, 4.0, 2.0, 3.5, 2.5, 4.0, 2.0];
    for (actual, expected) in values(&out).iter().zip(expected) {
        assert!((*actual - expected).abs() < 0.01);
    }
}

#[test]
fn attention_supports_mha_gqa_mqa_and_causal_masking() {
    let ctx = CudaContext::new(0).unwrap();
    let (batch, query_tokens, key_tokens, query_heads, head_dim) = (1, 2, 3, 4, 2);
    let query_values: Vec<_> = (0..batch * query_tokens * query_heads * head_dim)
        .map(|index| (index as f32 - 7.0) / 8.0)
        .collect();
    let scale = 0.5;

    for (kv_heads, causal) in [(4, false), (2, false), (1, false), (2, true)] {
        let key_values: Vec<_> = (0..batch * key_tokens * kv_heads * head_dim)
            .map(|index| ((index * 3 % 11) as f32 - 5.0) / 8.0)
            .collect();
        let value_values: Vec<_> = (0..batch * key_tokens * kv_heads * head_dim)
            .map(|index| ((index * 5 % 13) as f32 - 6.0) / 8.0)
            .collect();
        let query = tensor(
            0,
            vec![batch, query_tokens, query_heads, head_dim],
            &query_values,
        );
        let key = tensor(0, vec![batch, key_tokens, kv_heads, head_dim], &key_values);
        let value = tensor(
            0,
            vec![batch, key_tokens, kv_heads, head_dim],
            &value_values,
        );
        let mut out = tensor(
            0,
            vec![batch, query_tokens, query_heads, head_dim],
            &vec![0.0; query_values.len()],
        );
        let mut args = AttentionArgs::new(&query, &key, &value, &mut out);
        args.scale = scale;
        args.mask = if causal {
            AttentionMask::Causal
        } else {
            AttentionMask::None
        };
        args.policy.online_tune = false;
        attention(&ctx, args).unwrap();

        let expected = attention_reference(
            &query_values,
            &key_values,
            &value_values,
            batch,
            query_tokens,
            key_tokens,
            query_heads,
            kv_heads,
            head_dim,
            scale,
            causal,
        );
        for (index, (actual, expected)) in values(&out).iter().zip(expected).enumerate() {
            assert!(
                (*actual - expected).abs() < 0.02,
                "attention element {index} for kv_heads={kv_heads}, causal={causal} is {actual}, expected {expected}"
            );
        }
    }
}

#[test]
fn attention_prepares_before_capture_and_replays_from_cache() {
    let ctx = CudaContext::new(0).unwrap();
    let query = tensor(0, vec![1, 2, 2, 2], &[0.25; 8]);
    let key = tensor(0, vec![1, 3, 1, 2], &[0.5; 6]);
    let value = tensor(0, vec![1, 3, 1, 2], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let mut out = tensor(0, vec![1, 2, 2, 2], &[0.0; 8]);
    let session = ExecutionSession::with_capacity(4096, 0).unwrap();

    prepare_with_session(&session, || {
        let mut prepare = AttentionArgs::new(&query, &key, &value, &mut out);
        prepare.policy.online_tune = false;
        attention(&ctx, prepare)
    })
    .unwrap();
    let expected = values(&out);

    let graph = crate::capture(&ctx, || {
        with_session(&session, || {
            let mut captured = AttentionArgs::new(&query, &key, &value, &mut out);
            captured.policy.online_tune = false;
            attention(&ctx, captured)
        })
    })
    .unwrap();
    let output = crate::CudaBuffer::from_tensor(&out).unwrap();
    let sentinel: Vec<_> = (0..8)
        .flat_map(|_| bf16::from_f32(-123.0).to_bits().to_ne_bytes())
        .collect();
    output.copy_from_host(&sentinel).unwrap();
    graph.replay().unwrap();
    ctx.synchronize().unwrap();
    assert_eq!(values(&out), expected);
}

#[test]
fn kv_cache_attention_uses_trailing_causal_positions() {
    let ctx = CudaContext::new(0).unwrap();
    let query_values = vec![0.5, -0.25, 0.25, 0.75];
    let key_values = vec![0.25, 0.5, -0.5, 0.25, 0.75, -0.25];
    let value_values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let query = tensor(0, vec![1, 2, 1, 2], &query_values);
    let key = tensor(0, vec![1, 3, 1, 2], &key_values);
    let value = tensor(0, vec![1, 3, 1, 2], &value_values);
    let mut out = tensor(0, vec![1, 2, 1, 2], &[0.0; 4]);
    let mut args = KvCacheAttentionArgs::new(&query, &key, &value, &mut out);
    args.policy.online_tune = false;
    kv_cache_attention(&ctx, args).unwrap();

    let expected = attention_reference(
        &query_values,
        &key_values,
        &value_values,
        1,
        2,
        3,
        1,
        1,
        2,
        1.0 / 2.0_f32.sqrt(),
        true,
    );
    for (actual, expected) in values(&out).iter().zip(expected) {
        assert!((*actual - expected).abs() < 0.02);
    }
}

#[test]
fn segmented_attention_keeps_packed_segments_isolated() {
    let ctx = CudaContext::new(0).unwrap();
    let query_values = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
    let key_values = vec![1.0, 0.0, 0.0, 1.0, 1.0, -1.0];
    let value_values = vec![2.0, 4.0, 6.0, 8.0, 100.0, 200.0];
    let query = tensor(0, vec![3, 1, 2], &query_values);
    let key = tensor(0, vec![3, 1, 2], &key_values);
    let value = tensor(0, vec![3, 1, 2], &value_values);
    let mut out = tensor(0, vec![3, 1, 2], &[0.0; 6]);
    let mut args = SegmentedAttentionArgs::new(&query, &key, &value, &mut out, &[0, 2, 3]);
    args.policy.online_tune = false;
    segmented_attention(&ctx, args).unwrap();

    let mut expected = attention_reference(
        &query_values[..4],
        &key_values[..4],
        &value_values[..4],
        1,
        2,
        2,
        1,
        1,
        2,
        1.0 / 2.0_f32.sqrt(),
        false,
    );
    expected.extend_from_slice(&value_values[4..]);
    for (actual, expected) in values(&out).iter().zip(expected) {
        assert!((*actual - expected).abs() < 0.02);
    }
}
