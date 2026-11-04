//! Same-prepared-Q/K/V BF16 attention gate for Qwen3.6's 16/2/256 geometry.
//! This tests the attention core only, not model projection/norm/RoPE or FP8.
//! The independent f64 scalar oracle lives in a dependency-free test module.
//! GPU test timings are never reported as inference benchmark measurements.

#[path = "support/bf16_attention_reference.rs"]
mod reference;

use apxinf_core::{DType, Shape, Tensor};
use apxinf_cuda_new::{ops, transfers, CudaBuffer, CudaContext};
use half::bf16;
use reference::{causal_attention, metrics, Geometry};

const QH: usize = 16;
const KVH: usize = 2;
const D: usize = 256;
// Existing BF16 same-cache scalar contract:
// src/ops/tests/precision/precision.rs::kv_cache_cold_fa2_fallback_matches_reference_without_autotune
// checks every output element with abs error <0.01. Do not tune this ceiling to
// a candidate. Relative/cosine/head-row metrics are retained for frozen-baseline
// comparison; no undocumented relative-error allowance is invented here.
const MAX_ABS: f64 = 0.01;

fn upload(ctx: &CudaContext, shape: Vec<usize>, values: &[f32]) -> Tensor {
    assert!(values.iter().all(|value| value.is_finite()));
    let host: Vec<bf16> = values.iter().map(|&value| bf16::from_f32(value)).collect();
    // Fixtures are exactly representable binary fractions. This asserts that
    // the CPU oracle receives exactly the device BF16 values, not F32 inputs
    // from before a hidden conversion.
    assert!(host.iter().zip(values).all(|(bits, value)| bits.to_f32().to_bits() == value.to_bits()));
    let tensor = Tensor::from_bf16(shape, &host).unwrap();
    transfers::to_cuda(&tensor, ctx.device_id()).unwrap()
}

fn read(tensor: &Tensor) -> Vec<f32> {
    transfers::to_cpu(tensor).unwrap().as_bf16().unwrap().iter().map(|value| value.to_f32()).collect()
}

fn bytes(tensor: &Tensor) -> Vec<u8> {
    let buffer = CudaBuffer::from_tensor(tensor).unwrap();
    let mut result = vec![0u8; buffer.len()];
    buffer.copy_to_host(&mut result).unwrap();
    result
}

fn assert_same_bytes(label: &str, got: &[u8], want: &[u8]) {
    assert_eq!(got.len(), want.len(), "{label} storage size changed");
    let first = got.iter().zip(want).position(|(actual, expected)| actual != expected);
    if let Some(index) = first {
        let mismatches = got.iter().zip(want).filter(|(a, b)| a != b).count();
        panic!("{label}: {mismatches} byte mismatches; first at {index}: device={} original={}", got[index], want[index]);
    }
}

// Same allocation and bytes as the padded cache, with the exact live prefix
// shape used by qwen35moe::attention_backend::cache_prefix_view. This makes
// FA2 eligible; a padded capacity-shaped binding exercises the correctness
// fallback instead. No device copy or preparation arithmetic occurs here.
fn exact_live_view(cache: &Tensor, valid: usize) -> Tensor {
    CudaBuffer::from_tensor(cache).unwrap()
        .view(0, valid * KVH * D * DType::BF16.size_in_bytes()).unwrap()
        .as_tensor(Shape::new(vec![1, valid, KVH, D]), DType::BF16).unwrap()
}

fn execute(ctx: &CudaContext, g: Geometry, q: &Tensor, k: &Tensor, v: &Tensor) -> Tensor {
    let output = CudaBuffer::alloc_zeros_async(g.queries * QH * D * 2, ctx.device_id(), ctx.stream()).unwrap()
        .as_tensor(Shape::new(vec![1, g.queries, QH, D]), DType::BF16).unwrap();
    let mut out = output.clone();
    let mut args = ops::KvCacheAttentionArgs::new(q, k, v, &mut out);
    args.valid_key_tokens = g.valid_keys;
    args.query_start = g.query_start;
    args.scale = 1.0 / (D as f32).sqrt();
    // Precision gate only: exercise the same public static-KV boundary as the
    // model without tuning or altering disk recipes. Native diagnostics must
    // be captured with APXINF_ATTENTION_REPORT=1 to prove which provider ran.
    // Passing numerics alone does not certify that FA2 was selected.
    args.policy.online_tune = false;
    args.policy.allow_fallback = true;
    args.policy.cache_dir = Some(String::new());
    ops::kv_cache_attention(ctx, args).unwrap();
    ctx.synchronize().unwrap();
    output
}

fn case(ctx: &CudaContext, prefix: usize, queries: usize) {
    let valid = prefix + queries;
    let capacity = (valid + 17).div_ceil(128) * 128;
    let g = Geometry { queries, valid_keys: valid, capacity, query_heads: QH, kv_heads: KVH, head_dim: D, query_start: prefix };
    let q: Vec<f32> = (0..queries * QH * D).map(|i| ((i * 13 % 61) as f32 - 30.0) / 32.0).collect();
    let mut k = vec![3.5; capacity * KVH * D];
    let mut v = vec![7.5; k.len()];
    for position in 0..valid {
        for head in 0..KVH {
            for dim in 0..D {
                let i = (position * KVH + head) * D + dim;
                k[i] = (((position * 11 + head * 7 + dim * 3) % 59) as f32 - 29.0) / 64.0;
                v[i] = (((position * 7 + head * 17 + dim * 5) % 31) as f32 - 15.0) / 32.0 + 0.5;
            }
        }
    }
    let expected = causal_attention(g, &q, &k, &v, 1.0 / (D as f32).sqrt()).unwrap();
    let query = upload(ctx, vec![1, queries, QH, D], &q);
    let key = upload(ctx, vec![1, capacity, KVH, D], &k);
    let value = upload(ctx, vec![1, capacity, KVH, D], &v);
    let original_q = bytes(&query);
    let original_k = bytes(&key);
    let original_v = bytes(&value);
    for binding in ["padded-capacity", "exact-live"] {
        let (bound_key, bound_value) = if binding == "exact-live" {
            (exact_live_view(&key, valid), exact_live_view(&value, valid))
        } else {
            (key.clone(), value.clone())
        };
        // Report binding separately from actual provider: the latter is only
        // established by the native APXINF_ATTENTION_REPORT=1 diagnostics.
        println!("bf16-attention begin binding={binding} prefix={prefix} queries={queries} valid={valid} physical_capacity={capacity} bound_capacity={}", bound_key.shape().dims()[1]);
        let output = execute(ctx, g, &query, &bound_key, &bound_value);
        let actual = read(&output);
        let m = metrics(&actual, &expected, D).unwrap();
        println!("bf16-attention binding={binding} prefix={prefix} queries={queries} valid={valid} physical_capacity={capacity} oracle=f64 Metrics {m:?}");
        assert!(m.relative_l2.is_finite() && m.cosine.is_finite() && m.max_head_row_relative_l2.is_finite());
        assert!(m.max_abs < MAX_ABS && m.max_head_row_abs < MAX_ABS,
                "existing BF16 scalar absolute contract failed for {binding}: {m:?}");
        assert_same_bytes("Q was modified", &bytes(&query), &original_q);
        assert_same_bytes("K prefix/tail/canary was modified", &bytes(&key), &original_k);
        assert_same_bytes("V prefix/tail/canary was modified", &bytes(&value), &original_v);

        if prefix == 127 && queries == 4 {
            // Poison query0's causal future, including later query positions,
            // and all physically allocated invalid rows. Both binding modes
            // must leave query0 bitwise unchanged. Exact-live still shares the
            // padded poisoned allocation; its invalid tail is outside shape.
            let mut poisoned_k = k.clone();
            let mut poisoned_v = v.clone();
            for i in (prefix + 1) * KVH * D..k.len() { poisoned_k[i] = -6.5; poisoned_v[i] = 9.5; }
            let poisoned_key = upload(ctx, vec![1, capacity, KVH, D], &poisoned_k);
            let poisoned_value = upload(ctx, vec![1, capacity, KVH, D], &poisoned_v);
            let (poisoned_bound_key, poisoned_bound_value) = if binding == "exact-live" {
                (exact_live_view(&poisoned_key, valid), exact_live_view(&poisoned_value, valid))
            } else {
                (poisoned_key.clone(), poisoned_value.clone())
            };
            let poisoned = read(&execute(ctx, g, &query, &poisoned_bound_key, &poisoned_bound_value));
            let first_row = QH * D;
            let first_difference = actual[..first_row].iter().zip(&poisoned[..first_row])
                .position(|(a, b)| a.to_bits() != b.to_bits());
            assert!(first_difference.is_none(),
                    "{binding}: future keys leaked into query0 at output element {first_difference:?}");
        }
    }
}

#[test]
fn prepared_bf16_short_queries_match_independent_reference_at_page_boundaries() {
    let ctx = CudaContext::new_with_autotune(0, false).unwrap();
    for prefix in [0, 127, 128, 129, 4096] {
        for queries in 1..=4 { case(&ctx, prefix, queries); }
    }
}

#[test]
#[ignore = "full 8K/16K/24K/32K/48K BF16 same-cache numerical gate"]
fn prepared_bf16_short_queries_match_independent_reference_at_long_contexts() {
    let ctx = CudaContext::new_with_autotune(0, false).unwrap();
    for prefix in [8192, 16384, 24576, 32768, 49152] {
        for queries in 1..=4 { case(&ctx, prefix, queries); }
    }
}
