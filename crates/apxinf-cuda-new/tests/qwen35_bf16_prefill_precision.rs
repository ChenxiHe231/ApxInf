//! Aligned Qwen3.6 BF16 prefill attention-core numerical gate.
//!
//! The GPU executes each complete cold/append query shape. Only the explicitly
//! declared query rows receive independent f64 CPU attention references: this
//! is NOT an independent reference check of every output row. Projection,
//! normalization, RoPE, GDN, MoE, FP8 and complete-model quality are out of scope.
//! Run the ignored test under the global GPU lock with native attention REPORT
//! enabled; the report, not passing numerics, establishes the actual provider.

#[path = "support/bf16_attention_reference.rs"]
mod reference;

use apxinf_core::{DType, Shape, Tensor};
use apxinf_cuda_new::{ops, transfers, CudaBuffer, CudaContext};
use half::bf16;
use reference::{causal_attention, metrics, Geometry};

const QH: usize = 16;
const KVH: usize = 2;
const D: usize = 256;
// Frozen existing BF16 attention absolute ceiling; do not relax for a candidate.
// See qwen35_bf16_attention_precision.rs and the existing scalar contract in
// src/ops/tests/precision/precision.rs. Relative/cosine values are reported,
// without inventing a new relative-error acceptance threshold.
const MAX_ABS: f64 = 0.01;
const CASES: [(usize, usize); 9] = [
    (0, 2048),
    (0, 4096),
    (0, 8192),
    (0, 16384),
    (4096, 4096),
    (8192, 4096),
    (16384, 4096),
    (32768, 4096),
    (49152, 4096),
];

fn selected_rows(queries: usize) -> Vec<usize> {
    let mut rows: Vec<usize> = [0, 63, 64, 127, 128, 2047, 2048, queries - 1]
        .into_iter()
        .filter(|&row| row < queries)
        .collect();
    rows.sort_unstable();
    rows.dedup();
    rows
}

fn assert_finite_nonconstant(label: &str, values: &[f32]) {
    assert!(!values.is_empty(), "{label} is empty");
    assert!(values.iter().all(|value| value.is_finite()), "{label} contains NaN/Inf");
    assert!(
        values.iter().any(|value| value.to_bits() != values[0].to_bits()),
        "{label} unexpectedly constant"
    );
}

fn upload(ctx: &CudaContext, shape: Vec<usize>, values: &[f32]) -> Tensor {
    assert!(values.iter().all(|value| value.is_finite()));
    let prepared: Vec<bf16> = values.iter().map(|&value| bf16::from_f32(value)).collect();
    // The oracle and device consume the same exact prepared BF16 numbers.
    assert!(prepared.iter().zip(values).all(|(value, original)| {
        value.to_f32().to_bits() == original.to_bits()
    }));
    transfers::to_cuda(&Tensor::from_bf16(shape, &prepared).unwrap(), ctx.device_id()).unwrap()
}

fn read(tensor: &Tensor) -> Vec<f32> {
    transfers::to_cpu(tensor)
        .unwrap()
        .as_bf16()
        .unwrap()
        .iter()
        .map(|value| value.to_f32())
        .collect()
}

fn bytes(tensor: &Tensor) -> Vec<u8> {
    let buffer = CudaBuffer::from_tensor(tensor).unwrap();
    let mut result = vec![0; buffer.len()];
    buffer.copy_to_host(&mut result).unwrap();
    result
}

fn byte_digest(values: &[u8]) -> String {
    let mut hash = 0xcbf29ce484222325u64;
    for &byte in values {
        hash = (hash ^ u64::from(byte)).wrapping_mul(0x100000001b3);
    }
    format!("{hash:016x}")
}

fn assert_same_bytes(label: &str, tensor: &Tensor, original: &[u8]) {
    let actual = bytes(tensor);
    assert_eq!(actual.len(), original.len(), "{label}: allocation size changed");
    if let Some(first) = actual.iter().zip(original).position(|(a, b)| a != b) {
        let count = actual.iter().zip(original).filter(|(a, b)| a != b).count();
        panic!("{label}: {count} changed bytes; first at {first}: {} != {}", actual[first], original[first]);
    }
}

fn exact_live_view(cache: &Tensor, valid: usize) -> Tensor {
    CudaBuffer::from_tensor(cache)
        .unwrap()
        .view(0, valid * KVH * D * DType::BF16.size_in_bytes())
        .unwrap()
        .as_tensor(Shape::new(vec![1, valid, KVH, D]), DType::BF16)
        .unwrap()
}

fn execute(ctx: &CudaContext, g: Geometry, q: &Tensor, k: &Tensor, v: &Tensor) -> Tensor {
    assert_eq!(q.shape().dims(), &[1, g.queries, QH, D]);
    let output = CudaBuffer::alloc_zeros_async(g.queries * QH * D * 2, ctx.device_id(), ctx.stream())
        .unwrap()
        .as_tensor(Shape::new(vec![1, g.queries, QH, D]), DType::BF16)
        .unwrap();
    let key = exact_live_view(k, g.valid_keys);
    let value = exact_live_view(v, g.valid_keys);
    let mut out = output.clone();
    let mut args = ops::KvCacheAttentionArgs::new(q, &key, &value, &mut out);
    args.valid_key_tokens = g.valid_keys;
    args.query_start = g.query_start;
    args.scale = 1.0 / (D as f32).sqrt();
    // Numerical tests are separate from strict-recipe timing. Do not tune or
    // write recipes here. Exact live binding makes FA2 eligible; the native
    // APXINF_ATTENTION_REPORT=1 output must independently confirm selection.
    args.policy.online_tune = false;
    args.policy.allow_fallback = true;
    args.policy.cache_dir = Some(String::new());
    ops::kv_cache_attention(ctx, args).unwrap();
    ctx.synchronize().unwrap();
    output
}

fn future_poison(
    ctx: &CudaContext,
    g: Geometry,
    query: &Tensor,
    k: &[f32],
    v: &[f32],
    baseline: &[f32],
) {
    let mut poisoned_k = k.to_vec();
    let mut poisoned_v = v.to_vec();
    // Includes all later query positions as well as physical allocation tail.
    // Query0 must see exactly the same causal prefix as the baseline run.
    let future = (g.query_start + 1) * KVH * D;
    poisoned_k[future..].fill(-6.5);
    poisoned_v[future..].fill(9.5);
    let key = upload(ctx, vec![1, g.capacity, KVH, D], &poisoned_k);
    let value = upload(ctx, vec![1, g.capacity, KVH, D], &poisoned_v);
    let original_k = bytes(&key);
    let original_v = bytes(&value);
    let output = execute(ctx, g, query, &key, &value);
    let first_row = CudaBuffer::from_tensor(&output)
        .unwrap()
        .view(0, QH * D * 2)
        .unwrap()
        .as_tensor(Shape::new(vec![QH, D]), DType::BF16)
        .unwrap();
    let actual = read(&first_row);
    let mismatch = actual.iter().zip(&baseline[..QH * D])
        .position(|(a, b)| a.to_bits() != b.to_bits());
    assert!(mismatch.is_none(), "future keys leaked into query0 at {mismatch:?}");
    assert_same_bytes("poisoned K readonly canary", &key, &original_k);
    assert_same_bytes("poisoned V readonly canary", &value, &original_v);
    println!("bf16-prefill future_poison prefix={} queries={} query0_bitwise_unchanged=true", g.query_start, g.queries);
}

fn case(ctx: &CudaContext, prefix: usize, queries: usize) {
    assert_eq!(queries % 64, 0);
    let valid = prefix + queries;
    // Keep an actual invalid physical page even when valid is page-aligned.
    let capacity = (valid + 17).div_ceil(128) * 128;
    let g = Geometry {
        queries, valid_keys: valid, capacity,
        query_heads: QH, kv_heads: KVH, head_dim: D, query_start: prefix,
    };
    let rows = selected_rows(queries);
    let q: Vec<f32> = (0..queries * QH * D)
        .map(|i| ((i * 13 % 61) as f32 - 30.0) / 32.0)
        .collect();
    let mut k = vec![3.5; capacity * KVH * D];
    let mut v = vec![7.5; k.len()];
    for position in 0..valid {
        for head in 0..KVH {
            for dim in 0..D {
                let i = (position * KVH + head) * D + dim;
                k[i] = (((position * 11 + head * 7 + dim * 3) % 59) as f32 - 29.0) / 64.0;
                // The old periodic V averages close enough to0.5 that valid
                // long-context outputs can all round to the same BF16 number.
                // A position-independent head/dim bias survives softmax and
                // makes nonconstancy meaningful without relaxing precision.
                // Every prepared value remains exactly BF16-representable.
                let bias = head as f32 / 4.0 + (dim % 8) as f32 / 32.0;
                v[i] = (((position * 7 + head * 17 + dim * 5) % 31) as f32 - 15.0)
                    / 32.0 + 0.5 + bias;
            }
        }
    }
    assert_finite_nonconstant("prepared Q", &q);
    assert_finite_nonconstant("prepared K", &k);
    assert_finite_nonconstant("prepared V", &v);
    let query = upload(ctx, vec![1, queries, QH, D], &q);
    let key = upload(ctx, vec![1, capacity, KVH, D], &k);
    let value = upload(ctx, vec![1, capacity, KVH, D], &v);
    let original_q = bytes(&query);
    let original_k = bytes(&key);
    let original_v = bytes(&value);
    println!("bf16-prefill fixture version=head_dim_bias_v2 prefix={prefix} queries={queries} byte_order=little_endian_bf16 storage_scope=full_physical_allocation q_fnv1a64={} k_fnv1a64={} v_fnv1a64={}", byte_digest(&original_q), byte_digest(&original_k), byte_digest(&original_v));
    println!("bf16-prefill begin binding=exact-live prefix={prefix} queries={queries} valid={valid} physical_capacity={capacity} gpu_scope=full_shape oracle_scope=selected_query_rows selected_rows={rows:?}");
    let actual = read(&execute(ctx, g, &query, &key, &value));
    assert_finite_nonconstant("full GPU output", &actual);
    let mut selected_actual = Vec::with_capacity(rows.len() * QH * D);
    let mut selected_reference = Vec::with_capacity(rows.len() * QH * D);
    for &row in &rows {
        let start = row * QH * D;
        let row_geometry = Geometry { queries: 1, query_start: prefix + row, ..g };
        let expected = causal_attention(
            row_geometry, &q[start..start + QH * D], &k, &v, 1.0 / (D as f32).sqrt(),
        ).unwrap();
        for head in 0..QH {
            let head_start = head * D;
            let m = metrics(
                &actual[start + head_start..start + head_start + D],
                &expected[head_start..head_start + D], D,
            ).unwrap();
            println!("bf16-prefill selected_row prefix={prefix} queries={queries} query_row={row} head={head} oracle=f64 checked_elements={} max_abs={:.10} relative_l2={:.10} cosine={:.10}", m.count, m.max_abs, m.relative_l2, m.cosine);
            assert!(m.relative_l2.is_finite() && m.cosine.is_finite());
            assert!(m.max_abs < MAX_ABS, "frozen BF16 ceiling failed prefix={prefix} queries={queries} row={row} head={head}: {m:?}");
        }
        selected_actual.extend_from_slice(&actual[start..start + QH * D]);
        selected_reference.extend_from_slice(&expected);
    }
    let m = metrics(&selected_actual, &selected_reference, D).unwrap();
    assert!(m.max_abs < MAX_ABS && m.max_head_row_abs < MAX_ABS);
    assert!(m.relative_l2.is_finite() && m.cosine.is_finite() && m.max_head_row_relative_l2.is_finite());
    println!("bf16-prefill selected_summary prefix={prefix} queries={queries} independent_reference_rows={} gpu_output_rows={queries} all_output_rows_independently_checked=false Metrics {m:?}", rows.len());
    assert_same_bytes("Q readonly canary", &query, &original_q);
    assert_same_bytes("K prefix/padding readonly canary", &key, &original_k);
    assert_same_bytes("V prefix/padding readonly canary", &value, &original_v);
    if matches!((prefix, queries), (0, 2048) | (4096, 4096)) {
        future_poison(ctx, g, &query, &k, &v, &actual);
        assert_same_bytes("Q readonly after future poison", &query, &original_q);
    }
}

#[test]
#[ignore = "nine full aligned cold/append GPU shapes; independent f64 oracle on declared query rows only; run under global GPU lock"]
fn prepared_bf16_full_prefill_shapes_match_selected_independent_rows() {
    let ctx = CudaContext::new_with_autotune(0, false).unwrap();
    for (prefix, queries) in CASES {
        case(&ctx, prefix, queries);
    }
}
