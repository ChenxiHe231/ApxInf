//! Differential-precision harness: legacy `apxinf-cuda` vs `apxinf-cuda-new`.
//!
//! Every migrated model family runs the same host inputs through both crates'
//! identically named kernels and compares the device outputs. The kernels the
//! migration ported bit-for-bit must agree *exactly*; the ones that route
//! through a tuned GEMM/attention provider are allowed a small relative
//! tolerance because different providers sum in a different order.
//!
//! This binary is the only one that can do this: `apxinf-model` is the single
//! package that depends on both CUDA crates (see `cuda_crates_coexist.rs`).
//!
//! Run with a CUDA device present:
//! ```text
//! flock /tmp/apxinf-gpu.lock cargo test -p apxinf-model --features cuda \
//!   --test cuda_new_parity -- --ignored --test-threads=1
//! ```
#![cfg(feature = "cuda")]

use apxinf_core::{DType, Shape, Tensor};
use apxinf_cuda_new::CudaBuffer as NewBuffer;
use apxinf_cuda_new::CudaContext as NewContext;
use apxinf_cuda::{CudaBuffer as LegacyBuffer, CudaContext as LegacyContext};
use half::bf16;

// ── host plumbing ──────────────────────────────────────────────────────────

/// Deterministic signed pseudo-random f32 in roughly [-2, 2), matching the
/// LCG the existing cuda-new tests use, so runs are reproducible across hosts.
struct Lcg(u32);

impl Lcg {
    fn new(seed: u32) -> Self {
        Self(seed)
    }

    fn next(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(1664525).wrapping_add(1013904223);
        bf16::from_f32(((self.0 >> 16) as i32 - 32768) as f32 / 16384.0).to_f32()
    }

    fn values(&mut self, count: usize) -> Vec<f32> {
        (0..count).map(|_| self.next()).collect()
    }

    /// Positive values in (0, 2) for kernels whose contract needs weight-like
    /// or std-like nonnegative data.
    fn positive(&mut self, count: usize) -> Vec<f32> {
        (0..count).map(|_| self.next().abs() + 0.01).collect()
    }
}

fn to_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| bf16::from_f32(*value).to_bits().to_ne_bytes())
        .collect()
}

fn bf16_tensor_new(ctx: &NewContext, dims: Vec<usize>, values: &[f32]) -> Tensor {
    let bytes = to_bytes(values);
    let buffer = NewBuffer::alloc(bytes.len(), ctx.device_id()).unwrap();
    buffer.copy_from_host(&bytes).unwrap();
    buffer.as_tensor(Shape::new(dims), DType::BF16).unwrap()
}

fn bf16_tensor_legacy(ctx: &LegacyContext, dims: Vec<usize>, values: &[f32]) -> Tensor {
    let bytes = to_bytes(values);
    let buffer = LegacyBuffer::alloc(bytes.len(), ctx.device_id()).unwrap();
    buffer.copy_from_host(&bytes).unwrap();
    buffer.as_tensor(Shape::new(dims), DType::BF16).unwrap()
}

fn host_bf16_new(tensor: &Tensor) -> Vec<f32> {
    let buffer = NewBuffer::from_tensor(tensor).unwrap();
    let mut bytes = vec![0_u8; buffer.len()];
    buffer.copy_to_host(&mut bytes).unwrap();
    decode_bf16(&bytes)
}

fn host_bf16_legacy(tensor: &Tensor) -> Vec<f32> {
    let buffer = LegacyBuffer::from_tensor(tensor).unwrap();
    let mut bytes = vec![0_u8; buffer.len()];
    buffer.copy_to_host(&mut bytes).unwrap();
    decode_bf16(&bytes)
}

fn decode_bf16(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|chunk| bf16::from_bits(u16::from_ne_bytes([chunk[0], chunk[1]])).to_f32())
        .collect()
}

/// Raw F8E4M3 bytes of a device tensor, for the quantization comparisons.
fn host_f8_new(tensor: &Tensor) -> Vec<u8> {
    let buffer = NewBuffer::from_tensor(tensor).unwrap();
    let mut bytes = vec![0_u8; buffer.len()];
    buffer.copy_to_host(&mut bytes).unwrap();
    bytes
}

fn host_f8_legacy(tensor: &Tensor) -> Vec<u8> {
    let buffer = LegacyBuffer::from_tensor(tensor).unwrap();
    let mut bytes = vec![0_u8; buffer.len()];
    buffer.copy_to_host(&mut bytes).unwrap();
    bytes
}

fn host_f32_new(tensor: &Tensor) -> Vec<f32> {
    let buffer = NewBuffer::from_tensor(tensor).unwrap();
    let mut bytes = vec![0_u8; buffer.len()];
    buffer.copy_to_host(&mut bytes).unwrap();
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

fn host_f32_legacy(tensor: &Tensor) -> Vec<f32> {
    let buffer = LegacyBuffer::from_tensor(tensor).unwrap();
    let mut bytes = vec![0_u8; buffer.len()];
    buffer.copy_to_host(&mut bytes).unwrap();
    bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_ne_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
        .collect()
}

// ── comparison ─────────────────────────────────────────────────────────────

/// Bit-exact comparison for kernels the migration ported verbatim.
fn assert_bit_exact(label: &str, got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len(), "{label}: length mismatch");
    for (index, (a, b)) in got.iter().zip(want.iter()).enumerate() {
        assert!(
            a.to_bits() == b.to_bits(),
            "{label}: element {index} differs: cuda-new {a} ({:#010x}) vs legacy {b} ({:#010x})",
            a.to_bits(),
            b.to_bits()
        );
    }
    println!("PASS {label}: bit-exact over {} elements", got.len());
}

/// Byte-exact comparison with a compact failure report (raw F8E4M3 values,
/// where dumping whole arrays is unreadable).
fn assert_bytes_exact(label: &str, got: &[u8], want: &[u8]) {
    assert_eq!(got.len(), want.len(), "{label}: length mismatch");
    let diffs: Vec<(usize, u8, u8)> = got
        .iter()
        .zip(want.iter())
        .enumerate()
        .filter(|(_, (a, b))| a != b)
        .map(|(index, (a, b))| (index, *a, *b))
        .collect();
    assert!(
        diffs.is_empty(),
        "{label}: {} of {} bytes differ; first 8: {:?}",
        diffs.len(),
        got.len(),
        &diffs[..diffs.len().min(8)]
    );
    println!("PASS {label}: byte-exact over {} bytes", got.len());
}

/// Tolerance comparison for kernels whose provider may sum in a different
/// order (tuned GEMM, attention).
///
/// Tolerance is relative to the tensor's own scale, `|a - b| <= rel_tol *
/// max|b| + abs_floor`. A per-element relative test is wrong here: two GEMM
/// providers that accumulate in a different order agree to a fixed number of
/// BF16 ULPs of the accumulator, and an output element that cancels to near
/// zero still carries that absolute error. Comparing against the scale of the
/// whole result is the honest statement — "these agree to `rel_tol` of the
/// output magnitude".
fn assert_close(label: &str, got: &[f32], want: &[f32], rel_tol: f64, abs_floor: f64) {
    assert_eq!(got.len(), want.len(), "{label}: length mismatch");
    let scale = want
        .iter()
        .fold(0.0_f64, |acc, value| acc.max(value.abs() as f64));
    let allowed = rel_tol * scale + abs_floor;
    let mut worst_abs = 0.0_f64;
    let mut worst_index = 0_usize;
    let mut dot = 0.0_f64;
    let mut norm_got = 0.0_f64;
    let mut norm_want = 0.0_f64;
    for (index, (a, b)) in got.iter().zip(want.iter()).enumerate() {
        let (a, b) = (*a as f64, *b as f64);
        let diff = (a - b).abs();
        if diff > worst_abs {
            worst_abs = diff;
            worst_index = index;
        }
        dot += a * b;
        norm_got += a * a;
        norm_want += b * b;
    }
    let cosine = if norm_got > 0.0 && norm_want > 0.0 {
        dot / (norm_got.sqrt() * norm_want.sqrt())
    } else {
        1.0
    };
    assert!(
        worst_abs <= allowed && cosine >= 0.9999,
        "{label}: out of tolerance at element {worst_index}: cuda-new {} vs legacy {} \
         (max_abs={worst_abs:.3e}, allowed={allowed:.3e}, scale={scale:.3e}, cosine={cosine:.7})",
        got[worst_index],
        want[worst_index],
    );
    println!(
        "PASS {label}: max_abs={worst_abs:.3e} allowed={allowed:.3e} cosine={cosine:.6}",
    );
}

// ── shared contexts ────────────────────────────────────────────────────────

fn contexts() -> (LegacyContext, NewContext) {
    let legacy = LegacyContext::new(0).expect("legacy context");
    let new = NewContext::new(0).expect("cuda-new context");
    (legacy, new)
}

// ── Workstream A: per-operator parity ───────────────────────────────────────

/// Pure pointwise / norm kernels ported bit-for-bit.
#[test]
#[ignore = "requires a CUDA device"]
fn parity_pointwise_and_norm_bit_exact() {
    let (legacy, new) = contexts();
    let mut rng = Lcg::new(0xA11CE);

    // RMS norm over [64, 256].
    let (rows, cols) = (64_usize, 256_usize);
    let input = rng.values(rows * cols);
    let weight = rng.positive(cols);
    let eps = 1e-6_f32;

    let li = bf16_tensor_legacy(&legacy, vec![rows, cols], &input);
    let ni = bf16_tensor_new(&new, vec![rows, cols], &input);
    let lw = bf16_tensor_legacy(&legacy, vec![cols], &weight);
    let nw = bf16_tensor_new(&new, vec![cols], &weight);

    let l = apxinf_cuda::kernels::norm::rms_bf16(&legacy, &li, &lw, eps).unwrap();
    let n = apxinf_cuda_new::kernels::norm::rms_bf16(&new, &ni, &nw, eps).unwrap();
    legacy.synchronize().unwrap();
    new.synchronize().unwrap();
    assert_bit_exact("norm::rms_bf16", &host_bf16_new(&n), &host_bf16_legacy(&l));

    // Layer norm.
    let bias = rng.values(cols);
    let lb = bf16_tensor_legacy(&legacy, vec![cols], &bias);
    let nb = bf16_tensor_new(&new, vec![cols], &bias);
    let l = apxinf_cuda::kernels::norm::layer_bf16(&legacy, &li, &lw, &lb, eps).unwrap();
    let n = apxinf_cuda_new::kernels::norm::layer_bf16(&new, &ni, &nw, &nb, eps).unwrap();
    new.synchronize().unwrap();
    assert_bit_exact("norm::layer_bf16", &host_bf16_new(&n), &host_bf16_legacy(&l));

    // Silu / tanh-GELU over [128, 512].
    let (r2, c2) = (128_usize, 512_usize);
    let x = rng.values(r2 * c2);
    let lx = bf16_tensor_legacy(&legacy, vec![r2, c2], &x);
    let nx = bf16_tensor_new(&new, vec![r2, c2], &x);

    let l = apxinf_cuda::kernels::activation::silu(&legacy, &lx).unwrap();
    let n = apxinf_cuda_new::kernels::activation::silu(&new, &nx).unwrap();
    new.synchronize().unwrap();
    assert_bit_exact("activation::silu", &host_bf16_new(&n), &host_bf16_legacy(&l));

    let l = apxinf_cuda::kernels::activation::gelu_tanh(&legacy, &lx).unwrap();
    let n = apxinf_cuda_new::kernels::activation::gelu_tanh(&new, &nx).unwrap();
    new.synchronize().unwrap();
    assert_bit_exact("activation::gelu_tanh", &host_bf16_new(&n), &host_bf16_legacy(&l));

    // SwiGLU over a packed [rows, 2*inner] projection.
    let inner = 256_usize;
    let gate_up = rng.values(rows * 2 * inner);
    let lgu = bf16_tensor_legacy(&legacy, vec![rows, 2 * inner], &gate_up);
    let ngu = bf16_tensor_new(&new, vec![rows, 2 * inner], &gate_up);
    let l = apxinf_cuda::kernels::activation::swiglu_bf16(&legacy, &lgu).unwrap();
    let n = apxinf_cuda_new::kernels::activation::swiglu_bf16(&new, &ngu).unwrap();
    new.synchronize().unwrap();
    assert_bit_exact("activation::swiglu_bf16", &host_bf16_new(&n), &host_bf16_legacy(&l));

    // Bias + GELU.
    let bg_bias = rng.values(cols);
    let lbb = bf16_tensor_legacy(&legacy, vec![cols], &bg_bias);
    let nbb = bf16_tensor_new(&new, vec![cols], &bg_bias);
    let l = apxinf_cuda::kernels::activation::bias_gelu_bf16(&legacy, &li, Some(&lbb)).unwrap();
    let n = apxinf_cuda_new::kernels::activation::bias_gelu_bf16(&new, &ni, Some(&nbb)).unwrap();
    new.synchronize().unwrap();
    assert_bit_exact("activation::bias_gelu_bf16", &host_bf16_new(&n), &host_bf16_legacy(&l));

    // elementwise add / scale / bias.
    let a = rng.values(r2 * c2);
    let b = rng.values(r2 * c2);
    let la = bf16_tensor_legacy(&legacy, vec![r2, c2], &a);
    let na = bf16_tensor_new(&new, vec![r2, c2], &a);
    let lb = bf16_tensor_legacy(&legacy, vec![r2, c2], &b);
    let nb = bf16_tensor_new(&new, vec![r2, c2], &b);

    let l = apxinf_cuda::kernels::elementwise::add(&legacy, &la, &lb).unwrap();
    let n = apxinf_cuda_new::kernels::elementwise::add(&new, &na, &nb).unwrap();
    new.synchronize().unwrap();
    assert_bit_exact("elementwise::add", &host_bf16_new(&n), &host_bf16_legacy(&l));

    let l = apxinf_cuda::kernels::elementwise::scale(&legacy, &la, 1.75).unwrap();
    let n = apxinf_cuda_new::kernels::elementwise::scale(&new, &na, 1.75).unwrap();
    new.synchronize().unwrap();
    assert_bit_exact("elementwise::scale", &host_bf16_new(&n), &host_bf16_legacy(&l));

    let eb = rng.values(c2);
    let leb = bf16_tensor_legacy(&legacy, vec![c2], &eb);
    let neb = bf16_tensor_new(&new, vec![c2], &eb);
    let l = apxinf_cuda::kernels::elementwise::bias_bf16(&legacy, &la, Some(&leb)).unwrap();
    let n = apxinf_cuda_new::kernels::elementwise::bias_bf16(&new, &na, Some(&neb)).unwrap();
    new.synchronize().unwrap();
    assert_bit_exact("elementwise::bias_bf16", &host_bf16_new(&n), &host_bf16_legacy(&l));
}

/// Fused residual kernels ported bit-for-bit.
#[test]
#[ignore = "requires a CUDA device"]
fn parity_fused_residual_bit_exact() {
    let (legacy, new) = contexts();
    let mut rng = Lcg::new(0xB0B);
    let (rows, cols) = (96_usize, 384_usize);
    let projection = rng.values(rows * cols);
    let residual = rng.values(rows * cols);
    let bias = rng.values(cols);
    let weight = rng.positive(cols);
    let eps = 1e-5_f32;

    let lp = bf16_tensor_legacy(&legacy, vec![rows, cols], &projection);
    let np = bf16_tensor_new(&new, vec![rows, cols], &projection);
    let lr = bf16_tensor_legacy(&legacy, vec![rows, cols], &residual);
    let nr = bf16_tensor_new(&new, vec![rows, cols], &residual);
    let lb = bf16_tensor_legacy(&legacy, vec![cols], &bias);
    let nb = bf16_tensor_new(&new, vec![cols], &bias);
    let lw = bf16_tensor_legacy(&legacy, vec![cols], &weight);
    let nw = bf16_tensor_new(&new, vec![cols], &weight);

    let l = apxinf_cuda::kernels::fused::bias_residual_bf16(&legacy, &lp, Some(&lb), &lr).unwrap();
    let n = apxinf_cuda_new::kernels::fused::bias_residual_bf16(&new, &np, Some(&nb), &nr).unwrap();
    new.synchronize().unwrap();
    assert_bit_exact("fused::bias_residual_bf16", &host_bf16_new(&n), &host_bf16_legacy(&l));

    let l = apxinf_cuda::kernels::fused::bias_residual_rms_bf16(&legacy, &lp, Some(&lb), &lr, &lw, eps)
        .unwrap();
    let n = apxinf_cuda_new::kernels::fused::bias_residual_rms_bf16(&new, &np, Some(&nb), &nr, &nw, eps)
        .unwrap();
    new.synchronize().unwrap();
    assert_bit_exact(
        "fused::bias_residual_rms_bf16.hidden",
        &host_bf16_new(&n.hidden),
        &host_bf16_legacy(&l.hidden),
    );
    assert_bit_exact(
        "fused::bias_residual_rms_bf16.normalized",
        &host_bf16_new(&n.normalized),
        &host_bf16_legacy(&l.normalized),
    );
}

/// Rowwise E4M3 quantization ported bit-for-bit (scales + raw value bytes).
#[test]
#[ignore = "requires a CUDA device"]
fn parity_quantization_bit_exact() {
    let (legacy, new) = contexts();
    let mut rng = Lcg::new(0x00E4_3A1B);
    let (rows, cols) = (128_usize, 320_usize);
    let input = rng.values(rows * cols);

    let li = bf16_tensor_legacy(&legacy, vec![rows, cols], &input);
    let ni = bf16_tensor_new(&new, vec![rows, cols], &input);

    let l = apxinf_cuda::kernels::quantization::quantize_rows_bf16_e4m3(&legacy, &li).unwrap();
    let n = apxinf_cuda_new::kernels::quantization::quantize_rows_bf16_e4m3(&new, &ni).unwrap();
    new.synchronize().unwrap();

    assert_bytes_exact(
        "quantization::quantize_rows_bf16_e4m3 values",
        &host_f8_new(&n.values),
        &host_f8_legacy(&l.values),
    );
    assert_bit_exact(
        "quantization::quantize_rows_bf16_e4m3 scales",
        &host_f32_new(&n.scales),
        &host_f32_legacy(&l.scales),
    );
}

/// Attention and GEMM run through tuned providers, so they are compared within
/// a relative tolerance rather than bit-for-bit.
#[test]
#[ignore = "requires a CUDA device"]
fn parity_attention_and_gemm_within_tolerance() {
    let (legacy, new) = contexts();
    let mut rng = Lcg::new(0x9E3779B9);

    // Dense MHA over [tokens, heads, head_dim], non-causal.
    let (tokens, heads, head_dim) = (48_usize, 8_usize, 64_usize);
    let q = rng.values(tokens * heads * head_dim);
    let k = rng.values(tokens * heads * head_dim);
    let v = rng.values(tokens * heads * head_dim);
    let lq = bf16_tensor_legacy(&legacy, vec![tokens, heads, head_dim], &q);
    let nq = bf16_tensor_new(&new, vec![tokens, heads, head_dim], &q);
    let lk = bf16_tensor_legacy(&legacy, vec![tokens, heads, head_dim], &k);
    let nk = bf16_tensor_new(&new, vec![tokens, heads, head_dim], &k);
    let lv = bf16_tensor_legacy(&legacy, vec![tokens, heads, head_dim], &v);
    let nv = bf16_tensor_new(&new, vec![tokens, heads, head_dim], &v);

    let l = apxinf_cuda::kernels::attention::mha_bf16(&legacy, &lq, &lk, &lv, tokens).unwrap();
    let n = apxinf_cuda_new::kernels::attention::mha_bf16(&new, &nq, &nk, &nv, tokens).unwrap();
    new.synchronize().unwrap();
    assert_close("attention::mha_bf16", &host_bf16_new(&n), &host_bf16_legacy(&l), 1e-2, 1e-3);

    // GEMM [m,k] @ [k,n] -> [m,n].
    let (m, kk, nn) = (128_usize, 256_usize, 384_usize);
    let x = rng.values(m * kk);
    let w = rng.values(kk * nn);
    let bias = rng.values(nn);
    let lx = bf16_tensor_legacy(&legacy, vec![m, kk], &x);
    let nx = bf16_tensor_new(&new, vec![m, kk], &x);
    let lw = bf16_tensor_legacy(&legacy, vec![kk, nn], &w);
    let nw = bf16_tensor_new(&new, vec![kk, nn], &w);

    let l = apxinf_cuda::kernels::gemm::bf16(&legacy, &lx, &lw).unwrap();
    let n = apxinf_cuda_new::kernels::gemm::bf16(&new, &nx, &nw).unwrap();
    new.synchronize().unwrap();
    assert_close("gemm::bf16", &host_bf16_new(&n), &host_bf16_legacy(&l), 1e-2, 1e-3);

    let lb = bf16_tensor_legacy(&legacy, vec![nn], &bias);
    let nb = bf16_tensor_new(&new, vec![nn], &bias);
    let l = apxinf_cuda::kernels::gemm::bf16_bias(&legacy, &lx, &lw, &lb).unwrap();
    let n = apxinf_cuda_new::kernels::gemm::bf16_bias(&new, &nx, &nw, &nb).unwrap();
    new.synchronize().unwrap();
    assert_close("gemm::bf16_bias", &host_bf16_new(&n), &host_bf16_legacy(&l), 1e-2, 1e-3);
}

/// walloss-only fused-quantization epilogues and preprocess, ported bit-for-bit.
#[test]
#[ignore = "requires a CUDA device"]
fn parity_walloss_fused_and_preprocess_bit_exact() {
    let (legacy, new) = contexts();
    let mut rng = Lcg::new(0x5A11);
    let (rows, cols) = (72_usize, 256_usize);
    let output_cols = 320_usize; // padded
    let input = rng.values(rows * cols);
    let weight = rng.positive(cols);
    let eps = 1e-6_f32;

    let li = bf16_tensor_legacy(&legacy, vec![rows, cols], &input);
    let ni = bf16_tensor_new(&new, vec![rows, cols], &input);
    let lw = bf16_tensor_legacy(&legacy, vec![cols], &weight);
    let nw = bf16_tensor_new(&new, vec![cols], &weight);

    // norm::rms_quantize_rows_bf16_e4m3
    let l = apxinf_cuda::kernels::norm::rms_quantize_rows_bf16_e4m3(
        &legacy, &li, &lw, eps, output_cols,
    )
    .unwrap();
    let n = apxinf_cuda_new::kernels::norm::rms_quantize_rows_bf16_e4m3(
        &new, &ni, &nw, eps, output_cols,
    )
    .unwrap();
    new.synchronize().unwrap();
    assert_bytes_exact(
        "norm::rms_quantize_rows_bf16_e4m3 values",
        &host_f8_new(&n.values),
        &host_f8_legacy(&l.values),
    );
    assert_bit_exact(
        "norm::rms_quantize_rows_bf16_e4m3 scales",
        &host_f32_new(&n.scales),
        &host_f32_legacy(&l.scales),
    );

    // fused::bias_residual_rms_quantize_rows_bf16_e4m3
    let projection = rng.values(rows * cols);
    let residual = rng.values(rows * cols);
    let bias = rng.values(cols);
    let lp = bf16_tensor_legacy(&legacy, vec![rows, cols], &projection);
    let np = bf16_tensor_new(&new, vec![rows, cols], &projection);
    let lr = bf16_tensor_legacy(&legacy, vec![rows, cols], &residual);
    let nr = bf16_tensor_new(&new, vec![rows, cols], &residual);
    let lb = bf16_tensor_legacy(&legacy, vec![cols], &bias);
    let nb = bf16_tensor_new(&new, vec![cols], &bias);

    let l = apxinf_cuda::kernels::fused::bias_residual_rms_quantize_rows_bf16_e4m3(
        &legacy, &lp, Some(&lb), &lr, &lw, eps, output_cols,
    )
    .unwrap();
    let n = apxinf_cuda_new::kernels::fused::bias_residual_rms_quantize_rows_bf16_e4m3(
        &new, &np, Some(&nb), &nr, &nw, eps, output_cols,
    )
    .unwrap();
    new.synchronize().unwrap();
    assert_bit_exact(
        "fused::bias_residual_rms_quantize_rows_bf16_e4m3.hidden",
        &host_bf16_new(&n.hidden),
        &host_bf16_legacy(&l.hidden),
    );
    assert_bytes_exact(
        "fused::bias_residual_rms_quantize_rows_bf16_e4m3 normalized values",
        &host_f8_new(&n.normalized.values),
        &host_f8_legacy(&l.normalized.values),
    );
    assert_bit_exact(
        "fused::bias_residual_rms_quantize_rows_bf16_e4m3 normalized scales",
        &host_f32_new(&n.normalized.scales),
        &host_f32_legacy(&l.normalized.scales),
    );

    // activation::swiglu_quantize_rows_bf16_e4m3
    let inner = 128_usize;
    let gate_up = rng.values(rows * 2 * inner);
    let lgu = bf16_tensor_legacy(&legacy, vec![rows, 2 * inner], &gate_up);
    let ngu = bf16_tensor_new(&new, vec![rows, 2 * inner], &gate_up);
    let l = apxinf_cuda::kernels::activation::swiglu_quantize_rows_bf16_e4m3(
        &legacy, &lgu, None, inner, output_cols,
    )
    .unwrap();
    let n = apxinf_cuda_new::kernels::activation::swiglu_quantize_rows_bf16_e4m3(
        &new, &ngu, None, inner, output_cols,
    )
    .unwrap();
    new.synchronize().unwrap();
    assert_bytes_exact(
        "activation::swiglu_quantize_rows_bf16_e4m3 values",
        &host_f8_new(&n.values),
        &host_f8_legacy(&l.values),
    );
    assert_bit_exact(
        "activation::swiglu_quantize_rows_bf16_e4m3 scales",
        &host_f32_new(&n.scales),
        &host_f32_legacy(&l.scales),
    );

    // preprocess::rgb_u8_to_normalized_temporal_merged_patches_bf16
    let (views, image_size, patch_size, temporal, merge) = (2_usize, 28, 14, 2, 2);
    let bytes = views * 3 * image_size * image_size;
    let rgb: Vec<u8> = (0..bytes).map(|index| (index * 7 % 251) as u8).collect();
    let patch_rows = views * (image_size / patch_size).pow(2);
    let patch_width = 3 * temporal * patch_size * patch_size;
    let lrgb = LegacyBuffer::alloc(bytes, legacy.device_id()).unwrap();
    lrgb.copy_from_host(&rgb).unwrap();
    let nrgb = NewBuffer::alloc(bytes, new.device_id()).unwrap();
    nrgb.copy_from_host(&rgb).unwrap();
    let lpatches = bf16_tensor_legacy(&legacy, vec![patch_rows, patch_width], &vec![0.0; patch_rows * patch_width]);
    let npatches = bf16_tensor_new(&new, vec![patch_rows, patch_width], &vec![0.0; patch_rows * patch_width]);

    apxinf_cuda::kernels::preprocess::rgb_u8_to_normalized_temporal_merged_patches_bf16(
        &legacy,
        &lrgb,
        &lpatches,
        views,
        image_size,
        patch_size,
        temporal,
        merge,
        apxinf_cuda::kernels::preprocess::ImageLayout::Nhwc,
        1.0 / 255.0,
        [0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5],
    )
    .unwrap();
    apxinf_cuda_new::kernels::preprocess::rgb_u8_to_normalized_temporal_merged_patches_bf16(
        &new,
        &nrgb,
        &npatches,
        views,
        image_size,
        patch_size,
        temporal,
        merge,
        apxinf_cuda_new::kernels::preprocess::ImageLayout::Nhwc,
        1.0 / 255.0,
        [0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5],
    )
    .unwrap();
    new.synchronize().unwrap();
    assert_bit_exact(
        "preprocess::rgb_u8_to_normalized_temporal_merged_patches_bf16",
        &host_bf16_new(&npatches),
        &host_bf16_legacy(&lpatches),
    );
}
