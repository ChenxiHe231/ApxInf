//! Does the vendored FlashInfer Cake kernel hold at Qwen3.5's head ratio?
//!
//! `qwen38_flashinfer_gdn` validates Cake against our `gdn_chunk_scan` at
//! 16 query / 48 value heads. Qwen3.5 runs 16 / **32** -- group ratio 2 instead
//! of 3 -- and that combination has never been checked. The kernel takes the
//! head counts as runtime arguments and divides generically, so it should hold,
//! but "should" is why this file exists: the production prefill now routes
//! through Cake, and the only thing standing behind it is this comparison.
//!
//! `gdn_chunk_scan` is the oracle here. It is checked against official
//! reference tensors elsewhere, and it is kept for exactly this purpose -- it
//! must not be used in production, where its single launch grows with the
//! sequence.
//!
//! The comparison deliberately reaches Cake the way the model does, through
//! `gdn_prepare_prefill`, rather than normalizing on the host. An earlier
//! version of the sibling test normalized itself and so left our own
//! preparation pass uncovered; that is precisely where a real bug lived (a warp
//! reduction where a block reduction was needed, so the L2 norm summed 32 of
//! 128 elements). Hand the device what the model hands it.

use apxinf_core::{DType, Shape, Tensor};
use apxinf_cuda_new::{ops, CudaBuffer, CudaContext};

const HEAD_DIM: usize = ops::QWEN35_GDN_HEAD_DIM;
const Q_HEADS: usize = ops::QWEN35_GDN_KEY_HEADS;
const V_HEADS: usize = ops::QWEN35_GDN_VALUE_HEADS;
const CHUNK: usize = ops::QWEN35_GDN_SCAN_CHUNK;

fn upload(ctx: &CudaContext, bytes: &[u8], dims: Vec<usize>, dtype: DType) -> Tensor {
    let buffer = CudaBuffer::alloc(bytes.len(), ctx.device_id()).unwrap();
    buffer.copy_from_host(bytes).unwrap();
    buffer.as_tensor(Shape::new(dims), dtype).unwrap()
}

fn zeros(ctx: &CudaContext, dims: Vec<usize>, dtype: DType) -> Tensor {
    CudaBuffer::alloc_zeros(
        dims.iter().product::<usize>() * dtype.size_in_bytes(),
        ctx.device_id(),
    )
    .unwrap()
    .as_tensor(Shape::new(dims), dtype)
    .unwrap()
}

fn bf16_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| half::bf16::from_f32(*value).to_bits().to_le_bytes())
        .collect()
}

fn f16_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| half::f16::from_f32(*value).to_bits().to_le_bytes())
        .collect()
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|value| value.to_le_bytes()).collect()
}

fn read_bf16(tensor: &Tensor, count: usize) -> Vec<f32> {
    apxinf_cuda_new::transfers::to_cpu(tensor)
        .unwrap()
        .as_bf16()
        .unwrap()
        .iter()
        .take(count)
        .map(|value| value.to_f32())
        .collect()
}

fn read_f16(tensor: &Tensor, count: usize) -> Vec<f32> {
    apxinf_cuda_new::transfers::to_cpu(tensor)
        .unwrap()
        .as_f16()
        .unwrap()
        .iter()
        .take(count)
        .map(|value| value.to_f32())
        .collect()
}

fn read_f32(tensor: &Tensor, count: usize) -> Vec<f32> {
    apxinf_cuda_new::transfers::to_cpu(tensor)
        .unwrap()
        .as_f32()
        .unwrap()
        .iter()
        .take(count)
        .copied()
        .collect()
}

/// Cosine and relative L2 in f64, the pair the acceptance bars are written in.
fn cosine_rel_l2(got: &[f32], want: &[f32]) -> (f64, f64) {
    assert_eq!(got.len(), want.len());
    assert!(!got.is_empty());
    assert!(got.iter().chain(want.iter()).all(|value| value.is_finite()));
    let mut dot = 0.0f64;
    let mut ng = 0.0f64;
    let mut nw = 0.0f64;
    let mut err = 0.0f64;
    for (g, w) in got.iter().zip(want.iter()) {
        let (g, w) = (*g as f64, *w as f64);
        dot += g * w;
        ng += g * g;
        nw += w * w;
        err += (g - w) * (g - w);
    }
    assert!(ng > 0.0 && nw > 0.0);
    (dot / (ng.sqrt() * nw.sqrt()), (err / nw).sqrt())
}

fn rms(values: &[f32]) -> f64 {
    (values.iter().map(|x| (*x as f64) * (*x as f64)).sum::<f64>() / values.len() as f64).sqrt()
}

/// Deterministic inputs with q and k L2-normalized per head, so both kernels
/// see the same vectors: ours normalizes again (identity on a unit vector) and
/// Cake requires it done upstream. The unnormalized copies feed the fused row.
#[allow(clippy::type_complexity)]
fn make_inputs(
    seq: usize,
) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut q = vec![0.0f32; seq * Q_HEADS * HEAD_DIM];
    let mut k = vec![0.0f32; seq * Q_HEADS * HEAD_DIM];
    let mut v = vec![0.0f32; seq * V_HEADS * HEAD_DIM];
    for (index, value) in q.iter_mut().enumerate() {
        *value = ((index as f32) * 0.0013).sin();
    }
    for (index, value) in k.iter_mut().enumerate() {
        *value = ((index as f32) * 0.0017).cos();
    }
    for (index, value) in v.iter_mut().enumerate() {
        *value = ((index as f32) * 0.0007).sin() * 0.8;
    }
    let q_raw = q.clone();
    let k_raw = k.clone();
    for row in 0..seq * Q_HEADS {
        for buffer in [&mut q, &mut k] {
            let slice = &mut buffer[row * HEAD_DIM..(row + 1) * HEAD_DIM];
            let norm = slice.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-6);
            for x in slice.iter_mut() {
                *x /= norm;
            }
        }
    }
    // g is the natural-log decay and must stay negative; beta is the gate.
    let g: Vec<f32> = (0..seq * V_HEADS)
        .map(|i| -0.01 - 0.05 * (((i as f32) * 0.011).sin().abs()))
        .collect();
    let beta: Vec<f32> = (0..seq * V_HEADS)
        .map(|i| 0.2 + 0.6 * (((i as f32) * 0.019).cos().abs()))
        .collect();
    (q, k, v, g, beta, q_raw, k_raw)
}

#[test]
#[ignore = "requires a GPU"]
fn cake_matches_chunk_scan_at_qwen35_head_ratio() {
    let ctx = CudaContext::new(0).unwrap();
    assert_eq!(V_HEADS % Q_HEADS, 0, "Cake requires v_heads % q_heads == 0");
    assert_eq!(HEAD_DIM, 128, "the Cake kernel is hardwired to head_dim 128");
    let scale = 1.0f32 / (HEAD_DIM as f32).sqrt();
    let row_width = 2 * Q_HEADS * HEAD_DIM + V_HEADS * HEAD_DIM;
    assert_eq!(row_width, ops::QWEN35_GDN_QKV_WIDTH);

    // 1024 is the length the model crashed at on the DRIVE board, so it is the
    // one that matters most here; the rest bracket it.
    for &(seq, nonzero_initial) in &[
        (CHUNK, false),
        (2 * CHUNK, false),
        (1024, false),
        (2048, false),
        (4096, false),
        (1024, true),
    ] {
        let (q, k, v, g, beta, q_raw, k_raw) = make_inputs(seq);
        let initial_state: Vec<f32> = (0..V_HEADS * HEAD_DIM * HEAD_DIM)
            .map(|index| {
                if nonzero_initial {
                    (index as f32 * 0.013).sin() * 0.05
                } else {
                    0.0
                }
            })
            .collect();

        // --- oracle: our BF16 scan, which normalizes and scales internally.
        let q_bf = upload(&ctx, &bf16_bytes(&q), vec![seq, Q_HEADS, HEAD_DIM], DType::BF16);
        let k_bf = upload(&ctx, &bf16_bytes(&k), vec![seq, Q_HEADS, HEAD_DIM], DType::BF16);
        let v_bf = upload(&ctx, &bf16_bytes(&v), vec![seq, V_HEADS, HEAD_DIM], DType::BF16);
        let g_ours = upload(&ctx, &f32_bytes(&g), vec![seq, V_HEADS], DType::F32);
        let beta_ours = upload(&ctx, &f32_bytes(&beta), vec![seq, V_HEADS], DType::F32);
        let out_ours = zeros(&ctx, vec![seq, V_HEADS, HEAD_DIM], DType::BF16);
        let state_ours = upload(
            &ctx,
            &f32_bytes(&initial_state),
            vec![V_HEADS, HEAD_DIM, HEAD_DIM],
            DType::F32,
        );
        ops::gdn_chunk_scan(
            &ctx, &q_bf, &k_bf, &v_bf, &g_ours, &beta_ours, &out_ours, &state_ours, V_HEADS,
            Q_HEADS, CHUNK,
        )
        .unwrap();

        // --- Cake, reached through the preparation pass the model uses: one
        // interleaved, unnormalized BF16 q|k|v row plus natural-log decay.
        let mut interleaved = vec![0.0f32; seq * row_width];
        for token in 0..seq {
            let row = token * row_width;
            for head in 0..Q_HEADS {
                for d in 0..HEAD_DIM {
                    let source = (token * Q_HEADS + head) * HEAD_DIM + d;
                    interleaved[row + head * HEAD_DIM + d] = q_raw[source];
                    interleaved[row + (Q_HEADS + head) * HEAD_DIM + d] = k_raw[source];
                }
            }
            for index in 0..V_HEADS * HEAD_DIM {
                interleaved[row + 2 * Q_HEADS * HEAD_DIM + index] =
                    v[token * V_HEADS * HEAD_DIM + index];
            }
        }
        let fused = upload(&ctx, &bf16_bytes(&interleaved), vec![seq, row_width], DType::BF16);
        let q_h = zeros(&ctx, vec![seq, Q_HEADS, HEAD_DIM], DType::F16);
        let k_h = zeros(&ctx, vec![seq, Q_HEADS, HEAD_DIM], DType::F16);
        let v_h = zeros(&ctx, vec![seq, V_HEADS, HEAD_DIM], DType::F16);
        let g_fi = zeros(&ctx, vec![seq, V_HEADS], DType::F32);
        let g_log = upload(&ctx, &f32_bytes(&g), vec![seq, V_HEADS], DType::F32);
        ops::gdn_prepare_prefill(
            &ctx, &fused, &q_h, &k_h, &v_h, &g_log, &g_fi, seq, row_width, Q_HEADS, V_HEADS,
            HEAD_DIM, ops::QWEN35_GDN_EPSILON,
        )
        .unwrap();

        // FP16 I/O is the only form this head-generic variant has, so check the
        // activations actually fit before trusting the comparison.
        let v_f16 = read_f16(&v_h, seq * V_HEADS * HEAD_DIM);
        let v_peak = v_f16.iter().fold(0.0f32, |acc, x| acc.max(x.abs()));
        assert!(
            v_peak.is_finite() && v_peak < 65_504.0,
            "seq={seq}: v overflowed FP16 at peak {v_peak}"
        );

        let beta_fi = upload(&ctx, &f32_bytes(&beta), vec![seq, V_HEADS], DType::F32);
        let out_fi = zeros(&ctx, vec![seq, V_HEADS, HEAD_DIM], DType::F16);
        let state_fi = upload(
            &ctx,
            &f32_bytes(&initial_state),
            vec![V_HEADS, HEAD_DIM, HEAD_DIM],
            DType::F32,
        );
        let cu = upload(
            &ctx,
            &[0i32, seq as i32]
                .iter()
                .flat_map(|x| x.to_le_bytes())
                .collect::<Vec<u8>>(),
            vec![2],
            DType::I32,
        );
        let workspace_bytes = ops::gdn_prefill_workspace_bytes(V_HEADS, 1);
        assert!(workspace_bytes > 0, "workspace query returned 0");
        let workspace = zeros(&ctx, vec![workspace_bytes.div_ceil(4)], DType::F32);
        ops::gdn_prefill(
            &ctx, &q_h, &k_h, &v_h, &out_fi, &g_fi, &beta_fi, &cu, &state_fi, &workspace, seq,
            Q_HEADS, V_HEADS, 1, scale,
        )
        .unwrap();
        ctx.synchronize().unwrap();

        let ours_out = read_bf16(&out_ours, seq * V_HEADS * HEAD_DIM);
        let cake_out = read_f16(&out_fi, seq * V_HEADS * HEAD_DIM);
        let (out_cos, out_rel) = cosine_rel_l2(&cake_out, &ours_out);
        let ours_state = read_f32(&state_ours, V_HEADS * HEAD_DIM * HEAD_DIM);
        let cake_state = read_f32(&state_fi, V_HEADS * HEAD_DIM * HEAD_DIM);
        let (state_cos, state_rel) = cosine_rel_l2(&cake_state, &ours_state);
        println!(
            "q{Q_HEADS}/v{V_HEADS} seq={seq:4} nonzero_initial={nonzero_initial} \
out: cosine {out_cos:.6} relL2 {out_rel:.5}  state: cosine {state_cos:.6} relL2 {state_rel:.5}"
        );

        // BF16 carries 8 mantissa bits against FP16's 10 over the same
        // recurrence, so cosine is the bar and relative L2 is reported but
        // allowed to be looser -- it charges the dtype gap twice.
        assert!(
            out_cos >= 0.9999 && out_rel <= 0.01,
            "seq={seq}: Cake output diverged from the oracle, cosine {out_cos}"
        );
        assert!(
            state_cos >= 0.9999 && state_rel <= 0.01,
            "seq={seq}: Cake state diverged from the oracle, cosine {state_cos}"
        );
        // Cosine is blind to scale: the right direction times 1e28 still scores
        // 0.9999, which is a failure the sibling test actually caught.
        let (ours_rms, cake_rms) = (rms(&ours_state), rms(&cake_state));
        let ratio = if ours_rms > 0.0 {
            cake_rms / ours_rms
        } else {
            f64::INFINITY
        };
        assert!(
            ratio > 0.5 && ratio < 2.0,
            "seq={seq}: Cake state magnitude is {ratio:.3e}x the oracle's \
(ours rms {ours_rms:.3e}, theirs {cake_rms:.3e}) -- same direction, wrong scale"
        );
    }
}
