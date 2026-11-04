//! Exact-geometry tests for the Qwen3.5-35B-A3B GDN mixer adapter.
//!
//! These tests cover two separate claims:
//! 1. one decode step agrees with an independent CPU implementation of the
//!    Hugging Face recurrence and state update; and
//! 2. the 64-token scan leaves state that the one-token decode path can use
//!    without conversion or loss of causal-convolution history.

use apxinf_core::{DType, Shape, Tensor};
use apxinf_cuda_new::{ops, CudaBuffer, CudaContext};

const KH: usize = ops::QWEN35_GDN_KEY_HEADS;
const VH: usize = ops::QWEN35_GDN_VALUE_HEADS;
const D: usize = ops::QWEN35_GDN_HEAD_DIM;
const QW: usize = ops::QWEN35_GDN_Q_WIDTH;
const VW: usize = ops::QWEN35_GDN_VALUE_WIDTH;
const QKV: usize = ops::QWEN35_GDN_QKV_WIDTH;
const CW: usize = ops::QWEN35_GDN_CONV_WIDTH;

fn rb(value: f32) -> f32 {
    half::bf16::from_f32(value).to_f32()
}

fn upload_bf16(ctx: &CudaContext, values: &[f32], shape: Vec<usize>) -> Tensor {
    let bytes: Vec<u8> = values
        .iter()
        .flat_map(|value| half::bf16::from_f32(*value).to_bits().to_le_bytes())
        .collect();
    let buffer = CudaBuffer::alloc(bytes.len(), ctx.device_id()).unwrap();
    buffer.copy_from_host(&bytes).unwrap();
    buffer.as_tensor(Shape::new(shape), DType::BF16).unwrap()
}

fn upload_f32(ctx: &CudaContext, values: &[f32], shape: Vec<usize>) -> Tensor {
    let bytes: Vec<u8> = values.iter().flat_map(|value| value.to_le_bytes()).collect();
    let buffer = CudaBuffer::alloc(bytes.len(), ctx.device_id()).unwrap();
    buffer.copy_from_host(&bytes).unwrap();
    buffer.as_tensor(Shape::new(shape), DType::F32).unwrap()
}

fn zeros(ctx: &CudaContext, shape: Vec<usize>, dtype: DType) -> Tensor {
    let bytes = shape.iter().product::<usize>() * dtype.size_in_bytes();
    // Initialization must precede consumers on the nonblocking context stream.
    // Plain alloc_zeros uses legacy-default-stream cudaMemset, which has no
    // ordering dependency with this stream and can erase computed Q/K/V later.
    let buffer = CudaBuffer::alloc_zeros_async(bytes, ctx.device_id(), ctx.stream()).unwrap();
    buffer.as_tensor(Shape::new(shape), dtype).unwrap()
}

fn read_bf16(tensor: &Tensor) -> Vec<f32> {
    let buffer = CudaBuffer::from_tensor(tensor).unwrap();
    let mut bytes = vec![0u8; buffer.len()];
    buffer.copy_to_host(&mut bytes).unwrap();
    bytes
        .chunks_exact(2)
        .map(|raw| half::bf16::from_bits(u16::from_le_bytes([raw[0], raw[1]])).to_f32())
        .collect()
}

fn read_f32(tensor: &Tensor) -> Vec<f32> {
    let buffer = CudaBuffer::from_tensor(tensor).unwrap();
    let mut bytes = vec![0u8; buffer.len()];
    buffer.copy_to_host(&mut bytes).unwrap();
    bytes
        .chunks_exact(4)
        .map(|raw| f32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]))
        .collect()
}

fn view(tensor: &Tensor, offset: usize, shape: Vec<usize>, dtype: DType) -> Tensor {
    let element = dtype.size_in_bytes();
    let bytes = shape.iter().product::<usize>() * element;
    CudaBuffer::from_tensor(tensor)
        .unwrap()
        .view(offset * element, bytes)
        .unwrap()
        .as_tensor(Shape::new(shape), dtype)
        .unwrap()
}

fn cosine_rel_l2(got: &[f32], want: &[f32]) -> (f64, f64) {
    assert_eq!(got.len(), want.len());
    let mut dot = 0.0f64;
    let mut got_norm = 0.0f64;
    let mut want_norm = 0.0f64;
    let mut diff = 0.0f64;
    for (&got, &want) in got.iter().zip(want) {
        let (got, want) = (got as f64, want as f64);
        dot += got * want;
        got_norm += got * got;
        want_norm += want * want;
        diff += (got - want) * (got - want);
    }
    (
        dot / (got_norm.sqrt() * want_norm.sqrt()).max(1.0e-30),
        (diff / want_norm.max(1.0e-30)).sqrt(),
    )
}

fn bitwise_mismatches(got: &[f32], want: &[f32]) -> usize {
    assert_eq!(got.len(), want.len());
    got.iter()
        .zip(want)
        .filter(|(got, want)| got.to_bits() != want.to_bits())
        .count()
}

fn report_stage_difference(name: &str, got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len());
    let (cosine, relative) = cosine_rel_l2(got, want);
    let finite = got.iter().chain(want).all(|value| value.is_finite());
    let max_abs = got.iter().zip(want).map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    let first: Vec<_> = got.iter().zip(want).take(8).map(|(&a, &b)| (a, b)).collect();
    let mismatches: Vec<_> = got.iter().zip(want).enumerate()
        .filter(|(_, (a, b))| a.to_bits() != b.to_bits()).take(8)
        .map(|(index, (&a, &b))| (index, a, b)).collect();
    println!("gdn-stage {name}: finite={finite} cosine={cosine:.9} relL2={relative:.9} max_abs={max_abs:.9} first_actual_expected={first:?} first_bit_differences={mismatches:?}");
}

fn report_qkv_stage(name: &str, got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), QKV);
    assert_eq!(want.len(), QKV);
    for (segment, range) in [("Q", 0..QW), ("K", QW..2 * QW), ("V", 2 * QW..QKV)] {
        report_stage_difference(&format!("{name}/{segment}"), &got[range.clone()], &want[range.clone()]);
        let got_norms: Vec<_> = got[range.clone()].chunks_exact(D)
            .map(|head| head.iter().map(|value| (*value as f64).powi(2)).sum::<f64>().sqrt()).collect();
        let want_norms: Vec<_> = want[range].chunks_exact(D)
            .map(|head| head.iter().map(|value| (*value as f64).powi(2)).sum::<f64>().sqrt()).collect();
        println!("gdn-stage {name}/{segment}: per_head_l2_actual={got_norms:?} per_head_l2_expected={want_norms:?}");
    }
}

fn assert_long_parity(
    name: &str,
    got: &[f32],
    want: &[f32],
    min_cosine: f64,
    max_relative_l2: f64,
    require_bit_exact: bool,
) {
    assert!(got.iter().chain(want).all(|value| value.is_finite()),
            "{name} contains a non-finite value");
    let mismatches = bitwise_mismatches(got, want);
    let max_abs = got
        .iter()
        .zip(want)
        .map(|(got, want)| (got - want).abs())
        .fold(0.0f32, f32::max);
    let (cosine, relative) = cosine_rel_l2(got, want);
    println!(
        "{name}: bitwise_mismatches={mismatches}/{} max_abs={max_abs:.8} cosine={cosine:.9} relL2={relative:.9}",
        got.len()
    );
    if require_bit_exact {
        assert_eq!(mismatches, 0, "{name} is not bit-exact");
    } else {
        assert!(
            cosine >= min_cosine && relative <= max_relative_l2,
            "{name} mismatch: cosine={cosine} relL2={relative}"
        );
    }
}

struct HostDecode<'a> {
    projected_qkv: &'a [f32],
    z: &'a [f32],
    a: &'a [f32],
    b: &'a [f32],
    conv_weight: &'a [f32],
    a_log: &'a [f32],
    dt_bias: &'a [f32],
    norm_weight: &'a [f32],
    conv_state: &'a mut [f32],
    recurrent_state: &'a mut [f32],
}

/// Independent scalar form of Qwen3.5's decode mixer.  Inputs and weights are
/// rounded exactly where the checkpoint/device path has a BF16 boundary.
fn cpu_decode(args: HostDecode<'_>) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut convolved = vec![0.0f32; QKV];
    for channel in 0..QKV {
        let window = &mut args.conv_state[channel * CW..(channel + 1) * CW];
        window.rotate_left(1);
        window[CW - 1] = rb(args.projected_qkv[channel]);
        let mut sum = 0.0f32;
        for tap in 0..CW {
            sum += window[tap] * rb(args.conv_weight[channel * CW + tap]);
        }
        convolved[channel] = rb(sum / (1.0 + (-sum).exp()));
    }

    for offset in [0, QW] {
        for head in 0..KH {
            let row = offset + head * D;
            let sum: f32 = convolved[row..row + D]
                .iter()
                .map(|value| value * value)
                .sum();
            let inverse = 1.0 / (sum + ops::QWEN35_GDN_EPSILON).sqrt();
            for value in &mut convolved[row..row + D] {
                *value = rb(*value * inverse);
            }
        }
    }

    let mut decay = vec![0.0f32; VH];
    let mut beta = vec![0.0f32; VH];
    for head in 0..VH {
        let shifted = rb(args.a[head]) + rb(args.dt_bias[head]);
        let softplus = if shifted > 20.0 {
            shifted
        } else {
            (1.0 + shifted.exp()).ln()
        };
        decay[head] = -rb(args.a_log[head]).exp() * softplus;
        beta[head] = 1.0 / (1.0 + (-rb(args.b[head])).exp());
    }

    let mut core = vec![0.0f32; VW];
    for head in 0..VH {
        let key_head = head / (VH / KH);
        let q = &convolved[key_head * D..(key_head + 1) * D];
        let k = &convolved[QW + key_head * D..QW + (key_head + 1) * D];
        let v = &convolved[2 * QW + head * D..2 * QW + (head + 1) * D];
        let head_decay = decay[head].exp();
        for value_dim in 0..D {
            let state_row = (head * D + value_dim) * D;
            let state = &mut args.recurrent_state[state_row..state_row + D];
            for value in state.iter_mut() {
                *value *= head_decay;
            }
            let predicted: f32 = state.iter().zip(k).map(|(s, k)| s * k).sum();
            let delta = (v[value_dim] - predicted) * beta[head];
            for (value, key) in state.iter_mut().zip(k) {
                *value += delta * key;
            }
            let output: f32 = state.iter().zip(q).map(|(s, q)| s * q).sum();
            core[head * D + value_dim] = rb(output / (D as f32).sqrt());
        }
    }

    let mut gated = vec![0.0f32; VW];
    for head in 0..VH {
        let row = head * D;
        let variance = core[row..row + D]
            .iter()
            .map(|value| value * value)
            .sum::<f32>()
            / D as f32;
        let inverse = 1.0 / (variance + ops::QWEN35_GDN_EPSILON).sqrt();
        for index in 0..D {
            let normalized = rb(core[row + index] * inverse) * rb(args.norm_weight[index]);
            let gate = rb(args.z[row + index]);
            gated[row + index] = rb(normalized * gate / (1.0 + (-gate).exp()));
        }
    }
    (convolved, decay, beta, gated)
}

/// CPU-only analytic sanity check of the independent reference, not a CUDA
/// correctness claim. With zero q/k/v, a=0, A_log=0, dt_bias=0, the nonzero
/// recurrent state must halve each step and readout remains exactly zero.
#[test]
fn cpu_reference_zero_readout_has_analytic_multistep_decay() {
    let projected = vec![0.0; QKV];
    let z = vec![0.5; VW];
    let gates = vec![0.0; VH];
    let weights = vec![0.0; QKV * CW];
    let norm = vec![1.0; D];
    let mut conv_state = vec![0.0; QKV * CW];
    let initial = 0.125f32;
    let mut recurrent_state = vec![initial; VH * D * D];
    for step in 1..=4 {
        let (convolved, decay, beta, gated) = cpu_decode(HostDecode {
            projected_qkv: &projected, z: &z, a: &gates, b: &gates,
            conv_weight: &weights, a_log: &gates, dt_bias: &gates,
            norm_weight: &norm, conv_state: &mut conv_state,
            recurrent_state: &mut recurrent_state,
        });
        assert!(convolved.iter().chain(&gated).all(|value| *value == 0.0));
        assert!(conv_state.iter().all(|value| *value == 0.0));
        assert!(decay.iter().all(|value| (*value + std::f32::consts::LN_2).abs() < 2.0e-5));
        assert!(beta.iter().all(|value| *value == 0.5));
        let expected = initial / (1u32 << step) as f32;
        assert!(recurrent_state.iter().all(|value| value.is_finite() && (*value - expected).abs() < 2.0e-7));
    }
}

struct DeviceFixture {
    projected_qkv: Tensor,
    z: Tensor,
    a: Tensor,
    b: Tensor,
    conv_weight: Tensor,
    a_log: Tensor,
    dt_bias: Tensor,
    norm_weight: Tensor,
    conv_state: Tensor,
    recurrent_state: Tensor,
    convolved: Tensor,
    decay: Tensor,
    beta: Tensor,
    core: Tensor,
    gated: Tensor,
    /// FP16 Cake buffers, allocated only where that kernel exists so this
    /// fixture drives the same path production does.
    cake: Option<CakeFixture>,
}

struct CakeFixture {
    q16: Tensor,
    k16: Tensor,
    v16: Tensor,
    out16: Tensor,
    alpha: Tensor,
    cu_seqlens: Tensor,
    workspace: Tensor,
}

impl CakeFixture {
    fn scratch(&self) -> ops::Qwen35GdnCakeScratch<'_> {
        ops::Qwen35GdnCakeScratch {
            q16: &self.q16,
            k16: &self.k16,
            v16: &self.v16,
            out16: &self.out16,
            alpha: &self.alpha,
            cu_seqlens: &self.cu_seqlens,
            workspace: &self.workspace,
        }
    }
}

impl DeviceFixture {
    fn cake_scratch(&self) -> Option<ops::Qwen35GdnCakeScratch<'_>> {
        self.cake.as_ref().map(CakeFixture::scratch)
    }

    /// Prefix views for a prefill shorter than the fixture's profile, so a
    /// fixture built for `TOKENS` can drive a `SCANNED_TOKENS` prefill.
    fn cake_views(&self, tokens: usize) -> Option<CakeFixture> {
        let cake = self.cake.as_ref()?;
        let key = vec![tokens, KH, D];
        let value = vec![tokens, VH, D];
        Some(CakeFixture {
            q16: view(&cake.q16, 0, key.clone(), DType::F16),
            k16: view(&cake.k16, 0, key, DType::F16),
            v16: view(&cake.v16, 0, value.clone(), DType::F16),
            out16: view(&cake.out16, 0, value, DType::F16),
            alpha: view(&cake.alpha, 0, vec![tokens, VH], DType::F32),
            cu_seqlens: view(&cake.cu_seqlens, 0, vec![2], DType::I32),
            workspace: view(
                &cake.workspace,
                0,
                cake.workspace.shape().dims().to_vec(),
                DType::F32,
            ),
        })
    }

    fn decode<'a>(&'a self) -> ops::Qwen35GdnDecodeArgs<'a> {
        ops::Qwen35GdnDecodeArgs {
            projected_qkv: &self.projected_qkv,
            z: &self.z,
            a: &self.a,
            b: &self.b,
            weights: ops::Qwen35GdnCoreWeights {
                conv: &self.conv_weight,
                a_log: &self.a_log,
                dt_bias: &self.dt_bias,
                norm: &self.norm_weight,
            },
            state: ops::Qwen35GdnState {
                conv: &self.conv_state,
                recurrent: &self.recurrent_state,
            },
            scratch: ops::Qwen35GdnDecodeScratch {
                convolved_qkv: &self.convolved,
                decay: &self.decay,
                beta: &self.beta,
                core_output: &self.core,
                gated_output: &self.gated,
            },
        }
    }
}

#[test]
fn decode_matches_independent_cpu_reference_at_qwen35_geometry() {
    let ctx = CudaContext::new(0).unwrap();
    let projected: Vec<f32> = (0..QKV)
        .map(|index| ((index * 17 % 53) as f32 - 26.0) / 41.0)
        .collect();
    let z: Vec<f32> = (0..VW)
        .map(|index| ((index * 7 % 29) as f32 - 14.0) / 13.0)
        .collect();
    let a: Vec<f32> = (0..VH).map(|head| head as f32 / 31.0 - 0.5).collect();
    let b: Vec<f32> = (0..VH).map(|head| 0.7 - head as f32 / 37.0).collect();
    let conv_weight: Vec<f32> = (0..QKV * CW)
        .map(|index| ((index * 5 % 17) as f32 - 8.0) / 23.0)
        .collect();
    let a_log: Vec<f32> = (0..VH).map(|head| -1.4 + head as f32 / 53.0).collect();
    let dt_bias: Vec<f32> = (0..VH).map(|head| -0.3 + head as f32 / 47.0).collect();
    let norm_weight: Vec<f32> = (0..D).map(|index| 0.8 + index as f32 / 311.0).collect();
    let initial_conv: Vec<f32> = (0..QKV * CW)
        .map(|index| ((index * 3 % 19) as f32 - 9.0) * 1.0e-3)
        .collect();
    let initial_recurrent: Vec<f32> = (0..VH * D * D)
        .map(|index| ((index * 11 % 31) as f32 - 15.0) * 1.0e-5)
        .collect();

    let fixture = DeviceFixture {
        projected_qkv: upload_bf16(&ctx, &projected, vec![QKV]),
        z: upload_bf16(&ctx, &z, vec![VW]),
        a: upload_bf16(&ctx, &a, vec![VH]),
        b: upload_bf16(&ctx, &b, vec![VH]),
        conv_weight: upload_bf16(&ctx, &conv_weight, vec![QKV, 1, CW]),
        a_log: upload_bf16(&ctx, &a_log, vec![VH]),
        dt_bias: upload_bf16(&ctx, &dt_bias, vec![VH]),
        norm_weight: upload_bf16(&ctx, &norm_weight, vec![D]),
        conv_state: upload_f32(&ctx, &initial_conv, vec![QKV, CW]),
        recurrent_state: upload_f32(&ctx, &initial_recurrent, vec![VH, D, D]),
        convolved: zeros(&ctx, vec![QKV], DType::BF16),
        decay: zeros(&ctx, vec![VH], DType::F32),
        beta: zeros(&ctx, vec![VH], DType::F32),
        core: zeros(&ctx, vec![VW], DType::BF16),
        gated: zeros(&ctx, vec![VW], DType::BF16),
        // Decode-only fixture: the Cake buffers are a prefill concern.
        cake: None,
    };
    ops::qwen35_gdn_decode(&ctx, &fixture.decode()).unwrap();
    ctx.synchronize().unwrap();

    let mut expected_conv_state = initial_conv;
    let mut expected_recurrent = initial_recurrent;
    let (expected_convolved, expected_decay, expected_beta, expected_gated) = cpu_decode(HostDecode {
        projected_qkv: &projected,
        z: &z,
        a: &a,
        b: &b,
        conv_weight: &conv_weight,
        a_log: &a_log,
        dt_bias: &dt_bias,
        norm_weight: &norm_weight,
        conv_state: &mut expected_conv_state,
        recurrent_state: &mut expected_recurrent,
    });

    assert_eq!(read_f32(&fixture.conv_state), expected_conv_state);
    // Keep the original full-vector assertions below. This diagnostic splits
    // the normalized Q/K regions from unnormalized V before a failure aborts
    // the test; it does not substitute a weaker per-segment gate.
    report_qkv_stage("full-decode/post-QKnorm", &read_bf16(&fixture.convolved), &expected_convolved);
    for (name, got, want, min_cos, max_rel) in [
        ("convolved", read_bf16(&fixture.convolved), expected_convolved, 0.99999, 0.01),
        ("recurrent", read_f32(&fixture.recurrent_state), expected_recurrent, 0.99999, 0.01),
        ("gated", read_bf16(&fixture.gated), expected_gated, 0.9999, 0.025),
    ] {
        let (cosine, relative) = cosine_rel_l2(&got, &want);
        println!("{name}: cosine={cosine:.7} relL2={relative:.7}");
        assert!(cosine >= min_cos && relative <= max_rel, "{name} mismatch");
    }
    let max_decay = read_f32(&fixture.decay)
        .iter()
        .zip(&expected_decay)
        .map(|(got, want)| (got - want).abs())
        .fold(0.0f32, f32::max);
    let max_beta = read_f32(&fixture.beta)
        .iter()
        .zip(&expected_beta)
        .map(|(got, want)| (got - want).abs())
        .fold(0.0f32, f32::max);
    assert!(max_decay < 2.0e-5, "decay max abs {max_decay}");
    assert!(max_beta < 2.0e-5, "beta max abs {max_beta}");
}

/// Isolate the failing preprocessing stages from recurrence/output gating.
/// Same deterministic fixture as the complete independent decode test, but
/// only public raw-conv and in-place Q/K-normalization operators are called.
/// CPU raw-conv uses the *read-back prepared BF16* input/weight values. The
/// existing independent CPU reference provides the post-QKnorm expected row.
#[test]
fn decode_preprocessing_stages_match_independent_cpu_reference() {
    let ctx = CudaContext::new_with_autotune(0, false).unwrap();
    let projected: Vec<f32> = (0..QKV)
        .map(|index| ((index * 17 % 53) as f32 - 26.0) / 41.0).collect();
    let conv_weight: Vec<f32> = (0..QKV * CW)
        .map(|index| ((index * 5 % 17) as f32 - 8.0) / 23.0).collect();
    let initial_conv: Vec<f32> = (0..QKV * CW)
        .map(|index| ((index * 3 % 19) as f32 - 9.0) * 1.0e-3).collect();
    let projected_device = upload_bf16(&ctx, &projected, vec![QKV]);
    let weight_device = upload_bf16(&ctx, &conv_weight, vec![QKV, CW]);
    let state_device = upload_f32(&ctx, &initial_conv, vec![QKV, CW]);
    let output = zeros(&ctx, vec![QKV], DType::BF16);
    let prepared_projected = read_bf16(&projected_device);
    let prepared_weight = read_bf16(&weight_device);
    assert_eq!(prepared_projected, projected.iter().map(|&value| rb(value)).collect::<Vec<_>>());
    assert_eq!(prepared_weight, conv_weight.iter().map(|&value| rb(value)).collect::<Vec<_>>());

    // Independent raw convolution oracle; preserve all F32 conv-history
    // values and BF16 representation boundaries of the existing CPU math.
    let mut expected_conv = initial_conv.clone();
    let mut expected_raw = vec![0.0f32; QKV];
    for channel in 0..QKV {
        let history = &mut expected_conv[channel * CW..(channel + 1) * CW];
        history.rotate_left(1);
        history[CW - 1] = prepared_projected[channel];
        let mut sum = 0.0f32;
        for tap in 0..CW { sum += history[tap] * prepared_weight[channel * CW + tap]; }
        expected_raw[channel] = rb(sum / (1.0 + (-sum).exp()));
    }
    let zero_z = vec![0.0; VW];
    let zero_heads = vec![0.0; VH];
    let unit_norm = vec![1.0; D];
    let mut cpu_conv = initial_conv;
    let mut cpu_recurrent = vec![0.0; VH * D * D];
    let (expected_normalized, _, _, _) = cpu_decode(HostDecode {
        projected_qkv: &prepared_projected, z: &zero_z,
        a: &zero_heads, b: &zero_heads, conv_weight: &prepared_weight,
        a_log: &zero_heads, dt_bias: &zero_heads, norm_weight: &unit_norm,
        conv_state: &mut cpu_conv, recurrent_state: &mut cpu_recurrent,
    });
    assert_eq!(cpu_conv, expected_conv, "CPU raw and complete reference disagree on conv history");
    let mut expected_after_q = expected_raw.clone();
    expected_after_q[..QW].copy_from_slice(&expected_normalized[..QW]);

    ops::gdn_causal_conv_step(&ctx, &state_device, &projected_device, &weight_device, &output).unwrap();
    ctx.synchronize().unwrap();
    let raw = read_bf16(&output);
    assert_eq!(read_f32(&state_device), expected_conv);
    ops::gdn_l2_normalize_heads(&ctx, &view(&output, 0, vec![KH, D], DType::BF16), ops::QWEN35_GDN_EPSILON).unwrap();
    ctx.synchronize().unwrap();
    let after_q = read_bf16(&output);
    ops::gdn_l2_normalize_heads(&ctx, &view(&output, QW, vec![KH, D], DType::BF16), ops::QWEN35_GDN_EPSILON).unwrap();
    ctx.synchronize().unwrap();
    let after_qk = read_bf16(&output);

    // Print all stages before the first numerical assert, so a raw-conv
    // failure does not hide the Q/K behavior. These are diagnostic numbers;
    // the complete decode test and its original full-vector gate remain.
    report_qkv_stage("isolated/raw-conv", &raw, &expected_raw);
    report_qkv_stage("isolated/after-Qnorm", &after_q, &expected_after_q);
    report_qkv_stage("isolated/after-QKnorm", &after_qk, &expected_normalized);
    assert_eq!(bitwise_mismatches(&raw[QW..], &after_q[QW..]), 0, "Qnorm modified K/V");
    assert_eq!(bitwise_mismatches(&after_q[..QW], &after_qk[..QW]), 0, "Knorm modified Q");
    assert_eq!(bitwise_mismatches(&raw[2 * QW..], &after_qk[2 * QW..]), 0, "QKnorm modified V");
    assert_eq!(read_bf16(&projected_device), prepared_projected, "input changed");
    assert_eq!(read_bf16(&weight_device), prepared_weight, "weight changed");
    assert_eq!(read_f32(&state_device), expected_conv, "QKnorm changed conv history");
    for (name, got, want) in [
        ("raw convolution", raw.as_slice(), expected_raw.as_slice()),
        ("after Q norm", after_q.as_slice(), expected_after_q.as_slice()),
        ("after QK norm", after_qk.as_slice(), expected_normalized.as_slice()),
    ] {
        assert_long_parity(name, got, want, 0.99999, 0.01, false);
    }
}

fn prefill_fixture(
    ctx: &CudaContext,
    tokens: usize,
    projected: &[f32],
    z: &[f32],
    a: &[f32],
    b: &[f32],
    conv_weight: &[f32],
    a_log: &[f32],
    dt_bias: &[f32],
    norm_weight: &[f32],
    with_cake: bool,
) -> DeviceFixture {
    DeviceFixture {
        projected_qkv: upload_bf16(ctx, projected, vec![tokens, QKV]),
        z: upload_bf16(ctx, z, vec![tokens, VW]),
        a: upload_bf16(ctx, a, vec![tokens, VH]),
        b: upload_bf16(ctx, b, vec![tokens, VH]),
        conv_weight: upload_bf16(ctx, conv_weight, vec![QKV, 1, CW]),
        a_log: upload_bf16(ctx, a_log, vec![VH]),
        dt_bias: upload_bf16(ctx, dt_bias, vec![VH]),
        norm_weight: upload_bf16(ctx, norm_weight, vec![D]),
        conv_state: zeros(ctx, vec![QKV, CW], DType::F32),
        recurrent_state: zeros(ctx, vec![VH, D, D], DType::F32),
        convolved: zeros(ctx, vec![tokens, QKV], DType::BF16),
        decay: zeros(ctx, vec![tokens, VH], DType::F32),
        beta: zeros(ctx, vec![tokens, VH], DType::F32),
        core: zeros(ctx, vec![tokens, VW], DType::BF16),
        gated: zeros(ctx, vec![tokens, VW], DType::BF16),
        cake: (with_cake && ctx.caps().sm >= 100).then(|| CakeFixture {
            q16: zeros(ctx, vec![tokens, KH, D], DType::F16),
            k16: zeros(ctx, vec![tokens, KH, D], DType::F16),
            v16: zeros(ctx, vec![tokens, VH, D], DType::F16),
            out16: zeros(ctx, vec![tokens, VH, D], DType::F16),
            alpha: zeros(ctx, vec![tokens, VH], DType::F32),
            cu_seqlens: zeros(ctx, vec![2], DType::I32),
            workspace: zeros(
                ctx,
                vec![ops::gdn_prefill_workspace_bytes(VH, 1).div_ceil(4)],
                DType::F32,
            ),
        }),
    }
}

#[test]
fn chunk_prefill_and_decode_share_exact_qwen35_state_contract() {
    let ctx = CudaContext::new(0).unwrap();
    let tokens = ops::QWEN35_GDN_SCAN_CHUNK;
    let projected: Vec<f32> = (0..tokens * QKV)
        .map(|index| ((index * 13 % 61) as f32 - 30.0) / 47.0)
        .collect();
    let z: Vec<f32> = (0..tokens * VW)
        .map(|index| ((index * 7 % 31) as f32 - 15.0) / 17.0)
        .collect();
    let a: Vec<f32> = (0..tokens * VH)
        .map(|index| ((index * 3 % 23) as f32 - 11.0) / 19.0)
        .collect();
    let b: Vec<f32> = (0..tokens * VH)
        .map(|index| ((index * 5 % 29) as f32 - 14.0) / 13.0)
        .collect();
    // A delta-like convolution still exercises and persists all four history
    // slots while keeping the recurrence comparison numerically well-scaled.
    let mut conv_weight = vec![0.0f32; QKV * CW];
    for channel in 0..QKV {
        conv_weight[channel * CW + 3] = 0.75;
        conv_weight[channel * CW + 2] = -0.125;
    }
    let a_log: Vec<f32> = (0..VH).map(|head| -1.1 + head as f32 / 71.0).collect();
    let dt_bias: Vec<f32> = (0..VH).map(|head| -0.2 + head as f32 / 83.0).collect();
    let norm_weight: Vec<f32> = (0..D).map(|index| 0.9 + index as f32 / 521.0).collect();

    let fast = prefill_fixture(
        &ctx,
        tokens,
        &projected,
        &z,
        &a,
        &b,
        &conv_weight,
        &a_log,
        &dt_bias,
        &norm_weight,
        true,
    );
    let stepped = prefill_fixture(
        &ctx,
        tokens,
        &projected,
        &z,
        &a,
        &b,
        &conv_weight,
        &a_log,
        &dt_bias,
        &norm_weight,
        false,
    );

    ops::qwen35_gdn_prefill(
        &ctx,
        tokens,
        &ops::Qwen35GdnPrefillArgs {
            projected_qkv: &fast.projected_qkv,
            z: &fast.z,
            a: &fast.a,
            b: &fast.b,
            weights: ops::Qwen35GdnCoreWeights {
                conv: &fast.conv_weight,
                a_log: &fast.a_log,
                dt_bias: &fast.dt_bias,
                norm: &fast.norm_weight,
            },
            state: ops::Qwen35GdnState {
                conv: &fast.conv_state,
                recurrent: &fast.recurrent_state,
            },
            checkpoints: None,
            scratch: ops::Qwen35GdnPrefillScratch {
                convolved_qkv: &fast.convolved,
                decay: &fast.decay,
                beta: &fast.beta,
                core_output: &fast.core,
                gated_output: &fast.gated,
            },
            cake: fast.cake_scratch(),
        },
    )
    .unwrap();

    for token in 0..tokens {
        ops::qwen35_gdn_decode(
            &ctx,
            &ops::Qwen35GdnDecodeArgs {
                projected_qkv: &view(&stepped.projected_qkv, token * QKV, vec![QKV], DType::BF16),
                z: &view(&stepped.z, token * VW, vec![VW], DType::BF16),
                a: &view(&stepped.a, token * VH, vec![VH], DType::BF16),
                b: &view(&stepped.b, token * VH, vec![VH], DType::BF16),
                weights: ops::Qwen35GdnCoreWeights {
                    conv: &stepped.conv_weight,
                    a_log: &stepped.a_log,
                    dt_bias: &stepped.dt_bias,
                    norm: &stepped.norm_weight,
                },
                state: ops::Qwen35GdnState {
                    conv: &stepped.conv_state,
                    recurrent: &stepped.recurrent_state,
                },
                scratch: ops::Qwen35GdnDecodeScratch {
                    convolved_qkv: &view(&stepped.convolved, token * QKV, vec![QKV], DType::BF16),
                    decay: &view(&stepped.decay, token * VH, vec![VH], DType::F32),
                    beta: &view(&stepped.beta, token * VH, vec![VH], DType::F32),
                    core_output: &view(&stepped.core, token * VW, vec![VW], DType::BF16),
                    gated_output: &view(&stepped.gated, token * VW, vec![VW], DType::BF16),
                },
            },
        )
        .unwrap();
    }
    ctx.synchronize().unwrap();

    assert_eq!(read_f32(&fast.conv_state), read_f32(&stepped.conv_state));
    for (name, got, want, min_cos, max_rel) in [
        (
            "prefill recurrent state",
            read_f32(&fast.recurrent_state),
            read_f32(&stepped.recurrent_state),
            0.9998,
            0.025,
        ),
        (
            "prefill gated output",
            read_bf16(&fast.gated),
            read_bf16(&stepped.gated),
            0.999,
            0.05,
        ),
    ] {
        let (cosine, relative) = cosine_rel_l2(&got, &want);
        println!("{name}: cosine={cosine:.7} relL2={relative:.7}");
        assert!(cosine >= min_cos && relative <= max_rel, "{name} mismatch");
    }

    // The real hand-off condition: run one more token from both final states.
    let next_projected: Vec<f32> = (0..QKV)
        .map(|index| ((index * 19 % 67) as f32 - 33.0) / 51.0)
        .collect();
    let next_z: Vec<f32> = (0..VW)
        .map(|index| ((index * 11 % 37) as f32 - 18.0) / 19.0)
        .collect();
    let next_a: Vec<f32> = (0..VH).map(|head| head as f32 / 43.0 - 0.3).collect();
    let next_b: Vec<f32> = (0..VH).map(|head| 0.4 - head as f32 / 59.0).collect();
    let next = |conv_state: &Tensor, recurrent_state: &Tensor| DeviceFixture {
        projected_qkv: upload_bf16(&ctx, &next_projected, vec![QKV]),
        z: upload_bf16(&ctx, &next_z, vec![VW]),
        a: upload_bf16(&ctx, &next_a, vec![VH]),
        b: upload_bf16(&ctx, &next_b, vec![VH]),
        conv_weight: upload_bf16(&ctx, &conv_weight, vec![QKV, 1, CW]),
        a_log: upload_bf16(&ctx, &a_log, vec![VH]),
        dt_bias: upload_bf16(&ctx, &dt_bias, vec![VH]),
        norm_weight: upload_bf16(&ctx, &norm_weight, vec![D]),
        conv_state: conv_state.clone(),
        recurrent_state: recurrent_state.clone(),
        convolved: zeros(&ctx, vec![QKV], DType::BF16),
        decay: zeros(&ctx, vec![VH], DType::F32),
        beta: zeros(&ctx, vec![VH], DType::F32),
        core: zeros(&ctx, vec![VW], DType::BF16),
        gated: zeros(&ctx, vec![VW], DType::BF16),
        // Decode-only fixture: the Cake buffers are a prefill concern.
        cake: None,
    };
    let fast_next = next(&fast.conv_state, &fast.recurrent_state);
    let stepped_next = next(&stepped.conv_state, &stepped.recurrent_state);
    ops::qwen35_gdn_decode(&ctx, &fast_next.decode()).unwrap();
    ops::qwen35_gdn_decode(&ctx, &stepped_next.decode()).unwrap();
    ctx.synchronize().unwrap();
    let (cosine, relative) = cosine_rel_l2(&read_bf16(&fast_next.gated), &read_bf16(&stepped_next.gated));
    println!("first decode after prefill: cosine={cosine:.7} relL2={relative:.7}");
    assert!(cosine >= 0.999 && relative <= 0.05);
}

/// Run a contiguous slice through the same prefill adapter as production.
/// Cake scratch is reused from row zero; input/output views keep the global
/// row offset. The causal conv and recurrent states are never reset between
/// segments, which is the prefix -> append contract being tested.
fn prefill_rows(ctx: &CudaContext, fixture: &DeviceFixture, start: usize, rows: usize) {
    let cake = fixture.cake_views(rows);
    ops::qwen35_gdn_prefill(
        ctx,
        rows,
        &ops::Qwen35GdnPrefillArgs {
            projected_qkv: &view(&fixture.projected_qkv, start * QKV, vec![rows, QKV], DType::BF16),
            z: &view(&fixture.z, start * VW, vec![rows, VW], DType::BF16),
            a: &view(&fixture.a, start * VH, vec![rows, VH], DType::BF16),
            b: &view(&fixture.b, start * VH, vec![rows, VH], DType::BF16),
            weights: ops::Qwen35GdnCoreWeights {
                conv: &fixture.conv_weight,
                a_log: &fixture.a_log,
                dt_bias: &fixture.dt_bias,
                norm: &fixture.norm_weight,
            },
            state: ops::Qwen35GdnState {
                conv: &fixture.conv_state,
                recurrent: &fixture.recurrent_state,
            },
            checkpoints: None,
            scratch: ops::Qwen35GdnPrefillScratch {
                convolved_qkv: &view(&fixture.convolved, start * QKV, vec![rows, QKV], DType::BF16),
                decay: &view(&fixture.decay, start * VH, vec![rows, VH], DType::F32),
                beta: &view(&fixture.beta, start * VH, vec![rows, VH], DType::F32),
                core_output: &view(&fixture.core, start * VW, vec![rows, VW], DType::BF16),
                gated_output: &view(&fixture.gated, start * VW, vec![rows, VW], DType::BF16),
            },
            cake: cake.as_ref().map(CakeFixture::scratch),
        },
    ).unwrap();
}

fn decode_row(ctx: &CudaContext, fixture: &DeviceFixture, row: usize) {
    ops::qwen35_gdn_decode(
        ctx,
        &ops::Qwen35GdnDecodeArgs {
            projected_qkv: &view(&fixture.projected_qkv, row * QKV, vec![QKV], DType::BF16),
            z: &view(&fixture.z, row * VW, vec![VW], DType::BF16),
            a: &view(&fixture.a, row * VH, vec![VH], DType::BF16),
            b: &view(&fixture.b, row * VH, vec![VH], DType::BF16),
            weights: ops::Qwen35GdnCoreWeights {
                conv: &fixture.conv_weight,
                a_log: &fixture.a_log,
                dt_bias: &fixture.dt_bias,
                norm: &fixture.norm_weight,
            },
            state: ops::Qwen35GdnState {
                conv: &fixture.conv_state,
                recurrent: &fixture.recurrent_state,
            },
            scratch: ops::Qwen35GdnDecodeScratch {
                convolved_qkv: &view(&fixture.convolved, row * QKV, vec![QKV], DType::BF16),
                decay: &view(&fixture.decay, row * VH, vec![VH], DType::F32),
                beta: &view(&fixture.beta, row * VH, vec![VH], DType::F32),
                core_output: &view(&fixture.core, row * VW, vec![VW], DType::BF16),
                gated_output: &view(&fixture.gated, row * VW, vec![VW], DType::BF16),
            },
        },
    ).unwrap();
}

fn assert_gdn_boundary(
    label: &str,
    fast: &DeviceFixture,
    canonical: &DeviceFixture,
    output_start: usize,
    output_rows: usize,
    require_bit_exact: bool,
) {
    let fast_conv = read_f32(&fast.conv_state);
    let canonical_conv = read_f32(&canonical.conv_state);
    assert!(fast_conv.iter().chain(&canonical_conv).all(|value| value.is_finite()),
            "{label}: convolution state is not finite");
    assert_eq!(bitwise_mismatches(&fast_conv, &canonical_conv), 0,
               "{label}: causal-convolution window is not bit-exact");
    // Frozen existing 64-token / 2,049-token contracts. A longer sequence does
    // not justify relaxing them. Numeric parity and optional bitwise diagnosis
    // remain separate; scan and recurrence can reorder F32 arithmetic.
    assert_long_parity(
        &format!("{label} recurrent state"),
        &read_f32(&fast.recurrent_state),
        &read_f32(&canonical.recurrent_state),
        0.9998, 0.025, require_bit_exact,
    );
    assert_long_parity(
        &format!("{label} gated output rows {output_start}..{}", output_start + output_rows),
        &read_bf16(&view(&fast.gated, output_start * VW, vec![output_rows, VW], DType::BF16)),
        &read_bf16(&view(&canonical.gated, output_start * VW, vec![output_rows, VW], DType::BF16)),
        0.999, 0.05, require_bit_exact,
    );
    if output_rows > ops::QWEN35_GDN_SCAN_CHUNK {
        // A reset at an append boundary could be diluted by thousands of
        // later rows after the recurrent dynamics forget the prefix. Apply
        // the existing 64-token output contract to the first chunk as well.
        let rows = ops::QWEN35_GDN_SCAN_CHUNK;
        assert_long_parity(
            &format!("{label} first append/chunk output rows {output_start}..{}", output_start + rows),
            &read_bf16(&view(&fast.gated, output_start * VW, vec![rows, VW], DType::BF16)),
            &read_bf16(&view(&canonical.gated, output_start * VW, vec![rows, VW], DType::BF16)),
            0.999, 0.05, require_bit_exact,
        );
    }
}

/// Long GPU *scheduling/state* oracle: production chunk/Cake prefill versus
/// one-token CUDA recurrence over identical prepared BF16 inputs. This is not
/// an independent mathematical oracle; the short CPU reference test supplies
/// that separate claim. Every segment's full outputs and final state are
/// checked before a subsequent segment can obscure a boundary failure.
fn long_prefill_canonical_case(segments: &[usize]) {
    let started = std::time::Instant::now();
    assert!(!segments.is_empty());
    assert!(segments.iter().all(|rows| *rows > 0 && rows % ops::QWEN35_GDN_SCAN_CHUNK == 0));
    let scanned: usize = segments.iter().sum();
    let tokens = scanned + 1; // first scalar decode after the last prefill
    println!("gdn-long prepare segments={segments:?} rows={tokens} target_geometry={KH}/{VH}/{D} tuning=false");
    let ctx = CudaContext::new_with_autotune(0, false).unwrap();
    let projected: Vec<f32> = (0..tokens * QKV)
        .map(|index| ((index * 13 % 61) as f32 - 30.0) / 47.0).collect();
    let z: Vec<f32> = (0..tokens * VW)
        .map(|index| ((index * 7 % 31) as f32 - 15.0) / 17.0).collect();
    let a: Vec<f32> = (0..tokens * VH)
        .map(|index| ((index * 3 % 23) as f32 - 11.0) / 19.0).collect();
    let b: Vec<f32> = (0..tokens * VH)
        .map(|index| ((index * 5 % 29) as f32 - 14.0) / 13.0).collect();
    let mut conv_weight = vec![0.0f32; QKV * CW];
    for channel in 0..QKV {
        conv_weight[channel * CW + 3] = 0.75;
        conv_weight[channel * CW + 2] = -0.125;
    }
    let a_log: Vec<f32> = (0..VH).map(|head| -1.1 + head as f32 / 71.0).collect();
    let dt_bias: Vec<f32> = (0..VH).map(|head| -0.2 + head as f32 / 83.0).collect();
    let norm_weight: Vec<f32> = (0..D).map(|index| 0.9 + index as f32 / 521.0).collect();
    let fast = prefill_fixture(&ctx, tokens, &projected, &z, &a, &b,
                               &conv_weight, &a_log, &dt_bias, &norm_weight, true);
    let canonical = prefill_fixture(&ctx, tokens, &projected, &z, &a, &b,
                                    &conv_weight, &a_log, &dt_bias, &norm_weight, false);
    // Independent device inputs are retained on both sides. Host F32 source
    // matrices and unused canonical Cake buffers need not occupy long tests.
    drop((projected, z, a, b));
    let require_bit_exact = std::env::var("APXINF_QWEN35_REQUIRE_BIT_EXACT_GDN")
        .is_ok_and(|value| value == "1");
    println!("gdn-long segments={segments:?} rows={tokens} oracle=GPU-canonical-recurrence independent_math=false cake={} bit_exact_requested={require_bit_exact}", fast.cake.is_some());
    let mut start = 0;
    for &rows in segments {
        prefill_rows(&ctx, &fast, start, rows);
        for token in start..start + rows { decode_row(&ctx, &canonical, token); }
        ctx.synchronize().unwrap();
        let label = format!("segments={segments:?} boundary={}", start + rows);
        assert_gdn_boundary(&label, &fast, &canonical, start, rows, require_bit_exact);
        if segments.len() > 1 && start == 0 {
            assert!(read_f32(&fast.recurrent_state).iter().any(|value| *value != 0.0),
                    "append test did not establish a nonzero prefix recurrent state");
            assert!(read_f32(&fast.conv_state).iter().any(|value| *value != 0.0),
                    "append test did not establish a nonzero prefix convolution window");
        }
        println!("gdn-long {label} elapsed_s={:.3}", started.elapsed().as_secs_f64());
        start += rows;
    }
    decode_row(&ctx, &fast, scanned);
    decode_row(&ctx, &canonical, scanned);
    ctx.synchronize().unwrap();
    assert_gdn_boundary(&format!("segments={segments:?} next-decode"), &fast, &canonical,
                        scanned, 1, require_bit_exact);
    println!("gdn-long segments={segments:?} PASS elapsed_s={:.3} (test duration, not inference throughput)", started.elapsed().as_secs_f64());
}

/// Original 2,048 scan + 1 decode gate, now also checking the scan boundary.
#[test]
#[ignore = "long-running 2,049-token exact-geometry CUDA parity gate"]
fn long_prefill_output_window_and_state_match_canonical_decode() {
    long_prefill_canonical_case(&[2_048]);
}

macro_rules! long_gdn_case {
    ($name:ident, $segments:expr) => {
        #[test]
        #[ignore = "long-running exact-geometry GDN prefill/state GPU recurrence gate"]
        fn $name() { long_prefill_canonical_case($segments); }
    };
}

long_gdn_case!(long_prefill_4096_matches_canonical_decode, &[4_096]);
long_gdn_case!(long_prefill_8192_matches_canonical_decode, &[8_192]);
long_gdn_case!(long_prefill_16384_matches_canonical_decode, &[16_384]);
long_gdn_case!(long_prefix_4096_append_4096_matches_canonical_decode, &[4_096, 4_096]);
long_gdn_case!(long_prefix_8192_append_4096_matches_canonical_decode, &[8_192, 4_096]);
long_gdn_case!(long_prefix_16384_append_4096_matches_canonical_decode, &[16_384, 4_096]);
long_gdn_case!(long_prefix_32768_append_4096_matches_canonical_decode, &[32_768, 4_096]);
long_gdn_case!(long_prefix_49152_append_4096_matches_canonical_decode, &[49_152, 4_096]);

#[test]
fn invalid_decode_is_rejected_before_state_mutation() {
    let ctx = CudaContext::new(0).unwrap();
    let fixture = DeviceFixture {
        projected_qkv: zeros(&ctx, vec![QKV], DType::BF16),
        z: zeros(&ctx, vec![VH, D], DType::BF16), // deliberately not flat [4096]
        a: zeros(&ctx, vec![VH], DType::BF16),
        b: zeros(&ctx, vec![VH], DType::BF16),
        conv_weight: zeros(&ctx, vec![QKV, 1, CW], DType::BF16),
        a_log: zeros(&ctx, vec![VH], DType::BF16),
        dt_bias: zeros(&ctx, vec![VH], DType::BF16),
        norm_weight: zeros(&ctx, vec![D], DType::BF16),
        conv_state: upload_f32(&ctx, &vec![0.25; QKV * CW], vec![QKV, CW]),
        recurrent_state: zeros(&ctx, vec![VH, D, D], DType::F32),
        convolved: zeros(&ctx, vec![QKV], DType::BF16),
        decay: zeros(&ctx, vec![VH], DType::F32),
        beta: zeros(&ctx, vec![VH], DType::F32),
        core: zeros(&ctx, vec![VW], DType::BF16),
        gated: zeros(&ctx, vec![VW], DType::BF16),
        // Decode-only fixture: the Cake buffers are a prefill concern.
        cake: None,
    };
    let before = read_f32(&fixture.conv_state);
    assert!(ops::qwen35_gdn_decode(&ctx, &fixture.decode()).is_err());
    assert_eq!(read_f32(&fixture.conv_state), before);
}
