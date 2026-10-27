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
    let buffer = CudaBuffer::alloc_zeros(bytes, ctx.device_id()).unwrap();
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

fn assert_long_parity(
    name: &str,
    got: &[f32],
    want: &[f32],
    min_cosine: f64,
    max_relative_l2: f64,
    require_bit_exact: bool,
) {
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
        cake: (ctx.caps().sm >= 100).then(|| CakeFixture {
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

/// Long-sequence gate corresponding to the 2,049-token GDN check used by the
/// Qwen3.8 MTP tuning. The fast side scans 2,048 tokens in 64-token chunks and
/// then consumes token 2,049 through decode; the canonical side executes the
/// same 2,049 rows through the one-token recurrence.
///
/// The convolution window is required to be bit-exact unconditionally. Set
/// `APXINF_QWEN35_REQUIRE_BIT_EXACT_GDN=1` for the final canonical gate, which
/// additionally requires the complete BF16 output and F32 recurrent state to
/// be bit-identical. Without that flag the test still prints mismatch counts
/// and enforces the existing numerical scan contract, allowing it to
/// characterize a non-canonical kernel before the implementation is changed.
#[test]
#[ignore = "long-running 2,049-token exact-geometry CUDA parity gate"]
fn long_prefill_output_window_and_state_match_canonical_decode() {
    const TOKENS: usize = 2_049;
    const SCANNED_TOKENS: usize = 2_048;

    let ctx = CudaContext::new(0).unwrap();
    let projected: Vec<f32> = (0..TOKENS * QKV)
        .map(|index| ((index * 13 % 61) as f32 - 30.0) / 47.0)
        .collect();
    let z: Vec<f32> = (0..TOKENS * VW)
        .map(|index| ((index * 7 % 31) as f32 - 15.0) / 17.0)
        .collect();
    let a: Vec<f32> = (0..TOKENS * VH)
        .map(|index| ((index * 3 % 23) as f32 - 11.0) / 19.0)
        .collect();
    let b: Vec<f32> = (0..TOKENS * VH)
        .map(|index| ((index * 5 % 29) as f32 - 14.0) / 13.0)
        .collect();
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
        TOKENS,
        &projected,
        &z,
        &a,
        &b,
        &conv_weight,
        &a_log,
        &dt_bias,
        &norm_weight,
    );
    let canonical = prefill_fixture(
        &ctx,
        TOKENS,
        &projected,
        &z,
        &a,
        &b,
        &conv_weight,
        &a_log,
        &dt_bias,
        &norm_weight,
    );

    let cake_views = fast.cake_views(SCANNED_TOKENS);
    ops::qwen35_gdn_prefill(
        &ctx,
        SCANNED_TOKENS,
        &ops::Qwen35GdnPrefillArgs {
            projected_qkv: &view(
                &fast.projected_qkv,
                0,
                vec![SCANNED_TOKENS, QKV],
                DType::BF16,
            ),
            z: &view(&fast.z, 0, vec![SCANNED_TOKENS, VW], DType::BF16),
            a: &view(&fast.a, 0, vec![SCANNED_TOKENS, VH], DType::BF16),
            b: &view(&fast.b, 0, vec![SCANNED_TOKENS, VH], DType::BF16),
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
                convolved_qkv: &view(
                    &fast.convolved,
                    0,
                    vec![SCANNED_TOKENS, QKV],
                    DType::BF16,
                ),
                decay: &view(
                    &fast.decay,
                    0,
                    vec![SCANNED_TOKENS, VH],
                    DType::F32,
                ),
                beta: &view(
                    &fast.beta,
                    0,
                    vec![SCANNED_TOKENS, VH],
                    DType::F32,
                ),
                core_output: &view(
                    &fast.core,
                    0,
                    vec![SCANNED_TOKENS, VW],
                    DType::BF16,
                ),
                gated_output: &view(
                    &fast.gated,
                    0,
                    vec![SCANNED_TOKENS, VW],
                    DType::BF16,
                ),
            },
            // Prefix views, because the fixture is profiled for TOKENS while
            // this prefill covers the chunk-aligned SCANNED_TOKENS. With these
            // present the comparison below is Cake-prefill versus canonical
            // per-token decode, which is the handoff production depends on.
            cake: cake_views.as_ref().map(CakeFixture::scratch),
        },
    )
    .unwrap();
    ops::qwen35_gdn_decode(
        &ctx,
        &ops::Qwen35GdnDecodeArgs {
            projected_qkv: &view(
                &fast.projected_qkv,
                SCANNED_TOKENS * QKV,
                vec![QKV],
                DType::BF16,
            ),
            z: &view(
                &fast.z,
                SCANNED_TOKENS * VW,
                vec![VW],
                DType::BF16,
            ),
            a: &view(
                &fast.a,
                SCANNED_TOKENS * VH,
                vec![VH],
                DType::BF16,
            ),
            b: &view(
                &fast.b,
                SCANNED_TOKENS * VH,
                vec![VH],
                DType::BF16,
            ),
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
            scratch: ops::Qwen35GdnDecodeScratch {
                convolved_qkv: &view(
                    &fast.convolved,
                    SCANNED_TOKENS * QKV,
                    vec![QKV],
                    DType::BF16,
                ),
                decay: &view(
                    &fast.decay,
                    SCANNED_TOKENS * VH,
                    vec![VH],
                    DType::F32,
                ),
                beta: &view(
                    &fast.beta,
                    SCANNED_TOKENS * VH,
                    vec![VH],
                    DType::F32,
                ),
                core_output: &view(
                    &fast.core,
                    SCANNED_TOKENS * VW,
                    vec![VW],
                    DType::BF16,
                ),
                gated_output: &view(
                    &fast.gated,
                    SCANNED_TOKENS * VW,
                    vec![VW],
                    DType::BF16,
                ),
            },
        },
    )
    .unwrap();

    for token in 0..TOKENS {
        ops::qwen35_gdn_decode(
            &ctx,
            &ops::Qwen35GdnDecodeArgs {
                projected_qkv: &view(
                    &canonical.projected_qkv,
                    token * QKV,
                    vec![QKV],
                    DType::BF16,
                ),
                z: &view(&canonical.z, token * VW, vec![VW], DType::BF16),
                a: &view(&canonical.a, token * VH, vec![VH], DType::BF16),
                b: &view(&canonical.b, token * VH, vec![VH], DType::BF16),
                weights: ops::Qwen35GdnCoreWeights {
                    conv: &canonical.conv_weight,
                    a_log: &canonical.a_log,
                    dt_bias: &canonical.dt_bias,
                    norm: &canonical.norm_weight,
                },
                state: ops::Qwen35GdnState {
                    conv: &canonical.conv_state,
                    recurrent: &canonical.recurrent_state,
                },
                scratch: ops::Qwen35GdnDecodeScratch {
                    convolved_qkv: &view(
                        &canonical.convolved,
                        token * QKV,
                        vec![QKV],
                        DType::BF16,
                    ),
                    decay: &view(
                        &canonical.decay,
                        token * VH,
                        vec![VH],
                        DType::F32,
                    ),
                    beta: &view(
                        &canonical.beta,
                        token * VH,
                        vec![VH],
                        DType::F32,
                    ),
                    core_output: &view(
                        &canonical.core,
                        token * VW,
                        vec![VW],
                        DType::BF16,
                    ),
                    gated_output: &view(
                        &canonical.gated,
                        token * VW,
                        vec![VW],
                        DType::BF16,
                    ),
                },
            },
        )
        .unwrap();
    }
    ctx.synchronize().unwrap();

    let fast_conv = read_f32(&fast.conv_state);
    let canonical_conv = read_f32(&canonical.conv_state);
    assert_eq!(
        bitwise_mismatches(&fast_conv, &canonical_conv),
        0,
        "2,049-token causal-convolution window is not bit-exact"
    );

    let require_bit_exact = std::env::var("APXINF_QWEN35_REQUIRE_BIT_EXACT_GDN")
        .is_ok_and(|value| value == "1");
    assert_long_parity(
        "2,049-token recurrent state",
        &read_f32(&fast.recurrent_state),
        &read_f32(&canonical.recurrent_state),
        0.9998,
        0.025,
        require_bit_exact,
    );
    assert_long_parity(
        "2,049-token gated output",
        &read_bf16(&fast.gated),
        &read_bf16(&canonical.gated),
        0.999,
        0.05,
        require_bit_exact,
    );
}

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
