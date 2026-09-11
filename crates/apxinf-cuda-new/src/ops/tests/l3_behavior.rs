//! Per-operator L3 semantic and conditional Graph behavior tests.
//!
//! Every new L3 operator must add a public semantic-contract test here. Add a
//! Graph replay test only when the operator has an independent execution path,
//! resource lifetime, binding rule, or capture behavior not already covered by
//! the shared framework tests.

use super::framework::{tensor, values};
use super::*;
use crate::{CudaBuffer, CudaContext};
use half::bf16;

fn assert_close(actual: &[f32], expected: &[f32], tolerance: f32) {
    assert_eq!(actual.len(), expected.len(), "output length mismatch");
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            actual.is_finite(),
            "element {index} is not finite: {actual}"
        );
        assert!(
            (actual - expected).abs() <= tolerance,
            "element {index}: got {actual}, expected {expected}"
        );
    }
}

fn matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut output = vec![0.0; m * n];
    for row in 0..m {
        for column in 0..n {
            output[row * n + column] = (0..k)
                .map(|inner| a[row * k + inner] * b[inner * n + column])
                .sum();
        }
    }
    output
}

fn gelu_tanh(value: f32) -> f32 {
    0.5 * value * (1.0 + (0.79788456 * (value + 0.044715 * value.powi(3))).tanh())
}

fn write_bf16(tensor: &apxinf_core::Tensor, input: &[f32]) {
    assert_eq!(tensor.shape().numel(), input.len());
    let bytes: Vec<u8> = input
        .iter()
        .flat_map(|&value| bf16::from_f32(value).to_bits().to_ne_bytes())
        .collect();
    CudaBuffer::from_tensor(tensor)
        .unwrap()
        .copy_from_host(&bytes)
        .unwrap();
}

const A: [f32; 6] = [1.0, 2.0, -1.0, 0.0, 3.0, 1.0];
const B: [f32; 6] = [1.0, 2.0, 0.0, -1.0, 2.0, 1.0];
const BIAS: [f32; 2] = [0.5, -1.0];
const ALPHA: f32 = 0.75;
const OUTPUT_SCALE: f32 = 1.25;

#[test]
fn gemm_has_its_own_semantic_api() {
    let ctx = CudaContext::new(0).unwrap();
    let a = tensor(0, vec![2, 3], &A);
    let b = tensor(0, vec![3, 2], &B);
    let mut out = tensor(0, vec![2, 2], &[0.0; 4]);
    let mut args = GemmArgs::new(&a, &b, &mut out);
    args.alpha = ALPHA;
    args.output_scale = OUTPUT_SCALE;
    args.policy.online_tune = false;
    gemm(&ctx, args).unwrap();

    let expected: Vec<_> = matmul(&A, &B, 2, 3, 2)
        .into_iter()
        .map(|value| ALPHA * value / OUTPUT_SCALE)
        .collect();
    assert_close(&values(&out), &expected, 0.02);
}

#[test]
fn gemm_bias_has_its_own_semantic_api() {
    let ctx = CudaContext::new(0).unwrap();
    let a = tensor(0, vec![2, 3], &A);
    let b = tensor(0, vec![3, 2], &B);
    let bias = tensor(0, vec![2], &BIAS);
    let mut out = tensor(0, vec![2, 2], &[0.0; 4]);
    let mut args = GemmArgs::new(&a, &b, &mut out);
    args.alpha = ALPHA;
    args.output_scale = OUTPUT_SCALE;
    args.policy.online_tune = false;
    gemm_bias(
        &ctx,
        GemmBiasArgs {
            gemm: args,
            bias: &bias,
        },
    )
    .unwrap();

    let expected: Vec<_> = matmul(&A, &B, 2, 3, 2)
        .into_iter()
        .enumerate()
        .map(|(index, value)| (ALPHA * value + BIAS[index % 2]) / OUTPUT_SCALE)
        .collect();
    assert_close(&values(&out), &expected, 0.02);
}

#[test]
fn gemm_bias_gelu_has_its_own_semantic_api() {
    let ctx = CudaContext::new(0).unwrap();
    let a = tensor(0, vec![2, 3], &A);
    let b = tensor(0, vec![3, 2], &B);
    let bias = tensor(0, vec![2], &BIAS);
    let mut out = tensor(0, vec![2, 2], &[0.0; 4]);
    let mut args = GemmArgs::new(&a, &b, &mut out);
    args.alpha = ALPHA;
    args.output_scale = OUTPUT_SCALE;
    args.policy.online_tune = false;
    gemm_bias_gelu(
        &ctx,
        GemmBiasGeluArgs {
            gemm: args,
            bias: &bias,
        },
    )
    .unwrap();

    let expected: Vec<_> = matmul(&A, &B, 2, 3, 2)
        .into_iter()
        .enumerate()
        .map(|(index, value)| gelu_tanh(ALPHA * value + BIAS[index % 2]) / OUTPUT_SCALE)
        .collect();
    assert_close(&values(&out), &expected, 0.02);
}

#[test]
fn gemm_geglu_is_a_separate_semantic_domain() {
    let ctx = CudaContext::new(0).unwrap();
    let b_geglu = [
        1.0, 2.0, -1.0, 0.5, 0.0, -1.0, 2.0, 1.0, 2.0, 1.0, 0.5, -2.0,
    ];
    let a = tensor(0, vec![2, 3], &A);
    let b = tensor(0, vec![3, 4], &b_geglu);
    let mut out = tensor(0, vec![2, 2], &[0.0; 4]);
    let mut args = GemmArgs::new(&a, &b, &mut out);
    args.alpha = ALPHA;
    args.output_scale = OUTPUT_SCALE;
    args.policy.online_tune = false;
    gemm_geglu(&ctx, GemmGegluArgs { gemm: args }).unwrap();

    let mut expected = vec![0.0; 4];
    for row in 0..2 {
        for column in 0..2 {
            let gate: f32 = (0..3)
                .map(|inner| A[row * 3 + inner] * b_geglu[inner * 4 + column])
                .sum();
            let up: f32 = (0..3)
                .map(|inner| A[row * 3 + inner] * b_geglu[inner * 4 + column + 2])
                .sum();
            expected[row * 2 + column] = gelu_tanh(ALPHA * gate) * (ALPHA * up) / OUTPUT_SCALE;
        }
    }
    assert_close(&values(&out), &expected, 0.03);
}

#[test]
fn gemm_uses_current_values_after_resource_reuse() {
    let ctx = CudaContext::new(0).unwrap();
    let a = tensor(0, vec![2, 3], &A);
    let b = tensor(0, vec![3, 2], &B);
    let mut out = tensor(0, vec![2, 2], &[0.0; 4]);
    let workspace = GraphWorkspace::new(4096, 0).unwrap();

    prepare_with_workspace(&workspace, || {
        let mut args = GemmArgs::new(&a, &b, &mut out);
        args.alpha = ALPHA;
        args.output_scale = OUTPUT_SCALE;
        args.policy.online_tune = false;
        gemm(&ctx, args)
    })
    .unwrap();

    let changed_a = [2.0, -1.0, 0.5, -2.0, 1.0, 3.0];
    write_bf16(&a, &changed_a);
    with_workspace(&workspace, || {
        let mut args = GemmArgs::new(&a, &b, &mut out);
        args.alpha = ALPHA;
        args.output_scale = OUTPUT_SCALE;
        args.policy.online_tune = false;
        gemm(&ctx, args)
    })
    .unwrap();
    let expected: Vec<_> = matmul(&changed_a, &B, 2, 3, 2)
        .into_iter()
        .map(|value| ALPHA * value / OUTPUT_SCALE)
        .collect();
    assert_close(&values(&out), &expected, 0.02);

    let mut args = GemmArgs::new(&a, &b, &mut out);
    args.alpha = 0.5;
    args.output_scale = 2.0;
    args.policy.online_tune = false;
    gemm(&ctx, args).unwrap();
    let expected: Vec<_> = matmul(&changed_a, &B, 2, 3, 2)
        .into_iter()
        .map(|value| 0.5 * value / 2.0)
        .collect();
    assert_close(&values(&out), &expected, 0.02);
}
