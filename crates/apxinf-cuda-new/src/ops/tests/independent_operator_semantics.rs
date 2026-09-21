//! Independent black-box references for the norm, pointwise, RoPE, and gather
//! families.  Expected values in this file are deliberately computed from the
//! public mathematical contracts rather than from native implementation code.

use super::framework::{f16_tensor, tensor};
use super::*;
use crate::{CudaBuffer, CudaContext};
use apxinf_core::{DType, Tensor};
use half::{bf16, f16};

const EPS: f32 = 1.0e-6;

fn tensor_for(dtype: DType, shape: Vec<usize>, values: &[f32]) -> Tensor {
    match dtype {
        DType::BF16 => tensor(0, shape, values),
        DType::F16 => f16_tensor(0, shape, values),
        other => panic!("unsupported test dtype {other}"),
    }
}

fn tensor_values(value: &Tensor) -> Vec<f32> {
    let buffer = CudaBuffer::from_tensor(value).unwrap();
    let mut bytes = vec![0; buffer.len()];
    buffer.copy_to_host(&mut bytes).unwrap();
    match value.dtype() {
        DType::BF16 => bytes
            .chunks_exact(2)
            .map(|chunk| bf16::from_bits(u16::from_ne_bytes([chunk[0], chunk[1]])).to_f32())
            .collect(),
        DType::F16 => bytes
            .chunks_exact(2)
            .map(|chunk| f16::from_bits(u16::from_ne_bytes([chunk[0], chunk[1]])).to_f32())
            .collect(),
        other => panic!("unsupported output dtype {other}"),
    }
}

fn assert_close(actual: &Tensor, expected: &[f32], tolerance: f32) {
    let actual = tensor_values(actual);
    assert_eq!(actual.len(), expected.len());
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).abs() <= tolerance,
            "element {index}: got {actual}, expected {expected} (tolerance {tolerance})"
        );
    }
}

fn rms_rows(input: &[f32], rows: usize, cols: usize, eps: f32) -> Vec<f32> {
    let mut output = vec![0.0; input.len()];
    for row in 0..rows {
        let values = &input[row * cols..(row + 1) * cols];
        let inverse = (values.iter().map(|x| x * x).sum::<f32>() / cols as f32 + eps)
            .sqrt()
            .recip();
        for col in 0..cols {
            output[row * cols + col] = values[col] * inverse;
        }
    }
    output
}

fn rms_weighted(input: &[f32], weight: &[f32], rows: usize, cols: usize, eps: f32) -> Vec<f32> {
    let mut output = rms_rows(input, rows, cols, eps);
    for row in 0..rows {
        for col in 0..cols {
            output[row * cols + col] *= weight[col];
        }
    }
    output
}

fn layer_rows(
    input: &[f32],
    weight: &[f32],
    bias: &[f32],
    rows: usize,
    cols: usize,
    eps: f32,
) -> Vec<f32> {
    let mut output = vec![0.0; input.len()];
    for row in 0..rows {
        let values = &input[row * cols..(row + 1) * cols];
        let mean = values.iter().sum::<f32>() / cols as f32;
        let variance = values
            .iter()
            .map(|x| {
                let centered = x - mean;
                centered * centered
            })
            .sum::<f32>()
            / cols as f32;
        let inverse = (variance + eps).sqrt().recip();
        for col in 0..cols {
            output[row * cols + col] = (values[col] - mean) * inverse * weight[col] + bias[col];
        }
    }
    output
}

fn adaptive_rms(input: &[f32], style: &[f32], rows: usize, cols: usize, eps: f32) -> Vec<f32> {
    let normalized = rms_rows(input, rows, cols, eps);
    let (scale, shift) = style.split_at(cols);
    normalized
        .iter()
        .enumerate()
        .map(|(index, &x)| x * (1.0 + scale[index % cols]) + shift[index % cols])
        .collect()
}

fn bias_residual_reference(
    input: &[f32],
    bias: &[f32],
    residual: &[f32],
    cols: usize,
) -> Vec<f32> {
    input
        .iter()
        .zip(residual)
        .enumerate()
        .map(|(index, (&input, &residual))| input + bias[index % cols] + residual)
        .collect()
}

fn gated_residual(input: &[f32], residual: &[f32], gate_style: &[f32], cols: usize) -> Vec<f32> {
    let gate = &gate_style[2 * cols..3 * cols];
    input
        .iter()
        .zip(residual)
        .enumerate()
        .map(|(index, (&input, &residual))| residual + input * gate[index % cols])
        .collect()
}

#[test]
fn norm_all_semantics_match_independent_cpu_references() {
    let ctx = CudaContext::new(0).unwrap();
    let (rows, cols) = (2, 4);
    let input_values = [1.0, 2.0, 3.0, 4.0, -1.0, 0.5, 2.0, -0.5];
    let residual_values = [0.5, -1.0, 1.5, 0.25, 2.0, -0.5, 0.25, 1.0];
    let bias_values = [0.25, -0.5, 0.75, 1.0];
    let weight_values = [1.0, 0.5, 1.5, -1.0];
    let norm_bias_values = [0.25, -0.25, 0.5, -0.5];
    let norm_style_values = [0.5, -0.25, 1.0, 0.0, 1.0, -1.0, 0.5, 0.25];
    let gate_style_values = [
        7.0, 7.0, 7.0, 7.0, // unrelated scale segment
        9.0, 9.0, 9.0, 9.0, // unrelated shift segment
        0.5, -1.0, 2.0, 0.25,
    ];

    for dtype in [DType::BF16, DType::F16] {
        let input = tensor_for(dtype, vec![rows, cols], &input_values);
        let residual = tensor_for(dtype, vec![rows, cols], &residual_values);
        let bias = tensor_for(dtype, vec![cols], &bias_values);
        let weight = tensor_for(dtype, vec![cols], &weight_values);
        let norm_bias = tensor_for(dtype, vec![cols], &norm_bias_values);
        let norm_style = tensor_for(dtype, vec![2 * cols], &norm_style_values);
        let gate_style = tensor_for(dtype, vec![3 * cols], &gate_style_values);

        let mut normalized = tensor_for(dtype, vec![rows, cols], &[0.0; 8]);
        rms_norm(
            &ctx,
            RmsNormArgs::new(&input, &weight, &mut normalized, EPS),
        )
        .unwrap();
        assert_close(
            &normalized,
            &rms_weighted(&input_values, &weight_values, rows, cols, EPS),
            0.035,
        );

        let mut normalized = tensor_for(dtype, vec![rows, cols], &[0.0; 8]);
        layer_norm(
            &ctx,
            LayerNormArgs::new(&input, &weight, &norm_bias, &mut normalized, EPS),
        )
        .unwrap();
        assert_close(
            &normalized,
            &layer_rows(
                &input_values,
                &weight_values,
                &norm_bias_values,
                rows,
                cols,
                EPS,
            ),
            0.035,
        );

        let mut normalized = tensor_for(dtype, vec![rows, cols], &[0.0; 8]);
        adaptive_rms_norm(
            &ctx,
            AdaptiveRmsNormArgs::new(&input, &norm_style, &mut normalized, EPS),
        )
        .unwrap();
        assert_close(
            &normalized,
            &adaptive_rms(&input_values, &norm_style_values, rows, cols, EPS),
            0.035,
        );

        let hidden_reference =
            bias_residual_reference(&input_values, &bias_values, &residual_values, cols);
        let mut hidden = tensor_for(dtype, vec![rows, cols], &[0.0; 8]);
        bias_residual(
            &ctx,
            BiasResidualArgs::new(&input, Some(&bias), &residual, &mut hidden),
        )
        .unwrap();
        assert_close(&hidden, &hidden_reference, 0.02);

        let mut hidden = tensor_for(dtype, vec![rows, cols], &[0.0; 8]);
        let mut normalized = tensor_for(dtype, vec![rows, cols], &[0.0; 8]);
        bias_residual_rms_norm(
            &ctx,
            BiasResidualRmsNormArgs::new(
                &input,
                Some(&bias),
                &residual,
                &weight,
                &mut hidden,
                &mut normalized,
                EPS,
            ),
        )
        .unwrap();
        assert_close(&hidden, &hidden_reference, 0.02);
        assert_close(
            &normalized,
            &rms_weighted(&hidden_reference, &weight_values, rows, cols, EPS),
            0.035,
        );

        let mut hidden = tensor_for(dtype, vec![rows, cols], &[0.0; 8]);
        let mut normalized = tensor_for(dtype, vec![rows, cols], &[0.0; 8]);
        bias_residual_layer_norm(
            &ctx,
            BiasResidualLayerNormArgs::new(
                &input,
                Some(&bias),
                &residual,
                &weight,
                &norm_bias,
                &mut hidden,
                &mut normalized,
                EPS,
            ),
        )
        .unwrap();
        assert_close(&hidden, &hidden_reference, 0.02);
        assert_close(
            &normalized,
            &layer_rows(
                &hidden_reference,
                &weight_values,
                &norm_bias_values,
                rows,
                cols,
                EPS,
            ),
            0.035,
        );

        let gated_reference =
            gated_residual(&input_values, &residual_values, &gate_style_values, cols);
        let mut hidden = tensor_for(dtype, vec![rows, cols], &[0.0; 8]);
        ada_gate_residual(
            &ctx,
            AdaGateResidualArgs::new(&input, &residual, &gate_style, &mut hidden),
        )
        .unwrap();
        assert_close(&hidden, &gated_reference, 0.02);

        let mut hidden = tensor_for(dtype, vec![rows, cols], &[0.0; 8]);
        let mut normalized = tensor_for(dtype, vec![rows, cols], &[0.0; 8]);
        ada_gate_residual_rms_norm(
            &ctx,
            AdaGateResidualRmsNormArgs::new(
                &input,
                &residual,
                &norm_style,
                &gate_style,
                &mut hidden,
                &mut normalized,
                EPS,
            ),
        )
        .unwrap();
        assert_close(&hidden, &gated_reference, 0.02);
        assert_close(
            &normalized,
            &adaptive_rms(&gated_reference, &norm_style_values, rows, cols, EPS),
            0.035,
        );
    }
}

#[test]
fn typed_norm_rejects_invalid_tensor_shapes() {
    let ctx = CudaContext::new(0).unwrap();
    let input = tensor(0, vec![1, 4], &[1.0; 4]);
    let weight = tensor(0, vec![4], &[1.0; 4]);
    let wrong_style = tensor(0, vec![4], &[0.0; 4]);
    let mut output = tensor(0, vec![1, 4], &[0.0; 4]);

    assert!(rms_norm(&ctx, RmsNormArgs::new(&input, &weight, &mut output, -1.0)).is_err());
    assert!(
        adaptive_rms_norm(
            &ctx,
            AdaptiveRmsNormArgs::new(&input, &wrong_style, &mut output, EPS),
        )
        .is_err(),
        "adaptive RMSNorm must reject a style tensor with the wrong typed shape"
    );
}

fn gelu_tanh(x: f32) -> f32 {
    0.5 * x * (1.0 + (0.797_884_6 * (x + 0.044_715 * x * x * x)).tanh())
}

fn activate(kind: PointwiseActivation, x: f32) -> f32 {
    match kind {
        PointwiseActivation::None => x,
        PointwiseActivation::Gelu => gelu_tanh(x),
        PointwiseActivation::Silu => x / (1.0 + (-x).exp()),
    }
}

#[test]
fn pointwise_all_semantics_and_activations_match_cpu_references() {
    let ctx = CudaContext::new(0).unwrap();

    let geglu_values = [-1.0, -0.5, 0.5, 1.5, 2.0, -2.0, 0.25, 4.0];
    let geglu_expected: Vec<_> = (0..4)
        .map(|column| gelu_tanh(geglu_values[column]) * geglu_values[column + 4])
        .collect();

    let input_values = [-2.0, -0.5, 0.25, 2.0, 1.0, -1.0, 0.5, 3.0];
    let bias_values = [0.5, -0.5, 1.0, -1.0];
    let state_values = [1.0, -2.0, 3.0, -4.0];
    let velocity_values = [2.0, 4.0, -1.0, -2.0];
    let euler_expected: Vec<_> = state_values
        .iter()
        .zip(velocity_values)
        .map(|(&state, velocity)| state - 0.25 * velocity)
        .collect();

    for dtype in [DType::BF16, DType::F16] {
        let geglu_input = tensor_for(dtype, vec![1, 8], &geglu_values);
        let mut geglu_out = tensor_for(dtype, vec![1, 4], &[0.0; 4]);
        pointwise(
            &ctx,
            PointwiseArgs::new(PointwiseSemantic::Geglu, &geglu_input, &mut geglu_out),
        )
        .unwrap();
        assert_close(&geglu_out, &geglu_expected, 0.025);

        let input = tensor_for(dtype, vec![2, 4], &input_values);
        let bias = tensor_for(dtype, vec![4], &bias_values);
        for activation in [
            PointwiseActivation::None,
            PointwiseActivation::Gelu,
            PointwiseActivation::Silu,
        ] {
            let mut out = tensor_for(dtype, vec![2, 4], &[0.0; 8]);
            let mut args = PointwiseArgs::new(PointwiseSemantic::BiasActivation, &input, &mut out);
            args.bias = Some(&bias);
            args.activation = activation;
            pointwise(&ctx, args).unwrap();
            let expected: Vec<_> = input_values
                .iter()
                .enumerate()
                .map(|(index, &x)| activate(activation, x + bias_values[index % 4]))
                .collect();
            assert_close(&out, &expected, 0.025);
        }

        let state = tensor_for(dtype, vec![2, 2], &state_values);
        let velocity = tensor_for(dtype, vec![2, 2], &velocity_values);
        let mut out = tensor_for(dtype, vec![2, 2], &[0.0; 4]);
        let mut args = PointwiseArgs::new(PointwiseSemantic::EulerUpdate, &state, &mut out);
        args.secondary = Some(&velocity);
        args.dt = -0.25;
        pointwise(&ctx, args).unwrap();
        assert_close(&out, &euler_expected, 0.01);
    }
}

#[test]
fn pointwise_rejects_missing_and_unrelated_bindings() {
    let ctx = CudaContext::new(0).unwrap();
    let input = tensor(0, vec![1, 2], &[1.0, 2.0]);
    let secondary = tensor(0, vec![1, 2], &[3.0, 4.0]);
    let mut out = tensor(0, vec![1, 2], &[0.0; 2]);

    let missing = PointwiseArgs::new(PointwiseSemantic::EulerUpdate, &input, &mut out);
    assert!(
        pointwise(&ctx, missing).is_err(),
        "Euler update requires velocity"
    );

    let mut unrelated = PointwiseArgs::new(PointwiseSemantic::BiasActivation, &input, &mut out);
    unrelated.secondary = Some(&secondary);
    unrelated.activation = PointwiseActivation::Silu;
    assert!(
        pointwise(&ctx, unrelated).is_err(),
        "bias-activation must reject an unrelated velocity"
    );
}

fn rotate_pair(x: f32, y: f32, position: usize) -> [f32; 2] {
    let angle = position as f32;
    let (sin, cos) = angle.sin_cos();
    [x * cos - y * sin, x * sin + y * cos]
}

#[test]
fn rope_split_rotate_and_bias_split_match_cpu_references() {
    let ctx = CudaContext::new(0).unwrap();

    // head_dim=2 makes the standard RoPE frequency exactly one, which keeps
    // this reference independent of any frequency-table implementation.
    let qkv_values = [
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, -1.0, 0.5, 2.0, -3.0, 4.0, -2.0, 0.25, 1.5,
    ];
    let mut expected_q = Vec::new();
    let mut expected_k = Vec::new();
    let mut expected_v = Vec::new();
    for token in 0..2 {
        let row = &qkv_values[token * 8..(token + 1) * 8];
        expected_q.extend(rotate_pair(row[0], row[1], token + 1));
        expected_q.extend(rotate_pair(row[2], row[3], token + 1));
        expected_k.extend(rotate_pair(row[4], row[5], token + 1));
        expected_v.extend_from_slice(&row[6..8]);
    }

    for dtype in [DType::BF16, DType::F16] {
        let qkv = tensor_for(dtype, vec![2, 8], &qkv_values);
        let mut q = tensor_for(dtype, vec![2, 2, 2], &[0.0; 8]);
        let mut k = tensor_for(dtype, vec![2, 1, 2], &[0.0; 4]);
        let mut v = tensor_for(dtype, vec![2, 1, 2], &[0.0; 4]);
        rope(
            &ctx,
            RopeArgs {
                semantic: RopeSemantic::SplitQkvRope,
                qkv: &qkv,
                bias: None,
                q: &mut q,
                k: &mut k,
                v: &mut v,
                q_heads: 2,
                kv_heads: 1,
                head_dim: 2,
                theta: 10_000.0,
                position_offset: 1,
                kv_output_offset: 0,
            },
        )
        .unwrap();
        assert_close(&q, &expected_q, 0.025);
        assert_close(&k, &expected_k, 0.025);
        assert_close(&v, &expected_v, 0.01);
    }

    let packed_values = [
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, -1.0, -2.0, -3.0, -4.0, -5.0, -6.0,
    ];
    let bias_values = [0.5, -0.5, 1.0, -1.0, 2.0, -2.0];
    let expected: Vec<_> = packed_values
        .iter()
        .enumerate()
        .map(|(index, &x)| x + bias_values[index % 6])
        .collect();
    for dtype in [DType::BF16, DType::F16] {
        let packed = tensor_for(dtype, vec![2, 6], &packed_values);
        let bias = tensor_for(dtype, vec![6], &bias_values);
        let mut q = tensor_for(dtype, vec![2, 1, 2], &[0.0; 4]);
        let mut k = tensor_for(dtype, vec![2, 1, 2], &[0.0; 4]);
        let mut v = tensor_for(dtype, vec![2, 1, 2], &[0.0; 4]);
        rope(
            &ctx,
            RopeArgs {
                semantic: RopeSemantic::SplitQkvBias,
                qkv: &packed,
                bias: Some(&bias),
                q: &mut q,
                k: &mut k,
                v: &mut v,
                q_heads: 1,
                kv_heads: 1,
                head_dim: 2,
                theta: 10_000.0,
                position_offset: 0,
                kv_output_offset: 0,
            },
        )
        .unwrap();
        assert_close(
            &q,
            &[expected[0], expected[1], expected[6], expected[7]],
            0.01,
        );
        assert_close(
            &k,
            &[expected[2], expected[3], expected[8], expected[9]],
            0.01,
        );
        assert_close(
            &v,
            &[expected[4], expected[5], expected[10], expected[11]],
            0.01,
        );
    }
}

#[test]
fn rope_rejects_illegal_or_unrelated_arguments() {
    let ctx = CudaContext::new(0).unwrap();
    let packed = tensor(0, vec![1, 6], &[0.0; 6]);
    let mut q = tensor(0, vec![1, 1, 2], &[0.0; 2]);
    let mut k = tensor(0, vec![1, 1, 2], &[0.0; 2]);
    let mut v = tensor(0, vec![1, 1, 2], &[0.0; 2]);
    let result = rope(
        &ctx,
        RopeArgs {
            semantic: RopeSemantic::SplitQkvBias,
            qkv: &packed,
            bias: None,
            q: &mut q,
            k: &mut k,
            v: &mut v,
            q_heads: 1,
            kv_heads: 1,
            head_dim: 2,
            theta: 10_000.0,
            position_offset: 1,
            kv_output_offset: 0,
        },
    );
    assert!(
        result.is_err(),
        "the non-rotary split must reject a rotary position offset"
    );
}

fn u32_buffer(values: &[u32]) -> CudaBuffer {
    let bytes: Vec<_> = values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect();
    let buffer = CudaBuffer::alloc(bytes.len(), 0).unwrap();
    buffer.copy_from_host(&bytes).unwrap();
    buffer
}

fn u8_buffer(values: &[u8]) -> CudaBuffer {
    let buffer = CudaBuffer::alloc(values.len(), 0).unwrap();
    buffer.copy_from_host(values).unwrap();
    buffer
}

#[test]
fn gather_all_semantics_match_cpu_references() {
    let ctx = CudaContext::new(0).unwrap();

    let table_values = [
        1.0, 2.0, 3.0, 4.0, -1.0, -2.0, -3.0, -4.0, 0.5, 1.5, -0.5, -1.5,
    ];
    let ids = u32_buffer(&[2, 0]);
    let embedding_expected: Vec<_> = table_values[8..12]
        .iter()
        .chain(&table_values[0..4])
        .map(|value| value * 2.0)
        .collect();
    for dtype in [DType::BF16, DType::F16] {
        let table = tensor_for(dtype, vec![3, 4], &table_values);
        let mut out = tensor_for(dtype, vec![2, 4], &[0.0; 8]);
        let mut args = GatherArgs::new(GatherSemantic::EmbeddingLookup, &table, &mut out);
        args.ids = Some(&ids);
        args.vocab_size = 3;
        gather(&ctx, args).unwrap();
        assert_close(&out, &embedding_expected, 0.01);
    }

    let projection_values = [
        1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0,
    ];
    let bias_values = [0.5, -0.5, 1.0];
    let position_values = [1.0, 2.0, 3.0, -1.0, -2.0, -3.0];
    let position_expected: Vec<_> = projection_values
        .iter()
        .enumerate()
        .map(|(index, &x)| {
            let row = index / 3;
            let col = index % 3;
            x + bias_values[col] + position_values[(row % 2) * 3 + col]
        })
        .collect();
    for dtype in [DType::BF16, DType::F16] {
        let projection = tensor_for(dtype, vec![4, 3], &projection_values);
        let bias = tensor_for(dtype, vec![3], &bias_values);
        let position = tensor_for(dtype, vec![2, 3], &position_values);
        let mut out = tensor_for(dtype, vec![4, 3], &[0.0; 12]);
        let mut args = GatherArgs::new(GatherSemantic::BiasPosition, &projection, &mut out);
        args.bias = Some(&bias);
        args.position = Some(&position);
        args.tokens_per_view = 2;
        gather(&ctx, args).unwrap();
        assert_close(&out, &position_expected, 0.02);
    }

    let nhwc = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110];
    let nchw = [0, 30, 60, 90, 10, 40, 70, 100, 20, 50, 80, 110];
    let expected_u8 = [0, 30, 60, 90, 10, 40, 70, 100, 20, 50, 80, 110];
    let expected: Vec<_> = expected_u8
        .iter()
        .map(|&value| value as f32 / 127.5 - 1.0)
        .collect();
    for dtype in [DType::BF16, DType::F16] {
        for (bytes, nhwc) in [(&nhwc[..], true), (&nchw[..], false)] {
            let images = u8_buffer(bytes);
            let mut out = tensor_for(dtype, vec![1, 12], &[0.0; 12]);
            let args = GatherArgs::rgb_to_patches(
                &images,
                &mut out,
                GatherPatchGeometry {
                    views: 1,
                    image_size: 2,
                    patch_size: 2,
                    nhwc,
                },
            );
            gather(&ctx, args).unwrap();
            assert_close(&out, &expected, 0.012);
        }
    }
}

#[test]
fn gather_rejects_missing_and_unrelated_bindings() {
    let ctx = CudaContext::new(0).unwrap();
    let input = tensor(0, vec![2, 2], &[1.0; 4]);
    let images = u8_buffer(&[0; 12]);
    let mut out = tensor(0, vec![1, 2], &[0.0; 2]);

    let mut missing = GatherArgs::new(GatherSemantic::EmbeddingLookup, &input, &mut out);
    missing.vocab_size = 2;
    assert!(
        gather(&ctx, missing).is_err(),
        "embedding lookup requires token ids"
    );

    let mut unrelated = GatherArgs::new(GatherSemantic::BiasPosition, &input, &mut out);
    unrelated.images = Some(&images);
    unrelated.position = Some(&input);
    unrelated.tokens_per_view = 1;
    assert!(
        gather(&ctx, unrelated).is_err(),
        "bias-position must reject raw-image storage"
    );
}

#[test]
fn stateless_pointwise_captures_without_a_prepare_cache() {
    let ctx = CudaContext::new(0).unwrap();
    let input = tensor(0, vec![1, 4], &[1.0, -2.0, 3.0, -4.0]);
    let velocity = tensor(0, vec![1, 4], &[2.0, 4.0, -1.0, -2.0]);
    let mut out = tensor(0, vec![1, 4], &[0.0; 4]);
    let graph = crate::capture(&ctx, || {
        let mut args = PointwiseArgs::new(PointwiseSemantic::EulerUpdate, &input, &mut out);
        args.secondary = Some(&velocity);
        args.dt = -0.25;
        pointwise(&ctx, args)
    })
    .unwrap();
    let output = CudaBuffer::from_tensor(&out).unwrap();
    let sentinel: Vec<_> = (0..4)
        .flat_map(|_| bf16::from_f32(-123.0).to_bits().to_ne_bytes())
        .collect();
    output.copy_from_host(&sentinel).unwrap();
    graph.replay().unwrap();
    ctx.synchronize().unwrap();
    assert_close(&out, &[0.5, -3.0, 3.25, -3.5], 0.0);
}
