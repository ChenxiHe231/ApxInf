//! Precision contracts used by vision, through safe public backend APIs.
use apxinf_core::{DType, Tensor};
use apxinf_cuda_new::{ops, transfers, CudaContext, ExecutionSession, PreparedPhase};

fn device(ctx: &CudaContext, shape: Vec<usize>, values: &[f32]) -> Tensor {
    let data: Vec<_> = values.iter().map(|&x| half::bf16::from_f32(x)).collect();
    transfers::to_cuda(&Tensor::from_bf16(shape, &data).unwrap(), ctx.device_id()).unwrap()
}
fn read(x: &Tensor) -> Vec<f32> {
    transfers::to_cpu(x).unwrap().to_f32_vec().unwrap()
}

#[test]
fn ordered_pairwise_dot_preserves_small_cancellation_in_graph() {
    let ctx = CudaContext::new(0).unwrap();
    let a = device(&ctx, vec![2, 4], &[1.; 8]);
    let b = device(
        &ctx,
        vec![3, 4],
        &[
            1.,
            2f32.powi(-24),
            -1.,
            0.,
            2.,
            -2.,
            3.,
            -3.,
            1.,
            2f32.powi(-100),
            -1.,
            0.,
        ],
    );
    let bias = device(&ctx, vec![3], &[0., 0.5, 0.]);
    let output = device(&ctx, vec![2, 3], &[0.; 6]);
    let run = || {
        let mut out = output.clone();
        let mut gemm = ops::GemmArgs::new(&a, &b, &mut out).with_output_major_weight();
        gemm.policy.numerics = ops::GemmNumerics::PairwiseBf16Odd;
        gemm.policy.allow_fallback = false;
        ops::gemm_bias(&ctx, ops::GemmBiasArgs { gemm, bias: &bias })
    };
    let graph = PreparedPhase::prepare_and_capture(
        &ctx,
        ExecutionSession::with_capacity(1024 * 1024, 0).unwrap(),
        run,
    )
    .unwrap();
    graph.replay().unwrap();
    ctx.synchronize().unwrap();
    assert_eq!(
        read(&output),
        [
            2f32.powi(-23),
            0.5,
            2f32.powi(-23),
            2f32.powi(-23),
            0.5,
            2f32.powi(-23)
        ]
    );
    assert_eq!(read(&a), [1.; 8]);
    assert_eq!(read(&bias), [0., 0.5, 0.]);
    let mut invalid_out = output.clone();
    let mut invalid = ops::GemmArgs::new(&a, &b, &mut invalid_out);
    invalid.policy.numerics = ops::GemmNumerics::PairwiseBf16Odd;
    assert!(ops::gemm_bias(
        &ctx,
        ops::GemmBiasArgs {
            gemm: invalid,
            bias: &bias
        }
    )
    .is_err());
    // A two-lane dot with an odd reduction extent is outside this contract.
    let odd = device(&ctx, vec![2, 3], &[1.; 6]);
    let odd_weight = device(&ctx, vec![3, 3], &[1.; 9]);
    let mut invalid_out = output.clone();
    let mut invalid =
        ops::GemmArgs::new(&odd, &odd_weight, &mut invalid_out).with_output_major_weight();
    invalid.policy.numerics = ops::GemmNumerics::PairwiseBf16Odd;
    assert!(ops::gemm_bias(
        &ctx,
        ops::GemmBiasArgs {
            gemm: invalid,
            bias: &bias
        }
    )
    .is_err());
}

#[test]
fn ordered_pairwise_tiles_preserve_cancellation_tails_and_guards_in_graph() {
    use apxinf_core::Shape;
    use apxinf_cuda_new::CudaBuffer;

    let ctx = CudaContext::new(0).unwrap();
    // Cross both register-tile boundaries and the 64-element shared K tile.
    let (m, n, k) = (17, 35, 66);
    let input: Vec<f32> = (0..m)
        .flat_map(|row| vec![if row % 2 == 0 { 1. } else { -1. }; k])
        .collect();
    let mut weight = vec![0.; n * k];
    for column in 0..n {
        let base = (1 + column % 3) as f32;
        weight[column * k] = base;
        weight[column * k + 1] = 2f32.powi(-24);
        weight[column * k + 2] = -base;
        // This exact zero pair must be loaded from the partial final K tile.
        weight[column * k + 64] = 1.;
        weight[column * k + 65] = -1.;
    }
    let a = device(&ctx, vec![m, k], &input);
    let b = device(&ctx, vec![n, k], &weight);
    let bias = device(&ctx, vec![n], &vec![0.; n]);
    let guarded = device(&ctx, vec![m * n + 32], &vec![42.; m * n + 32]);
    let output = CudaBuffer::from_tensor(&guarded)
        .unwrap()
        .view(16 * 2, m * n * 2)
        .unwrap()
        .as_tensor(Shape::new(vec![m, n]), DType::BF16)
        .unwrap();
    let graph = PreparedPhase::prepare_and_capture(
        &ctx,
        ExecutionSession::with_capacity(1024 * 1024, 0).unwrap(),
        || {
            let mut out = output.clone();
            let mut gemm = ops::GemmArgs::new(&a, &b, &mut out).with_output_major_weight();
            gemm.policy.numerics = ops::GemmNumerics::PairwiseBf16Odd;
            gemm.policy.allow_fallback = false;
            ops::gemm_bias(&ctx, ops::GemmBiasArgs { gemm, bias: &bias })
        },
    )
    .unwrap();
    for _ in 0..3 {
        graph.replay().unwrap();
    }
    ctx.synchronize().unwrap();
    let expected: Vec<f32> = (0..m)
        .flat_map(|row| {
            (0..n).map(move |column| {
                let residual = 2f32.powi(if column % 3 == 0 { -23 } else { -22 });
                if row % 2 == 0 {
                    residual
                } else {
                    -residual
                }
            })
        })
        .collect();
    assert_eq!(read(&output), expected);
    let guards = read(&guarded);
    assert_eq!(&guards[..16], &[42.; 16]);
    assert_eq!(&guards[16 + m * n..], &[42.; 16]);
    assert_eq!(read(&a), input);
    assert_eq!(read(&b), weight);
    assert_eq!(read(&bias), vec![0.; n]);
}

#[test]
fn ordered_pairwise_flushes_products_and_handles_overflow_and_nan() {
    let ctx = CudaContext::new(0).unwrap();
    let tiny = 2f32.powi(-133); // Smallest positive BF16 subnormal.
    let normal = 2f32.powi(-126);
    let max = half::bf16::MAX.to_f32();
    let input = device(
        &ctx,
        vec![2, 4],
        &[1., 1., 1., 1., normal, normal, normal, normal],
    );
    let weights = device(
        &ctx,
        vec![6, 4],
        &[
            tiny,
            tiny,
            0.,
            0.,
            0.5,
            0.5,
            0.,
            0.,
            max,
            max,
            0.,
            0.,
            -max,
            -max,
            0.,
            0.,
            f32::INFINITY,
            f32::NEG_INFINITY,
            0.,
            0.,
            normal,
            -normal,
            0.,
            0.,
        ],
    );
    let bias = device(&ctx, vec![6], &[0.; 6]);
    let mut output = device(&ctx, vec![2, 6], &[42.; 12]);
    let mut gemm = ops::GemmArgs::new(&input, &weights, &mut output).with_output_major_weight();
    gemm.policy.numerics = ops::GemmNumerics::PairwiseBf16Odd;
    gemm.policy.allow_fallback = false;
    ops::gemm_bias(&ctx, ops::GemmBiasArgs { gemm, bias: &bias }).unwrap();
    ctx.synchronize().unwrap();
    let actual = read(&output);
    let expected = [
        0.,
        1.,
        f32::INFINITY,
        f32::NEG_INFINITY,
        f32::NAN,
        0.,
        0.,
        0.,
        7.96875,
        -7.96875,
        f32::NAN,
        0.,
    ];
    for (actual, expected) in actual.into_iter().zip(expected) {
        if expected.is_nan() {
            assert!(actual.is_nan());
        } else {
            assert_eq!(actual.to_bits(), expected.to_bits());
        }
    }
}

#[test]
fn transpose_padding_and_welford_norm_keep_inputs_and_graph_storage() {
    let ctx = CudaContext::new(0).unwrap();
    let input = device(
        &ctx,
        vec![2, 3, 2],
        &(1..=12).map(|x| x as f32).collect::<Vec<_>>(),
    );
    let packed = device(&ctx, vec![2, 2, 4], &[99.; 16]);
    let x = device(&ctx, vec![2, 8], &[0.; 16]);
    let gamma = device(&ctx, vec![8], &[2.; 8]);
    let beta = device(&ctx, vec![8], &[-1., 0., 1., 2., 3., 4., 5., 6.]);
    let normalized = device(&ctx, vec![2, 8], &[99.; 16]);
    let run = || {
        let mut out = packed.clone();
        ops::transpose_pad_channels(&ctx, &input, &mut out)?;
        let mut out = normalized.clone();
        let mut norm = ops::NormArgs::layer(&x, &gamma, &beta, &mut out, 1e-6);
        norm.numerics = ops::NormNumerics::Welford4F32;
        ops::norm::norm(&ctx, norm)
    };
    let graph = PreparedPhase::prepare_and_capture(
        &ctx,
        ExecutionSession::with_capacity(1024 * 1024, 0).unwrap(),
        run,
    )
    .unwrap();
    for _ in 0..2 {
        graph.replay().unwrap();
    }
    ctx.synchronize().unwrap();
    assert_eq!(
        read(&packed),
        [1., 3., 5., 0., 2., 4., 6., 0., 7., 9., 11., 0., 8., 10., 12., 0.]
    );
    assert_eq!(
        read(&normalized),
        [-1., 0., 1., 2., 3., 4., 5., 6., -1., 0., 1., 2., 3., 4., 5., 6.]
    );
    assert_eq!(read(&x), [0.; 16]);
    assert_eq!(read(&input), (1..=12).map(|x| x as f32).collect::<Vec<_>>());
    let mut alias = input.reshape(vec![2, 2, 3]).unwrap();
    assert!(ops::transpose_pad_channels(&ctx, &input, &mut alias).is_err());
    let bad = device(&ctx, vec![2, 7], &[1.; 14]);
    let mut bad_out = bad.clone();
    let mut args = ops::NormArgs::layer(&bad, &gamma, &beta, &mut bad_out, 1e-6);
    args.numerics = ops::NormNumerics::Welford4F32;
    assert!(ops::norm::norm(&ctx, args).is_err());
    assert_eq!(packed.dtype(), DType::BF16);
}

#[test]
fn staged_exp2_gelu_matches_pinned_bf16_cpu_values() {
    // PyTorch 2.14 CPU, oneDNN 3.12 SVE128, gelu(approximate="tanh").
    let values = [
        -8., -4., -3., -2., -1.5, -1., -0.5, -0.125, -0.015625, 0., 0.015625, 0.125, 0.5, 1., 1.5,
        2., 3., 4., 8.,
    ];
    let expected = [
        0.,
        -0.00007009506225585938,
        -0.003631591796875,
        -0.04541015625,
        -0.1005859375,
        -0.1591796875,
        -0.154296875,
        -0.056396484375,
        -0.007720947265625,
        0.,
        0.0079345703125,
        0.06884765625,
        0.345703125,
        0.83984375,
        1.3984375,
        1.953125,
        3.,
        4.,
        8.,
    ];
    let ctx = CudaContext::new(0).unwrap();
    let input = device(&ctx, vec![1, values.len()], &values);
    let output = device(&ctx, vec![1, values.len()], &vec![0.; values.len()]);
    let run = || {
        let mut out = output.clone();
        let mut args =
            ops::PointwiseArgs::new(ops::PointwiseSemantic::BiasActivation, &input, &mut out);
        args.activation = ops::PointwiseActivation::GeluTanhExp2;
        ops::pointwise(&ctx, args)
    };
    let graph = PreparedPhase::prepare_and_capture(
        &ctx,
        ExecutionSession::with_capacity(1024 * 1024, 0).unwrap(),
        run,
    )
    .unwrap();
    graph.replay().unwrap();
    ctx.synchronize().unwrap();
    assert_eq!(read(&output), expected);
    assert_eq!(read(&input), values);
}

#[test]
fn ordered_wide_projection_keeps_cancellation_partial_tiles_and_graph_guards() {
    use apxinf_core::Shape;
    use apxinf_cuda_new::CudaBuffer;

    let ctx = CudaContext::new_with_autotune(0, false).unwrap();
    // Partial 32x64 output tiles and a partial 64-element K tile, with
    // cancellation examples whose expected results depend on round-to-odd.
    let (m, n, k) = (129, 1025, 1026);
    let a = device(&ctx, vec![m, k], &vec![1.; m * k]);
    let mut weights = vec![0.; n * k];
    let mut bias_values = vec![0.; n];
    let mut expected = vec![0.; m * n];
    for c in 0..n {
        let row = &mut weights[c * k..(c + 1) * k];
        let value = if c % 3 == 1 {
            row[62..66].copy_from_slice(&[2., -2., 3., -3.]);
            bias_values[c] = 0.5;
            0.5
        } else {
            row[..4].copy_from_slice(&[
                1.,
                2f32.powi(if c % 3 == 0 { -24 } else { -100 }),
                -1.,
                0.,
            ]);
            2f32.powi(-23)
        };
        for r in 0..m {
            expected[r * n + c] = value;
        }
    }
    let b = device(&ctx, vec![n, k], &weights);
    let bias = device(&ctx, vec![n], &bias_values);
    let guarded = device(&ctx, vec![m * n + 32], &vec![42.; m * n + 32]);
    let output = CudaBuffer::from_tensor(&guarded)
        .unwrap()
        .view(32, m * n * 2)
        .unwrap()
        .as_tensor(Shape::new(vec![m, n]), DType::BF16)
        .unwrap();
    let phase = PreparedPhase::prepare_and_capture(
        &ctx,
        ExecutionSession::with_capacity(1024 * 1024, 0).unwrap(),
        || {
            let mut out = output.clone();
            let mut gemm = ops::GemmArgs::new(&a, &b, &mut out).with_output_major_weight();
            gemm.policy.numerics = ops::GemmNumerics::PairwiseBf16Odd;
            gemm.policy.online_tune = false;
            gemm.policy.cache_dir = Some(String::new());
            ops::gemm_bias(&ctx, ops::GemmBiasArgs { gemm, bias: &bias })
        },
    )
    .unwrap();
    for _ in 0..3 {
        phase.replay().unwrap();
    }
    ctx.synchronize().unwrap();
    assert_eq!(read(&output), expected);
    assert_eq!(read(&a), vec![1.; m * k]);
    assert_eq!(read(&b), weights);
    assert_eq!(read(&bias), bias_values);
    let guards = read(&guarded);
    assert_eq!(&guards[..16], &[42.; 16]);
    assert_eq!(&guards[m * n + 16..], &[42.; 16]);
}
