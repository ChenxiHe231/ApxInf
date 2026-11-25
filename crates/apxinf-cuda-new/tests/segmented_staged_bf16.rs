//! Explicit BF16 attention boundaries, image independence, and prepared/Graph replay.
use apxinf_core::Tensor;
use apxinf_cuda_new::{ops, transfers, CudaBuffer, CudaContext, ExecutionSession, PreparedPhase};

fn round(value: f32) -> f32 {
    half::bf16::from_f32(value).to_f32()
}
fn upload(ctx: &CudaContext, values: &[f32], shape: Vec<usize>) -> Tensor {
    transfers::to_cuda(
        &Tensor::from_bf16(
            shape,
            &values
                .iter()
                .map(|&v| half::bf16::from_f32(v))
                .collect::<Vec<_>>(),
        )
        .unwrap(),
        ctx.device_id(),
    )
    .unwrap()
}
fn read(tensor: &Tensor) -> Vec<f32> {
    transfers::to_cpu(tensor).unwrap().to_f32_vec().unwrap()
}

#[test]
fn staged_attention_matches_explicit_bf16_graph_without_cross_image_attention() {
    let ctx = CudaContext::new(0).unwrap();
    let (tokens, heads, dim) = (8, 2, 72);
    let make = |shift: usize| {
        (0..tokens * heads * dim)
            .map(|i| round((((i * 17 + shift) % 127) as f32 - 63.) / 37.))
            .collect::<Vec<_>>()
    };
    let (q, k, v) = (make(3), make(17), make(31));
    let query = upload(&ctx, &q, vec![tokens, heads, dim]);
    let key = upload(&ctx, &k, vec![tokens, heads, dim]);
    let value = upload(&ctx, &v, vec![tokens, heads, dim]);
    let output = upload(&ctx, &vec![0.; q.len()], vec![tokens, heads, dim]);
    let offsets = [0u32, 3, 8];
    let buffer = CudaBuffer::alloc(offsets.len() * 4, ctx.device_id()).unwrap();
    buffer
        .copy_from_host(
            &offsets
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap();
    let run = || {
        let mut out = output.clone();
        let mut args =
            ops::SegmentedAttentionArgs::new(&query, &key, &value, &mut out, &buffer, &offsets);
        args.numerics = ops::AttentionNumerics::StagedBf16;
        // This numerical implementation is a valid candidate with fallback disabled.
        args.policy.allow_fallback = false;
        ops::segmented_attention(&ctx, args)
    };
    let phase = PreparedPhase::prepare_and_capture(
        &ctx,
        ExecutionSession::with_capacity(4 * 1024 * 1024, 0).unwrap(),
        run,
    )
    .unwrap();
    phase.replay().unwrap();
    ctx.synchronize().unwrap();
    let got = read(&output);
    let mut expected = vec![0.; got.len()];
    for segment in offsets.windows(2) {
        let (begin, end) = (segment[0] as usize, segment[1] as usize);
        for i in begin..end {
            for h in 0..heads {
                let mut scores = (begin..end)
                    .map(|j| {
                        let dot = (0..dim)
                            .map(|d| q[(i * heads + h) * dim + d] * k[(j * heads + h) * dim + d])
                            .sum::<f32>();
                        round(round(dot) / (dim as f32).sqrt())
                    })
                    .collect::<Vec<_>>();
                let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                for s in &mut scores {
                    *s = (*s - max).exp();
                }
                let sum = scores.iter().sum::<f32>();
                for s in &mut scores {
                    *s = round(*s / sum);
                }
                for d in 0..dim {
                    expected[(i * heads + h) * dim + d] = round(
                        (begin..end)
                            .map(|j| scores[j - begin] * v[(j * heads + h) * dim + d])
                            .sum(),
                    );
                }
            }
        }
    }
    let max_error = got
        .iter()
        .zip(&expected)
        .map(|(a, b)| (a - b).abs())
        .fold(0., f32::max);
    assert!(
        max_error <= 0.0078125,
        "BF16 reference max error {max_error}"
    );
    assert_eq!(read(&query), q);
    assert_eq!(read(&key), k);
    assert_eq!(read(&value), v);
    // Change only image two; image one's rows must be bitwise unchanged.
    let mut altered = v.clone();
    altered[3 * heads * dim..].fill(100.);
    let replacement = upload(&ctx, &altered, vec![tokens, heads, dim]);
    ops::copy_tensor(&ctx, &replacement, &value).unwrap();
    phase.replay().unwrap();
    ctx.synchronize().unwrap();
    let changed = read(&output);
    assert_eq!(&changed[..3 * heads * dim], &got[..3 * heads * dim]);
    assert!(changed[3 * heads * dim..]
        .iter()
        .all(|x| (*x - 100.).abs() <= 0.5));
}

#[test]
fn staged_parallel_softmax_keeps_graph_guards_and_images_independent() {
    use apxinf_core::{DType, Shape};

    let ctx = CudaContext::new(0).unwrap();
    let (tokens, heads, dim) = (192, 2, 72);
    let zeros = vec![0.; tokens * heads * dim];
    let values: Vec<f32> = (0..tokens)
        .flat_map(|row| {
            (0..heads * dim).map(move |column| {
                if row < 64 {
                    2. + (column % 9) as f32 / 8.
                } else {
                    -3. + (column % 5) as f32 / 4.
                }
            })
        })
        .collect();
    let query = upload(&ctx, &zeros, vec![tokens, heads, dim]);
    let key = upload(&ctx, &zeros, vec![tokens, heads, dim]);
    let value = upload(&ctx, &values, vec![tokens, heads, dim]);
    let guarded = upload(&ctx, &vec![42.; values.len() + 32], vec![values.len() + 32]);
    let output = CudaBuffer::from_tensor(&guarded)
        .unwrap()
        .view(16 * 2, values.len() * 2)
        .unwrap()
        .as_tensor(Shape::new(vec![tokens, heads, dim]), DType::BF16)
        .unwrap();
    let offsets = [0u32, 64, 192];
    let buffer = CudaBuffer::alloc(12, ctx.device_id()).unwrap();
    buffer
        .copy_from_host(
            &offsets
                .iter()
                .flat_map(|x| x.to_le_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap();
    let phase = PreparedPhase::prepare_and_capture(
        &ctx,
        ExecutionSession::with_capacity(4 * 1024 * 1024, 0).unwrap(),
        || {
            let mut out = output.clone();
            let mut args =
                ops::SegmentedAttentionArgs::new(&query, &key, &value, &mut out, &buffer, &offsets);
            args.numerics = ops::AttentionNumerics::StagedBf16;
            args.policy.allow_fallback = false;
            ops::segmented_attention(&ctx, args)
        },
    )
    .unwrap();
    for _ in 0..3 {
        phase.replay().unwrap();
    }
    ctx.synchronize().unwrap();
    // Zero scores give exact 1/64 and 1/128 probabilities. Each segment is
    // constant along its token axis, so its expected output is independent.
    assert_eq!(read(&output), values);
    assert_eq!(read(&query), zeros);
    assert_eq!(read(&key), zeros);
    assert_eq!(read(&value), values);
    let guards = read(&guarded);
    assert_eq!(&guards[..16], &[42.; 16]);
    assert_eq!(&guards[16 + values.len()..], &[42.; 16]);

    let mut altered = values.clone();
    altered[64 * heads * dim..].fill(100.);
    let replacement = upload(&ctx, &altered, vec![tokens, heads, dim]);
    ops::copy_tensor(&ctx, &replacement, &value).unwrap();
    phase.replay().unwrap();
    ctx.synchronize().unwrap();
    let changed = read(&output);
    assert_eq!(&changed[..64 * heads * dim], &values[..64 * heads * dim]);
    assert!(changed[64 * heads * dim..].iter().all(|x| *x == 100.));
}

#[test]
fn staged_vision_positions_match_explicit_four_tap_rounding() {
    let ctx = CudaContext::new(0).unwrap();
    let (side, height, width, hidden) = (48usize, 12usize, 20usize, 7usize);
    let table_values = (0..side * side * hidden)
        .map(|i| round((((i * 29) % 211) as f32 - 105.) / 53.))
        .collect::<Vec<_>>();
    let table = upload(&ctx, &table_values, vec![side * side, hidden]);
    let embeddings = upload(
        &ctx,
        &vec![0.; height * width * hidden],
        vec![height * width, hidden],
    );
    let ids = CudaBuffer::alloc(height * width * 2 * 4, 0)
        .unwrap()
        .as_tensor(
            apxinf_core::Shape::new(vec![height * width, 2]),
            apxinf_core::DType::I32,
        )
        .unwrap();
    let run = || {
        let (mut out, mut positions) = (embeddings.clone(), ids.clone());
        ops::vision_position_staged(
            &ctx,
            ops::VisionPositionArgs {
                table: &table,
                embeddings: &mut out,
                position_ids: &mut positions,
                height,
                width,
                merge_size: 2,
            },
        )
    };
    let phase = PreparedPhase::prepare_and_capture(
        &ctx,
        ExecutionSession::with_capacity(1024 * 1024, 0).unwrap(),
        run,
    )
    .unwrap();
    phase.replay().unwrap();
    ctx.synchronize().unwrap();
    let mut expected = Vec::new();
    for token in 0..height * width {
        let (group, inner) = (token / 4, token % 4);
        let (r, c) = (
            group / (width / 2) * 2 + inner / 2,
            group % (width / 2) * 2 + inner % 2,
        );
        let axis = |index: usize, size: usize| {
            let x = index as f32 * (side - 1) as f32 / (size - 1) as f32;
            let floor = x.floor() as usize;
            let fraction = x - floor as f32;
            (
                floor,
                (floor + 1).min(side - 1),
                1. - fraction,
                1. - (fraction - 1.).abs(),
            )
        };
        let (r0, r1, wr0, wr1) = axis(r, height);
        let (c0, c1, wc0, wc1) = axis(c, width);
        for d in 0..hidden {
            let parts = [
                wr0 * wc0 * table_values[(r0 * side + c0) * hidden + d],
                wr0 * wc1 * table_values[(r0 * side + c1) * hidden + d],
                wr1 * wc0 * table_values[(r1 * side + c0) * hidden + d],
                wr1 * wc1 * table_values[(r1 * side + c1) * hidden + d],
            ];
            expected.push(round(((parts[0] + parts[1]) + parts[2]) + parts[3]));
        }
    }
    assert_eq!(read(&embeddings), expected);
    assert_eq!(read(&table), table_values);
}
