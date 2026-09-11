//! Public-API integration tests from an external executor's point of view.
//!
//! Adding an L3 operator normally does not require editing this file. Extend it
//! only when the operator changes the public preparation, capture, replay,
//! stream, or lifetime contract exposed to downstream crates.

use apxinf_core::{DType, Shape, Tensor};
use apxinf_cuda::{
    capture,
    ops::{
        gemm, gemm_bias, prepare_with_workspace, with_workspace, GemmArgs, GemmBiasArgs,
        GraphWorkspace,
    },
    CudaBuffer, CudaContext,
};
use half::bf16;

fn bf16_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|&value| bf16::from_f32(value).to_bits().to_ne_bytes())
        .collect()
}

fn bf16_tensor(device: usize, shape: Vec<usize>, values: &[f32]) -> Tensor {
    let bytes = bf16_bytes(values);
    let buffer = CudaBuffer::alloc(bytes.len(), device).unwrap();
    buffer.copy_from_host(&bytes).unwrap();
    buffer.as_tensor(Shape::new(shape), DType::BF16).unwrap()
}

fn write_bf16(buffer: &CudaBuffer, values: &[f32]) {
    let bytes = bf16_bytes(values);
    assert_eq!(buffer.len(), bytes.len());
    buffer.copy_from_host(&bytes).unwrap();
}

fn buffer_values(buffer: &CudaBuffer) -> Vec<f32> {
    let mut bytes = vec![0; buffer.len()];
    buffer.copy_to_host(&mut bytes).unwrap();
    bytes
        .chunks_exact(2)
        .map(|bytes| bf16::from_bits(u16::from_ne_bytes([bytes[0], bytes[1]])).to_f32())
        .collect()
}

fn values(tensor: &Tensor) -> Vec<f32> {
    buffer_values(&CudaBuffer::from_tensor(tensor).unwrap())
}

fn assert_values(actual: &[f32], expected: &[f32]) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        assert!(actual.is_finite(), "element {index} is not finite");
        assert!(
            (actual - expected).abs() < 0.02,
            "element {index}: got {actual}, expected {expected}"
        );
    }
}

fn cpu_gemm(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f32> {
    let mut out = vec![0.0; m * n];
    for row in 0..m {
        for column in 0..n {
            out[row * n + column] = (0..k)
                .map(|inner| a[row * k + inner] * b[inner * n + column])
                .sum();
        }
    }
    out
}

fn run_gemm(
    ctx: &CudaContext,
    a: &Tensor,
    b: &Tensor,
    out: &mut Tensor,
) -> apxinf_core::Result<()> {
    let mut args = GemmArgs::new(a, b, out);
    args.policy.online_tune = false;
    gemm(ctx, args)
}

fn run_forward(
    ctx: &CudaContext,
    a: &Tensor,
    first_weight: &Tensor,
    intermediate: &mut Tensor,
    second_weight: &Tensor,
    bias: &Tensor,
    out: &mut Tensor,
) -> apxinf_core::Result<()> {
    run_gemm(ctx, a, first_weight, intermediate)?;
    let mut args = GemmArgs::new(intermediate, second_weight, out);
    args.policy.online_tune = false;
    gemm_bias(ctx, GemmBiasArgs { gemm: args, bias })
}

#[test]
fn same_public_l3_forward_prepares_captures_and_replays() {
    let ctx = CudaContext::new(0).unwrap();
    let first_weight_values = [1.0, 0.0, -1.0, 2.0, 0.5, -0.5];
    let second_weight_values = [1.0, -1.0, 0.5, 2.0];
    let bias_values = [0.25, -0.5];
    let initial_a = [1.0, 2.0, -1.0, 0.0, 3.0, 1.0];
    let eager_a = [2.0, -1.0, 0.5, -2.0, 1.0, 3.0];
    let replay_a = [-1.0, 0.5, 2.0, 3.0, -2.0, 1.0];
    let a = bf16_tensor(0, vec![2, 3], &initial_a);
    let first_weight = bf16_tensor(0, vec![3, 2], &first_weight_values);
    let mut intermediate = bf16_tensor(0, vec![2, 2], &[0.0; 4]);
    let second_weight = bf16_tensor(0, vec![2, 2], &second_weight_values);
    let bias = bf16_tensor(0, vec![2], &bias_values);
    let mut out = bf16_tensor(0, vec![2, 2], &[0.0; 4]);
    let a_buffer = CudaBuffer::from_tensor(&a).unwrap();
    let observed = CudaBuffer::from_tensor(&out).unwrap();
    let workspace = GraphWorkspace::new(4096, 0).unwrap();

    let expected = |a_values: &[f32]| {
        let intermediate = cpu_gemm(a_values, &first_weight_values, 2, 3, 2);
        let mut output = cpu_gemm(&intermediate, &second_weight_values, 2, 2, 2);
        for (index, value) in output.iter_mut().enumerate() {
            *value += bias_values[index % 2];
        }
        output
    };

    prepare_with_workspace(&workspace, || {
        run_forward(
            &ctx,
            &a,
            &first_weight,
            &mut intermediate,
            &second_weight,
            &bias,
            &mut out,
        )
    })
    .unwrap();

    write_bf16(&a_buffer, &eager_a);
    with_workspace(&workspace, || {
        run_forward(
            &ctx,
            &a,
            &first_weight,
            &mut intermediate,
            &second_weight,
            &bias,
            &mut out,
        )
    })
    .unwrap();
    assert_values(&values(&out), &expected(&eager_a));

    let graph = capture(&ctx, || {
        with_workspace(&workspace, || {
            run_forward(
                &ctx,
                &a,
                &first_weight,
                &mut intermediate,
                &second_weight,
                &bias,
                &mut out,
            )
        })
    })
    .unwrap();

    write_bf16(&a_buffer, &replay_a);
    write_bf16(&observed, &[-123.0; 4]);
    graph.replay().unwrap();
    ctx.synchronize().unwrap();

    assert_values(&buffer_values(&observed), &expected(&replay_a));
    drop(graph);
    drop(workspace);
}

#[test]
fn public_graph_retains_resources_and_reads_updated_inputs() {
    let ctx = CudaContext::new(0).unwrap();
    let b_values = [1.0, 2.0, 0.0, -1.0, 2.0, 1.0];
    let a = bf16_tensor(0, vec![2, 3], &[0.0; 6]);
    let b = bf16_tensor(0, vec![3, 2], &b_values);
    let mut out = bf16_tensor(0, vec![2, 2], &[0.0; 4]);
    let input = CudaBuffer::from_tensor(&a).unwrap();
    let observed = CudaBuffer::from_tensor(&out).unwrap();
    let workspace = GraphWorkspace::new(4096, 0).unwrap();

    prepare_with_workspace(&workspace, || run_gemm(&ctx, &a, &b, &mut out)).unwrap();
    let graph = capture(&ctx, || {
        with_workspace(&workspace, || run_gemm(&ctx, &a, &b, &mut out))
    })
    .unwrap();

    drop(workspace);
    drop(a);
    drop(b);
    drop(out);

    for a_values in [
        [1.0, 2.0, -1.0, 0.0, 3.0, 1.0],
        [2.0, -1.0, 0.5, -2.0, 1.0, 3.0],
    ] {
        write_bf16(&input, &a_values);
        write_bf16(&observed, &[-123.0; 4]);
        graph.replay().unwrap();
        ctx.synchronize().unwrap();
        assert_values(
            &buffer_values(&observed),
            &cpu_gemm(&a_values, &b_values, 2, 3, 2),
        );
    }
}

#[test]
fn public_graphs_keep_independent_bindings_and_specializations() {
    let ctx = CudaContext::new(0).unwrap();

    let a1 = bf16_tensor(0, vec![2, 3], &[0.0; 6]);
    let b1_values = [1.0, 2.0, 0.0, -1.0, 2.0, 1.0];
    let b1 = bf16_tensor(0, vec![3, 2], &b1_values);
    let mut out1 = bf16_tensor(0, vec![2, 2], &[0.0; 4]);
    let input1 = CudaBuffer::from_tensor(&a1).unwrap();
    let observed1 = CudaBuffer::from_tensor(&out1).unwrap();
    let workspace1 = GraphWorkspace::new(4096, 0).unwrap();
    prepare_with_workspace(&workspace1, || run_gemm(&ctx, &a1, &b1, &mut out1)).unwrap();
    let graph1 = capture(&ctx, || {
        with_workspace(&workspace1, || run_gemm(&ctx, &a1, &b1, &mut out1))
    })
    .unwrap();

    let a2 = bf16_tensor(0, vec![1, 2], &[0.0; 2]);
    let b2_values = [1.0, -1.0, 2.0, 0.5, 1.0, -2.0];
    let b2 = bf16_tensor(0, vec![2, 3], &b2_values);
    let mut out2 = bf16_tensor(0, vec![1, 3], &[0.0; 3]);
    let input2 = CudaBuffer::from_tensor(&a2).unwrap();
    let observed2 = CudaBuffer::from_tensor(&out2).unwrap();
    let workspace2 = GraphWorkspace::new(4096, 0).unwrap();
    prepare_with_workspace(&workspace2, || run_gemm(&ctx, &a2, &b2, &mut out2)).unwrap();
    let graph2 = capture(&ctx, || {
        with_workspace(&workspace2, || run_gemm(&ctx, &a2, &b2, &mut out2))
    })
    .unwrap();

    let a1_values = [1.0, 2.0, -1.0, 0.0, 3.0, 1.0];
    let a2_values = [2.0, -1.0];
    write_bf16(&input1, &a1_values);
    write_bf16(&input2, &a2_values);
    write_bf16(&observed1, &[-123.0; 4]);
    write_bf16(&observed2, &[-456.0; 3]);

    graph2.replay().unwrap();
    graph1.replay().unwrap();
    ctx.synchronize().unwrap();
    assert_values(
        &buffer_values(&observed1),
        &cpu_gemm(&a1_values, &b1_values, 2, 3, 2),
    );
    assert_values(
        &buffer_values(&observed2),
        &cpu_gemm(&a2_values, &b2_values, 1, 2, 3),
    );

    let changed_a1 = [-1.0, 0.5, 2.0, 3.0, -2.0, 1.0];
    write_bf16(&input1, &changed_a1);
    write_bf16(&observed1, &[-789.0; 4]);
    graph1.replay().unwrap();
    ctx.synchronize().unwrap();
    assert_values(
        &buffer_values(&observed1),
        &cpu_gemm(&changed_a1, &b1_values, 2, 3, 2),
    );
    assert_values(
        &buffer_values(&observed2),
        &cpu_gemm(&a2_values, &b2_values, 1, 2, 3),
    );
}

#[test]
fn public_l3_capture_rejects_cached_instance_from_another_stream() {
    let capture_ctx = CudaContext::new(0).unwrap();
    let execution_ctx = CudaContext::new(0).unwrap();
    let a = bf16_tensor(0, vec![2, 3], &[1.0; 6]);
    let b = bf16_tensor(0, vec![3, 4], &[1.0; 12]);
    let mut out = bf16_tensor(0, vec![2, 4], &[0.0; 8]);
    let workspace = GraphWorkspace::new(4096, 0).unwrap();

    prepare_with_workspace(&workspace, || run_gemm(&execution_ctx, &a, &b, &mut out)).unwrap();
    let error = match capture(&capture_ctx, || {
        with_workspace(&workspace, || run_gemm(&execution_ctx, &a, &b, &mut out))
    }) {
        Ok(_) => panic!("capture accepted an execution bound to another stream"),
        Err(error) => error,
    };
    assert!(
        error
            .to_string()
            .contains("multi-stream capture is not supported"),
        "unexpected error: {error}"
    );

    // A failed mismatched capture must not poison later capture on the bound
    // stream.
    let graph = capture(&execution_ctx, || {
        with_workspace(&workspace, || run_gemm(&execution_ctx, &a, &b, &mut out))
    })
    .unwrap();
    graph.replay().unwrap();
    execution_ctx.synchronize().unwrap();
    assert!(values(&out).iter().all(|&value| value == 3.0));
}

#[test]
fn public_l3_capture_rejects_an_unprepared_cache_miss() {
    let ctx = CudaContext::new(0).unwrap();
    let a = bf16_tensor(0, vec![2, 3], &[1.0; 6]);
    let b = bf16_tensor(0, vec![3, 4], &[1.0; 12]);
    let mut out = bf16_tensor(0, vec![2, 4], &[0.0; 8]);
    let workspace = GraphWorkspace::new(4096, 0).unwrap();

    let error = match capture(&ctx, || {
        with_workspace(&workspace, || run_gemm(&ctx, &a, &b, &mut out))
    }) {
        Ok(_) => panic!("capture unexpectedly prepared a GEMM instance"),
        Err(error) => error,
    };
    assert!(
        error.to_string().contains("cache miss during capture"),
        "unexpected error: {error}"
    );

    prepare_with_workspace(&workspace, || run_gemm(&ctx, &a, &b, &mut out)).unwrap();
    let graph = capture(&ctx, || {
        with_workspace(&workspace, || run_gemm(&ctx, &a, &b, &mut out))
    })
    .unwrap();
    let observed = CudaBuffer::from_tensor(&out).unwrap();
    write_bf16(&observed, &[-123.0; 8]);
    graph.replay().unwrap();
    ctx.synchronize().unwrap();
    assert!(values(&out).iter().all(|&value| value == 3.0));
}
