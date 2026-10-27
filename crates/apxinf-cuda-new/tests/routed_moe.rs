//! CUDA routed-MoE primitive against the model-neutral CPU reference.

use std::time::Instant;

use apxinf_core::{DType, Shape, Tensor};
use apxinf_cuda_new::{
    ops, transfers, CudaBuffer, CudaContext, RoutePlan, RoutedMoeCombineArgs, RoutedMoeConfig,
    RoutedMoeRouteArgs, RoutedMoeRouteTopkArgs, SharedExpertOutput,
};
use half::bf16;

fn to_cuda(ctx: &CudaContext, tensor: &Tensor) -> Tensor {
    transfers::to_cuda(tensor, ctx.device_id()).unwrap()
}

fn zeros(ctx: &CudaContext, shape: Vec<usize>, dtype: DType) -> Tensor {
    let bytes = shape.iter().product::<usize>() * dtype.size_in_bytes();
    CudaBuffer::alloc_zeros(bytes, ctx.device_id())
        .unwrap()
        .as_tensor(Shape::new(shape), dtype)
        .unwrap()
}

fn read_i32(tensor: &Tensor) -> Vec<i32> {
    transfers::to_cpu(tensor)
        .unwrap()
        .as_i32()
        .unwrap()
        .to_vec()
}

fn read_f32(tensor: &Tensor) -> Vec<f32> {
    transfers::to_cpu(tensor)
        .unwrap()
        .as_f32()
        .unwrap()
        .to_vec()
}

fn read_bf16(tensor: &Tensor) -> Vec<f32> {
    transfers::to_cpu(tensor)
        .unwrap()
        .as_bf16()
        .unwrap()
        .iter()
        .map(|value| value.to_f32())
        .collect()
}

#[test]
fn qwen35_top8_assign_and_shared_combine_match_cpu_reference() {
    let ctx = CudaContext::new(0).unwrap();
    let config = RoutedMoeConfig::new(256, 8);
    let tokens = 2usize;
    let routes = tokens * config.top_k;
    let hidden = 4usize;

    let mut logits = vec![-100.0f32; tokens * config.experts];
    for (token, winners) in [
        [
            (91, 4.0),
            (7, 7.0),
            (2, 7.0),
            (88, 3.0),
            (12, 6.0),
            (255, 2.0),
            (0, 5.0),
            (31, 1.0),
        ],
        [
            (9, 4.0),
            (17, 7.0),
            (3, 7.0),
            (188, 3.0),
            (112, 6.0),
            (254, 2.0),
            (1, 5.0),
            (131, 1.0),
        ],
    ]
    .into_iter()
    .enumerate()
    {
        for (expert, score) in winners {
            logits[token * config.experts + expert] = score;
        }
    }
    let reference = ops::route_topk_reference(config, &logits, tokens).unwrap();
    let logits = to_cuda(
        &ctx,
        &Tensor::from_f32(vec![tokens, config.experts], &logits).unwrap(),
    );
    let mut expert_ids = zeros(&ctx, vec![tokens, config.top_k], DType::I32);
    let mut expert_weights = zeros(&ctx, vec![tokens, config.top_k], DType::F32);
    let mut assignment_expert_ids = zeros(&ctx, vec![routes], DType::I32);
    let mut assignment_token_ids = zeros(&ctx, vec![routes], DType::I32);
    let mut assignment_slots = zeros(&ctx, vec![routes], DType::I32);
    let mut token_assignment_indices = zeros(&ctx, vec![routes], DType::I32);
    let mut assignment_weights = zeros(&ctx, vec![routes], DType::F32);
    let mut expert_offsets = zeros(&ctx, vec![config.experts + 1], DType::I32);

    ops::routed_moe_route(
        &ctx,
        RoutedMoeRouteArgs {
            config,
            logits: &logits,
            expert_ids: &mut expert_ids,
            expert_weights: &mut expert_weights,
            assignment_expert_ids: &mut assignment_expert_ids,
            assignment_token_ids: &mut assignment_token_ids,
            assignment_slots: &mut assignment_slots,
            token_assignment_indices: &mut token_assignment_indices,
            assignment_weights: &mut assignment_weights,
            expert_offsets: &mut expert_offsets,
        },
    )
    .unwrap();
    ctx.synchronize().unwrap();

    let ids = read_i32(&expert_ids);
    assert_eq!(
        ids,
        reference
            .expert_ids()
            .iter()
            .map(|value| *value as i32)
            .collect::<Vec<_>>()
    );
    let token_weights = read_f32(&expert_weights);
    for (actual, expected) in token_weights.iter().zip(reference.expert_weights()) {
        assert!((actual - expected).abs() < 2e-6, "{actual} != {expected}");
    }
    if let Ok(iterations) = std::env::var("APXINF_ROUTE_TOPK_BENCH_ITERS") {
        let iterations: usize = iterations.parse().unwrap();
        ctx.synchronize().unwrap();
        let start = Instant::now();
        for _ in 0..iterations {
            ops::routed_moe_route_topk(
                &ctx,
                RoutedMoeRouteTopkArgs {
                    config,
                    logits: &logits,
                    expert_ids: &mut expert_ids,
                    expert_weights: &mut expert_weights,
                },
            )
            .unwrap();
        }
        ctx.synchronize().unwrap();
        println!(
            "Qwen3.5 route top-k tokens={tokens}: {:?}/iteration",
            start.elapsed() / iterations as u32
        );
    }
    let assignments = reference.assignments();
    assert_eq!(
        read_i32(&assignment_expert_ids),
        assignments
            .iter()
            .map(|assignment| assignment.expert_id as i32)
            .collect::<Vec<_>>()
    );
    assert_eq!(
        read_i32(&assignment_token_ids),
        assignments
            .iter()
            .map(|assignment| assignment.token_id as i32)
            .collect::<Vec<_>>()
    );
    assert_eq!(
        read_i32(&assignment_slots),
        assignments
            .iter()
            .map(|assignment| assignment.route_slot as i32)
            .collect::<Vec<_>>()
    );
    assert_eq!(
        read_i32(&expert_offsets),
        reference
            .expert_offsets()
            .iter()
            .map(|value| *value as i32)
            .collect::<Vec<_>>()
    );

    let assignment_values = (0..routes * hidden)
        .map(|index| bf16::from_f32((index as f32 - 17.0) / 9.0))
        .collect::<Vec<_>>();
    let shared_values = (0..tokens * hidden)
        .map(|index| bf16::from_f32((index as f32 + 1.0) / 5.0))
        .collect::<Vec<_>>();
    let gate_values = [bf16::from_f32(0.0), bf16::from_f32(3.0f32.ln())];
    let assignment_output = to_cuda(
        &ctx,
        &Tensor::from_bf16(vec![routes, hidden], &assignment_values).unwrap(),
    );
    let shared_output = to_cuda(
        &ctx,
        &Tensor::from_bf16(vec![tokens, hidden], &shared_values).unwrap(),
    );
    let shared_gate_logits = to_cuda(
        &ctx,
        &Tensor::from_bf16(vec![tokens], &gate_values).unwrap(),
    );
    let mut output = zeros(&ctx, vec![tokens, hidden], DType::BF16);
    ops::routed_moe_combine(
        &ctx,
        RoutedMoeCombineArgs {
            assignment_output: &assignment_output,
            assignment_token_ids: &assignment_token_ids,
            assignment_weights: &assignment_weights,
            token_assignment_indices: &token_assignment_indices,
            shared_output: &shared_output,
            shared_gate_logits: &shared_gate_logits,
            output: &mut output,
        },
    )
    .unwrap();
    ctx.synchronize().unwrap();

    let gpu_plan = RoutePlan::from_token_major(
        config,
        tokens,
        ids.into_iter().map(|value| value as u32).collect(),
        token_weights,
    )
    .unwrap();
    let assignment_values = assignment_values
        .iter()
        .map(|value| value.to_f32())
        .collect::<Vec<_>>();
    let shared_values = shared_values
        .iter()
        .map(|value| value.to_f32())
        .collect::<Vec<_>>();
    let gate_values = gate_values
        .iter()
        .map(|value| value.to_f32())
        .collect::<Vec<_>>();
    let expected = ops::routed_moe_combine_reference(
        &gpu_plan,
        &assignment_values,
        hidden,
        Some(SharedExpertOutput {
            output: &shared_values,
            gate_logits: &gate_values,
        }),
    )
    .unwrap();
    for (actual, expected) in read_bf16(&output).into_iter().zip(expected) {
        let rounded = bf16::from_f32(expected).to_f32();
        let tolerance = 0.01 * rounded.abs().max(1.0);
        assert!(
            (actual - rounded).abs() <= tolerance,
            "{actual} != {rounded} within {tolerance}"
        );
    }
}
