//! Qwen3.5-sized routed NVFP4 expert correctness on a CUDA device.

use std::time::Instant;

use apxinf_core::{DType, Shape, Tensor};
use apxinf_cuda_new::{
    ops, transfers, CudaBuffer, CudaContext, Nvfp4ExpertProjection, RoutePlan,
    RoutedMoeCombineArgs, RoutedMoeConfig, RoutedMoeNvfp4ExpertsArgs, RoutedMoeRouteArgs,
    SharedExpertOutput,
};
use half::bf16;

const EXPERTS: usize = 256;
const TOP_K: usize = 8;
const HIDDEN: usize = 2048;
const INTERMEDIATE: usize = 512;
const BLOCK: usize = 16;

fn zeros(ctx: &CudaContext, shape: Vec<usize>, dtype: DType) -> Tensor {
    let bytes = shape.iter().product::<usize>() * dtype.size_in_bytes();
    CudaBuffer::alloc_zeros(bytes, ctx.device_id())
        .unwrap()
        .as_tensor(Shape::new(shape), dtype)
        .unwrap()
}

fn upload(ctx: &CudaContext, tensor: &Tensor) -> Tensor {
    transfers::to_cuda(tensor, ctx.device_id()).unwrap()
}

fn upload_bytes(ctx: &CudaContext, bytes: &[u8], shape: Vec<usize>, dtype: DType) -> Tensor {
    let buffer = CudaBuffer::alloc(bytes.len(), ctx.device_id()).unwrap();
    buffer.copy_from_host(bytes).unwrap();
    buffer.as_tensor(Shape::new(shape), dtype).unwrap()
}

fn stacked_packed(
    ctx: &CudaContext,
    rows: usize,
    inputs: usize,
    code: impl Fn(usize, usize) -> u8,
) -> Tensor {
    let expert_bytes = rows * inputs / 2;
    let buffer = CudaBuffer::alloc(EXPERTS * expert_bytes, ctx.device_id()).unwrap();
    let mut staging = vec![0u8; expert_bytes];
    for expert in 0..EXPERTS {
        for (row, packed_row) in staging.chunks_exact_mut(inputs / 2).enumerate() {
            let nibble = code(expert, row) & 0x0f;
            packed_row.fill(nibble | (nibble << 4));
        }
        buffer
            .view(expert * expert_bytes, expert_bytes)
            .unwrap()
            .copy_from_host(&staging)
            .unwrap();
    }
    buffer
        .as_tensor(Shape::new(vec![EXPERTS, rows, inputs / 2]), DType::E2M1Pair)
        .unwrap()
}

fn stacked_unit_scales(ctx: &CudaContext, rows: usize, inputs: usize) -> Tensor {
    let expert_bytes = rows * inputs / BLOCK;
    let buffer = CudaBuffer::alloc(EXPERTS * expert_bytes, ctx.device_id()).unwrap();
    let staging = vec![0x38u8; expert_bytes]; // E4M3 1.0
    for expert in 0..EXPERTS {
        buffer
            .view(expert * expert_bytes, expert_bytes)
            .unwrap()
            .copy_from_host(&staging)
            .unwrap();
    }
    buffer
        .as_tensor(
            Shape::new(vec![EXPERTS, rows, inputs / BLOCK]),
            DType::F8E4M3,
        )
        .unwrap()
}

fn expert_alphas(ctx: &CudaContext, base: f32, step: f32) -> Tensor {
    let values = (0..EXPERTS)
        .map(|expert| base + step * (expert % 5) as f32)
        .collect::<Vec<_>>();
    upload(ctx, &Tensor::from_f32(vec![EXPERTS], &values).unwrap())
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

fn decode_e2m1(code: u8) -> f32 {
    let magnitude = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0][(code & 7) as usize];
    if code & 8 == 0 {
        magnitude
    } else {
        -magnitude
    }
}

fn decode_e4m3(code: u8) -> f32 {
    let code = code & 0x7f;
    let exponent = (code >> 3) & 0x0f;
    let fraction = (code & 7) as f32;
    if exponent == 0 {
        fraction * 0.001953125
    } else {
        (1.0 + fraction * 0.125) * 2.0f32.powi(exponent as i32 - 7)
    }
}

fn constant_projection(
    input: &[f32],
    outputs: usize,
    code: impl Fn(usize) -> u8,
    scale: u8,
    alpha: f32,
) -> Vec<f32> {
    let mut result = vec![0.0f32; outputs];
    for (row, output) in result.iter_mut().enumerate() {
        let weight = decode_e2m1(code(row)) * decode_e4m3(scale) * alpha;
        for activation in input {
            *output += activation * weight;
        }
        *output = bf16::from_f32(*output).to_f32();
    }
    result
}

fn cpu_expert(
    input: &[f32],
    expert: usize,
    shared: bool,
    gate_alpha: f32,
    up_alpha: f32,
    down_alpha: f32,
) -> Vec<f32> {
    let gate = constant_projection(
        input,
        INTERMEDIATE,
        |row| {
            if shared {
                shared_gate_code(row)
            } else {
                expert_gate_code(expert, row)
            }
        },
        0x38,
        gate_alpha,
    );
    let up = constant_projection(
        input,
        INTERMEDIATE,
        |row| {
            if shared {
                shared_up_code(row)
            } else {
                expert_up_code(expert, row)
            }
        },
        0x38,
        up_alpha,
    );
    let intermediate = gate
        .into_iter()
        .zip(up)
        .map(|(gate, up)| bf16::from_f32((gate / (1.0 + (-gate).exp())) * up).to_f32())
        .collect::<Vec<_>>();
    constant_projection(
        &intermediate,
        HIDDEN,
        |row| {
            if shared {
                shared_down_code(row)
            } else {
                down_code(expert, row)
            }
        },
        0x38,
        down_alpha,
    )
}

fn expert_gate_code(expert: usize, row: usize) -> u8 {
    1 + ((expert + row) % 3) as u8
}

fn expert_up_code(expert: usize, row: usize) -> u8 {
    1 + ((expert + row + 2) % 3) as u8
}

fn down_code(expert: usize, row: usize) -> u8 {
    1 + ((expert + row + 1) % 3) as u8
}

fn shared_gate_code(row: usize) -> u8 {
    1 + (row % 3) as u8
}

fn shared_up_code(row: usize) -> u8 {
    1 + ((row + 2) % 3) as u8
}

fn shared_down_code(row: usize) -> u8 {
    1 + ((row + 1) % 3) as u8
}

fn packed_rows(rows: usize, inputs: usize, code: impl Fn(usize) -> u8) -> Vec<u8> {
    let mut packed = vec![0u8; rows * inputs / 2];
    for (row, packed_row) in packed.chunks_exact_mut(inputs / 2).enumerate() {
        let nibble = code(row) & 0x0f;
        packed_row.fill(nibble | (nibble << 4));
    }
    packed
}

fn expert_alpha(base: f32, step: f32, expert: usize) -> f32 {
    base + step * (expert % 5) as f32
}

fn assert_bf16_close(actual: &[f32], expected: &[f32], label: &str) {
    assert_eq!(actual.len(), expected.len());
    for (index, (&actual, &expected)) in actual.iter().zip(expected).enumerate() {
        let expected = bf16::from_f32(expected).to_f32();
        let tolerance = 0.012 * expected.abs().max(1.0);
        assert!(
            (actual - expected).abs() <= tolerance,
            "{label}[{index}] {actual} != {expected} within {tolerance}"
        );
    }
}

#[test]
fn qwen35_decode_top8_nvfp4_experts_match_cpu_dequant_reference() {
    let ctx = CudaContext::new(0).unwrap();
    let config = RoutedMoeConfig::new(EXPERTS, TOP_K);
    let tokens = 1024usize;
    let routes = tokens * TOP_K;

    let mut single = vec![-100.0f32; EXPERTS];
    for (expert, score) in [
        (255, 1.0),
        (91, 2.0),
        (12, 3.0),
        (7, 4.0),
        (3, 5.0),
        (2, 6.0),
        (1, 7.0),
        (0, 8.0),
    ] {
        single[expert] = score;
    }
    let logits = single.repeat(tokens);
    let reference_plan = ops::route_topk_reference(config, &logits, tokens).unwrap();
    let logits = upload(
        &ctx,
        &Tensor::from_f32(vec![tokens, EXPERTS], &logits).unwrap(),
    );
    let mut expert_ids = zeros(&ctx, vec![tokens, TOP_K], DType::I32);
    let mut expert_weights = zeros(&ctx, vec![tokens, TOP_K], DType::F32);
    let mut assignment_expert_ids = zeros(&ctx, vec![routes], DType::I32);
    let mut assignment_token_ids = zeros(&ctx, vec![routes], DType::I32);
    let mut assignment_slots = zeros(&ctx, vec![routes], DType::I32);
    let mut assignment_weights = zeros(&ctx, vec![routes], DType::F32);
    let mut expert_offsets = zeros(&ctx, vec![EXPERTS + 1], DType::I32);
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
            assignment_weights: &mut assignment_weights,
            expert_offsets: &mut expert_offsets,
        },
    )
    .unwrap();

    let input_values = (0..tokens * HIDDEN)
        .map(|index| bf16::from_f32(0.001 * (1 + (index % HIDDEN) % 5) as f32))
        .collect::<Vec<_>>();
    let input = upload(
        &ctx,
        &Tensor::from_bf16(vec![tokens, HIDDEN], &input_values).unwrap(),
    );

    let expert_gate_weight = stacked_packed(&ctx, INTERMEDIATE, HIDDEN, expert_gate_code);
    let expert_gate_scales = stacked_unit_scales(&ctx, INTERMEDIATE, HIDDEN);
    let expert_up_weight = stacked_packed(&ctx, INTERMEDIATE, HIDDEN, expert_up_code);
    let expert_up_scales = stacked_unit_scales(&ctx, INTERMEDIATE, HIDDEN);
    let expert_down_weight = stacked_packed(&ctx, HIDDEN, INTERMEDIATE, down_code);
    let expert_down_scales = stacked_unit_scales(&ctx, HIDDEN, INTERMEDIATE);
    let expert_gate_alpha = expert_alphas(&ctx, 0.20, 0.01);
    let expert_up_alpha = expert_alphas(&ctx, 0.10, 0.005);
    let expert_down_alpha = expert_alphas(&ctx, 0.002, 0.0001);

    let shared_gate_weight = upload_bytes(
        &ctx,
        &packed_rows(INTERMEDIATE, HIDDEN, shared_gate_code),
        vec![INTERMEDIATE, HIDDEN / 2],
        DType::E2M1Pair,
    );
    let shared_gate_scales = upload_bytes(
        &ctx,
        &vec![0x38u8; INTERMEDIATE * HIDDEN / BLOCK],
        vec![INTERMEDIATE, HIDDEN / BLOCK],
        DType::F8E4M3,
    );
    let shared_up_weight = upload_bytes(
        &ctx,
        &packed_rows(INTERMEDIATE, HIDDEN, shared_up_code),
        vec![INTERMEDIATE, HIDDEN / 2],
        DType::E2M1Pair,
    );
    let shared_up_scales = upload_bytes(
        &ctx,
        &vec![0x38u8; INTERMEDIATE * HIDDEN / BLOCK],
        vec![INTERMEDIATE, HIDDEN / BLOCK],
        DType::F8E4M3,
    );
    let shared_down_weight = upload_bytes(
        &ctx,
        &packed_rows(HIDDEN, INTERMEDIATE, shared_down_code),
        vec![HIDDEN, INTERMEDIATE / 2],
        DType::E2M1Pair,
    );
    let shared_down_scales = upload_bytes(
        &ctx,
        &vec![0x38u8; HIDDEN * INTERMEDIATE / BLOCK],
        vec![HIDDEN, INTERMEDIATE / BLOCK],
        DType::F8E4M3,
    );
    let shared_gate_alpha = upload(&ctx, &Tensor::from_f32(vec![1], &[0.18]).unwrap());
    let shared_up_alpha = upload(&ctx, &Tensor::from_f32(vec![1], &[0.09]).unwrap());
    let shared_down_alpha = upload(&ctx, &Tensor::from_f32(vec![1], &[0.0015]).unwrap());

    let mut route_intermediate = zeros(&ctx, vec![routes, INTERMEDIATE], DType::BF16);
    let mut shared_intermediate = zeros(&ctx, vec![tokens, INTERMEDIATE], DType::BF16);
    let mut assignment_output = zeros(&ctx, vec![routes, HIDDEN], DType::BF16);
    let mut shared_output = zeros(&ctx, vec![tokens, HIDDEN], DType::BF16);
    let mut run_experts = || {
        ops::routed_moe_nvfp4_experts(
            &ctx,
            RoutedMoeNvfp4ExpertsArgs {
                config,
                intermediate: INTERMEDIATE,
                block_size: BLOCK as u32,
                scale_layout: ops::ScaleLayout::RowMajor,
                input: &input,
                assignment_expert_ids: &assignment_expert_ids,
                assignment_token_ids: &assignment_token_ids,
                expert_gate: Nvfp4ExpertProjection {
                    packed_weight: &expert_gate_weight,
                    block_scales: &expert_gate_scales,
                    alpha: &expert_gate_alpha,
                },
                expert_up: Nvfp4ExpertProjection {
                    packed_weight: &expert_up_weight,
                    block_scales: &expert_up_scales,
                    alpha: &expert_up_alpha,
                },
                expert_down: Nvfp4ExpertProjection {
                    packed_weight: &expert_down_weight,
                    block_scales: &expert_down_scales,
                    alpha: &expert_down_alpha,
                },
                shared_gate: Nvfp4ExpertProjection {
                    packed_weight: &shared_gate_weight,
                    block_scales: &shared_gate_scales,
                    alpha: &shared_gate_alpha,
                },
                shared_up: Nvfp4ExpertProjection {
                    packed_weight: &shared_up_weight,
                    block_scales: &shared_up_scales,
                    alpha: &shared_up_alpha,
                },
                shared_down: Nvfp4ExpertProjection {
                    packed_weight: &shared_down_weight,
                    block_scales: &shared_down_scales,
                    alpha: &shared_down_alpha,
                },
                route_intermediate: &mut route_intermediate,
                shared_intermediate: &mut shared_intermediate,
                assignment_output: &mut assignment_output,
                shared_output: &mut shared_output,
            },
        )
        .unwrap();
    };

    // One untimed launch removes first-use effects.  CI keeps one measured
    // iteration; the Thor benchmark sets APXINF_MOE_BENCH_ITERS to amortize
    // host launch and synchronization jitter without changing the oracle.
    run_experts();
    ctx.synchronize().unwrap();
    let iterations = std::env::var("APXINF_MOE_BENCH_ITERS")
        .ok()
        .and_then(|value| value.parse::<u32>().ok())
        .filter(|&value| value > 0)
        .unwrap_or(1);
    let started = Instant::now();
    for _ in 0..iterations {
        run_experts();
    }
    ctx.synchronize().unwrap();
    drop(run_experts);
    let elapsed = started.elapsed();
    println!(
        "Qwen3.5 routed NVFP4 experts: total={elapsed:?}, iterations={iterations}, mean={:?}",
        elapsed / iterations
    );

    let assignment_ids = read_i32(&assignment_expert_ids);
    assert_eq!(
        assignment_ids,
        reference_plan
            .assignments()
            .iter()
            .map(|assignment| assignment.expert_id as i32)
            .collect::<Vec<_>>()
    );
    let input_f32 = input_values
        .iter()
        .map(|value| value.to_f32())
        .collect::<Vec<_>>();
    let mut expected_assignments = Vec::with_capacity(routes * HIDDEN);
    for &expert in &assignment_ids {
        let expert = expert as usize;
        expected_assignments.extend(cpu_expert(
            &input_f32,
            expert,
            false,
            expert_alpha(0.20, 0.01, expert),
            expert_alpha(0.10, 0.005, expert),
            expert_alpha(0.002, 0.0001, expert),
        ));
    }
    let expected_shared = cpu_expert(&input_f32, 0, true, 0.18, 0.09, 0.0015);
    assert_bf16_close(
        &read_bf16(&assignment_output),
        &expected_assignments,
        "assignment_output",
    );
    assert_bf16_close(
        &read_bf16(&shared_output),
        &expected_shared,
        "shared_output",
    );

    let token_ids = read_i32(&assignment_token_ids);
    let route_weights = read_f32(&expert_weights);
    let gpu_plan = RoutePlan::from_token_major(
        config,
        tokens,
        read_i32(&expert_ids)
            .into_iter()
            .map(|value| value as u32)
            .collect(),
        route_weights,
    )
    .unwrap();
    assert_eq!(token_ids, vec![0; routes]);
    let gate_logits = upload(
        &ctx,
        &Tensor::from_bf16(vec![tokens], &[bf16::from_f32(0.0)]).unwrap(),
    );
    let mut combined = zeros(&ctx, vec![tokens, HIDDEN], DType::BF16);
    ops::routed_moe_combine(
        &ctx,
        RoutedMoeCombineArgs {
            assignment_output: &assignment_output,
            assignment_token_ids: &assignment_token_ids,
            assignment_weights: &assignment_weights,
            shared_output: &shared_output,
            shared_gate_logits: &gate_logits,
            output: &mut combined,
        },
    )
    .unwrap();
    ctx.synchronize().unwrap();
    let expected_combined = ops::routed_moe_combine_reference(
        &gpu_plan,
        &expected_assignments,
        HIDDEN,
        Some(SharedExpertOutput {
            output: &expected_shared,
            gate_logits: &[0.0],
        }),
    )
    .unwrap();
    assert_bf16_close(&read_bf16(&combined), &expected_combined, "combined");
}
