//! SM110 TensorRT-Edge-LLM NVFP4 MoE synthetic parity and timing.

use std::time::Instant;

use apxinf_core::{DType, Shape, Tensor};
use apxinf_cuda_new::{
    edge_nvfp4_moe, edge_nvfp4_moe_available, edge_nvfp4_moe_workspace_size,
    edge_nvfp4_swizzle_scales, encode_e4m3, transfers, CudaBuffer, CudaContext, EdgeNvfp4MoeArgs,
    EdgeNvfp4MoeWeights, RoutedMoeConfig,
};
use half::bf16;

const EXPERTS: usize = 256;
const TOP_K: usize = 8;
const HIDDEN: usize = 2048;
const INTERMEDIATE: usize = 512;
const BLOCK: usize = 16;

fn upload(ctx: &CudaContext, tensor: &Tensor) -> Tensor {
    transfers::to_cuda(tensor, ctx.device_id()).unwrap()
}

fn repeated_expert_tensor(
    ctx: &CudaContext,
    one_expert: &[u8],
    shape: Vec<usize>,
    dtype: DType,
) -> Tensor {
    let buffer = CudaBuffer::alloc(EXPERTS * one_expert.len(), ctx.device_id()).unwrap();
    for expert in 0..EXPERTS {
        buffer
            .view(expert * one_expert.len(), one_expert.len())
            .unwrap()
            .copy_from_host(one_expert)
            .unwrap();
    }
    buffer.as_tensor(Shape::new(shape), dtype).unwrap()
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

fn nearest_e2m1(value: f32) -> f32 {
    const VALUES: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    let sign = value.signum();
    let magnitude = value.abs();
    sign * VALUES
        .into_iter()
        .min_by(|left, right| {
            (magnitude - left)
                .abs()
                .total_cmp(&(magnitude - right).abs())
        })
        .unwrap()
}

fn quant_dequant_block(value: f32, global_scale: f32) -> f32 {
    let sf = decode_e4m3(encode_e4m3(value.abs() / (6.0 * global_scale)));
    if sf == 0.0 {
        0.0
    } else {
        nearest_e2m1(value / (sf * global_scale)) * sf * global_scale
    }
}

#[test]
fn qwen35_edge_nvfp4_synthetic_matches_quantized_constant_reference() {
    if !edge_nvfp4_moe_available() {
        eprintln!("skipping SM110-only Edge NVFP4 MoE test");
        return;
    }
    let ctx = CudaContext::new(0).unwrap();
    let tokens = std::env::var("APXINF_EDGE_MOE_TOKENS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .filter(|&value| value > 0)
        .unwrap_or(1);
    let config = RoutedMoeConfig::new(EXPERTS, TOP_K);

    let input_values = vec![bf16::from_f32(1.0); tokens * HIDDEN];
    let input = upload(
        &ctx,
        &Tensor::from_bf16(vec![tokens, HIDDEN], &input_values).unwrap(),
    );
    let expert_ids = (0..tokens)
        .flat_map(|_| 0..TOP_K as i32)
        .collect::<Vec<_>>();
    let expert_ids = upload(
        &ctx,
        &Tensor::from_i32(vec![tokens, TOP_K], &expert_ids).unwrap(),
    );
    let expert_weights = upload(
        &ctx,
        &Tensor::from_f32(
            vec![tokens, TOP_K],
            &vec![1.0 / TOP_K as f32; tokens * TOP_K],
        )
        .unwrap(),
    );

    // Both E2M1 nibbles are 0.5. The small E4M3 block scale keeps the
    // constant-reference accumulation away from saturation.
    let weight_scale = 1.0 / 256.0;
    let scale_code = encode_e4m3(weight_scale);
    let fc1_weight = repeated_expert_tensor(
        &ctx,
        &vec![0x11; 2 * INTERMEDIATE * HIDDEN / 2],
        vec![EXPERTS, 2 * INTERMEDIATE, HIDDEN / 2],
        DType::E2M1Pair,
    );
    let (fc1_scale_one, _) = edge_nvfp4_swizzle_scales(
        &vec![scale_code; 2 * INTERMEDIATE * HIDDEN / BLOCK],
        2 * INTERMEDIATE,
        HIDDEN / BLOCK,
    )
    .unwrap();
    let fc1_scales = repeated_expert_tensor(
        &ctx,
        &fc1_scale_one,
        vec![
            EXPERTS,
            (2 * INTERMEDIATE).div_ceil(128),
            (HIDDEN / BLOCK).div_ceil(4),
            32,
            4,
            4,
        ],
        DType::F8E4M3,
    );
    let fc2_weight = repeated_expert_tensor(
        &ctx,
        &vec![0x11; HIDDEN * INTERMEDIATE / 2],
        vec![EXPERTS, HIDDEN, INTERMEDIATE / 2],
        DType::E2M1Pair,
    );
    let (fc2_scale_one, _) = edge_nvfp4_swizzle_scales(
        &vec![scale_code; HIDDEN * INTERMEDIATE / BLOCK],
        HIDDEN,
        INTERMEDIATE / BLOCK,
    )
    .unwrap();
    let fc2_scales = repeated_expert_tensor(
        &ctx,
        &fc2_scale_one,
        vec![
            EXPERTS,
            HIDDEN.div_ceil(128),
            (INTERMEDIATE / BLOCK).div_ceil(4),
            32,
            4,
            4,
        ],
        DType::F8E4M3,
    );
    let unit = upload(
        &ctx,
        &Tensor::from_f32(vec![EXPERTS], &vec![1.0; EXPERTS]).unwrap(),
    );
    let workspace = CudaBuffer::alloc(
        edge_nvfp4_moe_workspace_size(config, tokens, HIDDEN, INTERMEDIATE).unwrap(),
        ctx.device_id(),
    )
    .unwrap();
    let mut output = CudaBuffer::alloc_zeros(
        tokens * HIDDEN * DType::BF16.size_in_bytes(),
        ctx.device_id(),
    )
    .unwrap()
    .as_tensor(Shape::new(vec![tokens, HIDDEN]), DType::BF16)
    .unwrap();

    let mut run = || {
        edge_nvfp4_moe(
            &ctx,
            EdgeNvfp4MoeArgs {
                config,
                intermediate: INTERMEDIATE,
                input: &input,
                expert_ids: &expert_ids,
                expert_weights: &expert_weights,
                weights: EdgeNvfp4MoeWeights {
                    fc1_weight: &fc1_weight,
                    fc1_scales: &fc1_scales,
                    fc1_alpha: &unit,
                    input_global_scale: &unit,
                    fc2_weight: &fc2_weight,
                    fc2_scales: &fc2_scales,
                    fc2_alpha: &unit,
                    down_input_scale: &unit,
                },
                workspace: &workspace,
                output: &mut output,
            },
        )
        .unwrap();
    };
    run();
    ctx.synchronize().unwrap();
    let iterations = std::env::var("APXINF_EDGE_MOE_BENCH_ITERS")
        .ok()
        .and_then(|value| value.parse::<u32>().ok())
        .filter(|&value| value > 0)
        .unwrap_or(1);
    let started = Instant::now();
    for _ in 0..iterations {
        run();
    }
    ctx.synchronize().unwrap();
    drop(run);
    println!(
        "Edge SM110 NVFP4 MoE tokens={tokens}: {:?}/iteration",
        started.elapsed() / iterations
    );

    let effective_input = quant_dequant_block(1.0, 1.0);
    let projection_weight = 0.5 * decode_e4m3(scale_code);
    let gate = bf16::from_f32(HIDDEN as f32 * effective_input * projection_weight).to_f32();
    let activated = (gate / (1.0 + (-gate).exp())) * gate;
    let effective_intermediate = quant_dequant_block(activated, 1.0);
    let expected =
        bf16::from_f32(INTERMEDIATE as f32 * effective_intermediate * projection_weight).to_f32();
    let actual = transfers::to_cpu(&output)
        .unwrap()
        .as_bf16()
        .unwrap()
        .iter()
        .map(|value| value.to_f32())
        .collect::<Vec<_>>();
    let tolerance = expected.abs().max(1.0) * 0.08;
    for (index, value) in actual.into_iter().enumerate() {
        assert!(value.is_finite(), "output[{index}] is not finite");
        assert!(
            (value - expected).abs() <= tolerance,
            "output[{index}]={value}, expected {expected} within {tolerance}"
        );
    }
}
