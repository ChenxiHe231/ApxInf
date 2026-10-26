//! Exact-geometry CUDA check for the BF16 Qwen3.5 MTP expert layer.

use apxinf_core::{DType, Shape, Tensor};
use apxinf_cuda_new::{
    edge_bf16_decode_moe, edge_bf16_decode_moe_available,
    edge_bf16_decode_moe_workspace_size, ops, qwen35_mtp_moe, qwen35_mtp_moe_reference,
    transfers, CudaBuffer, CudaContext, EdgeBf16DecodeMoeArgs, EdgeBf16DecodeMoeWeights,
    Qwen35MtpMoeArgs, Qwen35MtpMoeScratch, Qwen35MtpMoeState, Qwen35MtpMoeWeights,
    Qwen35MtpReferenceWeights, RoutedMoeConfig, QWEN35_MTP_EXPERTS, QWEN35_MTP_HIDDEN,
    QWEN35_MTP_INTERMEDIATE, QWEN35_MTP_TOP_K,
};
use half::bf16;

fn zeros(ctx: &CudaContext, shape: Vec<usize>, dtype: DType) -> (CudaBuffer, Tensor) {
    let bytes = shape.iter().product::<usize>() * dtype.size_in_bytes();
    let buffer = CudaBuffer::alloc_zeros(bytes, ctx.device_id()).unwrap();
    let tensor = buffer.as_tensor(Shape::new(shape), dtype).unwrap();
    (buffer, tensor)
}

fn set_bf16(buffer: &CudaBuffer, element: usize, value: f32) {
    let bytes = bf16::from_f32(value).to_bits().to_ne_bytes();
    buffer.view(element * 2, 2).unwrap().copy_from_host(&bytes).unwrap();
}

struct SparseReference;

impl Qwen35MtpReferenceWeights for SparseReference {
    fn router(&self, input: usize, expert: usize) -> f32 {
        if input == 0 && expert < QWEN35_MTP_TOP_K {
            (QWEN35_MTP_TOP_K - expert) as f32
        } else {
            0.0
        }
    }

    fn expert_gate(&self, expert: usize, input: usize, output: usize) -> f32 {
        if expert < QWEN35_MTP_TOP_K && input == 0 && output == 0 {
            0.5 + expert as f32 * 0.125
        } else {
            0.0
        }
    }

    fn expert_up(&self, expert: usize, input: usize, output: usize) -> f32 {
        if expert < QWEN35_MTP_TOP_K && input == 0 && output == 0 {
            1.0
        } else {
            0.0
        }
    }

    fn expert_down(&self, expert: usize, input: usize, output: usize) -> f32 {
        if expert < QWEN35_MTP_TOP_K && input == 0 && output == 0 {
            1.0
        } else {
            0.0
        }
    }

    fn shared_gate(&self, input: usize, output: usize) -> f32 {
        if input == 0 && output == 0 { 1.0 } else { 0.0 }
    }

    fn shared_up(&self, input: usize, output: usize) -> f32 {
        if input == 0 && output == 0 { 1.0 } else { 0.0 }
    }

    fn shared_down(&self, input: usize, output: usize) -> f32 {
        if input == 0 && output == 0 { 1.0 } else { 0.0 }
    }

    fn shared_expert_gate(&self, _: usize) -> f32 { 0.0 }
}

#[test]
fn mtp_swiglu_rounds_only_the_final_product() {
    let ctx = CudaContext::new(0).unwrap();
    let input = Tensor::from_bf16(
        vec![1, 2],
        &[bf16::from_f32(4.71875), bf16::from_f32(-3.640625)],
    )
    .unwrap();
    let input = transfers::to_cuda(&input, ctx.device_id()).unwrap();
    let (_output_buffer, output) = zeros(&ctx, vec![1, 1], DType::BF16);
    ops::swiglu_round_product(&ctx, &input, &output).unwrap();
    ctx.synchronize().unwrap();
    let actual = transfers::to_cpu(&output)
        .unwrap()
        .as_bf16()
        .unwrap()[0];
    assert_eq!(actual, bf16::from_f32(-17.0));
    assert_ne!(actual, bf16::from_f32(-17.125));
}

#[test]
fn exact_qwen35_mtp_bf16_selected_experts_match_cpu_reference() {
    let ctx = CudaContext::new(0).unwrap();

    let (input_buffer, input) = zeros(&ctx, vec![1, QWEN35_MTP_HIDDEN], DType::BF16);
    set_bf16(&input_buffer, 0, 1.0);

    let (router_buffer, router) = zeros(
        &ctx,
        vec![QWEN35_MTP_HIDDEN, QWEN35_MTP_EXPERTS],
        DType::BF16,
    );
    for expert in 0..QWEN35_MTP_TOP_K {
        set_bf16(
            &router_buffer,
            expert,
            (QWEN35_MTP_TOP_K - expert) as f32,
        );
    }

    let (expert_gate_buffer, expert_gate) = zeros(
        &ctx,
        vec![
            QWEN35_MTP_EXPERTS,
            QWEN35_MTP_HIDDEN,
            QWEN35_MTP_INTERMEDIATE,
        ],
        DType::BF16,
    );
    let (expert_up_buffer, expert_up) = zeros(
        &ctx,
        vec![
            QWEN35_MTP_EXPERTS,
            QWEN35_MTP_HIDDEN,
            QWEN35_MTP_INTERMEDIATE,
        ],
        DType::BF16,
    );
    let (expert_down_buffer, expert_down) = zeros(
        &ctx,
        vec![
            QWEN35_MTP_EXPERTS,
            QWEN35_MTP_INTERMEDIATE,
            QWEN35_MTP_HIDDEN,
        ],
        DType::BF16,
    );
    for expert in 0..QWEN35_MTP_TOP_K {
        let gate_index = expert * QWEN35_MTP_HIDDEN * QWEN35_MTP_INTERMEDIATE;
        let down_index = expert * QWEN35_MTP_INTERMEDIATE * QWEN35_MTP_HIDDEN;
        set_bf16(
            &expert_gate_buffer,
            gate_index,
            0.5 + expert as f32 * 0.125,
        );
        set_bf16(&expert_up_buffer, gate_index, 1.0);
        set_bf16(&expert_down_buffer, down_index, 1.0);
    }

    let (shared_gate_buffer, shared_gate) = zeros(
        &ctx,
        vec![QWEN35_MTP_HIDDEN, QWEN35_MTP_INTERMEDIATE],
        DType::BF16,
    );
    let (shared_up_buffer, shared_up) = zeros(
        &ctx,
        vec![QWEN35_MTP_HIDDEN, QWEN35_MTP_INTERMEDIATE],
        DType::BF16,
    );
    let (shared_down_buffer, shared_down) = zeros(
        &ctx,
        vec![QWEN35_MTP_INTERMEDIATE, QWEN35_MTP_HIDDEN],
        DType::BF16,
    );
    set_bf16(&shared_gate_buffer, 0, 1.0);
    set_bf16(&shared_up_buffer, 0, 1.0);
    set_bf16(&shared_down_buffer, 0, 1.0);
    let (_shared_expert_gate_buffer, shared_expert_gate) =
        zeros(&ctx, vec![QWEN35_MTP_HIDDEN, 1], DType::BF16);

    let (_router_logits_buffer, mut router_logits) =
        zeros(&ctx, vec![1, QWEN35_MTP_EXPERTS], DType::BF16);
    let (_expert_ids_buffer, mut expert_ids) =
        zeros(&ctx, vec![QWEN35_MTP_TOP_K], DType::I32);
    let (_expert_weights_buffer, mut expert_weights) =
        zeros(&ctx, vec![QWEN35_MTP_TOP_K], DType::F32);
    let (_route_intermediate_buffer, mut route_intermediate) = zeros(
        &ctx,
        vec![QWEN35_MTP_TOP_K, QWEN35_MTP_INTERMEDIATE],
        DType::BF16,
    );
    let (_route_output_buffer, mut route_output) = zeros(
        &ctx,
        vec![QWEN35_MTP_TOP_K, QWEN35_MTP_HIDDEN],
        DType::BF16,
    );
    let (_shared_intermediate_buffer, mut shared_intermediate) =
        zeros(&ctx, vec![1, QWEN35_MTP_INTERMEDIATE], DType::BF16);
    let (_shared_output_buffer, mut shared_output) =
        zeros(&ctx, vec![1, QWEN35_MTP_HIDDEN], DType::BF16);
    let (_shared_gate_logit_buffer, mut shared_gate_logit) =
        zeros(&ctx, vec![1], DType::BF16);
    let (_output_buffer, mut output) =
        zeros(&ctx, vec![1, QWEN35_MTP_HIDDEN], DType::BF16);

    qwen35_mtp_moe(
        &ctx,
        Qwen35MtpMoeArgs {
            state: Qwen35MtpMoeState {
                base_committed_position: 31,
                mtp_committed_position: 31,
                draft_step: 2,
            },
            input: &input,
            weights: Qwen35MtpMoeWeights {
                router: &router,
                expert_gate: &expert_gate,
                expert_up: &expert_up,
                expert_down: &expert_down,
                shared_gate: &shared_gate,
                shared_up: &shared_up,
                shared_down: &shared_down,
                shared_expert_gate: &shared_expert_gate,
            },
            scratch: Qwen35MtpMoeScratch {
                router_logits: &mut router_logits,
                expert_ids: &mut expert_ids,
                expert_weights: &mut expert_weights,
                route_intermediate: &mut route_intermediate,
                route_output: &mut route_output,
                shared_intermediate: &mut shared_intermediate,
                shared_output: &mut shared_output,
                shared_gate_logit: &mut shared_gate_logit,
            },
            output: &mut output,
        },
    )
    .unwrap();
    ctx.synchronize().unwrap();

    let ids = transfers::to_cpu(&expert_ids)
        .unwrap()
        .as_i32()
        .unwrap()
        .to_vec();
    assert_eq!(ids, vec![0, 1, 2, 3, 4, 5, 6, 7]);
    let actual_weights = transfers::to_cpu(&expert_weights)
        .unwrap()
        .as_f32()
        .unwrap()
        .to_vec();
    let actual_output = transfers::to_cpu(&output)
        .unwrap()
        .as_bf16()
        .unwrap()
        .iter()
        .map(|value| value.to_f32())
        .collect::<Vec<_>>();

    if edge_bf16_decode_moe_available() {
        let (fc1_buffer, fc1) = zeros(
            &ctx,
            vec![
                QWEN35_MTP_EXPERTS,
                2 * QWEN35_MTP_INTERMEDIATE,
                QWEN35_MTP_HIDDEN,
            ],
            DType::BF16,
        );
        let (fc2_buffer, fc2) = zeros(
            &ctx,
            vec![
                QWEN35_MTP_EXPERTS,
                QWEN35_MTP_HIDDEN,
                QWEN35_MTP_INTERMEDIATE,
            ],
            DType::BF16,
        );
        for expert in 0..QWEN35_MTP_TOP_K {
            let fc1_base = expert
                * 2
                * QWEN35_MTP_INTERMEDIATE
                * QWEN35_MTP_HIDDEN;
            // TensorRT-Edge-LLM stores each 64-row chunk as [up, gate].
            set_bf16(&fc1_buffer, fc1_base, 1.0);
            set_bf16(
                &fc1_buffer,
                fc1_base + 64 * QWEN35_MTP_HIDDEN,
                0.5 + expert as f32 * 0.125,
            );
            let fc2_base = expert
                * QWEN35_MTP_HIDDEN
                * QWEN35_MTP_INTERMEDIATE;
            set_bf16(&fc2_buffer, fc2_base, 1.0);
        }
        let ids = _expert_ids_buffer
            .as_tensor(
                Shape::new(vec![1, QWEN35_MTP_TOP_K]),
                DType::I32,
            )
            .unwrap();
        let weights = _expert_weights_buffer
            .as_tensor(
                Shape::new(vec![1, QWEN35_MTP_TOP_K]),
                DType::F32,
            )
            .unwrap();
        let config = RoutedMoeConfig::new(QWEN35_MTP_EXPERTS, QWEN35_MTP_TOP_K);
        let workspace = CudaBuffer::alloc(
            edge_bf16_decode_moe_workspace_size(
                config,
                QWEN35_MTP_HIDDEN,
                QWEN35_MTP_INTERMEDIATE,
            )
            .unwrap(),
            ctx.device_id(),
        )
        .unwrap();
        let (_edge_output_buffer, mut edge_output) =
            zeros(&ctx, vec![1, QWEN35_MTP_HIDDEN], DType::BF16);
        edge_bf16_decode_moe(
            &ctx,
            EdgeBf16DecodeMoeArgs {
                config,
                intermediate: QWEN35_MTP_INTERMEDIATE,
                input: &input,
                expert_ids: &ids,
                expert_weights: &weights,
                weights: EdgeBf16DecodeMoeWeights {
                    fc1_weight: &fc1,
                    fc2_weight: &fc2,
                },
                workspace: &workspace,
                output: &mut edge_output,
            },
        )
        .unwrap();
        ctx.synchronize().unwrap();
        let edge_output = transfers::to_cpu(&edge_output)
            .unwrap()
            .as_bf16()
            .unwrap()
            .iter()
            .map(|value| value.to_f32())
            .collect::<Vec<_>>();
        let route_output = transfers::to_cpu(&route_output)
            .unwrap()
            .as_bf16()
            .unwrap()
            .iter()
            .map(|value| value.to_f32())
            .collect::<Vec<_>>();
        for feature in 0..QWEN35_MTP_HIDDEN {
            let expected = (0..QWEN35_MTP_TOP_K)
                .map(|route| {
                    actual_weights[route]
                        * route_output[route * QWEN35_MTP_HIDDEN + feature]
                })
                .sum::<f32>();
            let tolerance = 0.01 * expected.abs().max(1.0);
            assert!(
                (edge_output[feature] - expected).abs() <= tolerance,
                "edge BF16 feature {feature}: {} != {expected}",
                edge_output[feature]
            );
        }
    }
    let mut reference_input = vec![0.0f32; QWEN35_MTP_HIDDEN];
    reference_input[0] = 1.0;
    let reference = qwen35_mtp_moe_reference(&reference_input, &SparseReference).unwrap();
    for (actual, expected) in actual_weights.iter().zip(reference.expert_weights) {
        assert!((actual - expected).abs() <= 2e-6, "{actual} != {expected}");
    }
    for (index, (actual, expected)) in actual_output.iter().zip(reference.output).enumerate() {
        let tolerance = 0.01 * expected.abs().max(1.0);
        assert!(
            (actual - expected).abs() <= tolerance,
            "output[{index}] {actual} != {expected} within {tolerance}"
        );
    }
}
