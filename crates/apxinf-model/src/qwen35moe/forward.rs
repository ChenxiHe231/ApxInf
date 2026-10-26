//! Executable CUDA slices of the Qwen3.5 text path.
//!
//! These kernels back the registered [`crate::llm_trait::LlmTrait`] runtime,
//! including the target full-attention/FP8-KV backend and BF16 MTP executor.
//! Together with `decoder`/`session`, they keep inputs,
//! intermediates, GDN state, routing, expert output, and logits on CUDA and use
//! only cuda-new's safe L3 interfaces.

use apxinf_core::{DType, Device, Error, Result, Shape, Tensor};
use apxinf_cuda_new::{
    ops, transfers, CudaBuffer, CudaContext, EdgeNvfp4MoeArgs, EdgeNvfp4MoeWeights,
    Nvfp4ExpertProjection, RoutedMoeAddSharedArgs, RoutedMoeCombineArgs, RoutedMoeConfig,
    RoutedMoeNvfp4ExpertsArgs, RoutedMoeRouteArgs, RoutedMoeRouteTopkArgs,
};

use super::device_weights::{
    DeviceGdnWeights, DeviceNvfp4Moe, DeviceNvfp4RoutedExperts, DeviceStaticFp8Linear,
    Qwen35DeviceWeights,
};
use super::execution::Qwen35MoeActivation;

const NVFP4_BLOCK_SIZE: u32 = 16;

/// Stable buffers for checkpoint embedding and final normalization/head.
///
/// `logits` intentionally contains every row. This preserves the required
/// `LlmTrait::forward -> [sequence, vocab]` contract; the generation-oriented
/// prefill path projects only the final row separately.
#[derive(Debug)]
pub struct TextBoundaryScratch {
    token_ids: Tensor,
    hidden: Tensor,
    normalized: Tensor,
    logits: Tensor,
    tokens: usize,
    hidden_size: usize,
    vocab_size: usize,
}

impl TextBoundaryScratch {
    pub fn new(
        ctx: &CudaContext,
        tokens: usize,
        hidden_size: usize,
        vocab_size: usize,
    ) -> Result<Self> {
        if tokens == 0 || hidden_size == 0 || vocab_size == 0 {
            return Err(invalid("Qwen3.5 text boundary dimensions must be non-zero"));
        }
        Ok(Self {
            token_ids: zeros(ctx, vec![tokens], DType::I32)?,
            hidden: zeros(ctx, vec![tokens, hidden_size], DType::BF16)?,
            normalized: zeros(ctx, vec![tokens, hidden_size], DType::BF16)?,
            logits: zeros(ctx, vec![tokens, vocab_size], DType::BF16)?,
            tokens,
            hidden_size,
            vocab_size,
        })
    }

    pub fn write_token_ids(&self, token_ids: &[u32]) -> Result<()> {
        if token_ids.len() != self.tokens {
            return Err(invalid(format!(
                "Qwen3.5 token count mismatch: prepared {}, got {}",
                self.tokens,
                token_ids.len()
            )));
        }
        let token_ids = token_ids
            .iter()
            .map(|&token| {
                i32::try_from(token)
                    .map_err(|_| invalid(format!("Qwen3.5 token ID {token} exceeds i32")))
            })
            .collect::<Result<Vec<_>>>()?;
        let host = Tensor::from_i32(vec![self.tokens], &token_ids)?;
        transfers::copy_cpu_to_cuda(&host, &self.token_ids)
    }

    /// Run checkpoint embedding lookup into resident hidden storage.
    pub fn embed(&self, ctx: &CudaContext, weights: &Qwen35DeviceWeights) -> Result<()> {
        validate_tensor(
            &weights.embedding,
            ctx,
            DType::BF16,
            &[self.vocab_size, self.hidden_size],
            "embedding",
        )?;
        ops::embedding_gather(ctx, &weights.embedding, &self.token_ids, &self.hidden)
    }

    /// Apply the final RMSNorm and untied language head on CUDA.
    pub fn finish(
        &mut self,
        ctx: &CudaContext,
        weights: &Qwen35DeviceWeights,
        eps: f32,
    ) -> Result<()> {
        validate_tensor(
            &weights.final_norm,
            ctx,
            DType::BF16,
            &[self.hidden_size],
            "final norm",
        )?;
        if weights.lm_head.input_features != self.hidden_size
            || weights.lm_head.output_features != self.vocab_size
        {
            return Err(invalid("Qwen3.6 lm_head geometry mismatch"));
        }
        // Qwen3.5's ordinary RMSNorm is zero-centered: the checkpoint stores
        // a delta and the effective affine is `(1 + weight)` in F32 before the
        // BF16 output cast. `ops::mlp::rms_norm` implements that exact rule;
        // the generic `ops::rms_norm` semantic multiplies by `weight` directly
        // and is therefore not interchangeable here.
        zero_centered_rms_norm(
            ctx,
            &self.hidden,
            &weights.final_norm,
            &self.normalized,
            eps,
        )?;
        project_lm_head(ctx, weights, &self.normalized, &mut self.logits)
    }

    pub fn hidden(&self) -> &Tensor {
        &self.hidden
    }

    pub fn hidden_mut(&mut self) -> &mut Tensor {
        &mut self.hidden
    }

    pub fn logits(&self) -> &Tensor {
        &self.logits
    }
}

/// Use the checkpoint-native NVFP4 head. The Edge W4A16 kernel is one-row, so
/// multi-row verification dispatches stable row views without materializing a
/// BF16 vocabulary matrix.
pub(crate) fn project_lm_head(
    ctx: &CudaContext,
    weights: &Qwen35DeviceWeights,
    input: &Tensor,
    output: &mut Tensor,
) -> Result<()> {
    let rows = input
        .shape()
        .dims()
        .first()
        .copied()
        .ok_or_else(|| invalid("Qwen3.5 lm_head input must be rank 2"))?;
    let weight = weights
        .lm_head
        .a16
        .as_ref()
        .ok_or_else(|| invalid("Qwen3.6 lm_head W4A16 layout is unavailable"))?;
    let input_row_bytes = weight.input_features * DType::BF16.size_in_bytes();
    let output_row_bytes = weight.output_features * DType::BF16.size_in_bytes();
    for row in 0..rows {
        let input_row = CudaBuffer::from_tensor(input)
            .map_err(Error::Cuda)?
            .view(row * input_row_bytes, input_row_bytes)
            .map_err(Error::Cuda)?
            .as_tensor(Shape::new(vec![1, weight.input_features]), DType::BF16)
            .map_err(Error::Cuda)?;
        let output_row = CudaBuffer::from_tensor(output)
            .map_err(Error::Cuda)?
            .view(row * output_row_bytes, output_row_bytes)
            .map_err(Error::Cuda)?
            .as_tensor(Shape::new(vec![1, weight.output_features]), DType::BF16)
            .map_err(Error::Cuda)?;
        ops::nvfp4_a16_blackwell_gemv(
            ctx,
            &input_row,
            &weight.qweight,
            &weight.block_scales,
            &weight.global_scale,
            &output_row,
        )?;
    }
    Ok(())
}

pub(crate) fn quantize_static_fp8(
    ctx: &CudaContext,
    input: &Tensor,
    quantized: &Tensor,
    scale: f32,
) -> Result<()> {
    ops::quantize_fp8_per_tensor(ctx, input, quantized, scale)
}

pub(crate) fn project_static_fp8_prequantized(
    ctx: &CudaContext,
    weight: &DeviceStaticFp8Linear,
    quantized: &Tensor,
    output: &mut Tensor,
    rows: usize,
) -> Result<()> {
    if rows == 1 {
        ops::fp8_gemv(ctx, &weight.weight, quantized, output, weight.alpha)
    } else {
        let mut args = ops::GemmArgs::new(quantized, &weight.transposed, output);
        args.quantization = ops::GemmQuantization::Fp8UnitScale;
        args.alpha = weight.alpha;
        ops::gemm(ctx, args)
    }
}

pub(crate) fn project_static_fp8(
    ctx: &CudaContext,
    weight: &DeviceStaticFp8Linear,
    input: &Tensor,
    quantized: &Tensor,
    output: &mut Tensor,
    rows: usize,
) -> Result<()> {
    quantize_static_fp8(ctx, input, quantized, weight.input_scale)?;
    project_static_fp8_prequantized(ctx, weight, quantized, output, rows)
}

/// Stable, capture-compatible scratch for one base routed-MoE invocation.
///
/// The `expert_*` / `act_*` / `inter_*` fields back either the experimental
/// per-expert cuBLASLt path or the shared expert paired with the SM110 Edge
/// routed kernel. Unsupported builds keep the original warp-kernel fallback.
pub struct BaseMoeScratch {
    router_logits: Tensor,
    expert_ids: Tensor,
    expert_weights: Tensor,
    assignment_expert_ids: Tensor,
    assignment_token_ids: Tensor,
    assignment_slots: Tensor,
    token_assignment_indices: Tensor,
    assignment_weights: Tensor,
    expert_offsets: Tensor,
    route_intermediate: Tensor,
    shared_intermediate: Tensor,
    assignment_output: Tensor,
    shared_output: Tensor,
    shared_gate_logits: Tensor,
    output: Tensor,
    /// Per-expert gate+up interleaved workspace.  The first half is gate output,
    /// the second half is up output; `ops::swiglu` consumes this layout.
    expert_fused: Option<Tensor>,
    /// Per-expert quantised-gathered activation, `[routes, hidden/2]` E2M1Pair.
    act_packed: Option<Tensor>,
    /// Block scales for the per-expert quantised activation, `[routes, hidden/16]` F8E4M3.
    act_scales: Option<Tensor>,
    /// Per-expert quantised intermediate, `[routes, intermediate/2]` E2M1Pair.
    inter_packed: Option<Tensor>,
    /// Block scales for the per-expert quantised intermediate, `[routes, intermediate/16]` F8E4M3.
    inter_scales: Option<Tensor>,
    /// CuTeDSL scratch required by the SM110 TensorRT-Edge-LLM provider.
    edge_workspace: Option<CudaBuffer>,
    tokens: usize,
    hidden: usize,
    experts: usize,
    top_k: usize,
    intermediate: usize,
}

impl BaseMoeScratch {
    pub fn new(
        ctx: &CudaContext,
        tokens: usize,
        hidden: usize,
        experts: usize,
        top_k: usize,
        intermediate: usize,
    ) -> Result<Self> {
        let config = RoutedMoeConfig::new(experts, top_k);
        config.validate()?;
        if tokens == 0 || hidden == 0 || intermediate == 0 {
            return Err(invalid("Qwen3.5 routed-MoE dimensions must be non-zero"));
        }
        if hidden % NVFP4_BLOCK_SIZE as usize != 0 || intermediate % NVFP4_BLOCK_SIZE as usize != 0
        {
            return Err(invalid(
                "Qwen3.5 routed-MoE hidden widths must be divisible by 16",
            ));
        }
        let routes = tokens
            .checked_mul(top_k)
            .ok_or_else(|| invalid("Qwen3.5 routed-MoE route count overflow"))?;
        // Keep both execution candidates allocation-stable. The load-time
        // Qwen3.5 policy selects which one runs; process-global environment
        // variables must not override the CLI contract.
        let nvfp4_ok = true;
        let edge_ok = ops::edge_nvfp4_moe_available();
        let quant_rows = if nvfp4_ok { routes } else { tokens };
        let edge_workspace = edge_ok
            .then(|| {
                let bytes =
                    ops::edge_nvfp4_moe_workspace_size(config, tokens, hidden, intermediate)?;
                CudaBuffer::alloc(bytes, ctx.device_id()).map_err(Error::Cuda)
            })
            .transpose()?;
        Ok(Self {
            router_logits: zeros(ctx, vec![tokens, experts], DType::BF16)?,
            expert_ids: zeros(ctx, vec![tokens, top_k], DType::I32)?,
            expert_weights: zeros(ctx, vec![tokens, top_k], DType::F32)?,
            assignment_expert_ids: zeros(ctx, vec![routes], DType::I32)?,
            assignment_token_ids: zeros(ctx, vec![routes], DType::I32)?,
            assignment_slots: zeros(ctx, vec![routes], DType::I32)?,
            token_assignment_indices: zeros(ctx, vec![routes], DType::I32)?,
            assignment_weights: zeros(ctx, vec![routes], DType::F32)?,
            expert_offsets: zeros(ctx, vec![experts + 1], DType::I32)?,
            route_intermediate: zeros(ctx, vec![routes, intermediate], DType::BF16)?,
            shared_intermediate: zeros(ctx, vec![tokens, intermediate], DType::BF16)?,
            assignment_output: zeros(ctx, vec![routes, hidden], DType::BF16)?,
            shared_output: zeros(ctx, vec![tokens, hidden], DType::BF16)?,
            shared_gate_logits: zeros(ctx, vec![tokens, 1], DType::BF16)?,
            output: zeros(ctx, vec![tokens, hidden], DType::BF16)?,
            expert_fused: (nvfp4_ok || edge_ok)
                .then(|| zeros(ctx, vec![quant_rows, 2 * intermediate], DType::BF16))
                .transpose()?,
            act_packed: (nvfp4_ok || edge_ok)
                .then(|| zeros(ctx, vec![quant_rows, hidden / 2], DType::E2M1Pair))
                .transpose()?,
            act_scales: if edge_ok {
                let bytes = ops::nvfp4_scale_buffer_bytes(tokens, hidden, NVFP4_BLOCK_SIZE)?;
                Some(zeros(ctx, vec![bytes], DType::F8E4M3)?)
            } else {
                nvfp4_ok
                    .then(|| zeros(ctx, vec![quant_rows, hidden / 16], DType::F8E4M3))
                    .transpose()?
            },
            inter_packed: (nvfp4_ok || edge_ok)
                .then(|| zeros(ctx, vec![quant_rows, intermediate / 2], DType::E2M1Pair))
                .transpose()?,
            inter_scales: if edge_ok {
                let bytes = ops::nvfp4_scale_buffer_bytes(tokens, intermediate, NVFP4_BLOCK_SIZE)?;
                Some(zeros(ctx, vec![bytes], DType::F8E4M3)?)
            } else {
                nvfp4_ok
                    .then(|| zeros(ctx, vec![quant_rows, intermediate / 16], DType::F8E4M3))
                    .transpose()?
            },
            edge_workspace,
            tokens,
            hidden,
            experts,
            top_k,
            intermediate,
        })
    }

    /// Execute router, deterministic top-k assignment, routed/shared NVFP4
    /// SwiGLU experts, shared gating, and weighted combine entirely on CUDA.
    pub fn run(
        &mut self,
        ctx: &CudaContext,
        weights: &DeviceNvfp4Moe,
        input: &Tensor,
    ) -> Result<()> {
        self.validate(ctx, weights, input)?;
        let config = RoutedMoeConfig::new(self.experts, self.top_k);
        let nvfp4_ok = weights.activation == Qwen35MoeActivation::Nvfp4;

        if self.tokens == 1 && std::env::var("APXINF_QWEN35_ROUTER_GEMV").as_deref() == Ok("1") {
            ops::bf16_gemv(ctx, input, &weights.router.weight, &self.router_logits)?;
        } else {
            ops::gemm(
                ctx,
                ops::GemmArgs::new(input, &weights.router.weight, &mut self.router_logits),
            )?;
        }
        if matches!(&weights.routed_experts, DeviceNvfp4RoutedExperts::Edge(_)) {
            ops::routed_moe_route_topk(
                ctx,
                RoutedMoeRouteTopkArgs {
                    config,
                    logits: &self.router_logits,
                    expert_ids: &mut self.expert_ids,
                    expert_weights: &mut self.expert_weights,
                },
            )?;
        } else {
            ops::routed_moe_route(
                ctx,
                RoutedMoeRouteArgs {
                    config,
                    logits: &self.router_logits,
                    expert_ids: &mut self.expert_ids,
                    expert_weights: &mut self.expert_weights,
                    assignment_expert_ids: &mut self.assignment_expert_ids,
                    assignment_token_ids: &mut self.assignment_token_ids,
                    assignment_slots: &mut self.assignment_slots,
                    token_assignment_indices: &mut self.token_assignment_indices,
                    assignment_weights: &mut self.assignment_weights,
                    expert_offsets: &mut self.expert_offsets,
                },
            )?;
        }
        if let DeviceNvfp4RoutedExperts::Edge(edge) = &weights.routed_experts {
            let workspace = self
                .edge_workspace
                .as_ref()
                .ok_or_else(|| invalid("SM110 Edge NVFP4 workspace is unavailable"))?;
            // Diagnostic only, default off. The decode layout kernel
            // (`small_batch_setup_kernel`, tokens <= 4) writes only the live
            // `tokens * top_k` permutation slots and leaves the padded
            // remainder of `permuted_m` untouched, unlike the prefill
            // `build_layout_kernel` which fills padding with -1 behind a
            // barrier. `permuted_m` is 32,640 for a single decode token, so
            // 32,632 slots carry whatever the previous call left behind.
            // Setting APXINF_QWEN35_MOE_WS_POISON=<byte> fills the whole
            // workspace before every MoE call; if two runs with different
            // poison bytes produce different tokens, the generated FC1/FC2
            // kernels are reading past `tile_limit`.
            if let Some(byte) = std::env::var("APXINF_QWEN35_MOE_WS_POISON")
                .ok()
                .and_then(|value| value.parse::<u8>().ok())
            {
                workspace
                    .memset_async(byte, ctx.stream())
                    .map_err(Error::Cuda)?;
            }
            let mut routed_output = CudaBuffer::from_tensor(&self.assignment_output)
                .map_err(Error::Cuda)?
                .view(0, self.tokens * self.hidden * DType::BF16.size_in_bytes())
                .map_err(Error::Cuda)?
                .as_tensor(Shape::new(vec![self.tokens, self.hidden]), DType::BF16)
                .map_err(Error::Cuda)?;
            ops::edge_nvfp4_moe(
                ctx,
                EdgeNvfp4MoeArgs {
                    config,
                    intermediate: self.intermediate,
                    input,
                    expert_ids: &self.expert_ids,
                    expert_weights: &self.expert_weights,
                    weights: EdgeNvfp4MoeWeights {
                        fc1_weight: &edge.fc1_weight,
                        fc1_scales: &edge.fc1_scales,
                        fc1_alpha: &edge.fc1_alpha,
                        input_global_scale: &edge.input_global_scale,
                        fc2_weight: &edge.fc2_weight,
                        fc2_scales: &edge.fc2_scales,
                        fc2_alpha: &edge.fc2_alpha,
                        down_input_scale: &edge.down_input_scale,
                    },
                    workspace,
                    output: &mut routed_output,
                },
            )?;
            self.run_shared_nvfp4(ctx, weights, input)?;
            if self.tokens == 1 && std::env::var("APXINF_QWEN35_GATE_GEMV").as_deref() == Ok("1") {
                ops::bf16_gemv(
                    ctx,
                    input,
                    &weights.shared_expert_gate.weight,
                    &self.shared_gate_logits,
                )?;
            } else {
                ops::gemm(
                    ctx,
                    ops::GemmArgs::new(
                        input,
                        &weights.shared_expert_gate.weight,
                        &mut self.shared_gate_logits,
                    ),
                )?;
            }
            let shared_gate_logits =
                view(&self.shared_gate_logits, vec![self.tokens], DType::BF16)?;
            return ops::routed_moe_add_shared(
                ctx,
                RoutedMoeAddSharedArgs {
                    routed_output: &routed_output,
                    shared_output: &self.shared_output,
                    shared_gate_logits: &shared_gate_logits,
                    output: &mut self.output,
                },
            );
        }

        let (expert_gate, expert_up, expert_down) = match &weights.routed_experts {
            DeviceNvfp4RoutedExperts::Reference { gate, up, down } => (gate, up, down),
            DeviceNvfp4RoutedExperts::Edge(_) => unreachable!(),
        };
        if let (
            true,
            Some(expert_fused),
            Some(act_packed),
            Some(act_scales),
            Some(inter_packed),
            Some(inter_scales),
        ) = (
            nvfp4_ok,
            self.expert_fused.as_ref(),
            self.act_packed.as_ref(),
            self.act_scales.as_ref(),
            self.inter_packed.as_ref(),
            self.inter_scales.as_ref(),
        ) {
            // --- NVFP4 cuBLASLt per-expert path ---
            // D2H the expert offsets and per-expert scales so the CPU loop knows
            // how many tokens each expert owns and what its per-tensor alpha is.
            let host_offsets = transfers::to_cpu(&self.expert_offsets)?;
            let off: Vec<i32> = host_offsets.as_i32()?.to_vec();
            let host_scales = transfers::to_cpu(&expert_gate.input_scales)?;
            let input_scales: Vec<f32> = host_scales.as_f32()?.to_vec();
            let host_ws2_gate = transfers::to_cpu(&expert_gate.weight_scales_2)?;
            let ws2_gate: Vec<f32> = host_ws2_gate.as_f32()?.to_vec();
            let host_ws2_up = transfers::to_cpu(&expert_up.weight_scales_2)?;
            let ws2_up: Vec<f32> = host_ws2_up.as_f32()?.to_vec();
            let host_ws2_down = transfers::to_cpu(&expert_down.weight_scales_2)?;
            let ws2_down: Vec<f32> = host_ws2_down.as_f32()?.to_vec();

            fn nvfp4_gemm_expert(
                ctx: &CudaContext,
                a: &Tensor,
                a_s: &Tensor,
                b: &Tensor,
                b_s: &Tensor,
                alpha: f32,
                out: &mut Tensor,
            ) -> Result<()> {
                let mut args = ops::GemmArgs::nvfp4(a, a_s, b, b_s, NVFP4_BLOCK_SIZE, alpha, out);
                args.policy.allow_fallback = false;
                ops::gemm(ctx, args)
            }

            // ---------- routed experts ----------
            for e in 0..self.experts {
                let start = off[e] as usize;
                let end = off.get(e + 1).copied().unwrap_or(0) as usize;
                let expert_tokens = end.saturating_sub(start);
                if expert_tokens == 0 {
                    continue;
                }
                let alpha_gate = input_scales[e] * ws2_gate[e];
                let alpha_up = input_scales[e] * ws2_up[e];
                let alpha_down = input_scales[e] * ws2_down[e];

                let ids_view = i32_slice(&self.assignment_token_ids, start, expert_tokens)?;
                let gather_offset = start * self.hidden * DType::BF16.size_in_bytes();
                let gather_bytes = expert_tokens * self.hidden * DType::BF16.size_in_bytes();
                let gather_out = CudaBuffer::from_tensor(&self.assignment_output)
                    .map_err(Error::Cuda)?
                    .view(gather_offset, gather_bytes)
                    .map_err(Error::Cuda)?
                    .as_tensor(Shape::new(vec![expert_tokens, self.hidden]), DType::BF16)
                    .map_err(Error::Cuda)?;
                ops::embedding_gather(ctx, input, &ids_view, &gather_out)?;
                // After gather, the expert's input is in the assignment_output
                // slice. Quantise it in-place inside the per-expert packed
                // workspace, then gate/up → swiglu → down writes the final
                // result back over the same slice.
                let packed_view = view(
                    act_packed,
                    vec![expert_tokens, self.hidden / 2],
                    DType::E2M1Pair,
                )?;
                let scales_view = view(
                    act_scales,
                    vec![expert_tokens, self.hidden / 16],
                    DType::F8E4M3,
                )?;
                ops::nvfp4_quantize_activation(
                    ctx,
                    &gather_out,
                    &packed_view,
                    &scales_view,
                    input_scales[e],
                    NVFP4_BLOCK_SIZE,
                    ops::ScaleLayout::RowMajor,
                )?;

                let gw = expert_weight_view(
                    &expert_gate.packed_weight,
                    e,
                    self.intermediate,
                    self.hidden / 2,
                    DType::E2M1Pair,
                )?;
                let gws = expert_weight_view(
                    &expert_gate.block_scales,
                    e,
                    self.intermediate,
                    self.hidden / 16,
                    DType::F8E4M3,
                )?;
                let uw = expert_weight_view(
                    &expert_up.packed_weight,
                    e,
                    self.intermediate,
                    self.hidden / 2,
                    DType::E2M1Pair,
                )?;
                let uws = expert_weight_view(
                    &expert_up.block_scales,
                    e,
                    self.intermediate,
                    self.hidden / 16,
                    DType::F8E4M3,
                )?;
                let dw = expert_weight_view(
                    &expert_down.packed_weight,
                    e,
                    self.hidden,
                    self.intermediate / 2,
                    DType::E2M1Pair,
                )?;
                let dws = expert_weight_view(
                    &expert_down.block_scales,
                    e,
                    self.hidden,
                    self.intermediate / 16,
                    DType::F8E4M3,
                )?;

                let mut gate_out = view(
                    expert_fused,
                    vec![expert_tokens, self.intermediate],
                    DType::BF16,
                )?;
                nvfp4_gemm_expert(
                    ctx,
                    &packed_view,
                    &scales_view,
                    &gw,
                    &gws,
                    alpha_gate,
                    &mut gate_out,
                )?;
                let up_offset_bytes =
                    expert_tokens * self.intermediate * DType::BF16.size_in_bytes();
                let mut up_out = CudaBuffer::from_tensor(expert_fused)
                    .map_err(Error::Cuda)?
                    .view(
                        up_offset_bytes,
                        expert_tokens * self.intermediate * DType::BF16.size_in_bytes(),
                    )
                    .map_err(Error::Cuda)?
                    .as_tensor(
                        Shape::new(vec![expert_tokens, self.intermediate]),
                        DType::BF16,
                    )
                    .map_err(Error::Cuda)?;
                nvfp4_gemm_expert(
                    ctx,
                    &packed_view,
                    &scales_view,
                    &uw,
                    &uws,
                    alpha_up,
                    &mut up_out,
                )?;

                let mut inter_out = view(
                    &self.route_intermediate,
                    vec![expert_tokens, self.intermediate],
                    DType::BF16,
                )?;
                ops::swiglu_split(ctx, &gate_out, &up_out, &mut inter_out)?;

                let inter_p = view(
                    inter_packed,
                    vec![expert_tokens, self.intermediate / 2],
                    DType::E2M1Pair,
                )?;
                let inter_s = view(
                    inter_scales,
                    vec![expert_tokens, self.intermediate / 16],
                    DType::F8E4M3,
                )?;
                ops::nvfp4_quantize_activation(
                    ctx,
                    &inter_out,
                    &inter_p,
                    &inter_s,
                    1.0,
                    NVFP4_BLOCK_SIZE,
                    ops::ScaleLayout::RowMajor,
                )?;

                let mut down_out = CudaBuffer::from_tensor(&self.assignment_output)
                    .map_err(Error::Cuda)?
                    .view(gather_offset, gather_bytes)
                    .map_err(Error::Cuda)?
                    .as_tensor(Shape::new(vec![expert_tokens, self.hidden]), DType::BF16)
                    .map_err(Error::Cuda)?;
                nvfp4_gemm_expert(
                    ctx,
                    &inter_p,
                    &inter_s,
                    &dw,
                    &dws,
                    alpha_down,
                    &mut down_out,
                )?;
            }

            // ---------- shared expert ----------
            let s_alpha_gate = weights.shared_gate.input_scale;
            let s_alpha_up = weights.shared_up.input_scale;
            let s_alpha_down = weights.shared_down.input_scale;

            let s_packed = view(
                act_packed,
                vec![self.tokens, self.hidden / 2],
                DType::E2M1Pair,
            )?;
            let s_scales = view(
                act_scales,
                vec![self.tokens, self.hidden / 16],
                DType::F8E4M3,
            )?;
            ops::nvfp4_quantize_activation(
                ctx,
                input,
                &s_packed,
                &s_scales,
                s_alpha_gate,
                NVFP4_BLOCK_SIZE,
                ops::ScaleLayout::RowMajor,
            )?;

            let mut s_gate = view(
                expert_fused,
                vec![self.tokens, self.intermediate],
                DType::BF16,
            )?;
            nvfp4_gemm_expert(
                ctx,
                &s_packed,
                &s_scales,
                &weights.shared_gate.packed_weight,
                &weights.shared_gate.block_scales,
                s_alpha_gate,
                &mut s_gate,
            )?;
            let s_up_offset = self.tokens * self.intermediate * DType::BF16.size_in_bytes();
            let mut s_up = CudaBuffer::from_tensor(expert_fused)
                .map_err(Error::Cuda)?
                .view(
                    s_up_offset,
                    self.tokens * self.intermediate * DType::BF16.size_in_bytes(),
                )
                .map_err(Error::Cuda)?
                .as_tensor(
                    Shape::new(vec![self.tokens, self.intermediate]),
                    DType::BF16,
                )
                .map_err(Error::Cuda)?;
            nvfp4_gemm_expert(
                ctx,
                &s_packed,
                &s_scales,
                &weights.shared_up.packed_weight,
                &weights.shared_up.block_scales,
                s_alpha_up,
                &mut s_up,
            )?;

            ops::swiglu_split(ctx, &s_gate, &s_up, &self.shared_intermediate)?;

            let s_inter_p = view(
                inter_packed,
                vec![self.tokens, self.intermediate / 2],
                DType::E2M1Pair,
            )?;
            let s_inter_s = view(
                inter_scales,
                vec![self.tokens, self.intermediate / 16],
                DType::F8E4M3,
            )?;
            ops::nvfp4_quantize_activation(
                ctx,
                &self.shared_intermediate,
                &s_inter_p,
                &s_inter_s,
                1.0,
                NVFP4_BLOCK_SIZE,
                ops::ScaleLayout::RowMajor,
            )?;

            nvfp4_gemm_expert(
                ctx,
                &s_inter_p,
                &s_inter_s,
                &weights.shared_down.packed_weight,
                &weights.shared_down.block_scales,
                s_alpha_down,
                &mut self.shared_output,
            )?;
        } else {
            // --- original correctness-first warp kernel ---
            ops::routed_moe_nvfp4_experts(
                ctx,
                RoutedMoeNvfp4ExpertsArgs {
                    config,
                    intermediate: self.intermediate,
                    block_size: NVFP4_BLOCK_SIZE,
                    scale_layout: ops::ScaleLayout::RowMajor,
                    input,
                    assignment_expert_ids: &self.assignment_expert_ids,
                    assignment_token_ids: &self.assignment_token_ids,
                    expert_gate: Nvfp4ExpertProjection {
                        packed_weight: &expert_gate.packed_weight,
                        block_scales: &expert_gate.block_scales,
                        alpha: &expert_gate.weight_scales_2,
                    },
                    expert_up: Nvfp4ExpertProjection {
                        packed_weight: &expert_up.packed_weight,
                        block_scales: &expert_up.block_scales,
                        alpha: &expert_up.weight_scales_2,
                    },
                    expert_down: Nvfp4ExpertProjection {
                        packed_weight: &expert_down.packed_weight,
                        block_scales: &expert_down.block_scales,
                        alpha: &expert_down.weight_scales_2,
                    },
                    shared_gate: Nvfp4ExpertProjection {
                        packed_weight: &weights.shared_gate.packed_weight,
                        block_scales: &weights.shared_gate.block_scales,
                        alpha: &weights.shared_gate.weight_scale_2,
                    },
                    shared_up: Nvfp4ExpertProjection {
                        packed_weight: &weights.shared_up.packed_weight,
                        block_scales: &weights.shared_up.block_scales,
                        alpha: &weights.shared_up.weight_scale_2,
                    },
                    shared_down: Nvfp4ExpertProjection {
                        packed_weight: &weights.shared_down.packed_weight,
                        block_scales: &weights.shared_down.block_scales,
                        alpha: &weights.shared_down.weight_scale_2,
                    },
                    route_intermediate: &mut self.route_intermediate,
                    shared_intermediate: &mut self.shared_intermediate,
                    assignment_output: &mut self.assignment_output,
                    shared_output: &mut self.shared_output,
                },
            )?;
        }
        ops::gemm(
            ctx,
            ops::GemmArgs::new(
                input,
                &weights.shared_expert_gate.weight,
                &mut self.shared_gate_logits,
            ),
        )?;
        let shared_gate_logits = view(&self.shared_gate_logits, vec![self.tokens], DType::BF16)?;
        ops::routed_moe_combine(
            ctx,
            RoutedMoeCombineArgs {
                assignment_output: &self.assignment_output,
                assignment_token_ids: &self.assignment_token_ids,
                assignment_weights: &self.assignment_weights,
                token_assignment_indices: &self.token_assignment_indices,
                shared_output: &self.shared_output,
                shared_gate_logits: &shared_gate_logits,
                output: &mut self.output,
            },
        )
    }

    fn run_shared_nvfp4(
        &mut self,
        ctx: &CudaContext,
        weights: &DeviceNvfp4Moe,
        input: &Tensor,
    ) -> Result<()> {
        let expert_fused = self
            .expert_fused
            .as_ref()
            .ok_or_else(|| invalid("shared NVFP4 fused scratch is unavailable"))?;
        let act_packed = self
            .act_packed
            .as_ref()
            .ok_or_else(|| invalid("shared NVFP4 activation scratch is unavailable"))?;
        let act_scales = self
            .act_scales
            .as_ref()
            .ok_or_else(|| invalid("shared NVFP4 activation scales are unavailable"))?;
        let inter_packed = self
            .inter_packed
            .as_ref()
            .ok_or_else(|| invalid("shared NVFP4 intermediate scratch is unavailable"))?;
        let inter_scales = self
            .inter_scales
            .as_ref()
            .ok_or_else(|| invalid("shared NVFP4 intermediate scales are unavailable"))?;

        if matches!(self.tokens, 1 | 2 | 4 | 8 | 16) {
            if let (Some(gate), Some(up), Some(down)) = (
                weights.shared_gate.a16.as_ref(),
                weights.shared_up.a16.as_ref(),
                weights.shared_down.a16.as_ref(),
            ) {
                let half_fused_bytes =
                    self.tokens * self.intermediate * DType::BF16.size_in_bytes();
                let gate_output = CudaBuffer::from_tensor(expert_fused)
                    .map_err(Error::Cuda)?
                    .view(0, half_fused_bytes)
                    .map_err(Error::Cuda)?
                    .as_tensor(
                        Shape::new(vec![self.tokens, self.intermediate]),
                        DType::BF16,
                    )
                    .map_err(Error::Cuda)?;
                let up_output = CudaBuffer::from_tensor(expert_fused)
                    .map_err(Error::Cuda)?
                    .view(half_fused_bytes, half_fused_bytes)
                    .map_err(Error::Cuda)?
                    .as_tensor(
                        Shape::new(vec![self.tokens, self.intermediate]),
                        DType::BF16,
                    )
                    .map_err(Error::Cuda)?;
                ops::nvfp4_a16_blackwell_gemv(
                    ctx,
                    input,
                    &gate.qweight,
                    &gate.block_scales,
                    &gate.global_scale,
                    &gate_output,
                )?;
                ops::nvfp4_a16_blackwell_gemv(
                    ctx,
                    input,
                    &up.qweight,
                    &up.block_scales,
                    &up.global_scale,
                    &up_output,
                )?;
                ops::swiglu_split(ctx, &gate_output, &up_output, &self.shared_intermediate)?;
                return ops::nvfp4_a16_blackwell_gemv(
                    ctx,
                    &self.shared_intermediate,
                    &down.qweight,
                    &down.block_scales,
                    &down.global_scale,
                    &self.shared_output,
                );
            }
        }

        let nvfp4_gemm = |a: &Tensor,
                          a_scales: &Tensor,
                          b: &Tensor,
                          b_scales: &Tensor,
                          alpha: f32,
                          output: &mut Tensor|
         -> Result<()> {
            let mut args =
                ops::GemmArgs::nvfp4(a, a_scales, b, b_scales, NVFP4_BLOCK_SIZE, alpha, output);
            args.policy.allow_fallback = false;
            ops::gemm(ctx, args)
        };

        let packed = view(
            act_packed,
            vec![self.tokens, self.hidden / 2],
            DType::E2M1Pair,
        )?;
        let scales = act_scales;
        let half_fused_bytes = self.tokens * self.intermediate * DType::BF16.size_in_bytes();
        let mut gate_output = CudaBuffer::from_tensor(expert_fused)
            .map_err(Error::Cuda)?
            .view(0, half_fused_bytes)
            .map_err(Error::Cuda)?
            .as_tensor(
                Shape::new(vec![self.tokens, self.intermediate]),
                DType::BF16,
            )
            .map_err(Error::Cuda)?;
        ops::nvfp4_quantize_activation(
            ctx,
            input,
            &packed,
            &scales,
            weights.shared_gate.input_scale,
            NVFP4_BLOCK_SIZE,
            ops::ScaleLayout::GemmAtom,
        )?;
        let gate_weight_scales = weights
            .shared_gate
            .gemm_block_scales
            .as_ref()
            .ok_or_else(|| invalid("shared gate GEMM scales are unavailable"))?;
        nvfp4_gemm(
            &packed,
            scales,
            &weights.shared_gate.packed_weight,
            gate_weight_scales,
            weights.shared_gate.input_scale * weights.shared_gate.weight_scale_2_value,
            &mut gate_output,
        )?;

        let up_offset = half_fused_bytes;
        let mut up_output = CudaBuffer::from_tensor(expert_fused)
            .map_err(Error::Cuda)?
            .view(
                up_offset,
                self.tokens * self.intermediate * DType::BF16.size_in_bytes(),
            )
            .map_err(Error::Cuda)?
            .as_tensor(
                Shape::new(vec![self.tokens, self.intermediate]),
                DType::BF16,
            )
            .map_err(Error::Cuda)?;
        ops::nvfp4_quantize_activation(
            ctx,
            input,
            &packed,
            &scales,
            weights.shared_up.input_scale,
            NVFP4_BLOCK_SIZE,
            ops::ScaleLayout::GemmAtom,
        )?;
        let up_weight_scales = weights
            .shared_up
            .gemm_block_scales
            .as_ref()
            .ok_or_else(|| invalid("shared up GEMM scales are unavailable"))?;
        nvfp4_gemm(
            &packed,
            scales,
            &weights.shared_up.packed_weight,
            up_weight_scales,
            weights.shared_up.input_scale * weights.shared_up.weight_scale_2_value,
            &mut up_output,
        )?;

        ops::swiglu_split(ctx, &gate_output, &up_output, &self.shared_intermediate)?;
        let inter_packed = view(
            inter_packed,
            vec![self.tokens, self.intermediate / 2],
            DType::E2M1Pair,
        )?;
        ops::nvfp4_quantize_activation(
            ctx,
            &self.shared_intermediate,
            &inter_packed,
            inter_scales,
            weights.shared_down.input_scale,
            NVFP4_BLOCK_SIZE,
            ops::ScaleLayout::GemmAtom,
        )?;
        let down_weight_scales = weights
            .shared_down
            .gemm_block_scales
            .as_ref()
            .ok_or_else(|| invalid("shared down GEMM scales are unavailable"))?;
        nvfp4_gemm(
            &inter_packed,
            inter_scales,
            &weights.shared_down.packed_weight,
            down_weight_scales,
            weights.shared_down.input_scale * weights.shared_down.weight_scale_2_value,
            &mut self.shared_output,
        )
    }

    pub fn output(&self) -> &Tensor {
        &self.output
    }

    fn validate(&self, ctx: &CudaContext, weights: &DeviceNvfp4Moe, input: &Tensor) -> Result<()> {
        validate_tensor(
            input,
            ctx,
            DType::BF16,
            &[self.tokens, self.hidden],
            "MoE input",
        )?;
        validate_linear(
            &weights.router.weight,
            ctx,
            self.hidden,
            self.experts,
            "MoE router",
        )?;
        validate_linear(
            &weights.shared_expert_gate.weight,
            ctx,
            self.hidden,
            1,
            "shared expert gate",
        )?;
        match &weights.routed_experts {
            DeviceNvfp4RoutedExperts::Reference { gate, up, down } => {
                for (projection, input_features, output_features, label) in [
                    (gate, self.hidden, self.intermediate, "expert gate"),
                    (up, self.hidden, self.intermediate, "expert up"),
                    (down, self.intermediate, self.hidden, "expert down"),
                ] {
                    if projection.experts != self.experts
                        || projection.input_features != input_features
                        || projection.output_features != output_features
                    {
                        return Err(invalid(format!("Qwen3.5 {label} geometry mismatch")));
                    }
                }
            }
            DeviceNvfp4RoutedExperts::Edge(edge) => {
                if weights.activation != Qwen35MoeActivation::Nvfp4 {
                    return Err(invalid(
                        "Qwen3.5 Edge NVFP4 experts require runtime NVFP4 activations",
                    ));
                }
                if edge.experts != self.experts
                    || edge.hidden != self.hidden
                    || edge.intermediate != self.intermediate
                {
                    return Err(invalid("Qwen3.5 Edge NVFP4 expert geometry mismatch"));
                }
            }
        }
        for (projection, input_features, output_features, label) in [
            (
                &weights.shared_gate,
                self.hidden,
                self.intermediate,
                "shared gate",
            ),
            (
                &weights.shared_up,
                self.hidden,
                self.intermediate,
                "shared up",
            ),
            (
                &weights.shared_down,
                self.intermediate,
                self.hidden,
                "shared down",
            ),
        ] {
            if projection.input_features != input_features
                || projection.output_features != output_features
            {
                return Err(invalid(format!("Qwen3.5 {label} geometry mismatch")));
            }
        }
        Ok(())
    }
}

/// Persistent recurrent state owned by one base GDN layer.
#[derive(Debug)]
pub struct GdnState {
    conv: Tensor,
    recurrent: Tensor,
}

/// Four-row target verification checkpoints for one GDN layer.
#[derive(Debug)]
pub(super) struct GdnStateCheckpoints {
    conv: Tensor,
    recurrent: Tensor,
    rows: usize,
}

impl GdnStateCheckpoints {
    pub(super) fn new(ctx: &CudaContext, rows: usize) -> Result<Self> {
        if rows == 0 {
            return Err(invalid("Qwen3.5 GDN checkpoints require at least one row"));
        }
        Ok(Self {
            conv: zeros(
                ctx,
                vec![rows, ops::QWEN35_GDN_QKV_WIDTH, ops::QWEN35_GDN_CONV_WIDTH],
                DType::F32,
            )?,
            recurrent: zeros(
                ctx,
                vec![
                    rows,
                    ops::QWEN35_GDN_VALUE_HEADS,
                    ops::QWEN35_GDN_HEAD_DIM,
                    ops::QWEN35_GDN_HEAD_DIM,
                ],
                DType::F32,
            )?,
            rows,
        })
    }

    pub(super) fn copy_row_to(
        &self,
        ctx: &CudaContext,
        row: usize,
        destination: &GdnState,
    ) -> Result<()> {
        if row >= self.rows {
            return Err(invalid("Qwen3.5 GDN checkpoint row is out of range"));
        }
        let conv = tensor_row_view(
            &self.conv,
            row,
            ops::QWEN35_GDN_QKV_WIDTH * ops::QWEN35_GDN_CONV_WIDTH,
            vec![ops::QWEN35_GDN_QKV_WIDTH, ops::QWEN35_GDN_CONV_WIDTH],
            DType::F32,
        )?;
        let recurrent = tensor_row_view(
            &self.recurrent,
            row,
            ops::QWEN35_GDN_VALUE_HEADS * ops::QWEN35_GDN_HEAD_DIM * ops::QWEN35_GDN_HEAD_DIM,
            vec![
                ops::QWEN35_GDN_VALUE_HEADS,
                ops::QWEN35_GDN_HEAD_DIM,
                ops::QWEN35_GDN_HEAD_DIM,
            ],
            DType::F32,
        )?;
        transfers::copy_cuda_to_cuda(ctx, &conv, &destination.conv)?;
        transfers::copy_cuda_to_cuda(ctx, &recurrent, &destination.recurrent)
    }

    fn core(&self) -> ops::Qwen35GdnStateCheckpoints<'_> {
        ops::Qwen35GdnStateCheckpoints {
            conv: &self.conv,
            recurrent: &self.recurrent,
        }
    }
}

impl GdnState {
    pub fn new(ctx: &CudaContext) -> Result<Self> {
        Ok(Self {
            conv: zeros(
                ctx,
                vec![ops::QWEN35_GDN_QKV_WIDTH, ops::QWEN35_GDN_CONV_WIDTH],
                DType::F32,
            )?,
            recurrent: zeros(
                ctx,
                vec![
                    ops::QWEN35_GDN_VALUE_HEADS,
                    ops::QWEN35_GDN_HEAD_DIM,
                    ops::QWEN35_GDN_HEAD_DIM,
                ],
                DType::F32,
            )?,
        })
    }

    /// Reset at a public sequence boundary without a host staging transfer.
    pub fn reset(&self, ctx: &CudaContext) -> Result<()> {
        for tensor in [&self.conv, &self.recurrent] {
            CudaBuffer::from_tensor(tensor)
                .map_err(Error::Cuda)?
                .zero_async(ctx.stream())
                .map_err(Error::Cuda)?;
        }
        Ok(())
    }

    /// Copy a committed recurrent state into a shadow state on the context
    /// stream.  Decoder transactions run exclusively against the shadow and
    /// swap it into the committed slot only after the outer CUDA synchronize
    /// succeeds, so a later layer failure cannot advance part of a sequence.
    pub fn copy_from(&self, ctx: &CudaContext, source: &Self) -> Result<()> {
        transfers::copy_cuda_to_cuda(ctx, &source.conv, &self.conv)?;
        transfers::copy_cuda_to_cuda(ctx, &source.recurrent, &self.recurrent)
    }

    pub(super) fn first_mismatch(
        &self,
        other: &Self,
    ) -> Result<Option<(&'static str, usize, u32, u32)>> {
        for (label, left, right) in [
            ("conv", &self.conv, &other.conv),
            ("recurrent", &self.recurrent, &other.recurrent),
        ] {
            let left = transfers::to_cpu(left)?;
            let right = transfers::to_cpu(right)?;
            for (index, (left, right)) in left.as_f32()?.iter().zip(right.as_f32()?).enumerate() {
                if left.to_bits() != right.to_bits() {
                    return Ok(Some((label, index, left.to_bits(), right.to_bits())));
                }
            }
        }
        Ok(None)
    }
}

/// Projection, core, and output buffers for one base GDN layer/profile.
///
/// The `gdn_*16`/`cake_*` fields back the FlashInfer Cake prefill path and are
/// allocated only on Blackwell, where that kernel exists. Decode keeps using
/// `convolved_qkv`/`core_output`, so those stay regardless.
#[derive(Debug)]
pub struct GdnScratch {
    fp8_qkv_input: Tensor,
    fp8_z_input: Tensor,
    fp8_output_input: Tensor,
    projected_qkv: Tensor,
    z: Tensor,
    a: Tensor,
    b: Tensor,
    convolved_qkv: Tensor,
    decay: Tensor,
    beta: Tensor,
    core_output: Tensor,
    gated_output: Tensor,
    output: Tensor,
    q16: Option<Tensor>,
    k16: Option<Tensor>,
    v16: Option<Tensor>,
    out16: Option<Tensor>,
    alpha: Option<Tensor>,
    cu_seqlens: Option<Tensor>,
    cake_workspace: Option<Tensor>,
    tokens: usize,
    hidden: usize,
}

impl GdnScratch {
    pub fn new(ctx: &CudaContext, tokens: usize, hidden: usize) -> Result<Self> {
        if tokens == 0 || hidden == 0 {
            return Err(invalid("Qwen3.5 GDN profile dimensions must be non-zero"));
        }
        // The Cake prefill kernel is Blackwell-only, matching `d3`'s gate.
        let cake = ctx.caps().sm >= 100;
        let key_dims = vec![tokens, ops::QWEN35_GDN_KEY_HEADS, ops::QWEN35_GDN_HEAD_DIM];
        let value_dims = vec![
            tokens,
            ops::QWEN35_GDN_VALUE_HEADS,
            ops::QWEN35_GDN_HEAD_DIM,
        ];
        let workspace_bytes = ops::gdn_prefill_workspace_bytes(ops::QWEN35_GDN_VALUE_HEADS, 1);
        Ok(Self {
            fp8_qkv_input: zeros(ctx, vec![tokens, hidden], DType::F8E4M3)?,
            fp8_z_input: zeros(ctx, vec![tokens, hidden], DType::F8E4M3)?,
            fp8_output_input: zeros(
                ctx,
                vec![tokens, ops::QWEN35_GDN_VALUE_WIDTH],
                DType::F8E4M3,
            )?,
            projected_qkv: zeros(ctx, vec![tokens, ops::QWEN35_GDN_QKV_WIDTH], DType::BF16)?,
            z: zeros(ctx, vec![tokens, ops::QWEN35_GDN_VALUE_WIDTH], DType::BF16)?,
            a: zeros(ctx, vec![tokens, ops::QWEN35_GDN_VALUE_HEADS], DType::BF16)?,
            b: zeros(ctx, vec![tokens, ops::QWEN35_GDN_VALUE_HEADS], DType::BF16)?,
            convolved_qkv: zeros(ctx, vec![tokens, ops::QWEN35_GDN_QKV_WIDTH], DType::BF16)?,
            decay: zeros(ctx, vec![tokens, ops::QWEN35_GDN_VALUE_HEADS], DType::F32)?,
            beta: zeros(ctx, vec![tokens, ops::QWEN35_GDN_VALUE_HEADS], DType::F32)?,
            core_output: zeros(ctx, vec![tokens, ops::QWEN35_GDN_VALUE_WIDTH], DType::BF16)?,
            gated_output: zeros(ctx, vec![tokens, ops::QWEN35_GDN_VALUE_WIDTH], DType::BF16)?,
            output: zeros(ctx, vec![tokens, hidden], DType::BF16)?,
            q16: cake
                .then(|| zeros(ctx, key_dims.clone(), DType::F16))
                .transpose()?,
            k16: cake.then(|| zeros(ctx, key_dims, DType::F16)).transpose()?,
            v16: cake
                .then(|| zeros(ctx, value_dims.clone(), DType::F16))
                .transpose()?,
            out16: cake
                .then(|| zeros(ctx, value_dims, DType::F16))
                .transpose()?,
            alpha: cake
                .then(|| zeros(ctx, vec![tokens, ops::QWEN35_GDN_VALUE_HEADS], DType::F32))
                .transpose()?,
            cu_seqlens: cake.then(|| zeros(ctx, vec![2], DType::I32)).transpose()?,
            cake_workspace: cake
                .then(|| zeros(ctx, vec![workspace_bytes.div_ceil(4)], DType::F32))
                .transpose()?,
            tokens,
            hidden,
        })
    }

    /// Run all four BF16 projections, exact prompt GDN state update, gated
    /// normalization, and the output projection on CUDA.
    pub fn prefill(
        &mut self,
        ctx: &CudaContext,
        weights: &DeviceGdnWeights,
        state: &GdnState,
        normalized_hidden: &Tensor,
    ) -> Result<()> {
        self.prefill_impl(ctx, weights, state, normalized_hidden, None)
    }

    pub(super) fn prefill_with_checkpoints(
        &mut self,
        ctx: &CudaContext,
        weights: &DeviceGdnWeights,
        state: &GdnState,
        normalized_hidden: &Tensor,
        checkpoints: &GdnStateCheckpoints,
    ) -> Result<()> {
        self.prefill_impl(ctx, weights, state, normalized_hidden, Some(checkpoints))
    }

    fn prefill_impl(
        &mut self,
        ctx: &CudaContext,
        weights: &DeviceGdnWeights,
        state: &GdnState,
        normalized_hidden: &Tensor,
        checkpoints: Option<&GdnStateCheckpoints>,
    ) -> Result<()> {
        self.validate(ctx, weights, state, normalized_hidden)?;
        if checkpoints.is_some_and(|checkpoints| checkpoints.rows != self.tokens) {
            return Err(invalid(
                "Qwen3.5 GDN checkpoint rows do not match the token profile",
            ));
        }
        self.project(ctx, weights, normalized_hidden)?;
        let z = view(
            &self.z,
            vec![self.tokens, ops::QWEN35_GDN_VALUE_WIDTH],
            DType::BF16,
        )?;
        ops::qwen35_gdn_prefill(
            ctx,
            self.tokens,
            &ops::Qwen35GdnPrefillArgs {
                projected_qkv: &self.projected_qkv,
                z: &z,
                a: &self.a,
                b: &self.b,
                weights: core_weights(weights),
                state: core_state(state),
                checkpoints: checkpoints.map(GdnStateCheckpoints::core),
                scratch: ops::Qwen35GdnPrefillScratch {
                    convolved_qkv: &self.convolved_qkv,
                    decay: &self.decay,
                    beta: &self.beta,
                    core_output: &self.core_output,
                    gated_output: &self.gated_output,
                },
                cake: self.cake_scratch(),
            },
        )?;
        self.output_projection(ctx, weights)
    }

    /// The FP16 buffers the Cake prefill kernel needs, present together or not
    /// at all. `None` on pre-Blackwell devices, where prefill takes the exact
    /// per-token path instead.
    fn cake_scratch(&self) -> Option<ops::Qwen35GdnCakeScratch<'_>> {
        Some(ops::Qwen35GdnCakeScratch {
            q16: self.q16.as_ref()?,
            k16: self.k16.as_ref()?,
            v16: self.v16.as_ref()?,
            out16: self.out16.as_ref()?,
            alpha: self.alpha.as_ref()?,
            cu_seqlens: self.cu_seqlens.as_ref()?,
            workspace: self.cake_workspace.as_ref()?,
        })
    }

    /// Advance a prepared single-token profile from the current recurrent
    /// state. This is the decode counterpart of [`Self::prefill`].
    pub fn decode(
        &mut self,
        ctx: &CudaContext,
        weights: &DeviceGdnWeights,
        state: &GdnState,
        normalized_hidden: &Tensor,
    ) -> Result<()> {
        self.decode_core(ctx, weights, state, normalized_hidden)?;
        self.output_projection(ctx, weights)
    }

    pub(super) fn decode_core(
        &mut self,
        ctx: &CudaContext,
        weights: &DeviceGdnWeights,
        state: &GdnState,
        normalized_hidden: &Tensor,
    ) -> Result<()> {
        self.prepare_decode_inputs(ctx, weights, state, normalized_hidden)?;
        self.decode_prepared_core(ctx, weights, state)
    }

    pub(super) fn prepare_decode_inputs(
        &mut self,
        ctx: &CudaContext,
        weights: &DeviceGdnWeights,
        state: &GdnState,
        normalized_hidden: &Tensor,
    ) -> Result<()> {
        if self.tokens != 1 {
            return Err(invalid(
                "Qwen3.5 GDN decode requires a one-token scratch profile",
            ));
        }
        self.validate(ctx, weights, state, normalized_hidden)?;
        self.project(ctx, weights, normalized_hidden)
    }

    /// Validate the fixed one-row projection contract before the projection
    /// GEMMs are split across two CUDA streams.
    pub(super) fn validate_parallel_decode_inputs(
        &self,
        ctx: &CudaContext,
        weights: &DeviceGdnWeights,
        state: &GdnState,
        normalized_hidden: &Tensor,
    ) -> Result<()> {
        if self.tokens != 1 {
            return Err(invalid(
                "Qwen3.5 parallel GDN decode requires a one-token scratch profile",
            ));
        }
        self.validate(ctx, weights, state, normalized_hidden)
    }

    /// Enqueue the widest GDN input projection. The caller owns the execution
    /// session and stream dependency.
    pub(super) fn project_decode_qkv(
        &mut self,
        ctx: &CudaContext,
        weights: &DeviceGdnWeights,
        input: &Tensor,
    ) -> Result<()> {
        project_static_fp8(
            ctx,
            &weights.in_proj_qkv,
            input,
            &self.fp8_qkv_input,
            &mut self.projected_qkv,
            self.tokens,
        )
    }

    /// Enqueue the remaining independent GDN input projections. These share
    /// an auxiliary stream while QKV runs on the primary stream.
    pub(super) fn project_decode_zab(
        &mut self,
        ctx: &CudaContext,
        weights: &DeviceGdnWeights,
        input: &Tensor,
    ) -> Result<()> {
        project_static_fp8(
            ctx,
            &weights.in_proj_z,
            input,
            &self.fp8_z_input,
            &mut self.z,
            self.tokens,
        )?;
        for (weight, output) in [
            (&weights.in_proj_a.weight, &mut self.a),
            (&weights.in_proj_b.weight, &mut self.b),
        ] {
            ops::gemm(ctx, ops::GemmArgs::new(input, weight, output))?;
        }
        Ok(())
    }

    pub(super) fn decode_prepared_core(
        &mut self,
        ctx: &CudaContext,
        weights: &DeviceGdnWeights,
        state: &GdnState,
    ) -> Result<()> {
        let projected_qkv = view(
            &self.projected_qkv,
            vec![ops::QWEN35_GDN_QKV_WIDTH],
            DType::BF16,
        )?;
        let z = view(&self.z, vec![ops::QWEN35_GDN_VALUE_WIDTH], DType::BF16)?;
        let a = view(&self.a, vec![ops::QWEN35_GDN_VALUE_HEADS], DType::BF16)?;
        let b = view(&self.b, vec![ops::QWEN35_GDN_VALUE_HEADS], DType::BF16)?;
        let convolved_qkv = view(
            &self.convolved_qkv,
            vec![ops::QWEN35_GDN_QKV_WIDTH],
            DType::BF16,
        )?;
        let decay = view(&self.decay, vec![ops::QWEN35_GDN_VALUE_HEADS], DType::F32)?;
        let beta = view(&self.beta, vec![ops::QWEN35_GDN_VALUE_HEADS], DType::F32)?;
        let core_output = view(
            &self.core_output,
            vec![ops::QWEN35_GDN_VALUE_WIDTH],
            DType::BF16,
        )?;
        let gated_output = view(
            &self.gated_output,
            vec![ops::QWEN35_GDN_VALUE_WIDTH],
            DType::BF16,
        )?;
        ops::qwen35_gdn_decode(
            ctx,
            &ops::Qwen35GdnDecodeArgs {
                projected_qkv: &projected_qkv,
                z: &z,
                a: &a,
                b: &b,
                weights: core_weights(weights),
                state: core_state(state),
                scratch: ops::Qwen35GdnDecodeScratch {
                    convolved_qkv: &convolved_qkv,
                    decay: &decay,
                    beta: &beta,
                    core_output: &core_output,
                    gated_output: &gated_output,
                },
            },
        )
    }

    pub(super) fn finish_decode_projection(
        &mut self,
        ctx: &CudaContext,
        weights: &DeviceGdnWeights,
    ) -> Result<()> {
        self.output_projection(ctx, weights)
    }

    pub fn output(&self) -> &Tensor {
        &self.output
    }

    fn project(
        &mut self,
        ctx: &CudaContext,
        weights: &DeviceGdnWeights,
        input: &Tensor,
    ) -> Result<()> {
        self.project_decode_qkv(ctx, weights, input)?;
        self.project_decode_zab(ctx, weights, input)
    }

    fn output_projection(&mut self, ctx: &CudaContext, weights: &DeviceGdnWeights) -> Result<()> {
        let gated_output = view(
            &self.gated_output,
            vec![self.tokens, ops::QWEN35_GDN_VALUE_WIDTH],
            DType::BF16,
        )?;
        project_static_fp8(
            ctx,
            &weights.out_proj,
            &gated_output,
            &self.fp8_output_input,
            &mut self.output,
            self.tokens,
        )
    }

    fn validate(
        &self,
        ctx: &CudaContext,
        weights: &DeviceGdnWeights,
        state: &GdnState,
        input: &Tensor,
    ) -> Result<()> {
        validate_tensor(
            input,
            ctx,
            DType::BF16,
            &[self.tokens, self.hidden],
            "GDN normalized hidden",
        )?;
        for (linear, input_features, output_features, label) in [
            (
                &weights.in_proj_qkv,
                self.hidden,
                ops::QWEN35_GDN_QKV_WIDTH,
                "GDN qkv projection",
            ),
            (
                &weights.in_proj_z,
                self.hidden,
                ops::QWEN35_GDN_VALUE_WIDTH,
                "GDN z projection",
            ),
            (
                &weights.out_proj,
                ops::QWEN35_GDN_VALUE_WIDTH,
                self.hidden,
                "GDN output projection",
            ),
        ] {
            if linear.input_features != input_features || linear.output_features != output_features
            {
                return Err(invalid(format!("Qwen3.5 {label} metadata mismatch")));
            }
            validate_tensor(
                &linear.weight,
                ctx,
                DType::F8E4M3,
                &[output_features, input_features],
                label,
            )?;
            validate_tensor(
                &linear.transposed,
                ctx,
                DType::F8E4M3,
                &[input_features, output_features],
                label,
            )?;
        }
        for (linear, output_features, label) in [
            (
                &weights.in_proj_a,
                ops::QWEN35_GDN_VALUE_HEADS,
                "GDN a projection",
            ),
            (
                &weights.in_proj_b,
                ops::QWEN35_GDN_VALUE_HEADS,
                "GDN b projection",
            ),
        ] {
            if linear.input_features != self.hidden || linear.output_features != output_features {
                return Err(invalid(format!("Qwen3.6 {label} metadata mismatch")));
            }
            validate_linear(&linear.weight, ctx, self.hidden, output_features, label)?;
        }
        for (tensor, dtype, shape, label) in [
            (
                &weights.conv1d,
                DType::BF16,
                &[ops::QWEN35_GDN_QKV_WIDTH, 1, ops::QWEN35_GDN_CONV_WIDTH][..],
                "GDN conv",
            ),
            (
                &weights.a_log,
                DType::BF16,
                &[ops::QWEN35_GDN_VALUE_HEADS][..],
                "GDN A_log",
            ),
            (
                &weights.dt_bias,
                DType::BF16,
                &[ops::QWEN35_GDN_VALUE_HEADS][..],
                "GDN dt_bias",
            ),
            (
                &weights.norm,
                DType::BF16,
                &[ops::QWEN35_GDN_HEAD_DIM][..],
                "GDN gated norm",
            ),
            (
                &state.conv,
                DType::F32,
                &[ops::QWEN35_GDN_QKV_WIDTH, ops::QWEN35_GDN_CONV_WIDTH][..],
                "GDN convolution state",
            ),
            (
                &state.recurrent,
                DType::F32,
                &[
                    ops::QWEN35_GDN_VALUE_HEADS,
                    ops::QWEN35_GDN_HEAD_DIM,
                    ops::QWEN35_GDN_HEAD_DIM,
                ][..],
                "GDN recurrent state",
            ),
        ] {
            validate_tensor(tensor, ctx, dtype, shape, label)?;
        }
        Ok(())
    }
}

fn core_weights(weights: &DeviceGdnWeights) -> ops::Qwen35GdnCoreWeights<'_> {
    ops::Qwen35GdnCoreWeights {
        conv: &weights.conv1d,
        a_log: &weights.a_log,
        dt_bias: &weights.dt_bias,
        norm: &weights.norm,
    }
}

fn core_state(state: &GdnState) -> ops::Qwen35GdnState<'_> {
    ops::Qwen35GdnState {
        conv: &state.conv,
        recurrent: &state.recurrent,
    }
}

fn validate_linear(
    tensor: &Tensor,
    ctx: &CudaContext,
    input_features: usize,
    output_features: usize,
    label: &str,
) -> Result<()> {
    validate_tensor(
        tensor,
        ctx,
        DType::BF16,
        &[input_features, output_features],
        label,
    )
}

fn validate_tensor(
    tensor: &Tensor,
    ctx: &CudaContext,
    dtype: DType,
    shape: &[usize],
    label: &str,
) -> Result<()> {
    if tensor.device() != Device::Cuda(ctx.device_id())
        || tensor.dtype() != dtype
        || tensor.shape().dims() != shape
    {
        return Err(invalid(format!(
            "Qwen3.5 {label} expected CUDA:{} {dtype} {shape:?}, got {:?} {} {:?}",
            ctx.device_id(),
            tensor.device(),
            tensor.dtype(),
            tensor.shape().dims()
        )));
    }
    Ok(())
}

fn zeros(ctx: &CudaContext, shape: Vec<usize>, dtype: DType) -> Result<Tensor> {
    let bytes = shape
        .iter()
        .try_fold(dtype.size_in_bytes(), |bytes, dimension| {
            bytes.checked_mul(*dimension)
        })
        .ok_or_else(|| invalid("Qwen3.5 CUDA allocation size overflow"))?;
    Ok(CudaBuffer::alloc_zeros(bytes, ctx.device_id())
        .map_err(Error::Cuda)?
        .as_tensor(Shape::new(shape), dtype)
        .map_err(Error::Cuda)?)
}

fn view(tensor: &Tensor, shape: Vec<usize>, dtype: DType) -> Result<Tensor> {
    let bytes = shape
        .iter()
        .try_fold(dtype.size_in_bytes(), |bytes, dimension| {
            bytes.checked_mul(*dimension)
        })
        .ok_or_else(|| invalid("Qwen3.5 CUDA view size overflow"))?;
    CudaBuffer::from_tensor(tensor)
        .map_err(Error::Cuda)?
        .view(0, bytes)
        .map_err(Error::Cuda)?
        .as_tensor(Shape::new(shape), dtype)
        .map_err(Error::Cuda)
}

fn tensor_row_view(
    tensor: &Tensor,
    row: usize,
    row_elements: usize,
    shape: Vec<usize>,
    dtype: DType,
) -> Result<Tensor> {
    let element_size = dtype.size_in_bytes();
    let offset = row
        .checked_mul(row_elements)
        .and_then(|elements| elements.checked_mul(element_size))
        .ok_or_else(|| invalid("Qwen3.5 tensor row offset overflow"))?;
    let bytes = row_elements
        .checked_mul(element_size)
        .ok_or_else(|| invalid("Qwen3.5 tensor row size overflow"))?;
    CudaBuffer::from_tensor(tensor)
        .map_err(Error::Cuda)?
        .view(offset, bytes)
        .map_err(Error::Cuda)?
        .as_tensor(Shape::new(shape), dtype)
        .map_err(Error::Cuda)
}

/// A single expert's slice from a `[experts, rows, cols]` stacked tensor.
fn expert_weight_view(
    stacked: &Tensor,
    expert: usize,
    rows: usize,
    cols: usize,
    dtype: DType,
) -> Result<Tensor> {
    let elem_size = dtype.size_in_bytes();
    let expert_bytes = rows * cols * elem_size;
    let offset = expert * expert_bytes;
    CudaBuffer::from_tensor(stacked)
        .map_err(Error::Cuda)?
        .view(offset, expert_bytes)
        .map_err(Error::Cuda)?
        .as_tensor(Shape::new(vec![rows, cols]), dtype)
        .map_err(Error::Cuda)
}

/// View an `[N]` I32 tensor for `len` elements starting at `offset`.
fn i32_slice(tensor: &Tensor, offset: usize, len: usize) -> Result<Tensor> {
    let byte_offset = offset * 4;
    CudaBuffer::from_tensor(tensor)
        .map_err(Error::Cuda)?
        .view(byte_offset, len * 4)
        .map_err(Error::Cuda)?
        .as_tensor(Shape::new(vec![len]), DType::I32)
        .map_err(Error::Cuda)
}

fn invalid(message: impl Into<String>) -> Error {
    Error::Other(message.into())
}

fn zero_centered_rms_norm(
    ctx: &CudaContext,
    input: &Tensor,
    checkpoint_weight: &Tensor,
    output: &Tensor,
    eps: f32,
) -> Result<()> {
    ops::mlp::rms_norm(ctx, input, checkpoint_weight, output, eps)
}

#[cfg(test)]
mod tests {
    use half::bf16;

    use super::*;

    #[test]
    fn zero_centered_norm_matches_f32_one_plus_weight_reference() {
        let ctx = CudaContext::new(0).unwrap();
        let input_values = [
            bf16::from_f32(1.0),
            bf16::from_f32(-2.0),
            bf16::from_f32(3.0),
            bf16::from_f32(-4.0),
            bf16::from_f32(0.5),
            bf16::from_f32(1.5),
            bf16::from_f32(-2.5),
            bf16::from_f32(3.5),
        ];
        let weight_values = [
            bf16::from_f32(-0.5),
            bf16::from_f32(0.0),
            bf16::from_f32(0.25),
            bf16::from_f32(1.0),
        ];
        let input = transfers::to_cuda(
            &Tensor::from_bf16(vec![2, 4], &input_values).unwrap(),
            ctx.device_id(),
        )
        .unwrap();
        let weight = transfers::to_cuda(
            &Tensor::from_bf16(vec![4], &weight_values).unwrap(),
            ctx.device_id(),
        )
        .unwrap();
        let output = zeros(&ctx, vec![2, 4], DType::BF16).unwrap();
        let eps = 1.0e-6;
        zero_centered_rms_norm(&ctx, &input, &weight, &output, eps).unwrap();
        ctx.synchronize().unwrap();

        let actual = transfers::to_cpu(&output)
            .unwrap()
            .as_bf16()
            .unwrap()
            .iter()
            .map(|value| value.to_f32())
            .collect::<Vec<_>>();
        let expected = input_values
            .chunks_exact(4)
            .flat_map(|row| {
                let mean_square = row.iter().map(|value| value.to_f32().powi(2)).sum::<f32>() / 4.0;
                let inverse_rms = (mean_square + eps).sqrt().recip();
                row.iter().zip(&weight_values).map(move |(input, weight)| {
                    bf16::from_f32(input.to_f32() * inverse_rms * (1.0 + weight.to_f32())).to_f32()
                })
            })
            .collect::<Vec<_>>();
        assert_eq!(actual.len(), expected.len());
        for (index, (&actual, &expected)) in actual.iter().zip(&expected).enumerate() {
            assert!(
                (actual - expected).abs() <= 0.016,
                "zero-centered RMSNorm[{index}] {actual} != {expected}"
            );
        }
        // A raw checkpoint delta of zero must mean an affine of one, not an
        // output of zero as generic RMSNorm would produce.
        assert_ne!(actual[1], 0.0);
    }
}
