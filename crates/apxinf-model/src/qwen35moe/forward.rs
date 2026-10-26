//! Executable CUDA slices of the Qwen3.5 text path.
//!
//! These kernels back the registered [`crate::llm_trait::LlmTrait`] runtime,
//! including the target full-attention/FP8-KV backend and BF16 MTP executor.
//! Together with `decoder`/`session`, they keep inputs,
//! intermediates, GDN state, routing, expert output, and logits on CUDA and use
//! only cuda-new's safe L3 interfaces.

use apxinf_core::{DType, Device, Error, Result, Shape, Tensor};
use apxinf_cuda_new::{
    ops, transfers, CudaBuffer, CudaContext, Nvfp4ExpertProjection, RoutedMoeCombineArgs,
    RoutedMoeConfig, RoutedMoeNvfp4ExpertsArgs, RoutedMoeRouteArgs,
};

use super::device_weights::{DeviceGdnWeights, DeviceNvfp4Moe, Qwen35DeviceWeights};

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
        validate_tensor(
            &weights.lm_head,
            ctx,
            DType::BF16,
            &[self.hidden_size, self.vocab_size],
            "lm_head",
        )?;
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
        ops::gemm(
            ctx,
            ops::GemmArgs::new(&self.normalized, &weights.lm_head, &mut self.logits),
        )
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

/// Stable, capture-compatible scratch for one base routed-MoE invocation.
#[derive(Debug)]
pub struct BaseMoeScratch {
    router_logits: Tensor,
    expert_ids: Tensor,
    expert_weights: Tensor,
    assignment_expert_ids: Tensor,
    assignment_token_ids: Tensor,
    assignment_slots: Tensor,
    assignment_weights: Tensor,
    expert_offsets: Tensor,
    route_intermediate: Tensor,
    shared_intermediate: Tensor,
    assignment_output: Tensor,
    shared_output: Tensor,
    shared_gate_logits: Tensor,
    output: Tensor,
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
        Ok(Self {
            router_logits: zeros(ctx, vec![tokens, experts], DType::BF16)?,
            expert_ids: zeros(ctx, vec![tokens, top_k], DType::I32)?,
            expert_weights: zeros(ctx, vec![tokens, top_k], DType::F32)?,
            assignment_expert_ids: zeros(ctx, vec![routes], DType::I32)?,
            assignment_token_ids: zeros(ctx, vec![routes], DType::I32)?,
            assignment_slots: zeros(ctx, vec![routes], DType::I32)?,
            assignment_weights: zeros(ctx, vec![routes], DType::F32)?,
            expert_offsets: zeros(ctx, vec![experts + 1], DType::I32)?,
            route_intermediate: zeros(ctx, vec![routes, intermediate], DType::BF16)?,
            shared_intermediate: zeros(ctx, vec![tokens, intermediate], DType::BF16)?,
            assignment_output: zeros(ctx, vec![routes, hidden], DType::BF16)?,
            shared_output: zeros(ctx, vec![tokens, hidden], DType::BF16)?,
            shared_gate_logits: zeros(ctx, vec![tokens, 1], DType::BF16)?,
            output: zeros(ctx, vec![tokens, hidden], DType::BF16)?,
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

        ops::gemm(
            ctx,
            ops::GemmArgs::new(input, &weights.router.weight, &mut self.router_logits),
        )?;
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
                assignment_weights: &mut self.assignment_weights,
                expert_offsets: &mut self.expert_offsets,
            },
        )?;
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
                    packed_weight: &weights.expert_gate.packed_weight,
                    block_scales: &weights.expert_gate.block_scales,
                    alpha: &weights.expert_gate.weight_scales_2,
                },
                expert_up: Nvfp4ExpertProjection {
                    packed_weight: &weights.expert_up.packed_weight,
                    block_scales: &weights.expert_up.block_scales,
                    alpha: &weights.expert_up.weight_scales_2,
                },
                expert_down: Nvfp4ExpertProjection {
                    packed_weight: &weights.expert_down.packed_weight,
                    block_scales: &weights.expert_down.block_scales,
                    alpha: &weights.expert_down.weight_scales_2,
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
                shared_output: &self.shared_output,
                shared_gate_logits: &shared_gate_logits,
                output: &mut self.output,
            },
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
        for (projection, input_features, output_features, label) in [
            (
                &weights.expert_gate,
                self.hidden,
                self.intermediate,
                "expert gate",
            ),
            (
                &weights.expert_up,
                self.hidden,
                self.intermediate,
                "expert up",
            ),
            (
                &weights.expert_down,
                self.intermediate,
                self.hidden,
                "expert down",
            ),
        ] {
            if projection.experts != self.experts
                || projection.input_features != input_features
                || projection.output_features != output_features
            {
                return Err(invalid(format!("Qwen3.5 {label} geometry mismatch")));
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
}

/// Projection, core, and output buffers for one base GDN layer/profile.
///
/// The `gdn_*16`/`cake_*` fields back the FlashInfer Cake prefill path and are
/// allocated only on Blackwell, where that kernel exists. Decode keeps using
/// `convolved_qkv`/`core_output`, so those stay regardless.
#[derive(Debug)]
pub struct GdnScratch {
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
        let key_dims = vec![
            tokens,
            ops::QWEN35_GDN_KEY_HEADS,
            ops::QWEN35_GDN_HEAD_DIM,
        ];
        let value_dims = vec![
            tokens,
            ops::QWEN35_GDN_VALUE_HEADS,
            ops::QWEN35_GDN_HEAD_DIM,
        ];
        let workspace_bytes = ops::gdn_prefill_workspace_bytes(ops::QWEN35_GDN_VALUE_HEADS, 1);
        Ok(Self {
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
            k16: cake
                .then(|| zeros(ctx, key_dims, DType::F16))
                .transpose()?,
            v16: cake
                .then(|| zeros(ctx, value_dims.clone(), DType::F16))
                .transpose()?,
            out16: cake
                .then(|| zeros(ctx, value_dims, DType::F16))
                .transpose()?,
            alpha: cake
                .then(|| zeros(ctx, vec![tokens, ops::QWEN35_GDN_VALUE_HEADS], DType::F32))
                .transpose()?,
            cu_seqlens: cake
                .then(|| zeros(ctx, vec![2], DType::I32))
                .transpose()?,
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
        self.validate(ctx, weights, state, normalized_hidden)?;
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
        if self.tokens != 1 {
            return Err(invalid(
                "Qwen3.5 GDN decode requires a one-token scratch profile",
            ));
        }
        self.validate(ctx, weights, state, normalized_hidden)?;
        self.project(ctx, weights, normalized_hidden)?;
        let projected_qkv = view(
            &self.projected_qkv,
            vec![ops::QWEN35_GDN_QKV_WIDTH],
            DType::BF16,
        )?;
        let z = view(
            &self.z,
            vec![ops::QWEN35_GDN_VALUE_WIDTH],
            DType::BF16,
        )?;
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
        )?;
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
        for (weight, output) in [
            (&weights.in_proj_qkv.weight, &mut self.projected_qkv),
            (&weights.in_proj_z.weight, &mut self.z),
            (&weights.in_proj_a.weight, &mut self.a),
            (&weights.in_proj_b.weight, &mut self.b),
        ] {
            ops::gemm(ctx, ops::GemmArgs::new(input, weight, output))?;
        }
        Ok(())
    }

    fn output_projection(&mut self, ctx: &CudaContext, weights: &DeviceGdnWeights) -> Result<()> {
        let gated_output = view(
            &self.gated_output,
            vec![self.tokens, ops::QWEN35_GDN_VALUE_WIDTH],
            DType::BF16,
        )?;
        ops::gemm(
            ctx,
            ops::GemmArgs::new(&gated_output, &weights.out_proj.weight, &mut self.output),
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
                &weights.in_proj_a,
                self.hidden,
                ops::QWEN35_GDN_VALUE_HEADS,
                "GDN a projection",
            ),
            (
                &weights.in_proj_b,
                self.hidden,
                ops::QWEN35_GDN_VALUE_HEADS,
                "GDN b projection",
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
            validate_linear(&linear.weight, ctx, input_features, output_features, label)?;
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
    CudaBuffer::from_tensor(tensor)
        .map_err(Error::Cuda)?
        .as_tensor(Shape::new(shape), dtype)
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
