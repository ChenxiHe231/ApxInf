use apxinf_core::{DType, Result, Tensor};

use super::contracts::RoutedMoeConfig;
use crate::ffi::abi::{moe as abi, status};
use crate::ops::gemm::contracts::{dtype, invalid, tensor_storage};
use crate::ops::gemm::ScaleLayout;
use crate::{CudaBuffer, CudaContext};

pub struct RoutedMoeRouteArgs<'a> {
    pub config: RoutedMoeConfig,
    /// Router logits `[tokens, experts]`, in F32, F16, or BF16.
    pub logits: &'a Tensor,
    /// Token-major selected expert IDs `[tokens, top_k]`, I32.
    pub expert_ids: &'a mut Tensor,
    /// Token-major selected weights `[tokens, top_k]`, F32.
    pub expert_weights: &'a mut Tensor,
    /// Expert-major assignment IDs `[tokens * top_k]`, I32.
    pub assignment_expert_ids: &'a mut Tensor,
    /// Expert-major assignment token IDs `[tokens * top_k]`, I32.
    pub assignment_token_ids: &'a mut Tensor,
    /// Expert-major assignment route slots `[tokens * top_k]`, I32.
    pub assignment_slots: &'a mut Tensor,
    /// Token-major inverse map from `[token, route_slot]` to the
    /// corresponding expert-major assignment row, I32 `[tokens * top_k]`.
    pub token_assignment_indices: &'a mut Tensor,
    /// Expert-major assignment weights `[tokens * top_k]`, F32.
    pub assignment_weights: &'a mut Tensor,
    /// Prefix offsets into assignments `[experts + 1]`, I32.
    pub expert_offsets: &'a mut Tensor,
}

pub struct RoutedMoeRouteTopkArgs<'a> {
    pub config: RoutedMoeConfig,
    /// Router logits `[tokens, experts]`, in F32, F16, or BF16.
    pub logits: &'a Tensor,
    /// Token-major selected expert IDs `[tokens, top_k]`, I32.
    pub expert_ids: &'a mut Tensor,
    /// Token-major selected weights `[tokens, top_k]`, F32.
    pub expert_weights: &'a mut Tensor,
}

pub struct RoutedMoeCombineArgs<'a> {
    /// Expert-major outputs `[routes, hidden]`.
    pub assignment_output: &'a Tensor,
    pub assignment_token_ids: &'a Tensor,
    pub assignment_weights: &'a Tensor,
    /// Token-major inverse map produced by [`routed_moe_route`].
    pub token_assignment_indices: &'a Tensor,
    /// Shared-expert outputs `[tokens, hidden]`.
    pub shared_output: &'a Tensor,
    /// Pre-sigmoid shared gate logits `[tokens]`.
    pub shared_gate_logits: &'a Tensor,
    /// Combined output `[tokens, hidden]`.
    pub output: &'a mut Tensor,
}

pub struct RoutedMoeAddSharedArgs<'a> {
    /// Token-major routed contribution `[tokens, hidden]`.
    pub routed_output: &'a Tensor,
    /// Token-major shared-expert output `[tokens, hidden]`.
    pub shared_output: &'a Tensor,
    /// Pre-sigmoid shared gate logits `[tokens]`.
    pub shared_gate_logits: &'a Tensor,
    /// `routed + sigmoid(gate) * shared`, `[tokens, hidden]`.
    pub output: &'a mut Tensor,
}

#[derive(Clone, Copy)]
pub struct Nvfp4ExpertProjection<'a> {
    /// Packed E2M1 checkpoint weight, K contiguous and low nibble first.
    pub packed_weight: &'a Tensor,
    /// Unsigned-E4M3 scales in [`ScaleLayout::RowMajor`].
    pub block_scales: &'a Tensor,
    /// Per-expert `weight_scale_2`; a shared projection uses shape `[1]`.
    pub alpha: &'a Tensor,
}

pub struct RoutedMoeNvfp4ExpertsArgs<'a> {
    pub config: RoutedMoeConfig,
    pub intermediate: usize,
    pub block_size: u32,
    pub scale_layout: ScaleLayout,
    /// BF16 hidden states `[tokens, hidden]`.
    pub input: &'a Tensor,
    /// Expert-major assignment IDs `[routes]` produced by routing.
    pub assignment_expert_ids: &'a Tensor,
    /// Assignment token IDs `[routes]` produced by routing.
    pub assignment_token_ids: &'a Tensor,
    pub expert_gate: Nvfp4ExpertProjection<'a>,
    pub expert_up: Nvfp4ExpertProjection<'a>,
    pub expert_down: Nvfp4ExpertProjection<'a>,
    pub shared_gate: Nvfp4ExpertProjection<'a>,
    pub shared_up: Nvfp4ExpertProjection<'a>,
    pub shared_down: Nvfp4ExpertProjection<'a>,
    /// BF16 scratch `[routes, intermediate]`.
    pub route_intermediate: &'a mut Tensor,
    /// BF16 scratch `[tokens, intermediate]`.
    pub shared_intermediate: &'a mut Tensor,
    /// BF16 expert-major result `[routes, hidden]`.
    pub assignment_output: &'a mut Tensor,
    /// BF16 shared-expert result `[tokens, hidden]`.
    pub shared_output: &'a mut Tensor,
}

/// TensorRT-Edge-LLM/CuTeDSL packed routed-expert weights.
///
/// FC1 uses 64-row `[up, gate]` interleaving. Scale tensors use the physical
/// 128x4 CuTe MMA layout; their rank documents that layout but the kernel
/// consumes the storage as bytes.
#[derive(Clone, Copy)]
pub struct EdgeNvfp4MoeWeights<'a> {
    pub fc1_weight: &'a Tensor,
    pub fc1_scales: &'a Tensor,
    pub fc1_alpha: &'a Tensor,
    pub input_global_scale: &'a Tensor,
    pub fc2_weight: &'a Tensor,
    pub fc2_scales: &'a Tensor,
    pub fc2_alpha: &'a Tensor,
    pub down_input_scale: &'a Tensor,
}

pub struct EdgeNvfp4MoeArgs<'a> {
    pub config: RoutedMoeConfig,
    pub intermediate: usize,
    /// BF16 `[tokens, hidden]`.
    pub input: &'a Tensor,
    /// Token-major I32 `[tokens, top_k]`.
    pub expert_ids: &'a Tensor,
    /// Token-major F32 `[tokens, top_k]`.
    pub expert_weights: &'a Tensor,
    pub weights: EdgeNvfp4MoeWeights<'a>,
    pub workspace: &'a CudaBuffer,
    /// BF16 routed contribution `[tokens, hidden]`; routing weights are
    /// already applied by FC2.
    pub output: &'a mut Tensor,
}

pub(crate) struct NormalizedRoute {
    pub spec: abi::RouteSpec,
    pub bindings: abi::RouteBindings,
    pub storage: Vec<CudaBuffer>,
}

pub(crate) struct NormalizedRouteTopk {
    pub spec: abi::RouteSpec,
    pub bindings: abi::RouteBindings,
    pub storage: Vec<CudaBuffer>,
}

pub(crate) struct NormalizedCombine {
    pub spec: abi::CombineSpec,
    pub bindings: abi::CombineBindings,
    pub storage: Vec<CudaBuffer>,
}

pub(crate) struct NormalizedAddShared {
    pub spec: abi::CombineSpec,
    pub bindings: abi::AddSharedBindings,
    pub storage: Vec<CudaBuffer>,
}

pub(crate) struct NormalizedNvfp4Experts {
    pub spec: abi::Nvfp4ExpertsSpec,
    pub bindings: abi::Nvfp4ExpertsBindings,
    pub storage: Vec<CudaBuffer>,
}

pub(crate) struct NormalizedEdgeNvfp4 {
    pub spec: abi::EdgeNvfp4Spec,
    pub bindings: abi::EdgeNvfp4Bindings,
    pub storage: Vec<CudaBuffer>,
}

fn edge_spec(
    config: RoutedMoeConfig,
    tokens: usize,
    hidden: usize,
    intermediate: usize,
) -> Result<abi::EdgeNvfp4Spec> {
    config.validate()?;
    if config.experts != 256 || config.top_k != 8 {
        return Err(invalid(
            "Edge NVFP4 MoE currently requires exactly 256 experts and top-8",
        ));
    }
    if tokens == 0
        || hidden == 0
        || hidden % 128 != 0
        || intermediate == 0
        || intermediate % 64 != 0
    {
        return Err(invalid(
            "Edge NVFP4 MoE requires non-zero tokens, hidden multiple of 128, and intermediate multiple of 64",
        ));
    }
    Ok(abi::EdgeNvfp4Spec {
        version: abi::EDGE_NVFP4_SPEC_VERSION,
        dtype: dtype(DType::BF16)?,
        tokens: u32::try_from(tokens).map_err(|_| invalid("Edge NVFP4 token count exceeds u32"))?,
        experts: u32::try_from(config.experts)
            .map_err(|_| invalid("Edge NVFP4 expert count exceeds u32"))?,
        top_k: u32::try_from(config.top_k).map_err(|_| invalid("Edge NVFP4 top-k exceeds u32"))?,
        hidden: u32::try_from(hidden).map_err(|_| invalid("Edge NVFP4 hidden size exceeds u32"))?,
        intermediate: u32::try_from(intermediate)
            .map_err(|_| invalid("Edge NVFP4 intermediate size exceeds u32"))?,
    })
}

pub(crate) fn edge_nvfp4_workspace_size(
    config: RoutedMoeConfig,
    tokens: usize,
    hidden: usize,
    intermediate: usize,
) -> Result<usize> {
    let spec = edge_spec(config, tokens, hidden, intermediate)?;
    let mut bytes = 0usize;
    unsafe {
        status::check(abi::apxinf_moe_edge_nvfp4_workspace_size(&spec, &mut bytes))?;
    }
    Ok(bytes)
}

pub(crate) fn normalize_edge_nvfp4(
    ctx: &CudaContext,
    args: EdgeNvfp4MoeArgs<'_>,
) -> Result<NormalizedEdgeNvfp4> {
    let shape = args.input.shape().dims();
    if shape.len() != 2 {
        return Err(invalid("Edge NVFP4 input must have shape [tokens, hidden]"));
    }
    let (tokens, hidden) = (shape[0], shape[1]);
    let spec = edge_spec(args.config, tokens, hidden, args.intermediate)?;
    let input = tensor_storage(ctx, args.input, DType::BF16, &[tokens, hidden])?;
    let expert_ids = tensor_storage(
        ctx,
        args.expert_ids,
        DType::I32,
        &[tokens, args.config.top_k],
    )?;
    let expert_weights = tensor_storage(
        ctx,
        args.expert_weights,
        DType::F32,
        &[tokens, args.config.top_k],
    )?;
    let experts = args.config.experts;
    let inter = args.intermediate;
    let fc1_weight = tensor_storage(
        ctx,
        args.weights.fc1_weight,
        DType::E2M1Pair,
        &[experts, 2 * inter, hidden / 2],
    )?;
    let fc1_scales = tensor_storage(
        ctx,
        args.weights.fc1_scales,
        DType::F8E4M3,
        &[
            experts,
            (2 * inter).div_ceil(128),
            (hidden / 16).div_ceil(4),
            32,
            4,
            4,
        ],
    )?;
    let fc1_alpha = tensor_storage(ctx, args.weights.fc1_alpha, DType::F32, &[experts])?;
    let input_global_scale =
        tensor_storage(ctx, args.weights.input_global_scale, DType::F32, &[experts])?;
    let fc2_weight = tensor_storage(
        ctx,
        args.weights.fc2_weight,
        DType::E2M1Pair,
        &[experts, hidden, inter / 2],
    )?;
    let fc2_scales = tensor_storage(
        ctx,
        args.weights.fc2_scales,
        DType::F8E4M3,
        &[
            experts,
            hidden.div_ceil(128),
            (inter / 16).div_ceil(4),
            32,
            4,
            4,
        ],
    )?;
    let fc2_alpha = tensor_storage(ctx, args.weights.fc2_alpha, DType::F32, &[experts])?;
    let down_input_scale =
        tensor_storage(ctx, args.weights.down_input_scale, DType::F32, &[experts])?;
    let output = tensor_storage(ctx, args.output, DType::BF16, &[tokens, hidden])?;
    if args.workspace.device() != ctx.device_id() {
        return Err(invalid("Edge NVFP4 workspace is on the wrong CUDA device"));
    }
    let required = edge_nvfp4_workspace_size(args.config, tokens, hidden, inter)?;
    if args.workspace.len() < required {
        return Err(invalid(format!(
            "Edge NVFP4 workspace requires {required} bytes, got {}",
            args.workspace.len()
        )));
    }
    let workspace = args.workspace.clone();
    let bindings = abi::EdgeNvfp4Bindings {
        input: input.ptr(),
        expert_ids: expert_ids.ptr().cast(),
        expert_weights: expert_weights.ptr().cast(),
        fc1_weight: fc1_weight.ptr(),
        fc1_scales: fc1_scales.ptr(),
        fc1_alpha: fc1_alpha.ptr().cast(),
        input_global_scale: input_global_scale.ptr().cast(),
        fc2_weight: fc2_weight.ptr(),
        fc2_scales: fc2_scales.ptr(),
        fc2_alpha: fc2_alpha.ptr().cast(),
        down_input_scale: down_input_scale.ptr().cast(),
        workspace: workspace.ptr(),
        workspace_bytes: workspace.len(),
        output: output.ptr(),
        stream: ctx.stream().handle(),
    };
    Ok(NormalizedEdgeNvfp4 {
        spec,
        bindings,
        storage: vec![
            input,
            expert_ids,
            expert_weights,
            fc1_weight,
            fc1_scales,
            fc1_alpha,
            input_global_scale,
            fc2_weight,
            fc2_scales,
            fc2_alpha,
            down_input_scale,
            output,
            workspace,
        ],
    })
}

struct ProjectionBuffers {
    weight: CudaBuffer,
    scales: CudaBuffer,
    alpha: CudaBuffer,
}

fn supported_float(value: DType) -> Result<u32> {
    match value {
        DType::F32 | DType::F16 | DType::BF16 => dtype(value),
        _ => Err(invalid("routed MoE supports F32, F16, and BF16")),
    }
}

pub(crate) fn normalize_route(
    ctx: &CudaContext,
    args: RoutedMoeRouteArgs<'_>,
) -> Result<NormalizedRoute> {
    args.config.validate()?;
    if args.config.experts > 256 || args.config.top_k > 8 {
        return Err(invalid(
            "the routed MoE CUDA candidate supports at most 256 experts and top-8",
        ));
    }
    let logits_shape = args.logits.shape().dims();
    if logits_shape.len() != 2 || logits_shape[0] == 0 || logits_shape[1] != args.config.experts {
        return Err(invalid(
            "routed MoE logits must have shape [tokens, experts]",
        ));
    }
    let tokens = logits_shape[0];
    let routes = tokens
        .checked_mul(args.config.top_k)
        .ok_or_else(|| invalid("routed MoE route count overflow"))?;
    let logits_dtype = args.logits.dtype();
    let dtype_code = supported_float(logits_dtype)?;

    let logits = tensor_storage(
        ctx,
        args.logits,
        logits_dtype,
        &[tokens, args.config.experts],
    )?;
    let expert_ids = tensor_storage(
        ctx,
        args.expert_ids,
        DType::I32,
        &[tokens, args.config.top_k],
    )?;
    let expert_weights = tensor_storage(
        ctx,
        args.expert_weights,
        DType::F32,
        &[tokens, args.config.top_k],
    )?;
    let assignment_expert_ids =
        tensor_storage(ctx, args.assignment_expert_ids, DType::I32, &[routes])?;
    let assignment_token_ids =
        tensor_storage(ctx, args.assignment_token_ids, DType::I32, &[routes])?;
    let assignment_slots = tensor_storage(ctx, args.assignment_slots, DType::I32, &[routes])?;
    let token_assignment_indices =
        tensor_storage(ctx, args.token_assignment_indices, DType::I32, &[routes])?;
    let assignment_weights = tensor_storage(ctx, args.assignment_weights, DType::F32, &[routes])?;
    let expert_offsets = tensor_storage(
        ctx,
        args.expert_offsets,
        DType::I32,
        &[args.config.experts + 1],
    )?;

    let spec = abi::RouteSpec {
        version: abi::ROUTE_SPEC_VERSION,
        logits_dtype: dtype_code,
        tokens: u32::try_from(tokens)
            .map_err(|_| invalid("routed MoE token count exceeds the CUDA range"))?,
        experts: u32::try_from(args.config.experts)
            .map_err(|_| invalid("routed MoE expert count exceeds the CUDA range"))?,
        top_k: u32::try_from(args.config.top_k)
            .map_err(|_| invalid("routed MoE top_k exceeds the CUDA range"))?,
        renormalize: u32::from(args.config.renormalize),
    };
    let bindings = abi::RouteBindings {
        logits: logits.ptr(),
        expert_ids: expert_ids.ptr().cast(),
        expert_weights: expert_weights.ptr().cast(),
        assignment_expert_ids: assignment_expert_ids.ptr().cast(),
        assignment_token_ids: assignment_token_ids.ptr().cast(),
        assignment_slots: assignment_slots.ptr().cast(),
        token_assignment_indices: token_assignment_indices.ptr().cast(),
        assignment_weights: assignment_weights.ptr().cast(),
        expert_offsets: expert_offsets.ptr().cast(),
        stream: ctx.stream().handle(),
    };
    Ok(NormalizedRoute {
        spec,
        bindings,
        storage: vec![
            logits,
            expert_ids,
            expert_weights,
            assignment_expert_ids,
            assignment_token_ids,
            assignment_slots,
            token_assignment_indices,
            assignment_weights,
            expert_offsets,
        ],
    })
}

pub(crate) fn normalize_route_topk(
    ctx: &CudaContext,
    args: RoutedMoeRouteTopkArgs<'_>,
) -> Result<NormalizedRouteTopk> {
    args.config.validate()?;
    if args.config.experts > 256 || args.config.top_k > 8 {
        return Err(invalid(
            "the routed MoE CUDA candidate supports at most 256 experts and top-8",
        ));
    }
    let shape = args.logits.shape().dims();
    if shape.len() != 2 || shape[0] == 0 || shape[1] != args.config.experts {
        return Err(invalid(
            "routed MoE logits must have shape [tokens, experts]",
        ));
    }
    let tokens = shape[0];
    let logits_dtype = args.logits.dtype();
    let logits = tensor_storage(ctx, args.logits, logits_dtype, shape)?;
    let expert_ids = tensor_storage(
        ctx,
        args.expert_ids,
        DType::I32,
        &[tokens, args.config.top_k],
    )?;
    let expert_weights = tensor_storage(
        ctx,
        args.expert_weights,
        DType::F32,
        &[tokens, args.config.top_k],
    )?;
    let spec = abi::RouteSpec {
        version: abi::ROUTE_SPEC_VERSION,
        logits_dtype: supported_float(logits_dtype)?,
        tokens: u32::try_from(tokens)
            .map_err(|_| invalid("routed MoE token count exceeds the CUDA range"))?,
        experts: u32::try_from(args.config.experts)
            .map_err(|_| invalid("routed MoE expert count exceeds the CUDA range"))?,
        top_k: u32::try_from(args.config.top_k)
            .map_err(|_| invalid("routed MoE top-k exceeds the CUDA range"))?,
        renormalize: u32::from(args.config.renormalize),
    };
    let bindings = abi::RouteBindings {
        logits: logits.ptr(),
        expert_ids: expert_ids.ptr().cast(),
        expert_weights: expert_weights.ptr().cast(),
        assignment_expert_ids: std::ptr::null_mut(),
        assignment_token_ids: std::ptr::null_mut(),
        assignment_slots: std::ptr::null_mut(),
        token_assignment_indices: std::ptr::null_mut(),
        assignment_weights: std::ptr::null_mut(),
        expert_offsets: std::ptr::null_mut(),
        stream: ctx.stream().handle(),
    };
    Ok(NormalizedRouteTopk {
        spec,
        bindings,
        storage: vec![logits, expert_ids, expert_weights],
    })
}

pub(crate) fn normalize_combine(
    ctx: &CudaContext,
    args: RoutedMoeCombineArgs<'_>,
) -> Result<NormalizedCombine> {
    let assignment_shape = args.assignment_output.shape().dims();
    let output_shape = args.output.shape().dims();
    if assignment_shape.len() != 2
        || assignment_shape[0] == 0
        || assignment_shape[1] == 0
        || output_shape.len() != 2
        || output_shape[0] == 0
        || output_shape[1] != assignment_shape[1]
    {
        return Err(invalid(
            "routed MoE combine tensors must have shapes [routes, hidden] and [tokens, hidden]",
        ));
    }
    let (routes, hidden) = (assignment_shape[0], assignment_shape[1]);
    let tokens = output_shape[0];
    if routes % tokens != 0 {
        return Err(invalid(
            "routed MoE route count must be divisible by the token count",
        ));
    }
    let top_k = routes / tokens;
    if top_k == 0 || top_k > 8 {
        return Err(invalid("routed MoE combine supports top_k in 1..=8"));
    }
    let activation_dtype = args.assignment_output.dtype();
    let dtype_code = supported_float(activation_dtype)?;

    let assignment_output = tensor_storage(
        ctx,
        args.assignment_output,
        activation_dtype,
        &[routes, hidden],
    )?;
    let assignment_token_ids =
        tensor_storage(ctx, args.assignment_token_ids, DType::I32, &[routes])?;
    let assignment_weights = tensor_storage(ctx, args.assignment_weights, DType::F32, &[routes])?;
    let token_assignment_indices =
        tensor_storage(ctx, args.token_assignment_indices, DType::I32, &[routes])?;
    let shared_output =
        tensor_storage(ctx, args.shared_output, activation_dtype, &[tokens, hidden])?;
    let shared_gate_logits =
        tensor_storage(ctx, args.shared_gate_logits, activation_dtype, &[tokens])?;
    let output = tensor_storage(ctx, args.output, activation_dtype, &[tokens, hidden])?;

    let spec = abi::CombineSpec {
        version: abi::COMBINE_SPEC_VERSION,
        dtype: dtype_code,
        tokens: u32::try_from(tokens)
            .map_err(|_| invalid("routed MoE token count exceeds the CUDA range"))?,
        routes: u32::try_from(routes)
            .map_err(|_| invalid("routed MoE route count exceeds the CUDA range"))?,
        top_k: u32::try_from(top_k)
            .map_err(|_| invalid("routed MoE top_k exceeds the CUDA range"))?,
        hidden: u32::try_from(hidden)
            .map_err(|_| invalid("routed MoE hidden width exceeds the CUDA range"))?,
    };
    let bindings = abi::CombineBindings {
        assignment_output: assignment_output.ptr(),
        assignment_token_ids: assignment_token_ids.ptr().cast(),
        assignment_weights: assignment_weights.ptr().cast(),
        token_assignment_indices: token_assignment_indices.ptr().cast(),
        shared_output: shared_output.ptr(),
        shared_gate_logits: shared_gate_logits.ptr(),
        output: output.ptr(),
        stream: ctx.stream().handle(),
    };
    Ok(NormalizedCombine {
        spec,
        bindings,
        storage: vec![
            assignment_output,
            assignment_token_ids,
            assignment_weights,
            token_assignment_indices,
            shared_output,
            shared_gate_logits,
            output,
        ],
    })
}

pub(crate) fn normalize_add_shared(
    ctx: &CudaContext,
    args: RoutedMoeAddSharedArgs<'_>,
) -> Result<NormalizedAddShared> {
    let shape = args.routed_output.shape().dims();
    if shape.len() != 2 || shape[0] == 0 || shape[1] == 0 {
        return Err(invalid("routed MoE add-shared expects [tokens, hidden]"));
    }
    let (tokens, hidden) = (shape[0], shape[1]);
    let activation_dtype = args.routed_output.dtype();
    let routed_output = tensor_storage(ctx, args.routed_output, activation_dtype, shape)?;
    let shared_output = tensor_storage(ctx, args.shared_output, activation_dtype, shape)?;
    let shared_gate_logits =
        tensor_storage(ctx, args.shared_gate_logits, activation_dtype, &[tokens])?;
    let output = tensor_storage(ctx, args.output, activation_dtype, shape)?;
    let spec = abi::CombineSpec {
        version: abi::COMBINE_SPEC_VERSION,
        dtype: supported_float(activation_dtype)?,
        tokens: u32::try_from(tokens)
            .map_err(|_| invalid("routed MoE token count exceeds the CUDA range"))?,
        routes: u32::try_from(tokens)
            .map_err(|_| invalid("routed MoE token count exceeds the CUDA range"))?,
        top_k: 1,
        hidden: u32::try_from(hidden)
            .map_err(|_| invalid("routed MoE hidden width exceeds the CUDA range"))?,
    };
    let bindings = abi::AddSharedBindings {
        routed_output: routed_output.ptr(),
        shared_output: shared_output.ptr(),
        shared_gate_logits: shared_gate_logits.ptr(),
        output: output.ptr(),
        stream: ctx.stream().handle(),
    };
    Ok(NormalizedAddShared {
        spec,
        bindings,
        storage: vec![routed_output, shared_output, shared_gate_logits, output],
    })
}

fn projection_buffers(
    ctx: &CudaContext,
    projection: Nvfp4ExpertProjection<'_>,
    weight_shape: &[usize],
    scale_shape: &[usize],
    alpha_shape: &[usize],
) -> Result<ProjectionBuffers> {
    Ok(ProjectionBuffers {
        weight: tensor_storage(ctx, projection.packed_weight, DType::E2M1Pair, weight_shape)?,
        scales: tensor_storage(ctx, projection.block_scales, DType::F8E4M3, scale_shape)?,
        alpha: tensor_storage(ctx, projection.alpha, DType::F32, alpha_shape)?,
    })
}

pub(crate) fn normalize_nvfp4_experts(
    ctx: &CudaContext,
    args: RoutedMoeNvfp4ExpertsArgs<'_>,
) -> Result<NormalizedNvfp4Experts> {
    args.config.validate()?;
    if args.config.experts > 256 {
        return Err(invalid(
            "the routed NVFP4 expert candidate supports at most 256 experts",
        ));
    }
    if args.block_size != 16 {
        return Err(invalid(
            "the routed NVFP4 expert candidate requires block size 16",
        ));
    }
    if args.scale_layout != ScaleLayout::RowMajor {
        return Err(invalid(
            "the correctness routed NVFP4 candidate requires ScaleLayout::RowMajor",
        ));
    }
    let input_shape = args.input.shape().dims();
    if input_shape.len() != 2 || input_shape[0] == 0 || input_shape[1] == 0 {
        return Err(invalid(
            "routed NVFP4 input must have shape [tokens, hidden]",
        ));
    }
    let (tokens, hidden) = (input_shape[0], input_shape[1]);
    if args.intermediate == 0
        || hidden % args.block_size as usize != 0
        || args.intermediate % args.block_size as usize != 0
    {
        return Err(invalid(
            "routed NVFP4 hidden and intermediate widths must be non-zero multiples of 16",
        ));
    }
    let assignment_shape = args.assignment_expert_ids.shape().dims();
    if assignment_shape.len() != 1 || assignment_shape[0] == 0 {
        return Err(invalid(
            "routed NVFP4 assignment IDs must have shape [routes]",
        ));
    }
    let routes = assignment_shape[0];
    let input = tensor_storage(ctx, args.input, DType::BF16, &[tokens, hidden])?;
    let assignment_expert_ids =
        tensor_storage(ctx, args.assignment_expert_ids, DType::I32, &[routes])?;
    let assignment_token_ids =
        tensor_storage(ctx, args.assignment_token_ids, DType::I32, &[routes])?;

    let expert_gate = projection_buffers(
        ctx,
        args.expert_gate,
        &[args.config.experts, args.intermediate, hidden / 2],
        &[
            args.config.experts,
            args.intermediate,
            hidden / args.block_size as usize,
        ],
        &[args.config.experts],
    )?;
    let expert_up = projection_buffers(
        ctx,
        args.expert_up,
        &[args.config.experts, args.intermediate, hidden / 2],
        &[
            args.config.experts,
            args.intermediate,
            hidden / args.block_size as usize,
        ],
        &[args.config.experts],
    )?;
    let expert_down = projection_buffers(
        ctx,
        args.expert_down,
        &[args.config.experts, hidden, args.intermediate / 2],
        &[
            args.config.experts,
            hidden,
            args.intermediate / args.block_size as usize,
        ],
        &[args.config.experts],
    )?;
    let shared_gate = projection_buffers(
        ctx,
        args.shared_gate,
        &[args.intermediate, hidden / 2],
        &[args.intermediate, hidden / args.block_size as usize],
        &[1],
    )?;
    let shared_up = projection_buffers(
        ctx,
        args.shared_up,
        &[args.intermediate, hidden / 2],
        &[args.intermediate, hidden / args.block_size as usize],
        &[1],
    )?;
    let shared_down = projection_buffers(
        ctx,
        args.shared_down,
        &[hidden, args.intermediate / 2],
        &[hidden, args.intermediate / args.block_size as usize],
        &[1],
    )?;
    let route_intermediate = tensor_storage(
        ctx,
        args.route_intermediate,
        DType::BF16,
        &[routes, args.intermediate],
    )?;
    let shared_intermediate = tensor_storage(
        ctx,
        args.shared_intermediate,
        DType::BF16,
        &[tokens, args.intermediate],
    )?;
    let assignment_output =
        tensor_storage(ctx, args.assignment_output, DType::BF16, &[routes, hidden])?;
    let shared_output = tensor_storage(ctx, args.shared_output, DType::BF16, &[tokens, hidden])?;

    let spec = abi::Nvfp4ExpertsSpec {
        version: abi::NVFP4_EXPERTS_SPEC_VERSION,
        dtype: dtype(DType::BF16)?,
        tokens: u32::try_from(tokens)
            .map_err(|_| invalid("routed NVFP4 token count exceeds the CUDA range"))?,
        routes: u32::try_from(routes)
            .map_err(|_| invalid("routed NVFP4 route count exceeds the CUDA range"))?,
        experts: u32::try_from(args.config.experts)
            .map_err(|_| invalid("routed NVFP4 expert count exceeds the CUDA range"))?,
        hidden: u32::try_from(hidden)
            .map_err(|_| invalid("routed NVFP4 hidden width exceeds the CUDA range"))?,
        intermediate: u32::try_from(args.intermediate)
            .map_err(|_| invalid("routed NVFP4 intermediate width exceeds the CUDA range"))?,
        block_size: args.block_size,
        scale_layout: 1,
    };
    let bindings = abi::Nvfp4ExpertsBindings {
        input: input.ptr(),
        assignment_expert_ids: assignment_expert_ids.ptr().cast(),
        assignment_token_ids: assignment_token_ids.ptr().cast(),
        expert_gate_weight: expert_gate.weight.ptr(),
        expert_gate_scales: expert_gate.scales.ptr(),
        expert_gate_alpha: expert_gate.alpha.ptr().cast(),
        expert_up_weight: expert_up.weight.ptr(),
        expert_up_scales: expert_up.scales.ptr(),
        expert_up_alpha: expert_up.alpha.ptr().cast(),
        expert_down_weight: expert_down.weight.ptr(),
        expert_down_scales: expert_down.scales.ptr(),
        expert_down_alpha: expert_down.alpha.ptr().cast(),
        shared_gate_weight: shared_gate.weight.ptr(),
        shared_gate_scales: shared_gate.scales.ptr(),
        shared_gate_alpha: shared_gate.alpha.ptr().cast(),
        shared_up_weight: shared_up.weight.ptr(),
        shared_up_scales: shared_up.scales.ptr(),
        shared_up_alpha: shared_up.alpha.ptr().cast(),
        shared_down_weight: shared_down.weight.ptr(),
        shared_down_scales: shared_down.scales.ptr(),
        shared_down_alpha: shared_down.alpha.ptr().cast(),
        route_intermediate: route_intermediate.ptr(),
        shared_intermediate: shared_intermediate.ptr(),
        assignment_output: assignment_output.ptr(),
        shared_output: shared_output.ptr(),
        stream: ctx.stream().handle(),
    };
    Ok(NormalizedNvfp4Experts {
        spec,
        bindings,
        storage: vec![
            input,
            assignment_expert_ids,
            assignment_token_ids,
            expert_gate.weight,
            expert_gate.scales,
            expert_gate.alpha,
            expert_up.weight,
            expert_up.scales,
            expert_up.alpha,
            expert_down.weight,
            expert_down.scales,
            expert_down.alpha,
            shared_gate.weight,
            shared_gate.scales,
            shared_gate.alpha,
            shared_up.weight,
            shared_up.scales,
            shared_up.alpha,
            shared_down.weight,
            shared_down.scales,
            shared_down.alpha,
            route_intermediate,
            shared_intermediate,
            assignment_output,
            shared_output,
        ],
    })
}
