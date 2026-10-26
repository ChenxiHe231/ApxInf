//! CUDA-resident weight layout for Qwen3.5-35B-A3B.
//!
//! The checkpoint loader deliberately preserves checkpoint-native tensors:
//! dense weights are `[N, K]`, and every routed expert is a separate tensor.
//! cuda-new's GEMM contract consumes dense weights as `[K, N]`, while its
//! routed-NVFP4 operator consumes one expert-major allocation per projection.
//! This module performs those fixed transformations once, during model load.

use std::path::Path;

use apxinf_core::{DType, Device, Error, Result, Shape, Tensor};
use apxinf_cuda_new::{
    edge_bf16_decode_moe_available, edge_nvfp4_interleave_fc1, edge_nvfp4_moe_available,
    edge_nvfp4_rescale_e4m3, edge_nvfp4_swizzle_scales, ops, transfers, CudaBuffer, CudaContext,
};

use super::architecture::{MixerKind, Qwen35MoeArchitecture};
use super::config::Qwen35MoeConfig;
use super::loader::{
    load_nvfp4_moe, load_qwen35_layer, load_qwen35_mtp, ManifestAccess, Qwen35SafetensorsLoader,
};
use super::runtime_weights::{
    AttentionWeights, DenseLinearWeights, GdnWeights, MixerWeights, Nvfp4LinearWeights,
    Qwen35LayerWeights, Qwen35MtpLayerWeights, Qwen35MtpWeights, Qwen35RuntimeWeights,
    RoutedMoeWeights,
};

const NVFP4_BLOCK_SIZE: usize = 16;

/// A BF16 linear operand in cuda-new GEMM's canonical `[K, N]` layout.
#[derive(Debug)]
pub struct DeviceDenseLinear {
    pub weight: Tensor,
    pub input_features: usize,
    pub output_features: usize,
}

/// One checkpoint-native ModelOpt NVFP4 projection on CUDA.
#[derive(Debug)]
pub struct DeviceNvfp4Linear {
    /// Packed E2M1 pairs, `[N, K / 2]`.
    pub packed_weight: Tensor,
    /// Unsigned-E4M3 block scales, `[N, K / 16]`, row-major.
    pub block_scales: Tensor,
    /// tcgen05 atom layout used by the SM110 shared-expert GEMM.
    pub gemm_block_scales: Option<Tensor>,
    /// Static activation scale retained for a future quantized GEMM candidate.
    pub input_scale: f32,
    /// Per-tensor weight scale consumed by the current BF16-input correctness
    /// candidate.
    pub weight_scale_2: Tensor,
    /// Host scalar mirror used to configure quantized GEMM without a D2H copy.
    pub weight_scale_2_value: f32,
    pub input_features: usize,
    pub output_features: usize,
}

/// TensorRT-Edge-LLM's SM110 CuTeDSL routed-expert layout.
#[derive(Debug)]
pub struct DeviceEdgeNvfp4Experts {
    pub fc1_weight: Tensor,
    pub fc1_scales: Tensor,
    pub fc1_alpha: Tensor,
    pub input_global_scale: Tensor,
    pub fc2_weight: Tensor,
    pub fc2_scales: Tensor,
    pub fc2_alpha: Tensor,
    pub down_input_scale: Tensor,
    pub experts: usize,
    pub hidden: usize,
    pub intermediate: usize,
}

#[derive(Debug)]
pub enum DeviceNvfp4RoutedExperts {
    Reference {
        gate: DeviceNvfp4ExpertProjection,
        up: DeviceNvfp4ExpertProjection,
        down: DeviceNvfp4ExpertProjection,
    },
    Edge(DeviceEdgeNvfp4Experts),
}

/// One projection from all routed experts, packed into expert-major storage.
#[derive(Debug)]
pub struct DeviceNvfp4ExpertProjection {
    /// `[experts, N, K / 2]`.
    pub packed_weight: Tensor,
    /// `[experts, N, K / 16]`, row-major.
    pub block_scales: Tensor,
    /// `[experts]`; kept because an optimized quantized-input candidate needs
    /// the exact per-expert activation scale rather than one guessed value.
    pub input_scales: Tensor,
    /// `[experts]`; this is `Nvfp4ExpertProjection::alpha` for the current
    /// BF16-input routed-MoE CUDA semantic.
    pub weight_scales_2: Tensor,
    pub experts: usize,
    pub input_features: usize,
    pub output_features: usize,
}

#[derive(Debug)]
pub struct DeviceNvfp4Moe {
    pub router: DeviceDenseLinear,
    pub routed_experts: DeviceNvfp4RoutedExperts,
    pub shared_gate: DeviceNvfp4Linear,
    pub shared_up: DeviceNvfp4Linear,
    pub shared_down: DeviceNvfp4Linear,
    pub shared_expert_gate: DeviceDenseLinear,
}

/// BF16 MTP experts are packed by projection and remain `[E, K, N]`, the
/// native selected-expert layout consumed by `qwen35_mtp_moe`.
#[derive(Debug)]
pub struct DeviceDenseExpertProjection {
    pub weight: Tensor,
    pub experts: usize,
    pub input_features: usize,
    pub output_features: usize,
}

#[derive(Debug)]
pub struct DeviceEdgeBf16Experts {
    /// `[experts, 2 * intermediate, hidden]`, 64-row `[up, gate]` interleaved.
    pub fc1_weight: Tensor,
    /// `[experts, hidden, intermediate]`.
    pub fc2_weight: Tensor,
    pub experts: usize,
    pub hidden: usize,
    pub intermediate: usize,
}

#[derive(Debug)]
pub enum DeviceDenseRoutedExperts {
    Reference {
        gate: DeviceDenseExpertProjection,
        up: DeviceDenseExpertProjection,
        down: DeviceDenseExpertProjection,
    },
    Edge(DeviceEdgeBf16Experts),
}

#[derive(Debug)]
pub struct DeviceDenseMoe {
    pub router: DeviceDenseLinear,
    pub routed_experts: DeviceDenseRoutedExperts,
    pub shared_gate: DeviceDenseLinear,
    pub shared_up: DeviceDenseLinear,
    pub shared_down: DeviceDenseLinear,
    pub shared_expert_gate: DeviceDenseLinear,
}

#[derive(Debug)]
pub struct DeviceAttentionWeights {
    pub q_proj: DeviceDenseLinear,
    pub k_proj: DeviceDenseLinear,
    pub v_proj: DeviceDenseLinear,
    pub o_proj: DeviceDenseLinear,
    pub q_norm: Tensor,
    pub k_norm: Tensor,
    pub kv_cache_scales: Option<super::runtime_weights::KvCacheScales>,
}

#[derive(Debug)]
pub struct DeviceGdnWeights {
    pub a_log: Tensor,
    pub conv1d: Tensor,
    pub dt_bias: Tensor,
    pub in_proj_a: DeviceDenseLinear,
    pub in_proj_b: DeviceDenseLinear,
    pub in_proj_qkv: DeviceDenseLinear,
    pub in_proj_z: DeviceDenseLinear,
    pub norm: Tensor,
    pub out_proj: DeviceDenseLinear,
}

#[derive(Debug)]
pub enum DeviceMixerWeights {
    Gdn(DeviceGdnWeights),
    FullAttention(DeviceAttentionWeights),
}

#[derive(Debug)]
pub struct DeviceLayerWeights {
    pub input_layernorm: Tensor,
    pub post_attention_layernorm: Tensor,
    pub mixer: DeviceMixerWeights,
    pub moe: DeviceNvfp4Moe,
}

#[derive(Debug)]
pub struct DeviceMtpLayerWeights {
    pub input_layernorm: Tensor,
    pub post_attention_layernorm: Tensor,
    pub attention: DeviceAttentionWeights,
    pub moe: DeviceDenseMoe,
}

#[derive(Debug)]
pub struct DeviceMtpWeights {
    pub pre_fc_norm_embedding: Tensor,
    pub pre_fc_norm_hidden: Tensor,
    pub fc: DeviceDenseLinear,
    pub norm: Tensor,
    pub layer: DeviceMtpLayerWeights,
}

/// Complete resident weight tree. No CPU tensor is retained by this object.
#[derive(Debug)]
pub struct Qwen35DeviceWeights {
    pub embedding: Tensor,
    pub layers: Vec<DeviceLayerWeights>,
    pub final_norm: Tensor,
    /// `[hidden, vocab]`, ready for cuda-new GEMM.
    pub lm_head: Tensor,
    pub mtp: DeviceMtpWeights,
}

/// Validate, transform, and upload the complete text+MTP weight tree.
///
/// This operation intentionally consumes the host tree so individual source
/// tensors can be released after their device representation is materialized.
/// It does not catch CUDA OOM: the caller receives the original allocation
/// error, which is important on DRIVE guests with a restricted GPU carveout.
pub fn upload_qwen35_device_weights(
    ctx: &CudaContext,
    config: &Qwen35MoeConfig,
    host: Qwen35RuntimeWeights<Tensor>,
) -> Result<Qwen35DeviceWeights> {
    host.validate_against(config)?;
    let architecture = Qwen35MoeArchitecture::from_config(config)?;

    let Qwen35RuntimeWeights {
        embedding,
        layers,
        final_norm,
        lm_head,
        mtp,
    } = host;
    if layers.len() != architecture.layers.len() {
        return Err(invalid("Qwen3.5 device upload layer count mismatch"));
    }

    let embedding = upload_unchanged(ctx, embedding)?;
    let final_norm = upload_unchanged(ctx, final_norm)?;
    let lm_head = transpose_and_upload_bf16(
        ctx,
        lm_head,
        architecture.vocab_size,
        architecture.hidden_size,
        "lm_head",
    )?;

    let mut device_layers = Vec::with_capacity(layers.len());
    for (weights, layer) in layers.into_iter().zip(&architecture.layers) {
        device_layers.push(upload_layer(ctx, weights, layer.mixer)?);
    }

    Ok(Qwen35DeviceWeights {
        embedding,
        layers: device_layers,
        final_norm,
        lm_head,
        mtp: upload_mtp(ctx, mtp)?,
    })
}

/// Header-validate the exact checkpoint, then materialize and upload one
/// language-model layer at a time. The SafeTensors payload remains mmap-backed
/// until an individual tensor is requested, and visual tensors are never
/// copied into anonymous host memory.
pub fn load_qwen35_device_weights_from_path(
    ctx: &CudaContext,
    config: &Qwen35MoeConfig,
    path: &Path,
) -> Result<Qwen35DeviceWeights> {
    let architecture = Qwen35MoeArchitecture::from_config(config)?;
    let mut loader = Qwen35SafetensorsLoader::open(config, path)?;

    let embedding = upload_unchanged(
        ctx,
        loader.take("model.language_model.embed_tokens.weight")?,
    )?;
    let final_norm = upload_unchanged(ctx, loader.take("model.language_model.norm.weight")?)?;
    let lm_head = transpose_and_upload_bf16(
        ctx,
        loader.take("lm_head.weight")?,
        architecture.vocab_size,
        architecture.hidden_size,
        "lm_head",
    )?;

    let mut layers = Vec::with_capacity(architecture.layers.len());
    for layer in &architecture.layers {
        let host_layer = load_qwen35_layer(&mut loader, layer, architecture.hidden_size)?;
        layers.push(upload_layer(ctx, host_layer, layer.mixer)?);
    }
    let mtp = upload_mtp(
        ctx,
        load_qwen35_mtp(&mut loader, &architecture, architecture.hidden_size)?,
    )?;
    let _payload_bytes_copied = loader.finish()?;

    Ok(Qwen35DeviceWeights {
        embedding,
        layers,
        final_norm,
        lm_head,
        mtp,
    })
}

/// Load and prepare one real routed-MoE layer without materializing the rest
/// of the checkpoint. This is intended for operator validation and profiling;
/// full model execution should use [`load_qwen35_device_weights_from_path`].
pub fn load_qwen35_moe_layer_from_path(
    ctx: &CudaContext,
    config: &Qwen35MoeConfig,
    path: &Path,
    layer_index: usize,
) -> Result<DeviceNvfp4Moe> {
    let architecture = Qwen35MoeArchitecture::from_config(config)?;
    let layer = architecture
        .layers
        .get(layer_index)
        .ok_or_else(|| invalid(format!("Qwen3.5 layer index {layer_index} is out of range")))?;
    let mut loader = Qwen35SafetensorsLoader::open(config, path)?;
    let prefix = format!("model.language_model.layers.{layer_index}.mlp");
    let host = load_nvfp4_moe(
        &mut loader,
        &prefix,
        architecture.hidden_size,
        layer.moe.experts,
        layer.moe.expert_intermediate_size,
        layer.moe.shared_expert_intermediate_size,
    )?;
    upload_nvfp4_moe(ctx, host)
}

fn upload_layer(
    ctx: &CudaContext,
    weights: Qwen35LayerWeights<Tensor>,
    expected_mixer: MixerKind,
) -> Result<DeviceLayerWeights> {
    let Qwen35LayerWeights {
        input_layernorm,
        post_attention_layernorm,
        mixer,
        moe,
    } = weights;
    let mixer = match (mixer, expected_mixer) {
        (MixerWeights::Gdn(weights), MixerKind::Gdn(_)) => {
            DeviceMixerWeights::Gdn(upload_gdn(ctx, weights)?)
        }
        (MixerWeights::FullAttention(weights), MixerKind::FullAttention(_)) => {
            DeviceMixerWeights::FullAttention(upload_attention(ctx, weights)?)
        }
        _ => {
            return Err(invalid(
                "Qwen3.5 device upload mixer does not match architecture",
            ));
        }
    };
    Ok(DeviceLayerWeights {
        input_layernorm: upload_unchanged(ctx, input_layernorm)?,
        post_attention_layernorm: upload_unchanged(ctx, post_attention_layernorm)?,
        mixer,
        moe: upload_nvfp4_moe(ctx, moe)?,
    })
}

fn upload_mtp(ctx: &CudaContext, weights: Qwen35MtpWeights<Tensor>) -> Result<DeviceMtpWeights> {
    let Qwen35MtpWeights {
        pre_fc_norm_embedding,
        pre_fc_norm_hidden,
        fc,
        norm,
        layer,
    } = weights;
    let Qwen35MtpLayerWeights {
        input_layernorm,
        post_attention_layernorm,
        attention,
        moe,
    } = layer;
    Ok(DeviceMtpWeights {
        pre_fc_norm_embedding: upload_unchanged(ctx, pre_fc_norm_embedding)?,
        pre_fc_norm_hidden: upload_unchanged(ctx, pre_fc_norm_hidden)?,
        fc: upload_dense(ctx, fc, "MTP fc")?,
        norm: upload_unchanged(ctx, norm)?,
        layer: DeviceMtpLayerWeights {
            input_layernorm: upload_unchanged(ctx, input_layernorm)?,
            post_attention_layernorm: upload_unchanged(ctx, post_attention_layernorm)?,
            attention: upload_attention(ctx, attention)?,
            moe: upload_dense_moe(ctx, moe)?,
        },
    })
}

fn upload_attention(
    ctx: &CudaContext,
    weights: AttentionWeights<Tensor>,
) -> Result<DeviceAttentionWeights> {
    Ok(DeviceAttentionWeights {
        q_proj: upload_dense(ctx, weights.q_proj, "attention q_proj")?,
        k_proj: upload_dense(ctx, weights.k_proj, "attention k_proj")?,
        v_proj: upload_dense(ctx, weights.v_proj, "attention v_proj")?,
        o_proj: upload_dense(ctx, weights.o_proj, "attention o_proj")?,
        q_norm: upload_unchanged(ctx, weights.q_norm)?,
        k_norm: upload_unchanged(ctx, weights.k_norm)?,
        kv_cache_scales: weights.kv_cache_scales,
    })
}

fn upload_gdn(ctx: &CudaContext, weights: GdnWeights<Tensor>) -> Result<DeviceGdnWeights> {
    let qkv_width = weights.in_proj_qkv.output_features;
    let conv_kernel = *weights
        .conv1d
        .shape()
        .dims()
        .get(2)
        .ok_or_else(|| invalid("GDN conv1d checkpoint tensor must be rank 3"))?;
    Ok(DeviceGdnWeights {
        a_log: upload_unchanged(ctx, weights.a_log)?,
        // Keep the public cuda-new contract `[channels, 1, kernel]`. The GDN
        // adapter creates a byte-identical `[channels, kernel]` view only at
        // the native causal-convolution boundary.
        conv1d: {
            validate_cpu_tensor(
                &weights.conv1d,
                DType::BF16,
                &[qkv_width, 1, conv_kernel],
                "GDN conv1d",
            )?;
            upload_unchanged(ctx, weights.conv1d)?
        },
        dt_bias: upload_unchanged(ctx, weights.dt_bias)?,
        in_proj_a: upload_dense(ctx, weights.in_proj_a, "GDN in_proj_a")?,
        in_proj_b: upload_dense(ctx, weights.in_proj_b, "GDN in_proj_b")?,
        in_proj_qkv: upload_dense(ctx, weights.in_proj_qkv, "GDN in_proj_qkv")?,
        in_proj_z: upload_dense(ctx, weights.in_proj_z, "GDN in_proj_z")?,
        norm: upload_unchanged(ctx, weights.norm)?,
        out_proj: upload_dense(ctx, weights.out_proj, "GDN out_proj")?,
    })
}

fn upload_nvfp4_moe(
    ctx: &CudaContext,
    weights: RoutedMoeWeights<Tensor, Nvfp4LinearWeights<Tensor>>,
) -> Result<DeviceNvfp4Moe> {
    let RoutedMoeWeights {
        router,
        experts,
        shared_expert,
        shared_expert_gate,
    } = weights;
    if experts.is_empty() {
        return Err(invalid("Qwen3.5 routed expert set is empty"));
    }
    let mut gate = Vec::with_capacity(experts.len());
    let mut up = Vec::with_capacity(experts.len());
    let mut down = Vec::with_capacity(experts.len());
    for expert in experts {
        gate.push(expert.gate_proj);
        up.push(expert.up_proj);
        down.push(expert.down_proj);
    }
    let routed_experts = if edge_nvfp4_moe_available() {
        DeviceNvfp4RoutedExperts::Edge(pack_and_upload_edge_nvfp4_experts(ctx, gate, up, down)?)
    } else {
        DeviceNvfp4RoutedExperts::Reference {
            gate: upload_nvfp4_expert_projection(ctx, gate, "expert gate")?,
            up: upload_nvfp4_expert_projection(ctx, up, "expert up")?,
            down: upload_nvfp4_expert_projection(ctx, down, "expert down")?,
        }
    };
    Ok(DeviceNvfp4Moe {
        router: upload_dense(ctx, router, "MoE router")?,
        routed_experts,
        shared_gate: upload_nvfp4(ctx, shared_expert.gate_proj, "shared expert gate")?,
        shared_up: upload_nvfp4(ctx, shared_expert.up_proj, "shared expert up")?,
        shared_down: upload_nvfp4(ctx, shared_expert.down_proj, "shared expert down")?,
        shared_expert_gate: upload_dense(ctx, shared_expert_gate, "shared expert gate scalar")?,
    })
}

fn upload_dense_moe(
    ctx: &CudaContext,
    weights: RoutedMoeWeights<Tensor, DenseLinearWeights<Tensor>>,
) -> Result<DeviceDenseMoe> {
    let RoutedMoeWeights {
        router,
        experts,
        shared_expert,
        shared_expert_gate,
    } = weights;
    if experts.is_empty() {
        return Err(invalid("Qwen3.5 MTP expert set is empty"));
    }
    let mut gate = Vec::with_capacity(experts.len());
    let mut up = Vec::with_capacity(experts.len());
    let mut down = Vec::with_capacity(experts.len());
    for expert in experts {
        gate.push(expert.gate_proj);
        up.push(expert.up_proj);
        down.push(expert.down_proj);
    }
    let routed_experts = if edge_bf16_decode_moe_available() {
        DeviceDenseRoutedExperts::Edge(pack_and_upload_edge_bf16_experts(ctx, gate, up, down)?)
    } else {
        DeviceDenseRoutedExperts::Reference {
            gate: upload_dense_expert_projection(ctx, gate, "MTP expert gate")?,
            up: upload_dense_expert_projection(ctx, up, "MTP expert up")?,
            down: upload_dense_expert_projection(ctx, down, "MTP expert down")?,
        }
    };
    Ok(DeviceDenseMoe {
        router: upload_dense(ctx, router, "MTP MoE router")?,
        routed_experts,
        shared_gate: upload_dense(ctx, shared_expert.gate_proj, "MTP shared gate")?,
        shared_up: upload_dense(ctx, shared_expert.up_proj, "MTP shared up")?,
        shared_down: upload_dense(ctx, shared_expert.down_proj, "MTP shared down")?,
        shared_expert_gate: upload_dense(ctx, shared_expert_gate, "MTP shared expert gate")?,
    })
}

fn pack_and_upload_edge_bf16_experts(
    ctx: &CudaContext,
    gate: Vec<DenseLinearWeights<Tensor>>,
    up: Vec<DenseLinearWeights<Tensor>>,
    down: Vec<DenseLinearWeights<Tensor>>,
) -> Result<DeviceEdgeBf16Experts> {
    if gate.is_empty() || gate.len() != up.len() || gate.len() != down.len() {
        return Err(invalid("Edge BF16 expert projection counts differ"));
    }
    let experts = gate.len();
    let hidden = gate[0].input_features;
    let intermediate = gate[0].output_features;
    let mut fc1 = Vec::new();
    let mut fc2 = Vec::new();
    let fc1_bytes = experts
        .checked_mul(2 * intermediate)
        .and_then(|value| value.checked_mul(hidden))
        .and_then(|value| value.checked_mul(DType::BF16.size_in_bytes()))
        .ok_or_else(|| invalid("Edge BF16 FC1 byte size overflow"))?;
    let fc2_bytes = experts
        .checked_mul(hidden)
        .and_then(|value| value.checked_mul(intermediate))
        .and_then(|value| value.checked_mul(DType::BF16.size_in_bytes()))
        .ok_or_else(|| invalid("Edge BF16 FC2 byte size overflow"))?;
    fc1.try_reserve_exact(fc1_bytes)
        .map_err(|error| invalid(format!("allocate Edge BF16 FC1 staging: {error}")))?;
    fc2.try_reserve_exact(fc2_bytes)
        .map_err(|error| invalid(format!("allocate Edge BF16 FC2 staging: {error}")))?;
    for ((gate, up), down) in gate.into_iter().zip(up).zip(down) {
        if gate.input_features != hidden
            || gate.output_features != intermediate
            || up.input_features != hidden
            || up.output_features != intermediate
            || down.input_features != intermediate
            || down.output_features != hidden
        {
            return Err(invalid("Edge BF16 expert projection geometries differ"));
        }
        validate_cpu_tensor(
            &gate.weight,
            DType::BF16,
            &[intermediate, hidden],
            "Edge BF16 expert gate",
        )?;
        validate_cpu_tensor(
            &up.weight,
            DType::BF16,
            &[intermediate, hidden],
            "Edge BF16 expert up",
        )?;
        validate_cpu_tensor(
            &down.weight,
            DType::BF16,
            &[hidden, intermediate],
            "Edge BF16 expert down",
        )?;
        let packed = edge_nvfp4_interleave_fc1(
            cpu_bytes(&gate.weight, "Edge BF16 expert gate")?,
            cpu_bytes(&up.weight, "Edge BF16 expert up")?,
            intermediate,
            hidden * DType::BF16.size_in_bytes(),
        )?;
        fc1.extend_from_slice(&packed);
        fc2.extend_from_slice(cpu_bytes(&down.weight, "Edge BF16 expert down")?);
    }
    let fc1 = Tensor::from_raw(
        Shape::new(vec![experts, 2 * intermediate, hidden]),
        DType::BF16,
        Device::Cpu,
        fc1,
    )?;
    let fc2 = Tensor::from_raw(
        Shape::new(vec![experts, hidden, intermediate]),
        DType::BF16,
        Device::Cpu,
        fc2,
    )?;
    Ok(DeviceEdgeBf16Experts {
        fc1_weight: transfers::to_cuda(&fc1, ctx.device_id())?,
        fc2_weight: transfers::to_cuda(&fc2, ctx.device_id())?,
        experts,
        hidden,
        intermediate,
    })
}

fn upload_dense(
    ctx: &CudaContext,
    weights: DenseLinearWeights<Tensor>,
    label: &str,
) -> Result<DeviceDenseLinear> {
    let weight = transpose_and_upload_bf16(
        ctx,
        weights.weight,
        weights.output_features,
        weights.input_features,
        label,
    )?;
    Ok(DeviceDenseLinear {
        weight,
        input_features: weights.input_features,
        output_features: weights.output_features,
    })
}

fn upload_nvfp4(
    ctx: &CudaContext,
    weights: Nvfp4LinearWeights<Tensor>,
    label: &str,
) -> Result<DeviceNvfp4Linear> {
    weights.validate()?;
    validate_cpu_tensor(
        &weights.packed_weight,
        DType::E2M1Pair,
        &[weights.output_features, weights.input_features / 2],
        &format!("{label} packed weight"),
    )?;
    validate_cpu_tensor(
        &weights.block_scales,
        DType::F8E4M3,
        &[
            weights.output_features,
            weights.input_features / NVFP4_BLOCK_SIZE,
        ],
        &format!("{label} block scales"),
    )?;
    let weight_scale_2_value = weights.weight_scale_2;
    let packed_weight = upload_unchanged(ctx, weights.packed_weight)?;
    let block_scales = upload_unchanged(ctx, weights.block_scales)?;
    let gemm_block_scales = if edge_nvfp4_moe_available() {
        let bytes = ops::nvfp4_scale_buffer_bytes(
            weights.output_features,
            weights.input_features,
            NVFP4_BLOCK_SIZE as u32,
        )?;
        let tensor = CudaBuffer::alloc(bytes, ctx.device_id())
            .map_err(Error::Cuda)?
            .as_tensor(Shape::new(vec![bytes]), DType::F8E4M3)
            .map_err(Error::Cuda)?;
        ops::nvfp4_pack_block_scales(
            ctx,
            &block_scales,
            &tensor,
            weights.output_features,
            weights.input_features,
            NVFP4_BLOCK_SIZE as u32,
        )?;
        Some(tensor)
    } else {
        None
    };
    Ok(DeviceNvfp4Linear {
        packed_weight,
        block_scales,
        gemm_block_scales,
        input_scale: weights.input_scale,
        weight_scale_2: upload_f32(ctx, &[weight_scale_2_value])?,
        weight_scale_2_value,
        input_features: weights.input_features,
        output_features: weights.output_features,
    })
}

fn pack_and_upload_edge_nvfp4_experts(
    ctx: &CudaContext,
    gate: Vec<Nvfp4LinearWeights<Tensor>>,
    up: Vec<Nvfp4LinearWeights<Tensor>>,
    down: Vec<Nvfp4LinearWeights<Tensor>>,
) -> Result<DeviceEdgeNvfp4Experts> {
    let experts = gate.len();
    if experts != 256 || up.len() != experts || down.len() != experts {
        return Err(invalid(
            "Edge NVFP4 routed experts require 256 gate/up/down projections",
        ));
    }
    let first = gate
        .first()
        .ok_or_else(|| invalid("Edge NVFP4 routed expert set is empty"))?;
    let hidden = first.input_features;
    let intermediate = first.output_features;
    if hidden % 128 != 0 || intermediate % 64 != 0 {
        return Err(invalid(
            "Edge NVFP4 routed expert geometry is not SM110-kernel compatible",
        ));
    }

    let mut fc1_weight = Vec::new();
    let mut fc1_scales = Vec::new();
    let mut fc1_alpha = Vec::with_capacity(experts);
    let mut input_global_scale = Vec::with_capacity(experts);
    let mut fc2_weight = Vec::new();
    let mut fc2_scales = Vec::new();
    let mut fc2_alpha = Vec::with_capacity(experts);
    let mut down_input_scale = Vec::with_capacity(experts);
    let fc1_scale_columns = hidden / NVFP4_BLOCK_SIZE;
    let fc2_scale_columns = intermediate / NVFP4_BLOCK_SIZE;

    for (expert, ((gate, up), down)) in gate.into_iter().zip(up).zip(down).enumerate() {
        gate.validate()?;
        up.validate()?;
        down.validate()?;
        if gate.input_features != hidden
            || gate.output_features != intermediate
            || up.input_features != hidden
            || up.output_features != intermediate
            || down.input_features != intermediate
            || down.output_features != hidden
        {
            return Err(invalid(format!(
                "Edge NVFP4 expert {expert} projection geometry mismatch"
            )));
        }
        if gate.input_scale.to_bits() != up.input_scale.to_bits() {
            return Err(invalid(format!(
                "Edge NVFP4 expert {expert} gate/up input scales differ"
            )));
        }
        if !gate.weight_scale_2.is_finite()
            || !(gate.weight_scale_2 > 0.0)
            || !up.weight_scale_2.is_finite()
            || !(up.weight_scale_2 > 0.0)
            || !down.weight_scale_2.is_finite()
            || !(down.weight_scale_2 > 0.0)
            || !gate.input_scale.is_finite()
            || !(gate.input_scale > 0.0)
            || !down.input_scale.is_finite()
            || !(down.input_scale > 0.0)
        {
            return Err(invalid(format!(
                "Edge NVFP4 expert {expert} scales must be finite and positive"
            )));
        }

        let shared_alpha = gate.weight_scale_2.max(up.weight_scale_2);
        let gate_scales = edge_nvfp4_rescale_e4m3(
            cpu_bytes(&gate.block_scales, "Edge expert gate scales")?,
            gate.weight_scale_2 / shared_alpha,
        )?;
        let up_scales = edge_nvfp4_rescale_e4m3(
            cpu_bytes(&up.block_scales, "Edge expert up scales")?,
            up.weight_scale_2 / shared_alpha,
        )?;
        let interleaved_weight = edge_nvfp4_interleave_fc1(
            cpu_bytes(&gate.packed_weight, "Edge expert gate weight")?,
            cpu_bytes(&up.packed_weight, "Edge expert up weight")?,
            intermediate,
            hidden / 2,
        )?;
        let interleaved_scales =
            edge_nvfp4_interleave_fc1(&gate_scales, &up_scales, intermediate, fc1_scale_columns)?;
        let (swizzled_fc1, _) =
            edge_nvfp4_swizzle_scales(&interleaved_scales, 2 * intermediate, fc1_scale_columns)?;
        let (swizzled_fc2, _) = edge_nvfp4_swizzle_scales(
            cpu_bytes(&down.block_scales, "Edge expert down scales")?,
            hidden,
            fc2_scale_columns,
        )?;

        fc1_weight.extend_from_slice(&interleaved_weight);
        fc1_scales.extend_from_slice(&swizzled_fc1);
        fc1_alpha.push(shared_alpha);
        input_global_scale.push(gate.input_scale);
        fc2_weight.extend_from_slice(cpu_bytes(&down.packed_weight, "Edge expert down weight")?);
        fc2_scales.extend_from_slice(&swizzled_fc2);
        fc2_alpha.push(down.weight_scale_2);
        down_input_scale.push(down.input_scale);
    }

    let upload_raw = |shape: Vec<usize>, dtype: DType, bytes: Vec<u8>| -> Result<Tensor> {
        let host = Tensor::from_raw(Shape::new(shape), dtype, Device::Cpu, bytes)?;
        transfers::to_cuda(&host, ctx.device_id())
    };
    let fc1_scale_shape = vec![
        experts,
        (2 * intermediate).div_ceil(128),
        fc1_scale_columns.div_ceil(4),
        32,
        4,
        4,
    ];
    let fc2_scale_shape = vec![
        experts,
        hidden.div_ceil(128),
        fc2_scale_columns.div_ceil(4),
        32,
        4,
        4,
    ];
    Ok(DeviceEdgeNvfp4Experts {
        fc1_weight: upload_raw(
            vec![experts, 2 * intermediate, hidden / 2],
            DType::E2M1Pair,
            fc1_weight,
        )?,
        fc1_scales: upload_raw(fc1_scale_shape, DType::F8E4M3, fc1_scales)?,
        fc1_alpha: upload_f32(ctx, &fc1_alpha)?,
        input_global_scale: upload_f32(ctx, &input_global_scale)?,
        fc2_weight: upload_raw(
            vec![experts, hidden, intermediate / 2],
            DType::E2M1Pair,
            fc2_weight,
        )?,
        fc2_scales: upload_raw(fc2_scale_shape, DType::F8E4M3, fc2_scales)?,
        fc2_alpha: upload_f32(ctx, &fc2_alpha)?,
        down_input_scale: upload_f32(ctx, &down_input_scale)?,
        experts,
        hidden,
        intermediate,
    })
}

fn upload_nvfp4_expert_projection(
    ctx: &CudaContext,
    weights: Vec<Nvfp4LinearWeights<Tensor>>,
    label: &str,
) -> Result<DeviceNvfp4ExpertProjection> {
    let experts = weights.len();
    let first = weights
        .first()
        .ok_or_else(|| invalid(format!("{label} has no experts")))?;
    let input_features = first.input_features;
    let output_features = first.output_features;
    let mut packed = Vec::with_capacity(experts);
    let mut scales = Vec::with_capacity(experts);
    let mut input_scales = Vec::with_capacity(experts);
    let mut weight_scales_2 = Vec::with_capacity(experts);
    for weight in weights {
        weight.validate()?;
        if weight.input_features != input_features || weight.output_features != output_features {
            return Err(invalid(format!(
                "{label} expert projection geometries differ"
            )));
        }
        packed.push(weight.packed_weight);
        scales.push(weight.block_scales);
        input_scales.push(weight.input_scale);
        weight_scales_2.push(weight.weight_scale_2);
    }
    let packed_weight = stack_and_upload(
        ctx,
        packed,
        DType::E2M1Pair,
        &[output_features, input_features / 2],
        &[experts, output_features, input_features / 2],
        &format!("{label} packed weights"),
    )?;
    let block_scales = stack_and_upload(
        ctx,
        scales,
        DType::F8E4M3,
        &[output_features, input_features / NVFP4_BLOCK_SIZE],
        &[experts, output_features, input_features / NVFP4_BLOCK_SIZE],
        &format!("{label} block scales"),
    )?;
    Ok(DeviceNvfp4ExpertProjection {
        packed_weight,
        block_scales,
        input_scales: upload_f32(ctx, &input_scales)?,
        weight_scales_2: upload_f32(ctx, &weight_scales_2)?,
        experts,
        input_features,
        output_features,
    })
}

fn upload_dense_expert_projection(
    ctx: &CudaContext,
    weights: Vec<DenseLinearWeights<Tensor>>,
    label: &str,
) -> Result<DeviceDenseExpertProjection> {
    let experts = weights.len();
    let first = weights
        .first()
        .ok_or_else(|| invalid(format!("{label} has no experts")))?;
    let input_features = first.input_features;
    let output_features = first.output_features;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(
            experts
                .checked_mul(input_features)
                .and_then(|value| value.checked_mul(output_features))
                .and_then(|value| value.checked_mul(DType::BF16.size_in_bytes()))
                .ok_or_else(|| invalid(format!("{label} byte size overflow")))?,
        )
        .map_err(|error| invalid(format!("allocate {label} staging: {error}")))?;
    for weight in weights {
        if weight.input_features != input_features || weight.output_features != output_features {
            return Err(invalid(format!(
                "{label} expert projection geometries differ"
            )));
        }
        let transposed =
            transpose_bf16_bytes(weight.weight, output_features, input_features, label)?;
        bytes.extend_from_slice(&transposed);
    }
    let host = Tensor::from_raw(
        Shape::new(vec![experts, input_features, output_features]),
        DType::BF16,
        Device::Cpu,
        bytes,
    )?;
    Ok(DeviceDenseExpertProjection {
        weight: transfers::to_cuda(&host, ctx.device_id())?,
        experts,
        input_features,
        output_features,
    })
}

fn transpose_and_upload_bf16(
    ctx: &CudaContext,
    tensor: Tensor,
    rows: usize,
    columns: usize,
    label: &str,
) -> Result<Tensor> {
    let bytes = transpose_bf16_bytes(tensor, rows, columns, label)?;
    let host = Tensor::from_raw(
        Shape::new(vec![columns, rows]),
        DType::BF16,
        Device::Cpu,
        bytes,
    )?;
    transfers::to_cuda(&host, ctx.device_id())
}

fn transpose_bf16_bytes(
    tensor: Tensor,
    rows: usize,
    columns: usize,
    label: &str,
) -> Result<Vec<u8>> {
    validate_cpu_tensor(&tensor, DType::BF16, &[rows, columns], label)?;
    let source = cpu_bytes(&tensor, label)?;
    let element = DType::BF16.size_in_bytes();
    let mut transposed = vec![0u8; source.len()];
    for row in 0..rows {
        for column in 0..columns {
            let from = (row * columns + column) * element;
            let to = (column * rows + row) * element;
            transposed[to..to + element].copy_from_slice(&source[from..from + element]);
        }
    }
    Ok(transposed)
}

fn stack_and_upload(
    ctx: &CudaContext,
    tensors: Vec<Tensor>,
    dtype: DType,
    each_shape: &[usize],
    stacked_shape: &[usize],
    label: &str,
) -> Result<Tensor> {
    let expected = stacked_shape
        .iter()
        .try_fold(dtype.size_in_bytes(), |bytes, dimension| {
            bytes.checked_mul(*dimension)
        })
        .ok_or_else(|| invalid(format!("{label} byte size overflow")))?;
    let mut bytes = Vec::new();
    bytes
        .try_reserve_exact(expected)
        .map_err(|error| invalid(format!("allocate {label} staging: {error}")))?;
    for tensor in tensors {
        validate_cpu_tensor(&tensor, dtype, each_shape, label)?;
        bytes.extend_from_slice(cpu_bytes(&tensor, label)?);
    }
    if bytes.len() != expected {
        return Err(invalid(format!(
            "{label} stacked byte size mismatch: expected {expected}, got {}",
            bytes.len()
        )));
    }
    let host = Tensor::from_raw(
        Shape::new(stacked_shape.to_vec()),
        dtype,
        Device::Cpu,
        bytes,
    )?;
    transfers::to_cuda(&host, ctx.device_id())
}

fn upload_f32(ctx: &CudaContext, values: &[f32]) -> Result<Tensor> {
    let host = Tensor::from_f32(vec![values.len()], values)?;
    transfers::to_cuda(&host, ctx.device_id())
}

fn upload_unchanged(ctx: &CudaContext, tensor: Tensor) -> Result<Tensor> {
    if tensor.device() != Device::Cpu {
        return Err(invalid("Qwen3.5 device upload expects CPU source tensors"));
    }
    transfers::to_cuda(&tensor, ctx.device_id())
}

fn validate_cpu_tensor(tensor: &Tensor, dtype: DType, shape: &[usize], label: &str) -> Result<()> {
    if tensor.device() != Device::Cpu || tensor.dtype() != dtype || tensor.shape().dims() != shape {
        return Err(invalid(format!(
            "{label} expected CPU {dtype} {shape:?}, got {:?} {} {:?}",
            tensor.device(),
            tensor.dtype(),
            tensor.shape().dims()
        )));
    }
    Ok(())
}

fn cpu_bytes<'a>(tensor: &'a Tensor, label: &str) -> Result<&'a [u8]> {
    tensor
        .storage()
        .as_cpu()
        .ok_or_else(|| invalid(format!("{label} is not backed by CPU storage")))
}

fn invalid(message: impl Into<String>) -> Error {
    Error::Other(message.into())
}

#[cfg(test)]
mod tests {
    use half::bf16;

    use super::*;

    #[test]
    fn dense_checkpoint_weight_is_physically_transposed_to_gemm_layout() {
        let values = (0..6)
            .map(|value| bf16::from_f32(value as f32))
            .collect::<Vec<_>>();
        let tensor = Tensor::from_bf16(vec![2, 3], &values).unwrap();
        let bytes = transpose_bf16_bytes(tensor, 2, 3, "test").unwrap();
        let tensor =
            Tensor::from_raw(Shape::new(vec![3, 2]), DType::BF16, Device::Cpu, bytes).unwrap();
        let actual = tensor
            .as_bf16()
            .unwrap()
            .iter()
            .map(|value| value.to_f32())
            .collect::<Vec<_>>();
        assert_eq!(actual, vec![0.0, 3.0, 1.0, 4.0, 2.0, 5.0]);
    }

    #[test]
    fn dense_transpose_rejects_shape_or_dtype_drift() {
        let wrong_shape = Tensor::from_bf16(vec![3, 2], &[bf16::ZERO; 6]).unwrap();
        assert!(transpose_bf16_bytes(wrong_shape, 2, 3, "shape").is_err());
        let wrong_dtype = Tensor::from_f32(vec![2, 3], &[0.0; 6]).unwrap();
        assert!(transpose_bf16_bytes(wrong_dtype, 2, 3, "dtype").is_err());
    }

    #[test]
    fn gdn_conv_keeps_the_public_three_dimensional_contract() {
        let values = (0..8)
            .map(|value| bf16::from_f32(value as f32))
            .collect::<Vec<_>>();
        let source = Tensor::from_bf16(vec![2, 1, 4], &values).unwrap();
        validate_cpu_tensor(&source, DType::BF16, &[2, 1, 4], "GDN conv").unwrap();
        assert_eq!(source.shape().dims(), &[2, 1, 4]);
        assert_eq!(source.as_bf16().unwrap(), values.as_slice());

        assert!(validate_cpu_tensor(&source, DType::BF16, &[2, 4], "GDN conv").is_err());
    }
}
