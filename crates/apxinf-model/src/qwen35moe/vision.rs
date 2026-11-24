//! Native BF16 Qwen3.5 vision tower and image-embedding injection.
//!
//! Processor output packs ordered static images as canonical
//! `[patches, 1536]` F32/BF16 plus `grid_thw`, whose `t` must be 1.
//! Segmented attention keeps patches from different images independent.
//!
//! All per-request tower computation remains on CUDA. One fused kernel
//! interpolates the learned 48x48 position table, applies spatial-merge order,
//! and emits the matching axial IDs on every request. Vision and language
//! computation share the same cuda-new context and stream.

use std::path::Path;

use apxinf_core::{DType, Device, Error, Result, Shape, Tensor};
use apxinf_cuda_new::{ops, transfers, CudaBuffer, CudaContext};

use super::vision_config::*;
use super::vision_weights::VisionWeights;

struct PreparedVision {
    grids: Vec<[u32; 3]>,
    offsets: Vec<u32>,
    total_patches: usize,
}

fn upload_u32(ctx: &CudaContext, values: &[u32]) -> Result<CudaBuffer> {
    let bytes: Vec<u8> = values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect();
    let buffer = CudaBuffer::alloc(bytes.len().max(1), ctx.device_id()).map_err(Error::Cuda)?;
    buffer.copy_from_host(&bytes).map_err(Error::Cuda)?;
    Ok(buffer)
}

fn upload_i32(ctx: &CudaContext, dims: Vec<usize>, values: &[i32]) -> Result<Tensor> {
    transfers::to_cuda(&Tensor::from_i32(dims, values)?, ctx.device_id())
}

fn device_tensor(ctx: &CudaContext, dims: &[usize], dtype: DType) -> Result<Tensor> {
    let bytes = dims
        .iter()
        .product::<usize>()
        .checked_mul(dtype.size_in_bytes())
        .ok_or_else(|| Error::Other("qwen35moe vision tensor size overflow".into()))?;
    CudaBuffer::alloc(bytes.max(1), ctx.device_id())
        .map_err(Error::Cuda)?
        .as_tensor(Shape::new(dims.to_vec()), dtype)
        .map_err(Error::Cuda)
}

fn dump_vision_stage(stage: &str, tensor: &Tensor) -> Result<()> {
    let Ok(directory) = std::env::var("APXINF_QWEN35_VISION_DUMP_DIR") else {
        return Ok(());
    };
    if stage.starts_with("block_") && !stage.ends_with("_output") {
        let index: usize = stage[6..8].parse().unwrap_or(0);
        if index != 0 && index < 24 {
            return Ok(());
        }
    }
    let directory = Path::new(&directory);
    std::fs::create_dir_all(directory).map_err(|error| {
        Error::Other(format!(
            "create qwen35moe vision dump directory {}: {error}",
            directory.display()
        ))
    })?;
    let values = transfers::to_cpu(tensor)?.to_f32_vec()?;
    let bytes: Vec<u8> = values
        .iter()
        .flat_map(|value| value.to_le_bytes())
        .collect();
    let data_path = directory.join(format!("{stage}.f32"));
    std::fs::write(&data_path, bytes).map_err(|error| {
        Error::Other(format!(
            "write qwen35moe vision stage {}: {error}",
            data_path.display()
        ))
    })?;
    let shape = tensor
        .shape()
        .dims()
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>()
        .join(" ");
    let shape_path = directory.join(format!("{stage}.shape"));
    std::fs::write(&shape_path, format!("{shape}\n")).map_err(|error| {
        Error::Other(format!(
            "write qwen35moe vision stage shape {}: {error}",
            shape_path.display()
        ))
    })?;
    Ok(())
}

pub(crate) fn upload_pixels(ctx: &CudaContext, pixels: &Tensor) -> Result<Tensor> {
    let on_device = if pixels.device() != Device::Cuda(ctx.device_id()) {
        transfers::to_cuda(pixels, ctx.device_id())?
    } else {
        pixels.clone()
    };
    match on_device.dtype() {
        DType::BF16 => Ok(on_device),
        DType::F32 => {
            let dims = on_device.shape().dims().to_vec();
            let mut out = device_tensor(ctx, &dims, DType::BF16)?;
            ops::quantization(
                ctx,
                ops::QuantizationArgs::new(
                    ops::QuantizationSemantic::CastF32ToBf16,
                    &on_device,
                    &mut out,
                ),
            )?;
            Ok(out)
        }
        dtype => Err(Error::Other(format!(
            "qwen35moe image pixel_values must be f32 or bf16, got {dtype}"
        ))),
    }
}

/// Run the complete vision tower over ordered packed images. Each image has
/// its own position grid and independent segmented attention span.
///
/// `scratch` supplies stable per-shape buffers and `session` owns the prepared
/// GEMM/Attention executions. The first call for a given scratch must pass
/// `prepared = false` so the traversal may tune and create native resources;
/// later calls on the same scratch replay them without allocating. Sessions do
/// not nest, so this must not be called inside the language prefill session.
pub(crate) fn forward(
    weights: &VisionWeights,
    ctx: &CudaContext,
    scratch: &VisionScratch,
    session: &ops::ExecutionSession,
    prepared: bool,
    pixel_values: &Tensor,
    grid_thw: &[[u32; 3]],
) -> Result<Tensor> {
    if weights.pos_embed.shape().dims() != [VISION_POSITIONS, VISION_HIDDEN] {
        return Err(Error::Other(
            "qwen35moe image position table shape mismatch".into(),
        ));
    }
    let plan = prepare(pixel_values, grid_thw)?;
    if scratch.grids != plan.grids {
        return Err(Error::Other(
            "Qwen3.5 vision scratch does not match this image".into(),
        ));
    }
    // Prepared executions bind addresses. Each request may upload pixels to a
    // different address, so stage them into this traversal's retained input.
    ops::copy_tensor(ctx, pixel_values, &scratch.pixels)?;
    let run = || {
        for (index, &[_, height, width]) in plan.grids.iter().enumerate() {
            let start = plan.offsets[index] as usize;
            let patches = (plan.offsets[index + 1] - plan.offsets[index]) as usize;
            let mut pos_embeds = row_view(&scratch.pos_embeds, start, patches)?;
            let mut pos_ids = row_view(&scratch.pos_ids, start, patches)?;
            ops::vision_position_staged(
                ctx,
                ops::VisionPositionArgs {
                    table: &weights.pos_embed,
                    embeddings: &mut pos_embeds,
                    position_ids: &mut pos_ids,
                    height: height as usize,
                    width: width as usize,
                    merge_size: VISION_MERGE,
                },
            )?;
        }
        forward_device(weights, ctx, scratch, &scratch.pixels, &plan)
    };
    let output = if prepared {
        ops::with_session(session, run)?
    } else {
        ops::prepare_with_session(session, run)?
    };
    ctx.synchronize().map_err(Error::Cuda)?;
    if let Ok(path) = std::env::var("APXINF_QWEN35_VISION_DUMP") {
        let values = transfers::to_cpu(&output)?.to_f32_vec()?;
        let bytes: Vec<u8> = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        std::fs::write(&path, bytes).map_err(|error| {
            Error::Other(format!("write qwen35moe vision dump {path}: {error}"))
        })?;
        eprintln!(
            "qwen35moe vision dump: {} rows x {} columns -> {path}",
            output.shape().dims()[0],
            output.shape().dims()[1]
        );
    }
    Ok(output)
}

fn row_view(tensor: &Tensor, start: usize, rows: usize) -> Result<Tensor> {
    let columns = tensor.shape().dims()[1];
    let row_bytes = columns * tensor.dtype().size_in_bytes();
    CudaBuffer::from_tensor(tensor)
        .map_err(Error::Cuda)?
        .view(start * row_bytes, rows * row_bytes)
        .map_err(Error::Cuda)?
        .as_tensor(Shape::new(vec![rows, columns]), tensor.dtype())
        .map_err(Error::Cuda)
}

fn prepare(pixel_values: &Tensor, grid_thw: &[[u32; 3]]) -> Result<PreparedVision> {
    let patch_width =
        VISION_CHANNELS * VISION_TEMPORAL_PATCH_SIZE * VISION_PATCH_SIZE * VISION_PATCH_SIZE;
    let dims = pixel_values.shape().dims();
    if grid_thw.is_empty() || dims.len() != 2 || dims[1] != patch_width {
        return Err(Error::Other(format!(
            "qwen35moe image pixel_values must be [patches, {patch_width}], got {dims:?}"
        )));
    }

    let total_patches = dims[0];
    let offsets = super::vision_input::image_patch_offsets(grid_thw)?;
    if offsets.last().copied() != Some(total_patches as u32) || total_patches > i32::MAX as usize {
        return Err(Error::Other(
            "image grid patch count does not match pixel_values".into(),
        ));
    }
    Ok(PreparedVision {
        grids: grid_thw.to_vec(),
        offsets,
        total_patches,
    })
}

/// Device scratch for one patch count, allocated once and reused by every
/// block.
///
/// Reuse needs no in-place operator support: each block writes only buffers it
/// does not read. `x` alternates between two buffers because a residual add
/// cannot write its own input, and `normed` is rewritten only after the GEMM
/// that reads it has been enqueued. Addresses are stable for the lifetime of
/// the scratch, which is what lets the session's GEMM/Attention execution
/// cache hit on a repeated grid -- `ExecutionKey` includes binding pointers.
pub(crate) struct VisionScratch {
    patches: usize,
    grids: Vec<[u32; 3]>,
    offsets: CudaBuffer,
    pixels: Tensor,
    packed_pixels: Tensor,
    pos_embeds: Tensor,
    pos_ids: Tensor,
    x: [Tensor; 2],
    normed: Tensor,
    qkv: Tensor,
    query: Tensor,
    key: Tensor,
    value: Tensor,
    attn: Tensor,
    projected: Tensor,
    mlp_projected: Tensor,
    mlp_hidden: Tensor,
    mlp_out: Tensor,
    merger_hidden: Tensor,
    merger_activated: Tensor,
    merger_out: Tensor,
}

impl VisionScratch {
    pub(crate) fn patches(&self) -> usize {
        self.patches
    }

    pub(crate) fn matches_grids(&self, grids: &[[u32; 3]]) -> bool {
        self.grids == grids
    }

    pub(crate) fn new(ctx: &CudaContext, grids: &[[u32; 3]]) -> Result<Self> {
        let offsets = super::vision_input::image_patch_offsets(grids)?;
        let patches = *offsets.last().unwrap() as usize;
        let factor = VISION_MERGE * VISION_MERGE;
        if patches == 0 || patches % factor != 0 {
            return Err(Error::Other(format!(
                "qwen35moe vision scratch needs a non-zero patch count divisible by {factor}, got {patches}"
            )));
        }
        let merged = patches / factor;
        let rows = [patches, VISION_HIDDEN];
        let heads = [patches, VISION_HEADS, VISION_HEAD_DIM];
        let bf16 = DType::BF16;
        Ok(Self {
            patches,
            grids: grids.to_vec(),
            offsets: upload_u32(ctx, &offsets)?,
            pixels: device_tensor(
                ctx,
                &[
                    patches,
                    VISION_CHANNELS
                        * VISION_TEMPORAL_PATCH_SIZE
                        * VISION_PATCH_SIZE
                        * VISION_PATCH_SIZE,
                ],
                bf16,
            )?,
            packed_pixels: device_tensor(
                ctx,
                &[
                    patches,
                    VISION_TEMPORAL_PATCH_SIZE * VISION_PATCH_SIZE * VISION_PATCH_SIZE * 4,
                ],
                bf16,
            )?,
            pos_embeds: device_tensor(ctx, &rows, bf16)?,
            pos_ids: device_tensor(ctx, &[patches, 2], DType::I32)?,
            x: [
                device_tensor(ctx, &rows, bf16)?,
                device_tensor(ctx, &rows, bf16)?,
            ],
            normed: device_tensor(ctx, &rows, bf16)?,
            qkv: device_tensor(ctx, &[patches, 3 * VISION_HIDDEN], bf16)?,
            query: device_tensor(ctx, &heads, bf16)?,
            key: device_tensor(ctx, &heads, bf16)?,
            value: device_tensor(ctx, &heads, bf16)?,
            // Dense Attention wants a leading batch axis; the rows are identical.
            attn: device_tensor(ctx, &[1, patches, VISION_HEADS, VISION_HEAD_DIM], bf16)?,
            projected: device_tensor(ctx, &rows, bf16)?,
            mlp_projected: device_tensor(ctx, &[patches, VISION_INTERMEDIATE], bf16)?,
            mlp_hidden: device_tensor(ctx, &[patches, VISION_INTERMEDIATE], bf16)?,
            mlp_out: device_tensor(ctx, &rows, bf16)?,
            merger_hidden: device_tensor(ctx, &[merged, VISION_HIDDEN * factor], bf16)?,
            merger_activated: device_tensor(ctx, &[merged, VISION_HIDDEN * factor], bf16)?,
            merger_out: device_tensor(ctx, &[merged, VISION_OUT_HIDDEN], bf16)?,
        })
    }
}

fn gemm_bias_into(
    ctx: &CudaContext,
    input: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    out: &mut Tensor,
) -> Result<()> {
    let input_dims = input.shape().dims();
    let weight_dims = weight.shape().dims();
    if input_dims.len() != 2 || weight_dims.len() != 2 || input_dims[1] != weight_dims[1] {
        return Err(Error::Other("qwen35moe vision GEMM shape mismatch".into()));
    }
    let mut gemm = ops::GemmArgs::new(input, weight, out).with_output_major_weight();
    gemm.policy.numerics = ops::GemmNumerics::PairwiseBf16Odd;
    // The numerical contract admits only the ordered-pair GPU kernel. Permit
    // its fixed configuration when a default CLI context has no tuned recipe;
    // strict recipe mode still rejects missing records in the backend.
    ops::gemm_bias(ctx, ops::GemmBiasArgs { gemm, bias })
}

fn layer_norm_into(
    ctx: &CudaContext,
    input: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
    out: &mut Tensor,
    eps: f32,
) -> Result<()> {
    let mut args = ops::NormArgs::layer(input, weight, bias, out, eps);
    args.numerics = ops::NormNumerics::Welford4F32;
    ops::norm::norm(ctx, args)
}

#[allow(clippy::too_many_arguments)]
fn bias_residual_layer_norm_into(
    ctx: &CudaContext,
    input: &Tensor,
    residual: &Tensor,
    weight: &Tensor,
    norm_bias: &Tensor,
    hidden: &mut Tensor,
    normalized: &mut Tensor,
    eps: f32,
) -> Result<()> {
    ops::copy_tensor(ctx, input, hidden)?;
    ops::add_into(ctx, residual, hidden)?;
    layer_norm_into(ctx, hidden, weight, norm_bias, normalized, eps)
}

fn forward_device(
    weights: &VisionWeights,
    ctx: &CudaContext,
    scratch: &VisionScratch,
    pixel_values: &Tensor,
    prepared: &PreparedVision,
) -> Result<Tensor> {
    let eps = 1e-6f32;
    let patches = prepared.total_patches;
    let factor = VISION_MERGE * VISION_MERGE;
    if scratch.patches != patches {
        return Err(Error::Other(format!(
            "qwen35moe vision scratch holds {} patches but this request has {patches}",
            scratch.patches
        )));
    }
    let Some(first) = weights.blocks.first() else {
        return Err(Error::Other("qwen35moe vision tower has no blocks".into()));
    };

    // Cloning a Tensor copies its shape and an Arc handle, not device memory,
    // so these alias the scratch buffers at stable addresses. Cloning avoids
    // borrowing several fields of `scratch` mutably at once.
    let mut current = scratch.x[0].clone();
    let mut spare = scratch.x[1].clone();
    let mut normed = scratch.normed.clone();
    let mut qkv = scratch.qkv.clone();
    let mut query = scratch.query.clone();
    let mut key = scratch.key.clone();
    let mut value = scratch.value.clone();
    let attn = scratch.attn.clone();
    let mut projected = scratch.projected.clone();
    let mut mlp_projected = scratch.mlp_projected.clone();
    let mut mlp_hidden = scratch.mlp_hidden.clone();
    let mut mlp_out = scratch.mlp_out.clone();

    let spatial = VISION_TEMPORAL_PATCH_SIZE * VISION_PATCH_SIZE * VISION_PATCH_SIZE;
    let input_3d = pixel_values.reshape(vec![patches, VISION_CHANNELS, spatial])?;
    let mut packed_3d = scratch.packed_pixels.reshape(vec![patches, spatial, 4])?;
    ops::transpose_pad_channels(ctx, &input_3d, &mut packed_3d)?;
    gemm_bias_into(
        ctx,
        &scratch.packed_pixels,
        &weights.patch_w,
        &weights.patch_b,
        &mut current,
    )?;
    dump_vision_stage("patch_projection", &current)?;
    dump_vision_stage("position_embeddings", &scratch.pos_embeds)?;
    ops::add_into(ctx, &scratch.pos_embeds, &current)?;
    dump_vision_stage("00_patch_position", &current)?;
    layer_norm_into(
        ctx,
        &current,
        &first.norm1_w,
        &first.norm1_b,
        &mut normed,
        eps,
    )?;

    for (block_index, block) in weights.blocks.iter().enumerate() {
        dump_vision_stage(&format!("block_{block_index:02}_norm1"), &normed)?;
        gemm_bias_into(ctx, &normed, &block.qkv_w, &block.qkv_b, &mut qkv)?;
        dump_vision_stage(&format!("block_{block_index:02}_qkv"), &qkv)?;
        ops::split_qkv_axial_rope(
            ctx,
            ops::AxialRopeArgs {
                qkv: &qkv,
                bias: None,
                positions: &scratch.pos_ids,
                query: &mut query,
                key: &mut key,
                value: &mut value,
                theta: 10_000.0,
                numerics: ops::AxialRopeNumerics::StagedFp32,
            },
        )?;
        dump_vision_stage(&format!("block_{block_index:02}_query"), &query)?;
        dump_vision_stage(&format!("block_{block_index:02}_key"), &key)?;
        dump_vision_stage(&format!("block_{block_index:02}_value"), &value)?;
        let mut attention_out = attn.reshape(vec![patches, VISION_HEADS, VISION_HEAD_DIM])?;
        let offsets = &prepared.offsets;
        let mut attention = ops::SegmentedAttentionArgs::new(
            &query,
            &key,
            &value,
            &mut attention_out,
            &scratch.offsets,
            offsets,
        );
        attention.numerics = ops::AttentionNumerics::StagedBf16;
        attention.scale = (VISION_HEAD_DIM as f64).powf(-0.5) as f32;
        ops::segmented_attention(ctx, attention)?;
        dump_vision_stage(&format!("block_{block_index:02}_attention"), &attention_out)?;
        gemm_bias_into(
            ctx,
            &attn.reshape(vec![patches, VISION_HIDDEN])?,
            &block.proj_w,
            &block.proj_b,
            &mut projected,
        )?;
        dump_vision_stage(&format!("block_{block_index:02}_projected"), &projected)?;
        // Writes `spare`, reads `current`: the residual add cannot target its
        // own input. `normed` is free here because the QKV GEMM above consumed
        // it, and it becomes this block's norm2 output.
        bias_residual_layer_norm_into(
            ctx,
            &projected,
            &current,
            &block.norm2_w,
            &block.norm2_b,
            &mut spare,
            &mut normed,
            eps,
        )?;
        std::mem::swap(&mut current, &mut spare);

        dump_vision_stage(&format!("block_{block_index:02}_norm2"), &normed)?;
        // HF Linear rounds the biased projection to BF16 before GELU.
        // GEMM_BIAS_GELU rounds the unbiased projection first; keep the model's
        // explicit boundary instead of changing the activation's input.
        gemm_bias_into(ctx, &normed, &block.fc1_w, &block.fc1_b, &mut mlp_projected)?;
        dump_vision_stage(&format!("block_{block_index:02}_fc1"), &mlp_projected)?;
        let mut gelu = ops::PointwiseArgs::new(
            ops::PointwiseSemantic::BiasActivation,
            &mlp_projected,
            &mut mlp_hidden,
        );
        gelu.activation = ops::PointwiseActivation::GeluTanhExp2;
        ops::pointwise(ctx, gelu)?;
        dump_vision_stage(&format!("block_{block_index:02}_gelu"), &mlp_hidden)?;
        gemm_bias_into(ctx, &mlp_hidden, &block.fc2_w, &block.fc2_b, &mut mlp_out)?;
        dump_vision_stage(&format!("block_{block_index:02}_fc2"), &mlp_out)?;
        // The fused norm for the next block, or the merger's own norm after
        // the last one, so no separate LayerNorm pass is needed.
        let (next_w, next_b) = weights
            .blocks
            .get(block_index + 1)
            .map(|next| (&next.norm1_w, &next.norm1_b))
            .unwrap_or((&weights.merger_norm_w, &weights.merger_norm_b));
        bias_residual_layer_norm_into(
            ctx,
            &mlp_out,
            &current,
            next_w,
            next_b,
            &mut spare,
            &mut normed,
            eps,
        )?;
        std::mem::swap(&mut current, &mut spare);
        dump_vision_stage(&format!("block_{block_index:02}_output"), &current)?;
    }

    let merged = normed.reshape(vec![patches / factor, VISION_HIDDEN * factor])?;
    dump_vision_stage("merger_norm", &merged)?;
    let mut merger_hidden = scratch.merger_hidden.clone();
    let mut merger_activated = scratch.merger_activated.clone();
    let mut merger_out = scratch.merger_out.clone();
    gemm_bias_into(
        ctx,
        &merged,
        &weights.merger_fc1_w,
        &weights.merger_fc1_b,
        &mut merger_hidden,
    )?;
    let mut activation = ops::PointwiseArgs::new(
        ops::PointwiseSemantic::BiasActivation,
        &merger_hidden,
        &mut merger_activated,
    );
    activation.activation = ops::PointwiseActivation::GeluExact;
    ops::pointwise(ctx, activation)?;
    gemm_bias_into(
        ctx,
        &merger_activated,
        &weights.merger_fc2_w,
        &weights.merger_fc2_b,
        &mut merger_out,
    )?;
    dump_vision_stage("merger_output", &merger_out)?;
    Ok(merger_out)
}

/// Replace image-token embedding rows with visual features. The replacement
/// is assignment (`masked_scatter` in Transformers), not residual addition.
pub(crate) fn embed_and_replace(
    ctx: &CudaContext,
    embedding_table: &Tensor,
    token_ids: &[u32],
    visual: &Tensor,
) -> Result<Tensor> {
    let ids: Vec<i32> = token_ids.iter().map(|&value| value as i32).collect();
    let ids = upload_i32(ctx, vec![token_ids.len()], &ids)?;
    let hidden = embedding_table.shape().dims()[1];
    let base = device_tensor(ctx, &[token_ids.len(), hidden], DType::BF16)?;
    ops::embedding_gather(ctx, embedding_table, &ids, &base)?;
    let mut ordinal = 0u32;
    let row_map: Vec<u32> = token_ids
        .iter()
        .map(|&token| {
            if token == IMAGE_TOKEN_ID {
                let current = ordinal;
                ordinal += 1;
                current
            } else {
                u32::MAX
            }
        })
        .collect();
    if ordinal as usize != visual.shape().dims()[0] {
        return Err(Error::Other(format!(
            "qwen35moe image token count {} does not match {} visual rows",
            ordinal,
            visual.shape().dims()[0]
        )));
    }
    let row_map = upload_u32(ctx, &row_map)?;
    let mut output = device_tensor(ctx, &[token_ids.len(), hidden], DType::BF16)?;
    let mut gather = ops::GatherArgs::new(ops::GatherSemantic::ReplaceRows, &base, &mut output);
    gather.replacement = Some(visual);
    gather.row_map = Some(&row_map);
    ops::gather(ctx, gather)?;
    Ok(output)
}

/// Validate the input representation before allocation or visual weight upload.
pub(crate) fn validate_input(pixels: &Tensor, grids: &[[u32; 3]]) -> Result<()> {
    prepare(pixels, grids)?;
    if !matches!(pixels.dtype(), DType::F32 | DType::BF16) {
        return Err(Error::Other("Qwen3.5 pixels must be F32 or BF16".into()));
    }
    if pixels.device() == Device::Cpu && pixels.to_f32_vec()?.iter().any(|value| !value.is_finite())
    {
        return Err(Error::Other("Qwen3.5 pixels must be finite".into()));
    }
    Ok(())
}
