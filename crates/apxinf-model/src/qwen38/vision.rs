//! Native BF16 Qwen3.5 vision tower and image-embedding injection.
//!
//! Processor output enters as canonical `[patches, 1536]` F32/BF16 plus
//! `grid_thw`. All per-request tower computation remains on CUDA. The learned
//! 48x48 position table is interpolated on the host once per distinct grid and
//! retained in a small device cache, matching the maintained Qwen-Drive path.

use std::sync::{Mutex, OnceLock};

use apxinf_core::{DType, Device, Error, Result, Shape, Tensor};

use super::backend::{
    legacy_kernels as kernels, legacy_transfers as transfers, LegacyBackend, LegacyBuffer,
};
use super::config::*;
use super::vision_weights::VisionWeights;

use kernels::{attention, elementwise, embedding, fused, gemm, linear_attention as la, norm};

#[derive(Default)]
pub(crate) struct VisionState {
    pos_table_host: OnceLock<Vec<f32>>,
    pos_embed_cache: Mutex<Vec<(Vec<[u32; 3]>, Tensor)>>,
}

struct PreparedVision {
    offsets: Vec<u32>,
    offsets_dev: LegacyBuffer,
    pos_embeds: Tensor,
    pos_ids_dev: LegacyBuffer,
    max_tokens: usize,
    total_patches: usize,
    segments: usize,
}

fn upload_u32(backend: &LegacyBackend, values: &[u32]) -> Result<LegacyBuffer> {
    let bytes: Vec<u8> = values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect();
    let buffer =
        LegacyBuffer::alloc(bytes.len().max(1), backend.device_id()).map_err(Error::Cuda)?;
    buffer.copy_from_host(&bytes).map_err(Error::Cuda)?;
    Ok(buffer)
}

fn device_tensor(backend: &LegacyBackend, dims: &[usize], dtype: DType) -> Result<Tensor> {
    let bytes = dims
        .iter()
        .product::<usize>()
        .checked_mul(dtype.size_in_bytes())
        .ok_or_else(|| Error::Other("qwen38 vision tensor size overflow".into()))?;
    kernels::scratch_buffer(backend.context(), bytes.max(1))?
        .as_tensor(Shape::new(dims.to_vec()), dtype)
        .map_err(Error::Cuda)
}

pub(crate) fn upload_pixels(backend: &LegacyBackend, pixels: &Tensor) -> Result<Tensor> {
    let ctx = backend.context();
    let on_device = if pixels.device() != Device::Cuda(ctx.device_id()) {
        transfers::to_cuda(pixels, ctx.device_id())?
    } else {
        pixels.clone()
    };
    match on_device.dtype() {
        DType::BF16 => Ok(on_device),
        DType::F32 => {
            let dims = on_device.shape().dims().to_vec();
            let out = device_tensor(backend, &dims, DType::BF16)?;
            la::cast_f32_to_bf16(ctx, &on_device, &out)?;
            Ok(out)
        }
        dtype => Err(Error::Other(format!(
            "qwen38 image pixel_values must be f32 or bf16, got {dtype}"
        ))),
    }
}

/// Run the complete vision tower over one or more concatenated image patch
/// streams. Attention is segmented at image boundaries.
pub(crate) fn forward(
    weights: &VisionWeights,
    state: &VisionState,
    backend: &LegacyBackend,
    pixel_values: &Tensor,
    grid_thw: &[[u32; 3]],
) -> Result<Tensor> {
    let prepared = prepare(weights, state, backend, pixel_values, grid_thw)?;
    let output = forward_device(weights, backend, pixel_values, &prepared)?;
    if let Ok(path) = std::env::var("APXINF_QWEN38_VISION_DUMP") {
        let values = transfers::to_cpu(&output)?.to_f32_vec()?;
        let bytes: Vec<u8> = values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect();
        std::fs::write(&path, bytes)
            .map_err(|error| Error::Other(format!("write qwen38 vision dump {path}: {error}")))?;
        eprintln!(
            "qwen38 vision dump: {} rows x {} columns -> {path}",
            output.shape().dims()[0],
            output.shape().dims()[1]
        );
    }
    Ok(output)
}

fn prepare(
    weights: &VisionWeights,
    state: &VisionState,
    backend: &LegacyBackend,
    pixel_values: &Tensor,
    grid_thw: &[[u32; 3]],
) -> Result<PreparedVision> {
    let patch_width =
        VISION_CHANNELS * VISION_TEMPORAL_PATCH_SIZE * VISION_PATCH_SIZE * VISION_PATCH_SIZE;
    let dims = pixel_values.shape().dims();
    if grid_thw.is_empty() || dims.len() != 2 || dims[1] != patch_width {
        return Err(Error::Other(format!(
            "qwen38 image pixel_values must be [patches, {patch_width}], got {dims:?}"
        )));
    }

    let mut offsets = vec![0u32];
    let mut max_tokens = 0usize;
    for &[t, h, w] in grid_thw {
        let (t, h, w) = (t as usize, h as usize, w as usize);
        if t == 0 || h == 0 || w == 0 || h % VISION_MERGE != 0 || w % VISION_MERGE != 0 {
            return Err(Error::Other(format!(
                "qwen38 image grid [{t}, {h}, {w}] is invalid for merge {VISION_MERGE}"
            )));
        }
        let patches = t
            .checked_mul(h)
            .and_then(|value| value.checked_mul(w))
            .ok_or_else(|| Error::Other("qwen38 image grid is too large".into()))?;
        max_tokens = max_tokens.max(patches);
        let next = offsets[offsets.len() - 1]
            .checked_add(
                u32::try_from(patches)
                    .map_err(|_| Error::Other("qwen38 image patch count exceeds u32".into()))?,
            )
            .ok_or_else(|| Error::Other("qwen38 image patch count overflow".into()))?;
        offsets.push(next);
    }
    let total_patches = dims[0];
    if offsets[offsets.len() - 1] as usize != total_patches {
        return Err(Error::Other(format!(
            "qwen38 image grids cover {} patches but pixel_values has {total_patches}",
            offsets[offsets.len() - 1]
        )));
    }

    let offsets_dev = upload_u32(backend, &offsets)?;
    let pos_embeds = compute_pos_embeds(weights, state, backend, grid_thw)?;
    let pos_ids_dev = upload_u32(backend, &vision_position_ids(grid_thw))?;
    Ok(PreparedVision {
        offsets,
        offsets_dev,
        pos_embeds,
        pos_ids_dev,
        max_tokens,
        total_patches,
        segments: grid_thw.len(),
    })
}

fn forward_device(
    weights: &VisionWeights,
    backend: &LegacyBackend,
    pixel_values: &Tensor,
    prepared: &PreparedVision,
) -> Result<Tensor> {
    let ctx = backend.context();
    let eps = 1e-6f32;
    let mut x = gemm::bf16(ctx, pixel_values, &weights.patch_w)?;
    x = elementwise::bias_bf16(ctx, &x, Some(&weights.patch_b))?;
    x = elementwise::add(ctx, &x, &prepared.pos_embeds)?;

    let mut normalized = None;
    for (block_index, block) in weights.blocks.iter().enumerate() {
        let normed = match normalized.take() {
            Some(value) => value,
            None => norm::layer_bf16(ctx, &x, &block.norm1_w, &block.norm1_b, eps)?,
        };
        let qkv = gemm::bf16_bias(ctx, &normed, &block.qkv_w, &block.qkv_b)?;
        let qkv = attention::split_vision_qkv_rope_bf16(
            ctx,
            &qkv,
            None,
            &prepared.pos_ids_dev,
            VISION_HEADS,
            VISION_HEAD_DIM,
            10_000.0,
        )?;
        let attn = attention::segmented_mha_bf16(
            ctx,
            &qkv.q,
            &qkv.k,
            &qkv.v,
            &prepared.offsets_dev,
            &prepared.offsets,
            prepared.segments,
            prepared.max_tokens,
        )?
        .reshape(vec![prepared.total_patches, VISION_HIDDEN])?;
        let projected = gemm::bf16_bias(ctx, &attn, &block.proj_w, &block.proj_b)?;
        let residual_norm = fused::bias_residual_layer_bf16(
            ctx,
            &projected,
            None,
            &x,
            &block.norm2_w,
            &block.norm2_b,
            eps,
        )?;
        x = residual_norm.hidden;

        let hidden = if ctx.caps().sm == 110 {
            gemm::bf16_bias_gelu_tanh(ctx, &residual_norm.normalized, &block.fc1_w, &block.fc1_b)?
        } else {
            let value =
                gemm::bf16_bias(ctx, &residual_norm.normalized, &block.fc1_w, &block.fc1_b)?;
            kernels::activation::gelu_tanh(ctx, &value)?
        };
        let mlp = gemm::bf16_bias(ctx, &hidden, &block.fc2_w, &block.fc2_b)?;
        let (next_w, next_b) = weights
            .blocks
            .get(block_index + 1)
            .map(|next| (&next.norm1_w, &next.norm1_b))
            .unwrap_or((&weights.merger_norm_w, &weights.merger_norm_b));
        let residual_norm =
            fused::bias_residual_layer_bf16(ctx, &mlp, None, &x, next_w, next_b, eps)?;
        x = residual_norm.hidden;
        normalized = Some(residual_norm.normalized);
    }

    let normalized = match normalized {
        Some(value) => value,
        None => norm::layer_bf16(ctx, &x, &weights.merger_norm_w, &weights.merger_norm_b, eps)?,
    };
    let rows = normalized.shape().dims()[0];
    let factor = VISION_MERGE * VISION_MERGE;
    if rows % factor != 0 {
        return Err(Error::Other(format!(
            "qwen38 image merger rows {rows} do not divide by {factor}"
        )));
    }
    let merged = normalized.reshape(vec![rows / factor, VISION_HIDDEN * factor])?;
    let hidden = gemm::bf16_bias(ctx, &merged, &weights.merger_fc1_w, &weights.merger_fc1_b)?;
    let hidden = la::gelu_exact(ctx, &hidden)?;
    gemm::bf16_bias(ctx, &hidden, &weights.merger_fc2_w, &weights.merger_fc2_b)
}

/// Replace image-token embedding rows with visual features. The replacement
/// is assignment (`masked_scatter` in Transformers), not residual addition.
pub(crate) fn embed_and_replace(
    backend: &LegacyBackend,
    embedding_table: &Tensor,
    token_ids: &[u32],
    visual: &Tensor,
) -> Result<Tensor> {
    let ctx = backend.context();
    let ids = upload_u32(backend, token_ids)?;
    let base = embedding::lookup(ctx, embedding_table, &ids, token_ids.len())?;
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
            "qwen38 image token count {} does not match {} visual rows",
            ordinal,
            visual.shape().dims()[0]
        )));
    }
    let row_map = upload_u32(backend, &row_map)?;
    elementwise::replace_rows_bf16(ctx, &base, visual, &row_map)
}

fn compute_pos_embeds(
    weights: &VisionWeights,
    state: &VisionState,
    backend: &LegacyBackend,
    grid_thw: &[[u32; 3]],
) -> Result<Tensor> {
    if let Ok(cache) = state.pos_embed_cache.lock() {
        if let Some((_, tensor)) = cache.iter().find(|(key, _)| key.as_slice() == grid_thw) {
            return Ok(tensor.clone());
        }
    }
    if state.pos_table_host.get().is_none() {
        let table = transfers::to_cpu(&weights.pos_embed)?.to_f32_vec()?;
        let _ = state.pos_table_host.set(table);
    }
    let table = state
        .pos_table_host
        .get()
        .ok_or_else(|| Error::Other("qwen38 image position table unavailable".into()))?;
    let grid_side = (VISION_POSITIONS as f64).sqrt().round() as usize;
    if table.len() != grid_side * grid_side * VISION_HIDDEN {
        return Err(Error::Other(
            "qwen38 image position table shape mismatch".into(),
        ));
    }

    let total: usize = grid_thw
        .iter()
        .map(|grid| grid[0] as usize * grid[1] as usize * grid[2] as usize)
        .sum();
    let mut output = vec![0.0f32; total * VISION_HIDDEN];
    let mut destination_token = 0usize;
    for &[t, h, w] in grid_thw {
        let (t, h, w) = (t as usize, h as usize, w as usize);
        for _ in 0..t {
            for merged_h in 0..h / VISION_MERGE {
                for merged_w in 0..w / VISION_MERGE {
                    for inner_h in 0..VISION_MERGE {
                        for inner_w in 0..VISION_MERGE {
                            let row = merged_h * VISION_MERGE + inner_h;
                            let column = merged_w * VISION_MERGE + inner_w;
                            let row_position =
                                row as f32 * (grid_side - 1) as f32 / (h - 1).max(1) as f32;
                            let column_position =
                                column as f32 * (grid_side - 1) as f32 / (w - 1).max(1) as f32;
                            let row0 = row_position.floor() as usize;
                            let row1 = (row0 + 1).min(grid_side - 1);
                            let column0 = column_position.floor() as usize;
                            let column1 = (column0 + 1).min(grid_side - 1);
                            let row_weight = row_position - row0 as f32;
                            let column_weight = column_position - column0 as f32;
                            let destination = destination_token * VISION_HIDDEN;
                            for channel in 0..VISION_HIDDEN {
                                let value00 =
                                    table[(row0 * grid_side + column0) * VISION_HIDDEN + channel];
                                let value01 =
                                    table[(row0 * grid_side + column1) * VISION_HIDDEN + channel];
                                let value10 =
                                    table[(row1 * grid_side + column0) * VISION_HIDDEN + channel];
                                let value11 =
                                    table[(row1 * grid_side + column1) * VISION_HIDDEN + channel];
                                output[destination + channel] =
                                    (1.0 - row_weight) * (1.0 - column_weight) * value00
                                        + (1.0 - row_weight) * column_weight * value01
                                        + row_weight * (1.0 - column_weight) * value10
                                        + row_weight * column_weight * value11;
                            }
                            destination_token += 1;
                        }
                    }
                }
            }
        }
    }
    let rounded: Vec<half::bf16> = output
        .iter()
        .map(|&value| half::bf16::from_f32(value))
        .collect();
    let host = Tensor::from_bf16(vec![total, VISION_HIDDEN], &rounded)?;
    let uploaded = transfers::to_cuda(&host, backend.device_id())?;
    if let Ok(mut cache) = state.pos_embed_cache.lock() {
        const MAX_GRIDS: usize = 4;
        if cache.len() >= MAX_GRIDS {
            cache.remove(0);
        }
        cache.push((grid_thw.to_vec(), uploaded.clone()));
    }
    Ok(uploaded)
}

fn vision_position_ids(grid_thw: &[[u32; 3]]) -> Vec<u32> {
    let mut ids = Vec::new();
    for &[t, h, w] in grid_thw {
        for _ in 0..t {
            for merged_h in 0..h as usize / VISION_MERGE {
                for merged_w in 0..w as usize / VISION_MERGE {
                    for inner_h in 0..VISION_MERGE {
                        for inner_w in 0..VISION_MERGE {
                            ids.push((merged_h * VISION_MERGE + inner_h) as u32);
                            ids.push((merged_w * VISION_MERGE + inner_w) as u32);
                        }
                    }
                }
            }
        }
    }
    ids
}

/// Qwen3.5 three-axis language positions for an image-only multimodal prompt.
pub(crate) fn rope_positions(token_ids: &[u32], grid_thw: &[[u32; 3]]) -> Result<Vec<[u32; 3]>> {
    let mut positions = Vec::with_capacity(token_ids.len());
    let mut current = 0u32;
    let mut grid_index = 0usize;
    let mut index = 0usize;
    while index < token_ids.len() {
        if token_ids[index] != IMAGE_TOKEN_ID {
            positions.push([current; 3]);
            current += 1;
            index += 1;
            continue;
        }
        let &[t, h, w] = grid_thw.get(grid_index).ok_or_else(|| {
            Error::Other("qwen38 image prompt has more image groups than grids".into())
        })?;
        grid_index += 1;
        if h % VISION_MERGE as u32 != 0 || w % VISION_MERGE as u32 != 0 {
            return Err(Error::Other(
                "qwen38 image grid is not merge-aligned".into(),
            ));
        }
        let merged_h = h / VISION_MERGE as u32;
        let merged_w = w / VISION_MERGE as u32;
        let image_tokens = t as usize * merged_h as usize * merged_w as usize;
        if index + image_tokens > token_ids.len()
            || token_ids[index..index + image_tokens]
                .iter()
                .any(|&token| token != IMAGE_TOKEN_ID)
        {
            return Err(Error::Other(format!(
                "qwen38 image placeholder group at token {index} does not contain {image_tokens} tokens"
            )));
        }
        for temporal in 0..t {
            for row in 0..merged_h {
                for column in 0..merged_w {
                    positions.push([current + temporal, current + row, current + column]);
                }
            }
        }
        current += merged_h.max(merged_w);
        index += image_tokens;
    }
    if grid_index != grid_thw.len() {
        return Err(Error::Other(format!(
            "qwen38 image prompt used {grid_index} grids but received {}",
            grid_thw.len()
        )));
    }
    Ok(positions)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn image_positions_match_qwen35_group_semantics() {
        let tokens = [
            7,
            IMAGE_TOKEN_ID,
            IMAGE_TOKEN_ID,
            IMAGE_TOKEN_ID,
            IMAGE_TOKEN_ID,
            8,
        ];
        let positions = rope_positions(&tokens, &[[1, 4, 4]]).unwrap();
        assert_eq!(
            positions,
            vec![
                [0, 0, 0],
                [1, 1, 1],
                [1, 1, 2],
                [1, 2, 1],
                [1, 2, 2],
                [3, 3, 3],
            ]
        );
    }

    #[test]
    fn image_positions_reject_placeholder_mismatch() {
        let error = rope_positions(&[IMAGE_TOKEN_ID], &[[1, 4, 4]])
            .err()
            .expect("one placeholder cannot represent four merged image tokens");
        assert!(error.to_string().contains("does not contain 4"));
    }
}
