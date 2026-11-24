//! BF16 Qwen3.5 vision-tower weights from the same 35B-A3B checkpoint.
//!
//! The language stack is quantized, but every `model.visual.*` tensor in the
//! shipped checkpoint is BF16. Checkpoint output-major linear layouts remain unchanged during upload.

use std::collections::HashMap;

use apxinf_core::{DType, Error, Result, Tensor};
use apxinf_cuda_new::{transfers, CudaContext};

use super::vision_config::*;

pub(crate) struct VisionBlockWeights {
    pub(crate) norm1_w: Tensor,
    pub(crate) norm1_b: Tensor,
    pub(crate) qkv_w: Tensor,
    pub(crate) qkv_b: Tensor,
    pub(crate) proj_w: Tensor,
    pub(crate) proj_b: Tensor,
    pub(crate) norm2_w: Tensor,
    pub(crate) norm2_b: Tensor,
    pub(crate) fc1_w: Tensor,
    pub(crate) fc1_b: Tensor,
    pub(crate) fc2_w: Tensor,
    pub(crate) fc2_b: Tensor,
}

pub(crate) struct VisionWeights {
    /// Conv3d in spatial-major order with channels padded to four, `[hidden, 2048]`.
    pub(crate) patch_w: Tensor,
    pub(crate) patch_b: Tensor,
    pub(crate) pos_embed: Tensor,
    pub(crate) blocks: Vec<VisionBlockWeights>,
    pub(crate) merger_norm_w: Tensor,
    pub(crate) merger_norm_b: Tensor,
    pub(crate) merger_fc1_w: Tensor,
    pub(crate) merger_fc1_b: Tensor,
    pub(crate) merger_fc2_w: Tensor,
    pub(crate) merger_fc2_b: Tensor,
}

fn get<'a>(tensors: &'a HashMap<String, Tensor>, name: &str) -> Result<&'a Tensor> {
    tensors
        .get(name)
        .ok_or_else(|| Error::Other(format!("qwen35moe vision weights: missing {name}")))
}

fn upload(ctx: &CudaContext, tensor: &Tensor) -> Result<Tensor> {
    transfers::to_cuda(tensor, ctx.device_id())
}

fn upload_named(
    ctx: &CudaContext,
    tensors: &HashMap<String, Tensor>,
    name: &str,
) -> Result<Tensor> {
    upload(ctx, get(tensors, name)?)
}

impl VisionWeights {
    pub(crate) fn load(tensors: &HashMap<String, Tensor>, ctx: &CudaContext) -> Result<Self> {
        validate_weights(tensors)?;
        let patch = get(tensors, "model.visual.patch_embed.proj.weight")?;
        let patch_dims = patch.shape().dims();
        let patch_width =
            VISION_CHANNELS * VISION_TEMPORAL_PATCH_SIZE * VISION_PATCH_SIZE * VISION_PATCH_SIZE;
        if patch_dims.len() != 5
            || patch_dims[0] != VISION_HIDDEN
            || patch_dims.iter().product::<usize>() != VISION_HIDDEN * patch_width
        {
            return Err(Error::Other(format!(
                "qwen35moe vision patch weight has shape {patch_dims:?}"
            )));
        }
        if patch.dtype() != DType::BF16 {
            return Err(Error::Other(format!(
                "qwen35moe vision patch weight must be BF16, got {}",
                patch.dtype()
            )));
        }
        let spatial = patch_width / VISION_CHANNELS;
        let original = patch.as_bf16()?;
        let mut packed = vec![half::bf16::ZERO; VISION_HIDDEN * spatial * 4];
        for row in 0..VISION_HIDDEN {
            for position in 0..spatial {
                for channel in 0..VISION_CHANNELS {
                    packed[(row * spatial + position) * 4 + channel] =
                        original[(row * VISION_CHANNELS + channel) * spatial + position];
                }
            }
        }
        let patch_w = upload(
            ctx,
            &Tensor::from_bf16(vec![VISION_HIDDEN, spatial * 4], &packed)?,
        )?;

        let mut blocks = Vec::with_capacity(VISION_DEPTH);
        for index in 0..VISION_DEPTH {
            let prefix = format!("model.visual.blocks.{index}");
            blocks.push(VisionBlockWeights {
                norm1_w: upload_named(ctx, tensors, &format!("{prefix}.norm1.weight"))?,
                norm1_b: upload_named(ctx, tensors, &format!("{prefix}.norm1.bias"))?,
                qkv_w: upload_named(ctx, tensors, &format!("{prefix}.attn.qkv.weight"))?,
                qkv_b: upload_named(ctx, tensors, &format!("{prefix}.attn.qkv.bias"))?,
                proj_w: upload_named(ctx, tensors, &format!("{prefix}.attn.proj.weight"))?,
                proj_b: upload_named(ctx, tensors, &format!("{prefix}.attn.proj.bias"))?,
                norm2_w: upload_named(ctx, tensors, &format!("{prefix}.norm2.weight"))?,
                norm2_b: upload_named(ctx, tensors, &format!("{prefix}.norm2.bias"))?,
                fc1_w: upload_named(ctx, tensors, &format!("{prefix}.mlp.linear_fc1.weight"))?,
                fc1_b: upload_named(ctx, tensors, &format!("{prefix}.mlp.linear_fc1.bias"))?,
                fc2_w: upload_named(ctx, tensors, &format!("{prefix}.mlp.linear_fc2.weight"))?,
                fc2_b: upload_named(ctx, tensors, &format!("{prefix}.mlp.linear_fc2.bias"))?,
            });
        }

        let merged = VISION_HIDDEN * VISION_MERGE * VISION_MERGE;
        let merger_fc1 = get(tensors, "model.visual.merger.linear_fc1.weight")?;
        let merger_fc2 = get(tensors, "model.visual.merger.linear_fc2.weight")?;
        if merger_fc1.shape().dims() != [merged, merged]
            || merger_fc2.shape().dims() != [VISION_OUT_HIDDEN, merged]
        {
            return Err(Error::Other(format!(
                "qwen35moe vision merger shape mismatch: {:?}, {:?}",
                merger_fc1.shape().dims(),
                merger_fc2.shape().dims()
            )));
        }

        Ok(Self {
            patch_w,
            patch_b: upload_named(ctx, tensors, "model.visual.patch_embed.proj.bias")?,
            pos_embed: upload_named(ctx, tensors, "model.visual.pos_embed.weight")?,
            blocks,
            merger_norm_w: upload_named(ctx, tensors, "model.visual.merger.norm.weight")?,
            merger_norm_b: upload_named(ctx, tensors, "model.visual.merger.norm.bias")?,
            merger_fc1_w: upload(ctx, merger_fc1)?,
            merger_fc1_b: upload_named(ctx, tensors, "model.visual.merger.linear_fc1.bias")?,
            merger_fc2_w: upload(ctx, merger_fc2)?,
            merger_fc2_b: upload_named(ctx, tensors, "model.visual.merger.linear_fc2.bias")?,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_vision_geometry_matches_the_35b_checkpoint() {
        assert_eq!(VISION_HEAD_DIM, 72);
        assert_eq!(VISION_HIDDEN * VISION_MERGE * VISION_MERGE, 4608);
        assert_eq!(VISION_CHANNELS * VISION_TEMPORAL_PATCH_SIZE * 16 * 16, 1536);
    }
}

/// Kept only until the first image: text serving does not upload a visual tower.
pub(super) enum VisionSource {
    Path(std::path::PathBuf),
    Host(HashMap<String, Tensor>),
}
impl VisionSource {
    pub(super) fn load(&self, ctx: &CudaContext) -> Result<VisionWeights> {
        match self {
            Self::Host(tensors) => VisionWeights::load(tensors, ctx),
            Self::Path(path) => {
                let mut reader =
                    apxinf_loader::safetensors::NativeSafetensorsReader::open_path(path)
                        .map_err(Error::Other)?;
                let names: Vec<_> = reader
                    .tensor_names()
                    .filter(|name| name.starts_with("model.visual."))
                    .map(str::to_owned)
                    .collect();
                let mut tensors = HashMap::new();
                for name in names {
                    tensors.insert(
                        name.clone(),
                        reader.load_tensor(&name).map_err(Error::Other)?,
                    );
                }
                VisionWeights::load(&tensors, ctx)
            }
        }
    }
}

fn validate_weights(tensors: &HashMap<String, Tensor>) -> Result<()> {
    let h = VISION_HIDDEN;
    let merged = h * VISION_MERGE * VISION_MERGE;
    let mut specs = vec![
        (
            "model.visual.patch_embed.proj.weight".to_string(),
            vec![h, 3, 2, 16, 16],
        ),
        ("model.visual.patch_embed.proj.bias".to_string(), vec![h]),
        (
            "model.visual.pos_embed.weight".to_string(),
            vec![VISION_POSITIONS, h],
        ),
        ("model.visual.merger.norm.weight".to_string(), vec![h]),
        ("model.visual.merger.norm.bias".to_string(), vec![h]),
        (
            "model.visual.merger.linear_fc1.weight".to_string(),
            vec![merged, merged],
        ),
        (
            "model.visual.merger.linear_fc1.bias".to_string(),
            vec![merged],
        ),
        (
            "model.visual.merger.linear_fc2.weight".to_string(),
            vec![VISION_OUT_HIDDEN, merged],
        ),
        (
            "model.visual.merger.linear_fc2.bias".to_string(),
            vec![VISION_OUT_HIDDEN],
        ),
    ];
    for index in 0..VISION_DEPTH {
        for (name, shape) in [
            ("norm1.weight", vec![h]),
            ("norm1.bias", vec![h]),
            ("norm2.weight", vec![h]),
            ("norm2.bias", vec![h]),
            ("attn.qkv.weight", vec![3 * h, h]),
            ("attn.qkv.bias", vec![3 * h]),
            ("attn.proj.weight", vec![h, h]),
            ("attn.proj.bias", vec![h]),
            ("mlp.linear_fc1.weight", vec![VISION_INTERMEDIATE, h]),
            ("mlp.linear_fc1.bias", vec![VISION_INTERMEDIATE]),
            ("mlp.linear_fc2.weight", vec![h, VISION_INTERMEDIATE]),
            ("mlp.linear_fc2.bias", vec![h]),
        ] {
            specs.push((format!("model.visual.blocks.{index}.{name}"), shape));
        }
    }
    for (name, shape) in specs {
        let tensor = get(tensors, &name)?;
        if tensor.dtype() != DType::BF16 || tensor.shape().dims() != shape.as_slice() {
            return Err(Error::Other(format!(
                "Qwen3.5 vision weight {name}: expected BF16 {shape:?}, got {} {:?}",
                tensor.dtype(),
                tensor.shape().dims()
            )));
        }
    }
    Ok(())
}
