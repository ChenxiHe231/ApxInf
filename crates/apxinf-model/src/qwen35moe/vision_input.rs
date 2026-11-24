//! Checkpoint and position contracts for ordered still images. No device work here.
use super::vision_config::*;
use apxinf_core::{Error, Result};
use serde::Deserialize;

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct VisionConfig {
    pub depth: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_heads: usize,
    pub in_channels: usize,
    pub out_hidden_size: usize,
    pub patch_size: usize,
    pub spatial_merge_size: usize,
    pub temporal_patch_size: usize,
    pub num_position_embeddings: usize,
    pub hidden_act: String,
    pub dtype: String,
    pub model_type: String,
    #[serde(default)]
    pub deepstack_visual_indexes: Vec<usize>,
}
impl VisionConfig {
    pub fn validate(&self) -> Result<()> {
        if self.depth != VISION_DEPTH
            || self.hidden_size != VISION_HIDDEN
            || self.intermediate_size != VISION_INTERMEDIATE
            || self.num_heads != VISION_HEADS
            || self.in_channels != VISION_CHANNELS
            || self.out_hidden_size != VISION_OUT_HIDDEN
            || self.patch_size != VISION_PATCH_SIZE
            || self.spatial_merge_size != VISION_MERGE
            || self.temporal_patch_size != VISION_TEMPORAL_PATCH_SIZE
            || self.num_position_embeddings != VISION_POSITIONS
            || self.hidden_act != "gelu_pytorch_tanh"
            || self.dtype != "bfloat16"
            || self.model_type != "qwen3_5_moe_vision"
            || !self.deepstack_visual_indexes.is_empty()
        {
            return Err(invalid("unsupported Qwen3.5 MoE vision configuration"));
        }
        Ok(())
    }
}

/// Packed image boundaries. Each still image owns an independent attention span.
pub(crate) fn image_patch_offsets(grids: &[[u32; 3]]) -> Result<Vec<u32>> {
    if grids.is_empty() {
        return Err(invalid("image input requires image grids"));
    }
    let mut offsets = Vec::with_capacity(grids.len() + 1);
    offsets.push(0u32);
    for &[t, h, w] in grids {
        if t != 1 || h == 0 || w == 0 || h % 2 != 0 || w % 2 != 0 {
            return Err(invalid(
                "image grids must be still, nonzero and merge-aligned",
            ));
        }
        let patches = h
            .checked_mul(w)
            .ok_or_else(|| invalid("image grid overflow"))?;
        let next = offsets
            .last()
            .unwrap()
            .checked_add(patches)
            .filter(|&value| value <= i32::MAX as u32)
            .ok_or_else(|| invalid("image patch count exceeds native bounds"))?;
        offsets.push(next);
    }
    Ok(offsets)
}

/// Three-axis prompt positions and the compressed offset used for continuation.
#[derive(Clone, Debug)]
pub(crate) struct ImagePositions {
    pub prompt: Vec<[u32; 3]>,
    pub delta: i64,
}
impl ImagePositions {
    pub fn new(tokens: &[u32], grids: &[[u32; 3]]) -> Result<Self> {
        image_patch_offsets(grids)?;
        let mut positions = Vec::with_capacity(tokens.len());
        let mut cursor = 0;
        let mut current = 0u32;
        for &[t, h, w] in grids {
            if t != 1 || h == 0 || w == 0 || h % 2 != 0 || w % 2 != 0 {
                return Err(invalid(
                    "Qwen3.5 requires still images with nonzero merge-aligned grids",
                ));
            }
            let count = (h as usize / 2)
                .checked_mul(w as usize / 2)
                .ok_or_else(|| invalid("image grid overflow"))?;
            let start = tokens[cursor..]
                .iter()
                .position(|&id| id == IMAGE_TOKEN_ID)
                .map(|index| cursor + index)
                .ok_or_else(|| invalid("image prompt is missing image placeholders"))?;
            let end = start
                .checked_add(count)
                .ok_or_else(|| invalid("image placeholder overflow"))?;
            if end >= tokens.len()
                || start == 0
                || tokens[start - 1] != 248053
                || tokens[end] != 248054
                || tokens[start..end].iter().any(|&id| id != IMAGE_TOKEN_ID)
            {
                return Err(invalid(
                    "image placeholders must match each grid between vision_start and vision_end",
                ));
            }
            for _ in cursor..start {
                positions.push([current; 3]);
                current = current
                    .checked_add(1)
                    .ok_or_else(|| invalid("image position overflow"))?;
            }
            let base = current;
            for row in 0..h / 2 {
                for col in 0..w / 2 {
                    positions.push([
                        base,
                        base.checked_add(row)
                            .ok_or_else(|| invalid("image position overflow"))?,
                        base.checked_add(col)
                            .ok_or_else(|| invalid("image position overflow"))?,
                    ]);
                }
            }
            current = base
                .checked_add((h / 2).max(w / 2))
                .ok_or_else(|| invalid("image position overflow"))?;
            cursor = end;
        }
        if tokens[cursor..].contains(&IMAGE_TOKEN_ID) {
            return Err(invalid(
                "image prompt has more placeholders than image grids",
            ));
        }
        for _ in cursor..tokens.len() {
            positions.push([current; 3]);
            current = current
                .checked_add(1)
                .ok_or_else(|| invalid("image position overflow"))?;
        }
        Ok(Self {
            delta: current as i64 - tokens.len() as i64,
            prompt: positions,
        })
    }
    /// Token-major positions for shifted MTP teacher forcing and draft rows.
    pub fn token_major_values(&self, start: usize, tokens: usize) -> Result<Vec<i32>> {
        let values = self.values(start, tokens)?;
        Ok((0..tokens)
            .flat_map(|i| [values[i], values[tokens + i], values[2 * tokens + i]])
            .collect())
    }

    /// Axis-major [3, tokens] used by the target attention kernels.
    pub fn values(&self, start: usize, tokens: usize) -> Result<Vec<i32>> {
        let mut values = Vec::with_capacity(3 * tokens);
        for axis in 0..3 {
            for index in start
                ..start
                    .checked_add(tokens)
                    .ok_or_else(|| invalid("image position overflow"))?
            {
                let value = self
                    .prompt
                    .get(index)
                    .map(|p| p[axis] as i64)
                    .unwrap_or(index as i64 + self.delta);
                values.push(
                    i32::try_from(value)
                        .ok()
                        .filter(|v| *v >= 0)
                        .ok_or_else(|| invalid("image position exceeds i32"))?,
                );
            }
        }
        Ok(values)
    }
}
fn invalid(message: &str) -> Error {
    Error::Other(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn real_processor_prompt_matches_transformers_position_reference() {
        #[derive(Deserialize)]
        struct Reference {
            tokens: Vec<u32>,
            grids: Vec<[u32; 3]>,
            positions: Vec<[u32; 3]>,
            delta: i64,
        }
        for source in [
            include_str!("../../tests/fixtures/qwen35-image-positions.json"),
            include_str!("../../tests/fixtures/qwen35-multi-image-positions.json"),
        ] {
            let reference: Reference = serde_json::from_str(source).unwrap();
            let positions = ImagePositions::new(&reference.tokens, &reference.grids).unwrap();
            assert_eq!(positions.prompt, reference.positions);
            assert_eq!(positions.delta, reference.delta);
            let expected = reference.positions[1..]
                .iter()
                .flat_map(|p| p.iter().map(|&v| v as i32))
                .collect::<Vec<_>>();
            assert_eq!(
                positions
                    .token_major_values(1, reference.tokens.len() - 1)
                    .unwrap(),
                expected
            );
            let next = (reference.tokens.len() as i64 + reference.delta) as i32;
            assert_eq!(
                positions
                    .token_major_values(reference.tokens.len(), 3)
                    .unwrap(),
                [
                    next,
                    next,
                    next,
                    next + 1,
                    next + 1,
                    next + 1,
                    next + 2,
                    next + 2,
                    next + 2
                ]
            );
        }
    }

    #[test]
    fn image_prompt_compresses_positions_and_decode_continues_after_max_axis() {
        let p = ImagePositions::new(
            &[
                7,
                248053,
                IMAGE_TOKEN_ID,
                IMAGE_TOKEN_ID,
                IMAGE_TOKEN_ID,
                IMAGE_TOKEN_ID,
                248054,
                9,
            ],
            &[[1, 4, 4]],
        )
        .unwrap();
        assert_eq!(
            p.prompt,
            vec![
                [0; 3],
                [1; 3],
                [2, 2, 2],
                [2, 2, 3],
                [2, 3, 2],
                [2, 3, 3],
                [4; 3],
                [5; 3]
            ]
        );
        assert_eq!(p.delta, -2);
        assert_eq!(p.values(8, 2).unwrap(), vec![6, 7, 6, 7, 6, 7]);
    }
    #[test]
    fn nonsquare_grid_keeps_spatial_axes_and_compresses_decode_offset() {
        let tokens = [
            248053,
            IMAGE_TOKEN_ID,
            IMAGE_TOKEN_ID,
            IMAGE_TOKEN_ID,
            IMAGE_TOKEN_ID,
            IMAGE_TOKEN_ID,
            IMAGE_TOKEN_ID,
            248054,
            9,
        ];
        let p = ImagePositions::new(&tokens, &[[1, 4, 6]]).unwrap();
        assert_eq!(p.prompt[6], [1, 2, 3]);
        assert_eq!(p.prompt[7], [4; 3]);
        assert_eq!(p.delta, -3);
        assert_eq!(p.values(9, 1).unwrap(), vec![6; 3]);
    }

    #[test]
    fn packed_images_have_independent_checked_boundaries() {
        assert_eq!(
            image_patch_offsets(&[[1, 4, 6], [1, 2, 4]]).unwrap(),
            vec![0, 24, 32]
        );
        assert!(image_patch_offsets(&[[1, u32::MAX - 1, 2]]).is_err());
        assert!(image_patch_offsets(&[[1, 65536, 65536]]).is_err());
    }

    #[test]
    fn multiple_images_keep_independent_axes_and_accumulate_compression() {
        let tokens = [
            7,
            248053,
            IMAGE_TOKEN_ID,
            IMAGE_TOKEN_ID,
            IMAGE_TOKEN_ID,
            IMAGE_TOKEN_ID,
            248054,
            9,
            248053,
            IMAGE_TOKEN_ID,
            IMAGE_TOKEN_ID,
            248054,
            10,
        ];
        let p = ImagePositions::new(&tokens, &[[1, 4, 4], [1, 2, 4]]).unwrap();
        assert_eq!(
            &p.prompt[2..6],
            &[[2, 2, 2], [2, 2, 3], [2, 3, 2], [2, 3, 3]]
        );
        assert_eq!(&p.prompt[6..9], &[[4; 3], [5; 3], [6; 3]]);
        assert_eq!(&p.prompt[9..11], &[[7, 7, 7], [7, 7, 8]]);
        assert_eq!(&p.prompt[11..], &[[9; 3], [10; 3]]);
        assert_eq!(p.delta, -2);
        assert_eq!(
            p.values(tokens.len(), 2).unwrap(),
            vec![11, 12, 11, 12, 11, 12]
        );
        assert!(ImagePositions::new(&tokens, &[[1, 4, 4]]).is_err());
        assert!(ImagePositions::new(&tokens, &[[1, 4, 4], [1, 4, 2], [1, 2, 2]]).is_err());
    }

    #[test]
    fn rejects_zero_video_multiple_images_and_malformed_placeholders() {
        let good = [248053, IMAGE_TOKEN_ID, 248054];
        for grids in [
            vec![],
            vec![[0, 2, 2]],
            vec![[2, 2, 2]],
            vec![[1, 0, 2]],
            vec![[1, 3, 2]],
            vec![[1, 2, 2], [1, 2, 2]],
        ] {
            assert!(ImagePositions::new(&good, &grids).is_err());
        }
        for tokens in [
            vec![IMAGE_TOKEN_ID],
            vec![1, IMAGE_TOKEN_ID, 2],
            vec![248053, IMAGE_TOKEN_ID, 248054, IMAGE_TOKEN_ID],
            vec![248053, IMAGE_TOKEN_ID, 7, 248054],
        ] {
            assert!(ImagePositions::new(&tokens, &[[1, 2, 2]]).is_err());
        }
        assert!(ImagePositions::new(&good, &[[1, 4, 4]]).is_err());
    }
}
