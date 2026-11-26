//! Identity for processed media, independent of URLs and placeholder token IDs.
use apxinf_core::{DType, Device, Error, Result, Storage, Tensor};
use sha2::{Digest, Sha256};

use super::vision_config::IMAGE_TOKEN_ID;
use super::vision_input::{image_patch_offsets, ImagePositions};

#[derive(Clone, Debug, PartialEq, Eq)]
pub(super) struct ImageKey {
    pub grid: [u32; 3],
    pub digest: [u8; 32],
}

#[derive(Clone)]
pub(super) struct ImageIdentity {
    pub keys: Vec<ImageKey>,
    starts: Vec<usize>,
    pub positions: std::rc::Rc<ImagePositions>,
}

impl ImageIdentity {
    /// Keep the original no-reuse path free of hashing and host downloads.
    /// This position-only plan is deliberately ineligible for cache matching.
    pub fn uncached(tokens: &[u32], grids: &[[u32; 3]]) -> Result<Self> {
        Ok(Self {
            keys: Vec::new(),
            starts: Vec::new(),
            positions: std::rc::Rc::new(ImagePositions::new(tokens, grids)?),
        })
    }
    /// Hash the actual processor output. A changed shared pixel budget must
    /// miss even when the source URL/bytes and placeholder IDs are unchanged.
    pub fn new(tokens: &[u32], pixels: &Tensor, grids: &[[u32; 3]]) -> Result<Self> {
        let positions = std::rc::Rc::new(ImagePositions::new(tokens, grids)?);
        let offsets = image_patch_offsets(grids)?;
        let bytes = match pixels.storage() {
            Storage::Cpu(bytes) if pixels.device() == Device::Cpu => bytes,
            _ => return Err(Error::Other("image identity requires host pixels".into())),
        };
        if !matches!(pixels.dtype(), DType::F32 | DType::BF16)
            || pixels.shape().dims() != [*offsets.last().unwrap() as usize, 1536]
            || bytes.len() != pixels.size_in_bytes()
        {
            return Err(Error::Other("invalid pixels for image identity".into()));
        }
        let row_bytes = 1536 * pixels.dtype().size_in_bytes();
        let mut keys = Vec::with_capacity(grids.len());
        let mut starts = Vec::with_capacity(grids.len());
        let mut cursor = 0;
        for (index, grid) in grids.iter().enumerate() {
            let start = cursor
                + tokens[cursor..]
                    .iter()
                    .position(|&t| t == IMAGE_TOKEN_ID)
                    .ok_or_else(|| Error::Other("missing image span".into()))?;
            starts.push(start);
            cursor = start + (grid[1] as usize / 2) * (grid[2] as usize / 2);
            let mut hash = Sha256::new();
            hash.update(b"qwen35-processed-image-v1");
            hash.update([pixels.dtype().size_in_bytes() as u8]);
            for value in grid {
                hash.update(value.to_le_bytes());
            }
            hash.update(
                &bytes
                    [offsets[index] as usize * row_bytes..offsets[index + 1] as usize * row_bytes],
            );
            keys.push(ImageKey {
                grid: *grid,
                digest: hash.finalize().into(),
            });
        }
        Ok(Self {
            keys,
            starts,
            positions,
        })
    }

    pub fn text_prefix_limit(&self) -> usize {
        self.starts.first().copied().unwrap_or(0)
    }

    /// Tokens are compared separately. Include the entire image identity if
    /// even one row of its span belongs to the cached prefix. Also compare
    /// absolute three-axis positions, including a generated continuation.
    pub fn matches_prefix(&self, cached: Option<&Self>, prefix: usize) -> bool {
        if self.keys.is_empty() || cached.is_some_and(|c| c.keys.is_empty()) {
            return false;
        }
        let Some(cached) = cached else {
            return prefix <= self.text_prefix_limit();
        };
        let images = self.starts.iter().take_while(|&&s| s < prefix).count();
        let cached_images = cached.starts.iter().take_while(|&&s| s < prefix).count();
        images == cached_images
            && self.starts[..images] == cached.starts[..images]
            && self.keys[..images] == cached.keys[..images]
            && (0..prefix).all(|i| {
                let position = |plan: &ImagePositions| {
                    plan.prompt
                        .get(i)
                        .map(|p| p.map(i64::from))
                        .unwrap_or([i as i64 + plan.delta; 3])
                };
                position(&self.positions) == position(&cached.positions)
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    const VISION_START_TOKEN_ID: u32 = 248053;
    const VISION_END_TOKEN_ID: u32 = 248054;

    fn identity(pixels: f32, grids: &[[u32; 3]]) -> (Vec<u32>, ImageIdentity) {
        let mut tokens = vec![10, 11];
        for g in grids {
            tokens.push(VISION_START_TOKEN_ID);
            tokens.extend(std::iter::repeat_n(
                IMAGE_TOKEN_ID,
                (g[1] * g[2] / 4) as usize,
            ));
            tokens.push(VISION_END_TOKEN_ID);
        }
        tokens.extend([12, 13]);
        let rows = grids.iter().map(|g| (g[1] * g[2]) as usize).sum();
        let tensor = Tensor::from_f32(vec![rows, 1536], &vec![pixels; rows * 1536]).unwrap();
        let id = ImageIdentity::new(&tokens, &tensor, grids).unwrap();
        (tokens, id)
    }

    #[test]
    fn processed_pixels_grids_and_partial_image_spans_are_part_of_identity() {
        let (tokens, a) = identity(0.0, &[[1, 2, 4]]);
        let (_, same) = identity(0.0, &[[1, 2, 4]]);
        let (_, changed) = identity(1.0, &[[1, 2, 4]]);
        let (_, reshaped) = identity(0.0, &[[1, 4, 2]]);
        assert!(a.matches_prefix(Some(&same), tokens.len()));
        assert!(!a.matches_prefix(Some(&changed), 4));
        assert!(!a.matches_prefix(Some(&reshaped), 4));
        assert!(a.matches_prefix(Some(&changed), 3));
        assert!(a.matches_prefix(None, 3));
        assert!(!a.matches_prefix(None, 4));
        let uncached = ImageIdentity::uncached(&tokens, &[[1, 2, 4]]).unwrap();
        assert!(!uncached.matches_prefix(Some(&a), tokens.len()));
        assert!(!a.matches_prefix(Some(&uncached), tokens.len()));
    }

    #[test]
    fn appended_images_do_not_invalidate_earlier_media_or_generated_positions() {
        let (tokens, a) = identity(0.0, &[[1, 2, 4]]);
        let (_, b) = identity(0.0, &[[1, 2, 4], [1, 4, 4]]);
        // New images change positions only from their own span onwards.
        assert!(b.matches_prefix(Some(&a), tokens.len() - 2));
        assert!(!b.matches_prefix(Some(&a), tokens.len() + 2));
        assert!(a.matches_prefix(Some(&a), tokens.len() + 8));
    }

    #[test]
    fn same_grid_images_are_ordered_and_host_pixels_remain_read_only() {
        let grids = [[1, 2, 2], [1, 2, 2]];
        let (tokens, _) = identity(0.0, &grids);
        let values: Vec<f32> = vec![0.0; 4 * 1536]
            .into_iter()
            .chain(vec![1.0; 4 * 1536])
            .collect();
        let pixels = Tensor::from_f32(vec![8, 1536], &values).unwrap();
        let a = ImageIdentity::new(&tokens, &pixels, &grids).unwrap();
        let reversed: Vec<f32> = values[4 * 1536..]
            .iter()
            .chain(&values[..4 * 1536])
            .copied()
            .collect();
        let reversed = Tensor::from_f32(vec![8, 1536], &reversed).unwrap();
        let b = ImageIdentity::new(&tokens, &reversed, &grids).unwrap();
        assert_ne!(a.keys[0], a.keys[1]);
        assert_eq!(a.keys[0], b.keys[1]);
        assert!(!a.matches_prefix(Some(&b), tokens.len()));
        assert!(a.matches_prefix(Some(&b), 3));
        assert_eq!(pixels.to_f32_vec().unwrap(), values);
    }
}
