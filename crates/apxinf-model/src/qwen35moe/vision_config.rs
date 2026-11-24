//! Fixed vision geometry for Qwen3.5/3.6-35B-A3B.

pub(crate) const IMAGE_TOKEN_ID: u32 = 248056;
// Vision tower. The checkpoint is a conditional-generation checkpoint, not
// a text-only export: it contains all `model.visual.*` tensors even though the
// original ApxInf runtime used only `model.language_model.*`.
pub(crate) const VISION_DEPTH: usize = 27;
pub(crate) const VISION_HIDDEN: usize = 1152;
pub(crate) const VISION_INTERMEDIATE: usize = 4304;
pub(crate) const VISION_HEADS: usize = 16;
pub(crate) const VISION_HEAD_DIM: usize = VISION_HIDDEN / VISION_HEADS;
pub(crate) const VISION_PATCH_SIZE: usize = 16;
pub(crate) const VISION_TEMPORAL_PATCH_SIZE: usize = 2;
pub(crate) const VISION_CHANNELS: usize = 3;
pub(crate) const VISION_MERGE: usize = 2;
pub(crate) const VISION_POSITIONS: usize = 2304;
pub(crate) const VISION_OUT_HIDDEN: usize = 2048;
