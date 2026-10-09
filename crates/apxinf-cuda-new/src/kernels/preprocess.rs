//! Legacy `kernels::preprocess` names over the cuda-new gather operator.

use apxinf_core::{DType, Error, Result, Tensor};

use crate::{ops, CudaBuffer, CudaContext};

/// Memory layout of a resized RGB `u8` image batch. Mirrors the legacy enum.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImageLayout {
    Nhwc,
    Nchw,
}

/// `rgb_u8_to_patches_bf16`: normalize a `u8` RGB batch to `[-1, 1]` and
/// patchify into `[views*(size/patch)^2, 3*patch^2]` BF16.
pub fn rgb_u8_to_patches_bf16(
    ctx: &CudaContext,
    images: &CudaBuffer,
    patches: &Tensor,
    views: usize,
    image_size: usize,
    patch_size: usize,
    layout: ImageLayout,
) -> Result<()> {
    if patches.dtype() != DType::BF16 {
        return Err(Error::Other(
            "cuda-new RGB preprocessing writes BF16 patches".into(),
        ));
    }
    // GatherArgs mutates only device memory; clone the handle for the &mut
    // contract without copying storage.
    let mut out = patches.clone();
    let args = ops::GatherArgs::rgb_to_patches(
        images,
        &mut out,
        ops::GatherPatchGeometry {
            views,
            image_size,
            patch_size,
            nhwc: layout == ImageLayout::Nhwc,
        },
    );
    ops::gather(ctx, args)
}

/// `rgb_u8_to_patches_f32`: the F32-output variant of the patchification,
/// used by pi0-fast whose vision tower keeps F32 until the first projection.
pub fn rgb_u8_to_patches_f32(
    ctx: &CudaContext,
    images: &CudaBuffer,
    patches: &Tensor,
    views: usize,
    image_size: usize,
    patch_size: usize,
    layout: ImageLayout,
) -> Result<()> {
    if patches.dtype() != DType::F32 {
        return Err(Error::Other(
            "rgb_u8_to_patches_f32 writes F32 patches".into(),
        ));
    }
    let mut out = patches.clone();
    let args = ops::GatherArgs::rgb_to_patches(
        images,
        &mut out,
        ops::GatherPatchGeometry {
            views,
            image_size,
            patch_size,
            nhwc: layout == ImageLayout::Nhwc,
        },
    );
    ops::gather(ctx, args)
}
