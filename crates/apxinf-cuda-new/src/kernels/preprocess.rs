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

/// `rgb_u8_to_normalized_temporal_merged_patches_bf16`: normalize a `u8` RGB
/// batch by mean/std (float64 rescale, float32 normalization — the
/// Transformers boundary) and patchify with temporal repetition and spatial
/// merge reordering into
/// `[views*(size/patch)^2, 3*temporal*patch^2]` BF16.
#[allow(clippy::too_many_arguments)]
pub fn rgb_u8_to_normalized_temporal_merged_patches_bf16(
    ctx: &CudaContext,
    images: &CudaBuffer,
    patches: &Tensor,
    views: usize,
    image_size: usize,
    patch_size: usize,
    temporal_patch_size: usize,
    merge_size: usize,
    layout: ImageLayout,
    rescale_factor: f64,
    image_mean: [f32; 3],
    image_std: [f32; 3],
) -> Result<()> {
    use crate::ffi::abi::{preprocess as abi, status};
    if !rescale_factor.is_finite()
        || rescale_factor <= 0.0
        || image_mean.iter().any(|value| !value.is_finite())
        || image_std
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0)
    {
        return Err(Error::Other(
            "invalid temporal-merged BF16 image preprocessing parameters".into(),
        ));
    }
    if views == 0
        || image_size == 0
        || patch_size == 0
        || temporal_patch_size == 0
        || merge_size == 0
        || image_size % (patch_size * merge_size) != 0
    {
        return Err(Error::Other(
            "invalid temporal-merged BF16 image preprocessing dimensions".into(),
        ));
    }
    let grid_size = image_size / patch_size;
    let patch_rows = views * grid_size * grid_size;
    let patch_width = 3 * temporal_patch_size * patch_size * patch_size;
    let expected_bytes = views * 3 * image_size * image_size;
    if images.device() != ctx.device_id() || images.len() != expected_bytes {
        return Err(Error::Other(format!(
            "temporal-merged raw images must contain exactly {expected_bytes} bytes on CUDA {}, got {} bytes on CUDA {}",
            ctx.device_id(),
            images.len(),
            images.device()
        )));
    }
    if patches.dtype() != DType::BF16 || patches.shape().dims() != [patch_rows, patch_width] {
        return Err(Error::Other(format!(
            "temporal-merged patches must be BF16 [{patch_rows}, {patch_width}], got {} {:?}",
            patches.dtype(),
            patches.shape().dims()
        )));
    }
    let to_i32 = |value: usize, what: &str| {
        i32::try_from(value).map_err(|_| Error::Other(format!("{what} exceeds i32")))
    };
    let patches_buffer = CudaBuffer::from_tensor(patches).map_err(Error::Cuda)?;
    unsafe {
        status::check(abi::apxinf_preprocess_temporal_merged_patches_bf16(
            images.ptr(),
            patches_buffer.ptr(),
            to_i32(views, "views")?,
            to_i32(image_size, "image size")?,
            to_i32(patch_size, "patch size")?,
            to_i32(temporal_patch_size, "temporal patch size")?,
            to_i32(merge_size, "merge size")?,
            match layout {
                ImageLayout::Nhwc => 1,
                ImageLayout::Nchw => 0,
            },
            rescale_factor,
            image_mean[0],
            image_mean[1],
            image_mean[2],
            image_std[0],
            image_std[1],
            image_std[2],
            ctx.stream().handle(),
        ))
    }
}
