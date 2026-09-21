use apxinf_core::{DType, Result, Tensor};

use super::contracts::{invalid, tensor_storage};
use crate::ffi::abi::{gemm as abi, status};
use crate::CudaContext;

/// Bytes an NVFP4 block-scale buffer occupies in the layout the kernel reads.
///
/// `rows` is M for an activation operand and N for a weight. The layout pads
/// the logical `[rows, k / block_size]` grid up to the kernel's atom
/// boundaries, so the answer is larger than the checkpoint's own scale tensor
/// and has to be queried rather than derived.
pub fn nvfp4_scale_buffer_bytes(rows: usize, k: usize, block_size: u32) -> Result<usize> {
    if rows == 0 || k == 0 || rows > i32::MAX as usize || k > i32::MAX as usize {
        return Err(invalid("NVFP4 scale buffer needs a nonempty shape"));
    }
    let bytes =
        unsafe { abi::apxinf_gemm_nvfp4_scale_buffer_bytes(rows as i64, k as i64, block_size) };
    if bytes == 0 {
        return Err(invalid(format!(
            "unsupported NVFP4 block size {block_size}"
        )));
    }
    Ok(bytes as usize)
}

/// Rewrite checkpoint-order block scales into the layout the kernel reads.
///
/// `source` is the row-major `[rows, k / block_size]` E4M3 tensor as a
/// checkpoint stores it. `destination` must be an E4M3 tensor holding at least
/// [`nvfp4_scale_buffer_bytes`] bytes.
///
/// The target layout depends only on `block_size`, not on which tactic the
/// autotuner picks, so this runs once when weights are loaded and stays valid
/// for the life of the model.
pub fn nvfp4_pack_block_scales(
    ctx: &CudaContext,
    source: &Tensor,
    destination: &Tensor,
    rows: usize,
    k: usize,
    block_size: u32,
) -> Result<()> {
    let required = nvfp4_scale_buffer_bytes(rows, k, block_size)?;
    let blocks = k.div_ceil(block_size as usize);

    let source_dims = source.shape().dims().to_vec();
    let source_buffer = tensor_storage(ctx, source, DType::F8E4M3, &source_dims)?;
    if source_buffer.len() < rows * blocks {
        return Err(invalid(format!(
            "NVFP4 scale source holds {} bytes, needs {}",
            source_buffer.len(),
            rows * blocks
        )));
    }

    let destination_dims = destination.shape().dims().to_vec();
    let destination_buffer = tensor_storage(ctx, destination, DType::F8E4M3, &destination_dims)?;
    if destination_buffer.len() < required {
        return Err(invalid(format!(
            "NVFP4 scale destination holds {} bytes, needs {required}",
            destination_buffer.len()
        )));
    }

    unsafe {
        status::check(abi::apxinf_gemm_nvfp4_pack_block_scales(
            source_buffer.ptr(),
            destination_buffer.ptr(),
            rows as i64,
            k as i64,
            block_size,
            ctx.stream().handle(),
        ))
    }
}
