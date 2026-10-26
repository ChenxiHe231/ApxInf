//! Device-side token selection operators.

use apxinf_core::{DType, Error, Result, Tensor};

use crate::ffi::abi::{elementwise as elementwise_abi, sampling as sampling_abi, status};
use crate::ops::gemm::contracts::tensor_storage;
use crate::{CudaBuffer, CudaContext};

mod suppress;

pub use suppress::suppress_logits;

/// `argmax_bf16_remapped_into`: device argmax over BF16 logits, mapping the
/// winning index through a u32 remap table into `out`. Ported bit-identically
/// from the legacy selection kernel, including its tie-break.
pub fn argmax_bf16_remapped_into(
    ctx: &CudaContext,
    logits: &Tensor,
    remap: &CudaBuffer,
    out: &CudaBuffer,
) -> Result<()> {
    if logits.dtype() != DType::BF16 {
        return Err(Error::Other(format!(
            "argmax expects BF16 logits, got {}",
            logits.dtype()
        )));
    }
    let n = logits.shape().numel();
    let n = u32::try_from(n).map_err(|_| Error::Other("argmax logits exceed u32".into()))?;
    if n == 0 {
        return Err(Error::Other("argmax needs at least one logit".into()));
    }
    let logits_buffer = CudaBuffer::from_tensor(logits).map_err(Error::Cuda)?;
    unsafe {
        status::check(elementwise_abi::apxinf_elementwise_argmax_remap_bf16(
            logits_buffer.ptr(),
            n,
            remap.ptr(),
            out.ptr(),
            ctx.stream().handle(),
        ))
    }
}

/// Index of the largest logit, reduced on device.
pub fn argmax(ctx: &CudaContext, logits: &Tensor, index: &Tensor) -> Result<()> {
    let dims = logits.shape().dims().to_vec();
    let count = dims.iter().product::<usize>();
    let logits_buffer = tensor_storage(ctx, logits, DType::BF16, &dims)?;
    let index_buffer = tensor_storage(ctx, index, DType::I32, &[1])?;
    unsafe {
        status::check(sampling_abi::apxinf_sampling_argmax(
            logits_buffer.ptr(),
            index_buffer.ptr(),
            count as i64,
            ctx.stream().handle(),
        ))
    }
}
