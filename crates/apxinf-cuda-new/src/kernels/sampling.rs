//! Legacy `kernels::sampling` names over cuda-new model helpers.

use apxinf_core::{DType, Error, Result, Shape, Tensor};

use crate::{ops, CudaBuffer, CudaContext};

/// `argmax_bf16_remapped_into`: device argmax over BF16 logits, then map the
/// winning index through a `u32` remap table into `out`.
///
/// cuda-new's `argmax` writes the raw index; the remap step is a one-element
/// gather the caller previously fused. Until a fused operator exists, this
/// shim is absent on purpose — see the module doc for the failing-signal
/// policy. The plain unmapped variant is provided for ports that can remap on
/// the host.
pub fn argmax_bf16_into(ctx: &CudaContext, logits: &Tensor, index: &CudaBuffer) -> Result<()> {
    if logits.dtype() != DType::BF16 {
        return Err(Error::Other(format!(
            "argmax expects BF16 logits, got {}",
            logits.dtype()
        )));
    }
    let index_tensor = index
        .as_tensor(Shape::new(vec![1]), DType::I32)
        .map_err(Error::Cuda)?;
    ops::argmax(ctx, logits, &index_tensor)
}
