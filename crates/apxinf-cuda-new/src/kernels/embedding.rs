//! Legacy `kernels::embedding` names over cuda-new gather operators.
//!
//! The cuda-new `EmbeddingLookup` gather fuses the `sqrt(cols)` scaling that
//! the legacy pipeline applied as a separate `elementwise::scale` step.
//! The fused form is exposed as [`lookup_scaled`]; a bare unscaled `lookup`
//! has no cuda-new operator yet and is deliberately absent so a port cannot
//! silently double-scale.

use apxinf_core::{DType, Error, Result, Shape, Tensor};

use crate::{ops, CudaBuffer, CudaContext};

/// Gather `seq_len` rows of `table` by u32 token ids and multiply by
/// `sqrt(cols)` — the fused embedding step of the Gemma-style models.
///
/// Call sites that previously ran `embedding::lookup` followed by
/// `elementwise::scale(sqrt(width))` collapse into this single call.
pub fn lookup_scaled(
    ctx: &CudaContext,
    table: &Tensor,
    ids: &CudaBuffer,
    seq_len: usize,
) -> Result<Tensor> {
    let dims = table.shape().dims();
    if dims.len() != 2 {
        return Err(Error::Other("embedding lookup expects a rank-2 table".into()));
    }
    if table.dtype() != DType::BF16 {
        return Err(Error::Other(
            "cuda-new embedding lookup currently supports BF16 tables".into(),
        ));
    }
    let (vocab, cols) = (dims[0], dims[1]);
    let mut output = ctx.allocate_output(Shape::new(vec![seq_len, cols]), DType::BF16)?;
    let mut args = ops::GatherArgs::new(ops::GatherSemantic::EmbeddingLookup, table, &mut output);
    args.ids = Some(ids);
    args.vocab_size = vocab;
    ops::gather(ctx, args)?;
    Ok(output)
}
