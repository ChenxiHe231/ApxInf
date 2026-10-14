//! Whole-model helpers: embedding gather and greedy token selection.

use apxinf_core::{DType, Result, Tensor};

use crate::ffi::abi::{model as abi, status};
use crate::ops::gemm::contracts::{invalid, tensor_storage};
use crate::CudaContext;

/// `output[t, :] = table[ids[t], :]`
pub fn embedding_gather(
    ctx: &CudaContext,
    table: &Tensor,
    ids: &Tensor,
    output: &Tensor,
) -> Result<()> {
    let table_dims = table.shape().dims().to_vec();
    let output_dims = output.shape().dims().to_vec();
    if table_dims.len() != 2 || output_dims.len() != 2 {
        return Err(invalid("embedding gather expects rank-2 tensors"));
    }
    let (vocab, hidden) = (table_dims[0], table_dims[1]);
    if output_dims[1] != hidden {
        return Err(invalid("embedding output width must match the table"));
    }
    let tokens = output_dims[0];
    let table_buffer = tensor_storage(ctx, table, DType::BF16, &table_dims)?;
    let id_buffer = tensor_storage(ctx, ids, DType::I32, &[tokens])?;
    let output_buffer = tensor_storage(ctx, output, DType::BF16, &output_dims)?;
    unsafe {
        status::check(abi::apxinf_model_embedding_gather(
            table_buffer.ptr(),
            id_buffer.ptr(),
            output_buffer.ptr(),
            tokens as i64,
            hidden as i64,
            vocab as i64,
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
        status::check(abi::apxinf_model_argmax_bf16(
            logits_buffer.ptr(),
            index_buffer.ptr(),
            count as i64,
            ctx.stream().handle(),
        ))
    }
}

/// Replace rows of a BF16 destination embedding matrix.
///
/// `source_rows` is I32 `[destination_rows]`; `-1` preserves that row and
/// non-negative values select a row from `source`.
pub fn replace_rows_bf16(
    ctx: &CudaContext,
    destination: &Tensor,
    source: &Tensor,
    source_rows: &Tensor,
) -> Result<()> {
    let destination_dims = destination.shape().dims().to_vec();
    let source_dims = source.shape().dims().to_vec();
    if destination_dims.len() != 2
        || source_dims.len() != 2
        || destination_dims[1] != source_dims[1]
    {
        return Err(invalid(
            "row replacement expects equal-width rank-2 matrices",
        ));
    }
    let destination_buffer = tensor_storage(ctx, destination, DType::BF16, &destination_dims)?;
    let source_buffer = tensor_storage(ctx, source, DType::BF16, &source_dims)?;
    let rows_buffer = tensor_storage(ctx, source_rows, DType::I32, &[destination_dims[0]])?;
    unsafe {
        status::check(abi::apxinf_model_replace_rows_bf16(
            destination_buffer.ptr(),
            source_buffer.ptr(),
            rows_buffer.ptr(),
            destination_dims[0] as i64,
            source_dims[0] as i64,
            destination_dims[1] as i64,
            ctx.stream().handle(),
        ))
    }
}
