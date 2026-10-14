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

/// Convert an F32 activation to BF16 with round-to-nearest-even.
pub fn cast_f32_to_bf16(ctx: &CudaContext, input: &Tensor, output: &Tensor) -> Result<()> {
    let dims = input.shape().dims().to_vec();
    if dims != output.shape().dims() {
        return Err(invalid("F32 to BF16 cast requires matching shapes"));
    }
    let input = tensor_storage(ctx, input, DType::F32, &dims)?;
    let output = tensor_storage(ctx, output, DType::BF16, &dims)?;
    unsafe {
        status::check(abi::apxinf_model_cast_f32_to_bf16(
            input.ptr(),
            output.ptr(),
            dims.iter().product::<usize>() as i64,
            ctx.stream().handle(),
        ))
    }
}

/// Exact (erf) GELU with BF16 input/output storage.
pub fn gelu_exact(ctx: &CudaContext, input: &Tensor, output: &Tensor) -> Result<()> {
    let dims = input.shape().dims().to_vec();
    if dims != output.shape().dims() {
        return Err(invalid("exact GELU requires matching shapes"));
    }
    let input = tensor_storage(ctx, input, DType::BF16, &dims)?;
    let output = tensor_storage(ctx, output, DType::BF16, &dims)?;
    unsafe {
        status::check(abi::apxinf_model_gelu_exact_bf16(
            input.ptr(),
            output.ptr(),
            dims.iter().product::<usize>() as i64,
            ctx.stream().handle(),
        ))
    }
}

/// Replace selected BF16 rows according to an I32 map.
///
/// A negative entry keeps the base row; every non-negative entry selects that
/// row from `replacement`.
pub fn replace_rows(
    ctx: &CudaContext,
    base: &Tensor,
    replacement: &Tensor,
    row_map: &Tensor,
    output: &Tensor,
) -> Result<()> {
    let base_dims = base.shape().dims();
    let replacement_dims = replacement.shape().dims();
    if base_dims.len() != 2
        || replacement_dims.len() != 2
        || replacement_dims[0] == 0
        || replacement_dims[1] != base_dims[1]
        || output.shape().dims() != base_dims
        || row_map.shape().dims() != [base_dims[0]]
    {
        return Err(invalid("row replacement shape mismatch"));
    }
    let base = tensor_storage(ctx, base, DType::BF16, base_dims)?;
    let replacement = tensor_storage(ctx, replacement, DType::BF16, replacement_dims)?;
    let row_map = tensor_storage(ctx, row_map, DType::I32, &[base_dims[0]])?;
    let output = tensor_storage(ctx, output, DType::BF16, base_dims)?;
    unsafe {
        status::check(abi::apxinf_model_replace_rows_bf16(
            base.ptr(),
            replacement.ptr(),
            row_map.ptr(),
            output.ptr(),
            base_dims[0] as i64,
            base_dims[1] as i64,
            replacement_dims[0] as i64,
            ctx.stream().handle(),
        ))
    }
}
