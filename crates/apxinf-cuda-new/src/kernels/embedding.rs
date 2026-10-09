//! Legacy `kernels::embedding` names over cuda-new kernels.

use apxinf_core::{DType, Error, Result, Shape, Tensor};

use crate::ffi::abi::{elementwise as abi, status};
use crate::{CudaBuffer, CudaContext};

/// `lookup`: gather `seq_len` rows of `table` by u32 token ids held in a
/// device buffer — the unscaled legacy semantic. Callers that need the
/// Gemma-style `sqrt(width)` scale apply it separately, exactly as they did
/// against the legacy backend, so the rounding sequence is unchanged.
pub fn lookup(
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
    let cols = dims[1];
    let output = ctx.allocate_output(Shape::new(vec![seq_len, cols]), DType::BF16)?;
    let table_buffer = CudaBuffer::from_tensor(table).map_err(Error::Cuda)?;
    let output_buffer = CudaBuffer::from_tensor(&output).map_err(Error::Cuda)?;
    let rows = i64::try_from(seq_len).map_err(|_| Error::Other("lookup rows exceed i64".into()))?;
    let cols_i64 = i64::try_from(cols).map_err(|_| Error::Other("lookup cols exceed i64".into()))?;
    unsafe {
        status::check(abi::apxinf_elementwise_gather_rows_bf16(
            table_buffer.ptr(),
            ids.ptr(),
            output_buffer.ptr(),
            rows,
            cols_i64,
            ctx.stream().handle(),
        ))?;
    }
    Ok(output)
}

/// `add_position_f32_bf16`: F32 projection + learned position embedding
/// (+ optional bias), rounded once to BF16 — the vision patch-embedding
/// epilogue.
pub fn add_position_f32_bf16(
    ctx: &CudaContext,
    projection: &Tensor,
    bias: Option<&Tensor>,
    position: &Tensor,
    tokens_per_view: usize,
) -> Result<Tensor> {
    let dims = projection.shape().dims();
    if dims.len() != 2 || projection.dtype() != DType::F32 || position.dtype() != DType::F32 {
        return Err(Error::Other(
            "position embedding expects rank-2 F32 projection and position".into(),
        ));
    }
    let (rows, cols) = (dims[0], dims[1]);
    if tokens_per_view == 0 || rows % tokens_per_view != 0 {
        return Err(Error::Other(
            "position embedding rows must divide by tokens_per_view".into(),
        ));
    }
    let output = ctx.allocate_output(Shape::new(vec![rows, cols]), DType::BF16)?;
    let projection_buffer = CudaBuffer::from_tensor(projection).map_err(Error::Cuda)?;
    let position_buffer = CudaBuffer::from_tensor(position).map_err(Error::Cuda)?;
    let bias_buffer = bias
        .map(CudaBuffer::from_tensor)
        .transpose()
        .map_err(Error::Cuda)?;
    let output_buffer = CudaBuffer::from_tensor(&output).map_err(Error::Cuda)?;
    let count = i64::try_from(rows * cols)
        .map_err(|_| Error::Other("position embedding size exceeds i64".into()))?;
    let cols_i32 =
        i32::try_from(cols).map_err(|_| Error::Other("position embedding cols exceed i32".into()))?;
    let tokens_i32 = i32::try_from(tokens_per_view)
        .map_err(|_| Error::Other("tokens_per_view exceeds i32".into()))?;
    unsafe {
        status::check(abi::apxinf_elementwise_bias_position_f32_bf16(
            projection_buffer.ptr(),
            bias_buffer
                .as_ref()
                .map_or(std::ptr::null(), |buffer| buffer.ptr() as *const _),
            position_buffer.ptr(),
            output_buffer.ptr(),
            count,
            cols_i32,
            tokens_i32,
            ctx.stream().handle(),
        ))?;
    }
    Ok(output)
}
