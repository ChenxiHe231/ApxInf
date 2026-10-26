//! Attention query/gate split and output gate helpers.

use apxinf_core::{DType, Result, Tensor};

use crate::ffi::abi::{attention as abi, status};
use crate::ops::gemm::contracts::{invalid, tensor_storage};
use crate::CudaContext;

/// Split q_proj's `[tokens, heads, 2*head_dim]` output into query and gate.
pub fn split_query_and_gate(
    ctx: &CudaContext,
    fused: &Tensor,
    query: &Tensor,
    gate: &Tensor,
) -> Result<()> {
    let dims = query.shape().dims().to_vec();
    if dims.len() != 3 {
        return Err(invalid("query must be [tokens, heads, head_dim]"));
    }
    let (tokens, heads, head_dim) = (dims[0], dims[1], dims[2]);
    let fused_buffer = tensor_storage(ctx, fused, DType::BF16, &[tokens, heads, 2 * head_dim])?;
    let query_buffer = tensor_storage(ctx, query, DType::BF16, &dims)?;
    let gate_buffer = tensor_storage(ctx, gate, DType::BF16, &dims)?;
    unsafe {
        status::check(abi::apxinf_attention_split_query_and_gate(
            fused_buffer.ptr(),
            query_buffer.ptr(),
            gate_buffer.ptr(),
            tokens as i64,
            heads as i64,
            head_dim as i64,
            ctx.stream().handle(),
        ))
    }
}

/// `data *= sigmoid(gate)`, elementwise and in place.
pub fn apply_output_gate(ctx: &CudaContext, data: &Tensor, gate: &Tensor) -> Result<()> {
    let dims = data.shape().dims().to_vec();
    if dims != gate.shape().dims() {
        return Err(invalid("output gate requires matching shapes"));
    }
    let data_buffer = tensor_storage(ctx, data, DType::BF16, &dims)?;
    let gate_buffer = tensor_storage(ctx, gate, DType::BF16, &dims)?;
    unsafe {
        status::check(abi::apxinf_attention_apply_output_gate(
            data_buffer.ptr(),
            gate_buffer.ptr(),
            dims.iter().product::<usize>() as i64,
            ctx.stream().handle(),
        ))
    }
}
