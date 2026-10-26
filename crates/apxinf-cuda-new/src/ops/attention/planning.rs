use apxinf_core::{DType, Error, Result, Tensor};

use crate::context::CudaContext;
use crate::ffi::abi::attention as attention_abi;
use crate::ops::validation::{check_cuda, expect_bf16, gpu_ptr, matrix_shape};

/// Prepare Q/K/V and caches for full attention.
pub fn full_attn_prepare(
    ctx: &CudaContext,
    fused: &Tensor,
    q_norm_w: &Tensor,
    k_norm_w: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    q_out: &Tensor,
    k_cache: &Tensor,
    v_cache: &Tensor,
    cache_offset: usize,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    eps: f32,
) -> Result<()> {
    let (seq, width) = matrix_shape(fused, "full attention prepare")?;
    let expected = q_heads * 2 * head_dim + 2 * kv_heads * head_dim;
    let q_dims = q_out.shape().dims();
    let cache_dims = k_cache.shape().dims();
    if seq == 0
        || width < expected
        || q_dims != [seq, q_heads, head_dim]
        || cache_dims.len() != 3
        || cache_dims[1] != kv_heads
        || cache_dims[2] != head_dim
        || v_cache.shape().dims() != cache_dims
        || cache_offset + seq > cache_dims[0]
        || q_norm_w.shape().dims() != [head_dim]
        || k_norm_w.shape().dims() != [head_dim]
        || cos.shape().dims() != [seq, rotary_dim]
        || sin.shape().dims() != [seq, rotary_dim]
        || rotary_dim == 0
        || rotary_dim % 2 != 0
        || rotary_dim > head_dim
    {
        return Err(Error::Other("full attention prepare shape mismatch".into()));
    }
    for tensor in [fused, q_norm_w, k_norm_w, cos, sin, q_out, k_cache, v_cache] {
        expect_bf16(tensor, "full attention prepare")?;
    }
    unsafe {
        check_cuda(attention_abi::apxinf_attention_full_prepare_bf16(
            gpu_ptr(fused)?,
            gpu_ptr(q_norm_w)?,
            gpu_ptr(k_norm_w)?,
            gpu_ptr(cos)?,
            gpu_ptr(sin)?,
            gpu_ptr(q_out)?,
            gpu_ptr(k_cache)?,
            gpu_ptr(v_cache)?,
            seq as i32,
            cache_offset as i32,
            q_heads as i32,
            kv_heads as i32,
            head_dim as i32,
            rotary_dim as i32,
            width as i64,
            (kv_heads * head_dim) as i64,
            eps,
            ctx.stream().handle(),
        ))
    }
}

/// In-place post-attention sigmoid gate: `attn *= sigmoid(gate)` where the
/// gate is read from the (q|gate) fused projection rows.
pub fn sigmoid_gate_mul(
    ctx: &CudaContext,
    attn: &Tensor,
    fused: &Tensor,
    heads: usize,
    head_dim: usize,
) -> Result<()> {
    let (rows, width) = matrix_shape(attn, "sigmoid gate")?;
    let (fused_rows, fused_width) = matrix_shape(fused, "sigmoid gate")?;
    if rows == 0
        || width != heads * head_dim
        || fused_rows != rows
        || fused_width < 2 * heads * head_dim
    {
        return Err(Error::Other("sigmoid gate shape mismatch".into()));
    }
    expect_bf16(attn, "sigmoid gate")?;
    expect_bf16(fused, "sigmoid gate")?;
    unsafe {
        check_cuda(attention_abi::apxinf_attention_sigmoid_gate_mul_bf16(
            gpu_ptr(attn)?,
            gpu_ptr(fused)?,
            rows as i32,
            heads as i32,
            head_dim as i32,
            fused_width as i64,
            ctx.stream().handle(),
        ))
    }
}

/// Prepare group-major expert Q/K/V tensors.
pub fn expert_qkv_prepare(
    ctx: &CudaContext,
    fused: &Tensor,
    q_norm_w: &Tensor,
    k_norm_w: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    q_out: &Tensor,
    gate_out: &Tensor,
    k_out: &Tensor,
    v_out: &Tensor,
    q_heads: usize,
    kv_heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    eps: f32,
) -> Result<()> {
    let (seq, width) = matrix_shape(fused, "expert qkv prepare")?;
    let heads_per_group = q_heads.checked_div(kv_heads).unwrap_or(0);
    let expected = kv_heads * (2 * heads_per_group + 2) * head_dim;
    if seq == 0
        || kv_heads == 0
        || q_heads % kv_heads != 0
        || width < expected
        || q_out.shape().dims() != [seq, q_heads, head_dim]
        || gate_out.shape().dims() != [seq, q_heads, head_dim]
        || k_out.shape().dims() != [seq, kv_heads, head_dim]
        || v_out.shape().dims() != [seq, kv_heads, head_dim]
        || q_norm_w.shape().dims() != [head_dim]
        || k_norm_w.shape().dims() != [head_dim]
        || cos.shape().dims() != [seq, rotary_dim]
        || sin.shape().dims() != [seq, rotary_dim]
        || rotary_dim == 0
        || rotary_dim % 2 != 0
        || rotary_dim > head_dim
    {
        return Err(Error::Other("expert qkv prepare shape mismatch".into()));
    }
    for tensor in [
        fused, q_norm_w, k_norm_w, cos, sin, q_out, gate_out, k_out, v_out,
    ] {
        expect_bf16(tensor, "expert qkv prepare")?;
    }
    unsafe {
        check_cuda(attention_abi::apxinf_attention_expert_qkv_prepare_bf16(
            gpu_ptr(fused)?,
            gpu_ptr(q_norm_w)?,
            gpu_ptr(k_norm_w)?,
            gpu_ptr(cos)?,
            gpu_ptr(sin)?,
            gpu_ptr(q_out)?,
            gpu_ptr(gate_out)?,
            gpu_ptr(k_out)?,
            gpu_ptr(v_out)?,
            seq as i32,
            q_heads as i32,
            kv_heads as i32,
            head_dim as i32,
            rotary_dim as i32,
            width as i64,
            eps,
            ctx.stream().handle(),
        ))
    }
}

/// In-place post-attention sigmoid gate from a standalone gate tensor.
pub fn expert_sigmoid_gate_mul(ctx: &CudaContext, attn: &Tensor, gate: &Tensor) -> Result<()> {
    if attn.shape().dims() != gate.shape().dims() || attn.numel() == 0 {
        return Err(Error::Other("expert sigmoid gate shape mismatch".into()));
    }
    expect_bf16(attn, "expert sigmoid gate")?;
    expect_bf16(gate, "expert sigmoid gate")?;
    unsafe {
        check_cuda(
            attention_abi::apxinf_attention_expert_sigmoid_gate_mul_bf16(
                gpu_ptr(attn)?,
                gpu_ptr(gate)?,
                attn.numel() as i64,
                ctx.stream().handle(),
            ),
        )
    }
}

/// Non-causal grouped-query BF16 attention.
pub fn gqa_bf16(
    ctx: &CudaContext,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    key_tokens: usize,
) -> Result<Tensor> {
    let q_shape = q.shape().dims();
    let k_shape = k.shape().dims();
    if [q, k, v]
        .into_iter()
        .any(|tensor| tensor.dtype() != DType::BF16)
        || q_shape.len() != 3
        || k_shape.len() != 3
        || v.shape() != k.shape()
        || k_shape[0] < key_tokens
        || k_shape[2] != q_shape[2]
        || k_shape[1] == 0
        || q_shape[1] % k_shape[1] != 0
        || key_tokens == 0
    {
        return Err(Error::Other(
            "non-causal joint BF16 GQA shape mismatch".into(),
        ));
    }
    if q_shape[0] == 0 || q_shape[1] == 0 || q_shape[2] == 0 {
        return Err(Error::Other(
            "non-causal GQA requires nonempty queries".into(),
        ));
    }
    // cuda-new routes this through the tuned KV-cache attention operator;
    // the legacy FA2 dispatch is a candidate inside that operator now.
    super::noncausal_gqa_bf16(ctx, q, k, v, key_tokens)
}
