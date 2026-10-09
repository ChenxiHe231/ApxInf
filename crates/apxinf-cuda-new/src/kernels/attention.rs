//! Legacy `kernels::attention` names over cuda-new attention operators.
//!
//! The legacy helpers take `[tokens, heads, head_dim]` tensors; the cuda-new
//! Attention contract is rank-4 `[batch, tokens, heads, head_dim]`. These
//! adapters reshape at the boundary — reshape on a contiguous tensor is a
//! metadata change, not a copy.
//!
//! Causality follows the legacy helper it replaces: `mqa_bf16` and `mha_bf16`
//! are non-causal over the valid keys; `causal_gqa_bf16` is causal with the
//! query block at the end of the key range.

use apxinf_core::{DType, Error, Result, Shape, Tensor};

use crate::{ops, CudaContext};

/// Q/K/V triple produced by a packed-QKV split. Mirrors the legacy struct.
pub struct QkvTensors {
    pub q: Tensor,
    pub k: Tensor,
    pub v: Tensor,
}

fn rank3(tensor: &Tensor, what: &str) -> Result<[usize; 3]> {
    let dims = tensor.shape().dims();
    if dims.len() != 3 {
        return Err(Error::Other(format!("{what} requires rank-3 [tokens, heads, dim]")));
    }
    Ok([dims[0], dims[1], dims[2]])
}

fn dense_attention(
    ctx: &CudaContext,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    key_tokens: usize,
    kv_heads: usize,
    causal: bool,
) -> Result<Tensor> {
    let [query_tokens, query_heads, head_dim] = rank3(q, "attention query")?;
    let q4 = q.reshape(vec![1, query_tokens, query_heads, head_dim])?;
    // K/V may be larger caches; present exactly the valid prefix.
    let k4 = k.reshape(vec![1, k.shape().numel() / (kv_heads * head_dim), kv_heads, head_dim])?;
    let v4 = v.reshape(vec![1, v.shape().numel() / (kv_heads * head_dim), kv_heads, head_dim])?;
    let mut out = ctx.allocate_output(Shape::new(vec![1, query_tokens, query_heads, head_dim]), DType::BF16)?;
    let mut args = ops::KvCacheAttentionArgs::new(&q4, &k4, &v4, &mut out);
    args.valid_key_tokens = key_tokens;
    if causal {
        args.mask = ops::AttentionMask::Causal;
        args.query_start = key_tokens - query_tokens;
    } else {
        args.mask = ops::AttentionMask::None;
        args.query_start = 0;
    }
    args.scale = 1.0 / (head_dim as f32).sqrt();
    ops::kv_cache_attention(ctx, args)?;
    out.reshape(vec![query_tokens, query_heads, head_dim])
}

/// `mqa_bf16`: multi-query attention — one shared K/V head, non-causal over
/// the leading `key_tokens` rows of flat `[*, head_dim]` K/V storage.
pub fn mqa_bf16(
    ctx: &CudaContext,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    key_tokens: usize,
) -> Result<Tensor> {
    dense_attention(ctx, q, k, v, key_tokens, 1, false)
}

/// `mha_bf16`: dense multi-head attention over equal-shaped Q/K/V, non-causal,
/// batched by `tokens_per_batch`.
pub fn mha_bf16(
    ctx: &CudaContext,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    tokens_per_batch: usize,
) -> Result<Tensor> {
    let [tokens, heads, head_dim] = rank3(q, "MHA query")?;
    if tokens_per_batch == 0 || tokens % tokens_per_batch != 0 {
        return Err(Error::Other("MHA tokens_per_batch mismatch".into()));
    }
    let batches = tokens / tokens_per_batch;
    let q4 = q.reshape(vec![batches, tokens_per_batch, heads, head_dim])?;
    let k4 = k.reshape(vec![batches, tokens_per_batch, heads, head_dim])?;
    let v4 = v.reshape(vec![batches, tokens_per_batch, heads, head_dim])?;
    let mut out =
        ctx.allocate_output(Shape::new(vec![batches, tokens_per_batch, heads, head_dim]), DType::BF16)?;
    let args = ops::AttentionArgs::new(&q4, &k4, &v4, &mut out);
    ops::attention(ctx, args)?;
    out.reshape(vec![tokens, heads, head_dim])
}

/// `causal_gqa_bf16`: grouped-query causal attention with the query block at
/// the end of `key_tokens` keys.
pub fn causal_gqa_bf16(
    ctx: &CudaContext,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    key_tokens: usize,
) -> Result<Tensor> {
    let [_, kv_heads, _] = rank3(k, "GQA keys")?;
    dense_attention(ctx, q, k, v, key_tokens, kv_heads, true)
}

/// `split_qkv_bias_bf16`: split a `[tokens, 3*heads*dim]` packed projection
/// into Q/K/V with an optional packed bias, no rotation (the vision layout).
pub fn split_qkv_bias_bf16(
    ctx: &CudaContext,
    qkv: &Tensor,
    bias: Option<&Tensor>,
    heads: usize,
    head_dim: usize,
) -> Result<QkvTensors> {
    let dims = qkv.shape().dims();
    if dims.len() != 2 || dims[1] != 3 * heads * head_dim {
        return Err(Error::Other(
            "packed QKV split expects [tokens, 3*heads*dim]".into(),
        ));
    }
    let tokens = dims[0];
    let shape = Shape::new(vec![tokens, heads, head_dim]);
    let mut q = ctx.allocate_output(shape.clone(), DType::BF16)?;
    let mut k = ctx.allocate_output(shape.clone(), DType::BF16)?;
    let mut v = ctx.allocate_output(shape, DType::BF16)?;
    ops::rope(
        ctx,
        ops::RopeArgs {
            semantic: ops::RopeSemantic::SplitQkvBias,
            qkv,
            bias,
            q: &mut q,
            k: &mut k,
            v: &mut v,
            q_heads: heads,
            kv_heads: heads,
            head_dim,
            theta: 0.0,
            position_offset: 0,
            kv_output_offset: 0,
        },
    )?;
    Ok(QkvTensors { q, k, v })
}
