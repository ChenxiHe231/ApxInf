//! Gated DeltaNet primitives: the recurrent step, its gates, the causal
//! convolution window, and the gated output norm.
//!
//! The recurrence is implemented from the architecture's documented form and
//! has **not** been checked against a reference engine running this
//! checkpoint. The tests here establish that the kernels compute the stated
//! recurrence, not that the recurrence is the model's.

use apxinf_core::{DType, Result, Tensor};

use crate::ffi::abi::{gdn as abi, status};
use crate::ops::gemm::contracts::{invalid, tensor_storage};
use crate::CudaContext;

/// Shape of the recurrent state a GDN layer carries between tokens.
///
/// `[v_heads, v_dim, k_dim]` f32. For Qwen3.8-27B that is
/// 48 x 128 x 128 x 4 B = 3 MiB per layer, so 48 layers hold ~151 MiB of state
/// that decode reads and writes every token -- a bandwidth cost comparable to
/// a sizeable fraction of the weights, and one worth measuring rather than
/// assuming away.
pub fn gdn_state_elements(v_heads: usize, v_dim: usize, k_dim: usize) -> usize {
    v_heads * v_dim * k_dim
}

/// One recurrent step: decay, delta-rule correction, rank-1 update, readout.
///
/// `state` is `[v_heads, v_dim, k_dim]` f32, updated in place. `q`/`k` are
/// `[k_heads, k_dim]`, `v` is `[v_heads, v_dim]`; value head `h` reads k-head
/// `h / (v_heads / k_heads)`.
#[allow(clippy::too_many_arguments)]
pub fn gdn_recurrent_step(
    ctx: &CudaContext,
    state: &Tensor,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    decay: &Tensor,
    beta: &Tensor,
    output: &Tensor,
    k_heads: usize,
) -> Result<()> {
    let state_dims = state.shape().dims().to_vec();
    if state_dims.len() != 3 {
        return Err(invalid("GDN state must be [v_heads, v_dim, k_dim]"));
    }
    let (v_heads, v_dim, k_dim) = (state_dims[0], state_dims[1], state_dims[2]);
    if k_heads == 0 || v_heads % k_heads != 0 {
        return Err(invalid("GDN value heads must be a multiple of key heads"));
    }
    let state_buffer = tensor_storage(ctx, state, DType::F32, &state_dims)?;
    let q_buffer = tensor_storage(ctx, q, DType::BF16, &[k_heads, k_dim])?;
    let k_buffer = tensor_storage(ctx, k, DType::BF16, &[k_heads, k_dim])?;
    let v_buffer = tensor_storage(ctx, v, DType::BF16, &[v_heads, v_dim])?;
    let decay_buffer = tensor_storage(ctx, decay, DType::F32, &[v_heads])?;
    let beta_buffer = tensor_storage(ctx, beta, DType::F32, &[v_heads])?;
    let output_buffer = tensor_storage(ctx, output, DType::BF16, &[v_heads, v_dim])?;
    unsafe {
        status::check(abi::apxinf_gdn_recurrent_step(
            state_buffer.ptr(),
            q_buffer.ptr(),
            k_buffer.ptr(),
            v_buffer.ptr(),
            decay_buffer.ptr(),
            beta_buffer.ptr(),
            output_buffer.ptr(),
            v_heads as i64,
            k_heads as i64,
            v_dim as i64,
            k_dim as i64,
            ctx.stream().handle(),
        ))
    }
}

/// Per-head RMSNorm followed by the swish output gate.
pub fn gdn_gated_norm(
    ctx: &CudaContext,
    input: &Tensor,
    gate: &Tensor,
    weight: &Tensor,
    output: &Tensor,
    epsilon: f32,
) -> Result<()> {
    let dims = input.shape().dims().to_vec();
    if dims.len() != 2 {
        return Err(invalid("GDN gated norm expects [heads, head_dim]"));
    }
    let (heads, head_dim) = (dims[0], dims[1]);
    let input_buffer = tensor_storage(ctx, input, DType::BF16, &dims)?;
    let gate_buffer = tensor_storage(ctx, gate, DType::BF16, &dims)?;
    let weight_buffer = tensor_storage(ctx, weight, DType::BF16, &[head_dim])?;
    let output_buffer = tensor_storage(ctx, output, DType::BF16, &dims)?;
    unsafe {
        status::check(abi::apxinf_gdn_gated_norm(
            input_buffer.ptr(),
            gate_buffer.ptr(),
            weight_buffer.ptr(),
            output_buffer.ptr(),
            heads as i64,
            head_dim as i64,
            epsilon,
            ctx.stream().handle(),
        ))
    }
}

/// Advance the causal convolution by one token, then apply SiLU.
///
/// `window` is `[channels, kernel_width]` f32 holding the last
/// `kernel_width` inputs per channel.
pub fn gdn_causal_conv_step(
    ctx: &CudaContext,
    window: &Tensor,
    input: &Tensor,
    weight: &Tensor,
    output: &Tensor,
) -> Result<()> {
    let window_dims = window.shape().dims().to_vec();
    if window_dims.len() != 2 {
        return Err(invalid("GDN conv window must be [channels, kernel_width]"));
    }
    let (channels, kernel_width) = (window_dims[0], window_dims[1]);
    let window_buffer = tensor_storage(ctx, window, DType::F32, &window_dims)?;
    let input_buffer = tensor_storage(ctx, input, DType::BF16, &[channels])?;
    let weight_buffer = tensor_storage(ctx, weight, DType::BF16, &window_dims)?;
    let output_buffer = tensor_storage(ctx, output, DType::BF16, &[channels])?;
    unsafe {
        status::check(abi::apxinf_gdn_causal_conv_step(
            window_buffer.ptr(),
            input_buffer.ptr(),
            weight_buffer.ptr(),
            output_buffer.ptr(),
            channels as i64,
            kernel_width as i64,
            ctx.stream().handle(),
        ))
    }
}

/// L2-normalize each head in place.
pub fn gdn_l2_normalize_heads(ctx: &CudaContext, data: &Tensor, epsilon: f32) -> Result<()> {
    let dims = data.shape().dims().to_vec();
    if dims.len() != 2 {
        return Err(invalid("GDN head normalization expects [heads, head_dim]"));
    }
    let buffer = tensor_storage(ctx, data, DType::BF16, &dims)?;
    unsafe {
        status::check(abi::apxinf_gdn_l2_normalize_heads(
            buffer.ptr(),
            dims[0] as i64,
            dims[1] as i64,
            epsilon,
            ctx.stream().handle(),
        ))
    }
}

/// `decay = -exp(A_log) * softplus(a + dt_bias)`, `beta = sigmoid(b)`.
#[allow(clippy::too_many_arguments)]
pub fn gdn_decay_and_beta(
    ctx: &CudaContext,
    a: &Tensor,
    b: &Tensor,
    a_log: &Tensor,
    dt_bias: &Tensor,
    decay: &Tensor,
    beta: &Tensor,
) -> Result<()> {
    let dims = a.shape().dims().to_vec();
    if dims.len() != 1 {
        return Err(invalid("GDN gate inputs must be [heads]"));
    }
    let heads = dims[0];
    let a_buffer = tensor_storage(ctx, a, DType::BF16, &dims)?;
    let b_buffer = tensor_storage(ctx, b, DType::BF16, &dims)?;
    let a_log_buffer = tensor_storage(ctx, a_log, DType::BF16, &dims)?;
    let dt_bias_buffer = tensor_storage(ctx, dt_bias, DType::BF16, &dims)?;
    let decay_buffer = tensor_storage(ctx, decay, DType::F32, &dims)?;
    let beta_buffer = tensor_storage(ctx, beta, DType::F32, &dims)?;
    unsafe {
        status::check(abi::apxinf_gdn_decay_and_beta(
            a_buffer.ptr(),
            b_buffer.ptr(),
            a_log_buffer.ptr(),
            dt_bias_buffer.ptr(),
            decay_buffer.ptr(),
            beta_buffer.ptr(),
            heads as i64,
            ctx.stream().handle(),
        ))
    }
}

/// Chunked parallel scan over `seq_padded` tokens (prefill).
///
/// A faithful port of `torch_chunk_gated_delta_rule` (the forward-substitution
/// export path). Produces the same `core_attn_out` and leaves the recurrent
/// `state` in exactly the form the single-token `gdn_recurrent_step` expects,
/// so a prompt run through this scan and then decoded continues from an
/// identical state.
///
/// `q`/`k` are `[seq_padded, k_heads, k_dim]` BF16 (post-conv, pre-l2norm);
/// `v` is `[seq_padded, v_heads, k_dim]` BF16; `g`/`beta` are
/// `[seq_padded, v_heads]` f32. `out` is `[seq_padded, v_heads, k_dim]` BF16.
/// `state` is `[v_heads, v_dim, k_dim]` f32, updated in place. q/k are
/// L2-normalized in fp32 and q scaled by `k_dim**-0.5` inside the kernel, to
/// match the reference. `seq_padded` must be a multiple of `chunk_size`.
#[allow(clippy::too_many_arguments)]
pub fn gdn_chunk_scan(
    ctx: &CudaContext,
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    g: &Tensor,
    beta: &Tensor,
    out: &Tensor,
    state: &Tensor,
    v_heads: usize,
    k_heads: usize,
    chunk_size: usize,
) -> Result<()> {
    let state_dims = state.shape().dims().to_vec();
    if state_dims.len() != 3 {
        return Err(invalid("GDN state must be [v_heads, v_dim, k_dim]"));
    }
    let (sv_heads, v_dim, k_dim) = (state_dims[0], state_dims[1], state_dims[2]);
    if sv_heads != v_heads || v_dim != k_dim {
        return Err(invalid("GDN chunk scan expects square v_dim == k_dim state"));
    }
    if k_heads == 0 || v_heads % k_heads != 0 {
        return Err(invalid("GDN value heads must be a multiple of key heads"));
    }
    if chunk_size == 0 {
        return Err(invalid("GDN chunk size must be positive"));
    }
    let q_dims = q.shape().dims().to_vec();
    if q_dims.len() != 3 {
        return Err(invalid("GDN chunk scan q must be [seq, k_heads, k_dim]"));
    }
    let seq_padded = q_dims[0];
    if seq_padded % chunk_size != 0 {
        return Err(invalid("GDN chunk scan seq must be a multiple of chunk_size"));
    }
    let num_chunks = seq_padded / chunk_size;

    let q_buffer = tensor_storage(ctx, q, DType::BF16, &[seq_padded, k_heads, k_dim])?;
    let k_buffer = tensor_storage(ctx, k, DType::BF16, &[seq_padded, k_heads, k_dim])?;
    let v_buffer = tensor_storage(ctx, v, DType::BF16, &[seq_padded, v_heads, v_dim])?;
    let g_buffer = tensor_storage(ctx, g, DType::F32, &[seq_padded, v_heads])?;
    let beta_buffer = tensor_storage(ctx, beta, DType::F32, &[seq_padded, v_heads])?;
    let out_buffer = tensor_storage(ctx, out, DType::BF16, &[seq_padded, v_heads, v_dim])?;
    let state_buffer = tensor_storage(ctx, state, DType::F32, &state_dims)?;
    unsafe {
        status::check(abi::apxinf_gdn_chunk_scan(
            q_buffer.ptr(),
            k_buffer.ptr(),
            v_buffer.ptr(),
            g_buffer.ptr(),
            beta_buffer.ptr(),
            out_buffer.ptr(),
            state_buffer.ptr(),
            seq_padded as i64,
            v_heads as i64,
            k_heads as i64,
            chunk_size as i64,
            k_dim as i64,
            num_chunks as i64,
            ctx.stream().handle(),
        ))
    }
}
