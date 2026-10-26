//! Qwen3.5-35B-A3B text-mixer adapters.
//!
//! The low-level Gated DeltaNet kernels are shared with other Qwen3.5-family
//! models.  This module fixes their otherwise generic shape parameters to the
//! exact 35B-A3B checkpoint geometry and defines the state hand-off between
//! prompt prefill and single-token decode.  Keeping that contract in one
//! place prevents a caller from accidentally running the 48-value-head
//! Qwen3.8 geometry against this checkpoint.
//!
//! Scope: these entry points begin after the four BF16 GDN projections
//! (`in_proj_qkv`, `in_proj_z`, `in_proj_a`, `in_proj_b`) and end before the
//! BF16 `out_proj`.  They are real device-resident mixer execution, not a
//! model stub.  The projection and residual orchestration belongs to the
//! model runtime.

use apxinf_core::{DType, Error, Result, Shape, Tensor};

use crate::ops::gemm::contracts::tensor_storage;
use crate::{transfers, CudaBuffer, CudaContext};

use super::{
    gdn_causal_conv_step, gdn_conv_prepare, gdn_decay_and_beta, gdn_decay_and_beta_seq,
    gdn_gated_norm, gdn_gated_norm_seq_f16, gdn_l2_normalize_heads, gdn_prefill,
    gdn_recurrent_step,
};

pub const QWEN35_GDN_KEY_HEADS: usize = 16;
pub const QWEN35_GDN_VALUE_HEADS: usize = 32;
pub const QWEN35_GDN_HEAD_DIM: usize = 128;
pub const QWEN35_GDN_CONV_WIDTH: usize = 4;
pub const QWEN35_GDN_Q_WIDTH: usize = QWEN35_GDN_KEY_HEADS * QWEN35_GDN_HEAD_DIM;
pub const QWEN35_GDN_VALUE_WIDTH: usize = QWEN35_GDN_VALUE_HEADS * QWEN35_GDN_HEAD_DIM;
pub const QWEN35_GDN_QKV_WIDTH: usize = 2 * QWEN35_GDN_Q_WIDTH + QWEN35_GDN_VALUE_WIDTH;
pub const QWEN35_GDN_SCAN_CHUNK: usize = 64;
pub const QWEN35_GDN_EPSILON: f32 = 1.0e-6;

/// Device state retained by every Qwen3.5 GDN layer.
///
/// `conv` is `[8192, 4]` F32, oldest sample first. `recurrent` is
/// `[32, 128, 128]` F32 in ApxInf's transposed `[value_head, value_dim,
/// key_dim]` layout.  The checkpoint/reference representation is
/// `[value_head, key_dim, value_dim]`; the two 128-wide axes are transposed at
/// import/export boundaries, never between prefill and decode.
pub struct Qwen35GdnState<'a> {
    pub conv: &'a Tensor,
    pub recurrent: &'a Tensor,
}

/// Non-projection weights consumed by the GDN core.
pub struct Qwen35GdnCoreWeights<'a> {
    /// Checkpoint-native BF16 `[8192, 1, 4]` depthwise-convolution weight.
    pub conv: &'a Tensor,
    pub a_log: &'a Tensor,
    pub dt_bias: &'a Tensor,
    /// Gated RMSNorm uses this weight directly.  Unlike the decoder's
    /// non-gated RMSNorm, it does not use Qwen3.5's `(1 + weight)` convention.
    pub norm: &'a Tensor,
}

/// Caller-owned scratch/output tensors for one decode token.
pub struct Qwen35GdnDecodeScratch<'a> {
    pub convolved_qkv: &'a Tensor,
    pub decay: &'a Tensor,
    pub beta: &'a Tensor,
    /// BF16 `[4096]`, flattened exactly as expected by `out_proj`.
    pub core_output: &'a Tensor,
    /// BF16 `[4096]`, flattened exactly as expected by `out_proj`.
    pub gated_output: &'a Tensor,
}

/// One-token GDN core invocation.
pub struct Qwen35GdnDecodeArgs<'a> {
    /// BF16 `[8192]`, output of `in_proj_qkv`.
    pub projected_qkv: &'a Tensor,
    /// BF16 `[4096]`, output of `in_proj_z`.
    pub z: &'a Tensor,
    /// BF16 `[32]`, output of `in_proj_a`.
    pub a: &'a Tensor,
    /// BF16 `[32]`, output of `in_proj_b`.
    pub b: &'a Tensor,
    pub weights: Qwen35GdnCoreWeights<'a>,
    pub state: Qwen35GdnState<'a>,
    pub scratch: Qwen35GdnDecodeScratch<'a>,
}

/// Caller-owned scratch/output tensors for a prompt.
pub struct Qwen35GdnPrefillScratch<'a> {
    pub convolved_qkv: &'a Tensor,
    pub decay: &'a Tensor,
    pub beta: &'a Tensor,
    /// BF16 `[tokens, 4096]`, flattened exactly as expected by `out_proj`.
    pub core_output: &'a Tensor,
    /// BF16 `[tokens, 4096]`, flattened exactly as expected by `out_proj`.
    pub gated_output: &'a Tensor,
}

/// FP16 buffers and side tables the FlashInfer Cake prefill kernel consumes.
///
/// Supplied together or not at all. `None` means the device predates Blackwell
/// and prefill must take the exact per-token path.
pub struct Qwen35GdnCakeScratch<'a> {
    /// F16 `[tokens, 16, 128]`, L2-normalized by the prepare op.
    pub q16: &'a Tensor,
    /// F16 `[tokens, 16, 128]`, L2-normalized by the prepare op.
    pub k16: &'a Tensor,
    /// F16 `[tokens, 32, 128]`.
    pub v16: &'a Tensor,
    /// F16 `[tokens, 32, 128]`, the kernel's readout.
    pub out16: &'a Tensor,
    /// F32 `[tokens, 32]`, linear-space decay the kernel reads as its gate.
    pub alpha: &'a Tensor,
    /// I32 `[2]`, filled with `[0, tokens]` per call.
    pub cu_seqlens: &'a Tensor,
    /// Scratch sized by `gdn_prefill_workspace_bytes`.
    pub workspace: &'a Tensor,
}

/// Prompt GDN core invocation.
pub struct Qwen35GdnPrefillArgs<'a> {
    /// BF16 `[tokens, 8192]`, output of `in_proj_qkv`.
    pub projected_qkv: &'a Tensor,
    /// BF16 `[tokens, 4096]`, output of `in_proj_z`.
    pub z: &'a Tensor,
    /// BF16 `[tokens, 32]`, output of `in_proj_a`.
    pub a: &'a Tensor,
    /// BF16 `[tokens, 32]`, output of `in_proj_b`.
    pub b: &'a Tensor,
    pub weights: Qwen35GdnCoreWeights<'a>,
    pub state: Qwen35GdnState<'a>,
    pub scratch: Qwen35GdnPrefillScratch<'a>,
    /// Present on Blackwell, where the chunked prefill kernel exists.
    pub cake: Option<Qwen35GdnCakeScratch<'a>>,
}

/// Advance one Qwen3.5 GDN layer by one token.
///
/// Every tensor's device, dtype, shape, and backing range are validated before
/// the first state mutation.  On success the convolution and recurrent states
/// both represent the supplied token, and `gated_output` is the BF16 `[4096]`
/// input to `out_proj`.
pub fn qwen35_gdn_decode(ctx: &CudaContext, args: &Qwen35GdnDecodeArgs<'_>) -> Result<()> {
    validate_decode(ctx, args)?;
    run_decode(
        ctx,
        args.projected_qkv,
        args.z,
        args.a,
        args.b,
        &args.weights,
        &args.state,
        &args.scratch,
    )
}

/// Run a new prompt through one Qwen3.5 GDN layer.
///
/// The caller must provide zero-initialized state for a new sequence.  A prompt
/// whose length is a multiple of 64 takes the FlashInfer Cake kernel when the
/// device and its FP16 scratch are both present; every other case takes the
/// exact recurrent path rather than padding, because padding would corrupt the
/// causal-convolution window and therefore the first decoded token.  Both paths
/// leave byte-compatible state for `qwen35_gdn_decode`.
///
/// The hand-written `gdn_chunk_scan` is deliberately *not* reachable from here.
/// Its single launch grows with the sequence -- measured 48 ms for one layer at
/// 2,048 tokens -- which on Thor's shared-SoC GPU trips channel recovery and
/// resets the whole board. It survives only as the test oracle this path is
/// checked against; see `ops::gdn_chunk_scan`.
///
/// This API deliberately does not expose an attention KV cache: GDN layers
/// carry the two states above, while only the model's ten full-attention
/// layers own FP8 K/V state.
pub fn qwen35_gdn_prefill(
    ctx: &CudaContext,
    tokens: usize,
    args: &Qwen35GdnPrefillArgs<'_>,
) -> Result<()> {
    validate_prefill(ctx, tokens, args)?;

    if let (0, Some(cake)) = (tokens % QWEN35_GDN_SCAN_CHUNK, args.cake.as_ref()) {
        gdn_decay_and_beta_seq(
            ctx,
            args.a,
            args.b,
            args.weights.a_log,
            args.weights.dt_bias,
            args.scratch.decay,
            args.scratch.beta,
            tokens,
            QWEN35_GDN_VALUE_HEADS,
        )?;
        // The convolution is fused into the prepare op, which also writes the
        // sliding window back into `state.conv` so decode continues from it.
        let conv_weight = conv_weight_view(args.weights.conv)?;
        gdn_conv_prepare(
            ctx,
            args.projected_qkv,
            &conv_weight,
            args.state.conv,
            cake.q16,
            cake.k16,
            cake.v16,
            args.scratch.decay,
            cake.alpha,
            tokens,
            QWEN35_GDN_KEY_HEADS,
            QWEN35_GDN_VALUE_HEADS,
            QWEN35_GDN_EPSILON,
        )?;
        let seqlens = Tensor::from_i32(
            vec![2],
            &[
                0,
                i32::try_from(tokens).map_err(|_| invalid("Qwen3.5 GDN prompt exceeds i32"))?,
            ],
        )?;
        transfers::copy_cpu_to_cuda(&seqlens, cake.cu_seqlens)?;
        gdn_prefill(
            ctx,
            cake.q16,
            cake.k16,
            cake.v16,
            cake.out16,
            // The gate is the linear-space alpha the prepare op emits, not the
            // log-space `decay` that feeds the reference scan.
            cake.alpha,
            args.scratch.beta,
            cake.cu_seqlens,
            args.state.recurrent,
            cake.workspace,
            tokens,
            QWEN35_GDN_KEY_HEADS,
            QWEN35_GDN_VALUE_HEADS,
            1,
            1.0 / (QWEN35_GDN_HEAD_DIM as f32).sqrt(),
        )?;
        let z = element_view(
            args.z,
            0,
            vec![tokens, QWEN35_GDN_VALUE_HEADS, QWEN35_GDN_HEAD_DIM],
            DType::BF16,
        )?;
        let gated_output = element_view(
            args.scratch.gated_output,
            0,
            vec![tokens, QWEN35_GDN_VALUE_HEADS, QWEN35_GDN_HEAD_DIM],
            DType::BF16,
        )?;
        return gdn_gated_norm_seq_f16(
            ctx,
            cake.out16,
            &z,
            args.weights.norm,
            &gated_output,
            tokens,
            QWEN35_GDN_VALUE_HEADS,
            QWEN35_GDN_HEAD_DIM,
            QWEN35_GDN_EPSILON,
        );
    }

    // Exact correctness path for a partial scan chunk.  It is intentionally
    // slower, but does not invent padding tokens or damage the decode state.
    for token in 0..tokens {
        let scratch = Qwen35GdnDecodeScratch {
            convolved_qkv: &row_view(
                args.scratch.convolved_qkv,
                token,
                QWEN35_GDN_QKV_WIDTH,
                vec![QWEN35_GDN_QKV_WIDTH],
                DType::BF16,
            )?,
            decay: &row_view(
                args.scratch.decay,
                token,
                QWEN35_GDN_VALUE_HEADS,
                vec![QWEN35_GDN_VALUE_HEADS],
                DType::F32,
            )?,
            beta: &row_view(
                args.scratch.beta,
                token,
                QWEN35_GDN_VALUE_HEADS,
                vec![QWEN35_GDN_VALUE_HEADS],
                DType::F32,
            )?,
            core_output: &row_view(
                args.scratch.core_output,
                token,
                QWEN35_GDN_VALUE_WIDTH,
                vec![QWEN35_GDN_VALUE_WIDTH],
                DType::BF16,
            )?,
            gated_output: &row_view(
                args.scratch.gated_output,
                token,
                QWEN35_GDN_VALUE_WIDTH,
                vec![QWEN35_GDN_VALUE_WIDTH],
                DType::BF16,
            )?,
        };
        run_decode(
            ctx,
            &row_view(
                args.projected_qkv,
                token,
                QWEN35_GDN_QKV_WIDTH,
                vec![QWEN35_GDN_QKV_WIDTH],
                DType::BF16,
            )?,
            &row_view(
                args.z,
                token,
                QWEN35_GDN_VALUE_WIDTH,
                vec![QWEN35_GDN_VALUE_WIDTH],
                DType::BF16,
            )?,
            &row_view(
                args.a,
                token,
                QWEN35_GDN_VALUE_HEADS,
                vec![QWEN35_GDN_VALUE_HEADS],
                DType::BF16,
            )?,
            &row_view(
                args.b,
                token,
                QWEN35_GDN_VALUE_HEADS,
                vec![QWEN35_GDN_VALUE_HEADS],
                DType::BF16,
            )?,
            &args.weights,
            &args.state,
            &scratch,
        )?;
    }
    Ok(())
}

fn run_decode(
    ctx: &CudaContext,
    projected_qkv: &Tensor,
    z: &Tensor,
    a: &Tensor,
    b: &Tensor,
    weights: &Qwen35GdnCoreWeights<'_>,
    state: &Qwen35GdnState<'_>,
    scratch: &Qwen35GdnDecodeScratch<'_>,
) -> Result<()> {
    let conv_weight = conv_weight_view(weights.conv)?;
    gdn_causal_conv_step(
        ctx,
        state.conv,
        projected_qkv,
        &conv_weight,
        scratch.convolved_qkv,
    )?;

    let q = element_view(
        scratch.convolved_qkv,
        0,
        vec![QWEN35_GDN_KEY_HEADS, QWEN35_GDN_HEAD_DIM],
        DType::BF16,
    )?;
    let k = element_view(
        scratch.convolved_qkv,
        QWEN35_GDN_Q_WIDTH,
        vec![QWEN35_GDN_KEY_HEADS, QWEN35_GDN_HEAD_DIM],
        DType::BF16,
    )?;
    let v = element_view(
        scratch.convolved_qkv,
        2 * QWEN35_GDN_Q_WIDTH,
        vec![QWEN35_GDN_VALUE_HEADS, QWEN35_GDN_HEAD_DIM],
        DType::BF16,
    )?;
    gdn_l2_normalize_heads(ctx, &q, QWEN35_GDN_EPSILON)?;
    gdn_l2_normalize_heads(ctx, &k, QWEN35_GDN_EPSILON)?;
    gdn_decay_and_beta(
        ctx,
        a,
        b,
        weights.a_log,
        weights.dt_bias,
        scratch.decay,
        scratch.beta,
    )?;
    let core_output = element_view(
        scratch.core_output,
        0,
        vec![QWEN35_GDN_VALUE_HEADS, QWEN35_GDN_HEAD_DIM],
        DType::BF16,
    )?;
    gdn_recurrent_step(
        ctx,
        state.recurrent,
        &q,
        &k,
        &v,
        scratch.decay,
        scratch.beta,
        &core_output,
        QWEN35_GDN_KEY_HEADS,
    )?;
    let z = element_view(
        z,
        0,
        vec![QWEN35_GDN_VALUE_HEADS, QWEN35_GDN_HEAD_DIM],
        DType::BF16,
    )?;
    let gated_output = element_view(
        scratch.gated_output,
        0,
        vec![QWEN35_GDN_VALUE_HEADS, QWEN35_GDN_HEAD_DIM],
        DType::BF16,
    )?;
    gdn_gated_norm(
        ctx,
        &core_output,
        &z,
        weights.norm,
        &gated_output,
        QWEN35_GDN_EPSILON,
    )
}

fn validate_decode(ctx: &CudaContext, args: &Qwen35GdnDecodeArgs<'_>) -> Result<()> {
    validate_common(
        ctx,
        &args.weights,
        &args.state,
        args.projected_qkv,
        &[QWEN35_GDN_QKV_WIDTH],
        args.z,
        &[QWEN35_GDN_VALUE_WIDTH],
        args.a,
        &[QWEN35_GDN_VALUE_HEADS],
        args.b,
        &[QWEN35_GDN_VALUE_HEADS],
        args.scratch.convolved_qkv,
        &[QWEN35_GDN_QKV_WIDTH],
        args.scratch.decay,
        &[QWEN35_GDN_VALUE_HEADS],
        args.scratch.beta,
        &[QWEN35_GDN_VALUE_HEADS],
        args.scratch.core_output,
        &[QWEN35_GDN_VALUE_WIDTH],
        args.scratch.gated_output,
        &[QWEN35_GDN_VALUE_WIDTH],
    )
}

fn validate_prefill(
    ctx: &CudaContext,
    tokens: usize,
    args: &Qwen35GdnPrefillArgs<'_>,
) -> Result<()> {
    if tokens == 0 || tokens > i32::MAX as usize {
        return Err(invalid("Qwen3.5 GDN prefill requires a positive i32 token count"));
    }
    validate_common(
        ctx,
        &args.weights,
        &args.state,
        args.projected_qkv,
        &[tokens, QWEN35_GDN_QKV_WIDTH],
        args.z,
        &[tokens, QWEN35_GDN_VALUE_WIDTH],
        args.a,
        &[tokens, QWEN35_GDN_VALUE_HEADS],
        args.b,
        &[tokens, QWEN35_GDN_VALUE_HEADS],
        args.scratch.convolved_qkv,
        &[tokens, QWEN35_GDN_QKV_WIDTH],
        args.scratch.decay,
        &[tokens, QWEN35_GDN_VALUE_HEADS],
        args.scratch.beta,
        &[tokens, QWEN35_GDN_VALUE_HEADS],
        args.scratch.core_output,
        &[tokens, QWEN35_GDN_VALUE_WIDTH],
        args.scratch.gated_output,
        &[tokens, QWEN35_GDN_VALUE_WIDTH],
    )?;
    // The Cake buffers are only read on the chunk-aligned fast path, but
    // validating them up front keeps the "nothing mutates before every tensor
    // is checked" contract that the per-token path also relies on.
    if let Some(cake) = args.cake.as_ref() {
        let key_dims = [tokens, QWEN35_GDN_KEY_HEADS, QWEN35_GDN_HEAD_DIM];
        let value_dims = [tokens, QWEN35_GDN_VALUE_HEADS, QWEN35_GDN_HEAD_DIM];
        for (tensor, dtype, shape) in [
            (cake.q16, DType::F16, &key_dims[..]),
            (cake.k16, DType::F16, &key_dims[..]),
            (cake.v16, DType::F16, &value_dims[..]),
            (cake.out16, DType::F16, &value_dims[..]),
            (cake.alpha, DType::F32, &[tokens, QWEN35_GDN_VALUE_HEADS][..]),
            (cake.cu_seqlens, DType::I32, &[2][..]),
        ] {
            tensor_storage(ctx, tensor, dtype, shape)?;
        }
        if cake.workspace.device() != apxinf_core::Device::Cuda(ctx.device_id()) {
            return Err(invalid("Qwen3.5 GDN Cake workspace is on another device"));
        }
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn validate_common(
    ctx: &CudaContext,
    weights: &Qwen35GdnCoreWeights<'_>,
    state: &Qwen35GdnState<'_>,
    projected_qkv: &Tensor,
    projected_qkv_shape: &[usize],
    z: &Tensor,
    z_shape: &[usize],
    a: &Tensor,
    a_shape: &[usize],
    b: &Tensor,
    b_shape: &[usize],
    convolved_qkv: &Tensor,
    convolved_qkv_shape: &[usize],
    decay: &Tensor,
    decay_shape: &[usize],
    beta: &Tensor,
    beta_shape: &[usize],
    core_output: &Tensor,
    core_output_shape: &[usize],
    gated_output: &Tensor,
    gated_output_shape: &[usize],
) -> Result<()> {
    for (tensor, dtype, shape) in [
        (projected_qkv, DType::BF16, projected_qkv_shape),
        (z, DType::BF16, z_shape),
        (a, DType::BF16, a_shape),
        (b, DType::BF16, b_shape),
        (
            weights.conv,
            DType::BF16,
            &[QWEN35_GDN_QKV_WIDTH, 1, QWEN35_GDN_CONV_WIDTH],
        ),
        (weights.a_log, DType::BF16, &[QWEN35_GDN_VALUE_HEADS]),
        (weights.dt_bias, DType::BF16, &[QWEN35_GDN_VALUE_HEADS]),
        (weights.norm, DType::BF16, &[QWEN35_GDN_HEAD_DIM]),
        (state.conv, DType::F32, &[QWEN35_GDN_QKV_WIDTH, QWEN35_GDN_CONV_WIDTH]),
        (
            state.recurrent,
            DType::F32,
            &[QWEN35_GDN_VALUE_HEADS, QWEN35_GDN_HEAD_DIM, QWEN35_GDN_HEAD_DIM],
        ),
        (convolved_qkv, DType::BF16, convolved_qkv_shape),
        (decay, DType::F32, decay_shape),
        (beta, DType::F32, beta_shape),
        (core_output, DType::BF16, core_output_shape),
        (gated_output, DType::BF16, gated_output_shape),
    ] {
        tensor_storage(ctx, tensor, dtype, shape)?;
    }
    Ok(())
}

fn row_view(
    tensor: &Tensor,
    row: usize,
    row_elements: usize,
    shape: Vec<usize>,
    dtype: DType,
) -> Result<Tensor> {
    element_view(tensor, row * row_elements, shape, dtype)
}

fn conv_weight_view(tensor: &Tensor) -> Result<Tensor> {
    element_view(
        tensor,
        0,
        vec![QWEN35_GDN_QKV_WIDTH, QWEN35_GDN_CONV_WIDTH],
        DType::BF16,
    )
}

fn element_view(
    tensor: &Tensor,
    element_offset: usize,
    shape: Vec<usize>,
    dtype: DType,
) -> Result<Tensor> {
    let element = dtype.size_in_bytes();
    let bytes = shape
        .iter()
        .try_fold(element, |bytes, dim| bytes.checked_mul(*dim))
        .ok_or_else(|| invalid("Qwen3.5 GDN tensor view size overflow"))?;
    let offset = element_offset
        .checked_mul(element)
        .ok_or_else(|| invalid("Qwen3.5 GDN tensor view offset overflow"))?;
    CudaBuffer::from_tensor(tensor)
        .map_err(Error::Cuda)?
        .view(offset, bytes)
        .map_err(Error::Cuda)?
        .as_tensor(Shape::new(shape), dtype)
        .map_err(Error::Cuda)
}

fn invalid(message: impl Into<String>) -> Error {
    Error::Other(message.into())
}

