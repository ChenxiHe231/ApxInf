//! L3 semantic adapters for fixed-shape kernels translated from the old AOT bundle.
//!
//! The public contracts are model-neutral. Hardware and production-shape
//! guards remain in the L1 launchers; a miss is either handled by the ordinary
//! cuda-new implementation here or reported as `None` when the surrounding
//! semantic is larger than this candidate (the fused GDN prefill case).

use apxinf_core::{Result, Tensor};

use crate::{CudaBuffer, CudaContext};

pub use crate::kernels::gemm::Bf16SwigluWeightLayout;

/// Select the resident BF16 gate/up layout accepted by both the translated
/// fixed-shape candidate and its variable-row fallback.
pub fn bf16_swiglu_weight_layout(
    ctx: &CudaContext,
    weight_elements: usize,
) -> Bf16SwigluWeightLayout {
    crate::kernels::gemm::bf16_swiglu_weight_layout(ctx, weight_elements)
}

/// Pack a checkpoint `[2N,K]` BF16 gate/up weight into the layout selected by
/// [`bf16_swiglu_weight_layout`].
pub fn pack_bf16_swiglu_weight_host(
    source: &[f32],
    full_n: usize,
    k: usize,
) -> Result<Vec<half::bf16>> {
    crate::kernels::gemm::pack_bf16_swiglu_weight_host(source, full_n, k)
}

/// BF16 gate/up projection followed by rounded SwiGLU.
///
/// The translated SM110 candidate may return padded physical rows so a
/// following projection can consume its preferred tile geometry. Callers must
/// use the input row count as the logical row count.
pub fn bf16_swiglu_projection(
    ctx: &CudaContext,
    input: &Tensor,
    weight: &Tensor,
) -> Result<Tensor> {
    crate::kernels::gemm::bf16_swiglu_checkpoint(ctx, input, weight)
}

/// Inputs for the fused first-prefill causal-convolution + gated-delta scan.
pub struct FusedGdnPrefillArgs<'a> {
    pub zba: &'a Tensor,
    pub conv_weight: &'a Tensor,
    pub new_conv_state: &'a Tensor,
    pub a_log: &'a Tensor,
    pub dt_bias: &'a Tensor,
    pub final_state: &'a CudaBuffer,
    /// `[key_heads, value_heads, key_head_dim, value_head_dim]`.
    pub geometry: [usize; 4],
}

/// Try the translated fused GDN prefill candidate.
///
/// `None` means that the device or geometry is outside the measured support
/// policy. The caller then composes the public causal-convolution and GDN scan
/// operators because those state transitions are part of its larger semantic.
pub fn fused_gdn_prefill_bf16(
    ctx: &CudaContext,
    args: FusedGdnPrefillArgs<'_>,
) -> Result<Option<Tensor>> {
    crate::kernels::linear_attention::try_causal_conv_gdn_prefill_bf16(
        ctx,
        args.zba,
        args.conv_weight,
        args.new_conv_state,
        args.a_log,
        args.dt_bias,
        args.final_state,
        args.geometry,
    )
}

/// Causal BF16 grouped-query attention over the leading `key_tokens` rows.
/// The fixed D256 SM110 provider is selected internally; all other contracts
/// use the ordinary cuda-new attention implementation.
pub fn causal_gqa_bf16(
    ctx: &CudaContext,
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    key_tokens: usize,
) -> Result<Tensor> {
    if let Some(output) = crate::kernels::attention::try_gqa_bf16_fa4_d256_sm110(
        ctx, query, key, value, key_tokens,
    )? {
        return Ok(output);
    }
    crate::kernels::attention::causal_gqa_bf16(ctx, query, key, value, key_tokens)
}

/// Non-causal BF16 grouped-query attention over the leading `key_tokens` rows.
/// The fixed D256 split-batch SM110 provider is selected internally.
pub fn noncausal_gqa_bf16(
    ctx: &CudaContext,
    query: &Tensor,
    key: &Tensor,
    value: &Tensor,
    key_tokens: usize,
) -> Result<Tensor> {
    if let Some(output) = crate::kernels::attention::try_gqa_bf16_fa4_d256_splitbatch_sm110(
        ctx, query, key, value, key_tokens,
    )? {
        return Ok(output);
    }
    crate::kernels::attention::noncausal_gqa_bf16(ctx, query, key, value, key_tokens)
}

/// Contract for split-QKV + axial RoPE + segmented vision attention.
pub struct VisionSegmentedAttentionArgs<'a> {
    pub qkv: &'a Tensor,
    pub position_ids: &'a CudaBuffer,
    pub heads: usize,
    pub head_dim: usize,
    pub theta: f32,
    pub offsets: &'a CudaBuffer,
    pub host_offsets: &'a [u32],
    pub segments: usize,
    pub max_tokens: usize,
}

/// Split QKV, apply axial RoPE to Q/K, then run non-causal attention within
/// each segment. The translated 624/2200-token SM110 provider and its support
/// policy are hidden behind this semantic interface.
pub fn vision_segmented_attention_bf16(
    ctx: &CudaContext,
    args: VisionSegmentedAttentionArgs<'_>,
) -> Result<Tensor> {
    if let Some(output) =
        crate::kernels::attention::try_vision_qkv_rope_segmented_fa4_skip_v(
            ctx,
            args.qkv,
            args.position_ids,
            args.heads,
            args.head_dim,
            args.theta,
            args.offsets,
            args.host_offsets,
            args.segments,
            true,
        )?
    {
        return Ok(output);
    }

    let split = crate::kernels::attention::split_vision_qkv_rope_bf16(
        ctx,
        args.qkv,
        None,
        args.position_ids,
        args.heads,
        args.head_dim,
        args.theta,
    )?;
    crate::kernels::attention::segmented_mha_bf16(
        ctx,
        &split.q,
        &split.k,
        &split.v,
        args.offsets,
        args.host_offsets,
        args.segments,
        args.max_tokens,
    )
}
