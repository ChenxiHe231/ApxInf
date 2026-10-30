//! Semantic CUDA APIs with selection and native state behind L1.

mod attention;
mod attn_ops;
mod cache;
mod gather;
mod gdn;
mod gemm;
pub mod mlp;
mod model;
mod moe;
mod norm;
mod pointwise;
mod quantization;
mod qwen35_attention;
mod qwen35_mixers;
mod qwen35_mtp;
mod rope;

// Keep these crate-private aliases while graph/workspace and unit tests still
// refer to the GEMM implementation through `crate::ops`.
#[cfg(test)]
pub(crate) use attention::{
    contracts as attention_contracts, execution as attention_execution,
    normalize_kv_cache_attention, normalize_segmented_attention,
};
#[cfg(test)]
pub(crate) use gemm::contracts;
#[cfg(test)]
pub(crate) use gemm::gemm_execution as execution;

pub use crate::workspace::{ExecutionSession, GraphWorkspace};
pub use attention::{
    attention, decode_attention, decode_attention_workspace_bytes, kv_cache_attention,
    packed_qkv_attention, segmented_attention, AttentionArgs, AttentionMask, AttentionPolicy,
    KvCacheAttentionArgs, KvCacheDecodeMeta, PackedQkvAttentionArgs, SegmentedAttentionArgs,
};
pub use attn_ops::{
    apply_output_gate, head_rms_norm, partial_rope, rotary_dim, split_query_and_gate,
};
pub use cache::{concat_rows, copy_tensor, reserve_prefix};
pub use gather::{gather, GatherArgs, GatherSemantic, PatchGeometry as GatherPatchGeometry};
pub use gdn::{
    gdn_causal_conv_forward, gdn_causal_conv_step, gdn_chunk_scan, gdn_chunk_scan_interleaved,
    gdn_conv_prepare, gdn_decay_and_beta, gdn_decay_and_beta_seq, gdn_gated_norm,
    gdn_gated_norm_quantize, gdn_gated_norm_seq, gdn_gated_norm_seq_f16, gdn_l2_normalize_heads,
    gdn_prefill, gdn_prefill_workspace_bytes, gdn_prepare_prefill, gdn_recurrent_step,
    gdn_state_elements, gdn_widen_f16_to_bf16,
};
pub use gemm::{
    gemm, gemm_bias, gemm_bias_gelu, gemm_bias_residual, gemm_geglu, nvfp4_gemm_swiglu,
    nvfp4_pack_block_scales, nvfp4_quantize_activation, nvfp4_quantize_rms_norm,
    nvfp4_quantize_swiglu, nvfp4_scale_buffer_bytes, GemmArgs, GemmBiasArgs, GemmBiasGeluArgs,
    GemmBiasResidualArgs, GemmGegluArgs, GemmPolicy, GemmQuantization, ScaleLayout, WeightVersion,
};
pub use mlp::{
    add_into, bf16_gemv, fp8_gemv, nvfp4_a16_blackwell_gemv, nvfp4_gemv, quantize_fp8_per_tensor,
    swiglu, swiglu_round_product, swiglu_split,
};
pub use model::{argmax, embedding_gather, vision_position, VisionPositionArgs};
pub use moe::{
    combine_reference as routed_moe_combine_reference, edge_bf16_decode_moe,
    edge_bf16_decode_moe_available, edge_bf16_decode_moe_workspace_size, edge_nvfp4_interleave_fc1,
    edge_nvfp4_moe, edge_nvfp4_moe_available, edge_nvfp4_moe_workspace_size,
    edge_nvfp4_rescale_e4m3, edge_nvfp4_swizzle_scales, route_topk_reference,
    routed_moe_add_shared, routed_moe_combine, routed_moe_nvfp4_experts, routed_moe_route,
    routed_moe_route_topk, EdgeBf16DecodeMoeArgs, EdgeBf16DecodeMoeWeights, EdgeNvfp4MoeArgs,
    EdgeNvfp4MoeWeights, ExpertAssignment, Nvfp4ExpertProjection, RoutePlan,
    RoutedMoeAddSharedArgs, RoutedMoeCombineArgs, RoutedMoeConfig, RoutedMoeNvfp4ExpertsArgs,
    RoutedMoeRouteArgs, RoutedMoeRouteTopkArgs, RoutedMoeSemantic, SharedExpertOutput,
};
pub use norm::{
    ada_gate_residual, ada_gate_residual_rms_norm, adaptive_rms_norm, bias_residual,
    bias_residual_layer_norm, bias_residual_rms_norm, bias_then_residual, layer_norm, rms_norm,
    AdaGateResidualArgs, AdaGateResidualRmsNormArgs, AdaptiveRmsNormArgs, BiasResidualArgs,
    BiasResidualLayerNormArgs, BiasResidualRmsNormArgs, BiasThenResidualArgs, LayerNormArgs,
    RmsNormArgs,
};
pub use pointwise::{pointwise, PointwiseActivation, PointwiseArgs, PointwiseSemantic};
pub use quantization::{quantization, QuantizationArgs, QuantizationSemantic};
pub use qwen35_attention::{
    decode_e4m3, encode_e4m3, qwen35_attention_fp8, qwen35_attention_reference,
    Qwen35AttentionReferenceArgs, Qwen35AttentionReferenceOutput, Qwen35AttentionScratch,
    Qwen35Fp8AttentionArgs, Qwen35Fp8KvCache, Qwen35KvCacheScales, FP8_E4M3_MAX_FINITE,
    QWEN35_ATTENTION_HEAD_DIM, QWEN35_ATTENTION_KV_HEADS, QWEN35_ATTENTION_KV_WIDTH,
    QWEN35_ATTENTION_MAX_SPLITS, QWEN35_ATTENTION_Q_HEADS,
    QWEN35_ATTENTION_Q_PROJECTION_WIDTH, QWEN35_ATTENTION_RMS_EPSILON,
    QWEN35_ATTENTION_ROTARY_DIM, QWEN35_KV_CACHE_BLOCK_TOKENS,
};
pub use qwen35_mixers::{
    qwen35_gdn_decode, qwen35_gdn_prefill, Qwen35GdnCakeScratch, Qwen35GdnCoreWeights,
    Qwen35GdnStateCheckpoints,
    Qwen35GdnDecodeArgs, Qwen35GdnDecodeScratch, Qwen35GdnPrefillArgs, Qwen35GdnPrefillScratch,
    Qwen35GdnState, QWEN35_GDN_CONV_WIDTH, QWEN35_GDN_EPSILON, QWEN35_GDN_HEAD_DIM,
    QWEN35_GDN_KEY_HEADS, QWEN35_GDN_QKV_WIDTH, QWEN35_GDN_Q_WIDTH, QWEN35_GDN_SCAN_CHUNK,
    QWEN35_GDN_VALUE_HEADS, QWEN35_GDN_VALUE_WIDTH,
};
pub use qwen35_mtp::{
    qwen35_mtp_moe, qwen35_mtp_moe_reference, Qwen35MtpMoeArgs, Qwen35MtpMoeScratch,
    Qwen35MtpMoeState, Qwen35MtpMoeWeights, Qwen35MtpReferenceOutput, Qwen35MtpReferenceWeights,
    QWEN35_MTP_DRAFT_STEPS, QWEN35_MTP_EXPERTS, QWEN35_MTP_HIDDEN, QWEN35_MTP_INTERMEDIATE,
    QWEN35_MTP_LAYERS, QWEN35_MTP_TOP_K, QWEN35_MTP_VOCAB_SIZE,
};
pub use rope::{
    decode_rope, multimodal_rope, rope, split_qkv_axial_rope, AxialRopeArgs, DecodeRopeArgs,
    MultimodalRopeArgs, RopeArgs, RopeSemantic,
};

/// Run a fixed-shape forward pass that may tune and create native executions.
pub fn prepare_with_session<T>(
    session: &ExecutionSession,
    operation: impl FnOnce() -> apxinf_core::Result<T>,
) -> apxinf_core::Result<T> {
    crate::workspace::prepare_with_session(session, operation)
}

/// Run the same prepared forward pass without allocating or tuning.
/// Enqueues are asynchronous; synchronize at the outer execution boundary.
pub fn with_session<T>(
    session: &ExecutionSession,
    operation: impl FnOnce() -> apxinf_core::Result<T>,
) -> apxinf_core::Result<T> {
    crate::workspace::with_session(session, operation)
}
#[cfg(test)]
mod tests;
