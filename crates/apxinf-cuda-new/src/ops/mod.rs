//! Semantic CUDA APIs with selection and native state behind L1.

pub mod attention;
mod attn_ops;
pub mod cache;
pub mod elementwise;
pub mod embedding;
pub mod gather;
pub mod gdn;
pub mod gemm;
pub mod mlp;
pub mod norm;
pub mod pillow_bicubic;
pub mod pointwise;
pub mod preprocess;
pub mod quantization;
pub mod rope;
pub mod sampling;
pub(crate) mod validation;
mod volume;
pub use attn_ops::head_rms_norm;
mod model;
mod moe;
mod qwen35_attention;
mod qwen35_gdn_compat;
mod qwen35_mixers;
mod qwen35_mtp;

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
    apply_output_gate, split_query_and_gate, OutputGateArgs, OutputGateNumerics, OutputGateSource,
};
pub use attention::{
    attention, decode_attention, decode_attention_workspace_bytes, kv_cache_attention,
    segmented_attention, AttentionArgs, AttentionInput, AttentionMask, AttentionNumerics,
    AttentionPolicy, KvCacheAttentionArgs, KvCacheDecodeMeta, SegmentedAttentionArgs,
};
pub use cache::core::{append as cache_append, concat_rows, copy_tensor, reserve_prefix};
pub use elementwise::add_into;
pub use elementwise::core::{
    activation as elementwise_activation, add as elementwise_add, add_bias as elementwise_add_bias,
    mul as elementwise_mul, scale as elementwise_scale, ElementwiseActivation,
};
pub use gather::{
    embedding_gather, gather, EmbeddingScale, GatherArgs, GatherSemantic,
    PatchGeometry as GatherPatchGeometry,
};
pub use gdn::{
    gdn_causal_conv_forward, gdn_causal_conv_step, gdn_causal_conv_verify, gdn_chunk_scan,
    gdn_continue, gdn_conv_prepare, gdn_decay_and_beta, gdn_first_prefill, gdn_gated_norm,
    gdn_l2_normalize_heads, gdn_l2_normalize_heads_seq, gdn_prefill, gdn_prefill_workspace_bytes,
    gdn_prepare_prefill, gdn_recurrent_step, gdn_recurrent_verify, gdn_recurrent_verify_from_source, gdn_state_elements,
    GdnChunkScanArgs, GdnChunkScanInput, GdnContinuationArgs, GdnDecayAndBetaArgs,
    GdnFirstPrefillArgs, GdnGate, GdnGatedNormArgs, GdnQuantizedOutput,
};
pub use gemm::{
    gemm, gemm_bias, gemm_bias_gelu, gemm_bias_residual, gemm_geglu, gemm_gelu_exact, gemm_swiglu,
    gemv, gemv_bias, pack_swiglu_weight_host, BlockQuantizedSwiGluArgs, GemmArgs, GemmBiasArgs,
    GemmBiasGeluArgs, GemmBiasResidualArgs, GemmGegluArgs, GemmNumerics, GemmPolicy,
    GemmQuantization, GemmSwiGluArgs, GemmWeightLayout, GemvArgs, GemvBiasArgs, GemvBlockScales,
    RepackedWeight, WeightVersion,
};
pub use mlp::{
    block_quantized_gemv, quantize_activation_per_tensor, rms_norm, scaled_gemv, swiglu,
    swiglu_round_product,
};
pub use mlp::{block_quantized_weight_gemv, dense_gemv, swiglu_split};
pub use moe::{
    combine_reference as routed_moe_combine_reference, fused_routed_moe,
    fused_routed_moe_available, fused_routed_moe_workspace_size, prepare_expert_fc1,
    prepare_expert_scales, rescale_expert_weights, route_topk_reference, routed_moe_add_shared,
    routed_moe_combine, routed_moe_quantized_experts, routed_moe_route, routed_moe_route_topk,
    small_routed_moe, small_routed_moe_available, small_routed_moe_workspace_size,
    EdgeBf16DecodeMoeArgs, EdgeBf16DecodeMoeWeights, EdgeNvfp4MoeArgs, EdgeNvfp4MoeWeights,
    ExpertAssignment, Nvfp4ExpertProjection, RoutePlan, RoutedMoeAddSharedArgs,
    RoutedMoeCombineArgs, RoutedMoeConfig, RoutedMoeNvfp4ExpertsArgs, RoutedMoeRouteArgs,
    RoutedMoeRouteTopkArgs, RoutedMoeSemantic, SharedExpertOutput,
};
pub use norm::{
    channel_l2_norm, channel_l2_norm_silu, fused_add_channel_l2_norm_silu, fused_add_norm,
    residual_add, Affine, BiasOrder, ChannelL2NormArgs, FusedAddChannelL2NormArgs,
    FusedAddNormArgs, NormArgs, NormKind, NormNumerics, QuantOut, ResidualAddArgs, ResidualGate,
};
pub use pointwise::{
    pointwise, swiglu_into, PointwiseActivation, PointwiseArgs, PointwiseSemantic,
};
pub use preprocess::{
    transpose_pad_channels, vision_position, vision_position_staged, VisionPositionArgs,
};
pub use quantization::{
    block_quantize, block_scale_buffer_bytes, pack_block_scales, quantization, quantize_per_tensor,
    BlockQuantizationArgs, BlockQuantizationSemantic, PackBlockScalesArgs, QuantizationArgs,
    QuantizationSemantic, ScaleLayout,
};
pub use qwen35_attention::{
    decode_cache_element, encode_cache_element, qwen35_attention_draft_cache,
    qwen35_attention_reference, qwen35_attention_static_cache,
    qwen35_draft_attention_workspace_elements, qwen35_dynamic_decode_attention,
    qwen35_dynamic_decode_attention_gated, qwen35_dynamic_decode_max_splits,
    qwen35_dynamic_kv_append, qwen35_dynamic_kv_dequant, qwen35_dynamic_kv_prepare,
    qwen35_query_prepare, qwen35_static_kv_append, qwen35_suffix_fmha,
    Qwen35AttentionReferenceArgs, Qwen35AttentionReferenceOutput, Qwen35AttentionScratch,
    Qwen35DynamicDecodeScratch, Qwen35DynamicFp8KvCache, Qwen35Fp8AttentionArgs, Qwen35Fp8KvCache,
    Qwen35KvCacheScales, Qwen35StaticFp8KvAppendArgs, FP8_E4M3_MAX_FINITE,
    QWEN35_ATTENTION_HEAD_DIM, QWEN35_ATTENTION_KV_HEADS, QWEN35_ATTENTION_KV_WIDTH,
    QWEN35_ATTENTION_MAX_SPLITS, QWEN35_ATTENTION_Q_HEADS, QWEN35_ATTENTION_Q_PROJECTION_WIDTH,
    QWEN35_ATTENTION_RMS_EPSILON, QWEN35_ATTENTION_ROTARY_DIM, QWEN35_DECODE_SPLIT_TOKENS,
    QWEN35_KV_CACHE_BLOCK_TOKENS, QWEN35_SUFFIX_FMHA_AVAILABLE,
};
pub use qwen35_mixers::{
    qwen35_gdn_decode, qwen35_gdn_prefill, qwen35_gdn_verify_from_source, Qwen35GdnCakeScratch, Qwen35GdnCoreWeights,
    Qwen35GdnDecodeArgs, Qwen35GdnDecodeScratch, Qwen35GdnPrefillArgs, Qwen35GdnPrefillScratch,
    Qwen35GdnState, Qwen35GdnStateCheckpoints, QWEN35_GDN_CONV_WIDTH, QWEN35_GDN_EPSILON,
    QWEN35_GDN_HEAD_DIM, QWEN35_GDN_KEY_HEADS, QWEN35_GDN_QKV_WIDTH, QWEN35_GDN_Q_WIDTH,
    QWEN35_GDN_SCAN_CHUNK, QWEN35_GDN_VALUE_HEADS, QWEN35_GDN_VALUE_WIDTH,
};
pub use qwen35_mtp::{qwen35_mtp_accept, Qwen35MtpAcceptArgs, QWEN35_MTP_ACCEPT_OUTPUT_I32};
pub use qwen35_mtp::{
    qwen35_mtp_moe, qwen35_mtp_moe_reference, Qwen35MtpMoeArgs, Qwen35MtpMoeScratch,
    Qwen35MtpMoeState, Qwen35MtpMoeWeights, Qwen35MtpReferenceOutput, Qwen35MtpReferenceWeights,
    QWEN35_MTP_DRAFT_STEPS, QWEN35_MTP_EXPERTS, QWEN35_MTP_HIDDEN, QWEN35_MTP_INTERMEDIATE,
    QWEN35_MTP_LAYERS, QWEN35_MTP_TOP_K, QWEN35_MTP_VOCAB_SIZE,
};
pub use rope::apply_vision_2d as rope_apply_vision_2d;
pub use rope::{
    decode_rope, decode_rotation, execute_batched as rope_apply_batched,
    execute_mrope as rope_apply_mrope, multimodal_rope, partial_rope, rms_norm_apply_precomputed,
    rope, rotary_dim, split_grouped_qkv, split_qkv_axial_rope, split_qkv_cache,
    split_qkv_mrope_cache, AxialRopeArgs, AxialRopeNumerics, DecodeRopeArgs, DecodeRotationArgs,
    DecodeRotationPosition, GroupedQkvRopeArgs, MropeCacheArgs, MultimodalRopeArgs,
    QkvCacheRopeArgs, RopeArgs, RopeSemantic,
};
pub use sampling::{argmax, ArgmaxArgs};
pub use volume::{Conv3dArgs, Conv3dPolicy, PreparedConv3d, Unfold3dGeometry};

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

/// Session-aware scratch storage for direct-launch operators.
pub fn scratch_buffer(
    ctx: &crate::CudaContext,
    bytes: usize,
) -> apxinf_core::Result<crate::CudaBuffer> {
    crate::workspace::output_buffer(ctx, bytes)
}

/// Session-aware scratch storage that is explicitly cleared before use.
pub fn scratch_buffer_zeroed(
    ctx: &crate::CudaContext,
    bytes: usize,
) -> apxinf_core::Result<crate::CudaBuffer> {
    crate::workspace::output_buffer_zeroed(ctx, bytes)
}

/// Session-aware scratch storage with only each group's unused tail cleared.
pub fn scratch_buffer_tail_zeroed(
    ctx: &crate::CudaContext,
    groups: usize,
    group_bytes: usize,
    used_bytes: usize,
) -> apxinf_core::Result<crate::CudaBuffer> {
    crate::workspace::output_buffer_tail_zeroed(ctx, groups, group_bytes, used_bytes)
}
#[cfg(test)]
mod tests;

mod recurrence;
pub use recurrence::{F64RecurrenceState, PreparedRecurrence, RecurrenceCoefficients};

mod reduction;
pub use reduction::PreparedIndexedMean;
