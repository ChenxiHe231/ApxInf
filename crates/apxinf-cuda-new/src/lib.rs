pub mod buffer;
pub mod context;
pub mod device_caps;
mod ffi;
mod graph;
pub mod phase;
pub mod sampling;
pub mod stream;
pub mod transfers;
mod workspace;

pub use buffer::{CudaBuffer, CudaDeviceAddress, HostMappedBuffer};
pub use context::CudaContext;
pub use device_caps::{CudaArchFamily, CudaDeviceCaps};
pub use graph::{capture, CapturedGraph};
pub use ops::{
    ada_gate_residual, ada_gate_residual_rms_norm, adaptive_rms_norm, attention, bias_residual,
    bias_residual_layer_norm, bias_residual_rms_norm, bias_then_residual, concat_rows, decode_rope,
    gather, kv_cache_attention, layer_norm, multimodal_rope, packed_qkv_attention, pointwise,
    quantization, reserve_prefix, rms_norm, rope, route_topk_reference, routed_moe_combine,
    routed_moe_combine_reference, routed_moe_nvfp4_experts, routed_moe_route, segmented_attention,
    split_qkv_axial_rope, AdaGateResidualArgs, AdaGateResidualRmsNormArgs, AdaptiveRmsNormArgs,
    AttentionArgs, AttentionMask, AttentionPolicy, AxialRopeArgs, BiasResidualArgs,
    BiasResidualLayerNormArgs, BiasResidualRmsNormArgs, BiasThenResidualArgs, DecodeRopeArgs,
    ExecutionSession, ExpertAssignment, GatherArgs, GatherPatchGeometry, GatherSemantic,
    GraphWorkspace, KvCacheAttentionArgs, KvCacheDecodeMeta, LayerNormArgs, MultimodalRopeArgs,
    Nvfp4ExpertProjection, PackedQkvAttentionArgs, PointwiseActivation, PointwiseArgs,
    PointwiseSemantic, QuantizationArgs, QuantizationSemantic, RmsNormArgs, RopeArgs, RopeSemantic,
    RoutePlan, RoutedMoeCombineArgs, RoutedMoeConfig, RoutedMoeNvfp4ExpertsArgs,
    RoutedMoeRouteArgs, RoutedMoeSemantic, SegmentedAttentionArgs, SharedExpertOutput,
    decode_e4m3, encode_e4m3, qwen35_attention_fp8, qwen35_attention_reference,
    Qwen35AttentionReferenceArgs, Qwen35AttentionReferenceOutput, Qwen35AttentionScratch,
    Qwen35Fp8AttentionArgs, Qwen35Fp8KvCache, Qwen35KvCacheScales,
    FP8_E4M3_MAX_FINITE, QWEN35_ATTENTION_HEAD_DIM, QWEN35_ATTENTION_KV_HEADS,
    QWEN35_ATTENTION_KV_WIDTH, QWEN35_ATTENTION_Q_HEADS,
    QWEN35_ATTENTION_Q_PROJECTION_WIDTH, QWEN35_ATTENTION_RMS_EPSILON,
    QWEN35_ATTENTION_ROTARY_DIM,
    qwen35_mtp_moe, qwen35_mtp_moe_reference, Qwen35MtpMoeArgs, Qwen35MtpMoeScratch,
    Qwen35MtpMoeState, Qwen35MtpMoeWeights, Qwen35MtpReferenceOutput,
    Qwen35MtpReferenceWeights, QWEN35_MTP_DRAFT_STEPS, QWEN35_MTP_EXPERTS,
    QWEN35_MTP_HIDDEN, QWEN35_MTP_INTERMEDIATE, QWEN35_MTP_LAYERS,
    QWEN35_MTP_TOP_K, QWEN35_MTP_VOCAB_SIZE,
};
pub use phase::PreparedPhase;
pub use stream::CudaStream;

pub mod ops;

/// Whether the SM110 native NVFP4 GEMM+SwiGLU implementation is available.
///
/// The name describes the mathematical operator rather than its current model
/// consumer. Unsupported targets use the unfused GEMM + SwiGLU fallback.
pub const NVFP4_GEMM_SWIGLU: bool = cfg!(apxinf_nvfp4_gemm_swiglu);

/// Whether this build compiled the allocation-free FA2 decode kernel
/// (`apxinf_decode_attention`). When `false` that ABI returns
/// `APXINF_STATUS_UNSUPPORTED`, so callers must use the generic attention path.
///
/// Like [`NVFP4_GEMM_SWIGLU`], this is a zero-cost compile-time constant
/// carrying a build-time cfg across the crate boundary.
pub const FA2_DECODE: bool = cfg!(apxinf_fa2_decode);
