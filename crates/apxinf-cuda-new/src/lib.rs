pub mod backend;
pub mod buffer;
pub mod context;
pub mod cublas;
pub mod device_caps;
mod ffi;
mod graph;
mod kv_cache;
pub mod nvtx;
pub mod phase;
pub mod profiler;
pub mod runtime;
pub mod sampling;
pub mod stream;
pub mod transfers;
mod workspace;

#[cfg(test)]
pub(crate) static CUDA_TEST_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

pub use backend::CudaNewBackend;
pub use buffer::{CudaBuffer, CudaDeviceAddress, HostMappedBuffer};
pub use context::CudaContext;
pub use cublas::{CublasHandle, CublasTranspose};
pub use device_caps::{CudaArchFamily, CudaDeviceCaps};
pub use graph::{capture, CapturedGraph};
pub use kv_cache::CudaKvCache;
pub use ops::norm::norm;
pub use ops::{
    attention, channel_l2_norm, channel_l2_norm_silu, concat_rows, decode_cache_element,
    decode_rope, decode_rotation, encode_cache_element, fused_add_channel_l2_norm_silu,
    fused_add_norm, fused_routed_moe, fused_routed_moe_available, fused_routed_moe_workspace_size,
    gather, kv_cache_attention, multimodal_rope, pointwise, prepare_expert_fc1,
    prepare_expert_scales, quantization, qwen35_attention_reference, qwen35_attention_static_cache,
    qwen35_mtp_moe, qwen35_mtp_moe_reference, rescale_expert_weights, reserve_prefix, residual_add,
    rope, route_topk_reference, routed_moe_add_shared, routed_moe_combine,
    routed_moe_combine_reference, routed_moe_quantized_experts, routed_moe_route,
    routed_moe_route_topk, segmented_attention, small_routed_moe, small_routed_moe_available,
    small_routed_moe_workspace_size, split_grouped_qkv, split_qkv_axial_rope, split_qkv_cache,
    split_qkv_mrope_cache, vision_position, Affine, ArgmaxArgs, AttentionArgs, AttentionInput,
    AttentionMask, AttentionPolicy, AxialRopeArgs, AxialRopeNumerics, BiasOrder, ChannelL2NormArgs,
    DecodeRopeArgs, DecodeRotationArgs, DecodeRotationPosition, EdgeBf16DecodeMoeArgs,
    EdgeBf16DecodeMoeWeights, EdgeNvfp4MoeArgs, EdgeNvfp4MoeWeights, EmbeddingScale,
    ExecutionSession, ExpertAssignment, FusedAddChannelL2NormArgs, FusedAddNormArgs, GatherArgs,
    GatherPatchGeometry, GatherSemantic, GraphWorkspace, GroupedQkvRopeArgs, KvCacheAttentionArgs,
    KvCacheDecodeMeta, MropeCacheArgs, MultimodalRopeArgs, NormArgs, NormKind, NormNumerics,
    Nvfp4ExpertProjection, PointwiseActivation, PointwiseArgs, PointwiseSemantic, QkvCacheRopeArgs,
    QuantOut, QuantizationArgs, QuantizationSemantic, Qwen35AttentionReferenceArgs,
    Qwen35AttentionReferenceOutput, Qwen35AttentionScratch, Qwen35Fp8AttentionArgs,
    Qwen35Fp8KvCache, Qwen35KvCacheScales, Qwen35MtpMoeArgs, Qwen35MtpMoeScratch,
    Qwen35MtpMoeState, Qwen35MtpMoeWeights, Qwen35MtpReferenceOutput, Qwen35MtpReferenceWeights,
    ResidualAddArgs, RopeArgs, RopeSemantic, RoutePlan, RoutedMoeAddSharedArgs,
    RoutedMoeCombineArgs, RoutedMoeConfig, RoutedMoeNvfp4ExpertsArgs, RoutedMoeRouteArgs,
    RoutedMoeRouteTopkArgs, RoutedMoeSemantic, SegmentedAttentionArgs, SharedExpertOutput,
    VisionPositionArgs, FP8_E4M3_MAX_FINITE, QWEN35_ATTENTION_HEAD_DIM, QWEN35_ATTENTION_KV_HEADS,
    QWEN35_ATTENTION_KV_WIDTH, QWEN35_ATTENTION_Q_HEADS, QWEN35_ATTENTION_Q_PROJECTION_WIDTH,
    QWEN35_ATTENTION_RMS_EPSILON, QWEN35_ATTENTION_ROTARY_DIM, QWEN35_KV_CACHE_BLOCK_TOKENS,
    QWEN35_MTP_DRAFT_STEPS, QWEN35_MTP_EXPERTS, QWEN35_MTP_HIDDEN, QWEN35_MTP_INTERMEDIATE,
    QWEN35_MTP_LAYERS, QWEN35_MTP_TOP_K, QWEN35_MTP_VOCAB_SIZE,
};
pub use phase::PreparedPhase;
pub use stream::{CudaEvent, CudaStream};

pub mod ops;
pub use ops::{Conv3dArgs, Conv3dPolicy, PreparedConv3d, Unfold3dGeometry};

/// Whether the SM110 native NVFP4 GEMM+SwiGLU implementation is available.
///
/// The name describes the mathematical operator rather than its current model
/// consumer. Unsupported targets use the unfused GEMM + SwiGLU fallback.
pub const NVFP4_GEMM_SWIGLU: bool = cfg!(apxinf_nvfp4_gemm_swiglu);

/// Whether this build includes the aarch64 SM110 TensorRT-Edge-LLM CuTeDSL
/// routed NVFP4 MoE provider.
pub const EDGE_NVFP4_MOE: bool = cfg!(apxinf_edge_nvfp4_moe);

/// Whether this build includes the aarch64 SM110 TensorRT-Edge-LLM BF16
/// grouped-MoE provider used by Qwen3.5 MTP decode.
pub const EDGE_BF16_MOE: bool = cfg!(apxinf_edge_bf16_moe);

/// Whether this build compiled the allocation-free FA2 decode kernel
/// (`apxinf_decode_attention`). When `false` that ABI returns
/// `APXINF_STATUS_UNSUPPORTED`, so callers must use the generic attention path.
///
/// Like [`NVFP4_GEMM_SWIGLU`], this is a zero-cost compile-time constant
/// carrying a build-time cfg across the crate boundary.
pub const FA2_DECODE: bool = cfg!(apxinf_fa2_decode);

pub use ops::{F64RecurrenceState, PreparedRecurrence, RecurrenceCoefficients};

pub use ops::PreparedIndexedMean;

pub use ops::rms_norm_apply_precomputed;

/// BF16 staged exact-erf GELU GEMM.
pub use ops::gemm_gelu_exact;
