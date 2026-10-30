//! Checkpoint contract for Qwen3.5-35B-A3B NVFP4.
//!
//! The module includes strict checkpoint loading, CUDA-resident weight layout,
//! routed NVFP4 MoE and GDN execution slices, plus the transactional MTP
//! schedule. The full-attention backend, native BF16 MTP proposer and
//! batch-one serving surface are registered with [`crate::AutoModel`]. Exact
//! neutral greedy requests use transactional MTP speculation; other sampling
//! policies fall back to the base target decoder.

pub mod architecture;
#[cfg(feature = "qwen35-cuda")]
pub mod attention_backend;
pub mod config;
#[cfg(feature = "qwen35-cuda")]
pub mod decoder;
#[cfg(feature = "qwen35-cuda")]
pub mod device_weights;
pub mod execution;
#[cfg(feature = "qwen35-cuda")]
pub mod forward;
#[cfg(feature = "qwen35-cuda")]
pub mod llm;
pub mod loader;
#[cfg(feature = "qwen35-cuda")]
pub mod mtp_runtime;
pub mod mtp_schedule;
pub mod runtime_weights;
#[cfg(feature = "qwen35-cuda")]
pub mod session;
pub mod weights;

pub use architecture::{
    AttentionGeometry, GdnGeometry, LayerArchitecture, MixerKind, MoeGeometry, MtpArchitecture,
    Qwen35MoeArchitecture,
};
#[cfg(feature = "qwen35-cuda")]
pub use attention_backend::{Qwen35AttentionCache, Qwen35Fp8AttentionBackend};
pub use config::{load_trt_fp8_kv_cache_scales, Qwen35MoeConfig, Qwen35MoeTextConfig};
pub use execution::{Qwen35ExecutionOptions, Qwen35KvCacheDType, Qwen35MoeActivation};
#[cfg(feature = "qwen35-cuda")]
pub use llm::{Qwen35Moe, DEFAULT_KV_CAPACITY};
#[cfg(feature = "qwen35-cuda")]
pub use mtp_runtime::{
    MtpGreedyPolicy, MtpPositionContract, MtpPrefillOutcome, MtpRoundOutput, PreparedMtpCommit,
    PreparedMtpPrefill, Qwen35MtpCoordinator, Qwen35MtpRuntime, Qwen35MtpState,
    QWEN35_MTP_KV_DTYPE,
};
pub use runtime_weights::{
    AttentionWeights, DenseLinearWeights, GdnWeights, KvCacheScales, LinearWeightGeometry,
    MixerWeights, Nvfp4LinearWeights, Qwen35LayerWeights, Qwen35MtpLayerWeights, Qwen35MtpWeights,
    Qwen35RuntimeWeights, Qwen35TensorSource, RoutedMoeWeights, SwiGluExpertWeights,
};
pub use weights::{
    expected_text_and_mtp_weights, is_visual_weight, CheckpointDType, WeightRole, WeightSpec,
};
