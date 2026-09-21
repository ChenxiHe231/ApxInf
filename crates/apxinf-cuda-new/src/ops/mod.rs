//! Semantic CUDA APIs with selection and native state behind L1.

mod attention;
mod cache;
mod gather;
mod gemm;
mod norm;
mod pointwise;
mod quantization;
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
    attention, kv_cache_attention, segmented_attention, AttentionArgs, AttentionMask,
    AttentionPolicy, KvCacheAttentionArgs, SegmentedAttentionArgs,
};
pub use cache::{concat_rows, reserve_prefix};
pub use gather::{
    gather, GatherArgs, GatherPolicy, GatherSemantic, PatchGeometry as GatherPatchGeometry,
};
pub use gemm::{
    gemm, gemm_bias, gemm_bias_gelu, gemm_geglu, GemmArgs, GemmBiasArgs, GemmBiasGeluArgs,
    GemmGegluArgs, GemmPolicy, GemmQuantization, WeightVersion,
};
pub use norm::{norm, NormArgs, NormPolicy, NormSemantic};
pub use pointwise::{
    pointwise, PointwiseActivation, PointwiseArgs, PointwisePolicy, PointwiseSemantic,
};
pub use quantization::{
    quantization, QuantizationArgs, QuantizationPolicy, QuantizationSemantic,
};
pub use rope::{decode_rope, rope, DecodeRopeArgs, RopeArgs, RopePolicy, RopeSemantic};

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
