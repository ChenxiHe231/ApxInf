pub mod buffer;
pub mod context;
pub mod device_caps;
mod ffi;
mod graph;
pub mod nvtx;
pub mod profiler;
pub mod sampling;
pub mod stream;
pub mod timing;
pub mod transfers;
mod workspace;

pub use buffer::{CudaBuffer, CudaDeviceAddress, HostMappedBuffer};
pub use context::{device_memory_info, CudaContext, CudaMemoryInfo};
pub use device_caps::{CudaArchFamily, CudaDeviceCaps};
pub use graph::{capture, CapturedGraph};
pub use ops::{
    ada_gate_residual, ada_gate_residual_rms_norm, adaptive_rms_norm, attention, bias_residual,
    bias_residual_layer_norm, bias_residual_rms_norm, bias_then_residual, concat_rows, decode_rope,
<<<<<<< HEAD
    gather, kv_cache_attention, layer_norm, pointwise, quantization, reserve_prefix, rms_norm, rope,
    segmented_attention, AttentionArgs, AttentionMask, AttentionPolicy,
    DecodeRopeArgs, ExecutionSession,
    GatherArgs, GatherPatchGeometry, GatherSemantic, GraphWorkspace,
    AdaGateResidualArgs, AdaGateResidualRmsNormArgs, AdaptiveRmsNormArgs, BiasResidualArgs,
    BiasResidualLayerNormArgs, BiasResidualRmsNormArgs, BiasThenResidualArgs,
    KvCacheAttentionArgs, KvCacheDecodeMeta, LayerNormArgs, PointwiseActivation, PointwiseArgs,
    PointwiseSemantic, QuantizationArgs, RmsNormArgs, QuantizationSemantic, RopeArgs,
    RopeSemantic, SegmentedAttentionArgs,
=======
    gather, kv_cache_attention, layer_norm, packed_qkv_attention, pointwise, quantization,
    reserve_prefix, rms_norm, rope, segmented_attention, AdaGateResidualArgs,
    AdaGateResidualRmsNormArgs, AdaptiveRmsNormArgs, AttentionArgs, AttentionMask, AttentionPolicy,
    BiasResidualArgs, BiasResidualLayerNormArgs, BiasResidualRmsNormArgs, BiasThenResidualArgs,
    DecodeRopeArgs, ExecutionSession, GatherArgs, GatherPatchGeometry, GatherSemantic,
    GraphWorkspace, KvCacheAttentionArgs, LayerNormArgs, PackedQkvAttentionArgs,
    PointwiseActivation, PointwiseArgs, PointwiseSemantic, QuantizationArgs, QuantizationSemantic,
    RmsNormArgs, RopeArgs, RopeSemantic, SegmentedAttentionArgs,
>>>>>>> 7051409 (perf(pi05): fuse Thor vision QKV attention path)
};
pub use stream::CudaStream;
pub use timing::CudaEventTimer;

pub mod ops;
