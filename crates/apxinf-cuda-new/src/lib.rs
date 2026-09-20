pub mod backend;
pub mod buffer;
pub mod context;
pub mod device_caps;
mod ffi;
mod graph;
pub mod kernels;
pub mod kv_cache;
pub mod nvtx;
pub mod profiler;
pub mod sampling;
pub mod stream;
pub mod timing;
pub mod transfers;
mod workspace;

pub use backend::CudaBackend;
pub use buffer::{CudaBuffer, CudaDeviceAddress, HostMappedBuffer};
pub use context::{device_memory_info, CudaContext, CudaMemoryInfo};
pub use device_caps::{CudaArchFamily, CudaDeviceCaps};
pub use graph::{capture, CapturedGraph};
pub use kv_cache::CudaKVCache;
pub use ops::{
    attention, concat_rows, decode_rope, gather, kv_cache_attention, norm, pointwise, quantization,
    reserve_prefix, rope, segmented_attention, AttentionArgs, AttentionMask, AttentionPolicy,
    DecodeRopeArgs, ExecutionSession,
    GatherArgs, GatherPatchGeometry, GatherPolicy, GatherSemantic, GraphWorkspace,
    KvCacheAttentionArgs, NormArgs, NormPolicy, NormSemantic, PointwiseActivation, PointwiseArgs,
    PointwisePolicy, PointwiseSemantic, QuantizationArgs, QuantizationPolicy,
    QuantizationSemantic, RopeArgs, RopePolicy, RopeSemantic, SegmentedAttentionArgs,
};
pub use stream::CudaStream;
pub use timing::CudaEventTimer;

pub mod ops;
