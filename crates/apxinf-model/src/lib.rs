//! LLM model architectures and abstractions.

mod accelerator;
pub mod auto;
pub mod builtin;
pub mod debug;
mod generation_config;
#[cfg(not(feature = "pi05-only"))]
pub mod gr00t;
#[cfg(not(feature = "pi05-only"))]
pub mod llama;
pub mod llm_trait;
pub mod pi05;
#[cfg(not(feature = "pi05-only"))]
pub mod pi0fast;
pub mod profiling;
#[cfg(not(feature = "pi05-only"))]
pub mod qwen3vl;
pub mod qwen_drive;
pub mod registry;
pub mod vla;
#[cfg(not(feature = "pi05-only"))]
mod walloss;

pub use auto::{AutoModel, LoadOptions, LoadedModel, ModelPrecision, SyntheticWeights};
pub use builtin::register_builtin_models;
pub use debug::{DebugCapture, DebugConfig};
pub use generation_config::{GenerationConfigSource, GenerationOptions, SamplingMode};
#[cfg(all(feature = "cuda", not(feature = "pi05-only")))]
pub use llama::{DecodeGraph, DecodeGraphConfig, DecodeGraphWeights, DecodeLayerWeights};
#[cfg(not(feature = "pi05-only"))]
pub use llama::{GeneralLlama, KVCache, LlamaModel, LlamaWeights, TransformerLayer};
pub use llm_trait::{
    generate_streaming, generate_streaming_with_options, GeneratedToken, GenerationOutput,
    GenerationRequest, ImageInput, LlmCapabilities, LlmInput, LlmTrait,
};
pub use pi05::{Pi05Config, Pi05PerformanceProfile};
pub use profiling::GenerationProfile;
#[cfg(not(feature = "pi05-only"))]
pub use qwen3vl::{GeneralQwen3VL, Qwen3VLConfig, Qwen3VLTextWeights};
pub use qwen_drive::QwenDriveConfig;
#[cfg(feature = "cuda")]
pub use qwen_drive::QwenDriveModelRunner;
pub use registry::{get, list, register};
pub use vla::{
    Action, ExecutionMode, ExecutionPolicy, ImageLayout, InferenceSpec, InitialLatent, Observation,
    PreparationStatus, PreparedInference, VisionObservation, VlaContract, VlaMetadata, VlaRequest,
    VlaRuntime,
};
