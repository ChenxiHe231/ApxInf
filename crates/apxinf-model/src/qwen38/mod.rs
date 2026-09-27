//! Qwen3.8-27B-NVFP4 on Jetson Thor (sm_110).
//!
//! 64 layers: 48 Gated DeltaNet (linear attention) + 16 full attention, mixed
//! NVFP4/FP8/BF16 from a ModelOpt checkpoint. Loading builds the weight
//! structures; `core` owns the model dataflow (moved from the development
//! harness); `Qwen38` wraps it behind [`crate::LlmTrait`] with CUDA-graph
//! decode and batched FlashInfer-GDN prefill as the default execution.
//!
//! Numeric contracts: the quantizers match vLLM's encodings bit for bit
//! (RNE FP4, SATFINITE E4M3, double-rounded SwiGLU — see the
//! `qwen38_*_quant_contract` test suites in `apxinf-cuda`), and the
//! FlashInfer GDN scan is shadow-checked against the reference chunked scan.

#[cfg(feature = "cuda")]
mod config;
#[cfg(feature = "cuda")]
mod core;
#[cfg(feature = "cuda")]
mod model;

#[cfg(feature = "cuda")]
pub use config::{Qwen38Config, VOCAB};
#[cfg(feature = "cuda")]
pub use model::Qwen38;
