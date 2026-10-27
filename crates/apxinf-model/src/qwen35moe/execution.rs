//! Runtime precision policy for the exact Qwen3.5-35B-A3B implementation.
//!
//! These choices describe execution, not the on-disk checkpoint.  The same
//! ModelOpt NVFP4 checkpoint can therefore run its expert weights with BF16
//! activations (W4A16) or runtime-quantized NVFP4 activations (W4A4), and can
//! keep the target model's K/V cache in BF16 or FP8-E4M3.

use std::fmt;
use std::str::FromStr;

/// Activation precision used by the base model's NVFP4 MoE experts.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Qwen35MoeActivation {
    /// Keep activations in BF16 and consume the static NVFP4 weights as W4A16.
    Bf16,
    /// Quantize activations at runtime and use the SM110 W4A4 expert kernels.
    #[default]
    Nvfp4,
}

impl Qwen35MoeActivation {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Bf16 => "bf16",
            Self::Nvfp4 => "nvfp4",
        }
    }
}

impl fmt::Display for Qwen35MoeActivation {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

impl FromStr for Qwen35MoeActivation {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.to_ascii_lowercase().as_str() {
            "bf16" => Ok(Self::Bf16),
            "nvfp4" | "fp4" => Ok(Self::Nvfp4),
            _ => Err(format!(
                "unsupported Qwen3.5 MoE activation `{value}`; use bf16 or nvfp4"
            )),
        }
    }
}

/// Storage precision used by the base target model's ten full-attention K/V caches.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Qwen35KvCacheDType {
    /// Preserve normalized/rotated K and V as BF16 and execute FA2 attention.
    #[default]
    Bf16,
    /// Quantize K and V with the checkpoint's fixed scales when appending them.
    Fp8,
}

impl Qwen35KvCacheDType {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Bf16 => "bf16",
            Self::Fp8 => "fp8",
        }
    }

    pub const fn tensor_dtype(self) -> apxinf_core::DType {
        match self {
            Self::Bf16 => apxinf_core::DType::BF16,
            Self::Fp8 => apxinf_core::DType::F8E4M3,
        }
    }
}

impl fmt::Display for Qwen35KvCacheDType {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

impl FromStr for Qwen35KvCacheDType {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.to_ascii_lowercase().as_str() {
            "bf16" => Ok(Self::Bf16),
            "fp8" | "f8e4m3" => Ok(Self::Fp8),
            _ => Err(format!(
                "unsupported Qwen3.5 KV-cache dtype `{value}`; use bf16 or fp8"
            )),
        }
    }
}

/// Complete load-time execution policy for Qwen3.5-35B-A3B.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Qwen35ExecutionOptions {
    pub moe_activation: Qwen35MoeActivation,
    pub kv_cache: Qwen35KvCacheDType,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_supported_precision_values() {
        assert_eq!("bf16".parse(), Ok(Qwen35MoeActivation::Bf16));
        assert_eq!("nvfp4".parse(), Ok(Qwen35MoeActivation::Nvfp4));
        assert_eq!("bf16".parse(), Ok(Qwen35KvCacheDType::Bf16));
        assert_eq!("fp8".parse(), Ok(Qwen35KvCacheDType::Fp8));
    }

    #[test]
    fn defaults_preserve_the_current_runtime() {
        assert_eq!(
            Qwen35ExecutionOptions::default(),
            Qwen35ExecutionOptions {
                moe_activation: Qwen35MoeActivation::Nvfp4,
                kv_cache: Qwen35KvCacheDType::Bf16,
            }
        );
    }
}
