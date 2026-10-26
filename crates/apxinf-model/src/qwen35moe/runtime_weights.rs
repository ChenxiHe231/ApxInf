//! Opaque runtime weight containers for the Qwen3.5 MoE execution path.
//!
//! These types define what a future loader and runtime must exchange. They do
//! not load tensors, launch operators, or register the model with `AutoModel`.

use apxinf_core::{Error, Result};

use super::architecture::{MixerKind, Qwen35MoeArchitecture};
use super::config::Qwen35MoeConfig;
use super::weights::WeightSpec;

/// Minimal checkpoint-source seam needed by a concrete loader.
pub trait Qwen35TensorSource {
    type Tensor;

    fn take_tensor(&mut self, spec: &WeightSpec) -> Result<Self::Tensor>;
    fn take_scalar_f32(&mut self, spec: &WeightSpec) -> Result<f32>;
}

pub trait LinearWeightGeometry {
    fn input_features(&self) -> usize;
    fn output_features(&self) -> usize;
}

#[derive(Clone, Debug)]
pub struct DenseLinearWeights<T> {
    pub weight: T,
    pub input_features: usize,
    pub output_features: usize,
}

impl<T> LinearWeightGeometry for DenseLinearWeights<T> {
    fn input_features(&self) -> usize {
        self.input_features
    }

    fn output_features(&self) -> usize {
        self.output_features
    }
}

#[derive(Clone, Debug)]
pub struct Nvfp4LinearWeights<T> {
    pub packed_weight: T,
    pub block_scales: T,
    pub input_scale: f32,
    pub weight_scale_2: f32,
    pub input_features: usize,
    pub output_features: usize,
}

impl<T> LinearWeightGeometry for Nvfp4LinearWeights<T> {
    fn input_features(&self) -> usize {
        self.input_features
    }

    fn output_features(&self) -> usize {
        self.output_features
    }
}

impl<T> Nvfp4LinearWeights<T> {
    pub fn validate(&self) -> Result<()> {
        if self.input_features == 0
            || self.output_features == 0
            || !self.input_features.is_multiple_of(16)
        {
            return Err(invalid(
                "Qwen3.5 NVFP4 linear dimensions must be non-zero and K must be divisible by 16",
            ));
        }
        if !self.input_scale.is_finite()
            || self.input_scale <= 0.0
            || !self.weight_scale_2.is_finite()
            || self.weight_scale_2 <= 0.0
        {
            return Err(invalid(
                "Qwen3.5 NVFP4 per-tensor scales must be finite and positive",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct SwiGluExpertWeights<L> {
    pub gate_proj: L,
    pub up_proj: L,
    pub down_proj: L,
}

#[derive(Clone, Debug)]
pub struct RoutedMoeWeights<T, L> {
    pub router: DenseLinearWeights<T>,
    pub experts: Vec<SwiGluExpertWeights<L>>,
    pub shared_expert: SwiGluExpertWeights<L>,
    pub shared_expert_gate: DenseLinearWeights<T>,
}

impl<T, L: LinearWeightGeometry> RoutedMoeWeights<T, L> {
    fn validate(
        &self,
        hidden: usize,
        experts: usize,
        expert_intermediate: usize,
        shared_intermediate: usize,
    ) -> Result<()> {
        validate_linear(&self.router, hidden, experts, "router")?;
        if self.experts.len() != experts {
            return Err(invalid(
                "Qwen3.5 routed-MoE weight count does not match the architecture",
            ));
        }
        for expert in &self.experts {
            validate_expert(expert, hidden, expert_intermediate, "routed expert")?;
        }
        validate_expert(
            &self.shared_expert,
            hidden,
            shared_intermediate,
            "shared expert",
        )?;
        validate_linear(&self.shared_expert_gate, hidden, 1, "shared expert gate")
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KvCacheScales {
    /// Dequantization multiplier for one stored FP8 key value.
    pub key: f32,
    /// Dequantization multiplier for one stored FP8 value value.
    pub value: f32,
}

impl KvCacheScales {
    /// Largest finite magnitude represented by NVIDIA's E4M3 cache format.
    pub const FP8_E4M3_MAX: f32 = 448.0;

    /// Convert ModelOpt checkpoint `k_scale`/`v_scale` scalars from amax to
    /// the cache contract used by Edge-LLM and cuda-new:
    ///
    /// - cache write: `fp8 = original / dequant_scale`
    /// - cache read:  `original = fp8 * dequant_scale`
    pub fn from_checkpoint_amax(key_amax: f32, value_amax: f32) -> Result<Self> {
        if !key_amax.is_finite() || key_amax <= 0.0 || !value_amax.is_finite() || value_amax <= 0.0
        {
            return Err(invalid(
                "Qwen3.5 FP8 KV-cache checkpoint amax values must be finite and positive",
            ));
        }
        let scales = Self {
            key: key_amax / Self::FP8_E4M3_MAX,
            value: value_amax / Self::FP8_E4M3_MAX,
        };
        scales.validate()?;
        Ok(scales)
    }

    fn validate(self) -> Result<()> {
        if !self.key.is_finite() || self.key <= 0.0 || !self.value.is_finite() || self.value <= 0.0
        {
            return Err(invalid(
                "Qwen3.5 FP8 KV-cache scales must be finite and positive",
            ));
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
pub struct AttentionWeights<T> {
    pub q_proj: DenseLinearWeights<T>,
    pub k_proj: DenseLinearWeights<T>,
    pub v_proj: DenseLinearWeights<T>,
    pub o_proj: DenseLinearWeights<T>,
    pub q_norm: T,
    pub k_norm: T,
    /// Present for base full-attention layers; absent for the BF16 MTP subtree.
    pub kv_cache_scales: Option<KvCacheScales>,
}

#[derive(Clone, Debug)]
pub struct GdnWeights<T> {
    pub a_log: T,
    pub conv1d: T,
    pub dt_bias: T,
    pub in_proj_a: DenseLinearWeights<T>,
    pub in_proj_b: DenseLinearWeights<T>,
    pub in_proj_qkv: DenseLinearWeights<T>,
    pub in_proj_z: DenseLinearWeights<T>,
    pub norm: T,
    pub out_proj: DenseLinearWeights<T>,
}

#[derive(Clone, Debug)]
pub enum MixerWeights<T> {
    Gdn(GdnWeights<T>),
    FullAttention(AttentionWeights<T>),
}

#[derive(Clone, Debug)]
pub struct Qwen35LayerWeights<T> {
    pub input_layernorm: T,
    pub post_attention_layernorm: T,
    pub mixer: MixerWeights<T>,
    pub moe: RoutedMoeWeights<T, Nvfp4LinearWeights<T>>,
}

#[derive(Clone, Debug)]
pub struct Qwen35MtpLayerWeights<T> {
    pub input_layernorm: T,
    pub post_attention_layernorm: T,
    pub attention: AttentionWeights<T>,
    pub moe: RoutedMoeWeights<T, DenseLinearWeights<T>>,
}

#[derive(Clone, Debug)]
pub struct Qwen35MtpWeights<T> {
    pub pre_fc_norm_embedding: T,
    pub pre_fc_norm_hidden: T,
    pub fc: DenseLinearWeights<T>,
    pub norm: T,
    pub layer: Qwen35MtpLayerWeights<T>,
}

#[derive(Clone, Debug)]
pub struct Qwen35RuntimeWeights<T> {
    pub embedding: T,
    pub layers: Vec<Qwen35LayerWeights<T>>,
    pub final_norm: T,
    pub lm_head: T,
    pub mtp: Qwen35MtpWeights<T>,
}

impl<T> Qwen35RuntimeWeights<T> {
    /// Validate container geometry against the exact checkpoint contract.
    pub fn validate_against(&self, config: &Qwen35MoeConfig) -> Result<()> {
        let architecture = Qwen35MoeArchitecture::from_config(config)?;
        if self.layers.len() != architecture.layers.len() {
            return Err(invalid(
                "Qwen3.5 runtime layer count does not match the architecture",
            ));
        }

        for (weights, layer) in self.layers.iter().zip(&architecture.layers) {
            match (&weights.mixer, layer.mixer) {
                (MixerWeights::Gdn(weights), MixerKind::Gdn(geometry)) => {
                    validate_linear(
                        &weights.in_proj_a,
                        architecture.hidden_size,
                        geometry.value_heads,
                        "GDN in_proj_a",
                    )?;
                    validate_linear(
                        &weights.in_proj_b,
                        architecture.hidden_size,
                        geometry.value_heads,
                        "GDN in_proj_b",
                    )?;
                    validate_linear(
                        &weights.in_proj_qkv,
                        architecture.hidden_size,
                        geometry.qkv_width,
                        "GDN in_proj_qkv",
                    )?;
                    validate_linear(
                        &weights.in_proj_z,
                        architecture.hidden_size,
                        geometry.z_width,
                        "GDN in_proj_z",
                    )?;
                    validate_linear(
                        &weights.out_proj,
                        geometry.z_width,
                        architecture.hidden_size,
                        "GDN out_proj",
                    )?;
                }
                (MixerWeights::FullAttention(weights), MixerKind::FullAttention(geometry)) => {
                    validate_attention(weights, architecture.hidden_size, geometry, true)?;
                }
                _ => {
                    return Err(invalid(
                        "Qwen3.5 runtime mixer kinds do not match the hybrid layer schedule",
                    ));
                }
            }
            weights.moe.validate(
                architecture.hidden_size,
                layer.moe.experts,
                layer.moe.expert_intermediate_size,
                layer.moe.shared_expert_intermediate_size,
            )?;
            for expert in weights
                .moe
                .experts
                .iter()
                .chain(std::iter::once(&weights.moe.shared_expert))
            {
                expert.gate_proj.validate()?;
                expert.up_proj.validate()?;
                expert.down_proj.validate()?;
            }
        }

        validate_linear(
            &self.mtp.fc,
            2 * architecture.hidden_size,
            architecture.hidden_size,
            "MTP fc",
        )?;
        validate_attention(
            &self.mtp.layer.attention,
            architecture.hidden_size,
            architecture.mtp.attention,
            false,
        )?;
        self.mtp.layer.moe.validate(
            architecture.hidden_size,
            architecture.mtp.moe.experts,
            architecture.mtp.moe.expert_intermediate_size,
            architecture.mtp.moe.shared_expert_intermediate_size,
        )
    }
}

fn validate_attention<T>(
    weights: &AttentionWeights<T>,
    hidden: usize,
    geometry: super::architecture::AttentionGeometry,
    requires_kv_scales: bool,
) -> Result<()> {
    validate_linear(
        &weights.q_proj,
        hidden,
        geometry.query_projection_width,
        "attention q_proj",
    )?;
    validate_linear(
        &weights.k_proj,
        hidden,
        geometry.kv_projection_width,
        "attention k_proj",
    )?;
    validate_linear(
        &weights.v_proj,
        hidden,
        geometry.kv_projection_width,
        "attention v_proj",
    )?;
    validate_linear(
        &weights.o_proj,
        geometry.query_projection_width / 2,
        hidden,
        "attention o_proj",
    )?;
    match (weights.kv_cache_scales, requires_kv_scales) {
        (Some(scales), true) => scales.validate(),
        (None, false) => Ok(()),
        _ => Err(invalid(
            "Qwen3.5 base attention requires FP8 KV scales and MTP attention must omit them",
        )),
    }
}

fn validate_expert<L: LinearWeightGeometry>(
    weights: &SwiGluExpertWeights<L>,
    hidden: usize,
    intermediate: usize,
    label: &str,
) -> Result<()> {
    validate_linear(
        &weights.gate_proj,
        hidden,
        intermediate,
        &format!("{label} gate_proj"),
    )?;
    validate_linear(
        &weights.up_proj,
        hidden,
        intermediate,
        &format!("{label} up_proj"),
    )?;
    validate_linear(
        &weights.down_proj,
        intermediate,
        hidden,
        &format!("{label} down_proj"),
    )
}

fn validate_linear<L: LinearWeightGeometry>(
    weights: &L,
    input: usize,
    output: usize,
    label: &str,
) -> Result<()> {
    if weights.input_features() != input || weights.output_features() != output {
        return Err(invalid(format!(
            "Qwen3.5 {label} shape does not match the architecture"
        )));
    }
    Ok(())
}

fn invalid(message: impl Into<String>) -> Error {
    Error::Other(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nvfp4_container_rejects_bad_block_geometry_and_scales() {
        let valid = Nvfp4LinearWeights {
            packed_weight: (),
            block_scales: (),
            input_scale: 0.5,
            weight_scale_2: 0.25,
            input_features: 2_048,
            output_features: 512,
        };
        assert!(valid.validate().is_ok());

        let bad_geometry = Nvfp4LinearWeights {
            input_features: 2_047,
            ..valid.clone()
        };
        assert!(bad_geometry.validate().is_err());
        let bad_scale = Nvfp4LinearWeights {
            input_scale: f32::NAN,
            ..valid
        };
        assert!(bad_scale.validate().is_err());
    }

    #[test]
    fn kv_cache_checkpoint_amax_is_converted_to_dequant_scale() {
        let scales = KvCacheScales::from_checkpoint_amax(1.0, 224.0).unwrap();
        assert_eq!(scales.key, 1.0 / 448.0);
        assert_eq!(scales.value, 0.5);
        assert!(KvCacheScales::from_checkpoint_amax(0.0, 1.0).is_err());
        assert!(KvCacheScales::from_checkpoint_amax(1.0, f32::NAN).is_err());
    }

    #[test]
    fn routed_container_requires_256_experts_and_exact_projection_shapes() {
        let linear = |input, output| DenseLinearWeights {
            weight: (),
            input_features: input,
            output_features: output,
        };
        let expert = || SwiGluExpertWeights {
            gate_proj: linear(2_048, 512),
            up_proj: linear(2_048, 512),
            down_proj: linear(512, 2_048),
        };
        let mut weights = RoutedMoeWeights {
            router: linear(2_048, 256),
            experts: (0..256).map(|_| expert()).collect(),
            shared_expert: expert(),
            shared_expert_gate: linear(2_048, 1),
        };
        assert!(weights.validate(2_048, 256, 512, 512).is_ok());
        weights.experts.pop();
        assert!(weights.validate(2_048, 256, 512, 512).is_err());
    }
}
