//! Exact configuration contract for the Thor-U Qwen3.6-35B-A3B-NVFP4 checkpoint.

use std::collections::{HashMap, HashSet};
use std::path::Path;

use apxinf_core::{Error, Result};
use serde::Deserialize;

pub const CHECKPOINT_REPOSITORY: &str = "Qwen3.6-35B-A3B-NVFP4";
pub const CHECKPOINT_REVISION: &str = "local-thor-u";

const MODEL_TYPE: &str = "qwen3_5_moe";
const TEXT_MODEL_TYPE: &str = "qwen3_5_moe_text";
const ARCHITECTURE: &str = "Qwen3_5MoeForConditionalGeneration";

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum LayerType {
    LinearAttention,
    FullAttention,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct Qwen35MoeConfig {
    pub architectures: Vec<String>,
    pub dtype: String,
    pub model_type: String,
    pub tie_word_embeddings: bool,
    pub text_config: Qwen35MoeTextConfig,
    pub quantization_config: ModelOptNvfp4Config,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct Qwen35MoeTextConfig {
    pub attention_bias: bool,
    pub attention_dropout: f32,
    pub attn_output_gate: bool,
    pub bos_token_id: Option<u32>,
    pub dtype: String,
    pub eos_token_id: u32,
    pub full_attention_interval: usize,
    pub head_dim: usize,
    pub hidden_act: String,
    pub hidden_size: usize,
    pub layer_types: Vec<LayerType>,
    pub linear_conv_kernel_dim: usize,
    pub linear_key_head_dim: usize,
    pub linear_num_key_heads: usize,
    pub linear_num_value_heads: usize,
    pub linear_value_head_dim: usize,
    pub mamba_ssm_dtype: String,
    pub max_position_embeddings: usize,
    #[serde(default)]
    pub mlp_only_layers: Vec<usize>,
    pub model_type: String,
    pub moe_intermediate_size: usize,
    pub mtp_num_hidden_layers: usize,
    pub mtp_use_dedicated_embeddings: bool,
    pub num_attention_heads: usize,
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    pub num_hidden_layers: usize,
    pub num_key_value_heads: usize,
    pub output_router_logits: bool,
    pub pad_token_id: Option<u32>,
    pub partial_rotary_factor: f32,
    pub rms_norm_eps: f32,
    pub rope_parameters: RopeConfig,
    pub shared_expert_intermediate_size: usize,
    pub tie_word_embeddings: bool,
    pub use_cache: bool,
    pub vocab_size: usize,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct RopeConfig {
    pub mrope_interleaved: bool,
    pub mrope_section: [usize; 3],
    pub partial_rotary_factor: f32,
    pub rope_theta: f32,
    pub rope_type: String,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct ModelOptNvfp4Config {
    pub config_groups: HashMap<String, QuantizationGroup>,
    pub ignore: Vec<String>,
    #[serde(default)]
    pub kv_cache_scheme: Option<QuantizedTensor>,
    pub quant_algo: String,
    pub quant_method: String,
}

#[derive(Clone, Debug, Deserialize, PartialEq)]
pub struct QuantizationGroup {
    pub input_activations: QuantizedTensor,
    pub targets: Vec<String>,
    pub weights: QuantizedTensor,
}

#[derive(Clone, Debug, Deserialize, PartialEq, Eq)]
pub struct QuantizedTensor {
    pub dynamic: bool,
    pub num_bits: u8,
    #[serde(rename = "type")]
    pub kind: String,
    #[serde(default)]
    pub group_size: Option<usize>,
}

impl Qwen35MoeConfig {
    pub fn from_json_file(path: &Path) -> Result<Self> {
        let raw = std::fs::read_to_string(path)
            .map_err(|error| Error::Other(format!("read {}: {error}", path.display())))?;
        Self::from_json(&raw)
    }

    pub fn from_json(raw: &str) -> Result<Self> {
        let config: Self = serde_json::from_str(raw)
            .map_err(|error| invalid(format!("invalid Qwen3.5 MoE config: {error}")))?;
        config.validate()?;
        Ok(config)
    }

    pub fn validate(&self) -> Result<()> {
        if self.architectures.as_slice() != [ARCHITECTURE]
            || self.dtype != "bfloat16"
            || self.model_type != MODEL_TYPE
            || self.tie_word_embeddings
        {
            return Err(invalid(
                "unsupported Qwen3.5 MoE checkpoint identity, dtype, or embedding contract",
            ));
        }
        self.text_config.validate()?;
        self.quantization_config.validate(&self.text_config)
    }
}

impl Qwen35MoeTextConfig {
    pub fn validate(&self) -> Result<()> {
        let exact_geometry = !self.attention_bias
            && self.attention_dropout == 0.0
            && self.attn_output_gate
            && self.bos_token_id == Some(248_044)
            && self.dtype == "bfloat16"
            && self.eos_token_id == 248_044
            && self.full_attention_interval == 4
            && self.head_dim == 256
            && self.hidden_act == "silu"
            && self.hidden_size == 2_048
            && self.linear_conv_kernel_dim == 4
            && self.linear_key_head_dim == 128
            && self.linear_num_key_heads == 16
            && self.linear_num_value_heads == 32
            && self.linear_value_head_dim == 128
            && self.mamba_ssm_dtype == "float32"
            && self.max_position_embeddings == 262_144
            && self.mlp_only_layers.is_empty()
            && self.model_type == TEXT_MODEL_TYPE
            && self.moe_intermediate_size == 512
            && self.mtp_num_hidden_layers == 1
            && !self.mtp_use_dedicated_embeddings
            && self.num_attention_heads == 16
            && self.num_experts == 256
            && self.num_experts_per_tok == 8
            && self.num_hidden_layers == 40
            && self.num_key_value_heads == 2
            && !self.output_router_logits
            && self.pad_token_id.is_none()
            && self.shared_expert_intermediate_size == 512
            && !self.tie_word_embeddings
            && self.use_cache
            && self.vocab_size == 248_320;
        if !exact_geometry {
            return Err(invalid("unsupported Qwen3.5-35B-A3B text geometry"));
        }
        if self.layer_types.len() != self.num_hidden_layers {
            return Err(invalid(
                "Qwen3.5 MoE layer_types length does not match num_hidden_layers",
            ));
        }
        for (index, layer_type) in self.layer_types.iter().enumerate() {
            let expected = if (index + 1) % self.full_attention_interval == 0 {
                LayerType::FullAttention
            } else {
                LayerType::LinearAttention
            };
            if *layer_type != expected {
                return Err(invalid(
                    "Qwen3.5 MoE layers must follow the 3 GDN + 1 full-attention schedule",
                ));
            }
        }
        if self.partial_rotary_factor != 0.25
            || self.rms_norm_eps != 1e-6
            || !self.rope_parameters.mrope_interleaved
            || self.rope_parameters.mrope_section != [11, 11, 10]
            || self.rope_parameters.partial_rotary_factor != self.partial_rotary_factor
            || self.rope_parameters.rope_theta != 10_000_000.0
            || self.rope_parameters.rope_type != "default"
        {
            return Err(invalid(
                "unsupported Qwen3.5 MoE normalization or mRoPE contract",
            ));
        }
        Ok(())
    }

    pub fn rotary_dim(&self) -> usize {
        (self.head_dim as f32 * self.partial_rotary_factor) as usize
    }

    pub fn q_projection_width(&self) -> usize {
        2 * self.num_attention_heads * self.head_dim
    }

    pub fn kv_projection_width(&self) -> usize {
        self.num_key_value_heads * self.head_dim
    }

    pub fn gdn_qkv_width(&self) -> usize {
        2 * self.linear_num_key_heads * self.linear_key_head_dim
            + self.linear_num_value_heads * self.linear_value_head_dim
    }

    pub fn gdn_z_width(&self) -> usize {
        self.linear_num_value_heads * self.linear_value_head_dim
    }
}

impl ModelOptNvfp4Config {
    fn validate(&self, text: &Qwen35MoeTextConfig) -> Result<()> {
        if self.quant_method != "modelopt" || self.quant_algo != "MIXED_PRECISION" {
            return Err(invalid(
                "Qwen3.6 MoE requires the ModelOpt MIXED_PRECISION format",
            ));
        }
        if self.config_groups.len() != 2 {
            return Err(invalid(
                "Qwen3.6 MoE requires the FP8 mixer and NVFP4 linear groups",
            ));
        }
        let Some(fp8_group) = self.config_groups.get("group_0") else {
            return Err(invalid("Qwen3.6 MoE is missing FP8 group_0"));
        };
        let Some(nvfp4_group) = self.config_groups.get("group_1") else {
            return Err(invalid("Qwen3.6 MoE is missing NVFP4 group_1"));
        };
        let fp8 = |tensor: &QuantizedTensor| {
            !tensor.dynamic
                && tensor.num_bits == 8
                && tensor.kind == "float"
                && tensor.group_size.is_none()
        };
        let nvfp4 = |tensor: &QuantizedTensor| {
            !tensor.dynamic
                && tensor.num_bits == 4
                && tensor.kind == "float"
                && tensor.group_size == Some(16)
        };
        if !fp8(&fp8_group.input_activations)
            || !fp8(&fp8_group.weights)
            || !nvfp4(&nvfp4_group.input_activations)
            || !nvfp4(&nvfp4_group.weights)
        {
            return Err(invalid(
                "Qwen3.6 MoE quantization groups have an unsupported numeric contract",
            ));
        }
        if fp8_group.targets.iter().cloned().collect::<HashSet<_>>() != expected_fp8_modules(text)
            || nvfp4_group.targets.iter().cloned().collect::<HashSet<_>>()
                != expected_nvfp4_modules(text)
        {
            return Err(invalid(
                "Qwen3.6 MoE quantization targets do not match the mixer/MoE contract",
            ));
        }
        if self.kv_cache_scheme.is_some() {
            return Err(invalid(
                "Qwen3.6 checkpoint must not advertise unavailable FP8 KV-cache scales",
            ));
        }
        let actual: HashSet<_> = self.ignore.iter().cloned().collect();
        if actual.len() != self.ignore.len()
            || actual != HashSet::from(["mtp.layers.0*".to_owned(), "mtp*".to_owned()])
        {
            return Err(invalid(
                "Qwen3.5 MoE quantization ignore list does not match the text/MTP precision contract",
            ));
        }
        Ok(())
    }
}

fn expected_fp8_modules(text: &Qwen35MoeTextConfig) -> HashSet<String> {
    let mut targets = HashSet::new();
    for (index, layer_type) in text.layer_types.iter().enumerate() {
        let prefix = format!("model.language_model.layers.{index}");
        match layer_type {
            LayerType::LinearAttention => {
                for projection in ["in_proj_qkv", "in_proj_z", "out_proj"] {
                    targets.insert(format!("{prefix}.linear_attn.{projection}"));
                }
            }
            LayerType::FullAttention => {
                for projection in ["q_proj", "k_proj", "v_proj", "o_proj"] {
                    targets.insert(format!("{prefix}.self_attn.{projection}"));
                }
            }
        }
    }
    targets
}

fn expected_nvfp4_modules(text: &Qwen35MoeTextConfig) -> HashSet<String> {
    let mut targets = HashSet::from(["lm_head".to_owned()]);
    for index in 0..text.num_hidden_layers {
        let prefix = format!("model.language_model.layers.{index}.mlp");
        targets.insert(format!("{prefix}.experts"));
        for projection in ["gate_proj", "up_proj", "down_proj"] {
            targets.insert(format!("{prefix}.shared_expert.{projection}"));
        }
    }
    targets
}

fn invalid(message: impl Into<String>) -> Error {
    Error::Other(message.into())
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    fn valid_value() -> serde_json::Value {
        let layer_types: Vec<_> = (0..40)
            .map(|index| {
                if (index + 1) % 4 == 0 {
                    "full_attention"
                } else {
                    "linear_attention"
                }
            })
            .collect();
        let mut fp8_targets = Vec::new();
        let mut nvfp4_targets = vec!["lm_head".to_owned()];
        for (index, layer_type) in layer_types.iter().enumerate() {
            let layer = format!("model.language_model.layers.{index}");
            let projections: &[&str] = if *layer_type == "full_attention" {
                &[
                    "self_attn.q_proj",
                    "self_attn.k_proj",
                    "self_attn.v_proj",
                    "self_attn.o_proj",
                ]
            } else {
                &[
                    "linear_attn.in_proj_qkv",
                    "linear_attn.in_proj_z",
                    "linear_attn.out_proj",
                ]
            };
            fp8_targets.extend(
                projections
                    .iter()
                    .map(|projection| format!("{layer}.{projection}")),
            );
            let mlp = format!("{layer}.mlp");
            nvfp4_targets.push(format!("{mlp}.experts"));
            nvfp4_targets.extend(
                ["gate_proj", "up_proj", "down_proj"]
                    .map(|projection| format!("{mlp}.shared_expert.{projection}")),
            );
        }
        let text_config = serde_json::json!({
                "attention_bias": false,
                "attention_dropout": 0.0,
                "attn_output_gate": true,
                "bos_token_id": 248044,
                "dtype": "bfloat16",
                "eos_token_id": 248044,
                "full_attention_interval": 4,
                "head_dim": 256,
                "hidden_act": "silu",
                "hidden_size": 2048,
                "layer_types": layer_types,
                "linear_conv_kernel_dim": 4,
                "linear_key_head_dim": 128,
                "linear_num_key_heads": 16,
                "linear_num_value_heads": 32,
                "linear_value_head_dim": 128,
                "mamba_ssm_dtype": "float32",
                "max_position_embeddings": 262144,
                "mlp_only_layers": [],
                "model_type": "qwen3_5_moe_text",
                "moe_intermediate_size": 512,
                "mtp_num_hidden_layers": 1,
                "mtp_use_dedicated_embeddings": false,
                "num_attention_heads": 16,
                "num_experts": 256,
                "num_experts_per_tok": 8,
                "num_hidden_layers": 40,
                "num_key_value_heads": 2,
                "output_router_logits": false,
                "pad_token_id": null,
                "partial_rotary_factor": 0.25,
                "rms_norm_eps": 0.000001,
                "rope_parameters": {
                    "mrope_interleaved": true,
                    "mrope_section": [11, 11, 10],
                    "partial_rotary_factor": 0.25,
                    "rope_theta": 10000000,
                    "rope_type": "default"
                },
                "shared_expert_intermediate_size": 512,
                "tie_word_embeddings": false,
                "use_cache": true,
                "vocab_size": 248320
        });
        let quantization_config = serde_json::json!({
                "config_groups": {
                    "group_0": {
                        "input_activations": {
                            "dynamic": false, "num_bits": 8, "type": "float"
                        },
                        "targets": fp8_targets,
                        "weights": {
                            "dynamic": false, "num_bits": 8, "type": "float"
                        }
                    },
                    "group_1": {
                        "input_activations": {
                            "dynamic": false, "num_bits": 4, "type": "float", "group_size": 16
                        },
                        "targets": nvfp4_targets,
                        "weights": {
                            "dynamic": false, "num_bits": 4, "type": "float", "group_size": 16
                        }
                    }
                },
                "ignore": ["mtp.layers.0*", "mtp*"],
                "quant_algo": "MIXED_PRECISION",
                "quant_method": "modelopt"
        });
        serde_json::json!({
            "architectures": ["Qwen3_5MoeForConditionalGeneration"],
            "dtype": "bfloat16",
            "model_type": "qwen3_5_moe",
            "tie_word_embeddings": false,
            "text_config": text_config,
            "quantization_config": quantization_config
        })
    }

    pub(crate) fn valid_config() -> Qwen35MoeConfig {
        Qwen35MoeConfig::from_json(&valid_value().to_string()).unwrap()
    }

    #[test]
    fn parses_the_exact_35b_a3b_contract() {
        let config = valid_config();
        let text = &config.text_config;
        assert_eq!(text.layer_types.len(), 40);
        assert_eq!(
            text.layer_types
                .iter()
                .filter(|kind| **kind == LayerType::FullAttention)
                .count(),
            10
        );
        assert_eq!(text.rotary_dim(), 64);
        assert_eq!(text.q_projection_width(), 8_192);
        assert_eq!(text.kv_projection_width(), 512);
        assert_eq!(text.gdn_qkv_width(), 8_192);
        assert_eq!(text.gdn_z_width(), 4_096);
        assert_eq!(config.quantization_config.ignore.len(), 2);
    }

    #[test]
    fn rejects_dense_aliases_shape_drift_and_mtp_drift() {
        for (path, value) in [
            (&["model_type"][..], serde_json::json!("qwen3_5")),
            (&["text_config", "hidden_size"][..], serde_json::json!(5120)),
            (
                &["text_config", "mtp_num_hidden_layers"][..],
                serde_json::json!(0),
            ),
            (
                &["text_config", "mtp_use_dedicated_embeddings"][..],
                serde_json::json!(true),
            ),
        ] {
            let mut value_json = valid_value();
            let mut cursor = &mut value_json;
            for key in &path[..path.len() - 1] {
                cursor = &mut cursor[*key];
            }
            cursor[path[path.len() - 1]] = value;
            assert!(Qwen35MoeConfig::from_json(&value_json.to_string()).is_err());
        }
    }

    #[test]
    fn rejects_layer_and_quantization_contract_drift() {
        let mut schedule = valid_value();
        schedule["text_config"]["layer_types"][0] = serde_json::json!("full_attention");
        assert!(Qwen35MoeConfig::from_json(&schedule.to_string()).is_err());

        let mut ignored = valid_value();
        ignored["quantization_config"]["ignore"]
            .as_array_mut()
            .unwrap()
            .retain(|entry| entry != "mtp.layers.0*");
        assert!(Qwen35MoeConfig::from_json(&ignored.to_string()).is_err());

        let mut kv = valid_value();
        kv["quantization_config"]["kv_cache_scheme"] =
            serde_json::json!({"dynamic": false, "num_bits": 8, "type": "float"});
        assert!(Qwen35MoeConfig::from_json(&kv.to_string()).is_err());
    }
}
