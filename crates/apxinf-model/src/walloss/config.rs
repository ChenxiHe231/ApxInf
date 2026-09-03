//! Checkpoint-owned configuration for the Walloss VLA runtime.

use std::path::Path;

use apxinf_core::{Error, Result};
use serde_json::Value;

#[derive(Clone, Debug, PartialEq)]
pub struct WallossTextConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_layers: usize,
    pub num_attention_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub mrope_section: [usize; 3],
}

#[derive(Clone, Debug, PartialEq)]
pub struct WallossVisionConfig {
    pub depth: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_heads: usize,
    pub patch_size: usize,
    pub temporal_patch_size: usize,
    pub spatial_merge_size: usize,
    pub out_hidden_size: usize,
    pub window_size: usize,
    pub full_attention_blocks: Vec<usize>,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct WallossActionConfig {
    pub hidden_size: usize,
    pub state_hidden_size: usize,
    pub intermediate_size: usize,
    pub action_dim: usize,
    pub proprio_dim: usize,
    pub action_horizon: usize,
    pub solver_steps: usize,
    pub causal_attention: bool,
    pub use_x_prediction: bool,
    pub scheduler_s: f32,
}

#[derive(Clone, Debug, PartialEq)]
pub struct WallossConfig {
    pub text: WallossTextConfig,
    pub vision: WallossVisionConfig,
    pub action: WallossActionConfig,
    pub image_token_id: u32,
    pub video_token_id: u32,
    pub vision_start_token_id: u32,
    pub vision_end_token_id: u32,
}

impl WallossConfig {
    pub fn from_json_file(path: &Path) -> Result<Self> {
        let raw = std::fs::read_to_string(path)
            .map_err(|error| Error::Other(format!("read {}: {error}", path.display())))?;
        Self::from_json_str(&raw)
    }

    pub fn from_json_str(raw: &str) -> Result<Self> {
        let value: Value = serde_json::from_str(raw)
            .map_err(|error| Error::Other(format!("walloss config json: {error}")))?;
        let vision = required_object(&value, "vision_config")?;
        let experts = value
            .get("experts")
            .and_then(Value::as_array)
            .ok_or_else(|| Error::Other("walloss config: missing experts".into()))?;
        if experts.len() != 2 {
            return Err(Error::Other(format!(
                "walloss config: expected two execution branches, got {}",
                experts.len()
            )));
        }
        let language_expert = &experts[0];
        let action_expert = &experts[1];
        let hidden_size = usize_field(&value, "hidden_size")?;
        let intermediate_size = usize_field(&value, "intermediate_size")?;
        if usize_field(language_expert, "hidden_size")? != hidden_size
            || usize_field(language_expert, "intermediate_size")? != intermediate_size
        {
            return Err(Error::Other(
                "walloss config: language branch dimensions disagree with the backbone".into(),
            ));
        }

        let mrope = value
            .pointer("/rope_scaling/mrope_section")
            .and_then(Value::as_array)
            .ok_or_else(|| Error::Other("walloss config: missing mRoPE sections".into()))?;
        if mrope.len() != 3 {
            return Err(Error::Other(format!(
                "walloss config: expected three mRoPE sections, got {}",
                mrope.len()
            )));
        }

        let q_heads = usize_field(&value, "num_attention_heads")?;
        if hidden_size % q_heads != 0 {
            return Err(Error::Other(format!(
                "walloss config: hidden size {hidden_size} is not divisible by {q_heads} heads"
            )));
        }
        let action_hidden_size = usize_field(action_expert, "hidden_size")?;
        let declared_action_hidden = usize_field(&value, "action_hidden_size")?;
        if action_hidden_size != declared_action_hidden {
            return Err(Error::Other(
                "walloss config: action branch hidden sizes disagree".into(),
            ));
        }

        Ok(Self {
            text: WallossTextConfig {
                hidden_size,
                intermediate_size,
                num_layers: usize_field(&value, "num_hidden_layers")?,
                num_attention_heads: q_heads,
                num_kv_heads: usize_field(&value, "num_key_value_heads")?,
                head_dim: hidden_size / q_heads,
                // Some published configs retain the base tokenizer size while
                // the embedding tensor contains added control tokens. The
                // weight loader validates and replaces this value from the
                // embedding shape before allocating device weights.
                vocab_size: usize_field(&value, "vocab_size")?,
                max_position_embeddings: usize_field(&value, "max_position_embeddings")?,
                rms_norm_eps: f32_field(&value, "rms_norm_eps")?,
                rope_theta: f32_field(&value, "rope_theta")?,
                mrope_section: [
                    value_as_usize(&mrope[0], "mRoPE temporal section")?,
                    value_as_usize(&mrope[1], "mRoPE height section")?,
                    value_as_usize(&mrope[2], "mRoPE width section")?,
                ],
            },
            vision: WallossVisionConfig {
                depth: usize_field(vision, "depth")?,
                hidden_size: usize_field(vision, "hidden_size")?,
                intermediate_size: usize_field(vision, "intermediate_size")?,
                num_heads: usize_field(vision, "num_heads")?,
                patch_size: usize_field(vision, "patch_size")?,
                temporal_patch_size: usize_field(vision, "temporal_patch_size")?,
                spatial_merge_size: usize_field(vision, "spatial_merge_size")?,
                out_hidden_size: usize_field(vision, "out_hidden_size")?,
                window_size: usize_field(vision, "window_size")?,
                full_attention_blocks: usize_array(vision, "fullatt_block_indexes")?,
                rms_norm_eps: vision
                    .get("rms_norm_eps")
                    .and_then(Value::as_f64)
                    .unwrap_or(1e-6) as f32,
                rope_theta: vision
                    .get("rope_theta")
                    .and_then(Value::as_f64)
                    .unwrap_or(10_000.0) as f32,
            },
            action: WallossActionConfig {
                hidden_size: action_hidden_size,
                state_hidden_size: value
                    .get("state_hidden_size")
                    .map(|field| value_as_usize(field, "state_hidden_size"))
                    .transpose()?
                    .unwrap_or(hidden_size),
                intermediate_size: usize_field(action_expert, "intermediate_size")?,
                // These widths are not reliably serialized by published
                // WallOSS configs. Zero means "unspecified" until the weight
                // loader resolves the checkpoint-native dimensions.
                action_dim: 0,
                proprio_dim: 0,
                action_horizon: 10,
                solver_steps: value
                    .pointer("/noise_scheduler/num_inference_timesteps")
                    .map(|v| value_as_usize(v, "solver steps"))
                    .transpose()?
                    .unwrap_or(10),
                causal_attention: bool_field(&value, "causal_action_attention_mask")?,
                use_x_prediction: bool_field(&value, "use_x_pred")?,
                scheduler_s: value
                    .pointer("/noise_scheduler/s")
                    .and_then(Value::as_f64)
                    .unwrap_or(0.999) as f32,
            },
            image_token_id: u32_field(&value, "image_token_id")?,
            video_token_id: u32_field(&value, "video_token_id")?,
            vision_start_token_id: u32_field(&value, "vision_start_token_id")?,
            vision_end_token_id: u32_field(&value, "vision_end_token_id")?,
        })
    }
}

fn required_object<'a>(value: &'a Value, name: &str) -> Result<&'a Value> {
    value
        .get(name)
        .filter(|field| field.is_object())
        .ok_or_else(|| Error::Other(format!("walloss config: missing {name}")))
}

fn value_as_usize(value: &Value, name: &str) -> Result<usize> {
    value
        .as_u64()
        .and_then(|field| usize::try_from(field).ok())
        .ok_or_else(|| Error::Other(format!("walloss config: invalid {name}")))
}

fn usize_field(value: &Value, name: &str) -> Result<usize> {
    value
        .get(name)
        .ok_or_else(|| Error::Other(format!("walloss config: missing {name}")))
        .and_then(|field| value_as_usize(field, name))
}

fn u32_field(value: &Value, name: &str) -> Result<u32> {
    let field = usize_field(value, name)?;
    u32::try_from(field)
        .map_err(|_| Error::Other(format!("walloss config: {name} does not fit u32")))
}

fn f32_field(value: &Value, name: &str) -> Result<f32> {
    value
        .get(name)
        .and_then(Value::as_f64)
        .map(|field| field as f32)
        .ok_or_else(|| Error::Other(format!("walloss config: invalid {name}")))
}

fn bool_field(value: &Value, name: &str) -> Result<bool> {
    value
        .get(name)
        .and_then(Value::as_bool)
        .ok_or_else(|| Error::Other(format!("walloss config: invalid {name}")))
}

fn usize_array(value: &Value, name: &str) -> Result<Vec<usize>> {
    value
        .get(name)
        .and_then(Value::as_array)
        .ok_or_else(|| Error::Other(format!("walloss config: invalid {name}")))?
        .iter()
        .map(|field| value_as_usize(field, name))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    const CONFIG: &str = r#"{
        "hidden_size": 2048,
        "intermediate_size": 11008,
        "num_hidden_layers": 36,
        "num_attention_heads": 16,
        "num_key_value_heads": 2,
        "vocab_size": 151936,
        "max_position_embeddings": 128000,
        "rms_norm_eps": 0.000001,
        "rope_theta": 1000000.0,
        "rope_scaling": {"mrope_section": [16, 24, 24]},
        "action_hidden_size": 1024,
        "experts": [
            {"hidden_size": 2048, "intermediate_size": 11008},
            {"hidden_size": 1024, "intermediate_size": 2048}
        ],
        "noise_scheduler": {"s": 0.999, "num_inference_timesteps": 10},
        "causal_action_attention_mask": true,
        "use_x_pred": false,
        "image_token_id": 151655,
        "video_token_id": 151656,
        "vision_start_token_id": 151652,
        "vision_end_token_id": 151653,
        "vision_config": {
            "depth": 32,
            "hidden_size": 1280,
            "intermediate_size": 3420,
            "num_heads": 16,
            "patch_size": 14,
            "temporal_patch_size": 2,
            "spatial_merge_size": 2,
            "out_hidden_size": 2048,
            "window_size": 112,
            "fullatt_block_indexes": [7, 15, 23, 31]
        }
    }"#;

    #[test]
    fn parses_published_architecture() {
        let config = WallossConfig::from_json_str(CONFIG).unwrap();
        assert_eq!(config.text.head_dim, 128);
        assert_eq!(config.text.mrope_section, [16, 24, 24]);
        assert_eq!(config.vision.depth, 32);
        assert_eq!(config.action.hidden_size, 1024);
        assert_eq!(config.action.action_dim, 0);
        assert_eq!(config.action.proprio_dim, 0);
        assert_eq!(config.action.solver_steps, 10);
    }

    #[test]
    fn rejects_action_dimension_mismatch() {
        let bad = CONFIG.replace(
            "\"action_hidden_size\": 1024",
            "\"action_hidden_size\": 768",
        );
        assert!(WallossConfig::from_json_str(&bad).is_err());
    }
}
