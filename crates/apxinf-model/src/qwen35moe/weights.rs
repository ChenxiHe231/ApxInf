//! Text-model weight manifest for the pinned Qwen3.5-35B-A3B NVFP4 checkpoint.
//!
//! The checkpoint also contains a vision tower. Those tensors are deliberately
//! outside this manifest: this family contract covers the language model and
//! its single MTP layer only.

use std::collections::HashSet;

use apxinf_core::Result;

use super::architecture::{MixerKind, Qwen35MoeArchitecture};
use super::config::Qwen35MoeConfig;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CheckpointDType {
    Bf16,
    F32,
    U8,
    F8E4M3,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WeightRole {
    Embedding,
    Normalization,
    Mixer,
    Router,
    RoutedExpert,
    SharedExpert,
    SharedExpertGate,
    LmHead,
    MtpProjection,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WeightSpec {
    pub name: String,
    pub dtype: CheckpointDType,
    pub shape: Vec<usize>,
    pub role: WeightRole,
}

impl WeightSpec {
    fn new(
        name: impl Into<String>,
        dtype: CheckpointDType,
        shape: impl Into<Vec<usize>>,
        role: WeightRole,
    ) -> Self {
        Self {
            name: name.into(),
            dtype,
            shape: shape.into(),
            role,
        }
    }
}

/// Returns the exact language-model and MTP tensor manifest.
///
/// This is a validation/mapping contract, not an executable loader. In
/// particular, emitting NVFP4 scale tensors here does not imply that routed
/// NVFP4 MoE execution is implemented.
pub fn expected_text_and_mtp_weights(config: &Qwen35MoeConfig) -> Result<Vec<WeightSpec>> {
    let architecture = Qwen35MoeArchitecture::from_config(config)?;
    let hidden = architecture.hidden_size;
    let mut weights = Vec::with_capacity(124_658);

    push_bf16(
        &mut weights,
        "model.language_model.embed_tokens.weight",
        [architecture.vocab_size, hidden],
        WeightRole::Embedding,
    );
    push_bf16(
        &mut weights,
        "model.language_model.norm.weight",
        [hidden],
        WeightRole::Normalization,
    );
    push_bf16(
        &mut weights,
        "lm_head.weight",
        [architecture.vocab_size, hidden],
        WeightRole::LmHead,
    );

    for layer in &architecture.layers {
        let prefix = format!("model.language_model.layers.{}", layer.index);
        push_bf16(
            &mut weights,
            format!("{prefix}.input_layernorm.weight"),
            [hidden],
            WeightRole::Normalization,
        );
        push_bf16(
            &mut weights,
            format!("{prefix}.post_attention_layernorm.weight"),
            [hidden],
            WeightRole::Normalization,
        );

        match layer.mixer {
            MixerKind::Gdn(gdn) => {
                push_bf16(
                    &mut weights,
                    format!("{prefix}.linear_attn.A_log"),
                    [gdn.value_heads],
                    WeightRole::Mixer,
                );
                push_bf16(
                    &mut weights,
                    format!("{prefix}.linear_attn.conv1d.weight"),
                    [gdn.qkv_width, 1, gdn.conv_kernel_dim],
                    WeightRole::Mixer,
                );
                push_bf16(
                    &mut weights,
                    format!("{prefix}.linear_attn.dt_bias"),
                    [gdn.value_heads],
                    WeightRole::Mixer,
                );
                for projection in ["in_proj_a", "in_proj_b"] {
                    push_bf16(
                        &mut weights,
                        format!("{prefix}.linear_attn.{projection}.weight"),
                        [gdn.value_heads, hidden],
                        WeightRole::Mixer,
                    );
                }
                push_bf16(
                    &mut weights,
                    format!("{prefix}.linear_attn.in_proj_qkv.weight"),
                    [gdn.qkv_width, hidden],
                    WeightRole::Mixer,
                );
                push_bf16(
                    &mut weights,
                    format!("{prefix}.linear_attn.in_proj_z.weight"),
                    [gdn.z_width, hidden],
                    WeightRole::Mixer,
                );
                push_bf16(
                    &mut weights,
                    format!("{prefix}.linear_attn.norm.weight"),
                    [gdn.value_head_dim],
                    WeightRole::Normalization,
                );
                push_bf16(
                    &mut weights,
                    format!("{prefix}.linear_attn.out_proj.weight"),
                    [hidden, gdn.z_width],
                    WeightRole::Mixer,
                );
            }
            MixerKind::FullAttention(attention) => {
                push_attention(
                    &mut weights,
                    &format!("{prefix}.self_attn"),
                    hidden,
                    attention.query_projection_width,
                    attention.kv_projection_width,
                    attention.head_dim,
                    true,
                );
            }
        }

        push_bf16(
            &mut weights,
            format!("{prefix}.mlp.gate.weight"),
            [layer.moe.experts, hidden],
            WeightRole::Router,
        );
        for expert in 0..layer.moe.experts {
            let expert_prefix = format!("{prefix}.mlp.experts.{expert}");
            push_nvfp4_linear(
                &mut weights,
                format!("{expert_prefix}.gate_proj"),
                hidden,
                layer.moe.expert_intermediate_size,
                WeightRole::RoutedExpert,
            );
            push_nvfp4_linear(
                &mut weights,
                format!("{expert_prefix}.up_proj"),
                hidden,
                layer.moe.expert_intermediate_size,
                WeightRole::RoutedExpert,
            );
            push_nvfp4_linear(
                &mut weights,
                format!("{expert_prefix}.down_proj"),
                layer.moe.expert_intermediate_size,
                hidden,
                WeightRole::RoutedExpert,
            );
        }
        let shared_prefix = format!("{prefix}.mlp.shared_expert");
        push_nvfp4_linear(
            &mut weights,
            format!("{shared_prefix}.gate_proj"),
            hidden,
            layer.moe.shared_expert_intermediate_size,
            WeightRole::SharedExpert,
        );
        push_nvfp4_linear(
            &mut weights,
            format!("{shared_prefix}.up_proj"),
            hidden,
            layer.moe.shared_expert_intermediate_size,
            WeightRole::SharedExpert,
        );
        push_nvfp4_linear(
            &mut weights,
            format!("{shared_prefix}.down_proj"),
            layer.moe.shared_expert_intermediate_size,
            hidden,
            WeightRole::SharedExpert,
        );
        push_bf16(
            &mut weights,
            format!("{prefix}.mlp.shared_expert_gate.weight"),
            [1, hidden],
            WeightRole::SharedExpertGate,
        );
    }

    push_bf16(
        &mut weights,
        "mtp.fc.weight",
        [hidden, 2 * hidden],
        WeightRole::MtpProjection,
    );
    for name in [
        "mtp.pre_fc_norm_embedding.weight",
        "mtp.pre_fc_norm_hidden.weight",
        "mtp.norm.weight",
    ] {
        push_bf16(&mut weights, name, [hidden], WeightRole::Normalization);
    }

    let mtp_prefix = "mtp.layers.0";
    for name in ["input_layernorm.weight", "post_attention_layernorm.weight"] {
        push_bf16(
            &mut weights,
            format!("{mtp_prefix}.{name}"),
            [hidden],
            WeightRole::Normalization,
        );
    }
    let attention = architecture.mtp.attention;
    push_attention(
        &mut weights,
        &format!("{mtp_prefix}.self_attn"),
        hidden,
        attention.query_projection_width,
        attention.kv_projection_width,
        attention.head_dim,
        false,
    );
    push_bf16(
        &mut weights,
        format!("{mtp_prefix}.mlp.gate.weight"),
        [architecture.mtp.moe.experts, hidden],
        WeightRole::Router,
    );
    for expert in 0..architecture.mtp.moe.experts {
        let expert_prefix = format!("{mtp_prefix}.mlp.experts.{expert}");
        push_bf16_linear(
            &mut weights,
            format!("{expert_prefix}.gate_proj"),
            hidden,
            architecture.mtp.moe.expert_intermediate_size,
            WeightRole::RoutedExpert,
        );
        push_bf16_linear(
            &mut weights,
            format!("{expert_prefix}.up_proj"),
            hidden,
            architecture.mtp.moe.expert_intermediate_size,
            WeightRole::RoutedExpert,
        );
        push_bf16_linear(
            &mut weights,
            format!("{expert_prefix}.down_proj"),
            architecture.mtp.moe.expert_intermediate_size,
            hidden,
            WeightRole::RoutedExpert,
        );
    }
    let shared_prefix = format!("{mtp_prefix}.mlp.shared_expert");
    push_bf16_linear(
        &mut weights,
        format!("{shared_prefix}.gate_proj"),
        hidden,
        architecture.mtp.moe.shared_expert_intermediate_size,
        WeightRole::SharedExpert,
    );
    push_bf16_linear(
        &mut weights,
        format!("{shared_prefix}.up_proj"),
        hidden,
        architecture.mtp.moe.shared_expert_intermediate_size,
        WeightRole::SharedExpert,
    );
    push_bf16_linear(
        &mut weights,
        format!("{shared_prefix}.down_proj"),
        architecture.mtp.moe.shared_expert_intermediate_size,
        hidden,
        WeightRole::SharedExpert,
    );
    push_bf16(
        &mut weights,
        format!("{mtp_prefix}.mlp.shared_expert_gate.weight"),
        [1, hidden],
        WeightRole::SharedExpertGate,
    );

    debug_assert_eq!(weights.len(), 124_658);
    debug_assert_eq!(
        weights
            .iter()
            .map(|weight| &weight.name)
            .collect::<HashSet<_>>()
            .len(),
        weights.len()
    );
    Ok(weights)
}

pub fn is_visual_weight(name: &str) -> bool {
    name.starts_with("model.visual.")
}

pub(super) fn push_attention(
    weights: &mut Vec<WeightSpec>,
    prefix: &str,
    hidden: usize,
    query_width: usize,
    kv_width: usize,
    head_dim: usize,
    include_fp8_kv_scales: bool,
) {
    for (projection, output) in [
        ("q_proj", query_width),
        ("k_proj", kv_width),
        ("v_proj", kv_width),
    ] {
        push_bf16(
            weights,
            format!("{prefix}.{projection}.weight"),
            [output, hidden],
            WeightRole::Mixer,
        );
    }
    push_bf16(
        weights,
        format!("{prefix}.o_proj.weight"),
        [hidden, query_width / 2],
        WeightRole::Mixer,
    );
    for norm in ["q_norm", "k_norm"] {
        push_bf16(
            weights,
            format!("{prefix}.{norm}.weight"),
            [head_dim],
            WeightRole::Normalization,
        );
    }
    if include_fp8_kv_scales {
        // ModelOpt stores each KV-cache amax next to the projection which
        // produces that cache.  These scalars are not direct dequantization
        // scales; the loader converts them by FP8_E4M3_MAX (448).
        for (projection, scale) in [("k_proj", "k_scale"), ("v_proj", "v_scale")] {
            weights.push(WeightSpec::new(
                format!("{prefix}.{projection}.{scale}"),
                CheckpointDType::F32,
                Vec::new(),
                WeightRole::Mixer,
            ));
        }
    }
}

fn push_nvfp4_linear(
    weights: &mut Vec<WeightSpec>,
    prefix: String,
    input: usize,
    output: usize,
    role: WeightRole,
) {
    weights.push(WeightSpec::new(
        format!("{prefix}.weight"),
        CheckpointDType::U8,
        [output, input / 2],
        role,
    ));
    weights.push(WeightSpec::new(
        format!("{prefix}.weight_scale"),
        CheckpointDType::F8E4M3,
        [output, input / 16],
        role,
    ));
    for suffix in ["input_scale", "weight_scale_2"] {
        weights.push(WeightSpec::new(
            format!("{prefix}.{suffix}"),
            CheckpointDType::F32,
            Vec::new(),
            role,
        ));
    }
}

fn push_bf16_linear(
    weights: &mut Vec<WeightSpec>,
    prefix: String,
    input: usize,
    output: usize,
    role: WeightRole,
) {
    push_bf16(weights, format!("{prefix}.weight"), [output, input], role);
}

fn push_bf16(
    weights: &mut Vec<WeightSpec>,
    name: impl Into<String>,
    shape: impl Into<Vec<usize>>,
    role: WeightRole,
) {
    weights.push(WeightSpec::new(name, CheckpointDType::Bf16, shape, role));
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qwen35moe::config::tests::valid_config;

    fn lookup<'a>(weights: &'a [WeightSpec], name: &str) -> &'a WeightSpec {
        weights
            .iter()
            .find(|weight| weight.name == name)
            .unwrap_or_else(|| panic!("missing weight {name}"))
    }

    #[test]
    fn maps_every_text_and_mtp_tensor_once() {
        let weights = expected_text_and_mtp_weights(&valid_config()).unwrap();
        assert_eq!(weights.len(), 124_658);
        let names: HashSet<_> = weights.iter().map(|weight| &weight.name).collect();
        assert_eq!(names.len(), weights.len());
        assert!(weights.iter().all(|weight| !is_visual_weight(&weight.name)));
        assert!(is_visual_weight("model.visual.blocks.0.attn.qkv.weight"));
    }

    #[test]
    fn maps_representative_hybrid_and_nvfp4_tensors() {
        let weights = expected_text_and_mtp_weights(&valid_config()).unwrap();
        assert_eq!(
            lookup(
                &weights,
                "model.language_model.layers.0.linear_attn.conv1d.weight"
            ),
            &WeightSpec::new(
                "model.language_model.layers.0.linear_attn.conv1d.weight",
                CheckpointDType::Bf16,
                [8_192, 1, 4],
                WeightRole::Mixer,
            )
        );
        assert_eq!(
            lookup(
                &weights,
                "model.language_model.layers.3.self_attn.q_proj.weight"
            )
            .shape,
            [8_192, 2_048]
        );
        assert_eq!(
            lookup(
                &weights,
                "model.language_model.layers.3.self_attn.k_proj.k_scale"
            )
            .dtype,
            CheckpointDType::F32
        );
        assert_eq!(
            lookup(
                &weights,
                "model.language_model.layers.3.self_attn.v_proj.v_scale"
            )
            .dtype,
            CheckpointDType::F32
        );
        assert!(!weights
            .iter()
            .any(|weight| weight.name == "model.language_model.layers.3.self_attn.k_scale"));
        assert_eq!(
            lookup(
                &weights,
                "model.language_model.layers.0.mlp.experts.0.gate_proj.weight"
            ),
            &WeightSpec::new(
                "model.language_model.layers.0.mlp.experts.0.gate_proj.weight",
                CheckpointDType::U8,
                [512, 1_024],
                WeightRole::RoutedExpert,
            )
        );
        assert_eq!(
            lookup(
                &weights,
                "model.language_model.layers.0.mlp.experts.0.gate_proj.weight_scale"
            )
            .shape,
            [512, 128]
        );
        assert_eq!(
            lookup(
                &weights,
                "model.language_model.layers.0.mlp.experts.0.down_proj.weight"
            )
            .shape,
            [2_048, 256]
        );
    }

    #[test]
    fn keeps_the_ignored_mtp_subtree_in_bf16() {
        let weights = expected_text_and_mtp_weights(&valid_config()).unwrap();
        let mtp_expert = "mtp.layers.0.mlp.experts.0.gate_proj";
        assert_eq!(
            lookup(&weights, &format!("{mtp_expert}.weight")),
            &WeightSpec::new(
                format!("{mtp_expert}.weight"),
                CheckpointDType::Bf16,
                [512, 2_048],
                WeightRole::RoutedExpert,
            )
        );
        assert!(!weights
            .iter()
            .any(|weight| weight.name == format!("{mtp_expert}.weight_scale")));
        assert_eq!(lookup(&weights, "mtp.fc.weight").shape, [2_048, 4_096]);
        assert!(weights
            .iter()
            .all(|weight| !weight.name.starts_with("mtp.embed_tokens")));
    }
}
