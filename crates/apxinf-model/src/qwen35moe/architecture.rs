//! Model topology derived from the validated checkpoint configuration.

use apxinf_core::Result;

use super::config::{LayerType, Qwen35MoeConfig};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AttentionGeometry {
    pub query_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub rotary_dim: usize,
    pub query_projection_width: usize,
    pub kv_projection_width: usize,
    pub output_gate: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GdnGeometry {
    pub key_heads: usize,
    pub value_heads: usize,
    pub key_head_dim: usize,
    pub value_head_dim: usize,
    pub conv_kernel_dim: usize,
    pub qkv_width: usize,
    pub z_width: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MoeGeometry {
    pub experts: usize,
    pub experts_per_token: usize,
    pub expert_intermediate_size: usize,
    pub shared_expert_intermediate_size: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MixerKind {
    Gdn(GdnGeometry),
    FullAttention(AttentionGeometry),
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LayerArchitecture {
    pub index: usize,
    pub mixer: MixerKind,
    pub moe: MoeGeometry,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MtpArchitecture {
    pub layers: usize,
    pub uses_dedicated_embeddings: bool,
    pub attention: AttentionGeometry,
    pub moe: MoeGeometry,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Qwen35MoeArchitecture {
    pub hidden_size: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub layers: Vec<LayerArchitecture>,
    pub mtp: MtpArchitecture,
    /// The checkpoint declares static FP8 K/V storage for full-attention layers.
    pub fp8_kv_cache: bool,
}

impl Qwen35MoeArchitecture {
    pub fn from_config(config: &Qwen35MoeConfig) -> Result<Self> {
        config.validate()?;
        let text = &config.text_config;
        let attention = AttentionGeometry {
            query_heads: text.num_attention_heads,
            kv_heads: text.num_key_value_heads,
            head_dim: text.head_dim,
            rotary_dim: text.rotary_dim(),
            query_projection_width: text.q_projection_width(),
            kv_projection_width: text.kv_projection_width(),
            output_gate: text.attn_output_gate,
        };
        let gdn = GdnGeometry {
            key_heads: text.linear_num_key_heads,
            value_heads: text.linear_num_value_heads,
            key_head_dim: text.linear_key_head_dim,
            value_head_dim: text.linear_value_head_dim,
            conv_kernel_dim: text.linear_conv_kernel_dim,
            qkv_width: text.gdn_qkv_width(),
            z_width: text.gdn_z_width(),
        };
        let moe = MoeGeometry {
            experts: text.num_experts,
            experts_per_token: text.num_experts_per_tok,
            expert_intermediate_size: text.moe_intermediate_size,
            shared_expert_intermediate_size: text.shared_expert_intermediate_size,
        };
        let layers = text
            .layer_types
            .iter()
            .enumerate()
            .map(|(index, kind)| LayerArchitecture {
                index,
                mixer: match kind {
                    LayerType::LinearAttention => MixerKind::Gdn(gdn),
                    LayerType::FullAttention => MixerKind::FullAttention(attention),
                },
                moe,
            })
            .collect();
        Ok(Self {
            hidden_size: text.hidden_size,
            vocab_size: text.vocab_size,
            max_position_embeddings: text.max_position_embeddings,
            layers,
            mtp: MtpArchitecture {
                layers: text.mtp_num_hidden_layers,
                uses_dedicated_embeddings: text.mtp_use_dedicated_embeddings,
                attention,
                moe,
            },
            fp8_kv_cache: true,
        })
    }

    pub fn full_attention_layers(&self) -> impl Iterator<Item = &LayerArchitecture> {
        self.layers
            .iter()
            .filter(|layer| matches!(layer.mixer, MixerKind::FullAttention(_)))
    }

    pub fn gdn_layers(&self) -> impl Iterator<Item = &LayerArchitecture> {
        self.layers
            .iter()
            .filter(|layer| matches!(layer.mixer, MixerKind::Gdn(_)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::qwen35moe::config::tests::valid_config;

    #[test]
    fn derives_the_independent_hybrid_layer_plan() {
        let architecture = Qwen35MoeArchitecture::from_config(&valid_config()).unwrap();
        assert_eq!(architecture.layers.len(), 40);
        assert_eq!(architecture.gdn_layers().count(), 30);
        assert_eq!(architecture.full_attention_layers().count(), 10);
        assert!(matches!(architecture.layers[0].mixer, MixerKind::Gdn(_)));
        assert!(matches!(
            architecture.layers[3].mixer,
            MixerKind::FullAttention(_)
        ));
        assert_eq!(architecture.layers[0].moe.experts, 256);
        assert_eq!(architecture.layers[0].moe.experts_per_token, 8);
        assert_eq!(architecture.mtp.layers, 1);
        assert!(!architecture.mtp.uses_dedicated_embeddings);
        assert!(architecture.fp8_kv_cache);
    }

    #[test]
    fn records_the_target_attention_and_gdn_widths() {
        let architecture = Qwen35MoeArchitecture::from_config(&valid_config()).unwrap();
        let MixerKind::Gdn(gdn) = architecture.layers[0].mixer else {
            panic!("layer zero must be GDN");
        };
        assert_eq!((gdn.key_heads, gdn.value_heads), (16, 32));
        assert_eq!((gdn.qkv_width, gdn.z_width), (8_192, 4_096));
        let MixerKind::FullAttention(attention) = architecture.layers[3].mixer else {
            panic!("layer three must be full attention");
        };
        assert_eq!((attention.query_heads, attention.kv_heads), (16, 2));
        assert_eq!(attention.head_dim, 256);
        assert_eq!(attention.query_projection_width, 8_192);
        assert_eq!(attention.kv_projection_width, 512);
        assert_eq!(attention.rotary_dim, 64);
    }
}
