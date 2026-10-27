//! Strict host-side loader for the Qwen3.5-35B-A3B text and MTP checkpoint.
//!
//! This module maps native CPU tensors into the model-neutral runtime weight
//! tree. It deliberately performs no device upload and registers no model.

use std::collections::{HashMap, HashSet};
use std::path::Path;

use apxinf_core::{DType, Device, Error, Result, Tensor};
use apxinf_loader::safetensors::NativeSafetensorsReader;

use super::architecture::{
    AttentionGeometry, GdnGeometry, LayerArchitecture, MixerKind, Qwen35MoeArchitecture,
};
use super::config::Qwen35MoeConfig;
use super::runtime_weights::{
    AttentionWeights, DenseLinearWeights, DenseMoeWeights, GdnWeights, MixerWeights,
    Nvfp4LinearWeights, PackedDenseExperts, Qwen35LayerWeights, Qwen35MtpLayerWeights,
    Qwen35MtpWeights, Qwen35RuntimeWeights, Qwen35TensorSource, RoutedMoeWeights,
    StaticFp8LinearWeights, SwiGluExpertWeights,
};
use super::weights::{
    expected_text_and_mtp_weights, is_visual_weight, CheckpointDType, WeightSpec,
};

/// Consume a native host tensor map into the exact runtime container.
///
/// Every text/MTP tensor must occur exactly once with the manifest dtype and
/// physical shape. Vision tensors are intentionally discarded because this
/// runtime is text-only. Any other leftover checkpoint key is rejected.
pub fn load_qwen35_runtime_weights(
    config: &Qwen35MoeConfig,
    tensors: HashMap<String, Tensor>,
) -> Result<Qwen35RuntimeWeights<Tensor>> {
    let architecture = Qwen35MoeArchitecture::from_config(config)?;
    let mut loader = ManifestLoader::new(config, tensors)?;
    let hidden = architecture.hidden_size;

    let embedding = loader.take("model.language_model.embed_tokens.weight")?;
    let mut layers = Vec::with_capacity(architecture.layers.len());
    for layer in &architecture.layers {
        layers.push(load_qwen35_layer(&mut loader, layer, hidden)?);
    }

    let final_norm = loader.take("model.language_model.norm.weight")?;
    let lm_head =
        load_nvfp4_linear_for_device(&mut loader, "lm_head", hidden, architecture.vocab_size)?;
    let mtp = load_qwen35_mtp(&mut loader, &architecture, hidden)?;
    let weights = Qwen35RuntimeWeights {
        embedding,
        layers,
        final_norm,
        lm_head,
        mtp,
    };
    loader.finish()?;
    weights.validate_against(config)?;
    Ok(weights)
}

struct ManifestLoader {
    source: HostTensorSource,
    remaining_specs: HashMap<String, WeightSpec>,
}

pub(super) struct Qwen35SafetensorsLoader {
    reader: NativeSafetensorsReader,
    remaining_specs: HashMap<String, WeightSpec>,
    expected_payload_bytes: usize,
}

pub(super) trait ManifestAccess {
    fn take(&mut self, name: &str) -> Result<Tensor>;
    fn scalar(&mut self, name: &str) -> Result<f32>;
}

impl ManifestLoader {
    fn new(config: &Qwen35MoeConfig, tensors: HashMap<String, Tensor>) -> Result<Self> {
        let specs = expected_text_and_mtp_weights(config)?;
        let capacity = specs.len();
        let remaining_specs = specs
            .into_iter()
            .map(|spec| (spec.name.clone(), spec))
            .collect::<HashMap<_, _>>();
        if remaining_specs.len() != capacity {
            return Err(invalid("Qwen3.5 weight manifest contains duplicate names"));
        }
        Ok(Self {
            source: HostTensorSource { tensors },
            remaining_specs,
        })
    }

    fn take(&mut self, name: &str) -> Result<Tensor> {
        let spec = self.remaining_specs.remove(name).ok_or_else(|| {
            invalid(format!(
                "Qwen3.5 loader requested tensor `{name}` outside or twice from the manifest"
            ))
        })?;
        self.source.take_tensor(&spec)
    }

    fn scalar(&mut self, name: &str) -> Result<f32> {
        let spec = self.remaining_specs.remove(name).ok_or_else(|| {
            invalid(format!(
                "Qwen3.5 loader requested scalar `{name}` outside or twice from the manifest"
            ))
        })?;
        self.source.take_scalar_f32(&spec)
    }

    fn finish(self) -> Result<()> {
        if !self.remaining_specs.is_empty() {
            let mut names = self.remaining_specs.into_keys().collect::<Vec<_>>();
            names.sort();
            names.truncate(8);
            return Err(invalid(format!(
                "Qwen3.5 loader did not consume manifest tensors: {}",
                names.join(", ")
            )));
        }
        self.source.finish()
    }
}

impl ManifestAccess for ManifestLoader {
    fn take(&mut self, name: &str) -> Result<Tensor> {
        ManifestLoader::take(self, name)
    }

    fn scalar(&mut self, name: &str) -> Result<f32> {
        ManifestLoader::scalar(self, name)
    }
}

impl Qwen35SafetensorsLoader {
    /// Parse and validate the complete text/MTP manifest without copying any
    /// payload bytes. Vision entries are accepted by name but never loaded.
    pub(super) fn open(config: &Qwen35MoeConfig, path: &Path) -> Result<Self> {
        let reader = NativeSafetensorsReader::open_path(path)
            .map_err(|error| invalid(format!("open {}: {error}", path.display())))?;
        let specs = expected_text_and_mtp_weights(config)?;
        let expected_names = specs
            .iter()
            .map(|spec| spec.name.as_str())
            .collect::<HashSet<_>>();
        let mut unexpected = reader
            .tensor_names()
            .filter(|name| !expected_names.contains(name) && !is_visual_weight(name))
            .map(str::to_string)
            .collect::<Vec<_>>();
        unexpected.sort();
        if !unexpected.is_empty() {
            unexpected.truncate(8);
            return Err(invalid(format!(
                "unexpected Qwen3.5 checkpoint tensors before streaming load: {}",
                unexpected.join(", ")
            )));
        }
        let mut expected_payload_bytes = 0usize;
        for spec in &specs {
            let metadata = reader
                .tensor_metadata(&spec.name)
                .ok_or_else(|| invalid(format!("missing Qwen3.5 tensor `{}`", spec.name)))?;
            let expected_dtype = checkpoint_dtype(spec.dtype);
            if metadata.dtype != expected_dtype {
                return Err(invalid(format!(
                    "Qwen3.5 tensor `{}` dtype mismatch: expected {}, got {}",
                    spec.name, expected_dtype, metadata.dtype
                )));
            }
            if metadata.shape != spec.shape.as_slice() {
                return Err(invalid(format!(
                    "Qwen3.5 tensor `{}` shape mismatch: expected {:?}, got {:?}",
                    spec.name, spec.shape, metadata.shape
                )));
            }
            expected_payload_bytes = expected_payload_bytes
                .checked_add(metadata.byte_len)
                .ok_or_else(|| invalid("Qwen3.5 manifest payload byte count overflow"))?;
        }
        let capacity = specs.len();
        let remaining_specs = specs
            .into_iter()
            .map(|spec| (spec.name.clone(), spec))
            .collect::<HashMap<_, _>>();
        if remaining_specs.len() != capacity {
            return Err(invalid("Qwen3.5 weight manifest contains duplicate names"));
        }
        Ok(Self {
            reader,
            remaining_specs,
            expected_payload_bytes,
        })
    }

    pub(super) fn finish(self) -> Result<usize> {
        if !self.remaining_specs.is_empty() {
            let mut names = self.remaining_specs.into_keys().collect::<Vec<_>>();
            names.sort();
            names.truncate(8);
            return Err(invalid(format!(
                "Qwen3.5 streaming loader did not consume manifest tensors: {}",
                names.join(", ")
            )));
        }
        let copied = self.reader.payload_bytes_copied();
        if copied != self.expected_payload_bytes {
            return Err(invalid(format!(
                "Qwen3.5 streaming loader copied {copied} payload bytes, expected {}",
                self.expected_payload_bytes
            )));
        }
        Ok(copied)
    }
}

impl ManifestAccess for Qwen35SafetensorsLoader {
    fn take(&mut self, name: &str) -> Result<Tensor> {
        let spec = self.remaining_specs.remove(name).ok_or_else(|| {
            invalid(format!(
                "Qwen3.5 streaming loader requested tensor `{name}` outside or twice from the manifest"
            ))
        })?;
        let tensor = self
            .reader
            .load_tensor(name)
            .map_err(|error| invalid(format!("load Qwen3.5 tensor `{name}`: {error}")))?;
        validate_tensor(&tensor, &spec)?;
        Ok(tensor)
    }

    fn scalar(&mut self, name: &str) -> Result<f32> {
        let tensor = self.take(name)?;
        tensor
            .as_f32()?
            .first()
            .copied()
            .ok_or_else(|| invalid(format!("Qwen3.5 scalar `{name}` is empty")))
    }
}

struct HostTensorSource {
    tensors: HashMap<String, Tensor>,
}

impl Qwen35TensorSource for HostTensorSource {
    type Tensor = Tensor;

    fn take_tensor(&mut self, spec: &WeightSpec) -> Result<Tensor> {
        let tensor = self
            .tensors
            .remove(&spec.name)
            .ok_or_else(|| invalid(format!("missing Qwen3.5 tensor `{}`", spec.name)))?;
        validate_tensor(&tensor, spec)?;
        Ok(tensor)
    }

    fn take_scalar_f32(&mut self, spec: &WeightSpec) -> Result<f32> {
        let tensor = self.take_tensor(spec)?;
        let values = tensor.as_f32()?;
        values
            .first()
            .copied()
            .ok_or_else(|| invalid(format!("Qwen3.5 scalar `{}` is empty", spec.name)))
    }
}

impl HostTensorSource {
    fn finish(mut self) -> Result<()> {
        // The pinned checkpoint is multimodal, while this runtime contract is
        // explicitly text-only. Visual weights are accepted only under this
        // exact namespace and are deliberately dropped without interpretation.
        self.tensors.retain(|name, _| !is_visual_weight(name));
        if self.tensors.is_empty() {
            return Ok(());
        }
        let mut leftovers = self.tensors.into_keys().collect::<Vec<_>>();
        leftovers.sort();
        leftovers.truncate(8);
        Err(invalid(format!(
            "unexpected Qwen3.5 checkpoint tensors after text/MTP load: {}",
            leftovers.join(", ")
        )))
    }
}

fn validate_tensor(tensor: &Tensor, spec: &WeightSpec) -> Result<()> {
    if tensor.device() != Device::Cpu {
        return Err(invalid(format!(
            "Qwen3.5 tensor `{}` must be resident on the CPU host",
            spec.name
        )));
    }
    let expected_dtype = checkpoint_dtype(spec.dtype);
    if tensor.dtype() != expected_dtype {
        return Err(invalid(format!(
            "Qwen3.5 tensor `{}` dtype mismatch: expected {}, got {}",
            spec.name,
            expected_dtype,
            tensor.dtype()
        )));
    }
    if tensor.shape().dims() != spec.shape.as_slice() {
        return Err(invalid(format!(
            "Qwen3.5 tensor `{}` shape mismatch: expected {:?}, got {:?}",
            spec.name,
            spec.shape,
            tensor.shape().dims()
        )));
    }
    Ok(())
}

fn checkpoint_dtype(dtype: CheckpointDType) -> DType {
    match dtype {
        CheckpointDType::Bf16 => DType::BF16,
        CheckpointDType::F32 => DType::F32,
        // Native SafeTensors maps checkpoint U8 packed-FP4 bytes to the
        // semantically explicit E2M1Pair core dtype.
        CheckpointDType::U8 => DType::E2M1Pair,
        CheckpointDType::F8E4M3 => DType::F8E4M3,
    }
}

pub(super) fn load_qwen35_layer<L: ManifestAccess>(
    loader: &mut L,
    layer: &LayerArchitecture,
    hidden: usize,
) -> Result<Qwen35LayerWeights<Tensor>> {
    let prefix = format!("model.language_model.layers.{}", layer.index);
    let input_layernorm = loader.take(&format!("{prefix}.input_layernorm.weight"))?;
    let post_attention_layernorm =
        loader.take(&format!("{prefix}.post_attention_layernorm.weight"))?;
    let mixer = match layer.mixer {
        MixerKind::Gdn(geometry) => MixerWeights::Gdn(load_gdn(loader, &prefix, hidden, geometry)?),
        MixerKind::FullAttention(geometry) => MixerWeights::FullAttention(load_fp8_attention(
            loader,
            &format!("{prefix}.self_attn"),
            hidden,
            geometry,
        )?),
    };
    let moe = load_nvfp4_moe(
        loader,
        &format!("{prefix}.mlp"),
        hidden,
        layer.moe.experts,
        layer.moe.expert_intermediate_size,
        layer.moe.shared_expert_intermediate_size,
    )?;
    Ok(Qwen35LayerWeights {
        input_layernorm,
        post_attention_layernorm,
        mixer,
        moe,
    })
}

pub(super) fn load_qwen35_mtp<L: ManifestAccess>(
    loader: &mut L,
    architecture: &Qwen35MoeArchitecture,
    hidden: usize,
) -> Result<Qwen35MtpWeights<Tensor>> {
    let mtp_prefix = "mtp.layers.0";
    Ok(Qwen35MtpWeights {
        pre_fc_norm_embedding: loader.take("mtp.pre_fc_norm_embedding.weight")?,
        pre_fc_norm_hidden: loader.take("mtp.pre_fc_norm_hidden.weight")?,
        fc: load_dense_linear(loader, "mtp.fc", 2 * hidden, hidden)?,
        norm: loader.take("mtp.norm.weight")?,
        layer: Qwen35MtpLayerWeights {
            input_layernorm: loader.take(&format!("{mtp_prefix}.input_layernorm.weight"))?,
            post_attention_layernorm: loader
                .take(&format!("{mtp_prefix}.post_attention_layernorm.weight"))?,
            attention: load_dense_attention(
                loader,
                &format!("{mtp_prefix}.self_attn"),
                hidden,
                architecture.mtp.attention,
            )?,
            moe: load_dense_moe(
                loader,
                &format!("{mtp_prefix}.mlp"),
                hidden,
                architecture.mtp.moe.experts,
                architecture.mtp.moe.expert_intermediate_size,
                architecture.mtp.moe.shared_expert_intermediate_size,
            )?,
        },
    })
}

fn load_dense_linear<L: ManifestAccess>(
    loader: &mut L,
    prefix: &str,
    input_features: usize,
    output_features: usize,
) -> Result<DenseLinearWeights<Tensor>> {
    Ok(DenseLinearWeights {
        weight: loader.take(&format!("{prefix}.weight"))?,
        input_features,
        output_features,
    })
}

pub(super) fn load_nvfp4_linear_for_device<L: ManifestAccess>(
    loader: &mut L,
    prefix: &str,
    input_features: usize,
    output_features: usize,
) -> Result<Nvfp4LinearWeights<Tensor>> {
    let weights = Nvfp4LinearWeights {
        packed_weight: loader.take(&format!("{prefix}.weight"))?,
        block_scales: loader.take(&format!("{prefix}.weight_scale"))?,
        input_scale: loader.scalar(&format!("{prefix}.input_scale"))?,
        weight_scale_2: loader.scalar(&format!("{prefix}.weight_scale_2"))?,
        input_features,
        output_features,
    };
    weights.validate()?;
    Ok(weights)
}

fn load_fp8_linear<L: ManifestAccess>(
    loader: &mut L,
    prefix: &str,
    input_features: usize,
    output_features: usize,
) -> Result<StaticFp8LinearWeights<Tensor>> {
    let weights = StaticFp8LinearWeights {
        weight: loader.take(&format!("{prefix}.weight"))?,
        input_scale: loader.scalar(&format!("{prefix}.input_scale"))?,
        weight_scale: loader.scalar(&format!("{prefix}.weight_scale"))?,
        input_features,
        output_features,
    };
    weights.validate()?;
    Ok(weights)
}

fn load_dense_attention<L: ManifestAccess>(
    loader: &mut L,
    prefix: &str,
    hidden: usize,
    geometry: AttentionGeometry,
) -> Result<AttentionWeights<Tensor>> {
    Ok(AttentionWeights {
        q_proj: load_dense_linear(
            loader,
            &format!("{prefix}.q_proj"),
            hidden,
            geometry.query_projection_width,
        )?,
        k_proj: load_dense_linear(
            loader,
            &format!("{prefix}.k_proj"),
            hidden,
            geometry.kv_projection_width,
        )?,
        v_proj: load_dense_linear(
            loader,
            &format!("{prefix}.v_proj"),
            hidden,
            geometry.kv_projection_width,
        )?,
        o_proj: load_dense_linear(
            loader,
            &format!("{prefix}.o_proj"),
            geometry.query_projection_width / 2,
            hidden,
        )?,
        q_norm: loader.take(&format!("{prefix}.q_norm.weight"))?,
        k_norm: loader.take(&format!("{prefix}.k_norm.weight"))?,
        kv_cache_scales: None,
    })
}

fn load_fp8_attention<L: ManifestAccess>(
    loader: &mut L,
    prefix: &str,
    hidden: usize,
    geometry: AttentionGeometry,
) -> Result<AttentionWeights<Tensor, StaticFp8LinearWeights<Tensor>>> {
    let q_proj = load_fp8_linear(
        loader,
        &format!("{prefix}.q_proj"),
        hidden,
        geometry.query_projection_width,
    )?;
    let k_proj = load_fp8_linear(
        loader,
        &format!("{prefix}.k_proj"),
        hidden,
        geometry.kv_projection_width,
    )?;
    let v_proj = load_fp8_linear(
        loader,
        &format!("{prefix}.v_proj"),
        hidden,
        geometry.kv_projection_width,
    )?;
    if q_proj.input_scale.to_bits() != k_proj.input_scale.to_bits()
        || q_proj.input_scale.to_bits() != v_proj.input_scale.to_bits()
    {
        return Err(invalid(format!(
            "Qwen3.6 attention `{prefix}` Q/K/V input scales differ"
        )));
    }
    Ok(AttentionWeights {
        q_proj,
        k_proj,
        v_proj,
        o_proj: load_fp8_linear(
            loader,
            &format!("{prefix}.o_proj"),
            geometry.query_projection_width / 2,
            hidden,
        )?,
        q_norm: loader.take(&format!("{prefix}.q_norm.weight"))?,
        k_norm: loader.take(&format!("{prefix}.k_norm.weight"))?,
        kv_cache_scales: None,
    })
}

fn load_gdn<L: ManifestAccess>(
    loader: &mut L,
    layer_prefix: &str,
    hidden: usize,
    geometry: GdnGeometry,
) -> Result<GdnWeights<Tensor>> {
    let prefix = format!("{layer_prefix}.linear_attn");
    let in_proj_qkv = load_fp8_linear(
        loader,
        &format!("{prefix}.in_proj_qkv"),
        hidden,
        geometry.qkv_width,
    )?;
    let in_proj_z = load_fp8_linear(
        loader,
        &format!("{prefix}.in_proj_z"),
        hidden,
        geometry.z_width,
    )?;
    if in_proj_qkv.input_scale.to_bits() != in_proj_z.input_scale.to_bits() {
        return Err(invalid(format!(
            "Qwen3.6 GDN `{prefix}` QKV/Z input scales differ"
        )));
    }
    Ok(GdnWeights {
        a_log: loader.take(&format!("{prefix}.A_log"))?,
        conv1d: loader.take(&format!("{prefix}.conv1d.weight"))?,
        dt_bias: loader.take(&format!("{prefix}.dt_bias"))?,
        in_proj_a: load_dense_linear(
            loader,
            &format!("{prefix}.in_proj_a"),
            hidden,
            geometry.value_heads,
        )?,
        in_proj_b: load_dense_linear(
            loader,
            &format!("{prefix}.in_proj_b"),
            hidden,
            geometry.value_heads,
        )?,
        in_proj_qkv,
        in_proj_z,
        norm: loader.take(&format!("{prefix}.norm.weight"))?,
        out_proj: load_fp8_linear(
            loader,
            &format!("{prefix}.out_proj"),
            geometry.z_width,
            hidden,
        )?,
    })
}

fn load_nvfp4_expert<L: ManifestAccess>(
    loader: &mut L,
    prefix: &str,
    hidden: usize,
    intermediate: usize,
) -> Result<SwiGluExpertWeights<Nvfp4LinearWeights<Tensor>>> {
    Ok(SwiGluExpertWeights {
        gate_proj: load_nvfp4_linear_for_device(
            loader,
            &format!("{prefix}.gate_proj"),
            hidden,
            intermediate,
        )?,
        up_proj: load_nvfp4_linear_for_device(
            loader,
            &format!("{prefix}.up_proj"),
            hidden,
            intermediate,
        )?,
        down_proj: load_nvfp4_linear_for_device(
            loader,
            &format!("{prefix}.down_proj"),
            intermediate,
            hidden,
        )?,
    })
}

fn load_dense_expert<L: ManifestAccess>(
    loader: &mut L,
    prefix: &str,
    hidden: usize,
    intermediate: usize,
) -> Result<SwiGluExpertWeights<DenseLinearWeights<Tensor>>> {
    Ok(SwiGluExpertWeights {
        gate_proj: load_dense_linear(loader, &format!("{prefix}.gate_proj"), hidden, intermediate)?,
        up_proj: load_dense_linear(loader, &format!("{prefix}.up_proj"), hidden, intermediate)?,
        down_proj: load_dense_linear(loader, &format!("{prefix}.down_proj"), intermediate, hidden)?,
    })
}

pub(super) fn load_nvfp4_moe<L: ManifestAccess>(
    loader: &mut L,
    prefix: &str,
    hidden: usize,
    experts: usize,
    expert_intermediate: usize,
    shared_intermediate: usize,
) -> Result<RoutedMoeWeights<Tensor, Nvfp4LinearWeights<Tensor>>> {
    let mut expert_weights = Vec::with_capacity(experts);
    for expert in 0..experts {
        expert_weights.push(load_nvfp4_expert(
            loader,
            &format!("{prefix}.experts.{expert}"),
            hidden,
            expert_intermediate,
        )?);
    }
    Ok(RoutedMoeWeights {
        router: load_dense_linear(loader, &format!("{prefix}.gate"), hidden, experts)?,
        experts: expert_weights,
        shared_expert: load_nvfp4_expert(
            loader,
            &format!("{prefix}.shared_expert"),
            hidden,
            shared_intermediate,
        )?,
        shared_expert_gate: load_dense_linear(
            loader,
            &format!("{prefix}.shared_expert_gate"),
            hidden,
            1,
        )?,
    })
}

fn load_dense_moe<L: ManifestAccess>(
    loader: &mut L,
    prefix: &str,
    hidden: usize,
    experts: usize,
    expert_intermediate: usize,
    shared_intermediate: usize,
) -> Result<DenseMoeWeights<Tensor>> {
    Ok(DenseMoeWeights {
        router: load_dense_linear(loader, &format!("{prefix}.gate"), hidden, experts)?,
        routed_experts: PackedDenseExperts {
            gate_up_proj: loader.take(&format!("{prefix}.experts.gate_up_proj"))?,
            down_proj: loader.take(&format!("{prefix}.experts.down_proj"))?,
            experts,
            hidden_size: hidden,
            intermediate_size: expert_intermediate,
        },
        shared_expert: load_dense_expert(
            loader,
            &format!("{prefix}.shared_expert"),
            hidden,
            shared_intermediate,
        )?,
        shared_expert_gate: load_dense_linear(
            loader,
            &format!("{prefix}.shared_expert_gate"),
            hidden,
            1,
        )?,
    })
}

fn invalid(message: impl Into<String>) -> Error {
    Error::Other(message.into())
}

#[cfg(test)]
mod tests {
    use super::super::weights::WeightRole;
    use super::*;

    fn spec(name: &str, dtype: CheckpointDType, shape: &[usize]) -> WeightSpec {
        WeightSpec {
            name: name.into(),
            dtype,
            shape: shape.to_vec(),
            role: WeightRole::Mixer,
        }
    }

    #[test]
    fn minimal_host_source_reads_scalar_and_discards_only_visual_leftovers() {
        let mut tensors = HashMap::from([
            ("norm".into(), Tensor::zeros(vec![2], DType::BF16)),
            (
                "scale".into(),
                Tensor::from_f32(Vec::<usize>::new(), &[0.25]).unwrap(),
            ),
            (
                "model.visual.blocks.0.weight".into(),
                Tensor::zeros(vec![1], DType::F32),
            ),
        ]);
        let mut source = HostTensorSource {
            tensors: std::mem::take(&mut tensors),
        };
        let tensor = source
            .take_tensor(&spec("norm", CheckpointDType::Bf16, &[2]))
            .unwrap();
        assert_eq!(tensor.shape().dims(), &[2]);
        assert_eq!(
            source
                .take_scalar_f32(&spec("scale", CheckpointDType::F32, &[]))
                .unwrap(),
            0.25
        );
        assert!(source.finish().is_ok());
    }

    #[test]
    fn host_source_rejects_missing_wrong_dtype_and_wrong_shape() {
        let mut missing = HostTensorSource {
            tensors: HashMap::new(),
        };
        assert!(missing
            .take_tensor(&spec("missing", CheckpointDType::Bf16, &[2]))
            .is_err());

        let mut wrong_dtype = HostTensorSource {
            tensors: HashMap::from([("x".into(), Tensor::zeros(vec![2], DType::F32))]),
        };
        assert!(wrong_dtype
            .take_tensor(&spec("x", CheckpointDType::Bf16, &[2]))
            .is_err());

        let mut wrong_shape = HostTensorSource {
            tensors: HashMap::from([("x".into(), Tensor::zeros(vec![3], DType::BF16))]),
        };
        assert!(wrong_shape
            .take_tensor(&spec("x", CheckpointDType::Bf16, &[2]))
            .is_err());
    }

    #[test]
    fn host_source_accepts_native_packed_fp4_and_rejects_nonvisual_leftovers() {
        let mut source = HostTensorSource {
            tensors: HashMap::from([
                ("packed".into(), Tensor::zeros(vec![2, 3], DType::E2M1Pair)),
                ("unexpected.bias".into(), Tensor::zeros(vec![1], DType::F32)),
            ]),
        };
        source
            .take_tensor(&spec("packed", CheckpointDType::U8, &[2, 3]))
            .unwrap();
        let error = source.finish().unwrap_err().to_string();
        assert!(error.contains("unexpected.bias"), "{error}");
    }

    #[test]
    fn mtp_dense_attention_omits_unavailable_kv_scales() {
        let prefix = "model.language_model.layers.3.self_attn";
        let geometry = AttentionGeometry {
            query_heads: 16,
            kv_heads: 2,
            head_dim: 256,
            rotary_dim: 64,
            query_projection_width: 8_192,
            kv_projection_width: 512,
            output_gate: true,
        };
        let mut specs = Vec::new();
        super::super::weights::push_attention(
            &mut specs,
            prefix,
            2_048,
            geometry.query_projection_width,
            geometry.kv_projection_width,
            geometry.head_dim,
        );
        let mut remaining_specs = specs
            .iter()
            .cloned()
            .map(|spec| (spec.name.clone(), spec))
            .collect::<HashMap<_, _>>();
        let mut tensors = HashMap::new();
        for spec in specs {
            let tensor = Tensor::zeros(spec.shape.clone(), DType::BF16);
            tensors.insert(spec.name, tensor);
        }
        let mut loader = ManifestLoader {
            source: HostTensorSource { tensors },
            remaining_specs: std::mem::take(&mut remaining_specs),
        };
        let attention = load_dense_attention(&mut loader, prefix, 2_048, geometry).unwrap();
        assert!(attention.kv_cache_scales.is_none());
        loader.finish().unwrap();
    }
}
