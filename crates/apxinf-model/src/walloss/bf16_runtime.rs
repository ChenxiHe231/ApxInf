//! Owning BF16 VLA runtime for fixed-shape WallOSS inference.

use std::cell::RefCell;
use std::path::Path;
use std::rc::Rc;
use std::sync::Arc;

use apxinf_core::{
    Backend, DType, Device, Error, Graph, NormalGenerator, Result, SamplingBackend, Tensor,
};

use crate::accelerator::cuda::tuning;
use crate::auto::{LoadOptions, LoadedModel, ModelPrecision};
use crate::vla::{
    Action, InferenceSpec, InitialLatent, PreparedInference, VisionObservation, VlaRequest,
    VlaRuntime,
};

use super::backend::{kernels, transfers, DeviceBuffer, RuntimeBackend};
use super::bf16_executor::{
    action_stack, language_prefix, solver_update, vision_tower, TransformerWeights,
    VisionTowerWeights,
};
use super::{
    multimodal_position_ids, sinusoidal_time_embedding, solver_times, DeviceVisionGeometry,
    VisionGeometry, WallossConfig, WallossDynamicFp8Weights, WallossWeights,
};

const DEFAULT_GRIDS: [[usize; 3]; 2] = [[1, 18, 18], [1, 18, 18]];
const BF16_WORKSPACE_BYTES: usize = 12 * 1024 * 1024 * 1024;

pub struct WallossBf16Runtime {
    backend: Arc<RuntimeBackend>,
    config: Arc<WallossConfig>,
    weights: Arc<WallossDeviceWeights>,
    grids: Arc<Vec<[usize; 3]>>,
    geometry: Arc<DeviceVisionGeometry>,
    prepared: RefCell<Option<(InferenceSpec, Rc<WallossPreparedInference>)>>,
}

pub struct WallossPreparedInference {
    spec: InferenceSpec,
    backend: Arc<RuntimeBackend>,
    config: Arc<WallossConfig>,
    weights: Arc<WallossDeviceWeights>,
    grids: Arc<Vec<[usize; 3]>>,
    geometry: Arc<DeviceVisionGeometry>,
    noise: Tensor,
    normal_generator: RefCell<Box<dyn NormalGenerator>>,
    workspace: kernels::GraphWorkspace,
    captured: RefCell<Option<WallossBf16CapturedGraph>>,
}

struct WallossHostInputs {
    patches: Tensor,
    prefix_ids: Vec<u32>,
    vision_row_map: Vec<u32>,
    prefix_position_ids: Vec<u32>,
    action_position_ids: Vec<u32>,
    initial_state: Option<Tensor>,
    action_mask: Tensor,
    time_embeddings: Vec<Tensor>,
}

struct WallossDeviceInputs {
    patches: Tensor,
    prefix_ids: DeviceBuffer,
    vision_row_map: DeviceBuffer,
    prefix_position_ids: DeviceBuffer,
    action_position_ids: DeviceBuffer,
    initial_state: Tensor,
    action_mask: Tensor,
    time_embeddings: Vec<Tensor>,
    prefix_tokens: usize,
    generated_latent: bool,
}

struct WallossBf16CapturedGraph {
    graph: Box<dyn Graph>,
    output: Tensor,
    inputs: WallossDeviceInputs,
}

enum WallossDeviceWeights {
    Bf16(WallossWeights),
    DynamicFp8(WallossDynamicFp8Weights),
}

impl WallossPreparedInference {
    fn run_impl(&self, request: &VlaRequest<'_>) -> Result<Action> {
        let host = self.prepare_host_inputs(request)?;
        if let Some(captured) = self.captured.borrow_mut().as_mut() {
            self.update_device_inputs(&mut captured.inputs, &host)?;
            captured.graph.replay()?;
            return Ok(Action::new(captured.output.clone()));
        }

        let inputs = self.upload_inputs(&host)?;
        let eager_output =
            kernels::prepare_with_workspace(&self.workspace, || self.execute(&inputs))?;
        if std::env::var_os("APXINF_WALLOSS_NO_GRAPH").is_some() {
            return Ok(Action::new(eager_output));
        }
        self.backend.synchronize()?;
        drop(eager_output);

        self.backend.begin_capture()?;
        let output = match kernels::with_workspace(&self.workspace, || self.execute(&inputs)) {
            Ok(output) => output,
            Err(error) => {
                let _ = self.backend.end_capture();
                return Err(error);
            }
        };
        let graph = self.backend.end_capture()?;
        *self.captured.borrow_mut() = Some(WallossBf16CapturedGraph {
            graph,
            output: output.clone(),
            inputs,
        });
        Ok(Action::new(output))
    }

    fn prepare_host_inputs(&self, request: &VlaRequest<'_>) -> Result<WallossHostInputs> {
        let observation = request.observation;
        observation.validate()?;
        if !self.spec.matches(observation) {
            return Err(Error::Other(format!(
                "prepared walloss spec {:?} does not match observation {:?}",
                self.spec,
                observation.inference_spec()
            )));
        }
        let patches =
            match &observation.vision {
                VisionObservation::Patches(value) => normalize_host_bf16(
                    value,
                    vec![
                        self.geometry.patch_order.len() / 4,
                        patch_width(&self.config),
                    ],
                    "patches",
                )?,
                VisionObservation::RgbU8 { .. } => return Err(Error::Other(
                    "walloss raw RGB preprocessing is not connected yet; pass preprocessed patches"
                        .into(),
                )),
            };
        let action_tokens = self.config.action.action_horizon;
        let prefix_tokens = observation.token_ids.len() - action_tokens;
        let prefix_ids = observation.token_ids[..prefix_tokens].to_vec();
        let vision_rows = self.geometry.reverse_indices.len() / std::mem::size_of::<u32>();
        let mut vision_row_map = vec![u32::MAX; prefix_tokens];
        let mut vision_row = 0u32;
        for (row, &token) in prefix_ids.iter().enumerate() {
            if token == self.config.image_token_id {
                vision_row_map[row] = vision_row;
                vision_row += 1;
            }
        }
        if vision_row as usize != vision_rows {
            return Err(Error::Other(format!(
                "walloss prompt has {vision_row} image tokens, expected {vision_rows}"
            )));
        }
        let position_ids = multimodal_position_ids(
            &observation.token_ids,
            &self.grids,
            self.config.image_token_id,
            self.config.vision.spatial_merge_size,
        )?;
        let initial_state = match request.initial_latent {
            InitialLatent::Provided(value) => Some(normalize_host_bf16(
                value,
                vec![action_tokens, self.config.action.action_dim],
                "initial latent",
            )?),
            InitialLatent::Generate { rng } => {
                self.normal_generator.borrow_mut().generate(rng)?;
                None
            }
        };
        let mask_host = match observation.action_mask.as_ref() {
            Some(value) => normalize_host_bf16(
                value,
                vec![action_tokens, self.config.action.action_dim],
                "action mask",
            )?,
            None => Tensor::from_bf16(
                vec![action_tokens, self.config.action.action_dim],
                &vec![half::bf16::ONE; action_tokens * self.config.action.action_dim],
            )?,
        };
        let times = solver_times(
            self.config.action.solver_steps,
            self.config.action.scheduler_s,
            1.0,
        )?;
        let mut time_embeddings = Vec::with_capacity(self.config.action.solver_steps);
        for &time in times.iter().take(self.config.action.solver_steps) {
            let embedding = sinusoidal_time_embedding(time, self.config.action.hidden_size)?;
            let repeated = embedding
                .iter()
                .copied()
                .cycle()
                .take(action_tokens * embedding.len())
                .map(half::bf16::from_f32)
                .collect::<Vec<_>>();
            time_embeddings.push(Tensor::from_bf16(
                vec![action_tokens, self.config.action.hidden_size],
                &repeated,
            )?);
        }
        Ok(WallossHostInputs {
            patches,
            prefix_ids,
            vision_row_map,
            prefix_position_ids: position_ids[..prefix_tokens * 3].to_vec(),
            action_position_ids: position_ids[prefix_tokens * 3..].to_vec(),
            initial_state,
            action_mask: mask_host,
            time_embeddings,
        })
    }

    fn upload_inputs(&self, host: &WallossHostInputs) -> Result<WallossDeviceInputs> {
        let device = self.backend.context().device_id();
        Ok(WallossDeviceInputs {
            patches: self.backend.to_device(&host.patches)?,
            prefix_ids: upload_u32(device, &host.prefix_ids)?,
            vision_row_map: upload_u32(device, &host.vision_row_map)?,
            prefix_position_ids: upload_u32(device, &host.prefix_position_ids)?,
            action_position_ids: upload_u32(device, &host.action_position_ids)?,
            initial_state: match &host.initial_state {
                Some(state) => self.backend.to_device(state)?,
                None => self.noise.clone(),
            },
            action_mask: self.backend.to_device(&host.action_mask)?,
            time_embeddings: host
                .time_embeddings
                .iter()
                .map(|value| self.backend.to_device(value))
                .collect::<Result<Vec<_>>>()?,
            prefix_tokens: host.prefix_ids.len(),
            generated_latent: host.initial_state.is_none(),
        })
    }

    fn update_device_inputs(
        &self,
        device: &mut WallossDeviceInputs,
        host: &WallossHostInputs,
    ) -> Result<()> {
        if device.generated_latent != host.initial_state.is_none() {
            return Err(Error::Other(
                "walloss captured inference cannot switch initial-latent mode".into(),
            ));
        }
        self.backend.synchronize()?;
        transfers::copy_cpu_to_cuda(&host.patches, &device.patches)?;
        transfers::copy_cpu_to_cuda(&host.action_mask, &device.action_mask)?;
        if let Some(state) = &host.initial_state {
            transfers::copy_cpu_to_cuda(state, &device.initial_state)?;
        }
        copy_u32(&device.prefix_ids, &host.prefix_ids)?;
        copy_u32(&device.vision_row_map, &host.vision_row_map)?;
        copy_u32(&device.prefix_position_ids, &host.prefix_position_ids)?;
        copy_u32(&device.action_position_ids, &host.action_position_ids)?;
        Ok(())
    }

    fn execute(&self, inputs: &WallossDeviceInputs) -> Result<Tensor> {
        match self.weights.as_ref() {
            WallossDeviceWeights::Bf16(weights) => self.execute_with(
                inputs,
                &weights.vision,
                &weights.language_layers,
                &weights.action_layers,
                &weights.token_embedding,
                &weights.action,
                &weights.action_norm,
            ),
            WallossDeviceWeights::DynamicFp8(weights) => self.execute_with(
                inputs,
                &weights.vision,
                &weights.language_layers,
                &weights.action_layers,
                &weights.token_embedding,
                &weights.action,
                &weights.action_norm,
            ),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn execute_with<V: VisionTowerWeights, L: TransformerWeights, A: TransformerWeights>(
        &self,
        inputs: &WallossDeviceInputs,
        vision_weights: &V,
        language_weights: &[L],
        transformer_action_weights: &[A],
        token_embedding: &Tensor,
        action_weights: &super::WallossActionWeights,
        action_norm: &Tensor,
    ) -> Result<Tensor> {
        let context = self.backend.context();
        let vision = vision_tower(
            context,
            &self.config.vision,
            vision_weights,
            &self.geometry,
            &inputs.patches,
        )?;
        let prefix = language_prefix(
            context,
            &self.config.text,
            language_weights,
            token_embedding,
            &inputs.prefix_ids,
            &vision,
            &inputs.vision_row_map,
            &inputs.prefix_position_ids,
            inputs.prefix_tokens,
            inputs.prefix_tokens + self.config.action.action_horizon,
        )?;
        self.execute_action(
            inputs,
            transformer_action_weights,
            action_weights,
            action_norm,
            &prefix,
        )
    }

    fn execute_action<A: TransformerWeights>(
        &self,
        inputs: &WallossDeviceInputs,
        transformer_action_weights: &[A],
        action_weights: &super::WallossActionWeights,
        action_norm: &Tensor,
        prefix: &super::bf16_executor::PrefixCache,
    ) -> Result<Tensor> {
        let context = self.backend.context();
        let times = solver_times(
            self.config.action.solver_steps,
            self.config.action.scheduler_s,
            1.0,
        )?;
        let mut state = inputs.initial_state.clone();
        for (step, time_embedding) in inputs.time_embeddings.iter().enumerate() {
            let velocity = action_stack(
                context,
                &self.config.text,
                transformer_action_weights,
                action_weights,
                action_norm,
                &prefix,
                &state,
                &inputs.action_mask,
                time_embedding,
                &inputs.action_position_ids,
            )?;
            state = solver_update(context, &state, &velocity, times[step + 1] - times[step])?;
        }
        Ok(state)
    }
}

impl PreparedInference for WallossPreparedInference {
    fn spec(&self) -> &InferenceSpec {
        &self.spec
    }

    fn run(&self, request: &VlaRequest<'_>) -> Result<Action> {
        self.run_impl(request)
    }
}

impl WallossBf16Runtime {
    fn build_prepared(&self, spec: InferenceSpec) -> Result<WallossPreparedInference> {
        spec.validate()?;
        if spec.token_count <= self.config.action.action_horizon {
            return Err(Error::Other(
                "walloss token sequence must contain a language prefix and action suffix".into(),
            ));
        }
        let noise_host = Tensor::zeros(
            (
                self.config.action.action_horizon,
                self.config.action.action_dim,
            ),
            DType::BF16,
        );
        let noise = self.backend.to_device(&noise_host)?;
        let normal_generator = self.backend.create_normal_generator(noise.clone())?;
        let workspace =
            kernels::GraphWorkspace::new(BF16_WORKSPACE_BYTES, self.backend.context().device_id())?;
        Ok(WallossPreparedInference {
            spec,
            backend: Arc::clone(&self.backend),
            config: Arc::clone(&self.config),
            weights: Arc::clone(&self.weights),
            grids: Arc::clone(&self.grids),
            geometry: Arc::clone(&self.geometry),
            noise,
            normal_generator: RefCell::new(normal_generator),
            workspace,
            captured: RefCell::new(None),
        })
    }
}

impl VlaRuntime for WallossBf16Runtime {
    fn contract(&self) -> crate::VlaContract {
        crate::VlaContract {
            action_shape: [
                self.config.action.action_horizon,
                self.config.action.action_dim,
            ],
            patch_shape: [2 * 18 * 18, patch_width(&self.config)],
            max_token_len: self.config.text.max_position_embeddings,
            num_views: 2,
            image_size: 18 * self.config.vision.patch_size,
            patch_size: self.config.vision.patch_size,
            accepts_rgb_u8: false,
        }
    }

    fn infer(&self, request: &VlaRequest<'_>) -> Result<Action> {
        let spec = request.observation.inference_spec();
        let prepared = {
            let mut cache = self.prepared.borrow_mut();
            if cache.as_ref().is_none_or(|(cached, _)| *cached != spec) {
                *cache = Some((spec, Rc::new(self.build_prepared(spec)?)));
            }
            Rc::clone(&cache.as_ref().unwrap().1)
        };
        prepared.run(request)
    }

    fn prepare(&self, spec: &InferenceSpec) -> Result<Box<dyn PreparedInference>> {
        Ok(Box::new(self.build_prepared(*spec)?))
    }

    fn infer_host_f32(&self, request: &VlaRequest<'_>) -> Result<Vec<f32>> {
        let output = self.infer(request)?;
        self.backend.to_cpu(output.tensor())?.to_f32_vec()
    }
}

pub(super) fn load_registered(
    path: &Path,
    _device: Device,
    backend: Arc<dyn Backend>,
    options: &LoadOptions,
) -> Result<LoadedModel> {
    if options.config.is_some() {
        return Err(Error::Other(
            "walloss does not support action_horizon/num_views config overrides".into(),
        ));
    }
    if !matches!(
        options.precision,
        ModelPrecision::Auto | ModelPrecision::Bf16 | ModelPrecision::Fp8
    ) {
        return Err(Error::Other("walloss supports BF16 and FP8 on CUDA".into()));
    }
    if options.calibration_path.is_some() || options.uniform_fp8_scale.is_some() {
        return Err(Error::Other(
            "walloss FP8 uses dynamic rowwise activation scales and does not support calibration"
                .into(),
        ));
    }
    let backend = crate::accelerator::cuda::downcast_arc(backend)
        .ok_or_else(|| Error::Other("walloss is only registered for CUDA".into()))?;
    let root = if path.is_dir() {
        path
    } else {
        path.parent().unwrap_or_else(|| Path::new("."))
    };
    let mut config = WallossConfig::from_json_file(&root.join("config.json"))?;
    let host_weights = WallossWeights::from_safetensors(&mut config, path)?;
    let dynamic_fp8 = matches!(options.precision, ModelPrecision::Fp8);
    let tuning_path = options.tuning_path.clone().or_else(|| {
        if dynamic_fp8 {
            return None;
        }
        let candidate = root.join("tactics.json");
        candidate.is_file().then_some(candidate)
    });
    if let Some(path) = tuning_path.as_deref() {
        let database = tuning::TuningDb::from_json_file(path)?;
        crate::accelerator::cuda::kernels::gemm::install_tuning_db(backend.context(), &database)?;
    }
    let weights = match options.precision {
        ModelPrecision::Fp8 => WallossDeviceWeights::DynamicFp8(
            WallossDynamicFp8Weights::from_host(&host_weights, &*backend)?,
        ),
        ModelPrecision::Auto | ModelPrecision::Bf16 => {
            WallossDeviceWeights::Bf16(host_weights.to_bf16_device(&*backend)?)
        }
        ModelPrecision::W8A8 => unreachable!(),
    };
    let weights = Arc::new(weights);
    let grids = Arc::new(DEFAULT_GRIDS.to_vec());
    let host_geometry = VisionGeometry::new(&config.vision, &grids)?;
    let geometry = Arc::new(host_geometry.upload(backend.context())?);
    Ok(LoadedModel::Vla(Box::new(WallossBf16Runtime {
        backend,
        config: Arc::new(config),
        weights,
        grids,
        geometry,
        prepared: RefCell::new(None),
    })))
}

fn patch_width(config: &WallossConfig) -> usize {
    3 * config.vision.temporal_patch_size * config.vision.patch_size * config.vision.patch_size
}

fn normalize_host_bf16(value: &Tensor, shape: Vec<usize>, name: &str) -> Result<Tensor> {
    if value.shape().dims() != shape {
        return Err(Error::Other(format!(
            "walloss {name} shape {:?}, expected {shape:?}",
            value.shape().dims()
        )));
    }
    let values = value
        .to_f32_vec()?
        .into_iter()
        .map(half::bf16::from_f32)
        .collect::<Vec<_>>();
    Tensor::from_bf16(shape, &values)
}

fn upload_u32(device_id: usize, values: &[u32]) -> Result<DeviceBuffer> {
    let bytes = values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect::<Vec<_>>();
    let buffer = DeviceBuffer::alloc_zeros(bytes.len(), device_id).map_err(Error::Cuda)?;
    buffer.copy_from_host(&bytes).map_err(Error::Cuda)?;
    Ok(buffer)
}

fn copy_u32(buffer: &DeviceBuffer, values: &[u32]) -> Result<()> {
    let bytes = values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect::<Vec<_>>();
    if bytes.len() != buffer.len() {
        return Err(Error::Other(format!(
            "walloss captured u32 input has {} bytes, expected {}",
            bytes.len(),
            buffer.len()
        )));
    }
    buffer.copy_from_host(&bytes).map_err(Error::Cuda)
}
