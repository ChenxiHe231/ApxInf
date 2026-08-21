use apxinf_core::{DType, Device, Error, Result, Shape, Tensor};

use crate::buffer::CudaBuffer;
use crate::context::CudaContext;
use crate::ffi;
use crate::kernels::contracts::gpu_ptr;
use crate::tuning::{
    DeviceFingerprint, Epilogue, GemmLayout, GemmOp, GemmTuningKey, ScaleMode, TacticBackend,
    TacticStore, TuningDType, TuningDb,
};

#[derive(Clone, Copy, Debug)]
pub struct CutlassTacticTiming {
    pub tactic: i32,
    pub milliseconds: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct CublasLtAlgorithmTiming {
    pub heuristic_rank: i32,
    pub milliseconds: f64,
}

#[derive(Clone, Copy, Debug)]
pub struct ColdL2TuningMetadata {
    pub l2_cache_bytes: usize,
    pub eviction_buffer_bytes: usize,
}

pub fn cold_l2_tuning_metadata(ctx: &CudaContext) -> Result<ColdL2TuningMetadata> {
    let mut l2_cache_bytes = 0i32;
    unsafe {
        ffi::check_cuda(ffi::cudaDeviceGetAttribute(
            &mut l2_cache_bytes,
            ffi::CUDA_DEV_ATTR_L2_CACHE_SIZE,
            ctx.device_id() as i32,
        ))
        .map_err(Error::Cuda)?;
    }
    let l2_cache_bytes = usize::try_from(l2_cache_bytes)
        .ok()
        .filter(|bytes| *bytes > 0)
        .ok_or_else(|| Error::Other("CUDA reported an empty L2 cache".into()))?;
    let eviction_buffer_bytes = l2_cache_bytes
        .checked_mul(4)
        .and_then(|bytes| bytes.checked_add(255))
        .map(|bytes| bytes & !255usize)
        .ok_or_else(|| Error::Other("cold-L2 eviction buffer size overflow".into()))?;
    Ok(ColdL2TuningMetadata {
        l2_cache_bytes,
        eviction_buffer_bytes,
    })
}

struct ColdL2Evictor {
    buffer: CudaBuffer,
    metadata: ColdL2TuningMetadata,
    seed: u32,
}

impl ColdL2Evictor {
    fn new(ctx: &CudaContext) -> Result<Self> {
        let metadata = cold_l2_tuning_metadata(ctx)?;
        let buffer = CudaBuffer::alloc_zeros(metadata.eviction_buffer_bytes, ctx.device_id())
            .map_err(Error::Cuda)?;
        Ok(Self {
            buffer,
            metadata,
            seed: 0,
        })
    }

    fn evict(&mut self, ctx: &CudaContext) -> Result<()> {
        self.seed = self.seed.wrapping_add(1);
        unsafe {
            ffi::check_cuda(ffi::apxinf_static_evict_l2(
                self.buffer.ptr(),
                self.metadata.eviction_buffer_bytes,
                self.seed,
                ctx.stream().handle(),
            ))
            .map_err(Error::Cuda)
        }
    }
}

struct CudaEventPair {
    start: ffi::cudaEvent_t,
    stop: ffi::cudaEvent_t,
}

impl CudaEventPair {
    fn new() -> Result<Self> {
        let mut events = Self {
            start: std::ptr::null_mut(),
            stop: std::ptr::null_mut(),
        };
        unsafe {
            ffi::check_cuda(ffi::cudaEventCreate(&mut events.start)).map_err(Error::Cuda)?;
            if let Err(error) = ffi::check_cuda(ffi::cudaEventCreate(&mut events.stop)) {
                let _ = ffi::cudaEventDestroy(events.start);
                return Err(Error::Cuda(error));
            }
        }
        Ok(events)
    }

    fn measure(
        &self,
        ctx: &CudaContext,
        evictor: &mut ColdL2Evictor,
        launch: impl FnOnce() -> Result<()>,
    ) -> Result<f64> {
        evictor.evict(ctx)?;
        unsafe {
            ffi::check_cuda(ffi::cudaEventRecord(self.start, ctx.stream().handle()))
                .map_err(Error::Cuda)?;
        }
        launch()?;
        let mut milliseconds = 0.0f32;
        unsafe {
            ffi::check_cuda(ffi::cudaEventRecord(self.stop, ctx.stream().handle()))
                .map_err(Error::Cuda)?;
            ffi::check_cuda(ffi::cudaEventSynchronize(self.stop)).map_err(Error::Cuda)?;
            ffi::check_cuda(ffi::cudaEventElapsedTime(
                &mut milliseconds,
                self.start,
                self.stop,
            ))
            .map_err(Error::Cuda)?;
        }
        Ok(f64::from(milliseconds))
    }
}

impl Drop for CudaEventPair {
    fn drop(&mut self) {
        unsafe {
            if !self.start.is_null() {
                let _ = ffi::cudaEventDestroy(self.start);
            }
            if !self.stop.is_null() {
                let _ = ffi::cudaEventDestroy(self.stop);
            }
        }
    }
}

fn validate_fp8_dual_geglu_record(
    op: GemmOp,
    m: usize,
    n: usize,
    k: usize,
    tactic: i32,
) -> Result<()> {
    if op != GemmOp::Fp8F16 || !matches!(m, 522 | 533) || (n, k) != (32768, 2048) || tactic != 0 {
        return Err(Error::Other(format!(
            "FP8 dual GeGLU backend requires M522 or M533, N32768/K2048, tactic 0; got M{m}/N{n}/K{k} tactic {tactic}"
        )));
    }
    Ok(())
}

fn validate_bf16_dual_geglu_record(
    op: GemmOp,
    m: usize,
    n: usize,
    k: usize,
    tactic: i32,
    expected_m: usize,
    experiment: &str,
) -> Result<()> {
    if op != GemmOp::Bf16 || (m, n, k) != (expected_m, 32768, 2048) || tactic != 0 {
        return Err(Error::Other(format!(
            "{experiment} backend requires exact BF16 M{expected_m}/N32768/K2048 tactic 0"
        )));
    }
    Ok(())
}

/// Validate and install a read-only tactic database before graph capture.
pub fn install_tuning_db(ctx: &CudaContext, database: &TuningDb) -> Result<()> {
    install_tuning_dbs(ctx, std::slice::from_ref(database))
}

/// Validate and merge all databases before publishing one immutable store.
/// This is the service-startup path when tactics are split across files.
pub fn install_tuning_dbs(ctx: &CudaContext, databases: &[TuningDb]) -> Result<()> {
    let stores = databases
        .iter()
        .map(|database| database.build_store(ctx.caps(), ctx.library_versions()))
        .collect::<Result<Vec<_>>>()?;
    let store = TacticStore::merge(stores)?;
    if let Some(installed) = crate::tuning::installed() {
        return if installed == &store {
            Ok(())
        } else {
            Err(Error::Other(
                "a different CUDA tactic store is already installed".into(),
            ))
        };
    }
    // The current cuBLASLt C ABI caches plans internally. Seed its immutable
    // startup configuration before publishing the Rust store so inference
    // never mutates tactic state.
    for record in store.gemm_records() {
        match record.tactic.backend {
            TacticBackend::Cutlass => {}
            TacticBackend::CublasLt => match record.key.op {
                GemmOp::Bf16 => super::bf16::set_cublaslt_gemm_heuristic(
                    record.key.m,
                    record.key.n,
                    record.key.k,
                    record.tactic.value,
                )?,
                GemmOp::Fp8F16 => set_cublaslt_gemm_heuristic(
                    record.key.m,
                    record.key.n,
                    record.key.k,
                    record.tactic.value,
                )?,
                _ => {}
            },
            TacticBackend::CublasLtCustom => match record.key.op {
                GemmOp::Bf16 => super::bf16::set_cublaslt_gemm_custom(
                    record.key.m,
                    record.key.n,
                    record.key.k,
                    record.tactic.value,
                )?,
                GemmOp::Fp8F16 => set_cublaslt_gemm_custom(
                    record.key.m,
                    record.key.n,
                    record.key.k,
                    record.tactic.value,
                )?,
                op => {
                    return Err(Error::Other(format!(
                        "cublaslt_custom does not support tuning operation {op:?}"
                    )))
                }
            },
            TacticBackend::CublasLtCustomBias => match record.key.op {
                GemmOp::Fp8F16 => set_cublaslt_gemm_bias_custom(
                    record.key.m,
                    record.key.n,
                    record.key.k,
                    record.tactic.value,
                )?,
                op => {
                    return Err(Error::Other(format!(
                        "cublaslt_custom_bias does not support tuning operation {op:?}"
                    )))
                }
            },
            TacticBackend::CublasLtCustomSplitSerial => match record.key.op {
                GemmOp::Bf16 => super::bf16::set_cublaslt_gemm_split_custom(
                    record.key.m,
                    record.key.n,
                    record.key.k,
                    record.tactic.value,
                )?,
                GemmOp::Fp8F16 => set_cublaslt_gemm_split_custom(
                    record.key.m,
                    record.key.n,
                    record.key.k,
                    record.tactic.value,
                )?,
                op => {
                    return Err(Error::Other(format!(
                        "cublaslt_custom_split_serial does not support tuning operation {op:?}"
                    )))
                }
            },
            TacticBackend::CublasLtCustomSplitGeGluCutlass
            | TacticBackend::CublasLtCustomSplitGeGluCutlass2SmAuto
            | TacticBackend::CublasLtCustomSplitGeGluCutlass2SmStage3
            | TacticBackend::CublasLtCustomSplitGeGluCutlassM522Explicit2Sm => {
                match record.key.op {
                    GemmOp::Fp8F16 => set_cublaslt_gemm_split_custom(
                        record.key.m,
                        record.key.n,
                        record.key.k,
                        record.tactic.value,
                    )?,
                    op => {
                        return Err(Error::Other(format!(
                            "fused GeGLU tactic does not support tuning operation {op:?}"
                        )))
                    }
                }
            }
            TacticBackend::CutlassFp8DualGeGlu => {
                validate_fp8_dual_geglu_record(
                    record.key.op,
                    record.key.m,
                    record.key.n,
                    record.key.k,
                    record.tactic.value,
                )?;
            }
            TacticBackend::CutlassBf16DualGeGluM522 => {
                validate_bf16_dual_geglu_record(
                    record.key.op,
                    record.key.m,
                    record.key.n,
                    record.key.k,
                    record.tactic.value,
                    522,
                    "BF16 dual GeGLU",
                )?;
            }
            TacticBackend::CutlassBf16DualGeGluM533 => {
                validate_bf16_dual_geglu_record(
                    record.key.op,
                    record.key.m,
                    record.key.n,
                    record.key.k,
                    record.tactic.value,
                    533,
                    "BF16 dual GeGLU",
                )?;
            }
            TacticBackend::CublasLtCustomSplitGeGluCutlassBf16 => match record.key.op {
                GemmOp::Bf16 => super::bf16::set_cublaslt_gemm_split_custom(
                    record.key.m,
                    record.key.n,
                    record.key.k,
                    record.tactic.value,
                )?,
                op => {
                    return Err(Error::Other(format!(
                        "BF16 fused GeGLU tactic does not support tuning operation {op:?}"
                    )))
                }
            },
            TacticBackend::Vendor => {}
        }
    }
    crate::tuning::install(store)
}

/// Borrowed static-per-tensor FP8 weight contract.
#[derive(Clone, Copy)]
pub struct Fp8WeightView<'a> {
    pub values_e4m3: &'a Tensor,
    pub scale: f32,
    /// Exact dual-GeGLU [gate256,up256] physical layout. Plain GEMM must
    /// reject this layout; only the exact dual-GeGLU backend may consume it.
    pub dual_geglu_interleaved: bool,
    /// Optional auto-mode physical [gate256,up256] matrix. The primary tensor
    /// remains plain and is used by every non-dual route.
    pub dual_geglu_auto_interleaved: Option<&'a Tensor>,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Fp8DualGeGluMode {
    Auto,
    Off,
    On,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Fp8DualGeGluWeightRoute {
    Plain,
    InterleavedPrimary,
    InterleavedAuto,
}

fn parse_fp8_dual_geglu_mode(value: Option<&str>) -> Result<Fp8DualGeGluMode> {
    match value {
        None | Some("auto") => Ok(Fp8DualGeGluMode::Auto),
        Some("0" | "off") => Ok(Fp8DualGeGluMode::Off),
        Some("1" | "on") => Ok(Fp8DualGeGluMode::On),
        Some(value) => Err(Error::Other(format!(
            "APXINF_PI05_FP8_DUAL_GEGLU must be auto, 0/off, or 1/on; got {value}"
        ))),
    }
}

fn fp8_dual_geglu_mode() -> Result<Fp8DualGeGluMode> {
    const NAME: &str = "APXINF_PI05_FP8_DUAL_GEGLU";
    match std::env::var(NAME) {
        Err(std::env::VarError::NotPresent) => parse_fp8_dual_geglu_mode(None),
        Err(std::env::VarError::NotUnicode(_)) => {
            Err(Error::Other(format!("{NAME} must be valid Unicode")))
        }
        Ok(value) => parse_fp8_dual_geglu_mode(Some(&value)),
    }
}

fn fp8_dual_geglu_weight_route(
    mode: Fp8DualGeGluMode,
    dual_mega: bool,
    primary_interleaved: bool,
    auto_interleaved_available: bool,
) -> Result<Fp8DualGeGluWeightRoute> {
    match (mode, dual_mega, primary_interleaved, auto_interleaved_available) {
        (Fp8DualGeGluMode::Off, false, false, false) => Ok(Fp8DualGeGluWeightRoute::Plain),
        (Fp8DualGeGluMode::On, true, true, false) => Ok(Fp8DualGeGluWeightRoute::InterleavedPrimary),
        (Fp8DualGeGluMode::Auto, false, false, false) => Ok(Fp8DualGeGluWeightRoute::Plain),
        (Fp8DualGeGluMode::Auto, false, false, true) => Ok(Fp8DualGeGluWeightRoute::Plain),
        (Fp8DualGeGluMode::Auto, true, false, true) => Ok(Fp8DualGeGluWeightRoute::InterleavedAuto),
        _ => Err(Error::Other(format!(
            "FP8 dual GeGLU config/layout mismatch: mode={mode:?}, backend_dual={dual_mega}, primary_interleaved={primary_interleaved}, auto_interleaved={auto_interleaved_available}"
        ))),
    }
}

fn tuning_key(ctx: &CudaContext, m: usize, n: usize, k: usize) -> GemmTuningKey {
    GemmTuningKey {
        op: GemmOp::Fp8F16,
        device: DeviceFingerprint::from(ctx.caps()),
        m,
        n,
        k,
        activation_dtype: TuningDType::F8E4M3,
        weight_dtype: TuningDType::F8E4M3,
        output_dtype: TuningDType::F16,
        layout: GemmLayout::RowMajor,
        scale_mode: ScaleMode::PerTensor,
        epilogue: Epilogue::None,
        workspace_limit: usize::MAX,
    }
}

pub fn exact_fp8_tactic(
    ctx: &CudaContext,
    m: usize,
    n: usize,
    k: usize,
) -> Option<crate::tuning::TacticId> {
    crate::tuning::lookup_gemm_exact(&tuning_key(ctx, m, n, k))
}

#[cfg(apxinf_cutlass_gemm)]
fn selected_cutlass_tactic(ctx: &CudaContext, m: usize, n: usize, k: usize) -> i32 {
    let key = tuning_key(ctx, m, n, k);
    crate::tuning::lookup_gemm_exact(&key)
        .or_else(|| crate::tuning::lookup_gemm(&key))
        .filter(|tactic| tactic.backend == TacticBackend::Cutlass)
        .map(|tactic| tactic.value)
        .unwrap_or_else(|| {
            if m <= 16 {
                0
            } else if m <= 64 {
                1
            } else if m <= 256 {
                2
            } else {
                3
            }
        })
}

/// Physical static FP8 GEMM with FP16 output.
pub fn gemm_fp8(
    ctx: &CudaContext,
    activation: &Tensor,
    activation_scale: f32,
    weight: Fp8WeightView<'_>,
) -> Result<Tensor> {
    if weight.dual_geglu_interleaved {
        return Err(Error::Other(
            "FP8 dual GeGLU interleaved Gate/Up weight cannot be used by plain FP8 GEMM".into(),
        ));
    }
    if activation.dtype() != DType::F8E4M3 || weight.values_e4m3.dtype() != DType::F8E4M3 {
        return Err(Error::Other(format!(
            "gemm_fp8 expects E4M3 operands, got {} and {}",
            activation.dtype(),
            weight.values_e4m3.dtype()
        )));
    }
    if !activation_scale.is_finite()
        || activation_scale <= 0.0
        || !weight.scale.is_finite()
        || weight.scale <= 0.0
    {
        return Err(Error::Other(format!(
            "gemm_fp8 scales must be finite and positive, got activation={activation_scale}, weight={}",
            weight.scale
        )));
    }
    let a = activation.shape().dims();
    let b = weight.values_e4m3.shape().dims();
    if a.len() != 2 || b.len() != 2 || a[1] != b[0] {
        return Err(Error::Other(format!(
            "gemm_fp8 shape mismatch: {a:?} @ {b:?}"
        )));
    }
    let expected_device = Device::Cuda(ctx.device_id());
    if activation.device() != expected_device || weight.values_e4m3.device() != expected_device {
        return Err(Error::DeviceMismatch {
            expected: expected_device,
            got: if activation.device() != expected_device {
                activation.device()
            } else {
                weight.values_e4m3.device()
            },
        });
    }

    let (m, k, n) = (a[0], a[1], b[1]);
    let output = crate::workspace::output_buffer(ctx, m * n * DType::F16.size_in_bytes())?;
    let activation_buffer = CudaBuffer::from_tensor(activation).map_err(Error::Cuda)?;
    let weight_buffer = CudaBuffer::from_tensor(weight.values_e4m3).map_err(Error::Cuda)?;
    if crate::workspace::fp8_emulation_required(ctx)? {
        let activation_bytes = m
            .checked_mul(k)
            .and_then(|elements| elements.checked_mul(DType::F16.size_in_bytes()))
            .ok_or_else(|| Error::Other("FP8 activation decode size overflow".into()))?;
        let weight_bytes = k
            .checked_mul(n)
            .and_then(|elements| elements.checked_mul(DType::F16.size_in_bytes()))
            .ok_or_else(|| Error::Other("FP8 weight decode size overflow".into()))?;
        let (activation_f16, weight_f16) =
            crate::workspace::fp8_emulation_buffers(ctx, activation_bytes, weight_bytes)?;
        dequantize_e4m3_f16(
            ctx,
            &activation_buffer,
            &activation_f16,
            m * k,
            activation_scale,
        )?;
        dequantize_e4m3_f16(ctx, &weight_buffer, &weight_f16, k * n, weight.scale)?;
        ctx.cublas()
            .gemm(
                DType::F16,
                m,
                n,
                k,
                1.0,
                &activation_f16,
                &weight_f16,
                0.0,
                &output,
            )
            .map_err(Error::Cuda)?;
        return Ok(output.into_tensor(Shape::new(vec![m, n]), DType::F16));
    }

    let persisted_tactic = crate::tuning::lookup_gemm_exact(&tuning_key(ctx, m, n, k));
    let use_split_serial = persisted_tactic.is_some_and(|tactic| {
        matches!(
            tactic.backend,
            TacticBackend::CublasLtCustomSplitSerial
                | TacticBackend::CublasLtCustomSplitGeGluCutlass
                | TacticBackend::CublasLtCustomSplitGeGluCutlass2SmAuto
                | TacticBackend::CublasLtCustomSplitGeGluCutlass2SmStage3
                | TacticBackend::CublasLtCustomSplitGeGluCutlassM522Explicit2Sm
        )
    });
    let use_cublaslt = persisted_tactic.is_some_and(|tactic| {
        matches!(
            tactic.backend,
            TacticBackend::CublasLt
                | TacticBackend::CublasLtCustom
                | TacticBackend::CublasLtCustomBias
                | TacticBackend::CublasLtCustomSplitSerial
                | TacticBackend::CublasLtCustomSplitGeGluCutlass
                | TacticBackend::CublasLtCustomSplitGeGluCutlass2SmAuto
                | TacticBackend::CublasLtCustomSplitGeGluCutlass2SmStage3
                | TacticBackend::CublasLtCustomSplitGeGluCutlassM522Explicit2Sm
        )
    });
    #[cfg(not(apxinf_cutlass_gemm))]
    let _ = use_cublaslt;
    #[cfg(apxinf_cutlass_gemm)]
    if n >= 1024 && n % 16 == 0 && k % 16 == 0 && !use_cublaslt {
        let tactic = selected_cutlass_tactic(ctx, m, n, k);
        if cutlass_fp8_gemm_f16(
            ctx,
            &activation_buffer,
            &weight_buffer,
            &output,
            m,
            n,
            k,
            activation_scale * weight.scale,
            tactic,
        )? {
            return Ok(output.into_tensor(Shape::new(vec![m, n]), DType::F16));
        }
    }
    if crate::workspace::may_prepare_native_resources() {
        if use_split_serial {
            prepare_cublaslt_fp8_gemm_split(m, n, k)?;
        } else {
            prepare_cublaslt_fp8_gemm(m, n, k)?;
        }
    }
    if use_split_serial {
        cublaslt_fp8_gemm_split_f16(
            ctx,
            &activation_buffer,
            &weight_buffer,
            &output,
            m,
            n,
            k,
            activation_scale * weight.scale,
        )?;
    } else {
        cublaslt_fp8_gemm_f16(
            ctx,
            &activation_buffer,
            &weight_buffer,
            &output,
            m,
            n,
            k,
            activation_scale * weight.scale,
        )?;
    }
    Ok(output.into_tensor(Shape::new(vec![m, n]), DType::F16))
}

/// Run the configured cuBLASLt gate + CUTLASS up/GeGLU/E4M3 fused tactic.
/// Returns `None` unless the exact physical GEMM record selects this backend;
/// a selected but unsupported record fails closed instead of falling back.
pub fn gemm_fp8_geglu_fused(
    ctx: &CudaContext,
    activation: &Tensor,
    activation_scale: f32,
    packed_weight: Fp8WeightView<'_>,
    output_scale: f32,
) -> Result<Option<Tensor>> {
    if activation.dtype() != DType::F8E4M3 || packed_weight.values_e4m3.dtype() != DType::F8E4M3 {
        return Err(Error::Other(format!(
            "FP8 fused GeGLU expects E4M3 operands, got {} and {}",
            activation.dtype(),
            packed_weight.values_e4m3.dtype()
        )));
    }
    if !activation_scale.is_finite()
        || activation_scale <= 0.0
        || !packed_weight.scale.is_finite()
        || packed_weight.scale <= 0.0
        || !output_scale.is_finite()
        || output_scale <= 0.0
    {
        return Err(Error::Other(format!(
            "FP8 fused GeGLU scales must be finite and positive, got activation={activation_scale}, weight={}, output={output_scale}",
            packed_weight.scale
        )));
    }
    let a = activation.shape().dims();
    let b = packed_weight.values_e4m3.shape().dims();
    if a.len() != 2 || b.len() != 2 || a[1] != b[0] || b[1] % 2 != 0 {
        return Err(Error::Other(format!(
            "FP8 fused GeGLU shape mismatch: {a:?} @ {b:?}"
        )));
    }
    let expected_device = Device::Cuda(ctx.device_id());
    if activation.device() != expected_device
        || packed_weight.values_e4m3.device() != expected_device
    {
        return Err(Error::DeviceMismatch {
            expected: expected_device,
            got: if activation.device() != expected_device {
                activation.device()
            } else {
                packed_weight.values_e4m3.device()
            },
        });
    }

    let (m, k, full_n) = (a[0], a[1], b[1]);
    let fused_tactic = crate::tuning::lookup_gemm_exact(&tuning_key(ctx, m, full_n, k));
    let (cutlass_geglu_tactic, tuned_m, dual_mega) = match fused_tactic.map(|tactic| tactic.backend)
    {
        Some(TacticBackend::CublasLtCustomSplitGeGluCutlass) => (0, 778, false),
        Some(TacticBackend::CublasLtCustomSplitGeGluCutlass2SmAuto) => (1, 778, false),
        Some(TacticBackend::CublasLtCustomSplitGeGluCutlass2SmStage3) => (2, 778, false),
        Some(TacticBackend::CublasLtCustomSplitGeGluCutlassM522Explicit2Sm) => (3, 522, false),
        Some(TacticBackend::CutlassFp8DualGeGlu) => (0, m, true),
        _ => return Ok(None),
    };
    if (m, full_n, k) != (tuned_m, 32768, 2048) {
        return Err(Error::Other(format!(
            "fused FP8 GeGLU backend is tuned only for [{tuned_m},2048] @ [2048,32768], got [{m},{k}] @ [{k},{full_n}]"
        )));
    }
    if crate::workspace::fp8_emulation_required(ctx)? {
        return Err(Error::Other(
            "FP8 fused GeGLU requires native FP8 Tensor Cores".into(),
        ));
    }
    let weight_route = fp8_dual_geglu_weight_route(
        fp8_dual_geglu_mode()?,
        dual_mega,
        packed_weight.dual_geglu_interleaved,
        packed_weight.dual_geglu_auto_interleaved.is_some(),
    )?;
    let selected_weight = match weight_route {
        Fp8DualGeGluWeightRoute::Plain | Fp8DualGeGluWeightRoute::InterleavedPrimary => {
            packed_weight.values_e4m3
        }
        Fp8DualGeGluWeightRoute::InterleavedAuto => {
            packed_weight.dual_geglu_auto_interleaved.unwrap()
        }
    };
    if selected_weight.dtype() != DType::F8E4M3 || selected_weight.shape().dims() != b {
        return Err(Error::Other(format!(
            "FP8 dual GeGLU selected weight must be E4M3 {b:?}, got {} {:?}",
            selected_weight.dtype(),
            selected_weight.shape().dims()
        )));
    }
    if selected_weight.device() != expected_device {
        return Err(Error::DeviceMismatch {
            expected: expected_device,
            got: selected_weight.device(),
        });
    }

    #[cfg(not(apxinf_cutlass_gemm))]
    {
        let _ = (ctx, activation_scale, output_scale);
        return Err(Error::Other(
            "FP8 fused GeGLU requires the SM100-family CUTLASS build".into(),
        ));
    }

    #[cfg(apxinf_cutlass_gemm)]
    {
        let n = full_n / 2;
        let output = crate::workspace::output_buffer(
            ctx,
            m.checked_mul(n)
                .and_then(|elements| elements.checked_mul(DType::F8E4M3.size_in_bytes()))
                .ok_or_else(|| Error::Other("FP8 fused GeGLU output size overflow".into()))?,
        )?;
        let activation_buffer = CudaBuffer::from_tensor(activation).map_err(Error::Cuda)?;
        let weight_buffer = CudaBuffer::from_tensor(selected_weight).map_err(Error::Cuda)?;
        if dual_mega {
            let status = unsafe {
                ffi::apxinf_static_cutlass_fp8_dual_gemm_geglu_e4m3(
                    activation_buffer.ptr(),
                    weight_buffer.ptr(),
                    output.ptr(),
                    m as i32,
                    n as i32,
                    k as i32,
                    full_n as i32,
                    activation_scale * packed_weight.scale,
                    output_scale,
                    ctx.stream().handle(),
                )
            };
            if status != 0 {
                return Err(Error::Cuda(format!(
                    "FP8 dual-GEMM GeGLU rejected [{m},{n},{k}] ({status})"
                )));
            }
            return Ok(Some(
                output.into_tensor(Shape::new(vec![m, n]), DType::F8E4M3),
            ));
        }
        let gate = crate::workspace::output_buffer(
            ctx,
            m.checked_mul(full_n)
                .and_then(|elements| elements.checked_mul(DType::F16.size_in_bytes()))
                .ok_or_else(|| Error::Other("FP8 fused GeGLU gate size overflow".into()))?,
        )?;
        if crate::workspace::may_prepare_native_resources() {
            prepare_cublaslt_fp8_gemm_split(m, full_n, k)?;
        }
        cublaslt_fp8_gemm_split_first_f16(
            ctx,
            &activation_buffer,
            &weight_buffer,
            &gate,
            m,
            full_n,
            k,
            activation_scale * packed_weight.scale,
        )?;
        let status = unsafe {
            ffi::apxinf_static_cutlass_fp8_gemm_geglu_e4m3(
                activation_buffer.ptr(),
                weight_buffer.ptr(),
                gate.ptr(),
                output.ptr(),
                m as i32,
                n as i32,
                k as i32,
                full_n as i32,
                activation_scale * packed_weight.scale,
                output_scale,
                cutlass_geglu_tactic,
                ctx.stream().handle(),
            )
        };
        if status != 0 {
            return Err(Error::Cuda(format!(
                "FP8 fused GeGLU CUTLASS fused GeGLU rejected [{m},{n},{k}] ({status})"
            )));
        }
        Ok(Some(
            output.into_tensor(Shape::new(vec![m, n]), DType::F8E4M3),
        ))
    }
}

pub fn native_fp8_gemm_supported_for_device(device: usize) -> Result<bool> {
    let mut supported = 0i32;
    unsafe {
        ffi::check_cuda(ffi::apxinf_static_native_fp8_supported(
            device as i32,
            &mut supported,
        ))
        .map_err(Error::Cuda)?;
    }
    Ok(supported != 0)
}

/// Whether this CUDA device can execute E4M3 GEMMs directly on Tensor Cores.
pub fn native_fp8_gemm_supported(ctx: &CudaContext) -> Result<bool> {
    native_fp8_gemm_supported_for_device(ctx.device_id())
}

pub fn set_cublaslt_gemm_heuristic(
    m: usize,
    n: usize,
    k: usize,
    heuristic_rank: i32,
) -> Result<()> {
    if !(0..64).contains(&heuristic_rank) {
        return Err(Error::Other(format!(
            "invalid static inference cuBLASLt heuristic rank {heuristic_rank}"
        )));
    }
    let status = unsafe {
        ffi::apxinf_static_set_cublaslt_gemm_heuristic(m as i32, n as i32, k as i32, heuristic_rank)
    };
    ffi::check_cublas(status).map_err(Error::Cuda)
}

pub fn set_cublaslt_gemm_custom(m: usize, n: usize, k: usize, tactic: i32) -> Result<()> {
    let config = crate::tuning::decode_cublaslt_custom_tactic(tactic).ok_or_else(|| {
        Error::Other(format!(
            "invalid static inference cuBLASLt custom tactic {tactic}"
        ))
    })?;
    let status = unsafe {
        ffi::apxinf_static_set_cublaslt_fp8_gemm_custom(
            m as i32,
            n as i32,
            k as i32,
            config.tile_id,
            config.custom_option,
            config.stages_id,
            config.cluster_shape_id,
        )
    };
    ffi::check_cublas(status).map_err(Error::Cuda)
}

pub fn set_cublaslt_gemm_bias_custom(m: usize, n: usize, k: usize, tactic: i32) -> Result<()> {
    let config = crate::tuning::decode_cublaslt_custom_tactic(tactic).ok_or_else(|| {
        Error::Other(format!(
            "invalid static inference cuBLASLt fused-bias custom tactic {tactic}"
        ))
    })?;
    let status = unsafe {
        ffi::apxinf_static_set_cublaslt_fp8_gemm_bias_custom(
            m as i32,
            n as i32,
            k as i32,
            config.tile_id,
            config.custom_option,
            config.stages_id,
            config.cluster_shape_id,
        )
    };
    ffi::check_cublas(status).map_err(Error::Cuda)
}

pub fn set_cublaslt_gemm_split_custom(m: usize, n: usize, k: usize, tactic: i32) -> Result<()> {
    let config = crate::tuning::decode_cublaslt_custom_tactic(tactic).ok_or_else(|| {
        Error::Other(format!(
            "invalid static inference cuBLASLt split-serial custom tactic {tactic}"
        ))
    })?;
    let status = unsafe {
        ffi::apxinf_static_set_cublaslt_fp8_gemm_split_custom(
            m as i32,
            n as i32,
            k as i32,
            config.tile_id,
            config.custom_option,
            config.stages_id,
            config.cluster_shape_id,
        )
    };
    ffi::check_cublas(status).map_err(Error::Cuda)
}

pub fn dequantize_e4m3_f16(
    ctx: &CudaContext,
    input: &CudaBuffer,
    output: &CudaBuffer,
    elements: usize,
    scale: f32,
) -> Result<()> {
    unsafe {
        ffi::check_cuda(ffi::apxinf_static_dequantize_e4m3_f16(
            input.ptr(),
            output.ptr(),
            elements as i64,
            scale,
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)?;
    }
    Ok(())
}

#[cfg(apxinf_cutlass_gemm)]
#[allow(clippy::too_many_arguments)]
pub fn cutlass_fp8_gemm_f16(
    ctx: &CudaContext,
    activation: &CudaBuffer,
    weight: &CudaBuffer,
    output: &CudaBuffer,
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
    tactic: i32,
) -> Result<bool> {
    let status = unsafe {
        ffi::apxinf_static_cutlass_fp8_gemm_f16(
            activation.ptr(),
            weight.ptr(),
            output.ptr(),
            m as i32,
            n as i32,
            k as i32,
            alpha,
            tactic,
            ctx.stream().handle(),
        )
    };
    Ok(status == 0)
}

pub fn prepare_cublaslt_fp8_gemm(m: usize, n: usize, k: usize) -> Result<()> {
    let status = unsafe { ffi::apxinf_static_prepare_fp8_gemm_f16(m as i32, n as i32, k as i32) };
    ffi::check_cublas(status).map_err(Error::Cuda)
}

pub fn prepare_cublaslt_fp8_gemm_split(m: usize, n: usize, k: usize) -> Result<()> {
    let status =
        unsafe { ffi::apxinf_static_prepare_fp8_gemm_split_f16(m as i32, n as i32, k as i32) };
    ffi::check_cublas(status).map_err(Error::Cuda)
}

#[allow(clippy::too_many_arguments)]
pub fn cublaslt_fp8_gemm_f16(
    ctx: &CudaContext,
    activation: &CudaBuffer,
    weight: &CudaBuffer,
    output: &CudaBuffer,
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
) -> Result<()> {
    let status = unsafe {
        ffi::apxinf_static_fp8_gemm_f16(
            activation.ptr(),
            weight.ptr(),
            output.ptr(),
            m as i32,
            n as i32,
            k as i32,
            alpha,
            ctx.stream().handle(),
        )
    };
    ffi::check_cublas(status).map_err(Error::Cuda)
}

#[allow(clippy::too_many_arguments)]
pub fn cublaslt_fp8_gemm_split_f16(
    ctx: &CudaContext,
    activation: &CudaBuffer,
    weight: &CudaBuffer,
    output: &CudaBuffer,
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
) -> Result<()> {
    let status = unsafe {
        ffi::apxinf_static_fp8_gemm_split_f16(
            activation.ptr(),
            weight.ptr(),
            output.ptr(),
            m as i32,
            n as i32,
            k as i32,
            alpha,
            ctx.stream().handle(),
        )
    };
    ffi::check_cublas(status).map_err(Error::Cuda)
}

#[allow(clippy::too_many_arguments)]
pub fn cublaslt_fp8_gemm_split_first_f16(
    ctx: &CudaContext,
    activation: &CudaBuffer,
    weight: &CudaBuffer,
    output: &CudaBuffer,
    m: usize,
    n: usize,
    k: usize,
    alpha: f32,
) -> Result<()> {
    let status = unsafe {
        ffi::apxinf_static_fp8_gemm_split_first_f16(
            activation.ptr(),
            weight.ptr(),
            output.ptr(),
            m as i32,
            n as i32,
            k as i32,
            alpha,
            ctx.stream().handle(),
        )
    };
    ffi::check_cublas(status).map_err(Error::Cuda)
}

#[cfg(apxinf_cutlass_gemm)]
pub fn autotune_cutlass_gemm_f16(
    ctx: &CudaContext,
    activation: &Tensor,
    weight: &Tensor,
    activation_scale: f32,
    weight_scale: f32,
    warmup: usize,
    iterations: usize,
) -> Result<Vec<CutlassTacticTiming>> {
    if iterations == 0 {
        return Err(Error::Other(
            "CUTLASS autotune iterations must be non-zero".into(),
        ));
    }
    let a = activation.shape().dims();
    let b = weight.shape().dims();
    if activation.dtype() != DType::F8E4M3
        || weight.dtype() != DType::F8E4M3
        || a.len() != 2
        || b.len() != 2
        || a[1] != b[0]
        || b[1] % 16 != 0
        || a[1] % 16 != 0
    {
        return Err(Error::Other(
            "CUTLASS autotune expects aligned FP8 [M,K] @ [K,N]".into(),
        ));
    }
    let (m, k, n) = (a[0], a[1], b[1]);
    let output = CudaBuffer::alloc_zeros(m * n * 2, ctx.device_id()).map_err(Error::Cuda)?;
    let mut evictor = ColdL2Evictor::new(ctx)?;
    let events = CudaEventPair::new()?;
    let mut timings = Vec::new();
    // All exposed candidates are ordinary auto-scheduled one-SM kernels.
    // Explicit two-SM schedules are intentionally not compiled because they
    // can wedge CUDA graph replay on the current Thor-U software stack.
    for tactic in 0..=7 {
        let launch = || -> Result<()> {
            let status = unsafe {
                ffi::apxinf_static_cutlass_fp8_gemm_f16(
                    gpu_ptr(activation)?,
                    gpu_ptr(weight)?,
                    output.ptr(),
                    m as i32,
                    n as i32,
                    k as i32,
                    activation_scale * weight_scale,
                    tactic,
                    ctx.stream().handle(),
                )
            };
            if status == 0 {
                Ok(())
            } else {
                Err(Error::Cuda(format!(
                    "CUTLASS tactic {tactic} rejected shape [{m},{n},{k}] ({status})"
                )))
            }
        };
        if (0..warmup)
            .try_for_each(|_| {
                evictor.evict(ctx)?;
                launch()
            })
            .is_err()
        {
            continue;
        }
        ctx.stream().synchronize().map_err(Error::Cuda)?;
        let mut milliseconds = 0.0f64;
        for _ in 0..iterations {
            milliseconds += events.measure(ctx, &mut evictor, &launch)?;
        }
        timings.push(CutlassTacticTiming {
            tactic,
            milliseconds: milliseconds / iterations as f64,
        });
    }
    Ok(timings)
}

#[cfg(not(apxinf_cutlass_gemm))]
#[allow(clippy::too_many_arguments)]
pub fn autotune_cutlass_gemm_f16(
    _ctx: &CudaContext,
    _activation: &Tensor,
    _weight: &Tensor,
    _activation_scale: f32,
    _weight_scale: f32,
    _warmup: usize,
    _iterations: usize,
) -> Result<Vec<CutlassTacticTiming>> {
    Err(Error::Other(
        "CUTLASS FP8 autotune requires an SM100-family CUDA build".into(),
    ))
}

pub fn autotune_cublaslt_gemm_f16(
    ctx: &CudaContext,
    activation: &Tensor,
    weight: &Tensor,
    activation_scale: f32,
    weight_scale: f32,
    max_algorithms: usize,
    warmup: usize,
    iterations: usize,
) -> Result<Vec<CublasLtAlgorithmTiming>> {
    if max_algorithms == 0 || max_algorithms > 64 || iterations == 0 {
        return Err(Error::Other(
            "cuBLASLt autotune expects 1..=64 algorithms and non-zero iterations".into(),
        ));
    }
    let a = activation.shape().dims();
    let b = weight.shape().dims();
    if activation.dtype() != DType::F8E4M3
        || weight.dtype() != DType::F8E4M3
        || a.len() != 2
        || b.len() != 2
        || a[1] != b[0]
    {
        return Err(Error::Other(
            "cuBLASLt autotune expects FP8 [M,K] @ [K,N]".into(),
        ));
    }
    let (m, k, n) = (a[0], a[1], b[1]);
    let output = CudaBuffer::alloc_zeros(m * n * 2, ctx.device_id()).map_err(Error::Cuda)?;
    let evictor = ColdL2Evictor::new(ctx)?;
    let mut returned = 0i32;
    let mut milliseconds = vec![-1.0f32; max_algorithms];
    let status = unsafe {
        ffi::apxinf_static_autotune_cublaslt_fp8_gemm_f16(
            gpu_ptr(activation)?,
            gpu_ptr(weight)?,
            output.ptr(),
            evictor.buffer.ptr(),
            evictor.metadata.eviction_buffer_bytes,
            m as i32,
            n as i32,
            k as i32,
            activation_scale * weight_scale,
            max_algorithms as i32,
            warmup as i32,
            iterations as i32,
            &mut returned,
            milliseconds.as_mut_ptr(),
            ctx.stream().handle(),
        )
    };
    ffi::check_cublas(status).map_err(Error::Cuda)?;
    Ok(milliseconds
        .into_iter()
        .take(returned.max(0) as usize)
        .enumerate()
        .filter(|(_, milliseconds)| *milliseconds >= 0.0)
        .map(|(heuristic_rank, milliseconds)| CublasLtAlgorithmTiming {
            heuristic_rank: heuristic_rank as i32,
            milliseconds: milliseconds as f64,
        })
        .collect())
}

#[cfg(test)]
mod fp8_dual_geglu_tests {
    use super::*;

    #[test]
    fn mode_parser_is_tri_state_and_defaults_auto() {
        assert_eq!(
            parse_fp8_dual_geglu_mode(None).unwrap(),
            Fp8DualGeGluMode::Auto
        );
        assert_eq!(
            parse_fp8_dual_geglu_mode(Some("auto")).unwrap(),
            Fp8DualGeGluMode::Auto
        );
        assert_eq!(
            parse_fp8_dual_geglu_mode(Some("0")).unwrap(),
            Fp8DualGeGluMode::Off
        );
        assert_eq!(
            parse_fp8_dual_geglu_mode(Some("off")).unwrap(),
            Fp8DualGeGluMode::Off
        );
        assert_eq!(
            parse_fp8_dual_geglu_mode(Some("1")).unwrap(),
            Fp8DualGeGluMode::On
        );
        assert_eq!(
            parse_fp8_dual_geglu_mode(Some("on")).unwrap(),
            Fp8DualGeGluMode::On
        );
        assert!(parse_fp8_dual_geglu_mode(Some("invalid")).is_err());
    }

    #[test]
    fn auto_routes_dual_to_copy_and_other_backends_to_plain() {
        assert_eq!(
            fp8_dual_geglu_weight_route(Fp8DualGeGluMode::Auto, true, false, true).unwrap(),
            Fp8DualGeGluWeightRoute::InterleavedAuto
        );
        assert_eq!(
            fp8_dual_geglu_weight_route(Fp8DualGeGluMode::Auto, false, false, true).unwrap(),
            Fp8DualGeGluWeightRoute::Plain
        );
        assert_eq!(
            fp8_dual_geglu_weight_route(Fp8DualGeGluMode::Auto, false, false, false).unwrap(),
            Fp8DualGeGluWeightRoute::Plain
        );
        assert_eq!(
            fp8_dual_geglu_weight_route(Fp8DualGeGluMode::Off, false, false, false).unwrap(),
            Fp8DualGeGluWeightRoute::Plain
        );
        assert_eq!(
            fp8_dual_geglu_weight_route(Fp8DualGeGluMode::On, true, true, false).unwrap(),
            Fp8DualGeGluWeightRoute::InterleavedPrimary
        );
        assert!(fp8_dual_geglu_weight_route(Fp8DualGeGluMode::Off, true, false, false).is_err());
        assert!(fp8_dual_geglu_weight_route(Fp8DualGeGluMode::On, false, true, false).is_err());
        assert!(fp8_dual_geglu_weight_route(Fp8DualGeGluMode::Auto, true, false, false).is_err());
    }

    #[test]
    fn dual_backend_accepts_only_validated_m_values_and_tactic_zero() {
        for m in [522, 533] {
            assert!(validate_fp8_dual_geglu_record(GemmOp::Fp8F16, m, 32768, 2048, 0).is_ok());
        }

        for (op, m, n, k, tactic) in [
            (GemmOp::Bf16, 533, 32768, 2048, 0),
            (GemmOp::Fp8F16, 521, 32768, 2048, 0),
            (GemmOp::Fp8F16, 534, 32768, 2048, 0),
            (GemmOp::Fp8F16, 533, 16384, 2048, 0),
            (GemmOp::Fp8F16, 533, 32768, 1024, 0),
            (GemmOp::Fp8F16, 533, 32768, 2048, 1),
        ] {
            assert!(validate_fp8_dual_geglu_record(op, m, n, k, tactic).is_err());
        }
    }

    #[test]
    fn bf16_dual_geglu_backend_is_exact_m533_tactic_zero() {
        assert!(validate_bf16_dual_geglu_record(
            GemmOp::Bf16,
            522,
            32768,
            2048,
            0,
            522,
            "BF16 dual GeGLU",
        )
        .is_ok());
        assert!(validate_bf16_dual_geglu_record(
            GemmOp::Bf16,
            533,
            32768,
            2048,
            0,
            533,
            "BF16 dual GeGLU",
        )
        .is_ok());
        assert!(validate_bf16_dual_geglu_record(
            GemmOp::Bf16,
            533,
            32768,
            2048,
            0,
            522,
            "BF16 dual GeGLU",
        )
        .is_err());

        for (op, m, n, k, tactic) in [
            (GemmOp::Fp8F16, 533, 32768, 2048, 0),
            (GemmOp::Bf16, 522, 32768, 2048, 0),
            (GemmOp::Bf16, 534, 32768, 2048, 0),
            (GemmOp::Bf16, 533, 16384, 2048, 0),
            (GemmOp::Bf16, 533, 32768, 1024, 0),
            (GemmOp::Bf16, 533, 32768, 2048, 1),
        ] {
            assert!(
                validate_bf16_dual_geglu_record(op, m, n, k, tactic, 533, "BF16 dual GeGLU",)
                    .is_err()
            );
        }
    }
}
