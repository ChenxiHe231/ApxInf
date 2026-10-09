//! Legacy `kernels::gemm` names over the cuda-new GEMM operator.

use std::cell::RefCell;
use std::rc::Rc;

use apxinf_core::{DType, Error, Result, Shape, Tensor};

use crate::{ops, CudaContext};

/// Calibration hook: observes every BF16 activation/weight pair entering a
/// GEMM while installed. Thread-local, so normal inference pays one
/// empty-cell check and concurrent model threads cannot see each other's
/// activations. Ported from the legacy crate with identical semantics.
pub trait Bf16ActivationObserver {
    fn observe(&self, activation: &Tensor, weight: &Tensor) -> Result<()>;
}

thread_local! {
    static BF16_OBSERVER: RefCell<Option<Rc<dyn Bf16ActivationObserver>>> =
        const { RefCell::new(None) };
}

/// Uninstalls the observer when dropped.
pub struct Bf16ObserverGuard;

impl Drop for Bf16ObserverGuard {
    fn drop(&mut self) {
        BF16_OBSERVER.with(|slot| *slot.borrow_mut() = None);
    }
}

pub fn install_bf16_observer(
    observer: Rc<dyn Bf16ActivationObserver>,
) -> Result<Bf16ObserverGuard> {
    BF16_OBSERVER.with(|slot| {
        let mut slot = slot.borrow_mut();
        if slot.is_some() {
            return Err(Error::Other(
                "a BF16 activation observer is already installed".into(),
            ));
        }
        *slot = Some(observer);
        Ok(Bf16ObserverGuard)
    })
}

fn observe_bf16(activation: &Tensor, weight: &Tensor) -> Result<()> {
    BF16_OBSERVER.with(|slot| {
        if let Some(observer) = slot.borrow().as_ref() {
            observer.observe(activation, weight)?;
        }
        Ok(())
    })
}

fn output_for(ctx: &CudaContext, a: &Tensor, b: &Tensor, what: &str) -> Result<Tensor> {
    let a_dims = a.shape().dims();
    let b_dims = b.shape().dims();
    if a_dims.len() != 2 || b_dims.len() != 2 || a_dims[1] != b_dims[0] {
        return Err(Error::Other(format!(
            "{what} shape mismatch: {a_dims:?} @ {b_dims:?}"
        )));
    }
    ctx.allocate_output(Shape::new(vec![a_dims[0], b_dims[1]]), DType::BF16)
}

/// `bf16`: plain BF16 GEMM, `[m, k] @ [k, n] -> [m, n]`.
pub fn bf16(ctx: &CudaContext, activation: &Tensor, weight: &Tensor) -> Result<Tensor> {
    observe_bf16(activation, weight)?;
    let mut output = output_for(ctx, activation, weight, "BF16 GEMM")?;
    ops::gemm(ctx, ops::GemmArgs::new(activation, weight, &mut output))?;
    Ok(output)
}

/// `matmul`: alias of [`bf16`] under the legacy generic name.
pub fn matmul(ctx: &CudaContext, activation: &Tensor, weight: &Tensor) -> Result<Tensor> {
    bf16(ctx, activation, weight)
}

/// `bf16_bias`: BF16 GEMM with a fused `[n]` bias epilogue.
pub fn bf16_bias(
    ctx: &CudaContext,
    activation: &Tensor,
    weight: &Tensor,
    bias: &Tensor,
) -> Result<Tensor> {
    let mut output = output_for(ctx, activation, weight, "BF16 bias GEMM")?;
    let gemm = ops::GemmArgs::new(activation, weight, &mut output);
    ops::gemm_bias(ctx, ops::GemmBiasArgs { gemm, bias })?;
    Ok(output)
}

/// Pre-quantized E4M3 weight with one per-tensor scale, mirroring the legacy
/// `Fp8WeightView`. The dual-GeGLU interleaved layouts are a legacy-runtime
/// concept and are intentionally absent.
#[derive(Clone, Copy)]
pub struct Fp8WeightView<'a> {
    pub values_e4m3: &'a Tensor,
    pub scale: f32,
}

/// `fp8_bf16`: static per-tensor FP8 GEMM with BF16 output.
///
/// The legacy helper consumed unit-scaled E4M3 operands and applied
/// `activation_scale * weight_scale` as alpha; cuda-new's `Fp8UnitScale`
/// quantization is the same contract.
pub fn fp8_bf16(
    ctx: &CudaContext,
    activation: &Tensor,
    activation_scale: f32,
    weight: Fp8WeightView<'_>,
) -> Result<Tensor> {
    let a_dims = activation.shape().dims();
    let b_dims = weight.values_e4m3.shape().dims();
    if a_dims.len() != 2 || b_dims.len() != 2 || a_dims[1] != b_dims[0] {
        return Err(Error::Other(format!(
            "FP8 GEMM shape mismatch: {a_dims:?} @ {b_dims:?}"
        )));
    }
    let mut output =
        ctx.allocate_output(Shape::new(vec![a_dims[0], b_dims[1]]), DType::BF16)?;
    let mut gemm = ops::GemmArgs::new(activation, weight.values_e4m3, &mut output);
    gemm.quantization = ops::GemmQuantization::Fp8UnitScale;
    gemm.alpha = activation_scale * weight.scale;
    ops::gemm(ctx, gemm)?;
    Ok(output)
}

/// `bf16_geglu_fused`: fused gate/up GEMM + GeGLU over a packed
/// `[k, 2*cols]` weight, producing `[rows, cols]`.
///
/// The legacy interleaved dual-GeGLU weight layouts are autotune candidates of
/// the legacy runtime; cuda-new selects its own candidates from the plain
/// layout, so only the plain weight is accepted.
pub fn bf16_geglu_fused(
    ctx: &CudaContext,
    activation: &Tensor,
    packed_weight: &Tensor,
) -> Result<Tensor> {
    observe_bf16(activation, packed_weight)?;
    let a_dims = activation.shape().dims();
    let b_dims = packed_weight.shape().dims();
    if a_dims.len() != 2 || b_dims.len() != 2 || a_dims[1] != b_dims[0] || b_dims[1] % 2 != 0 {
        return Err(Error::Other(format!(
            "fused GeGLU shape mismatch: {a_dims:?} @ {b_dims:?}"
        )));
    }
    let mut output =
        ctx.allocate_output(Shape::new(vec![a_dims[0], b_dims[1] / 2]), DType::BF16)?;
    let gemm = ops::GemmArgs::new(activation, packed_weight, &mut output);
    ops::gemm_geglu(ctx, ops::GemmGegluArgs { gemm })?;
    Ok(output)
}

/// Pre-quantized rowwise-dynamic FP8 weight. Unlike the legacy view, the
/// value matrix is stored in cuda-new's canonical `[K, N]` orientation; the
/// `[N, K]` legacy layout must be transposed at load time (a pure byte
/// permutation with no numerical effect).
#[derive(Clone, Copy)]
pub struct DynamicFp8WeightView<'a> {
    /// Contiguous `[K, N]` E4M3 matrix.
    pub values_e4m3: &'a Tensor,
    /// FP32 scale for each output channel, shape `[N]`.
    pub channel_scales: &'a Tensor,
}

/// `gemm_fp8_dynamic_bf16`: rowwise-dynamic FP8 GEMM — per-row activation
/// scales, per-channel weight scales, BF16 output.
pub fn gemm_fp8_dynamic_bf16(
    ctx: &CudaContext,
    activation: &Tensor,
    activation_scales: &Tensor,
    weight: DynamicFp8WeightView<'_>,
    bias: Option<&Tensor>,
) -> Result<Tensor> {
    let a_dims = activation.shape().dims();
    let b_dims = weight.values_e4m3.shape().dims();
    if activation.dtype() != DType::F8E4M3
        || weight.values_e4m3.dtype() != DType::F8E4M3
        || a_dims.len() != 2
        || b_dims.len() != 2
        || a_dims[1] != b_dims[0]
    {
        return Err(Error::Other(format!(
            "dynamic FP8 GEMM shape mismatch: {a_dims:?} @ KN {b_dims:?}"
        )));
    }
    let (m, n) = (a_dims[0], b_dims[1]);
    if activation_scales.dtype() != DType::F32
        || weight.channel_scales.dtype() != DType::F32
        || activation_scales.shape().dims() != [m]
        || weight.channel_scales.shape().dims() != [n]
    {
        return Err(Error::Other(
            "dynamic FP8 GEMM scale vectors must be F32 [M] and [N]".into(),
        ));
    }
    let mut output = ctx.allocate_output(Shape::new(vec![m, n]), DType::BF16)?;
    let gemm = ops::GemmArgs::fp8(
        activation,
        activation_scales,
        weight.values_e4m3,
        weight.channel_scales,
        &mut output,
    );
    match bias {
        Some(bias) => ops::gemm_bias(ctx, ops::GemmBiasArgs { gemm, bias })?,
        None => ops::gemm(ctx, gemm)?,
    }
    Ok(output)
}
