use apxinf_core::{DType, Device, Error, Result, Tensor};
use std::ops::Range;

use crate::ffi::abi::gemm as abi;
use crate::{CudaBuffer, CudaContext};

#[derive(Clone, Debug)]
pub struct GemmPolicy {
    /// Accumulation precision is part of the GEMM key, not a candidate detail.
    pub accumulation_dtype: DType,
    pub workspace_limit: usize,
    pub online_tune: bool,
    pub allow_fallback: bool,
    pub graph_safe: bool,
    pub deterministic: bool,
    pub cache_dir: Option<String>,
}

impl Default for GemmPolicy {
    fn default() -> Self {
        Self {
            accumulation_dtype: DType::F32,
            workspace_limit: 256 * 1024 * 1024,
            online_tune: true,
            allow_fallback: true,
            graph_safe: true,
            deterministic: false,
            cache_dir: None,
        }
    }
}

/// Parameters common to the GEMM semantic operators.
///
/// Additional operands belong to their semantic operator Args instead of this
/// common contract, so invalid fused combinations cannot be expressed.
pub struct GemmArgs<'a> {
    pub a: &'a Tensor,
    pub b: &'a Tensor,
    /// Canonical row-major output `[M, N]`. GEMM mutates this tensor.
    pub out: &'a mut Tensor,
    pub quantization: GemmQuantization<'a>,
    pub alpha: f32,
    pub output_scale: f32,
    pub policy: GemmPolicy,
    /// Optional pre-quantization FP32 operands used only while autotuning.
    ///
    /// When omitted, validation reconstructs FP32 values from the supplied
    /// inference tensors and therefore measures implementation error only.
    /// Supplying this reference also includes the initial quantization loss.
    pub reference: Option<GemmReference<'a>>,
}

/// Original FP32 operands for provider-independent autotune validation.
///
/// The slices use the same canonical row-major shapes as the inference
/// tensors: `a=[M,K]`, `b=[K,N]`, and (for fused bias semantics) `bias=[N]`.
/// They are consumed synchronously during plan creation and are never retained
/// by the execution instance.
#[derive(Clone, Copy)]
pub struct GemmReference<'a> {
    pub a: &'a [f32],
    pub b: &'a [f32],
    pub bias: Option<&'a [f32]>,
}

impl<'a> GemmReference<'a> {
    pub fn new(a: &'a [f32], b: &'a [f32]) -> Self {
        Self { a, b, bias: None }
    }

    pub fn with_bias(a: &'a [f32], b: &'a [f32], bias: &'a [f32]) -> Self {
        Self {
            a,
            b,
            bias: Some(bias),
        }
    }
}

/// Quantization contract for the operands supplied to one GEMM call.
///
/// Inputs in the FP8 and W8A8 variants are already quantized. Dynamic
/// quantization is intentionally not represented until a candidate can perform
/// it as part of the operation.
#[derive(Clone, Copy)]
pub enum GemmQuantization<'a> {
    /// Ordinary floating-point GEMM. FP8 is excluded because its scale
    /// semantics must be explicit.
    None,
    /// Pre-quantized FP8 inputs with per-row A scales and per-channel B scales.
    Fp8 {
        row_scales: &'a Tensor,
        channel_scales: &'a Tensor,
    },
    /// Pre-quantized INT8 inputs with per-row A scales and per-channel B scales.
    W8A8 {
        row_scales: &'a Tensor,
        channel_scales: &'a Tensor,
    },
    /// FP8 inputs whose values already include the intended scaling.
    /// This supports existing FP8 kernels which do not consume scale tensors.
    Fp8UnitScale,
}

impl<'a> GemmArgs<'a> {
    /// Create a GEMM with canonical contiguous row-major inputs:
    /// `a=[M,K]`, `b=[K,N]`, and `out=[M,N]`.
    pub fn new(a: &'a Tensor, b: &'a Tensor, out: &'a mut Tensor) -> Self {
        Self {
            a,
            b,
            out,
            quantization: GemmQuantization::None,
            alpha: 1.0,
            output_scale: 1.0,
            policy: GemmPolicy::default(),
            reference: None,
        }
    }

    pub fn fp8(
        a: &'a Tensor,
        row_scales: &'a Tensor,
        b: &'a Tensor,
        channel_scales: &'a Tensor,
        out: &'a mut Tensor,
    ) -> Self {
        Self {
            a,
            b,
            out,
            quantization: GemmQuantization::Fp8 {
                row_scales,
                channel_scales,
            },
            alpha: 1.0,
            output_scale: 1.0,
            policy: GemmPolicy::default(),
            reference: None,
        }
    }

    pub fn w8a8(
        a: &'a Tensor,
        row_scales: &'a Tensor,
        b: &'a Tensor,
        channel_scales: &'a Tensor,
        out: &'a mut Tensor,
    ) -> Self {
        let policy = GemmPolicy {
            accumulation_dtype: DType::I32,
            ..GemmPolicy::default()
        };
        Self {
            a,
            b,
            out,
            quantization: GemmQuantization::W8A8 {
                row_scales,
                channel_scales,
            },
            alpha: 1.0,
            output_scale: 1.0,
            policy,
            reference: None,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Semantic {
    Gemm = 0,
    GemmBiasGelu = 1,
    GemmGeglu = 2,
    GemmBias = 3,
}

pub(crate) struct Normalized<'a> {
    pub api: super::execution::PlanApi,
    pub spec: abi::Spec,
    pub policy: GemmPolicy,
    pub bindings: abi::Bindings,
    pub storage: Vec<CudaBuffer>,
    pub reference: Option<GemmReference<'a>>,
}

pub(crate) fn invalid(message: impl Into<String>) -> Error {
    Error::Other(message.into())
}

pub(crate) fn dtype(dtype: DType) -> Result<u32> {
    match dtype {
        DType::F32 => Ok(0),
        DType::F16 => Ok(1),
        DType::BF16 => Ok(2),
        DType::F8E4M3 => Ok(3),
        DType::I8 => Ok(4),
        DType::I32 => Ok(5),
    }
}

fn required_bytes(dtype: DType, shape: &[usize]) -> Result<usize> {
    shape
        .iter()
        .try_fold(dtype.size_in_bytes(), |bytes, dimension| {
            bytes.checked_mul(*dimension)
        })
        .ok_or_else(|| invalid("GEMM size overflow"))
}

const MAX_RECORDED_ALIGNMENT: usize = 256;

fn alignment_class(ptr: *const std::ffi::c_void) -> u32 {
    let address = ptr as usize;
    if address == 0 {
        0
    } else {
        (1usize
            << address
                .trailing_zeros()
                .min(MAX_RECORDED_ALIGNMENT.trailing_zeros())) as u32
    }
}

fn checked_device_range(ptr: *mut std::ffi::c_void, len: usize) -> Result<Range<usize>> {
    let start = ptr as usize;
    if len != 0 && start == 0 {
        return Err(invalid("GEMM storage has a null device pointer"));
    }
    let end = start
        .checked_add(len)
        .ok_or_else(|| invalid("GEMM storage address range overflow"))?;
    Ok(start..end)
}

fn reject_output_overlap(
    output: &CudaBuffer,
    output_bytes: usize,
    input: &CudaBuffer,
    input_bytes: usize,
    input_name: &str,
) -> Result<()> {
    let output = checked_device_range(output.ptr(), output_bytes)?;
    let input = checked_device_range(input.ptr(), input_bytes)?;
    if output.start < input.end && input.start < output.end {
        return Err(invalid(format!(
            "GEMM output storage overlaps read-only {input_name} storage"
        )));
    }
    Ok(())
}

pub(crate) fn tensor_storage(
    ctx: &CudaContext,
    tensor: &Tensor,
    expected_dtype: DType,
    expected_shape: &[usize],
) -> Result<CudaBuffer> {
    if tensor.device() != Device::Cuda(ctx.device_id())
        || tensor.dtype() != expected_dtype
        || tensor.shape().dims() != expected_shape
    {
        return Err(invalid("GEMM tensor device/dtype/shape mismatch"));
    }
    let expected_bytes = required_bytes(expected_dtype, expected_shape)?;
    let buffer = CudaBuffer::from_tensor(tensor).map_err(Error::Cuda)?;
    if buffer.len() < expected_bytes {
        return Err(invalid("GEMM storage is too small"));
    }
    let dtype_alignment = expected_dtype.size_in_bytes();
    if (buffer.ptr() as usize) % dtype_alignment != 0 {
        return Err(invalid(format!(
            "GEMM tensor storage is not aligned to its {}-byte dtype",
            dtype_alignment
        )));
    }
    Ok(buffer)
}

pub(crate) fn normalize<'a>(
    ctx: &CudaContext,
    args: GemmArgs<'a>,
    semantic: Semantic,
    api: super::execution::PlanApi,
    bias: Option<&Tensor>,
) -> Result<Normalized<'a>> {
    let a_shape = args.a.shape().dims();
    let b_shape = args.b.shape().dims();
    if a_shape.len() != 2 || b_shape.len() != 2 {
        return Err(invalid("GEMM requires rank-2 tensors"));
    }
    // Public storage is deliberately canonical and candidate-independent.
    // Candidates may transpose or pack internally while preparing a plan.
    let (m, k) = (a_shape[0], a_shape[1]);
    let (weight_k, n) = (b_shape[0], b_shape[1]);
    if k != weight_k
        || m == 0
        || n == 0
        || k == 0
        || m > i32::MAX as usize
        || n > i32::MAX as usize
        || k > i32::MAX as usize
    {
        return Err(invalid("invalid GEMM dimensions"));
    }
    if semantic == Semantic::GemmGeglu && n % 2 != 0 {
        return Err(invalid("GEMM+GeGLU requires an even projection width"));
    }
    if !args.alpha.is_finite() || !args.output_scale.is_finite() || args.output_scale <= 0.0 {
        return Err(invalid("invalid GEMM scales"));
    }
    let needs_bias = matches!(semantic, Semantic::GemmBias | Semantic::GemmBiasGelu);
    if bias.is_some() != needs_bias {
        return Err(invalid("fused operands do not match the semantic operator"));
    }
    if let Some(reference) = args.reference {
        let expected_a = m
            .checked_mul(k)
            .ok_or_else(|| invalid("GEMM reference A size overflow"))?;
        let expected_b = k
            .checked_mul(n)
            .ok_or_else(|| invalid("GEMM reference B size overflow"))?;
        if reference.a.len() != expected_a
            || reference.b.len() != expected_b
            || reference.bias.is_some() != needs_bias
            || reference.bias.is_some_and(|values| values.len() != n)
        {
            return Err(invalid(
                "original FP32 reference operands do not match the L3 semantic",
            ));
        }
        if reference
            .a
            .iter()
            .chain(reference.b)
            .chain(reference.bias.into_iter().flatten())
            .any(|value| !value.is_finite())
        {
            return Err(invalid("original FP32 reference operands must be finite"));
        }
    }

    let (quantization, row_scales, channel_scales) = match args.quantization {
        GemmQuantization::None => {
            if matches!(args.a.dtype(), DType::F8E4M3 | DType::I8)
                || args.a.dtype() != args.b.dtype()
            {
                return Err(invalid(
                    "plain GEMM requires matching non-quantized input dtypes",
                ));
            }
            (0, None, None)
        }
        GemmQuantization::Fp8UnitScale => {
            if args.a.dtype() != DType::F8E4M3 || args.b.dtype() != DType::F8E4M3 {
                return Err(invalid("unit-scale FP8 GEMM requires two FP8 tensors"));
            }
            (1, None, None)
        }
        GemmQuantization::Fp8 {
            row_scales,
            channel_scales,
        } => {
            if args.a.dtype() != DType::F8E4M3 || args.b.dtype() != DType::F8E4M3 {
                return Err(invalid("scaled FP8 GEMM requires two FP8 tensors"));
            }
            (2, Some(row_scales), Some(channel_scales))
        }
        GemmQuantization::W8A8 {
            row_scales,
            channel_scales,
        } => {
            if args.a.dtype() != DType::I8
                || args.b.dtype() != DType::I8
                || args.out.dtype() != DType::BF16
                || !matches!(semantic, Semantic::Gemm | Semantic::GemmBias)
                || k > 131071
            {
                return Err(invalid("invalid W8A8 GEMM contract"));
            }
            (3, Some(row_scales), Some(channel_scales))
        }
    };
    let has_row_channel_scales = matches!(quantization, 2 | 3);
    if semantic == Semantic::GemmGeglu && has_row_channel_scales {
        return Err(invalid("scaled GEMM+GeGLU is unsupported"));
    }

    let mut storage = vec![
        tensor_storage(ctx, args.a, args.a.dtype(), a_shape)?,
        tensor_storage(ctx, args.b, args.b.dtype(), b_shape)?,
    ];
    let output_width = if semantic == Semantic::GemmGeglu {
        n / 2
    } else {
        n
    };
    let output_buffer = tensor_storage(ctx, args.out, args.out.dtype(), &[m, output_width])?;
    let output_bytes = required_bytes(args.out.dtype(), &[m, output_width])?;
    reject_output_overlap(
        &output_buffer,
        output_bytes,
        &storage[0],
        required_bytes(args.a.dtype(), a_shape)?,
        "A",
    )?;
    reject_output_overlap(
        &output_buffer,
        output_bytes,
        &storage[1],
        required_bytes(args.b.dtype(), b_shape)?,
        "B",
    )?;
    let mut bindings = abi::Bindings {
        a: storage[0].ptr(),
        b: storage[1].ptr(),
        bias: std::ptr::null(),
        a_scales: std::ptr::null(),
        b_scales: std::ptr::null(),
        output: output_buffer.ptr(),
        stream: ctx.stream().handle(),
        alpha: args.alpha,
        output_scale: args.output_scale,
    };

    let projection_dtype = if has_row_channel_scales {
        args.out.dtype()
    } else if args.a.dtype() == DType::F8E4M3 {
        if args.out.dtype() == DType::F16 {
            DType::F16
        } else {
            // BF16 and F32 both have a wider exponent range than F16. Using
            // F16 here can turn a finite result into infinity before the
            // requested output conversion.
            DType::F32
        }
    } else {
        args.a.dtype()
    };
    for (tensor, expected_dtype, expected_shape, slot, name) in [
        (bias, projection_dtype, vec![n], 0, "bias"),
        (row_scales, DType::F32, vec![m], 2, "row scale"),
        (channel_scales, DType::F32, vec![n], 3, "channel scale"),
    ] {
        if let Some(tensor) = tensor {
            let buffer = tensor_storage(ctx, tensor, expected_dtype, &expected_shape)?;
            reject_output_overlap(
                &output_buffer,
                output_bytes,
                &buffer,
                required_bytes(expected_dtype, &expected_shape)?,
                name,
            )?;
            match slot {
                0 => bindings.bias = buffer.ptr(),
                2 => bindings.a_scales = buffer.ptr().cast(),
                _ => bindings.b_scales = buffer.ptr().cast(),
            }
            storage.push(buffer);
        }
    }
    storage.push(output_buffer);

    Ok(Normalized {
        api,
        spec: abi::Spec {
            version: 3,
            a_dtype: dtype(args.a.dtype())?,
            b_dtype: dtype(args.b.dtype())?,
            accumulation_dtype: dtype(args.policy.accumulation_dtype)?,
            output_dtype: dtype(args.out.dtype())?,
            quantization,
            a_alignment: alignment_class(bindings.a),
            b_alignment: alignment_class(bindings.b),
            bias_alignment: alignment_class(bindings.bias),
            a_scales_alignment: alignment_class(bindings.a_scales.cast::<std::ffi::c_void>()),
            b_scales_alignment: alignment_class(bindings.b_scales.cast::<std::ffi::c_void>()),
            output_alignment: alignment_class(bindings.output.cast_const()),
            m: m as i64,
            n: n as i64,
            k: k as i64,
            // Only the unit/non-unit distinction selects a candidate, so two
            // calls that differ only by scale share one tuned recipe.
            alpha_is_unit: u32::from(args.alpha == 1.0),
            output_scale_is_unit: u32::from(args.output_scale == 1.0),
        },
        policy: args.policy,
        bindings,
        storage,
        reference: args.reference,
    })
}

#[cfg(test)]
mod safety_tests {
    use super::checked_device_range;

    #[test]
    fn device_range_rejects_address_overflow() {
        let ptr = (usize::MAX - 3) as *mut std::ffi::c_void;
        assert!(checked_device_range(ptr, 8).is_err());
    }

    #[test]
    fn device_range_rejects_non_empty_null_pointer() {
        assert!(checked_device_range(std::ptr::null_mut(), 1).is_err());
    }
}
