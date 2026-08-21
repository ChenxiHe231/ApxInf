//! Device-ready static-FP8 linear weights.

use apxinf_core::{Backend, DType, Error, Result, Tensor};

#[cfg(feature = "cuda")]
use super::backend::{kernels, RuntimeBackend};
use super::{quantize_e4m3_absmax, LinearWeights};

#[derive(Debug)]
pub struct Fp8LinearWeights {
    /// `[input, output]` CUDA E4M3 matrix.
    pub weight: Tensor,
    pub weight_scale: f32,
    /// Physical [gate256,up256] column order for exact dual-GeGLU paths.
    /// When true this tensor must never be sent to a plain GEMM.
    pub dual_geglu_interleaved: bool,
    /// Additional [gate256,up256] resident matrix used only by auto routing.
    /// routing. The primary `weight` remains plain so every other shape and
    /// backend keeps its original physical contract.
    pub dual_geglu_auto_interleaved: Option<Tensor>,
    /// Bias stays FP16 and is fused into the consumer kernel.
    pub bias: Option<Tensor>,
}

impl Fp8LinearWeights {
    #[cfg(feature = "cuda")]
    pub fn as_kernel_view(&self) -> kernels::gemm::Fp8WeightView<'_> {
        kernels::gemm::Fp8WeightView {
            values_e4m3: &self.weight,
            scale: self.weight_scale,
            dual_geglu_interleaved: self.dual_geglu_interleaved,
            dual_geglu_auto_interleaved: self.dual_geglu_auto_interleaved.as_ref(),
        }
    }

    pub fn from_host(linear: &LinearWeights, backend: &dyn Backend) -> Result<Self> {
        Self::from_host_parts(&[linear], backend)
    }

    /// Concatenate projections along their output dimension before applying
    /// one absmax quantization scale. This produces graph-ready QKV and
    /// gate/up matrices without runtime concatenation or mixed descales.
    pub fn from_host_parts(linears: &[&LinearWeights], backend: &dyn Backend) -> Result<Self> {
        Self::from_host_parts_with_dual_layout(linears, backend, true)
    }

    pub(crate) fn from_host_parts_with_dual_layout(
        linears: &[&LinearWeights],
        backend: &dyn Backend,
        allow_dual_layout: bool,
    ) -> Result<Self> {
        if linears.is_empty() {
            return Err(Error::Other("cannot pack an empty FP8 linear group".into()));
        }
        let fp8_dual_geglu_mode = fp8_dual_geglu_mode()?;
        let dual_geglu_exact = allow_dual_layout
            && linears.len() == 2
            && linears
                .iter()
                .all(|linear| linear.weight.shape().dims() == [2048, 16384]);
        let dual_geglu_interleaved =
            dual_geglu_exact && fp8_dual_geglu_mode == Fp8DualGeGluMode::On;
        let plain_host = concat_host_2d(&linears.iter().map(|x| &x.weight).collect::<Vec<_>>())?;
        let interleaved_host = if dual_geglu_exact && fp8_dual_geglu_mode != Fp8DualGeGluMode::Off {
            Some(interleave_gate_up_host(
                &linears[0].weight,
                &linears[1].weight,
                256,
            )?)
        } else {
            None
        };
        let weight_host = if dual_geglu_interleaved {
            interleaved_host.as_ref().unwrap()
        } else {
            &plain_host
        };
        #[cfg(feature = "cuda")]
        let (weight, weight_scale) =
            if let Some(cuda_backend) = backend.as_any().downcast_ref::<RuntimeBackend>() {
                // Quantizing billions of parameters with the scalar CPU E4M3
                // encoder is prohibitively slow on Jetson. Upload FP16 once and
                // let the CUDA conversion kernel produce the resident FP8 matrix.
                let (weight_f16, amax) = fp16_host_and_amax(weight_host)?;
                let weight_scale = if amax == 0.0 {
                    1.0
                } else {
                    amax / super::E4M3_MAX
                };
                let weight_f16 = backend.to_device(&weight_f16)?;
                let weight = kernels::quantization::quantize_f16_e4m3(
                    cuda_backend.context(),
                    &weight_f16,
                    weight_scale,
                )?;
                (weight, weight_scale)
            } else {
                let quantized = quantize_e4m3_absmax(weight_host)?;
                (backend.to_device(&quantized.values)?, quantized.scale)
            };
        #[cfg(not(feature = "cuda"))]
        let (weight, weight_scale) = {
            let quantized = quantize_e4m3_absmax(weight_host)?;
            (backend.to_device(&quantized.values)?, quantized.scale)
        };
        let dual_geglu_auto_interleaved = if dual_geglu_exact
            && fp8_dual_geglu_mode == Fp8DualGeGluMode::Auto
        {
            let interleaved_host = interleaved_host.as_ref().unwrap();
            #[cfg(feature = "cuda")]
            let (interleaved, interleaved_scale) =
                if let Some(cuda_backend) = backend.as_any().downcast_ref::<RuntimeBackend>() {
                    let (weight_f16, amax) = fp16_host_and_amax(interleaved_host)?;
                    let interleaved_scale = if amax == 0.0 {
                        1.0
                    } else {
                        amax / super::E4M3_MAX
                    };
                    let weight_f16 = backend.to_device(&weight_f16)?;
                    let interleaved = kernels::quantization::quantize_f16_e4m3(
                        cuda_backend.context(),
                        &weight_f16,
                        weight_scale,
                    )?;
                    (interleaved, interleaved_scale)
                } else {
                    let quantized = quantize_e4m3_absmax(interleaved_host)?;
                    (backend.to_device(&quantized.values)?, quantized.scale)
                };
            #[cfg(not(feature = "cuda"))]
            let (interleaved, interleaved_scale) = {
                let quantized = quantize_e4m3_absmax(interleaved_host)?;
                (backend.to_device(&quantized.values)?, quantized.scale)
            };
            if weight_scale.to_bits() != interleaved_scale.to_bits() {
                return Err(Error::Other(format!(
                    "FP8 dual GeGLU auto layouts changed joint scale bits: plain={:#010x}, interleaved={:#010x}",
                    weight_scale.to_bits(),
                    interleaved_scale.to_bits()
                )));
            }
            Some(interleaved)
        } else {
            None
        };
        let bias = if linears.iter().all(|x| x.bias.is_none()) {
            None
        } else if linears.iter().all(|x| x.bias.is_some()) {
            let biases = linears
                .iter()
                .map(|x| x.bias.as_ref().unwrap())
                .collect::<Vec<_>>();
            Some(backend.to_device(&concat_host_1d_f16(&biases)?)?)
        } else {
            return Err(Error::Other(
                "cannot pack projections with a mixture of present and absent biases".into(),
            ));
        };
        Ok(Self {
            weight,
            weight_scale,
            dual_geglu_interleaved,
            dual_geglu_auto_interleaved,
            bias,
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum Fp8DualGeGluMode {
    Auto,
    Off,
    On,
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

fn interleave_gate_up_host(gate: &Tensor, up: &Tensor, tile: usize) -> Result<Tensor> {
    let gate_shape = gate.shape().dims();
    let up_shape = up.shape().dims();
    if gate_shape.len() != 2 || gate_shape != up_shape || tile == 0 || gate_shape[1] % tile != 0 {
        return Err(Error::Other(format!(
            "FP8 dual GeGLU requires equal 2D Gate/Up widths divisible by {tile}, got {gate_shape:?} and {up_shape:?}"
        )));
    }
    let rows = gate_shape[0];
    let width = gate_shape[1];
    let gate_values = gate.to_f32_vec()?;
    let up_values = up.to_f32_vec()?;
    let mut output = vec![0.0f32; rows * width * 2];
    for row in 0..rows {
        for tile_index in 0..width / tile {
            let src = row * width + tile_index * tile;
            let dst = row * width * 2 + tile_index * tile * 2;
            output[dst..dst + tile].copy_from_slice(&gate_values[src..src + tile]);
            output[dst + tile..dst + 2 * tile].copy_from_slice(&up_values[src..src + tile]);
        }
    }
    // Max is order-independent for finite model weights. Checking raw bits
    // here ensures the interleaved candidate keeps the exact joint scale.
    let source_amax = gate_values
        .iter()
        .chain(&up_values)
        .fold(0.0f32, |maximum, value| maximum.max(value.abs()));
    let interleaved_amax = output
        .iter()
        .fold(0.0f32, |maximum, value| maximum.max(value.abs()));
    if source_amax.to_bits() != interleaved_amax.to_bits() {
        return Err(Error::Other(
            "FP8 dual GeGLU interleaving changed joint amax raw bits".into(),
        ));
    }
    Tensor::from_f32(vec![rows, width * 2], &output)
}

#[cfg(feature = "cuda")]
fn fp16_host_and_amax(tensor: &Tensor) -> Result<(Tensor, f32)> {
    let values = tensor.to_f32_vec()?;
    let amax = values
        .iter()
        .fold(0.0f32, |maximum, value| maximum.max(value.abs()));
    let values = values
        .into_iter()
        .map(half::f16::from_f32)
        .collect::<Vec<_>>();
    Ok((
        Tensor::from_f16(tensor.shape().dims().to_vec(), &values)?,
        amax,
    ))
}

pub fn fp16_to_device(tensor: &Tensor, backend: &dyn Backend) -> Result<Tensor> {
    let values = tensor.to_f32_vec()?;
    let values = values
        .iter()
        .map(|value| half::f16::from_f32(*value))
        .collect::<Vec<_>>();
    backend.to_device(&Tensor::from_f16(tensor.shape().dims().to_vec(), &values)?)
}

pub(super) fn concat_host_2d(tensors: &[&Tensor]) -> Result<Tensor> {
    let first = tensors
        .first()
        .ok_or_else(|| Error::Other("empty tensor concatenation".into()))?;
    let dims = first.shape().dims();
    if dims.len() != 2 {
        return Err(Error::Other(format!("expected 2D weight, got {dims:?}")));
    }
    let rows = dims[0];
    let widths = tensors
        .iter()
        .map(|tensor| {
            let dims = tensor.shape().dims();
            if dims.len() != 2 || dims[0] != rows {
                return Err(Error::Other("packed linear input dimensions differ".into()));
            }
            Ok(dims[1])
        })
        .collect::<Result<Vec<_>>>()?;
    let total_cols = widths.iter().sum::<usize>();
    let sources = tensors
        .iter()
        .map(|tensor| tensor.to_f32_vec())
        .collect::<Result<Vec<_>>>()?;
    let mut output = vec![0.0f32; rows * total_cols];
    for row in 0..rows {
        let mut output_col = 0;
        for (source, width) in sources.iter().zip(&widths) {
            output[row * total_cols + output_col..row * total_cols + output_col + width]
                .copy_from_slice(&source[row * width..(row + 1) * width]);
            output_col += width;
        }
    }
    Tensor::from_f32(vec![rows, total_cols], &output)
}

fn concat_host_1d_f16(tensors: &[&Tensor]) -> Result<Tensor> {
    let mut output = Vec::new();
    for tensor in tensors {
        if tensor.shape().dims().len() != 1 || tensor.dtype() == DType::F8E4M3 {
            return Err(Error::Other("packed biases must be non-FP8 vectors".into()));
        }
        output.extend(tensor.to_f32_vec()?.into_iter().map(half::f16::from_f32));
    }
    Tensor::from_f16(vec![output.len()], &output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use apxinf_core::CpuBackend;

    fn linear(weight: &[f32], shape: [usize; 2], bias: Option<&[f32]>) -> LinearWeights {
        LinearWeights {
            weight: Tensor::from_f32(shape.to_vec(), weight).unwrap(),
            bias: bias.map(|x| Tensor::from_f32(vec![x.len()], x).unwrap()),
        }
    }

    #[test]
    fn packs_qkv_before_quantization() {
        let q = linear(&[1., 2., 3., 4.], [2, 2], Some(&[1., 2.]));
        let k = linear(&[5., 6.], [2, 1], Some(&[3.]));
        let v = linear(&[7., 8.], [2, 1], Some(&[4.]));
        let packed = Fp8LinearWeights::from_host_parts(&[&q, &k, &v], &CpuBackend).unwrap();
        assert_eq!(packed.weight.shape().dims(), &[2, 4]);
        assert_eq!(packed.weight.dtype(), DType::F8E4M3);
        let bias = packed.bias.unwrap();
        assert_eq!(bias.dtype(), DType::F16);
        assert_eq!(bias.to_f32_vec().unwrap(), vec![1., 2., 3., 4.]);
    }

    #[test]
    fn fp8_dual_geglu_mode_parser_is_tri_state_and_defaults_auto() {
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
    fn fp8_dual_geglu_eighteen_layer_interleave_preserves_bytes_and_scale() {
        const ROWS: usize = 2;
        const WIDTH: usize = 1024;
        const TILE: usize = 256;
        for layer in 0..18usize {
            let gate = (0..ROWS * WIDTH)
                .map(|index| ((index * 17 + layer * 31) % 1009) as f32 / 127.0 - 4.0)
                .collect::<Vec<_>>();
            let up = (0..ROWS * WIDTH)
                .map(|index| ((index * 29 + layer * 43) % 1013) as f32 / 131.0 - 3.5)
                .collect::<Vec<_>>();
            let gate = Tensor::from_f32(vec![ROWS, WIDTH], &gate).unwrap();
            let up = Tensor::from_f32(vec![ROWS, WIDTH], &up).unwrap();
            let plain = concat_host_2d(&[&gate, &up]).unwrap();
            let interleaved = interleave_gate_up_host(&gate, &up, TILE).unwrap();
            let plain_q = quantize_e4m3_absmax(&plain).unwrap();
            let interleaved_q = quantize_e4m3_absmax(&interleaved).unwrap();
            assert_eq!(plain_q.scale.to_bits(), interleaved_q.scale.to_bits());
            let plain_bytes = plain_q.values.as_f8_e4m3().unwrap();
            let interleaved_bytes = interleaved_q.values.as_f8_e4m3().unwrap();
            for row in 0..ROWS {
                for tile_index in 0..WIDTH / TILE {
                    let plain_gate = row * 2 * WIDTH + tile_index * TILE;
                    let plain_up = row * 2 * WIDTH + WIDTH + tile_index * TILE;
                    let packed = row * 2 * WIDTH + tile_index * 2 * TILE;
                    assert_eq!(
                        &interleaved_bytes[packed..packed + TILE],
                        &plain_bytes[plain_gate..plain_gate + TILE]
                    );
                    assert_eq!(
                        &interleaved_bytes[packed + TILE..packed + 2 * TILE],
                        &plain_bytes[plain_up..plain_up + TILE]
                    );
                }
            }
        }
    }
}
