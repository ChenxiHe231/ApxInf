//! Legacy `kernels::quantization` names over cuda-new quantization operators.

use apxinf_core::{DType, Result, Shape, Tensor};

use crate::{ops, CudaContext};

fn fixed_scale_e4m3(ctx: &CudaContext, input: &Tensor, scale: f32) -> Result<Tensor> {
    let mut out =
        ctx.allocate_output(Shape::new(input.shape().dims().to_vec()), DType::F8E4M3)?;
    let mut args =
        ops::QuantizationArgs::new(ops::QuantizationSemantic::FixedScaleE4m3, input, &mut out);
    args.scale = scale;
    ops::quantization(ctx, args)?;
    Ok(out)
}

/// `quantize_bf16_e4m3`: BF16 to E4M3 against one pre-calibrated scale.
pub fn quantize_bf16_e4m3(ctx: &CudaContext, input: &Tensor, scale: f32) -> Result<Tensor> {
    fixed_scale_e4m3(ctx, input, scale)
}

/// `quantize_f16_e4m3`: F16 to E4M3 against one pre-calibrated scale.
pub fn quantize_f16_e4m3(ctx: &CudaContext, input: &Tensor, scale: f32) -> Result<Tensor> {
    fixed_scale_e4m3(ctx, input, scale)
}
