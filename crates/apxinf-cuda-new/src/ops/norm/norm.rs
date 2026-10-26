use apxinf_core::{Result, Tensor};

use super::contracts::{normalize, BiasOrder, RawArgs, Semantic};
use super::launch;
use crate::CudaContext;

/// Inputs for ordinary RMS normalization.
pub struct RmsNormArgs<'a> {
    pub input: &'a Tensor,
    pub weight: &'a Tensor,
    pub normalized: &'a mut Tensor,
    pub eps: f32,
    /// Static dequantization scale when `normalized` is E4M3.
    pub output_scale: f32,
}

impl<'a> RmsNormArgs<'a> {
    pub fn new(
        input: &'a Tensor,
        weight: &'a Tensor,
        normalized: &'a mut Tensor,
        eps: f32,
    ) -> Self {
        Self {
            input,
            weight,
            normalized,
            eps,
            output_scale: 1.0,
        }
    }
}

/// Inputs for ordinary layer normalization.
pub struct LayerNormArgs<'a> {
    pub input: &'a Tensor,
    pub weight: &'a Tensor,
    pub bias: &'a Tensor,
    pub normalized: &'a mut Tensor,
    pub eps: f32,
    /// Static dequantization scale when `normalized` is E4M3.
    pub output_scale: f32,
}

impl<'a> LayerNormArgs<'a> {
    pub fn new(
        input: &'a Tensor,
        weight: &'a Tensor,
        bias: &'a Tensor,
        normalized: &'a mut Tensor,
        eps: f32,
    ) -> Self {
        Self {
            input,
            weight,
            bias,
            normalized,
            eps,
            output_scale: 1.0,
        }
    }
}

/// Inputs for adaptive RMS normalization. `norm_style` is `[2 * cols]`.
pub struct AdaptiveRmsNormArgs<'a> {
    pub input: &'a Tensor,
    pub norm_style: &'a Tensor,
    pub normalized: &'a mut Tensor,
    pub eps: f32,
    /// Static dequantization scale when `normalized` is E4M3.
    pub output_scale: f32,
}

impl<'a> AdaptiveRmsNormArgs<'a> {
    pub fn new(
        input: &'a Tensor,
        norm_style: &'a Tensor,
        normalized: &'a mut Tensor,
        eps: f32,
    ) -> Self {
        Self {
            input,
            norm_style,
            normalized,
            eps,
            output_scale: 1.0,
        }
    }
}

/// Inputs for bias plus residual. Bias is optional by contract.
pub struct BiasResidualArgs<'a> {
    pub input: &'a Tensor,
    pub bias: Option<&'a Tensor>,
    pub residual: &'a Tensor,
    pub hidden: &'a mut Tensor,
    pub bias_order: BiasOrder,
}

impl<'a> BiasResidualArgs<'a> {
    pub fn new(
        input: &'a Tensor,
        bias: Option<&'a Tensor>,
        residual: &'a Tensor,
        hidden: &'a mut Tensor,
    ) -> Self {
        Self {
            input,
            bias,
            residual,
            hidden,
            bias_order: BiasOrder::SingleRound,
        }
    }

    pub fn with_bias_order(mut self, bias_order: BiasOrder) -> Self {
        self.bias_order = bias_order;
        self
    }
}

pub struct BiasResidualRmsNormArgs<'a> {
    pub input: &'a Tensor,
    pub bias: Option<&'a Tensor>,
    pub residual: &'a Tensor,
    pub weight: &'a Tensor,
    pub hidden: &'a mut Tensor,
    pub normalized: &'a mut Tensor,
    pub eps: f32,
    /// Static dequantization scale when `normalized` is E4M3.
    pub output_scale: f32,
}

impl<'a> BiasResidualRmsNormArgs<'a> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        input: &'a Tensor,
        bias: Option<&'a Tensor>,
        residual: &'a Tensor,
        weight: &'a Tensor,
        hidden: &'a mut Tensor,
        normalized: &'a mut Tensor,
        eps: f32,
    ) -> Self {
        Self {
            input,
            bias,
            residual,
            weight,
            hidden,
            normalized,
            eps,
            output_scale: 1.0,
        }
    }
}

pub struct BiasResidualLayerNormArgs<'a> {
    pub input: &'a Tensor,
    pub bias: Option<&'a Tensor>,
    pub residual: &'a Tensor,
    pub weight: &'a Tensor,
    pub norm_bias: &'a Tensor,
    pub hidden: &'a mut Tensor,
    pub normalized: &'a mut Tensor,
    pub eps: f32,
    /// Static dequantization scale when `normalized` is E4M3.
    pub output_scale: f32,
    pub bias_order: BiasOrder,
}

impl<'a> BiasResidualLayerNormArgs<'a> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        input: &'a Tensor,
        bias: Option<&'a Tensor>,
        residual: &'a Tensor,
        weight: &'a Tensor,
        norm_bias: &'a Tensor,
        hidden: &'a mut Tensor,
        normalized: &'a mut Tensor,
        eps: f32,
    ) -> Self {
        Self {
            input,
            bias,
            residual,
            weight,
            norm_bias,
            hidden,
            normalized,
            eps,
            output_scale: 1.0,
            bias_order: BiasOrder::SingleRound,
        }
    }

    pub fn with_bias_order(mut self, bias_order: BiasOrder) -> Self {
        self.bias_order = bias_order;
        self
    }
}

/// Bias/residual combine followed by adaptive LayerNorm. `norm_style` packs
/// `[scale; shift]` as a `[2 * cols]` vector.
pub struct BiasResidualAdaptiveLayerNormArgs<'a> {
    pub input: &'a Tensor,
    pub bias: Option<&'a Tensor>,
    pub residual: &'a Tensor,
    pub norm_style: &'a Tensor,
    pub hidden: &'a mut Tensor,
    pub normalized: &'a mut Tensor,
    pub eps: f32,
    pub bias_order: BiasOrder,
}

impl<'a> BiasResidualAdaptiveLayerNormArgs<'a> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        input: &'a Tensor,
        bias: Option<&'a Tensor>,
        residual: &'a Tensor,
        norm_style: &'a Tensor,
        hidden: &'a mut Tensor,
        normalized: &'a mut Tensor,
        eps: f32,
    ) -> Self {
        Self {
            input,
            bias,
            residual,
            norm_style,
            hidden,
            normalized,
            eps,
            bias_order: BiasOrder::SingleRound,
        }
    }

    pub fn with_bias_order(mut self, bias_order: BiasOrder) -> Self {
        self.bias_order = bias_order;
        self
    }
}

/// Inputs for adaptive gate plus residual. `gate_style` is `[3 * cols]`.
pub struct AdaGateResidualArgs<'a> {
    pub input: &'a Tensor,
    pub residual: &'a Tensor,
    pub gate_style: &'a Tensor,
    pub hidden: &'a mut Tensor,
}

impl<'a> AdaGateResidualArgs<'a> {
    pub fn new(
        input: &'a Tensor,
        residual: &'a Tensor,
        gate_style: &'a Tensor,
        hidden: &'a mut Tensor,
    ) -> Self {
        Self {
            input,
            residual,
            gate_style,
            hidden,
        }
    }
}

pub struct AdaGateResidualRmsNormArgs<'a> {
    pub input: &'a Tensor,
    pub residual: &'a Tensor,
    pub norm_style: &'a Tensor,
    pub gate_style: &'a Tensor,
    pub hidden: &'a mut Tensor,
    pub normalized: &'a mut Tensor,
    pub eps: f32,
    /// Static dequantization scale when `normalized` is E4M3.
    pub output_scale: f32,
}

impl<'a> AdaGateResidualRmsNormArgs<'a> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        input: &'a Tensor,
        residual: &'a Tensor,
        norm_style: &'a Tensor,
        gate_style: &'a Tensor,
        hidden: &'a mut Tensor,
        normalized: &'a mut Tensor,
        eps: f32,
    ) -> Self {
        Self {
            input,
            residual,
            norm_style,
            gate_style,
            hidden,
            normalized,
            eps,
            output_scale: 1.0,
        }
    }
}

fn execute(ctx: &CudaContext, args: RawArgs<'_>) -> Result<()> {
    launch::execute(ctx, normalize(ctx, args)?)
}

pub fn rms_norm(ctx: &CudaContext, args: RmsNormArgs<'_>) -> Result<()> {
    execute(
        ctx,
        RawArgs {
            semantic: Semantic::Rms,
            bias_order: BiasOrder::SingleRound,
            input: args.input,
            bias: None,
            residual: None,
            weight: Some(args.weight),
            norm_bias: None,
            norm_style: None,
            gate_style: None,
            hidden: None,
            normalized: Some(args.normalized),
            eps: args.eps,
            output_scale: args.output_scale,
        },
    )
}

pub fn layer_norm(ctx: &CudaContext, args: LayerNormArgs<'_>) -> Result<()> {
    execute(
        ctx,
        RawArgs {
            semantic: Semantic::Layer,
            bias_order: BiasOrder::SingleRound,
            input: args.input,
            bias: None,
            residual: None,
            weight: Some(args.weight),
            norm_bias: Some(args.bias),
            norm_style: None,
            gate_style: None,
            hidden: None,
            normalized: Some(args.normalized),
            eps: args.eps,
            output_scale: args.output_scale,
        },
    )
}

pub fn adaptive_rms_norm(ctx: &CudaContext, args: AdaptiveRmsNormArgs<'_>) -> Result<()> {
    execute(
        ctx,
        RawArgs {
            semantic: Semantic::AdaptiveRms,
            bias_order: BiasOrder::SingleRound,
            input: args.input,
            bias: None,
            residual: None,
            weight: None,
            norm_bias: None,
            norm_style: Some(args.norm_style),
            gate_style: None,
            hidden: None,
            normalized: Some(args.normalized),
            eps: args.eps,
            output_scale: args.output_scale,
        },
    )
}

pub fn bias_residual(ctx: &CudaContext, args: BiasResidualArgs<'_>) -> Result<()> {
    execute(
        ctx,
        RawArgs {
            semantic: Semantic::BiasResidual,
            bias_order: args.bias_order,
            input: args.input,
            bias: args.bias,
            residual: Some(args.residual),
            weight: None,
            norm_bias: None,
            norm_style: None,
            gate_style: None,
            hidden: Some(args.hidden),
            normalized: None,
            eps: 1e-6,
            output_scale: 1.0,
        },
    )
}

pub fn bias_residual_rms_norm(ctx: &CudaContext, args: BiasResidualRmsNormArgs<'_>) -> Result<()> {
    execute(
        ctx,
        RawArgs {
            semantic: Semantic::BiasResidualRms,
            bias_order: BiasOrder::SingleRound,
            input: args.input,
            bias: args.bias,
            residual: Some(args.residual),
            weight: Some(args.weight),
            norm_bias: None,
            norm_style: None,
            gate_style: None,
            hidden: Some(args.hidden),
            normalized: Some(args.normalized),
            eps: args.eps,
            output_scale: args.output_scale,
        },
    )
}

pub fn bias_residual_layer_norm(
    ctx: &CudaContext,
    args: BiasResidualLayerNormArgs<'_>,
) -> Result<()> {
    execute(
        ctx,
        RawArgs {
            semantic: Semantic::BiasResidualLayer,
            bias_order: args.bias_order,
            input: args.input,
            bias: args.bias,
            residual: Some(args.residual),
            weight: Some(args.weight),
            norm_bias: Some(args.norm_bias),
            norm_style: None,
            gate_style: None,
            hidden: Some(args.hidden),
            normalized: Some(args.normalized),
            eps: args.eps,
            output_scale: args.output_scale,
        },
    )
}

pub fn ada_gate_residual(ctx: &CudaContext, args: AdaGateResidualArgs<'_>) -> Result<()> {
    execute(
        ctx,
        RawArgs {
            semantic: Semantic::AdaGateResidual,
            bias_order: BiasOrder::SingleRound,
            input: args.input,
            bias: None,
            residual: Some(args.residual),
            weight: None,
            norm_bias: None,
            norm_style: None,
            gate_style: Some(args.gate_style),
            hidden: Some(args.hidden),
            normalized: None,
            eps: 1e-6,
            output_scale: 1.0,
        },
    )
}

pub fn ada_gate_residual_rms_norm(
    ctx: &CudaContext,
    args: AdaGateResidualRmsNormArgs<'_>,
) -> Result<()> {
    execute(
        ctx,
        RawArgs {
            semantic: Semantic::AdaGateResidualRms,
            bias_order: BiasOrder::SingleRound,
            input: args.input,
            bias: None,
            residual: Some(args.residual),
            weight: None,
            norm_bias: None,
            norm_style: Some(args.norm_style),
            gate_style: Some(args.gate_style),
            hidden: Some(args.hidden),
            normalized: Some(args.normalized),
            eps: args.eps,
            output_scale: args.output_scale,
        },
    )
}

pub fn bias_residual_adaptive_layer_norm(
    ctx: &CudaContext,
    args: BiasResidualAdaptiveLayerNormArgs<'_>,
) -> Result<()> {
    execute(
        ctx,
        RawArgs {
            semantic: Semantic::BiasResidualAdaptiveLayer,
            bias_order: args.bias_order,
            input: args.input,
            bias: args.bias,
            residual: Some(args.residual),
            weight: None,
            norm_bias: None,
            norm_style: Some(args.norm_style),
            gate_style: None,
            hidden: Some(args.hidden),
            normalized: Some(args.normalized),
            eps: args.eps,
            output_scale: 1.0,
        },
    )
}
