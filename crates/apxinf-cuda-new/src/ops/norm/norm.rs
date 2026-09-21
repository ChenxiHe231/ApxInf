use apxinf_core::{Result, Tensor};

use super::contracts::{normalize, NormPolicy, RawArgs, Semantic};
use super::execution;
use crate::CudaContext;

macro_rules! impl_policy_default {
    ($name:ident) => {
        impl $name<'_> {
            pub fn with_policy(mut self, policy: NormPolicy) -> Self {
                self.policy = policy;
                self
            }
        }
    };
}

/// Inputs for ordinary RMS normalization.
pub struct RmsNormArgs<'a> {
    pub input: &'a Tensor,
    pub weight: &'a Tensor,
    pub normalized: &'a mut Tensor,
    pub eps: f32,
    pub policy: NormPolicy,
}

impl<'a> RmsNormArgs<'a> {
    pub fn new(input: &'a Tensor, weight: &'a Tensor, normalized: &'a mut Tensor, eps: f32) -> Self {
        Self { input, weight, normalized, eps, policy: NormPolicy::default() }
    }
}
impl_policy_default!(RmsNormArgs);

/// Inputs for ordinary layer normalization.
pub struct LayerNormArgs<'a> {
    pub input: &'a Tensor,
    pub weight: &'a Tensor,
    pub bias: &'a Tensor,
    pub normalized: &'a mut Tensor,
    pub eps: f32,
    pub policy: NormPolicy,
}

impl<'a> LayerNormArgs<'a> {
    pub fn new(input: &'a Tensor, weight: &'a Tensor, bias: &'a Tensor, normalized: &'a mut Tensor, eps: f32) -> Self {
        Self { input, weight, bias, normalized, eps, policy: NormPolicy::default() }
    }
}
impl_policy_default!(LayerNormArgs);

/// Inputs for adaptive RMS normalization. `norm_style` is `[2 * cols]`.
pub struct AdaptiveRmsNormArgs<'a> {
    pub input: &'a Tensor,
    pub norm_style: &'a Tensor,
    pub normalized: &'a mut Tensor,
    pub eps: f32,
    pub policy: NormPolicy,
}

impl<'a> AdaptiveRmsNormArgs<'a> {
    pub fn new(input: &'a Tensor, norm_style: &'a Tensor, normalized: &'a mut Tensor, eps: f32) -> Self {
        Self { input, norm_style, normalized, eps, policy: NormPolicy::default() }
    }
}
impl_policy_default!(AdaptiveRmsNormArgs);

/// Inputs for bias plus residual. Bias is optional by contract.
pub struct BiasResidualArgs<'a> {
    pub input: &'a Tensor,
    pub bias: Option<&'a Tensor>,
    pub residual: &'a Tensor,
    pub hidden: &'a mut Tensor,
    pub policy: NormPolicy,
}

impl<'a> BiasResidualArgs<'a> {
    pub fn new(input: &'a Tensor, bias: Option<&'a Tensor>, residual: &'a Tensor, hidden: &'a mut Tensor) -> Self {
        Self { input, bias, residual, hidden, policy: NormPolicy::default() }
    }
}
impl_policy_default!(BiasResidualArgs);

pub struct BiasResidualRmsNormArgs<'a> {
    pub input: &'a Tensor,
    pub bias: Option<&'a Tensor>,
    pub residual: &'a Tensor,
    pub weight: &'a Tensor,
    pub hidden: &'a mut Tensor,
    pub normalized: &'a mut Tensor,
    pub eps: f32,
    pub policy: NormPolicy,
}

impl<'a> BiasResidualRmsNormArgs<'a> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(input: &'a Tensor, bias: Option<&'a Tensor>, residual: &'a Tensor, weight: &'a Tensor, hidden: &'a mut Tensor, normalized: &'a mut Tensor, eps: f32) -> Self {
        Self { input, bias, residual, weight, hidden, normalized, eps, policy: NormPolicy::default() }
    }
}
impl_policy_default!(BiasResidualRmsNormArgs);

pub struct BiasResidualLayerNormArgs<'a> {
    pub input: &'a Tensor,
    pub bias: Option<&'a Tensor>,
    pub residual: &'a Tensor,
    pub weight: &'a Tensor,
    pub norm_bias: &'a Tensor,
    pub hidden: &'a mut Tensor,
    pub normalized: &'a mut Tensor,
    pub eps: f32,
    pub policy: NormPolicy,
}

impl<'a> BiasResidualLayerNormArgs<'a> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(input: &'a Tensor, bias: Option<&'a Tensor>, residual: &'a Tensor, weight: &'a Tensor, norm_bias: &'a Tensor, hidden: &'a mut Tensor, normalized: &'a mut Tensor, eps: f32) -> Self {
        Self { input, bias, residual, weight, norm_bias, hidden, normalized, eps, policy: NormPolicy::default() }
    }
}
impl_policy_default!(BiasResidualLayerNormArgs);

/// Inputs for adaptive gate plus residual. `gate_style` is `[3 * cols]`.
pub struct AdaGateResidualArgs<'a> {
    pub input: &'a Tensor,
    pub residual: &'a Tensor,
    pub gate_style: &'a Tensor,
    pub hidden: &'a mut Tensor,
    pub policy: NormPolicy,
}

impl<'a> AdaGateResidualArgs<'a> {
    pub fn new(input: &'a Tensor, residual: &'a Tensor, gate_style: &'a Tensor, hidden: &'a mut Tensor) -> Self {
        Self { input, residual, gate_style, hidden, policy: NormPolicy::default() }
    }
}
impl_policy_default!(AdaGateResidualArgs);

pub struct AdaGateResidualRmsNormArgs<'a> {
    pub input: &'a Tensor,
    pub residual: &'a Tensor,
    pub norm_style: &'a Tensor,
    pub gate_style: &'a Tensor,
    pub hidden: &'a mut Tensor,
    pub normalized: &'a mut Tensor,
    pub eps: f32,
    pub policy: NormPolicy,
}

impl<'a> AdaGateResidualRmsNormArgs<'a> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(input: &'a Tensor, residual: &'a Tensor, norm_style: &'a Tensor, gate_style: &'a Tensor, hidden: &'a mut Tensor, normalized: &'a mut Tensor, eps: f32) -> Self {
        Self { input, residual, norm_style, gate_style, hidden, normalized, eps, policy: NormPolicy::default() }
    }
}
impl_policy_default!(AdaGateResidualRmsNormArgs);

/// BF16-only two-rounding bias-then-residual contract.
pub struct BiasThenResidualArgs<'a> {
    pub input: &'a Tensor,
    pub bias: Option<&'a Tensor>,
    pub residual: &'a Tensor,
    pub hidden: &'a mut Tensor,
    pub policy: NormPolicy,
}

impl<'a> BiasThenResidualArgs<'a> {
    pub fn new(input: &'a Tensor, bias: Option<&'a Tensor>, residual: &'a Tensor, hidden: &'a mut Tensor) -> Self {
        Self { input, bias, residual, hidden, policy: NormPolicy::default() }
    }
}
impl_policy_default!(BiasThenResidualArgs);

fn execute(ctx: &CudaContext, args: RawArgs<'_>) -> Result<()> {
    execution::execute(ctx, normalize(ctx, args)?)
}

pub fn rms_norm(ctx: &CudaContext, args: RmsNormArgs<'_>) -> Result<()> {
    execute(ctx, RawArgs { semantic: Semantic::Rms, input: args.input, bias: None, residual: None, weight: Some(args.weight), norm_bias: None, norm_style: None, gate_style: None, hidden: None, normalized: Some(args.normalized), eps: args.eps, policy: args.policy })
}

pub fn layer_norm(ctx: &CudaContext, args: LayerNormArgs<'_>) -> Result<()> {
    execute(ctx, RawArgs { semantic: Semantic::Layer, input: args.input, bias: None, residual: None, weight: Some(args.weight), norm_bias: Some(args.bias), norm_style: None, gate_style: None, hidden: None, normalized: Some(args.normalized), eps: args.eps, policy: args.policy })
}

pub fn adaptive_rms_norm(ctx: &CudaContext, args: AdaptiveRmsNormArgs<'_>) -> Result<()> {
    execute(ctx, RawArgs { semantic: Semantic::AdaptiveRms, input: args.input, bias: None, residual: None, weight: None, norm_bias: None, norm_style: Some(args.norm_style), gate_style: None, hidden: None, normalized: Some(args.normalized), eps: args.eps, policy: args.policy })
}

pub fn bias_residual(ctx: &CudaContext, args: BiasResidualArgs<'_>) -> Result<()> {
    execute(ctx, RawArgs { semantic: Semantic::BiasResidual, input: args.input, bias: args.bias, residual: Some(args.residual), weight: None, norm_bias: None, norm_style: None, gate_style: None, hidden: Some(args.hidden), normalized: None, eps: 1e-6, policy: args.policy })
}

pub fn bias_residual_rms_norm(ctx: &CudaContext, args: BiasResidualRmsNormArgs<'_>) -> Result<()> {
    execute(ctx, RawArgs { semantic: Semantic::BiasResidualRms, input: args.input, bias: args.bias, residual: Some(args.residual), weight: Some(args.weight), norm_bias: None, norm_style: None, gate_style: None, hidden: Some(args.hidden), normalized: Some(args.normalized), eps: args.eps, policy: args.policy })
}

pub fn bias_residual_layer_norm(ctx: &CudaContext, args: BiasResidualLayerNormArgs<'_>) -> Result<()> {
    execute(ctx, RawArgs { semantic: Semantic::BiasResidualLayer, input: args.input, bias: args.bias, residual: Some(args.residual), weight: Some(args.weight), norm_bias: Some(args.norm_bias), norm_style: None, gate_style: None, hidden: Some(args.hidden), normalized: Some(args.normalized), eps: args.eps, policy: args.policy })
}

pub fn ada_gate_residual(ctx: &CudaContext, args: AdaGateResidualArgs<'_>) -> Result<()> {
    execute(ctx, RawArgs { semantic: Semantic::AdaGateResidual, input: args.input, bias: None, residual: Some(args.residual), weight: None, norm_bias: None, norm_style: None, gate_style: Some(args.gate_style), hidden: Some(args.hidden), normalized: None, eps: 1e-6, policy: args.policy })
}

pub fn ada_gate_residual_rms_norm(ctx: &CudaContext, args: AdaGateResidualRmsNormArgs<'_>) -> Result<()> {
    execute(ctx, RawArgs { semantic: Semantic::AdaGateResidualRms, input: args.input, bias: None, residual: Some(args.residual), weight: None, norm_bias: None, norm_style: Some(args.norm_style), gate_style: Some(args.gate_style), hidden: Some(args.hidden), normalized: Some(args.normalized), eps: args.eps, policy: args.policy })
}

pub fn bias_then_residual(ctx: &CudaContext, args: BiasThenResidualArgs<'_>) -> Result<()> {
    execute(ctx, RawArgs { semantic: Semantic::BiasThenResidual, input: args.input, bias: args.bias, residual: Some(args.residual), weight: None, norm_bias: None, norm_style: None, gate_style: None, hidden: Some(args.hidden), normalized: None, eps: 1e-6, policy: args.policy })
}
