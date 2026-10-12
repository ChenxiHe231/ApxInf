use apxinf_core::{Result, Tensor};

use super::contracts::{invalid, normalize, Fields, RawArgs};
use super::launch;
use crate::ffi::abi::norm as abi;
use crate::CudaContext;

/// Which reduction the row-wise normalization runs over a `[rows, cols]`
/// activation. Both reduce in f32 regardless of the F16/BF16 storage.
pub enum NormKind<'a> {
    /// `x[r,c] / sqrt(mean(x[r,:]^2) + eps)`
    Rms,
    /// `(x[r,c] - mean(x[r,:])) / sqrt(var(x[r,:]) + eps)`, then `+ bias[c]`
    /// after the affine. `bias` is the `[cols]` LayerNorm shift and is
    /// separate from the projection bias a fused add takes.
    Layer { bias: &'a Tensor },
}

/// How the normalized row is rescaled. There is no `None`: [`residual_add`],
/// the one form that applies no affine, takes no `Affine` at all.
pub enum Affine<'a> {
    /// `* weight[c]`, or `* (1 + weight[c])` when `offset`.
    ///
    /// The two settings are **not** one kernel with a flag: `offset` selects
    /// the Qwen3.8 RMSNorm kernel, which has its own block size and reduction
    /// tree, so the results differ by more than the `1 +`. That kernel covers
    /// exactly one pipeline -- [`norm`] with [`NormKind::Rms`], BF16 in and
    /// out, no `post` -- and every other combination is `UNSUPPORTED`; in
    /// particular [`fused_add_norm`] has no `offset` form.
    Weight { weight: &'a Tensor, offset: bool },
    /// `* (1 + style[c]) + style[cols + c]`, where `style` is `[2 * cols]`:
    /// scale in `[0, cols)` and shift in `[cols, 2 * cols)`.
    AdaScaleShift { style: &'a Tensor },
}

/// Fused output quantization, applied instead of writing the storage dtype.
pub enum QuantOut {
    /// E4M3 saturating at +/-448 against a static dequantization `scale`.
    /// F16 input only.
    E4m3 { scale: f32 },
}

/// Rounding order of a bias plus residual combine.
pub enum BiasOrder {
    /// One rounding of `input + bias + residual`.
    SingleRound,
    /// `input + bias` rounds to the storage dtype before the residual is added
    /// and rounded again. BF16 only, and observably different from
    /// [`Self::SingleRound`] at rounding boundaries: it reproduces the legacy
    /// two-kernel contract in one launch.
    RoundAfterBias,
}

/// Pure row-wise normalization: one input, one output, no residual.
pub struct NormArgs<'a> {
    pub input: &'a Tensor,
    pub out: &'a mut Tensor,
    pub kind: NormKind<'a>,
    pub affine: Affine<'a>,
    pub eps: f32,
    pub post: Option<QuantOut>,
}

impl<'a> NormArgs<'a> {
    pub fn new(
        input: &'a Tensor,
        out: &'a mut Tensor,
        kind: NormKind<'a>,
        affine: Affine<'a>,
        eps: f32,
    ) -> Self {
        Self {
            input,
            out,
            kind,
            affine,
            eps,
            post: None,
        }
    }

    /// `out[r,c] = input[r,c] * rms(input[r,:])^-1 * (1 + weight[c])`, the
    /// Qwen3.8 RMSNorm. See [`Affine::Weight`] for why this is not the same
    /// kernel as the `offset: false` form.
    pub fn rms_weight_offset(
        input: &'a Tensor,
        weight: &'a Tensor,
        out: &'a mut Tensor,
        eps: f32,
    ) -> Self {
        Self::new(
            input,
            out,
            NormKind::Rms,
            Affine::Weight {
                weight,
                offset: true,
            },
            eps,
        )
    }
}

/// Residual combine followed by normalization, writing both the combined row
/// (`hidden`) and the normalized row (`normalized`).
///
/// `bias` and `gate` are mutually exclusive: the gate multiplies the
/// projection where the bias would be added. `gate` is `[3 * cols]` and only
/// its third segment, `[2 * cols, 3 * cols)`, is read -- it is not
/// interchangeable with [`Affine::AdaScaleShift`]'s `[2 * cols]` style.
///
/// The gated form reads the gate from the *current* layer's conditioning and
/// normalizes with the *next* layer's: the pi05 action expert folds the next
/// layer's normalization into this layer's residual.
pub struct FusedAddNormArgs<'a> {
    pub input: &'a Tensor,
    pub residual: &'a Tensor,
    pub bias: Option<&'a Tensor>,
    pub gate: Option<&'a Tensor>,
    pub hidden: &'a mut Tensor,
    pub normalized: &'a mut Tensor,
    pub kind: NormKind<'a>,
    pub affine: Affine<'a>,
    pub eps: f32,
    pub post: Option<QuantOut>,
}

impl<'a> FusedAddNormArgs<'a> {
    pub fn new(
        input: &'a Tensor,
        residual: &'a Tensor,
        hidden: &'a mut Tensor,
        normalized: &'a mut Tensor,
        kind: NormKind<'a>,
        affine: Affine<'a>,
        eps: f32,
    ) -> Self {
        Self {
            input,
            residual,
            bias: None,
            gate: None,
            hidden,
            normalized,
            kind,
            affine,
            eps,
            post: None,
        }
    }
}

/// Residual combine with no normalization. `bias` and `gate` are mutually
/// exclusive; see [`FusedAddNormArgs`] for the gate layout.
pub struct ResidualAddArgs<'a> {
    pub input: &'a Tensor,
    pub residual: &'a Tensor,
    pub bias: Option<&'a Tensor>,
    pub gate: Option<&'a Tensor>,
    pub order: BiasOrder,
    pub out: &'a mut Tensor,
}

impl<'a> ResidualAddArgs<'a> {
    pub fn new(
        input: &'a Tensor,
        bias: Option<&'a Tensor>,
        residual: &'a Tensor,
        out: &'a mut Tensor,
    ) -> Self {
        Self {
            input,
            residual,
            bias,
            gate: None,
            order: BiasOrder::SingleRound,
            out,
        }
    }

    /// Gated combine: `out = residual + input * gate[2 * cols + c]`.
    pub fn gated(
        input: &'a Tensor,
        residual: &'a Tensor,
        gate: &'a Tensor,
        out: &'a mut Tensor,
    ) -> Self {
        Self {
            input,
            residual,
            bias: None,
            gate: Some(gate),
            order: BiasOrder::SingleRound,
            out,
        }
    }
}

/// The Spec stage fields and the tensors a `kind`/`affine` pair binds.
struct Stage<'a> {
    norm_kind: u32,
    affine_kind: u32,
    weight: Option<&'a Tensor>,
    norm_bias: Option<&'a Tensor>,
    norm_style: Option<&'a Tensor>,
}

fn stage<'a>(kind: &NormKind<'a>, affine: &Affine<'a>) -> Stage<'a> {
    let (norm_kind, norm_bias) = match kind {
        NormKind::Rms => (abi::NORM_KIND_RMS, None),
        NormKind::Layer { bias } => (abi::NORM_KIND_LAYER, Some(*bias)),
    };
    let (affine_kind, weight, norm_style) = match affine {
        Affine::Weight { weight, offset } => (
            if *offset {
                abi::AFFINE_WEIGHT_OFFSET
            } else {
                abi::AFFINE_WEIGHT
            },
            Some(*weight),
            None,
        ),
        Affine::AdaScaleShift { style } => (abi::AFFINE_ADA_SCALE_SHIFT, None, Some(*style)),
    };
    Stage {
        norm_kind,
        affine_kind,
        weight,
        norm_bias,
        norm_style,
    }
}

fn quant_scale(post: &Option<QuantOut>) -> f32 {
    match post {
        Some(QuantOut::E4m3 { scale }) => *scale,
        None => 1.0,
    }
}

fn execute(ctx: &CudaContext, args: RawArgs<'_>) -> Result<()> {
    launch::execute(ctx, normalize(ctx, args)?)
}

/// Row-wise normalization of a `[rows, cols]` activation.
pub fn norm(ctx: &CudaContext, args: NormArgs<'_>) -> Result<()> {
    let stage = stage(&args.kind, &args.affine);
    execute(
        ctx,
        RawArgs {
            fields: Fields {
                semantic: abi::SEMANTIC_NORM,
                norm_kind: stage.norm_kind,
                affine_kind: stage.affine_kind,
                has_gate: false,
                bias_order: abi::BIAS_ORDER_SINGLE_ROUND,
            },
            input: args.input,
            bias: None,
            residual: None,
            weight: stage.weight,
            norm_bias: stage.norm_bias,
            norm_style: stage.norm_style,
            gate_style: None,
            hidden: None,
            normalized: Some(args.out),
            eps: args.eps,
            output_scale: quant_scale(&args.post),
        },
    )
}

/// Residual combine plus normalization in one launch, writing the combined
/// row and the normalized row.
pub fn fused_add_norm(ctx: &CudaContext, args: FusedAddNormArgs<'_>) -> Result<()> {
    if args.bias.is_some() && args.gate.is_some() {
        return Err(invalid(
            "a fused add takes either a bias or a gate, not both",
        ));
    }
    let stage = stage(&args.kind, &args.affine);
    execute(
        ctx,
        RawArgs {
            fields: Fields {
                semantic: abi::SEMANTIC_FUSED_ADD_NORM,
                norm_kind: stage.norm_kind,
                affine_kind: stage.affine_kind,
                has_gate: args.gate.is_some(),
                bias_order: abi::BIAS_ORDER_SINGLE_ROUND,
            },
            input: args.input,
            bias: args.bias,
            residual: Some(args.residual),
            weight: stage.weight,
            norm_bias: stage.norm_bias,
            norm_style: stage.norm_style,
            gate_style: args.gate,
            hidden: Some(args.hidden),
            normalized: Some(args.normalized),
            eps: args.eps,
            output_scale: quant_scale(&args.post),
        },
    )
}

/// Bias or gate combine plus residual, with no normalization.
pub fn residual_add(ctx: &CudaContext, args: ResidualAddArgs<'_>) -> Result<()> {
    if args.bias.is_some() && args.gate.is_some() {
        return Err(invalid(
            "a residual add takes either a bias or a gate, not both",
        ));
    }
    let bias_order = match args.order {
        BiasOrder::SingleRound => abi::BIAS_ORDER_SINGLE_ROUND,
        BiasOrder::RoundAfterBias => abi::BIAS_ORDER_ROUND_AFTER_BIAS,
    };
    execute(
        ctx,
        RawArgs {
            fields: Fields {
                semantic: abi::SEMANTIC_RESIDUAL_ADD,
                norm_kind: abi::NORM_KIND_NONE,
                affine_kind: abi::AFFINE_NONE,
                has_gate: args.gate.is_some(),
                bias_order,
            },
            input: args.input,
            bias: args.bias,
            residual: Some(args.residual),
            weight: None,
            norm_bias: None,
            norm_style: None,
            gate_style: args.gate,
            hidden: Some(args.out),
            normalized: None,
            // The combine forms take no epsilon; the ABI validates it as a
            // positive float whether or not a reduction consumes it.
            eps: 1e-6,
            output_scale: 1.0,
        },
    )
}
