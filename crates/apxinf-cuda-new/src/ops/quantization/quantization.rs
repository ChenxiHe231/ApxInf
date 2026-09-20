use apxinf_core::Result;

use super::contracts::{normalize, QuantizationArgs};
use super::execution;
use crate::CudaContext;

/// Execute a validated quantization or representation-conversion semantic.
pub fn quantization(ctx: &CudaContext, args: QuantizationArgs<'_>) -> Result<()> {
    execution::execute(ctx, normalize(ctx, args)?)
}
