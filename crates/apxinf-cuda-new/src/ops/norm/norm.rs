use apxinf_core::Result;

use super::contracts::{normalize, NormArgs};
use super::execution;
use crate::CudaContext;

/// Runs one row-wise normalization semantic. The semantic selects which of the
/// optional bindings participate; see [`NormArgs`].
pub fn norm(ctx: &CudaContext, args: NormArgs<'_>) -> Result<()> {
    execution::execute(ctx, normalize(ctx, args)?)
}
