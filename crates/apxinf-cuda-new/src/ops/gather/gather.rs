use apxinf_core::Result;

use super::contracts::{normalize, GatherArgs};
use super::execution;
use crate::CudaContext;

/// Runs one gather / layout semantic. The semantic selects which of the
/// optional bindings participate; see [`GatherArgs`].
pub fn gather(ctx: &CudaContext, args: GatherArgs<'_>) -> Result<()> {
    execution::execute(ctx, normalize(ctx, args)?)
}
