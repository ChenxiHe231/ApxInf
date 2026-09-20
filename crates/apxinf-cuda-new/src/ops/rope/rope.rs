use apxinf_core::Result;

use super::contracts::{normalize, RopeArgs};
use super::execution;
use crate::CudaContext;

/// Splits a packed QKV projection, optionally applying rotary embedding.
pub fn rope(ctx: &CudaContext, args: RopeArgs<'_>) -> Result<()> {
    execution::execute(ctx, normalize(ctx, args)?)
}
