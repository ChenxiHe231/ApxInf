//! GEMM-only pilot: semantic APIs with selection and native state behind L1.

mod gemm;

// Keep these crate-private aliases while graph/workspace and unit tests still
// refer to the GEMM implementation through `crate::ops`.
#[cfg(test)]
pub(crate) use gemm::contracts;
pub(crate) use gemm::gemm_execution as execution;

pub use crate::workspace::GraphWorkspace;
pub use gemm::{
    gemm, gemm_bias, gemm_bias_gelu, gemm_geglu, GemmArgs, GemmBiasArgs, GemmBiasGeluArgs,
    GemmGegluArgs, GemmPolicy, GemmQuantization, WeightVersion,
};

pub fn prepare_with_workspace<T>(
    workspace: &GraphWorkspace,
    operation: impl FnOnce() -> apxinf_core::Result<T>,
) -> apxinf_core::Result<T> {
    crate::workspace::prepare_with_workspace(workspace, operation)
}

pub fn with_workspace<T>(
    workspace: &GraphWorkspace,
    operation: impl FnOnce() -> apxinf_core::Result<T>,
) -> apxinf_core::Result<T> {
    crate::workspace::with_workspace(workspace, operation)
}
#[cfg(test)]
mod tests;
#[cfg(test)]
mod torch_l3_fixtures;
