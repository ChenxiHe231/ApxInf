//! GEMM-only pilot: semantic APIs with selection and native state behind L1.

mod contracts;
pub(crate) mod execution;
mod gemm;
mod gemm_bias;
mod gemm_geglu;
mod gemm_gelu;

pub use crate::workspace::GraphWorkspace;
pub use contracts::{GemmArgs, GemmPolicy, GemmQuantization};
pub use gemm::gemm;
pub use gemm_bias::{gemm_bias, GemmBiasArgs};
pub use gemm_geglu::{gemm_geglu, GemmGegluArgs};
pub use gemm_gelu::{gemm_bias_gelu, GemmBiasGeluArgs};

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
