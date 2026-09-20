//! Compile-time backend seam for PI0.5 Blocks.
//!
//! Block code depends on this model-local alias and the model-neutral
//! kernel contract. Adding another accelerator backend changes this seam,
//! not the layer topology.

pub(crate) use crate::accelerator::cuda::{
    kernels, transfers, Context, DeviceBuffer, ExecutionSession, RuntimeBackend,
};

/// Memory layout of a fixed-shape batch of RGB `uint8` images.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ImageLayout {
    Nhwc,
    Nchw,
}

impl std::fmt::Display for ImageLayout {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Nhwc => formatter.write_str("nhwc"),
            Self::Nchw => formatter.write_str("nchw"),
        }
    }
}
