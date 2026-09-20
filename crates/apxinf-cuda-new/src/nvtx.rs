//! NVTX-compatible range API.
//!
//! cuda-new keeps the public profiling seam available even when the optional
//! NVTX runtime is not linked. The no-op guard lets model code remain
//! backend-independent; native marker emission can be enabled separately.

pub struct Range;

impl Range {
    pub fn new(_name: &str) -> Self {
        Self
    }
}

pub fn range(name: &str) -> Range {
    Range::new(name)
}
