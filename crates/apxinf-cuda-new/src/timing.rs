//! CUDA-event timing on the backend's execution stream.

use std::sync::Arc;

use apxinf_core::{Error, Result};

use crate::{CudaContext, CudaStream};

/// Reusable pair of CUDA events for device-side latency measurements.
pub struct CudaEventTimer {
    start: crate::ffi::cudaEvent_t,
    end: crate::ffi::cudaEvent_t,
    stream: Arc<CudaStream>,
}

impl CudaEventTimer {
    pub fn new(ctx: &CudaContext) -> Result<Self> {
        ctx.stream().set_current_device().map_err(Error::Cuda)?;
        let mut start = std::ptr::null_mut();
        let mut end = std::ptr::null_mut();
        unsafe {
            crate::ffi::check_cuda(crate::ffi::cudaEventCreate(&mut start)).map_err(Error::Cuda)?;
            if let Err(error) = crate::ffi::check_cuda(crate::ffi::cudaEventCreate(&mut end)) {
                let _ = crate::ffi::cudaEventDestroy(start);
                return Err(Error::Cuda(error));
            }
        }
        Ok(Self {
            start,
            end,
            stream: ctx.shared_stream(),
        })
    }

    /// Measure asynchronous work enqueued by `operation` on this CUDA stream.
    pub fn measure<T>(&self, operation: impl FnOnce() -> Result<T>) -> Result<(T, f64)> {
        self.stream.set_current_device().map_err(Error::Cuda)?;
        unsafe {
            crate::ffi::check_cuda(crate::ffi::cudaEventRecord(
                self.start,
                self.stream.handle(),
            ))
            .map_err(Error::Cuda)?;
        }
        let value = operation()?;
        let mut milliseconds = 0.0f32;
        unsafe {
            crate::ffi::check_cuda(crate::ffi::cudaEventRecord(
                self.end,
                self.stream.handle(),
            ))
            .map_err(Error::Cuda)?;
            crate::ffi::check_cuda(crate::ffi::cudaEventSynchronize(self.end))
                .map_err(Error::Cuda)?;
            crate::ffi::check_cuda(crate::ffi::cudaEventElapsedTime(
                &mut milliseconds,
                self.start,
                self.end,
            ))
            .map_err(Error::Cuda)?;
        }
        Ok((value, f64::from(milliseconds)))
    }
}

impl Drop for CudaEventTimer {
    fn drop(&mut self) {
        let _ = self.stream.set_current_device();
        unsafe {
            let _ = crate::ffi::cudaEventDestroy(self.start);
            let _ = crate::ffi::cudaEventDestroy(self.end);
        }
    }
}
