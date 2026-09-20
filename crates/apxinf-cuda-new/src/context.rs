//! CUDA device and stream context.

use std::sync::Arc;

use apxinf_core::{DType, Error, Result as CoreResult, Shape, Tensor};

use crate::ffi;
use crate::stream::CudaStream;
use crate::CudaDeviceCaps;

/// One `cudaMemGetInfo` snapshot for the context's device.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct CudaMemoryInfo {
    pub free_bytes: usize,
    pub total_bytes: usize,
}

impl CudaMemoryInfo {
    pub fn used_bytes(self) -> usize {
        self.total_bytes.saturating_sub(self.free_bytes)
    }
}

/// Query device-wide CUDA memory without requiring ownership of a context.
///
/// This is intended for monitoring threads. The CUDA current-device binding is
/// thread-local, so the function establishes it before every query.
pub fn device_memory_info(device_id: usize) -> Result<CudaMemoryInfo, String> {
    let device = i32::try_from(device_id)
        .map_err(|_| format!("CUDA device id {device_id} does not fit in i32"))?;
    let mut free_bytes = 0usize;
    let mut total_bytes = 0usize;
    unsafe {
        ffi::check_cuda(ffi::cudaSetDevice(device))?;
        ffi::check_cuda(ffi::cudaMemGetInfo(&mut free_bytes, &mut total_bytes))?;
    }
    if free_bytes > total_bytes {
        return Err(format!(
            "CUDA returned invalid memory info: free={free_bytes}, total={total_bytes}"
        ));
    }
    Ok(CudaMemoryInfo {
        free_bytes,
        total_bytes,
    })
}

/// Owns the stream and opaque native runtime used by GEMM executions.
pub struct CudaContext {
    device_id: usize,
    stream: Arc<CudaStream>,
    runtime: crate::ffi::abi::types::Runtime,
    caps: CudaDeviceCaps,
}

impl CudaContext {
    /// Create a context for the specified CUDA device.
    pub fn new(device_id: usize) -> Result<Self, String> {
        let device = i32::try_from(device_id)
            .map_err(|_| format!("CUDA device id {device_id} does not fit in i32"))?;
        unsafe {
            ffi::check_cuda(ffi::cudaSetDevice(device))?;
        }

        let stream = Arc::new(CudaStream::new_on(device_id)?);
        let mut runtime = std::ptr::null_mut();
        unsafe {
            crate::ffi::abi::status::check(crate::ffi::abi::runtime::apxinf_runtime_create(
                device,
                &mut runtime,
            ))
            .map_err(|error| error.to_string())?;
        }
        let caps = match CudaDeviceCaps::query(runtime) {
            Ok(caps) => caps,
            Err(error) => {
                unsafe { crate::ffi::abi::runtime::apxinf_runtime_destroy(runtime) };
                return Err(error);
            }
        };

        Ok(Self {
            device_id,
            stream,
            runtime,
            caps,
        })
    }

    pub fn device_id(&self) -> usize {
        self.device_id
    }
    pub fn stream(&self) -> &CudaStream {
        &self.stream
    }
    pub fn caps(&self) -> &CudaDeviceCaps {
        &self.caps
    }
    pub(crate) fn shared_stream(&self) -> Arc<CudaStream> {
        Arc::clone(&self.stream)
    }
    pub(crate) fn runtime(&self) -> crate::ffi::abi::types::Runtime {
        self.runtime
    }

    pub fn synchronize(&self) -> Result<(), String> {
        self.stream.synchronize()
    }

    /// Return a point-in-time device-memory snapshot for this CUDA device.
    ///
    /// CUDA reports device-wide free and total memory. Callers that need an
    /// attributable benchmark delta should retain a baseline snapshot and
    /// report both the absolute high-water observation and its baseline delta.
    pub fn memory_info(&self) -> Result<CudaMemoryInfo, String> {
        device_memory_info(self.device_id)
    }

    /// Allocate caller-owned output storage for an L3 semantic operation.
    ///
    /// Inside an [`crate::ExecutionSession`] this sub-allocates from the
    /// deterministic graph arena. Outside a session it creates an ordinary
    /// zero-initialized device allocation. Operator selection remains in L3;
    /// this method only provides stable runtime-owned storage for its output
    /// binding.
    pub fn allocate_output(&self, shape: Shape, dtype: DType) -> CoreResult<Tensor> {
        let bytes = shape
            .numel()
            .checked_mul(dtype.size_in_bytes())
            .ok_or_else(|| Error::Other("CUDA output size overflow".into()))?;
        let buffer = crate::workspace::output_buffer(self, bytes)?;
        Ok(buffer.into_tensor(shape, dtype))
    }
}

impl Drop for CudaContext {
    fn drop(&mut self) {
        unsafe { crate::ffi::abi::runtime::apxinf_runtime_destroy(self.runtime) }
    }
}
