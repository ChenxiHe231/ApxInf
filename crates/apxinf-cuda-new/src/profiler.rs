//! CUDA profiler control used to delimit an opt-in profiling interval.

/// Start CUDA profiler collection for the current process.
pub fn start() -> Result<(), String> {
    unsafe { crate::ffi::check_cuda(crate::ffi::cudaProfilerStart()) }
}

/// Stop CUDA profiler collection for the current process.
pub fn stop() -> Result<(), String> {
    unsafe { crate::ffi::check_cuda(crate::ffi::cudaProfilerStop()) }
}
