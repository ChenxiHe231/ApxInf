//! Optional CUDA profiler ranges for an isolated diagnostic invocation.
//! Default measurement runs make no profiler API calls.
#[link(name = "cudart")]
extern "C" {
    fn cudaProfilerStart() -> i32;
    fn cudaProfilerStop() -> i32;
}

pub struct CudaProfileRange {
    active: bool,
}
impl CudaProfileRange {
    pub fn start(enabled: bool) -> Result<Self, Box<dyn std::error::Error>> {
        if enabled {
            let status = unsafe { cudaProfilerStart() };
            if status != 0 {
                return Err(format!("cudaProfilerStart failed: {status}").into());
            }
        }
        Ok(Self { active: enabled })
    }
    pub fn finish(mut self) -> Result<(), Box<dyn std::error::Error>> {
        if self.active {
            self.active = false;
            let status = unsafe { cudaProfilerStop() };
            if status != 0 {
                return Err(format!("cudaProfilerStop failed: {status}").into());
            }
        }
        Ok(())
    }
}
impl Drop for CudaProfileRange {
    fn drop(&mut self) {
        if self.active {
            unsafe {
                cudaProfilerStop();
            }
        }
    }
}
