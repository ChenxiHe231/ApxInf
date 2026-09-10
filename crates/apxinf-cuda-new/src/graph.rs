//! Private CUDA Graph ABI and handle lifetime boundary.

use crate::context::CudaContext;
use crate::ffi;
use std::rc::Rc;
use std::sync::Arc;

#[derive(Clone, Copy)]
pub(crate) enum CaptureMode {
    ThreadLocal,
}

pub struct CapturedGraph {
    exec: ffi::cudaGraphExec_t,
    graph: ffi::cudaGraph_t,
    stream: Arc<crate::CudaStream>,
    _gemm_instances: Vec<Rc<crate::ops::execution::SharedExecution>>,
}

impl CapturedGraph {
    pub fn replay(&self) -> Result<(), String> {
        self.stream.with_current_device(|| unsafe {
            ffi::check_cuda(ffi::cudaGraphLaunch(self.exec, self.stream.handle()))
        })
    }
}

impl Drop for CapturedGraph {
    fn drop(&mut self) {
        let _ = self.stream.with_current_device(|| unsafe {
            ffi::check_cuda(ffi::cudaGraphExecDestroy(self.exec))?;
            ffi::check_cuda(ffi::cudaGraphDestroy(self.graph))
        });
    }
}

pub(crate) fn begin(ctx: &CudaContext, mode: CaptureMode) -> Result<(), String> {
    let mode = match mode {
        CaptureMode::ThreadLocal => ffi::cudaStreamCaptureMode::cudaStreamCaptureModeThreadLocal,
    };
    // Keep this device current until `end`; switching devices during capture
    // can invalidate a thread-local capture.
    ctx.stream().set_current_device()?;
    unsafe {
        ffi::check_cuda(ffi::cudaStreamBeginCapture(ctx.stream().handle(), mode))?;
    }
    crate::workspace::begin_capture_retention();
    Ok(())
}

pub(crate) fn end(ctx: &CudaContext) -> Result<CapturedGraph, String> {
    ctx.stream().set_current_device()?;
    let stream = ctx.stream().handle();
    let mut graph: ffi::cudaGraph_t = std::ptr::null_mut();
    let end_status = unsafe { ffi::check_cuda(ffi::cudaStreamEndCapture(stream, &mut graph)) };
    let retained = crate::workspace::end_capture_retention();
    end_status?;
    let mut exec: ffi::cudaGraphExec_t = std::ptr::null_mut();
    let status = unsafe {
        ffi::cudaGraphInstantiate(
            &mut exec,
            graph,
            std::ptr::null_mut(),
            std::ptr::null_mut(),
            0,
        )
    };
    if let Err(error) = ffi::check_cuda(status) {
        unsafe {
            let _ = ffi::cudaGraphDestroy(graph);
        }
        return Err(error);
    }
    Ok(CapturedGraph {
        exec,
        graph,
        stream: ctx.shared_stream(),
        _gemm_instances: retained,
    })
}
