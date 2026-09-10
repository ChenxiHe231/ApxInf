//! Persistent CUDA graph workspace and deterministic sub-allocation.

use std::cell::Cell;

use apxinf_core::{Error, Result};

use crate::buffer::CudaBuffer;
use crate::context::CudaContext;

const WORKSPACE_ALIGNMENT: usize = 256;

/// Persistent device arena used by a fixed-shape CUDA graph.
pub struct GraphWorkspace {
    gemm_instances: std::cell::RefCell<
        std::collections::HashMap<String, std::rc::Rc<crate::ops::execution::SharedExecution>>,
    >,
    storage: CudaBuffer,
    offset: Cell<usize>,
}

impl GraphWorkspace {
    pub fn new(capacity_bytes: usize, device: usize) -> Result<Self> {
        if capacity_bytes == 0 {
            return Err(Error::Other(
                "static inference workspace capacity must be non-zero".into(),
            ));
        }
        Ok(Self {
            gemm_instances: Default::default(),
            storage: CudaBuffer::alloc(capacity_bytes, device).map_err(Error::Cuda)?,
            offset: Cell::new(0),
        })
    }

    pub fn capacity(&self) -> usize {
        self.storage.len()
    }

    pub fn used(&self) -> usize {
        self.offset.get()
    }

    fn reset(&self) {
        self.offset.set(0);
    }

    fn allocate(&self, bytes: usize, device: usize) -> Result<CudaBuffer> {
        if device != self.storage.device() {
            return Err(Error::Other(format!(
                "static inference workspace is on CUDA {}, but operation targets CUDA {device}",
                self.storage.device()
            )));
        }
        let start = self
            .offset
            .get()
            .checked_add(WORKSPACE_ALIGNMENT - 1)
            .ok_or_else(|| Error::Other("static inference workspace offset overflow".into()))?
            & !(WORKSPACE_ALIGNMENT - 1);
        let end = start
            .checked_add(bytes)
            .ok_or_else(|| Error::Other("static inference workspace size overflow".into()))?;
        if end > self.storage.len() {
            return Err(Error::Other(format!(
                "static inference workspace exhausted: need {end} bytes, capacity is {} bytes",
                self.storage.len()
            )));
        }
        self.offset.set(end);
        self.storage.view(start, bytes).map_err(Error::Cuda)
    }
}

thread_local! {
    static ACTIVE_WORKSPACE: Cell<*const GraphWorkspace> = const { Cell::new(std::ptr::null()) };
    static PREPARING: Cell<bool> = const { Cell::new(false) };
    static CAPTURE_ACTIVE: Cell<bool> = const { Cell::new(false) };
    static CAPTURED_GEMM_INSTANCES: std::cell::RefCell<Vec<std::rc::Rc<crate::ops::execution::SharedExecution>>> = const { std::cell::RefCell::new(Vec::new()) };
}

struct ActiveWorkspaceGuard {
    workspace: *const GraphWorkspace,
    preparing: bool,
}

impl Drop for ActiveWorkspaceGuard {
    fn drop(&mut self) {
        ACTIVE_WORKSPACE.with(|active| active.set(self.workspace));
        PREPARING.with(|preparing| preparing.set(self.preparing));
    }
}

fn with_workspace_phase<T>(
    workspace: &GraphWorkspace,
    prepare: bool,
    operation: impl FnOnce() -> Result<T>,
) -> Result<T> {
    if prepare {
        workspace.gemm_instances.borrow_mut().clear();
    }
    workspace.reset();
    ACTIVE_WORKSPACE.with(|active| {
        if !active.get().is_null() {
            return Err(Error::Other(
                "nested static inference workspaces are not supported".into(),
            ));
        }
        let previous = active.replace(workspace as *const _);
        let previous_preparing = PREPARING.with(|preparing| preparing.replace(prepare));
        let _guard = ActiveWorkspaceGuard {
            workspace: previous,
            preparing: previous_preparing,
        };
        operation()
    })
}

pub(crate) fn prepare_with_workspace<T>(
    workspace: &GraphWorkspace,
    operation: impl FnOnce() -> Result<T>,
) -> Result<T> {
    with_workspace_phase(workspace, true, operation)
}

pub(crate) fn with_workspace<T>(
    workspace: &GraphWorkspace,
    operation: impl FnOnce() -> Result<T>,
) -> Result<T> {
    with_workspace_phase(workspace, false, operation)
}

/// Native execution resources may be installed only before capture or when an
/// operation is executed without a graph workspace.
pub(crate) fn may_prepare_native_resources() -> bool {
    PREPARING.with(Cell::get) || ACTIVE_WORKSPACE.with(|active| active.get().is_null())
}

/// Whether execution is the synthetic eager traversal used only to prepare a
/// graph workspace. Autotuning must wait for a real request instead of using
/// these placeholder inputs.
pub(crate) fn is_preparing_workspace() -> bool {
    PREPARING.with(Cell::get)
}

pub(crate) fn begin_capture_retention() {
    CAPTURED_GEMM_INSTANCES.with(|instances| instances.borrow_mut().clear());
    CAPTURE_ACTIVE.with(|active| active.set(true));
}

pub(crate) fn end_capture_retention() -> Vec<std::rc::Rc<crate::ops::execution::SharedExecution>> {
    CAPTURE_ACTIVE.with(|active| active.set(false));
    CAPTURED_GEMM_INSTANCES.with(|instances| std::mem::take(&mut *instances.borrow_mut()))
}

pub(crate) fn is_capturing() -> bool {
    CAPTURE_ACTIVE.with(Cell::get)
}

pub(crate) fn output_buffer(ctx: &CudaContext, bytes: usize) -> Result<CudaBuffer> {
    ACTIVE_WORKSPACE.with(|active| {
        let workspace = active.get();
        if workspace.is_null() {
            CudaBuffer::alloc_zeros(bytes, ctx.device_id()).map_err(Error::Cuda)
        } else {
            unsafe { &*workspace }.allocate(bytes, ctx.device_id())
        }
    })
}

pub(crate) fn has_active_workspace() -> bool {
    ACTIVE_WORKSPACE.with(|slot| !slot.get().is_null())
}

pub(crate) fn lookup_gemm_instance(
    key: &str,
) -> Option<std::rc::Rc<crate::ops::execution::SharedExecution>> {
    ACTIVE_WORKSPACE.with(|slot| {
        let workspace = slot.get();
        if workspace.is_null() {
            None
        } else {
            let instance = unsafe { &*workspace }
                .gemm_instances
                .borrow()
                .get(key)
                .cloned();
            if let Some(instance) = &instance {
                if CAPTURE_ACTIVE.with(Cell::get) {
                    CAPTURED_GEMM_INSTANCES.with(|instances| {
                        let mut instances = instances.borrow_mut();
                        if !instances
                            .iter()
                            .any(|stored| std::rc::Rc::ptr_eq(stored, instance))
                        {
                            instances.push(std::rc::Rc::clone(instance));
                        }
                    });
                }
            }
            instance
        }
    })
}

pub(crate) fn store_gemm_instance(
    key: String,
    instance: std::rc::Rc<crate::ops::execution::SharedExecution>,
) {
    ACTIVE_WORKSPACE.with(|slot| {
        let workspace = slot.get();
        if !workspace.is_null() {
            unsafe { &*workspace }
                .gemm_instances
                .borrow_mut()
                .insert(key, instance);
        }
    });
}
