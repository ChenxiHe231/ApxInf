#[cfg(test)]
use std::ffi::CStr;
use std::ffi::CString;
use std::marker::PhantomData;
use std::rc::Rc;
use std::sync::Arc;

use apxinf_core::{Error, Result};

use super::contracts::Normalized;
use crate::ffi::abi::{gather as abi, status};
use crate::{CudaBuffer, CudaContext};

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
struct ExecutionKey {
    spec: abi::Spec,
    device: usize,
    input: usize,
    ids: usize,
    bias: usize,
    position: usize,
    output: usize,
    stream: usize,
    workspace_limit: u64,
    graph_safe: bool,
    deterministic: bool,
}

impl ExecutionKey {
    fn new(ctx: &CudaContext, normalized: &Normalized) -> Self {
        Self {
            spec: normalized.spec,
            device: ctx.device_id(),
            input: normalized.bindings.input as usize,
            ids: normalized.bindings.ids as usize,
            bias: normalized.bindings.bias as usize,
            position: normalized.bindings.position as usize,
            output: normalized.bindings.output as usize,
            stream: normalized.bindings.stream as usize,
            workspace_limit: normalized.policy.workspace_limit as u64,
            graph_safe: normalized.policy.graph_safe,
            deterministic: normalized.policy.deterministic,
        }
    }
}

pub(crate) struct Execution {
    raw: abi::Execution,
    stream: Arc<crate::CudaStream>,
    _storage: Vec<CudaBuffer>,
    #[cfg(test)]
    summary: String,
    _not_send: PhantomData<Rc<()>>,
}

impl Drop for Execution {
    fn drop(&mut self) {
        let _ = self.stream.synchronize();
        unsafe { abi::apxinf_gather_destroy(self.raw) }
    }
}

pub(crate) fn prepare(ctx: &CudaContext, normalized: Normalized) -> Result<Rc<Execution>> {
    let key = ExecutionKey::new(ctx, &normalized);
    if let Some(execution) = crate::workspace::lookup_execution(&key) {
        crate::workspace::use_execution(&execution)?;
        return Ok(execution);
    }
    if !crate::workspace::may_prepare_native_resources() {
        return Err(Error::Other(
            "Gather execution cache miss during capture; prepare the same bindings first".into(),
        ));
    }

    let cache = normalized
        .policy
        .cache_dir
        .as_ref()
        .map(|path| CString::new(path.as_str()))
        .transpose()
        .map_err(|_| Error::Other("cache path contains NUL".into()))?;
    let policy = abi::Policy {
        workspace_limit: normalized.policy.workspace_limit as u64,
        online_tune: normalized.policy.online_tune as u32,
        allow_fallback: normalized.policy.allow_fallback as u32,
        graph_safe: normalized.policy.graph_safe as u32,
        deterministic: normalized.policy.deterministic as u32,
        cache_dir: cache
            .as_ref()
            .map_or(std::ptr::null(), |path| path.as_ptr()),
    };
    let mut raw = std::ptr::null_mut();
    unsafe {
        status::check(abi::apxinf_gather_prepare(
            ctx.runtime(),
            &normalized.spec,
            &policy,
            &normalized.bindings,
            &mut raw,
        ))?;
    }
    if raw.is_null() {
        return Err(Error::Other(
            "native Gather prepare returned a null execution".into(),
        ));
    }
    #[cfg(test)]
    let summary = unsafe { CStr::from_ptr(abi::apxinf_gather_summary(raw)) }
        .to_string_lossy()
        .into_owned();
    let execution = Rc::new(Execution {
        raw,
        stream: ctx.shared_stream(),
        _storage: normalized.storage,
        #[cfg(test)]
        summary,
        _not_send: PhantomData,
    });
    crate::workspace::store_execution(key, Rc::clone(&execution));
    crate::workspace::use_execution(&execution)?;
    Ok(execution)
}

pub(crate) fn execute(ctx: &CudaContext, normalized: Normalized) -> Result<()> {
    let execution = prepare(ctx, normalized)?;
    crate::workspace::validate_capture_target(
        execution.stream.device(),
        execution.stream.handle() as usize,
    )?;
    execution.stream.set_current_device().map_err(Error::Cuda)?;
    unsafe { status::check(abi::apxinf_gather_enqueue(execution.raw))? };
    if crate::workspace::is_capturing()
        || (crate::workspace::has_active_session() && !crate::workspace::is_preparing_session())
    {
        Ok(())
    } else {
        ctx.synchronize().map_err(Error::Cuda)
    }
}

#[cfg(test)]
impl Execution {
    pub(crate) fn summary(&self) -> &str {
        &self.summary
    }
}
