use std::ffi::{CStr, CString};
use std::marker::PhantomData;
use std::rc::Rc;
use std::sync::Arc;

use apxinf_core::{Error, Result};

use super::contracts::{invalid, Normalized};
use crate::ffi::abi::types::Runtime;
use crate::ffi::abi::{gemm as abi, status};
use crate::{CudaBuffer, CudaContext};

#[derive(Clone, Copy)]
pub(crate) struct PlanApi {
    pub name: &'static str,
    plan_create: unsafe extern "C" fn(
        Runtime,
        *const abi::Spec,
        *const abi::Policy,
        *const abi::TuningBindings,
        *mut abi::Plan,
    ) -> i32,
    instance_create:
        unsafe extern "C" fn(abi::Plan, *const abi::Bindings, *mut abi::Instance) -> i32,
    enqueue: unsafe extern "C" fn(abi::Instance) -> i32,
    instance_destroy: unsafe extern "C" fn(abi::Instance),
    plan_destroy: unsafe extern "C" fn(abi::Plan),
    plan_summary: unsafe extern "C" fn(abi::Plan) -> *const std::ffi::c_char,
}

macro_rules! plan_api {
    ($constructor:ident, $name:literal, $module:ident, $create:ident, $instance:ident, $enqueue:ident, $destroy_instance:ident, $destroy_plan:ident, $summary:ident) => {
        pub(crate) fn $constructor() -> Self {
            Self {
                name: $name,
                plan_create: crate::ffi::abi::$module::$create,
                instance_create: crate::ffi::abi::$module::$instance,
                enqueue: crate::ffi::abi::$module::$enqueue,
                instance_destroy: crate::ffi::abi::$module::$destroy_instance,
                plan_destroy: crate::ffi::abi::$module::$destroy_plan,
                plan_summary: crate::ffi::abi::$module::$summary,
            }
        }
    };
}

impl PlanApi {
    plan_api!(
        gemm,
        "gemm",
        gemm,
        apxinf_gemm_plan_create,
        apxinf_gemm_instance_create,
        apxinf_gemm_enqueue,
        apxinf_gemm_instance_destroy,
        apxinf_gemm_plan_destroy,
        apxinf_gemm_plan_summary
    );
    plan_api!(
        gemm_bias_gelu,
        "gemm_bias_gelu",
        gemm_gelu,
        apxinf_gemm_bias_gelu_plan_create,
        apxinf_gemm_bias_gelu_instance_create,
        apxinf_gemm_bias_gelu_enqueue,
        apxinf_gemm_bias_gelu_instance_destroy,
        apxinf_gemm_bias_gelu_plan_destroy,
        apxinf_gemm_bias_gelu_plan_summary
    );
    plan_api!(
        gemm_bias,
        "gemm_bias",
        gemm_bias,
        apxinf_gemm_bias_plan_create,
        apxinf_gemm_bias_instance_create,
        apxinf_gemm_bias_enqueue,
        apxinf_gemm_bias_instance_destroy,
        apxinf_gemm_bias_plan_destroy,
        apxinf_gemm_bias_plan_summary
    );
    plan_api!(
        gemm_geglu,
        "gemm_geglu",
        gemm_geglu,
        apxinf_gemm_geglu_plan_create,
        apxinf_gemm_geglu_instance_create,
        apxinf_gemm_geglu_enqueue,
        apxinf_gemm_geglu_instance_destroy,
        apxinf_gemm_geglu_plan_destroy,
        apxinf_gemm_geglu_plan_summary
    );
}

struct NativePlan {
    raw: abi::Plan,
    destroy: unsafe extern "C" fn(abi::Plan),
}

impl Drop for NativePlan {
    fn drop(&mut self) {
        unsafe { (self.destroy)(self.raw) }
    }
}

/// Owns the native instance and every device allocation referenced by it.
/// The native object is deliberately thread-confined.
pub(crate) struct SharedExecution {
    raw: abi::Instance,
    _plan: NativePlan,
    stream: Arc<crate::CudaStream>,
    _storage: Vec<CudaBuffer>,
    summary: String,
    enqueue: unsafe extern "C" fn(abi::Instance) -> i32,
    destroy_instance: unsafe extern "C" fn(abi::Instance),
    _not_send: PhantomData<Rc<()>>,
}

impl Drop for SharedExecution {
    fn drop(&mut self) {
        let _ = self.stream.synchronize();
        unsafe { (self.destroy_instance)(self.raw) }
    }
}

/// A fully prepared GEMM execution bound to fixed tensor addresses.
///
/// Preparation selects the candidate, creates provider resources and performs
/// any required warmup. Executors should retain this object and call
/// [`enqueue`](Self::enqueue) on their hot path. Enqueue is asynchronous; the
/// executor owns synchronization or CUDA Graph replay.
pub struct PreparedExecution {
    exec: Rc<SharedExecution>,
}

impl PreparedExecution {
    pub fn summary(&self) -> &str {
        &self.exec.summary
    }

    pub fn enqueue(&mut self) -> Result<()> {
        crate::workspace::validate_capture_target(
            self.exec.stream.device(),
            self.exec.stream.handle() as usize,
        )?;
        self.exec.stream.set_current_device().map_err(Error::Cuda)?;
        crate::workspace::retain_gemm_instance(&self.exec);
        unsafe { status::check((self.exec.enqueue)(self.exec.raw)) }
    }
}

pub(crate) fn prepare(ctx: &CudaContext, normalized: Normalized<'_>) -> Result<PreparedExecution> {
    let Normalized {
        api,
        spec,
        policy: options,
        bindings,
        storage,
        validation_reference,
    } = normalized;
    let cache = options
        .cache_dir
        .as_ref()
        .map(|path| CString::new(path.as_str()))
        .transpose()
        .map_err(|_| invalid("cache path contains NUL"))?;
    let policy = abi::Policy {
        workspace_limit: options.workspace_limit as u64,
        online_tune: options.online_tune as u32,
        allow_fallback: options.allow_fallback as u32,
        graph_safe: options.graph_safe as u32,
        deterministic: options.deterministic as u32,
        execution_mode: match options.execution_mode {
            super::GemmExecutionMode::Eager => 0,
            super::GemmExecutionMode::GraphReplay => 1,
        },
        cache_dir: cache
            .as_ref()
            .map_or(std::ptr::null(), |path| path.as_ptr()),
    };
    let tuning_bindings = if let Some(reference) = validation_reference {
        abi::TuningBindings {
            execution: bindings,
            original_a: reference.a.as_ptr(),
            original_a_len: reference.a.len() as u64,
            original_b: reference.b.as_ptr(),
            original_b_len: reference.b.len() as u64,
            original_bias: reference
                .bias
                .map_or(std::ptr::null(), |bias| bias.as_ptr()),
            original_bias_len: reference.bias.map_or(0, |bias| bias.len()) as u64,
            reference_kind: 1,
        }
    } else {
        abi::TuningBindings {
            execution: bindings,
            original_a: std::ptr::null(),
            original_a_len: 0,
            original_b: std::ptr::null(),
            original_b_len: 0,
            original_bias: std::ptr::null(),
            original_bias_len: 0,
            reference_kind: 0,
        }
    };
    let instance_key = format!(
        "{}|{:?}|{:?}|{}|{}|{}|{}",
        api.name,
        spec,
        bindings,
        policy.workspace_limit,
        policy.graph_safe,
        policy.deterministic,
        policy.execution_mode
    );
    if let Some(exec) = crate::workspace::lookup_gemm_instance(&instance_key) {
        return Ok(PreparedExecution { exec });
    }
    if !crate::workspace::may_prepare_native_resources() {
        return Err(invalid(
            "GEMM instance cache miss during capture; prepare the same bindings first",
        ));
    }

    let mut plan = std::ptr::null_mut();
    unsafe {
        status::check((api.plan_create)(
            ctx.runtime(),
            &spec,
            &policy,
            &tuning_bindings,
            &mut plan,
        ))?;
    }
    let plan = NativePlan {
        raw: plan,
        destroy: api.plan_destroy,
    };
    let summary = unsafe { CStr::from_ptr((api.plan_summary)(plan.raw)) }
        .to_string_lossy()
        .into_owned();
    let mut raw = std::ptr::null_mut();
    unsafe {
        status::check((api.instance_create)(plan.raw, &bindings, &mut raw))?;
    }
    #[cfg(test)]
    PREPARED_EXECUTION_CREATE_COUNT.with(|count| count.set(count.get() + 1));
    let exec = Rc::new(SharedExecution {
        raw,
        _plan: plan,
        stream: ctx.shared_stream(),
        _storage: storage,
        summary,
        enqueue: api.enqueue,
        destroy_instance: api.instance_destroy,
        _not_send: PhantomData,
    });
    crate::workspace::store_gemm_instance(instance_key, Rc::clone(&exec));
    Ok(PreparedExecution { exec })
}

pub(crate) fn execute(ctx: &CudaContext, normalized: Normalized<'_>) -> Result<()> {
    let mut instance = prepare(ctx, normalized)?;
    instance.enqueue()?;
    if crate::workspace::is_capturing() {
        Ok(())
    } else {
        ctx.synchronize().map_err(Error::Cuda)
    }
}

#[cfg(test)]
thread_local! {
    static PREPARED_EXECUTION_CREATE_COUNT: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
pub(crate) fn reset_prepared_execution_create_count() {
    PREPARED_EXECUTION_CREATE_COUNT.with(|count| count.set(0));
}

#[cfg(test)]
pub(crate) fn prepared_execution_create_count() -> usize {
    PREPARED_EXECUTION_CREATE_COUNT.with(std::cell::Cell::get)
}
