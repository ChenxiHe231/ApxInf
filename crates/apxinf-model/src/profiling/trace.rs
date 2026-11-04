//! Profiler trace ranges — hardware-neutral named time-span markers for an
//! external profiler.
//!
//! These emit no numbers themselves; they annotate the timeline so an external
//! profiler (NVTX / Nsight Systems on CUDA, MTTX / msys on MUSA) can attribute
//! time to named regions. Each backend's client is reached through the
//! accelerator seam; a build with neither provides a no-op stub so model code
//! can call `crate::profiling::trace::range("name")` unconditionally.
//!
//! One backend is chosen for the whole process, and the marker API carries no
//! device: when both features are compiled in CUDA wins, so a binary that runs
//! both records NVTX only and MUSA's ranges appear only in a build without it.

#[cfg(feature = "cuda")]
pub use apxinf_cuda_new::nvtx::{range, Range};

/// MUSA speaks NVTX's protocol, so the model's ranges reach msys unchanged.
#[cfg(all(feature = "musa", not(any(feature = "cuda", feature = "qwen35-cuda"))))]
pub use crate::accelerator::musa::mttx::{range, Range};

#[cfg(all(feature = "qwen35-cuda", not(feature = "cuda")))]
mod qwen35_nvtx {
    use std::{
        ffi::{c_void, CString},
        marker::PhantomData,
        os::raw::{c_char, c_int},
        rc::Rc,
        sync::OnceLock,
    };

    // Do not require a v2 NVTX library at link time: CUDA 13 may only ship
    // headers. Profiling on such a platform needs a loadable NVTX library.
    #[link(name = "dl")]
    extern "C" {
        fn dlopen(filename: *const c_char, flags: c_int) -> *mut c_void;
        fn dlsym(handle: *mut c_void, symbol: *const c_char) -> *mut c_void;
    }

    type Push = unsafe extern "C" fn(*const c_char) -> c_int;
    type Pop = unsafe extern "C" fn() -> c_int;

    struct Nvtx {
        push: Push,
        pop: Pop,
    }

    fn hooks() -> Option<&'static Nvtx> {
        static ENABLED: OnceLock<bool> = OnceLock::new();
        if !*ENABLED.get_or_init(|| std::env::var("APXINF_NVTX").as_deref() == Ok("1")) {
            return None;
        }
        static HOOKS: OnceLock<Option<Nvtx>> = OnceLock::new();
        HOOKS
            .get_or_init(|| {
                for library in [b"libnvToolsExt.so.1\0".as_slice(), b"libnvToolsExt.so\0"] {
                    // RTLD_NOW resolves the NVTX implementation once. We never
                    // dlclose: cached function pointers remain valid for the
                    // entire process, including guards active on other threads.
                    let handle = unsafe { dlopen(library.as_ptr().cast(), 2) };
                    if handle.is_null() {
                        continue;
                    }
                    let push = unsafe { dlsym(handle, b"nvtxRangePushA\0".as_ptr().cast()) };
                    let pop = unsafe { dlsym(handle, b"nvtxRangePop\0".as_ptr().cast()) };
                    if !push.is_null() && !pop.is_null() {
                        return Some(Nvtx {
                            // POSIX dlsym permits converting resolved function
                            // symbols to their declared C function-pointer type.
                            push: unsafe { std::mem::transmute::<*mut c_void, Push>(push) },
                            pop: unsafe { std::mem::transmute::<*mut c_void, Pop>(pop) },
                        });
                    }
                }
                eprintln!(
                    "[apxinf] APXINF_NVTX=1 but NVTX push/pop could not be loaded; trace markers disabled"
                );
                None
            })
            .as_ref()
    }

    /// NVTX push/pop ranges are thread-local and must be dropped on this thread.
    pub struct Range {
        _name: Option<CString>,
        pop: Option<Pop>,
        _thread: PhantomData<Rc<()>>,
    }

    impl Range {
        pub fn new(name: &str) -> Self {
            let Some(nvtx) = hooks() else {
                return Self {
                    _name: None,
                    pop: None,
                    _thread: PhantomData,
                };
            };
            let name = CString::new(name).unwrap_or_else(|_| CString::new("nvtx").unwrap());
            unsafe { (nvtx.push)(name.as_ptr()) };
            Self {
                _name: Some(name),
                pop: Some(nvtx.pop),
                _thread: PhantomData,
            }
        }
    }

    impl Drop for Range {
        fn drop(&mut self) {
            if let Some(pop) = self.pop {
                unsafe { pop() };
            }
        }
    }

    pub fn range(name: &str) -> Range {
        Range::new(name)
    }
}

#[cfg(all(feature = "qwen35-cuda", not(feature = "cuda")))]
pub use qwen35_nvtx::{range, Range};

#[cfg(not(any(feature = "cuda", feature = "qwen35-cuda", feature = "musa")))]
pub struct Range;

#[cfg(not(any(feature = "cuda", feature = "qwen35-cuda", feature = "musa")))]
#[inline]
pub fn range(_name: &str) -> Range {
    Range
}
