//! Private host-only ABI. CUDA13.2 types/signatures stay in the header-built C++ shim.
use std::ffi::c_void;
extern "C" {
    pub(crate) fn apxinf_capture_frontier_begin(stream:*mut c_void,device:i32,out:*mut *mut c_void)->i32;
    pub(crate) fn apxinf_capture_frontier_rewind(scope:*mut c_void)->i32;
    pub(crate) fn apxinf_capture_frontier_join(scope:*mut c_void)->i32;
    pub(crate) fn apxinf_capture_frontier_destroy(scope:*mut c_void)->i32;
    #[cfg(test)]
    pub(crate) fn apxinf_capture_frontier_test_inject(scope:*mut c_void,kind:i32)->i32;
    #[cfg(test)]
    pub(crate) fn apxinf_capture_frontier_test_topology(graph:*mut c_void,out:*mut usize)->i32;
    #[cfg(test)]
    pub(crate) fn apxinf_capture_frontier_test_edge(offset:u32,value:u8)->i32;
}
