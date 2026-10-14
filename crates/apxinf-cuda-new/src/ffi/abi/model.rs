use std::ffi::c_void;

use super::types::CudaStream;

unsafe extern "C" {
    pub(crate) fn apxinf_model_embedding_gather(
        table: *const c_void,
        ids: *const c_void,
        output: *mut c_void,
        tokens: i64,
        hidden: i64,
        vocab: i64,
        stream: CudaStream,
    ) -> i32;
    pub(crate) fn apxinf_model_argmax_bf16(
        logits: *const c_void,
        index: *mut c_void,
        count: i64,
        stream: CudaStream,
    ) -> i32;
    pub(crate) fn apxinf_model_cast_f32_to_bf16(
        input: *const c_void,
        output: *mut c_void,
        count: i64,
        stream: CudaStream,
    ) -> i32;
    pub(crate) fn apxinf_model_gelu_exact_bf16(
        input: *const c_void,
        output: *mut c_void,
        count: i64,
        stream: CudaStream,
    ) -> i32;
    pub(crate) fn apxinf_model_replace_rows_bf16(
        base: *const c_void,
        replacement: *const c_void,
        row_map: *const c_void,
        output: *mut c_void,
        rows: i64,
        columns: i64,
        replacement_rows: i64,
        stream: CudaStream,
    ) -> i32;
}
