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
    pub(crate) fn apxinf_model_replace_rows_bf16(
        destination: *mut c_void,
        source: *const c_void,
        source_rows: *const c_void,
        destination_rows: i64,
        source_row_count: i64,
        columns: i64,
        stream: CudaStream,
    ) -> i32;
}
