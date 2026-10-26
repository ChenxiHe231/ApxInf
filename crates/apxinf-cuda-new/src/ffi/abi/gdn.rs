use std::ffi::c_void;

use super::types::CudaStream;

unsafe extern "C" {
    pub(crate) fn apxinf_gdn_prepare_fixed(
        device: i32,
        head_k_dim: i32,
        head_v_dim: i32,
        chunk_size: i32,
    ) -> i32;
    pub(crate) fn apxinf_gdn_qk_bf16_supported(
        device: i32,
        recurrent: i32,
        head_k_dim: i32,
        head_v_dim: i32,
        chunk_size: i32,
    ) -> i32;
    pub(crate) fn apxinf_gdn_chunk_scan_prepare(device: i32) -> i32;
    pub(crate) fn apxinf_gdn_conv_prepare(
        input: *const c_void,
        weight: *const c_void,
        window: *mut c_void,
        q_out: *mut c_void,
        k_out: *mut c_void,
        v_out: *mut c_void,
        decay: *const c_void,
        alpha: *mut c_void,
        tokens: i64,
        k_heads: i64,
        v_heads: i64,
        epsilon: f32,
        stream: CudaStream,
    ) -> i32;
    pub(crate) fn apxinf_gdn_recurrent_step(
        state: *mut c_void,
        q: *const c_void,
        k: *const c_void,
        v: *const c_void,
        decay: *const c_void,
        beta: *const c_void,
        output: *mut c_void,
        v_heads: i64,
        k_heads: i64,
        v_dim: i64,
        k_dim: i64,
        stream: CudaStream,
    ) -> i32;
    pub(crate) fn apxinf_gdn_gated_norm(
        input: *const c_void,
        gate: *const c_void,
        weight: *const c_void,
        output: *mut c_void,
        heads: i64,
        head_dim: i64,
        epsilon: f32,
        stream: CudaStream,
    ) -> i32;
    pub(crate) fn apxinf_gdn_gated_norm_quantize(
        input: *const c_void,
        gate: *const c_void,
        weight: *const c_void,
        output: *mut c_void,
        quantized: *mut c_void,
        rows: i64,
        fp16_input: i32,
        epsilon: f32,
        input_scale: f32,
        stream: CudaStream,
    ) -> i32;
    pub(crate) fn apxinf_gdn_causal_conv_step(
        window: *mut c_void,
        input: *const c_void,
        weight: *const c_void,
        output: *mut c_void,
        channels: i64,
        kernel_width: i64,
        stream: CudaStream,
    ) -> i32;
    pub(crate) fn apxinf_gdn_prepare_prefill(
        fused: *const c_void,
        q_out: *mut c_void,
        k_out: *mut c_void,
        v_out: *mut c_void,
        g: *const c_void,
        alpha: *mut c_void,
        tokens: i64,
        row_width: i64,
        k_heads: i64,
        v_heads: i64,
        dim: i64,
        epsilon: f32,
        stream: CudaStream,
    ) -> i32;
    pub(crate) fn apxinf_gdn_prefill(
        q: *const c_void,
        k: *const c_void,
        v: *const c_void,
        out: *mut c_void,
        gate_log: *const c_void,
        beta: *const c_void,
        cu_seqlens: *const c_void,
        state: *mut c_void,
        tensor_map_workspace: *mut c_void,
        tokens: i64,
        q_heads: i64,
        v_heads: i64,
        num_seqs: i64,
        scale: f32,
        stream: CudaStream,
    ) -> i32;
    pub(crate) fn apxinf_gdn_prefill_workspace_bytes(v_heads: i64, num_seqs: i64) -> i64;
    pub(crate) fn apxinf_gdn_causal_conv_forward(
        input: *const c_void,
        weight: *const c_void,
        output: *mut c_void,
        window: *mut c_void,
        tokens: i64,
        channels: i64,
        kernel_width: i64,
        stream: CudaStream,
    ) -> i32;
    pub(crate) fn apxinf_gdn_decay_and_beta_seq(
        a: *const c_void,
        b: *const c_void,
        a_log: *const c_void,
        dt_bias: *const c_void,
        decay: *mut c_void,
        beta: *mut c_void,
        tokens: i64,
        heads: i64,
        stream: CudaStream,
    ) -> i32;
    pub(crate) fn apxinf_gdn_gated_norm_seq(
        input: *const c_void,
        gate: *const c_void,
        weight: *const c_void,
        output: *mut c_void,
        tokens: i64,
        heads: i64,
        head_dim: i64,
        epsilon: f32,
        stream: CudaStream,
    ) -> i32;
    pub(crate) fn apxinf_gdn_gated_norm_seq_f16(
        input: *const c_void,
        gate: *const c_void,
        weight: *const c_void,
        output: *mut c_void,
        tokens: i64,
        heads: i64,
        head_dim: i64,
        epsilon: f32,
        stream: CudaStream,
    ) -> i32;
    pub(crate) fn apxinf_gdn_l2_normalize_heads(
        data: *mut c_void,
        heads: i64,
        head_dim: i64,
        epsilon: f32,
        stream: CudaStream,
    ) -> i32;
    pub(crate) fn apxinf_gdn_decay_and_beta(
        a: *const c_void,
        b: *const c_void,
        a_log: *const c_void,
        dt_bias: *const c_void,
        decay: *mut c_void,
        beta: *mut c_void,
        heads: i64,
        stream: CudaStream,
    ) -> i32;
    pub(crate) fn apxinf_gdn_chunk_scan(
        q: *const c_void,
        k: *const c_void,
        v: *const c_void,
        g: *const c_void,
        beta: *const c_void,
        out: *mut c_void,
        state: *mut c_void,
        seq_padded: i64,
        v_heads: i64,
        k_heads: i64,
        chunk_size: i64,
        k_dim: i64,
        num_chunks: i64,
        q_row_stride: i64,
        k_row_stride: i64,
        v_row_stride: i64,
        stream: CudaStream,
    ) -> i32;
}

mod fixed_launchers {
    use std::ffi::c_void;

    use crate::ffi::raw::cuda_runtime::{cudaError_t, cudaStream_t};

    unsafe extern "C" {
        #[cfg(apxinf_segmented_attention_sm110)]
        pub(crate) fn apxinf_gdn_bf16_prefill_sm110_init() -> i32;
        #[cfg(apxinf_segmented_attention_sm110)]
        pub(crate) fn apxinf_gdn_bf16_prefill_sm110(
            zba: *const c_void,
            conv_weight: *const c_void,
            new_conv_state: *mut c_void,
            a_log: *const f32,
            dt_bias_bf16: *const c_void,
            raw_qk_scratch: *mut c_void,
            q: *mut c_void,
            k: *mut c_void,
            v: *mut c_void,
            a: *mut c_void,
            b: *mut c_void,
            h_output_vmajor: *mut f32,
            final_state_kmajor: *mut f32,
            output_pad: *mut c_void,
            cu_seqlens: *mut i32,
            seq: i32,
            seq_pad: i32,
            stream: cudaStream_t,
        ) -> i32;
        #[allow(clippy::too_many_arguments)]
        pub(crate) fn apxinf_gdn_causal_conv1d_silu_bf16(
            x: *const c_void,
            weight: *const c_void,
            state: *const c_void,
            out: *mut c_void,
            new_state: *mut c_void,
            channels: i32,
            seq: i32,
            kernel_size: i32,
            x_row_stride: i64,
            stream: cudaStream_t,
        ) -> cudaError_t;
        #[allow(clippy::too_many_arguments)]
        pub(crate) fn apxinf_gdn_qk_prep_bf16(
            conv_out: *const c_void,
            q_out: *mut c_void,
            k_out: *mut c_void,
            seq: i32,
            seq_pad: i32,
            conv_dim: i32,
            key_dim: i32,
            num_v_heads: i32,
            head_k_dim: i32,
            scale: f32,
            eps: f32,
            recurrent: i32,
            stream: cudaStream_t,
        ) -> cudaError_t;
        #[allow(clippy::too_many_arguments)]
        pub(crate) fn apxinf_gdn_qk_prep_qk_bf16(
            conv_out: *const c_void,
            q_out: *mut c_void,
            k_out: *mut c_void,
            seq: i32,
            seq_pad: i32,
            conv_dim: i32,
            key_dim: i32,
            num_v_heads: i32,
            head_k_dim: i32,
            scale: f32,
            eps: f32,
            recurrent: i32,
            stream: cudaStream_t,
        ) -> cudaError_t;
        #[allow(clippy::too_many_arguments)]
        pub(crate) fn apxinf_gdn_vb_prep_bf16(
            conv_out: *const c_void,
            b_proj: *const c_void,
            a_proj: *const c_void,
            dt_bias: *const c_void,
            a_log: *const c_void,
            v_out: *mut c_void,
            beta_out: *mut c_void,
            g_out: *mut c_void,
            seq: i32,
            seq_pad: i32,
            conv_dim: i32,
            v_offset: i32,
            num_v_heads: i32,
            ba_row_stride: i32,
            head_v_dim: i32,
            stream: cudaStream_t,
        ) -> cudaError_t;
        pub(crate) fn apxinf_gdn_gate_prep_bf16(
            b_proj: *const c_void,
            a_proj: *const c_void,
            dt_bias: *const c_void,
            a_log: *const c_void,
            beta_out: *mut c_void,
            g_out: *mut c_void,
            seq: i32,
            seq_pad: i32,
            num_v_heads: i32,
            ba_row_stride: i32,
            stream: cudaStream_t,
        ) -> cudaError_t;
        pub(crate) fn apxinf_gdn_cumsum_f32(
            g: *const c_void,
            g_cum: *mut c_void,
            seq_pad: i32,
            num_v_heads: i32,
            chunk_size: i32,
            stream: cudaStream_t,
        ) -> cudaError_t;
        #[allow(clippy::too_many_arguments)]
        pub(crate) fn apxinf_gdn_attn_raw_f32(
            q: *const c_void,
            k: *const c_void,
            beta: *const c_void,
            g_cum: *const c_void,
            a_out: *mut c_void,
            t_out: *mut c_void,
            seq_pad: i32,
            num_v_heads: i32,
            head_k_dim: i32,
            chunk_size: i32,
            stream: cudaStream_t,
        ) -> cudaError_t;
        #[allow(clippy::too_many_arguments)]
        #[allow(clippy::too_many_arguments)]
        pub(crate) fn apxinf_gdn_attn_raw_solve_f1_qk_bf16(
            q: *const c_void,
            k: *const c_void,
            beta: *const c_void,
            g_cum: *const c_void,
            a_out: *mut c_void,
            t_out: *mut c_void,
            seq_pad: i32,
            num_v_heads: i32,
            head_k_dim: i32,
            chunk_size: i32,
            stream: cudaStream_t,
        ) -> cudaError_t;
        #[allow(clippy::too_many_arguments)]
        pub(crate) fn apxinf_gdn_tri_solve_f32(
            a: *mut c_void,
            matrices: i32,
            chunk_size: i32,
            stream: cudaStream_t,
        ) -> cudaError_t;
        #[allow(clippy::too_many_arguments)]
        pub(crate) fn apxinf_gdn_chunk_gemm_f32(
            a: *const c_void,
            v: *const c_void,
            k: *const c_void,
            beta: *const c_void,
            g_cum: *const c_void,
            vt_out: *mut c_void,
            kcd_out: *mut c_void,
            seq_pad: i32,
            num_v_heads: i32,
            head_k_dim: i32,
            head_v_dim: i32,
            chunk_size: i32,
            device: i32,
            stream: cudaStream_t,
        ) -> cudaError_t;
        #[allow(clippy::too_many_arguments)]
        #[allow(clippy::too_many_arguments)]
        pub(crate) fn apxinf_gdn_chunk_gemm_tri_k_bf16_direct_v(
            a: *const c_void,
            conv_out: *const c_void,
            k: *const c_void,
            beta: *const c_void,
            g_cum: *const c_void,
            vt_out: *mut c_void,
            kcd_out: *mut c_void,
            seq: i32,
            seq_pad: i32,
            conv_dim: i32,
            v_offset: i32,
            num_v_heads: i32,
            head_k_dim: i32,
            head_v_dim: i32,
            chunk_size: i32,
            stream: cudaStream_t,
        ) -> cudaError_t;
        #[allow(clippy::too_many_arguments)]
        #[allow(clippy::too_many_arguments)]
        pub(crate) fn apxinf_gdn_chunk_state_f32(
            q: *const c_void,
            k: *const c_void,
            g_cum: *const c_void,
            t_in: *const c_void,
            vt_in: *const c_void,
            kcd_in: *const c_void,
            state: *mut c_void,
            out: *mut c_void,
            seq: i32,
            seq_pad: i32,
            num_v_heads: i32,
            head_k_dim: i32,
            head_v_dim: i32,
            chunk_size: i32,
            total_chunks: i32,
            out_row_width: i32,
            device: i32,
            stream: cudaStream_t,
        ) -> cudaError_t;
        #[allow(clippy::too_many_arguments)]
        pub(crate) fn apxinf_gdn_chunk_state_qk_bf16(
            q: *const c_void,
            k: *const c_void,
            g_cum: *const c_void,
            t_in: *const c_void,
            vt_in: *const c_void,
            kcd_in: *const c_void,
            state: *mut c_void,
            out: *mut c_void,
            seq: i32,
            seq_pad: i32,
            num_v_heads: i32,
            head_k_dim: i32,
            head_v_dim: i32,
            chunk_size: i32,
            total_chunks: i32,
            out_row_width: i32,
            stream: cudaStream_t,
        ) -> cudaError_t;
        #[allow(clippy::too_many_arguments)]
        pub(crate) fn apxinf_gdn_recurrent_f32(
            q: *const c_void,
            k: *const c_void,
            v: *const c_void,
            beta: *const c_void,
            g: *const c_void,
            state: *mut c_void,
            out: *mut c_void,
            num_v_heads: i32,
            head_k_dim: i32,
            head_v_dim: i32,
            device: i32,
            stream: cudaStream_t,
        ) -> cudaError_t;
    }
}
pub(crate) use fixed_launchers::*;
