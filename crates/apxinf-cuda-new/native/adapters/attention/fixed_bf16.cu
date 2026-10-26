#include "../../kernels/attention/fixed_bf16.cuh"

extern "C" cudaError_t apxinf_attention_row_softmax_f32_bf16(
    const void* input, void* output, uint32_t cols, uint32_t rows,
    cudaStream_t stream) {
  if (input == nullptr || output == nullptr || cols == 0 || rows == 0) {
    return cudaErrorInvalidValue;
  }
  row_softmax_f32_bf16_kernel<<<rows, 256, 0, stream>>>(
      static_cast<const float*>(input), static_cast<__nv_bfloat16*>(output),
      cols, rows);
  return cudaGetLastError();
}


extern "C" cudaError_t apxinf_attention_row_softmax_causal_f32(
    const void* input, void* output, uint32_t cols, uint32_t rows,
    uint32_t kv_offset, uint32_t n_heads, cudaStream_t stream) {
  if (input == nullptr || output == nullptr || cols == 0 || rows == 0 ||
      n_heads == 0 || n_heads > rows) {
    return cudaErrorInvalidValue;
  }
  row_softmax_causal_f32_kernel<<<rows, 256, 0, stream>>>(
      static_cast<const float*>(input), static_cast<float*>(output), cols, rows,
      kv_offset, n_heads);
  return cudaGetLastError();
}


extern "C" cudaError_t apxinf_attention_full_prepare_bf16(
    const void* fused, const void* q_norm_w, const void* k_norm_w,
    const void* cos, const void* sin, void* q_out, void* k_cache, void* v_cache,
    int seq, int cache_offset, int q_heads, int kv_heads, int head_dim,
    int rotary_dim, int64_t fused_width, int64_t cache_width, float eps,
    cudaStream_t stream) {
  if (fused == nullptr || q_norm_w == nullptr || k_norm_w == nullptr ||
      cos == nullptr || sin == nullptr || q_out == nullptr ||
      k_cache == nullptr || v_cache == nullptr || seq <= 0 ||
      cache_offset < 0 || q_heads <= 0 || kv_heads <= 0 || head_dim <= 0 ||
      rotary_dim <= 0 || (rotary_dim & 1) != 0 || rotary_dim > head_dim ||
      !(eps > 0.0f)) {
    return cudaErrorInvalidValue;
  }
  // Opt-in only for the measured D256 BF16 prefill geometry. Keep the
  // original C ABI and generic route for all other inputs and decode.
  if (seq > 1 && q_heads == 16 && kv_heads == 4 && head_dim == 256 &&
      rotary_dim == 64 && fused_width == 10240 && cache_width == 1024) {
    full_attn_prepare_bf16_warp_kernel
        <<<dim3(seq, 6), 128, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(fused),
        static_cast<const __nv_bfloat16*>(q_norm_w),
        static_cast<const __nv_bfloat16*>(k_norm_w),
        static_cast<const __nv_bfloat16*>(cos),
        static_cast<const __nv_bfloat16*>(sin),
        static_cast<__nv_bfloat16*>(q_out),
        static_cast<__nv_bfloat16*>(k_cache),
        static_cast<__nv_bfloat16*>(v_cache),
        cache_offset, q_heads, kv_heads, fused_width, cache_width, eps);
    return cudaGetLastError();
  }
  const size_t smem = static_cast<size_t>(head_dim) * sizeof(float);
  full_attn_prepare_bf16_kernel
      <<<dim3(seq, q_heads + 2 * kv_heads), 128, smem, stream>>>(
      static_cast<const __nv_bfloat16*>(fused),
      static_cast<const __nv_bfloat16*>(q_norm_w),
      static_cast<const __nv_bfloat16*>(k_norm_w),
      static_cast<const __nv_bfloat16*>(cos),
      static_cast<const __nv_bfloat16*>(sin),
      static_cast<__nv_bfloat16*>(q_out),
      static_cast<__nv_bfloat16*>(k_cache),
      static_cast<__nv_bfloat16*>(v_cache),
      cache_offset, q_heads, kv_heads, head_dim, rotary_dim, fused_width,
      cache_width, eps);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_attention_expert_qkv_prepare_bf16(
    const void* fused, const void* q_norm_w, const void* k_norm_w,
    const void* cos, const void* sin, void* q_out, void* gate_out, void* k_out,
    void* v_out, int seq, int q_heads, int kv_heads, int head_dim,
    int rotary_dim, int64_t fused_width, float eps, cudaStream_t stream) {
  if (fused == nullptr || q_norm_w == nullptr || k_norm_w == nullptr ||
      cos == nullptr || sin == nullptr || q_out == nullptr ||
      gate_out == nullptr || k_out == nullptr || v_out == nullptr ||
      seq <= 0 || q_heads <= 0 || kv_heads <= 0 || head_dim <= 0 ||
      rotary_dim <= 0 || (rotary_dim & 1) != 0 || rotary_dim > head_dim ||
      q_heads % kv_heads != 0 || !(eps > 0.0f)) {
    return cudaErrorInvalidValue;
  }
  const size_t smem = static_cast<size_t>(head_dim) * sizeof(float);
  expert_qkv_prepare_bf16_kernel
      <<<dim3(seq, q_heads + 2 * kv_heads), 128, smem, stream>>>(
      static_cast<const __nv_bfloat16*>(fused),
      static_cast<const __nv_bfloat16*>(q_norm_w),
      static_cast<const __nv_bfloat16*>(k_norm_w),
      static_cast<const __nv_bfloat16*>(cos),
      static_cast<const __nv_bfloat16*>(sin),
      static_cast<__nv_bfloat16*>(q_out),
      static_cast<__nv_bfloat16*>(gate_out),
      static_cast<__nv_bfloat16*>(k_out),
      static_cast<__nv_bfloat16*>(v_out),
      q_heads, kv_heads, head_dim, rotary_dim, fused_width, eps);
  return cudaGetLastError();
}
