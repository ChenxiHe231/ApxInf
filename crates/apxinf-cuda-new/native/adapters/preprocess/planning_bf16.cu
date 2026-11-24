// Copyright 2026 ApxInf contributors.
// Fixed family launcher; kernel body and launch parameters are unchanged.
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <algorithm>
#include <cstdint>
#include <limits>

namespace {
#include "../../kernels/preprocess/planning_bf16.cuh"
}  // namespace

extern "C" cudaError_t apxinf_preprocess_concat7_cols_bf16(
    const void* s0, const void* s1, const void* s2, const void* s3,
    const void* s4, const void* s5, const void* s6, void* dst,
    int rows, int cols, int broadcast_mask, cudaStream_t stream) {
  if (s0 == nullptr || s1 == nullptr || s2 == nullptr || s3 == nullptr ||
      s4 == nullptr || s5 == nullptr || s6 == nullptr || dst == nullptr ||
      rows <= 0 || cols <= 0) {
    return cudaErrorInvalidValue;
  }
  concat7_cols_bf16_kernel<<<rows, 256, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(s0),
      static_cast<const __nv_bfloat16*>(s1),
      static_cast<const __nv_bfloat16*>(s2),
      static_cast<const __nv_bfloat16*>(s3),
      static_cast<const __nv_bfloat16*>(s4),
      static_cast<const __nv_bfloat16*>(s5),
      static_cast<const __nv_bfloat16*>(s6),
      static_cast<__nv_bfloat16*>(dst), rows, cols, broadcast_mask);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_preprocess_flow_update_f32(
    void* w, const void* endpoint, float remaining, float step, int64_t count,
    cudaStream_t stream) {
  if (w == nullptr || endpoint == nullptr || count <= 0 ||
      !(remaining > 0.0f) || !std::isfinite(step)) {
    return cudaErrorInvalidValue;
  }
  int blocks = static_cast<int>((count + 255) / 256);
  blocks = blocks > 1024 ? 1024 : blocks;
  flow_update_f32_kernel<<<blocks, 256, 0, stream>>>(
      static_cast<float*>(w), static_cast<const float*>(endpoint),
      remaining, step, count);
  return cudaGetLastError();
}

extern "C" cudaError_t apxinf_preprocess_vision_position_bf16(
    const void* table, void* embeddings, void* position_ids,
    int source_side, int height, int width, int hidden, int merge_size,
    cudaStream_t stream) {
  if (table == nullptr || embeddings == nullptr || position_ids == nullptr ||
      source_side <= 0 || height <= 0 || width <= 0 || hidden <= 0 ||
      merge_size <= 0 || height % merge_size != 0 ||
      width % merge_size != 0 ||
      height > std::numeric_limits<int>::max() / width) {
    return cudaErrorInvalidValue;
  }
  const int tokens = height * width;
  vision_position_kernel<false><<<tokens, 256, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(table),
      static_cast<__nv_bfloat16*>(embeddings),
      static_cast<int32_t*>(position_ids), source_side, height, width, hidden,
      merge_size);
  return cudaGetLastError();
}


extern "C" cudaError_t apxinf_preprocess_vision_position_staged_bf16(
    const void* table, void* embeddings, void* position_ids,
    int source_side, int height, int width, int hidden, int merge_size,
    cudaStream_t stream) {
  if (table == nullptr || embeddings == nullptr || position_ids == nullptr ||
      source_side <= 0 || height <= 0 || width <= 0 || hidden <= 0 ||
      merge_size <= 0 || height % merge_size != 0 ||
      width % merge_size != 0 ||
      height > std::numeric_limits<int>::max() / width) {
    return cudaErrorInvalidValue;
  }
  const int tokens = height * width;
  vision_position_kernel<true><<<tokens, 256, 0, stream>>>(
      static_cast<const __nv_bfloat16*>(table),
      static_cast<__nv_bfloat16*>(embeddings),
      static_cast<int32_t*>(position_ids), source_side, height, width, hidden,
      merge_size);
  return cudaGetLastError();
}

namespace {
__global__ void transpose_pad_channels_kernel(const __nv_bfloat16* input,__nv_bfloat16* output,
    int channels,int spatial,int padded,int64_t count) {
  for(int64_t i=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;i<count;i+=int64_t(gridDim.x)*blockDim.x) {
    const int channel=i%padded,position=(i/padded)%spatial;
    const int64_t batch=i/(int64_t(padded)*spatial);
    output[i]=channel<channels?input[(batch*channels+channel)*spatial+position]:__float2bfloat16(0.f);
  }
}
}
extern "C" cudaError_t apxinf_preprocess_transpose_pad_channels_bf16(const void* input,void* output,
    int batches,int channels,int spatial,int padded,cudaStream_t stream) {
  if(!input||!output||batches<=0||channels<=0||spatial<=0||padded<channels ||
      int64_t(batches)*channels>INT64_MAX/spatial || int64_t(batches)*padded>INT64_MAX/spatial)
    return cudaErrorInvalidValue;
  const int64_t count=int64_t(batches)*spatial*padded;
  const int blocks=int(std::min<int64_t>(4096,(count+255)/256));
  transpose_pad_channels_kernel<<<blocks,256,0,stream>>>((const __nv_bfloat16*)input,
      (__nv_bfloat16*)output,channels,spatial,padded,count);
  return cudaGetLastError();
}
