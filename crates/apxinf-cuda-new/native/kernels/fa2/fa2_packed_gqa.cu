// Private packed-GQA wrapper. Existing FA2 entries and arithmetic remain unchanged.
#include <cuda_runtime.h>
#include <cutlass/numeric_types.h>
#include <cstdint>
#include "flash_attn/flash_fwd_launch_template.h"
#include "flash_attn/packed_gqa_fwd.h"
namespace FLASH_NAMESPACE {
template<typename Traits>
__global__ void flash_fwd_splitkv_packed_gqa_kernel(KERNEL_PARAM_MODIFIER const Flash_fwd_params params) {
    const int bidb=blockIdx.z/params.h, bidh=blockIdx.z-bidb*params.h;
    compute_attn_1rowblock_splitkv_packed_gqa<Traits,true,false,false,false,true,false,true,false>(
        params,bidb,bidh,blockIdx.x,blockIdx.y,params.num_splits);
}
template<typename Kernel_traits>
int run_packed(Flash_fwd_params& params,cudaStream_t stream) {
    constexpr size_t smem_size=Kernel_traits::kSmemSize;
    auto kernel=&flash_fwd_splitkv_packed_gqa_kernel<Kernel_traits>;
    if (smem_size>=48*1024) {
        auto err=cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,smem_size);
        if(err!=cudaSuccess) return static_cast<int>(err);
    }
    kernel<<<dim3(1,params.num_splits,params.b*params.h),Kernel_traits::kNThreads,smem_size,stream>>>(params);
    auto err=cudaGetLastError();if(err!=cudaSuccess)return static_cast<int>(err);
    const bool is_even_K=params.d==Kernel_traits::kHeadDim;
    if (params.num_splits > 1) {
        // We want kBlockM to be as small as possible for more parallelism.
        // With 128 threads we can load 512 elements at a time, so if headdim is divisible by 128, kBlockM = 4.
        // If headdim is divisible by 64, then we set kBlockM = 8, etc.
        constexpr static int kBlockM = Kernel_traits::kHeadDim % 128 == 0 ? 4 : (Kernel_traits::kHeadDim % 64 == 0 ? 8 : 16);
        dim3 grid_combine((params.b * params.h * params.seqlen_q + kBlockM - 1) / kBlockM);
        EVENK_SWITCH(is_even_K, IsEvenKConst, [&] {
            if (params.num_splits <= 2) {
                flash_fwd_splitkv_combine_kernel<Kernel_traits, kBlockM, 1, IsEvenKConst><<<grid_combine, Kernel_traits::kNThreads, 0, stream>>>(params);
            } else if (params.num_splits <= 4) {
                flash_fwd_splitkv_combine_kernel<Kernel_traits, kBlockM, 2, IsEvenKConst><<<grid_combine, Kernel_traits::kNThreads, 0, stream>>>(params);
            } else if (params.num_splits <= 8) {
                flash_fwd_splitkv_combine_kernel<Kernel_traits, kBlockM, 3, IsEvenKConst><<<grid_combine, Kernel_traits::kNThreads, 0, stream>>>(params);
            } else if (params.num_splits <= 16) {
                flash_fwd_splitkv_combine_kernel<Kernel_traits, kBlockM, 4, IsEvenKConst><<<grid_combine, Kernel_traits::kNThreads, 0, stream>>>(params);
            } else if (params.num_splits <= 32) {
                flash_fwd_splitkv_combine_kernel<Kernel_traits, kBlockM, 5, IsEvenKConst><<<grid_combine, Kernel_traits::kNThreads, 0, stream>>>(params);
            } else if (params.num_splits <= 64) {
                flash_fwd_splitkv_combine_kernel<Kernel_traits, kBlockM, 6, IsEvenKConst><<<grid_combine, Kernel_traits::kNThreads, 0, stream>>>(params);
            } else if (params.num_splits <= 128) {
                flash_fwd_splitkv_combine_kernel<Kernel_traits, kBlockM, 7, IsEvenKConst><<<grid_combine, Kernel_traits::kNThreads, 0, stream>>>(params);
            }
            C10_CUDA_KERNEL_LAUNCH_CHECK();
        });
    }
    return static_cast<int>(cudaGetLastError());
}
} // namespace FLASH_NAMESPACE
namespace apxinf_fa2_packed_detail {
constexpr float kLog2E=1.4426950408889634074f;
void fill_params(FLASH_NAMESPACE::Flash_fwd_params& params, bool is_bf16,
                 const void* q, const void* k, const void* v, void* output,
                 void* softmax_lse, int batch, int query_tokens,
                 int key_tokens, int query_heads, int kv_heads, int head_dim,
                 float softmax_scale) {
  params = {};
  params.is_bf16 = is_bf16;
  params.q_ptr = const_cast<void*>(q);
  params.k_ptr = const_cast<void*>(k);
  params.v_ptr = const_cast<void*>(v);
  params.o_ptr = output;
  params.softmax_lse_ptr = softmax_lse;

  const int64_t q_row_stride = static_cast<int64_t>(query_heads) * head_dim;
  const int64_t kv_row_stride = static_cast<int64_t>(kv_heads) * head_dim;
  params.q_batch_stride = static_cast<int64_t>(query_tokens) * q_row_stride;
  params.k_batch_stride = static_cast<int64_t>(key_tokens) * kv_row_stride;
  params.v_batch_stride = params.k_batch_stride;
  params.o_batch_stride = params.q_batch_stride;
  params.q_row_stride = q_row_stride;
  params.k_row_stride = kv_row_stride;
  params.v_row_stride = kv_row_stride;
  params.o_row_stride = q_row_stride;
  params.q_head_stride = head_dim;
  params.k_head_stride = head_dim;
  params.v_head_stride = head_dim;
  params.o_head_stride = head_dim;

  params.b = batch;
  params.h = query_heads;
  params.h_k = kv_heads;
  params.h_h_k_ratio = query_heads / kv_heads;
  params.seqlen_q = query_tokens;
  params.seqlen_k = key_tokens;
  params.seqlen_q_rounded = ((query_tokens + 127) / 128) * 128;
  params.seqlen_k_rounded = ((key_tokens + 127) / 128) * 128;
  params.d = head_dim;
  params.d_rounded = (head_dim + 31) & ~31;

  params.scale_softmax = softmax_scale;
  params.scale_softmax_log2 = softmax_scale * kLog2E;
  params.scale_softmax_rp_dropout = softmax_scale;
  params.p_dropout = 1.0f;
  params.p_dropout_in_uint8_t = 255;
  params.rp_dropout = 1.0f;

  params.is_causal = false;
  params.window_size_left = -1;
  params.window_size_right = -1;
  params.is_seqlens_k_cumulative = true;
  params.num_splits = 1;
}
__global__ void pack_q(const uint16_t* q,uint16_t* packed,int tokens) {
    const int elements=tokens*16*256;
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<elements;i+=blockDim.x*gridDim.x) {
        const int d=i%256,head=(i/256)%16,pos=i/(16*256);
        packed[((head/8)*(tokens*8)+pos*8+head%8)*256+d]=q[i];
    }
}
__global__ void unpack_o_lse(const uint16_t* packed,const float* packed_lse,
                             uint16_t* output,float* lse,int tokens) {
    const int elements=tokens*16*256;
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<elements;i+=blockDim.x*gridDim.x) {
        const int d=i%256,head=(i/256)%16,pos=i/(16*256);
        output[i]=packed[((head/8)*(tokens*8)+pos*8+head%8)*256+d];
    }
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<tokens*16;i+=blockDim.x*gridDim.x) {
        const int head=i/tokens,pos=i%tokens;
        lse[i]=packed_lse[(head/8)*(tokens*8)+pos*8+head%8];
    }
}
} // namespace apxinf_fa2_packed_detail
namespace apxinf::cuda_new::cutlass_ops {
int fa2_bf16_packed_gqa_splitkv(const void* q,const void* k,const void* v,void* output,
    void* lse,void* lse_accum,void* o_accum,void* packed_q,void* packed_o,void* packed_lse,
    int tokens,int keys,float scale,int splits,cudaStream_t stream) {
    if(!q||!k||!v||!output||!lse||!lse_accum||!o_accum||!packed_q||!packed_o||!packed_lse||
       tokens<1||tokens>4||keys<=64||keys<tokens||splits<2||splits>128)
        return static_cast<int>(cudaErrorInvalidValue);
    const int blocks=(tokens*16*256+255)/256;
    apxinf_fa2_packed_detail::pack_q<<<blocks,256,0,stream>>>(static_cast<const uint16_t*>(q),static_cast<uint16_t*>(packed_q),tokens);
    auto err=cudaGetLastError();if(err!=cudaSuccess)return static_cast<int>(err);
    FLASH_NAMESPACE::Flash_fwd_params params;
    apxinf_fa2_packed_detail::fill_params(params,true,packed_q,k,v,packed_o,packed_lse,1,tokens*8,keys,2,2,256,scale);
    params.q_row_stride=params.o_row_stride=256;
    params.q_head_stride=params.o_head_stride=tokens*8*256;
    params.is_causal=true;params.window_size_right=0;params.num_splits=splits;
    params.softmax_lseaccum_ptr=lse_accum;params.oaccum_ptr=o_accum;
    using Traits=::Flash_fwd_kernel_traits<256,64,64,4,false,false,cutlass::bfloat16_t>;
    int status=FLASH_NAMESPACE::run_packed<Traits>(params,stream);if(status)return status;
    apxinf_fa2_packed_detail::unpack_o_lse<<<blocks,256,0,stream>>>(static_cast<const uint16_t*>(packed_o),static_cast<const float*>(packed_lse),
        static_cast<uint16_t*>(output),static_cast<float*>(lse),tokens);
    return static_cast<int>(cudaGetLastError());
}
} // namespace apxinf::cuda_new::cutlass_ops
