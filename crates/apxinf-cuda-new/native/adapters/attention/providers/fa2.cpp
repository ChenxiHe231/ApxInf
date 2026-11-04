#include <cuda_runtime.h>

#include "../internal.h"

#if defined(APXINF_ATTENTION_FA2)

#include <algorithm>
#include <cuda.h>
#include <cstdlib>
#include <cstring>
#include <map>
#include <limits>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace apxinf::cuda_new::cutlass_ops {

int fa2_bf16_packed_gqa_splitkv(const void*,const void*,const void*,void*,void*,void*,void*,void*,void*,void*,int,int,float,int,cudaStream_t);

int fa2_bf16_decode_splitkv(const void* q, const void* k, const void* v,
                           void* output, void* workspace, int key_tokens,
                           float softmax_scale, cudaStream_t stream);

int fa2_bf16(const void* q, const void* k, const void* v, void* output,
             void* softmax_lse, int batch, int query_tokens, int key_tokens,
             int query_heads, int kv_heads, int head_dim, float softmax_scale,
             cudaStream_t stream);
int fa2_bf16_causal(const void* q, const void* k, const void* v, void* output,
                    void* softmax_lse, int batch, int query_tokens,
                    int key_tokens, int query_heads, int kv_heads,
                    int head_dim, float softmax_scale, cudaStream_t stream);
int fa2_bf16_splitkv(
    const void* q, const void* k, const void* v, void* output,
    void* softmax_lse, void* softmax_lse_accum, void* o_accum, int batch,
    int query_tokens, int key_tokens, int query_heads, int kv_heads,
    int head_dim, float softmax_scale, int num_splits, cudaStream_t stream);
int fa2_bf16_causal_splitkv(
    const void* q, const void* k, const void* v, void* output,
    void* softmax_lse, void* softmax_lse_accum, void* o_accum, int batch,
    int query_tokens, int key_tokens, int query_heads, int kv_heads,
    int head_dim, float softmax_scale, int num_splits, cudaStream_t stream);
int fa2_f16(const void* q, const void* k, const void* v, void* output,
            void* softmax_lse, int batch, int query_tokens, int key_tokens,
            int query_heads, int kv_heads, int head_dim, float softmax_scale,
            cudaStream_t stream);
int fa2_f16_splitkv(
    const void* q, const void* k, const void* v, void* output,
    void* softmax_lse, void* softmax_lse_accum, void* o_accum, int batch,
    int query_tokens, int key_tokens, int query_heads, int kv_heads,
    int head_dim, float softmax_scale, int num_splits, cudaStream_t stream);
int fa2_f16_packed_qkv(const void* q, const void* k, const void* v,
                       void* output, void* softmax_lse, int batch,
                       int tokens, int heads, int head_dim,
                       float softmax_scale, cudaStream_t stream);
#if defined(APXINF_ATTENTION_FA2_E4M3)
int fa2_f16_direct_e4m3_522(
    const void* q, const void* k, const void* v, void* output,
    void* softmax_lse, int batch, int query_tokens, int key_tokens,
    int query_heads, int kv_heads, int head_dim, float softmax_scale,
    float output_scale, cudaStream_t stream);
#endif
}  // namespace apxinf::cuda_new::cutlass_ops

namespace apxinf::attention {
namespace {

struct Fa2State {
  float* softmax_lse = nullptr;
  bool splitkv = false;
};

struct Fa2SegmentedState {
  float* softmax_lse = nullptr;
  std::vector<uint32_t> offsets;

  ~Fa2SegmentedState() {
    if (softmax_lse != nullptr) cudaFree(softmax_lse);
  }
};

bool use_decode_splitkv(const Spec& spec) {
  return spec.dtype == APXINF_DTYPE_BF16 &&
         spec.output_dtype == APXINF_DTYPE_BF16 && spec.batch == 1 &&
         spec.query_tokens == 1 && spec.query_heads == 24 &&
         spec.kv_heads == 4 && spec.head_dim == 256 &&
         spec.key_tokens >= 128 &&
         (spec.mask == APXINF_ATTENTION_MASK_NONE ||
          (spec.mask == APXINF_ATTENTION_MASK_CAUSAL &&
           spec.query_start == spec.key_tokens - 1));
}
struct Fa2SplitKvState {
  float* softmax_lse = nullptr;
  float* softmax_lse_accum = nullptr;
  float* output_accum = nullptr;

  ~Fa2SplitKvState() {
    if (output_accum != nullptr) cudaFree(output_accum);
    if (softmax_lse_accum != nullptr) cudaFree(softmax_lse_accum);
    if (softmax_lse != nullptr) cudaFree(softmax_lse);
  }
};

size_t checked_multiply(size_t lhs, size_t rhs, const char* label) {
  if (rhs != 0 && lhs > std::numeric_limits<size_t>::max() / rhs) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT, label);
  }
  return lhs * rhs;
}

size_t splitkv_lse_elements(const Spec& spec) {
  return checked_multiply(
      checked_multiply(static_cast<size_t>(spec.batch),
                       static_cast<size_t>(spec.query_heads),
                       "FA2 split-KV LSE workspace size overflow"),
      static_cast<size_t>(spec.query_tokens),
      "FA2 split-KV LSE workspace size overflow");
}

// cudaGetDeviceProperties is a slow driver query; prepare can run per
// autotune configuration, so cache the only field split-KV needs.
int cached_multiprocessor_count(int device) {
  static std::mutex mutex;
  static std::unordered_map<int, int> counts;
  std::lock_guard<std::mutex> lock(mutex);
  const auto cached = counts.find(device);
  if (cached != counts.end()) return cached->second;
  cudaDeviceProp properties{};
  check_cuda(cudaGetDeviceProperties(&properties, device));
  counts.emplace(device, properties.multiProcessorCount);
  return properties.multiProcessorCount;
}

}  // namespace

size_t fa2_resource_requirements(const Spec& spec, int) {
  if (use_decode_splitkv(spec)) return (24 + 5 * 24 * (1 + 256)) * sizeof(float);
  const uint64_t elements = static_cast<uint64_t>(spec.batch) *
                            spec.query_heads * spec.query_tokens;
  if (elements > SIZE_MAX / sizeof(float)) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                  "FA2 softmax LSE workspace size overflow");
  }
  return static_cast<size_t>(elements) * sizeof(float);
}

void prepare_fa2(Execution& execution) {
  auto state = std::make_unique<Fa2State>();
  state->splitkv = use_decode_splitkv(execution.spec);
  const size_t bytes =
      fa2_resource_requirements(execution.spec, execution.configuration);
  if (bytes > execution.resource_limit) {
    throw Failure(APXINF_STATUS_UNSUPPORTED,
                  "FA2 softmax LSE workspace exceeds policy");
  }
  check_cuda(cudaMalloc(&state->softmax_lse, bytes));
  execution.resource_bytes = bytes;
  execution.provider_state = state.release();
}

size_t fa2_splitkv_resource_requirements(const Spec& spec,
                                         int configuration) {
  if (configuration <= 1 || configuration > 128) {
    throw Failure(APXINF_STATUS_UNSUPPORTED,
                  "invalid FA2 split-KV configuration");
  }
  const size_t splits = static_cast<size_t>(configuration);
  const size_t lse_elements = splitkv_lse_elements(spec);
  const size_t lse_bytes = checked_multiply(
      lse_elements, sizeof(float),
      "FA2 split-KV LSE workspace byte size overflow");
  const size_t lse_accum_bytes = checked_multiply(
      checked_multiply(splits, lse_elements,
                       "FA2 split-KV LSE accumulation size overflow"),
      sizeof(float),
      "FA2 split-KV LSE accumulation byte size overflow");
  const size_t output_accum_bytes = checked_multiply(
      checked_multiply(
          checked_multiply(splits, lse_elements,
                           "FA2 split-KV output accumulation size overflow"),
          static_cast<size_t>(spec.head_dim),
          "FA2 split-KV output accumulation size overflow"),
      sizeof(float),
      "FA2 split-KV output accumulation byte size overflow");
  if (lse_bytes > std::numeric_limits<size_t>::max() - lse_accum_bytes ||
      lse_bytes + lse_accum_bytes >
          std::numeric_limits<size_t>::max() - output_accum_bytes) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                  "FA2 split-KV total workspace size overflow");
  }
  return lse_bytes + lse_accum_bytes + output_accum_bytes;
}

void prepare_fa2_splitkv(Execution& execution) {
  const auto& spec = execution.spec;
  const size_t multiprocessors = static_cast<size_t>(
      cached_multiprocessor_count(execution.device));
  const size_t key_tile = spec.head_dim == 128 ? 128 : 64;
  const size_t key_tiles =
      (static_cast<size_t>(spec.key_tokens) + key_tile - 1) / key_tile;
  const size_t device_limit = std::min<size_t>(
      128, std::min(key_tiles, multiprocessors * 2));
  if (static_cast<size_t>(execution.configuration) > device_limit) {
    throw Failure(APXINF_STATUS_UNSUPPORTED,
                  "FA2 split-KV configuration exceeds device split limit");
  }
  const size_t required =
      fa2_splitkv_resource_requirements(spec, execution.configuration);
  if (required > execution.resource_limit) {
    throw Failure(APXINF_STATUS_UNSUPPORTED,
                  "FA2 split-KV workspace exceeds policy");
  }
  auto state = std::make_unique<Fa2SplitKvState>();
  const size_t lse_elements = splitkv_lse_elements(spec);
  const size_t lse_bytes = lse_elements * sizeof(float);
  const size_t lse_accum_bytes =
      static_cast<size_t>(execution.configuration) * lse_bytes;
  const size_t output_accum_bytes =
      static_cast<size_t>(execution.configuration) * lse_bytes * spec.head_dim;
  check_cuda(cudaMalloc(&state->softmax_lse, lse_bytes));
  check_cuda(cudaMalloc(&state->softmax_lse_accum, lse_accum_bytes));
  check_cuda(cudaMalloc(&state->output_accum, output_accum_bytes));
  // The combine kernel reduces over all `configuration` split slots. A split
  // that covers no key tile never writes its accumulator, so an unzeroed
  // allocation leaks whatever the allocator last held into the reduction, and
  // a fresh cudaMalloc per prepare makes that garbage differ run to run.
  // Stream-ordered so this cannot race the kernels already queued.
  {
    const auto stream =
        static_cast<cudaStream_t>(execution.bindings.stream);
    check_cuda(cudaMemsetAsync(state->softmax_lse, 0, lse_bytes, stream));
    check_cuda(
        cudaMemsetAsync(state->softmax_lse_accum, 0, lse_accum_bytes, stream));
    check_cuda(
        cudaMemsetAsync(state->output_accum, 0, output_accum_bytes, stream));
  }
  execution.resource_bytes = required;
  execution.provider_state = state.release();
}

cudaError_t launch_fa2_splitkv(Execution& execution) {
  auto* state = static_cast<Fa2SplitKvState*>(execution.provider_state);
  const auto& spec = execution.spec;
  const auto& bindings = execution.bindings;
  const auto stream = static_cast<cudaStream_t>(bindings.stream);
  if (spec.dtype == APXINF_DTYPE_F16) {
    // The F16 contract is gated to APXINF_ATTENTION_MASK_NONE upstream.
    return static_cast<cudaError_t>(
        apxinf::cuda_new::cutlass_ops::fa2_f16_splitkv(
            bindings.query, bindings.key, bindings.value, bindings.output,
            state->softmax_lse, state->softmax_lse_accum, state->output_accum,
            static_cast<int>(spec.batch), static_cast<int>(spec.query_tokens),
            static_cast<int>(spec.key_tokens),
            static_cast<int>(spec.query_heads),
            static_cast<int>(spec.kv_heads), static_cast<int>(spec.head_dim),
            bindings.scale, execution.configuration, stream));
  }
  const auto launch = spec.mask == APXINF_ATTENTION_MASK_CAUSAL
                          ? apxinf::cuda_new::cutlass_ops::fa2_bf16_causal_splitkv
                          : apxinf::cuda_new::cutlass_ops::fa2_bf16_splitkv;
  return static_cast<cudaError_t>(launch(
      bindings.query, bindings.key, bindings.value, bindings.output,
      state->softmax_lse, state->softmax_lse_accum, state->output_accum,
      static_cast<int>(spec.batch), static_cast<int>(spec.query_tokens),
      static_cast<int>(spec.key_tokens), static_cast<int>(spec.query_heads),
      static_cast<int>(spec.kv_heads), static_cast<int>(spec.head_dim),
      bindings.scale, execution.configuration, stream));
}

void destroy_fa2_splitkv(Execution& execution) noexcept {
  delete static_cast<Fa2SplitKvState*>(execution.provider_state);
  execution.provider_state = nullptr;
}

size_t fa2_segmented_resource_requirements(const Spec& spec, int) {
  const size_t elements = checked_multiply(
      static_cast<size_t>(spec.query_heads),
      static_cast<size_t>(spec.max_segment_tokens),
      "segmented FA2 LSE workspace size overflow");
  return checked_multiply(elements, sizeof(float),
                          "segmented FA2 LSE workspace size overflow");
}

void prepare_fa2_segmented(Execution& execution) {
  const size_t bytes =
      fa2_segmented_resource_requirements(execution.spec, 0);
  if (bytes > execution.resource_limit) {
    throw Failure(APXINF_STATUS_UNSUPPORTED,
                  "segmented FA2 LSE workspace exceeds policy");
  }
  auto state = std::make_unique<Fa2SegmentedState>();
  state->offsets.resize(static_cast<size_t>(execution.spec.segments) + 1);
  const auto stream =
      static_cast<cudaStream_t>(execution.bindings.stream);
  check_cuda(cudaMemcpyAsync(
      state->offsets.data(), execution.bindings.offsets,
      state->offsets.size() * sizeof(uint32_t), cudaMemcpyDeviceToHost,
      stream));
  check_cuda(cudaStreamSynchronize(stream));
  if (state->offsets.front() != 0 ||
      state->offsets.back() != execution.spec.query_tokens) {
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                  "segmented FA2 offsets do not span all tokens");
  }
  const size_t row_bytes =
      checked_multiply(
          checked_multiply(static_cast<size_t>(execution.spec.query_heads),
                           static_cast<size_t>(execution.spec.head_dim),
                           "segmented FA2 row size overflow"),
          dtype_bytes(execution.spec.dtype),
          "segmented FA2 row size overflow");
  for (size_t index = 1; index < state->offsets.size(); ++index) {
    const uint32_t begin = state->offsets[index - 1];
    const uint32_t end = state->offsets[index];
    if (end < begin ||
        static_cast<uint64_t>(end - begin) >
            static_cast<uint64_t>(execution.spec.max_segment_tokens)) {
      throw Failure(APXINF_STATUS_INVALID_ARGUMENT,
                    "segmented FA2 offsets contradict the Spec");
    }
    // FA2 vector loads require every segment view, not only the original
    // allocation, to retain the candidate's 16-byte pointer alignment.
    if (((static_cast<size_t>(begin) % 16) * (row_bytes % 16)) % 16 != 0) {
      throw Failure(APXINF_STATUS_UNSUPPORTED,
                    "segmented FA2 segment pointer is not 16-byte aligned");
    }
  }
  check_cuda(cudaMalloc(&state->softmax_lse, bytes));
  execution.resource_bytes = bytes;
  execution.provider_state = state.release();
}

cudaError_t launch_fa2_segmented(Execution& execution) {
  auto* state =
      static_cast<Fa2SegmentedState*>(execution.provider_state);
  const auto& spec = execution.spec;
  const auto& bindings = execution.bindings;
  const auto stream = static_cast<cudaStream_t>(bindings.stream);
  const size_t row_bytes = static_cast<size_t>(spec.query_heads) *
                           static_cast<size_t>(spec.head_dim) *
                           dtype_bytes(spec.dtype);
  for (size_t index = 1; index < state->offsets.size(); ++index) {
    const size_t begin = state->offsets[index - 1];
    const size_t tokens = state->offsets[index] - begin;
    if (tokens == 0) continue;
    const size_t offset = begin * row_bytes;
    const auto* query = static_cast<const char*>(bindings.query) + offset;
    const auto* key = static_cast<const char*>(bindings.key) + offset;
    const auto* value = static_cast<const char*>(bindings.value) + offset;
    auto* output = static_cast<char*>(bindings.output) + offset;
    int status = cudaErrorInvalidValue;
    if (spec.dtype == APXINF_DTYPE_BF16) {
      status = apxinf::cuda_new::cutlass_ops::fa2_bf16(
          query, key, value, output, state->softmax_lse, 1,
          static_cast<int>(tokens), static_cast<int>(tokens),
          static_cast<int>(spec.query_heads),
          static_cast<int>(spec.kv_heads), static_cast<int>(spec.head_dim),
          bindings.scale, stream);
    } else if (spec.dtype == APXINF_DTYPE_F16) {
      status = apxinf::cuda_new::cutlass_ops::fa2_f16(
          query, key, value, output, state->softmax_lse, 1,
          static_cast<int>(tokens), static_cast<int>(tokens),
          static_cast<int>(spec.query_heads),
          static_cast<int>(spec.kv_heads), static_cast<int>(spec.head_dim),
          bindings.scale, stream);
    }
    if (status != 0) return static_cast<cudaError_t>(status);
  }
  return cudaSuccess;
}

void destroy_fa2_segmented(Execution& execution) noexcept {
  delete static_cast<Fa2SegmentedState*>(execution.provider_state);
  execution.provider_state = nullptr;
}

cudaError_t launch_fa2(Execution& execution) {
  auto* state = static_cast<Fa2State*>(execution.provider_state);
  const auto& spec = execution.spec;
  const auto& bindings = execution.bindings;
  const auto stream = static_cast<cudaStream_t>(bindings.stream);
  if (state->splitkv) {
    return static_cast<cudaError_t>(
        apxinf::cuda_new::cutlass_ops::fa2_bf16_decode_splitkv(
            bindings.query, bindings.key, bindings.value, bindings.output,
            state->softmax_lse, static_cast<int>(spec.key_tokens),
            bindings.scale, stream));
  }
  if (spec.qkv_layout == APXINF_ATTENTION_QKV_LAYOUT_PACKED_TOKEN) {
    return static_cast<cudaError_t>(
        apxinf::cuda_new::cutlass_ops::fa2_f16_packed_qkv(
            bindings.query, bindings.key, bindings.value, bindings.output,
            state->softmax_lse, static_cast<int>(spec.batch),
            static_cast<int>(spec.query_tokens),
            static_cast<int>(spec.query_heads), static_cast<int>(spec.head_dim),
            bindings.scale, stream));
  }
#if defined(APXINF_ATTENTION_FA2_E4M3)
  if (spec.dtype == APXINF_DTYPE_F16 &&
      spec.output_dtype == APXINF_DTYPE_E4M3) {
    return static_cast<cudaError_t>(
        apxinf::cuda_new::cutlass_ops::fa2_f16_direct_e4m3_522(
            bindings.query, bindings.key, bindings.value, bindings.output,
            state->softmax_lse, static_cast<int>(spec.batch),
            static_cast<int>(spec.query_tokens),
            static_cast<int>(spec.key_tokens),
            static_cast<int>(spec.query_heads),
            static_cast<int>(spec.kv_heads), static_cast<int>(spec.head_dim),
            bindings.scale, bindings.output_scale, stream));
  }
#endif
  int status = static_cast<int>(cudaErrorInvalidValue);
  if (spec.dtype == APXINF_DTYPE_BF16) {
    const auto launch = spec.mask == APXINF_ATTENTION_MASK_CAUSAL
                            ? apxinf::cuda_new::cutlass_ops::fa2_bf16_causal
                            : apxinf::cuda_new::cutlass_ops::fa2_bf16;
    status = launch(
        bindings.query, bindings.key, bindings.value, bindings.output,
        state->softmax_lse, static_cast<int>(spec.batch),
        static_cast<int>(spec.query_tokens), static_cast<int>(spec.key_tokens),
        static_cast<int>(spec.query_heads), static_cast<int>(spec.kv_heads),
        static_cast<int>(spec.head_dim), bindings.scale, stream);
  } else if (spec.dtype == APXINF_DTYPE_F16 &&
             spec.mask == APXINF_ATTENTION_MASK_NONE) {
    status = apxinf::cuda_new::cutlass_ops::fa2_f16(
        bindings.query, bindings.key, bindings.value, bindings.output,
        state->softmax_lse, static_cast<int>(spec.batch),
        static_cast<int>(spec.query_tokens), static_cast<int>(spec.key_tokens),
        static_cast<int>(spec.query_heads), static_cast<int>(spec.kv_heads),
        static_cast<int>(spec.head_dim), bindings.scale, stream);
  }
  return static_cast<cudaError_t>(status);
}

void destroy_fa2(Execution& execution) noexcept {
  auto* state = static_cast<Fa2State*>(execution.provider_state);
  if (state == nullptr) return;
  if (state->softmax_lse != nullptr) cudaFree(state->softmax_lse);
  delete state;
  execution.provider_state = nullptr;
}


namespace {
constexpr size_t kPackedQBytes=4*16*256*2;
constexpr size_t kPackedSlabBytes=2*kPackedQBytes+4*16*sizeof(float); //65792, all offsets256B aligned.
constexpr size_t kPackedIdleByteLimit=32*kPackedSlabBytes;
struct PackedScratch { void* raw=nullptr; int device=0; CUcontext context=nullptr; PackedScratch* next=nullptr; };
struct PackedPool {
  std::mutex mutex;
  std::map<int,CUcontext> retained_primary;
  std::map<std::pair<int,uintptr_t>,std::vector<std::unique_ptr<PackedScratch>>> idle;
  std::map<int,size_t> idle_bytes;
  PackedScratch* quarantined=nullptr;
};
// Bound idle storage survives until process exit, as does the retained primary
// context. It is never shared between live executions or capture graphs.
PackedPool& packed_pool() { static auto* pool=new PackedPool; return *pool; }
std::unique_ptr<PackedScratch> acquire_packed_scratch(int device) {
  CUcontext current=nullptr;
  if(cuCtxGetCurrent(&current)!=CUDA_SUCCESS || !current)
    throw Failure(APXINF_STATUS_PROVIDER_ERROR,"packed GQA requires current CUDA context");
  auto& pool=packed_pool();std::lock_guard<std::mutex> guard(pool.mutex);
  auto found=pool.retained_primary.find(device);
  if(found==pool.retained_primary.end()) {
    CUcontext primary=nullptr;
    if(cuDevicePrimaryCtxRetain(&primary,device)!=CUDA_SUCCESS)
      throw Failure(APXINF_STATUS_PROVIDER_ERROR,"packed GQA primary context retain failed");
    try { found=pool.retained_primary.emplace(device,primary).first; }
    catch(...) { (void)cuDevicePrimaryCtxRelease(device); throw; }
  }
  if(current!=found->second)
    throw Failure(APXINF_STATUS_UNSUPPORTED,"packed GQA pooled execution requires primary context");
  pool.idle_bytes.try_emplace(device,0); // no map-node allocation during noexcept retirement
  auto key=std::make_pair(device,reinterpret_cast<uintptr_t>(current));
  auto& slots=pool.idle[key];
  if(!slots.empty()) { auto result=std::move(slots.back());slots.pop_back();pool.idle_bytes[device]-=kPackedSlabBytes;return result; }
  auto result=std::make_unique<PackedScratch>();result->device=device;result->context=current;
  check_cuda(cudaMalloc(&result->raw,kPackedSlabBytes));return result;
}
void retire_packed_scratch(std::unique_ptr<PackedScratch> scratch,bool safe) noexcept {
  if(!scratch)return;
  auto& pool=packed_pool();CUcontext current=nullptr;
  safe=safe && cuCtxGetCurrent(&current)==CUDA_SUCCESS && current==scratch->context;
  std::lock_guard<std::mutex> guard(pool.mutex);
  if(!safe) {
    // A failed CUDA retirement must never make this memory available again.
    scratch->next=pool.quarantined;pool.quarantined=scratch.release();return;
  }
  if(pool.idle_bytes[scratch->device]+kPackedSlabBytes<=kPackedIdleByteLimit) {
    auto key=std::make_pair(scratch->device,reinterpret_cast<uintptr_t>(scratch->context));
    try { pool.idle[key].push_back(std::move(scratch));pool.idle_bytes[key.first]+=kPackedSlabBytes;return; }
    catch(...) { /* retired memory may be freed if host retention allocation fails */ }
  }
  if(scratch && cudaFree(scratch->raw)!=cudaSuccess) {
    scratch->next=pool.quarantined;pool.quarantined=scratch.release();
  }
}
struct Fa2PackedState {
  float* softmax_lse=nullptr;float* lse_accum=nullptr;float* output_accum=nullptr;
  std::unique_ptr<PackedScratch> scratch;
  bool enqueued=false;
  ~Fa2PackedState() {
    bool safe=true,had_nonnull_free=false;
    // Same three cudaMalloc/cudaFree boundaries as ordinary split-KV. Successful
    // non-async frees retire preceding work before a private slab is reusable.
    for(void* ptr:{static_cast<void*>(output_accum),static_cast<void*>(lse_accum),static_cast<void*>(softmax_lse)})
      if(ptr) { had_nonnull_free=true;safe=(cudaFree(ptr)==cudaSuccess)&&safe; }
    retire_packed_scratch(std::move(scratch),safe&&(had_nonnull_free||!enqueued));
  }
};
} // namespace
bool fa2_packed_gqa_requested(const Spec& spec) {
  static const int mode=[] {
    const char* value=std::getenv("APXINF_FA2_PACKED_GQA");
    if(!value || std::strcmp(value,"0")==0)return 0;
    if(std::strcmp(value,"1")==0)return 1;
    return -1;
  }();
  if(mode<0)throw Failure(APXINF_STATUS_INVALID_ARGUMENT,"APXINF_FA2_PACKED_GQA requires0/1");
  if(mode==0)return false;
  return spec.dtype==APXINF_DTYPE_BF16 && spec.output_dtype==APXINF_DTYPE_BF16 &&
    spec.batch==1 && spec.query_heads==16 && spec.kv_heads==2 && spec.head_dim==256 &&
    spec.query_tokens>=1 && spec.query_tokens<=4 && spec.key_tokens>64 &&
    spec.key_tokens>spec.query_tokens && spec.key_capacity==spec.key_tokens &&
    spec.mask==APXINF_ATTENTION_MASK_CAUSAL && spec.dynamic_decode==0 &&
    spec.query_start==spec.key_tokens-spec.query_tokens &&
    (spec.semantic==APXINF_ATTENTION_SEMANTIC_DENSE || spec.semantic==APXINF_ATTENTION_SEMANTIC_KV_CACHE);
}
size_t fa2_packed_gqa_resource_requirements(const Spec& spec,int configuration) {
  const size_t old=fa2_splitkv_resource_requirements(spec,configuration);
  if(old>std::numeric_limits<size_t>::max()-kPackedSlabBytes)
    throw Failure(APXINF_STATUS_INVALID_ARGUMENT,"packed GQA workspace overflow");
  return old+kPackedSlabBytes;
}
void prepare_fa2_packed_gqa(Execution& execution) {
  const auto& spec=execution.spec;
  if(!fa2_packed_gqa_requested(spec))throw Failure(APXINF_STATUS_UNSUPPORTED,"packed GQA unsupported Spec/policy");
  const size_t limit=std::min<size_t>(128,std::min<size_t>((spec.key_tokens+63)/64,cached_multiprocessor_count(execution.device)*2));
  if(execution.configuration<2 || static_cast<size_t>(execution.configuration)>limit)
    throw Failure(APXINF_STATUS_UNSUPPORTED,"packed GQA split count exceeds device limit");
  const size_t required=fa2_packed_gqa_resource_requirements(spec,execution.configuration);
  if(required>execution.resource_limit)throw Failure(APXINF_STATUS_UNSUPPORTED,"packed GQA workspace exceeds policy");
  auto state=std::make_unique<Fa2PackedState>();state->scratch=acquire_packed_scratch(execution.device);
  const size_t count=splitkv_lse_elements(spec),lse_bytes=count*sizeof(float);
  const size_t accum_bytes=execution.configuration*lse_bytes;
  check_cuda(cudaMalloc(&state->softmax_lse,lse_bytes));
  check_cuda(cudaMalloc(&state->lse_accum,accum_bytes));
  check_cuda(cudaMalloc(&state->output_accum,accum_bytes*spec.head_dim));
  const auto stream=static_cast<cudaStream_t>(execution.bindings.stream);
  check_cuda(cudaMemsetAsync(state->softmax_lse,0,lse_bytes,stream));
  check_cuda(cudaMemsetAsync(state->lse_accum,0,accum_bytes,stream));
  check_cuda(cudaMemsetAsync(state->output_accum,0,accum_bytes*spec.head_dim,stream));
  execution.resource_bytes=required;execution.provider_state=state.release();
}
cudaError_t launch_fa2_packed_gqa(Execution& execution) {
  auto* state=static_cast<Fa2PackedState*>(execution.provider_state);auto* raw=static_cast<unsigned char*>(state->scratch->raw);
  const auto& b=execution.bindings;state->enqueued=true;
  return static_cast<cudaError_t>(apxinf::cuda_new::cutlass_ops::fa2_bf16_packed_gqa_splitkv(
    b.query,b.key,b.value,b.output,state->softmax_lse,state->lse_accum,state->output_accum,
    raw,raw+kPackedQBytes,raw+2*kPackedQBytes,static_cast<int>(execution.spec.query_tokens),
    static_cast<int>(execution.spec.key_tokens),b.scale,execution.configuration,static_cast<cudaStream_t>(b.stream)));
}
void destroy_fa2_packed_gqa(Execution& execution) noexcept { delete static_cast<Fa2PackedState*>(execution.provider_state); }

}  // namespace apxinf::attention

#endif
