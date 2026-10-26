
#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include <stdint.h>


// Macro to check for cuda errors.
#ifndef CUTE_DSL_CUDA_ERROR_CHECK
#define CUTE_DSL_CUDA_ERROR_CHECK(err) { \
    if ((err) != cudaSuccess) { \
        printf("Got Cuda Error %s: %s\n", cudaGetErrorName(err), cudaGetErrorString(err)); \
    } \
}

#endif

typedef struct {
    cudaLibrary_t module;
} nvfp4_moe_sm110_fc1_swiglu_n128_Kernel_Module_t;

#ifdef __cplusplus
extern "C" {
#endif
void _mlir_nvfp4_moe_sm110_fc1_swiglu_n128_cuda_init(void **);
void _mlir_nvfp4_moe_sm110_fc1_swiglu_n128_cuda_load_to_device(void **);
static inline void nvfp4_moe_sm110_fc1_swiglu_n128_Kernel_Module_Load(nvfp4_moe_sm110_fc1_swiglu_n128_Kernel_Module_t *module) {
    cudaLibrary_t *libraryPtr = &(module->module);
    cudaError_t ret = cudaSuccess;
    struct {
        cudaLibrary_t **libraryPtr;
        cudaError_t *ret;
    } initArgs = {&libraryPtr, &ret};
    _mlir_nvfp4_moe_sm110_fc1_swiglu_n128_cuda_init((void **)(&initArgs));
    CUTE_DSL_CUDA_ERROR_CHECK(ret);
    int32_t device_id = 0;
    struct {
        cudaLibrary_t **library;
        int32_t *device_id;
        cudaError_t *ret;
    } loadArgs = {&libraryPtr, &device_id, &ret};
    int32_t device_count;
    CUTE_DSL_CUDA_ERROR_CHECK(cudaGetDeviceCount(&device_count));
    for (int32_t i = 0; i < device_count; i++) {
        device_id = i;
        _mlir_nvfp4_moe_sm110_fc1_swiglu_n128_cuda_load_to_device((void **)(&loadArgs));
        CUTE_DSL_CUDA_ERROR_CHECK(ret);
    }
}

static inline void nvfp4_moe_sm110_fc1_swiglu_n128_Kernel_Module_Unload(nvfp4_moe_sm110_fc1_swiglu_n128_Kernel_Module_t *module) {
    CUTE_DSL_CUDA_ERROR_CHECK(cudaLibraryUnload(module->module));
}

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C"
#endif
void _mlir_nvfp4_moe_sm110_fc1_swiglu_n128__mlir_ciface_cutlass_single_b_wrapper_Ptrgmem_Ptrgmem_Ptrgmem_Ptrgmem_Ptrgmem_Ptrgmem_Ptrgmem_262144__Ptrgmem_Ptrgmem_Ptrgmem_Ptrgmem_Ptrgmem_Ptrgmem_Ptrgmem_128_32768_1024_2048_256_128_16___CU(void **args, int32_t num_args);

static inline int32_t cute_dsl_nvfp4_moe_sm110_fc1_swiglu_n128_wrapper(nvfp4_moe_sm110_fc1_swiglu_n128_Kernel_Module_t *module, void *a_ptr, void *b_ptr, void *a_sf_ptr, void *b_sf_ptr, void *c_ptr, void *c_sf_ptr, void *output_ptr, int64_t output_elements, int32_t fuse_output_zero, void *alpha_ptr, void *input_global_scale_ptr, void *down_input_scale_ptr, void *tile_idx_to_group_idx_ptr, void *tile_idx_to_mn_limit_ptr, void *token_id_mapping_ptr, void *num_non_exiting_tiles_ptr, int64_t orig_m, int64_t m, int64_t n, int64_t k, int64_t l, int32_t max_active_clusters, int32_t enable_pdl, cudaStream_t stream) {
    int32_t ret = 0;
    void *args[25] = {
        &a_ptr, &b_ptr, &a_sf_ptr, &b_sf_ptr, &c_ptr, &c_sf_ptr, &output_ptr, &output_elements, &fuse_output_zero, &alpha_ptr, &input_global_scale_ptr, &down_input_scale_ptr, &tile_idx_to_group_idx_ptr, &tile_idx_to_mn_limit_ptr, &token_id_mapping_ptr, &num_non_exiting_tiles_ptr, &orig_m, &m, &n, &k, &l, &max_active_clusters, &enable_pdl, &stream,
        &ret
    };
    _mlir_nvfp4_moe_sm110_fc1_swiglu_n128__mlir_ciface_cutlass_single_b_wrapper_Ptrgmem_Ptrgmem_Ptrgmem_Ptrgmem_Ptrgmem_Ptrgmem_Ptrgmem_262144__Ptrgmem_Ptrgmem_Ptrgmem_Ptrgmem_Ptrgmem_Ptrgmem_Ptrgmem_128_32768_1024_2048_256_128_16___CU(args, 25);
    return ret;
}
