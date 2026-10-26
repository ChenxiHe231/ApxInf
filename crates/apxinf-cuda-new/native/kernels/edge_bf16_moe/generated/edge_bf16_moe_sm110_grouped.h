
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
} edge_bf16_moe_sm110_grouped_Kernel_Module_t;

#ifdef __cplusplus
extern "C" {
#endif
void _mlir_edge_bf16_moe_sm110_grouped_cuda_init(void **);
void _mlir_edge_bf16_moe_sm110_grouped_cuda_load_to_device(void **);
static inline void edge_bf16_moe_sm110_grouped_Kernel_Module_Load(edge_bf16_moe_sm110_grouped_Kernel_Module_t *module) {
    cudaLibrary_t *libraryPtr = &(module->module);
    cudaError_t ret = cudaSuccess;
    struct {
        cudaLibrary_t **libraryPtr;
        cudaError_t *ret;
    } initArgs = {&libraryPtr, &ret};
    _mlir_edge_bf16_moe_sm110_grouped_cuda_init((void **)(&initArgs));
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
        _mlir_edge_bf16_moe_sm110_grouped_cuda_load_to_device((void **)(&loadArgs));
        CUTE_DSL_CUDA_ERROR_CHECK(ret);
    }
}

static inline void edge_bf16_moe_sm110_grouped_Kernel_Module_Unload(edge_bf16_moe_sm110_grouped_Kernel_Module_t *module) {
    CUTE_DSL_CUDA_ERROR_CHECK(cudaLibraryUnload(module->module));
}

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
extern "C"
#endif
void _mlir_edge_bf16_moe_sm110_grouped__mlir_ciface_cutlass_wrapper___main__export_grouped_gemmlocalsExportWrapper_object_at__Ptrgmem_Ptrgmem_Ptrgmem_Ptrgmem_Ptrgmem_Ptrgmem_Ptrgmem_128_128_128_256__CUstream0x0(void **args, int32_t num_args);

static inline int32_t cute_dsl_edge_bf16_moe_sm110_grouped_wrapper(edge_bf16_moe_sm110_grouped_Kernel_Module_t *module, void *a_ptr, void *b_ptr, void *d_ptr, void *problem_shapes_ptr, void *strides_ptr, void *addresses_ptr, void *scratch_ptr, int32_t max_m, int32_t n, int32_t k, int32_t group_count, int32_t max_active_clusters, cudaStream_t stream) {
    int32_t ret = 0;
    void *args[14] = {
        &a_ptr, &b_ptr, &d_ptr, &problem_shapes_ptr, &strides_ptr, &addresses_ptr, &scratch_ptr, &max_m, &n, &k, &group_count, &max_active_clusters, &stream,
        &ret
    };
    _mlir_edge_bf16_moe_sm110_grouped__mlir_ciface_cutlass_wrapper___main__export_grouped_gemmlocalsExportWrapper_object_at__Ptrgmem_Ptrgmem_Ptrgmem_Ptrgmem_Ptrgmem_Ptrgmem_Ptrgmem_128_128_128_256__CUstream0x0(args, 14);
    return ret;
}
