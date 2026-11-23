#pragma once
#include <stdint.h>
// Private backend execution ABI; no dependency/graph handles reach the model.
#ifdef __cplusplus
extern "C" {
#define APXINF_GRAPH_NOEXCEPT noexcept
#else
#define APXINF_GRAPH_NOEXCEPT
#endif
int32_t apxinf_capture_frontier_begin(void* stream,int32_t device,void** scope) APXINF_GRAPH_NOEXCEPT;
int32_t apxinf_capture_frontier_rewind(void* scope) APXINF_GRAPH_NOEXCEPT;
int32_t apxinf_capture_frontier_join(void* scope) APXINF_GRAPH_NOEXCEPT;
int32_t apxinf_capture_frontier_destroy(void* scope) APXINF_GRAPH_NOEXCEPT;
#ifdef __cplusplus
}
#endif
#undef APXINF_GRAPH_NOEXCEPT
