// CUDA12.8+ host-only capture-frontier scope. No kernel/math/provider dispatch.
#include <cuda_runtime_api.h>
#include "../../include/apxinf_cuda/graph_capture.h"
#include <algorithm>
#include <cstring>
#include <new>
#include <memory>
#include <thread>
#include <vector>

#if CUDART_VERSION >= 12080
namespace {
thread_local bool inject_nondefault_edge = false; // Internal unit-test injection, never set by model code.
bool default_edge(const cudaGraphEdgeData& edge) {
  const cudaGraphEdgeData normal{};
  return std::memcmp(&edge, &normal, sizeof(edge)) == 0;
}
struct Snapshot {
  cudaGraph_t graph = nullptr;
  unsigned long long id = 0;
  std::vector<cudaGraphNode_t> nodes;
  std::vector<cudaGraphEdgeData> edges;
};
struct Scope {
  cudaStream_t stream = nullptr;
  int device = -1;
  std::thread::id thread;
  Snapshot prefix, routed;
  bool rewind_attempted = false;
  bool joined = false;
};
cudaError_t snapshot(cudaStream_t stream, Snapshot& out) {
  cudaStreamCaptureStatus status{};
  const cudaGraphNode_t* nodes = nullptr;
  const cudaGraphEdgeData* edges = nullptr;
  size_t count = 0;
  auto e = cudaStreamGetCaptureInfo_v3(stream, &status, &out.id, &out.graph,
                                    &nodes, &edges, &count);
  if (e != cudaSuccess) return e;
  if (status != cudaStreamCaptureStatusActive || !out.graph)
    return cudaErrorIllegalState;
  if (count && (!nodes || !edges)) return cudaErrorInvalidValue;
  // Own the arrays before the next API operating on this stream.
  out.nodes.clear(); out.edges.clear();
  if (count) { out.nodes.assign(nodes, nodes + count); out.edges.assign(edges, edges + count); }
  if (inject_nondefault_edge && !out.edges.empty()) {
    inject_nondefault_edge = false;
    reinterpret_cast<unsigned char*>(&out.edges.front())[0] = 1;
  }
  for (const auto& edge : out.edges)
    if (!default_edge(edge))
      return cudaErrorNotSupported; // No silent weakening of PDL/port semantics.
  return cudaSuccess;
}
cudaError_t validate(Scope& s, Snapshot& current) {
  if (s.thread != std::this_thread::get_id()) return cudaErrorInvalidValue;
  int device = -1; auto e = cudaGetDevice(&device); if (e) return e;
  if (device != s.device) return cudaErrorInvalidDevice;
  e = snapshot(s.stream, current); if (e) return e;
  if (current.graph != s.prefix.graph || current.id != s.prefix.id)
    return cudaErrorInvalidValue;
  return cudaSuccess;
}
cudaError_t set(Scope& s, std::vector<cudaGraphNode_t> nodes) {
  return cudaStreamUpdateCaptureDependencies_v2(s.stream,
      nodes.empty() ? nullptr : nodes.data(), nullptr, nodes.size(),
      cudaStreamSetCaptureDependencies);
}
std::vector<cudaGraphNode_t> unite(const Snapshot& a, const Snapshot& b) {
  auto nodes = a.nodes;
  for (auto node : b.nodes)
    if (std::find(nodes.begin(), nodes.end(), node) == nodes.end()) nodes.push_back(node);
  return nodes;
}
template<class F> int boundary(F function) noexcept {
  try { return static_cast<int>(function()); }
  catch (const std::bad_alloc&) { return cudaErrorMemoryAllocation; }
  catch (...) { return cudaErrorUnknown; }
}
}
extern "C" int apxinf_capture_frontier_begin(void* stream, int device, void** out) noexcept {
  if (!out) return cudaErrorInvalidValue;
  *out = nullptr;
  return boundary([&]() -> cudaError_t {
    if (!stream) return cudaErrorInvalidValue;
    auto s = std::make_unique<Scope>();
    s->stream = static_cast<cudaStream_t>(stream); s->device = device;
    s->thread = std::this_thread::get_id();
    auto e = snapshot(s->stream, s->prefix);
    if (e) { return e; }
    int current = -1; e = cudaGetDevice(&current);
    if (e || current != device) { return e ? e : cudaErrorInvalidDevice; }
    *out = s.release(); return cudaSuccess;
  });
}
extern "C" int apxinf_capture_frontier_rewind(void* raw) noexcept {
  return boundary([&]() -> cudaError_t {
    if (!raw) return cudaErrorInvalidValue;
    auto& s = *static_cast<Scope*>(raw);
    if (s.rewind_attempted || s.joined) return cudaErrorIllegalState;
    auto e = validate(s, s.routed); if (e) return e;
    s.rewind_attempted = true; // Failure is not assumed atomic.
    return set(s, s.prefix.nodes);
  });
}
extern "C" int apxinf_capture_frontier_join(void* raw) noexcept {
  return boundary([&]() -> cudaError_t {
    if (!raw) return cudaErrorInvalidValue;
    auto& s = *static_cast<Scope*>(raw);
    if (!s.rewind_attempted || s.joined) return cudaErrorIllegalState;
    Snapshot shared; auto e = validate(s, shared); if (e) return e;
    e = set(s, unite(s.routed, shared)); if (e) return e;
    Snapshot check; e = validate(s, check); if (e) return e;
    s.joined = true; return cudaSuccess;
  });
}
extern "C" int apxinf_capture_frontier_destroy(void* raw) noexcept {
  if (!raw) return cudaSuccess;
  auto* s = static_cast<Scope*>(raw);
  int result = boundary([&]() -> cudaError_t {
    if (s->rewind_attempted && !s->joined) {
      Snapshot current;
      auto e = validate(*s, current); if (e) return e;
      // Best effort only. Rust poisons the capture on every unfinished scope.
      return set(*s, unite(s->routed, current));
    }
    return cudaSuccess;
  });
  delete s; return result;
}
#else
// Older CUDA builds preserve default-off behavior without referencing the versioned capture ABI.
extern "C" int apxinf_capture_frontier_begin(void*, int, void** out) noexcept {
  if(out) *out = nullptr; return cudaErrorNotSupported;
}
extern "C" int apxinf_capture_frontier_rewind(void*) noexcept { return cudaErrorNotSupported; }
extern "C" int apxinf_capture_frontier_join(void*) noexcept { return cudaErrorNotSupported; }
extern "C" int apxinf_capture_frontier_destroy(void*) noexcept { return cudaSuccess; }
#endif

// Internal diagnostics called only by Rust unit tests; no graph handle is public.
extern "C" int apxinf_capture_frontier_test_edge(unsigned int offset, unsigned char value) noexcept {
#if CUDART_VERSION >= 12080
  cudaGraphEdgeData edge{};
  if (offset >= sizeof(edge)) return -1;
  reinterpret_cast<unsigned char*>(&edge)[offset] = value;
  return default_edge(edge);
#else
  return -1;
#endif
}
extern "C" int apxinf_capture_frontier_test_topology(void* raw, size_t* out) noexcept {
#if CUDART_VERSION >= 12080
  return boundary([&]() -> cudaError_t {
    if (!raw || !out) return cudaErrorInvalidValue;
    auto graph = static_cast<cudaGraph_t>(raw);
    size_t n = 0, edges = 0;
    auto e = cudaGraphGetNodes(graph, nullptr, &n); if (e) return e;
    e = cudaGraphGetEdges_v2(graph, nullptr, nullptr, nullptr, &edges); if (e) return e;
    std::vector<cudaGraphNode_t> nodes(n), from(edges), to(edges);
    std::vector<cudaGraphEdgeData> data(edges);
    e = cudaGraphGetNodes(graph, nodes.data(), &n); if (e) return e;
    e = cudaGraphGetEdges_v2(graph, from.data(), to.data(), data.data(), &edges); if (e) return e;
    out[0]=n;out[1]=edges;out[2]=out[3]=out[4]=0;
    for (auto node:nodes) {
      size_t in=std::count(to.begin(),to.end(),node), degree=std::count(from.begin(),from.end(),node);
      if (!in) ++out[2]; if (!degree) ++out[3]; out[4]=std::max(out[4],in);
    }
    return cudaSuccess;
  });
#else
  return cudaErrorNotSupported;
#endif
}

extern "C" int apxinf_capture_frontier_test_inject(void* raw,int kind) noexcept {
#if CUDART_VERSION >= 12080
  if (kind == 0) { inject_nondefault_edge = true; return cudaSuccess; }
  if (!raw) return cudaErrorInvalidValue;
  auto* scope=static_cast<Scope*>(raw);
  if(kind==1){++scope->prefix.id;return cudaSuccess;}
  if(kind==2){scope->prefix.graph=nullptr;return cudaSuccess;}
  return cudaErrorInvalidValue;
#else
  return cudaErrorNotSupported;
#endif
}
