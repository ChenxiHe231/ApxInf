#include "internal.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <cmath>
#include <limits>

namespace apxinf::gemm {
namespace {

struct Events {
  cudaEvent_t start = nullptr;
  cudaEvent_t stop = nullptr;

  Events() {
    check_cuda(cudaEventCreate(&start));
    check_cuda(cudaEventCreate(&stop));
  }
  ~Events() {
    if (start != nullptr) cudaEventDestroy(start);
    if (stop != nullptr) cudaEventDestroy(stop);
  }
};

struct Allocation {
  void* pointer = nullptr;

  explicit Allocation(size_t bytes) { check_cuda(cudaMalloc(&pointer, bytes)); }
  ~Allocation() {
    if (pointer != nullptr) cudaFree(pointer);
  }
};

std::vector<float> read_output(void* pointer, size_t count, int dtype,
                               cudaStream_t stream) {
  std::vector<unsigned char> bytes(count * dtype_bytes(dtype));
  check_cuda(cudaMemcpyAsync(bytes.data(), pointer, bytes.size(),
                             cudaMemcpyDeviceToHost, stream));
  check_cuda(cudaStreamSynchronize(stream));
  std::vector<float> result(count);
  for (size_t index = 0; index < count; ++index) {
    if (dtype == APXINF_DTYPE_F32) {
      std::memcpy(&result[index], bytes.data() + 4 * index, 4);
    } else if (dtype == APXINF_DTYPE_F16) {
      half value;
      std::memcpy(&value, bytes.data() + 2 * index, 2);
      result[index] = __half2float(value);
    } else if (dtype == APXINF_DTYPE_BF16) {
      __nv_bfloat16 value;
      std::memcpy(&value, bytes.data() + 2 * index, 2);
      result[index] = __bfloat162float(value);
    } else {
      __nv_fp8_e4m3 value;
      std::memcpy(&value, bytes.data() + index, 1);
      result[index] = static_cast<float>(value);
    }
  }
  return result;
}

void poison(const apxinf_gemm_bindings_t& bindings, size_t bytes) {
  const auto stream = static_cast<cudaStream_t>(bindings.stream);
  check_cuda(cudaMemsetAsync(bindings.output, 0xff, bytes, stream));
  check_cuda(cudaStreamSynchronize(stream));
}

struct CapturedExecution {
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t executable = nullptr;
  cudaStream_t stream = nullptr;

  CapturedExecution(State& candidate,
                    const apxinf_gemm_bindings_t& bindings)
      : stream(static_cast<cudaStream_t>(bindings.stream)) {
    check_cuda(cudaStreamBeginCapture(stream,
                                      cudaStreamCaptureModeThreadLocal));
    try {
      check_cuda(candidate.implementation->launch(candidate, bindings));
    } catch (...) {
      cudaStreamEndCapture(stream, &graph);
      if (graph != nullptr) cudaGraphDestroy(graph);
      graph = nullptr;
      throw;
    }
    check_cuda(cudaStreamEndCapture(stream, &graph));
    const auto status =
        cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0);
    if (status != cudaSuccess) {
      cudaGraphDestroy(graph);
      graph = nullptr;
      check_cuda(status);
    }
  }

  ~CapturedExecution() {
    if (executable != nullptr) cudaGraphExecDestroy(executable);
    if (graph != nullptr) cudaGraphDestroy(graph);
  }

  void launch() { check_cuda(cudaGraphLaunch(executable, stream)); }
};

}  // namespace

std::shared_ptr<State> tune(
    const Spec& spec, const apxinf_gemm_policy_t& policy,
    const apxinf_gemm_tuning_bindings_t& tuning_bindings, int device,
    std::string& report, const Recipe* preferred) {
  const size_t count = static_cast<size_t>(
      spec.m * (spec.semantic == APXINF_GEMM_SEMANTIC_GEMM_GEGLU
                    ? spec.n / 2
                    : spec.n));
  Allocation output(count * dtype_bytes(spec.output_dtype));
  auto bindings = tuning_bindings.execution;
  bindings.output = output.pointer;
  const size_t output_bytes = count * dtype_bytes(spec.output_dtype);
  const ReferenceOutput reference = cpu_reference(spec, tuning_bindings);

  std::shared_ptr<State> winner;
  float best = std::numeric_limits<float>::infinity();
  Events events;
  int checked = 0;
  int rejected = 0;
  std::vector<std::string> diagnostics;
  const auto& implementations = registry(spec.semantic);
  struct Candidate {
    const Implementation* implementation;
    int configuration;
  };
  std::vector<Candidate> candidates;
  for (const auto& implementation : implementations) {
    if (!supports_device(implementation, device)) {
      diagnostics.push_back(std::string(implementation.name) +
                            "=skip(device)");
      continue;
    }
    if (!implementation.supports(spec)) {
      diagnostics.push_back(std::string(implementation.name) +
                            "=skip(contract)");
      continue;
    }
    if (!supports_alignment(implementation, spec)) {
      diagnostics.push_back(std::string(implementation.name) +
                            "=skip(alignment)");
      continue;
    }
    if ((policy.graph_safe || policy.execution_mode == 1) &&
        !implementation.graph_safe) {
      diagnostics.push_back(std::string(implementation.name) +
                            "=skip(graph-safe)");
      continue;
    }
    if (policy.deterministic && !implementation.deterministic) {
      diagnostics.push_back(std::string(implementation.name) +
                            "=skip(determinism)");
      continue;
    }
    std::vector<int> configurations;
    implementation.enumerate_configs(spec, configurations);
    for (int configuration : configurations) {
      candidates.push_back({&implementation, configuration});
    }
  }
  bool preferred_first = false;
  if (preferred != nullptr) {
    const auto match = [&](const Candidate& candidate) {
      return candidate.implementation->provider_id == preferred->provider_id &&
             candidate.implementation->implementation_id ==
                 preferred->implementation_id &&
             candidate.implementation->implementation_version ==
                 preferred->implementation_version &&
             candidate.configuration == preferred->configuration;
    };
    const auto found = std::find_if(candidates.begin(), candidates.end(), match);
    if (found != candidates.end()) {
      std::rotate(candidates.begin(), found, std::next(found));
      preferred_first = true;
    }
  }
  for (const auto& selected : candidates) {
    const auto& implementation = *selected.implementation;
    const int configuration = selected.configuration;
    try {
        auto candidate =
            prepare(implementation, configuration, spec, policy, device);
        if (implementation.bind_state != nullptr) {
          implementation.bind_state(*candidate, bindings);
        }
        poison(bindings, output_bytes);
        check_cuda(implementation.launch(*candidate, bindings));
        const auto actual = read_output(
            output.pointer, count, spec.output_dtype,
            static_cast<cudaStream_t>(bindings.stream));
        const AccuracyMetrics accuracy =
            compare_reference(reference.values, actual);
        ++checked;
        const std::string label = std::string(implementation.name) + "#" +
                                  std::to_string(configuration);
        if (!accuracy.valid) {
          ++rejected;
          diagnostics.push_back(label + "=reject(numeric:" +
                                format_accuracy(accuracy) + ")");
          continue;
        }
        std::unique_ptr<CapturedExecution> graph;
        if (policy.graph_safe || policy.execution_mode == 1) {
          // Capture and instantiation are preparation costs. They are kept
          // outside the timing interval even when replay is the target mode.
          graph = std::make_unique<CapturedExecution>(*candidate, bindings);
          poison(bindings, output_bytes);
          graph->launch();
          const auto graph_actual = read_output(
              output.pointer, count, spec.output_dtype,
              static_cast<cudaStream_t>(bindings.stream));
          if (!compare_reference(reference.values, graph_actual).valid) {
            throw Failure(APXINF_STATUS_PROVIDER_ERROR,
                          "CUDA Graph replay failed numeric validation");
          }
        }
        const auto launch_target = [&] {
          if (policy.execution_mode == 1) {
            graph->launch();
          } else {
            check_cuda(implementation.launch(*candidate, bindings));
          }
        };
        for (int iteration = 0; iteration < 3; ++iteration) {
          launch_target();
        }
        check_cuda(cudaEventRecord(events.start,
                                   static_cast<cudaStream_t>(bindings.stream)));
        for (int iteration = 0; iteration < 10; ++iteration) {
          launch_target();
        }
        check_cuda(cudaEventRecord(events.stop,
                                   static_cast<cudaStream_t>(bindings.stream)));
        check_cuda(cudaEventSynchronize(events.stop));
        float milliseconds = 0.0F;
        check_cuda(
            cudaEventElapsedTime(&milliseconds, events.start, events.stop));
        milliseconds /= 10.0F;
        diagnostics.push_back(label + "=pass(" + format_accuracy(accuracy) +
                              ")");
        if (milliseconds < best) {
          best = milliseconds;
          winner = candidate;
        }
    } catch (const Failure& failure) {
      ++rejected;
      diagnostics.push_back(std::string(implementation.name) + "#" +
                            std::to_string(configuration) + "=reject(" +
                            failure.what() + ")");
      cudaGetLastError();
    }
  }
  if (winner == nullptr) {
    throw Failure(APXINF_STATUS_UNSUPPORTED,
                  "no candidate satisfies the fixed FP32 reference contract");
  }
  report = "tuned preferred=" + std::to_string(preferred_first) +
           std::string(" mode=") +
           (policy.execution_mode == 1 ? "graph-replay" : "eager") +
           " reference=" + std::string(reference.kind) +
           " checked=" + std::to_string(checked) +
           " rejected=" + std::to_string(rejected) +
           " ms=" + std::to_string(best) + " candidates=[";
  for (size_t index = 0; index < diagnostics.size(); ++index) {
    if (index != 0) report += ",";
    report += diagnostics[index];
  }
  report += "]";
  return winner;
}

}  // namespace apxinf::gemm
