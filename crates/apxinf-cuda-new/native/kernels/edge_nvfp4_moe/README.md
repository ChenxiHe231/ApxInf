# TensorRT-Edge-LLM NVFP4 MoE provider

This directory integrates NVIDIA's Apache-2.0 CuTeDSL NVFP4 MoE kernels for
the Qwen3.5-35B-A3B geometry on Jetson Thor (SM110 hardware):

- FC1: grouped NVFP4 GEMM, fused SwiGLU and FP4 requantization;
- FC2: grouped NVFP4 GEMM, router scaling and token scatter/reduction;
- BF16 input and BF16 token-major output;
- fused decode layout/quantization setup and a general prefill setup.

Upstream source: `NVIDIA/TensorRT-Edge-LLM`, commit
`1231d6415ce38d0b2f20b3308ac24e58ec4100a8`. The raw-pointer adapter in
`runner.cu` is derived from:

- `cpp/kernels/moe/nvfp4_cutedsl/cuteDslNvfp4MoeSm110Runner.cpp`;
- `cpp/kernels/moe/fp4SupportKernels/buildLayout.cu`;
- `cpp/kernels/moe/fp4SupportKernels/fp4Quantize.cu`.

The project root `LICENSE` is Apache-2.0. Adapted source files retain NVIDIA's
copyright and SPDX notice.

## AOT artifact

`aot/aarch64-sm110-cuda12/libapxinf_edge_nvfp4_moe_sm110_cuda12.a` is
generated on thor-u from the upstream commit with `nvidia-cutlass-dsl==4.7.0`
and the CUDA-12 backend. It contains pure SM110 FC1 SwiGLU and BF16 FC2
modules from one invocation; the previously mixed FC1/FC2 artifact is
intentionally rejected.

CUDA 12.8 reports this Thor as compute capability 10.1. CuTeDSL 4.7 defines
`sm_101a = sm_110a` and its CUDA-12 backend records `-arch sm_101a` in both
objects. This is NVIDIA's pre-CUDA-13 Thor compatibility encoding, not a mix
of two architectures. A physical `sm_110` cubin requires CUDA 13 and a driver
that reports Thor as 11.0; this DRIVE OS image provides neither.

Run `generate_sm110_aot.sh` on thor-u. The script fixes `CUTE_DSL_ARCH=sm_110a`
and constrains every compiler child to one low-priority CPU core. Copy the two
generated headers into `generated/`, the archive into
`aot/aarch64-sm110-cuda12/`, and record its SHA-256 here before committing.

SHA-256:
`7486d16d00dcb155647fb569b8c7eab0eba60df47103666369264f156648176e`.

The archive also contains CuTeDSL's matching CUDA-12 dialect runtime shim, so
consumers only need the normal CUDA runtime/driver libraries at link time and
do not depend on the generation virtual environment.

The CUDA-12 build enables this provider only for a single `aarch64` `sm_101`
target, which is the capability returned by `cudaGetDeviceProperties` here.
All other targets compile a stable unsupported stub.

## Thor validation

Validated on thor-u with the real `Qwen3.5-35B-A3B-NVFP4` checkpoint. Native
compilation used one low-priority CPU core throughout.

- Synthetic constant-reference parity passed at 1 and 1024 tokens. Ten-run
  timings were 70.58 us and 922.44 us per call, respectively.
- Real layer-0 MoE at 1024 tokens passed finite/non-zero output checks and ran
  in 3.102 ms per iteration (330,079 input tokens/s, ten iterations).
- The complete 40-layer model passed a 1024-token prefill plus MTP/scalar token
  parity check without a device reset. Scalar TTFT was 266.9 ms (about 3,837
  input tokens/s) in the debug test build.
- The current MTP prompt path performs draft teacher forcing one row at a time:
  its 1024-token TTFT was 3.051 s. This is a runtime scheduling optimization
  target, not an FC1/FC2 AOT limitation.

## Weight layout

The provider consumes checkpoint NVFP4 bytes without dequantizing them, but
requires a one-time load-time repack:

- FC1 rows are interleaved in 64-row chunks: `up[0:64], gate[0:64], ...`;
- scale bytes are swizzled from `[M, K/16]` to
  `[ceil(M/128), ceil((K/16)/4), 32, 4, 4]`;
- FC2 weights remain row-major and FC2 scales use the same swizzle;
- FC1 gate/up share one alpha. If their ModelOpt `weight_scale_2` differs,
  the smaller projection's E4M3 scale bytes must be rescaled before packing.

The original correctness-first routed-MoE operator remains available as a
fallback; callers opt into this provider by using `edge_nvfp4_moe`.
