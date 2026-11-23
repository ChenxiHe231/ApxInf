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

## Decode metadata prefix candidate

`APXINF_MOE_SETUP_PREFIX=1` opts into an integer metadata prefix in the
small-batch setup kernel, default off. Initial support is BF16 input,256 experts,
H2048,I512,top8 and exactly1 or4 tokens on this existing Thor provider. Other
shapes and flag values retain the original path. The original setup kernel,
quantize_row, checkpoint scales/layout, FC1/FC2 AOT and persistent caps are
unchanged. The new kernel shares the original quantization function; it only
replaces the serial expert-offset/tile scan with integer ballot/popcount/prefix.
Within-expert atomic scatter keeps its existing unordered physical-row contract.

Set the flag before model/Graph creation in a fresh worker. Captured Graph nodes
retain the selected kernel after environment changes. The aligned benchmark
worker admits this flag and echoes its effective boolean in `ready`; benchmark
controllers that clear `APXINF_` environment must explicitly restore0/1 before
launch. The product L3 comparison passed72 exact numerical records. On Thor2,
two public short-input cases (62/63 tokens,max-output512,warmup10,three repeats)
showed positive same-ELF off/on results in both pair directions: pooled model
end-to-end throughput +2.89% Vanilla and +1.22% MTP, with exact original-V4 output
parity. These are affected-case results, not a complete80-question result or a
claim of exceeding the published TensorRT value.

A paired Nsight trace verified5080 original/prefix setup calls in127 decode steps,
unchanged other kernel and runtime-launch counts, and diagnostic setup median
11.296us to3.200us. Profile timing is not the benchmark score. Numerical and model
runs used frozen worker SHA25673ce4ff0ac25c51467f384b07eb69b43111a015dd237491a6b63cc8ea5c3275f
with baseline forward8dd5ecfa and runner610aa0a4; this was a recorded dirty-source
build, not a clean rebuild of this commit. Full80 validation and a clean revision
build remain pending. Experiment commands, recipes, receipts and raw paths are
recorded under the local feature directory devlocal/trt-plus20/.

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

The provider is enabled for a single `aarch64` Thor target: `sm_101` on
CUDA 12 or its renamed `sm_110` on CUDA 13. The same compatibility archive
has passed independent numerical checks and full-model smoke generation on
Thor-3 / CUDA 13.2, including CUDA Graph. Its bytes and SHA-256 are unchanged.
All unrelated targets compile a stable unsupported stub. See
[CUDA 13 validation](../../../../../doc/qwen35-thoru/CUDA13_ACCEPTANCE.md).

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
