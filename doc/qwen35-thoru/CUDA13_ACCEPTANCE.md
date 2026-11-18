# CUDA 13 Thor validation

Baseline: Final `f534b034`. The repair is isolated from the accepted
`68b12f23` snapshot and its historical performance evidence.

CUDA 13 renamed Thor SM101 to SM110. The fused NVFP4 build admission and
runtime admission now accept both exact Thor spellings. The unchanged
CuTeDSL CUDA-12 compatibility archive has been exercised on CUDA 13.2 / SM110
against the independent quantized constant reference at 1 and 128 tokens.
The driver loads it successfully. Unrelated Blackwell architectures remain
excluded. The test now fails instead of silently skipping a missing provider
on Thor.

The rebase also left a conflicting GEMM baseline-count check and a second
designated baseline. Restore the upstream single-baseline invariant while
keeping the packed-format candidate available for selection. A small GEMM
numerical regression test covers the model's first failing operation.

The unvalidated host-routed NVFP4 fallback passes row-major scales to packed
GEMM and can return corrupted tokens. NVFP4 activation is now rejected before
upload on targets without the validated fused provider. BF16 activation keeps
the NVFP4 weights and the existing W4A16 reference path.

All full-model timings require frozen recipes, no online tuning/fallback,
recorded clocks and precision, the shared GPU lock, and repeated valid output.
Formal results are stored separately from this source checkout. Historical
16K=83.28 and 48K=67.54 tok/s remain references, not measurements of this repair.

CUDA reference: https://docs.nvidia.com/cuda/archive/13.1.2/cuda-features-archive/index.html
