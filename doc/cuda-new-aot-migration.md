# AOT-to-native operator migration

This note records where the seven translated CUDA operators belong in
`apxinf-cuda-new`. It follows the operator classification and L3-to-L0 layering
in `cuda-new-design.md`: public APIs describe mathematical semantics, while
model names and fixed production shapes stay inside candidate support checks.

| Translated operator | cuda-new family | Shape | Status in this change |
|---|---|---|---|
| NVFP4 GEMM + SwiGLU + requantization | GEMM | Fixed | L3/L2/ABI/L1/L0; used by Qwen3.8-27B |
| BF16 GEMM + SwiGLU | GEMM | GEMM family candidate | Native L3/L1/L0; used by Qwen-Drive prefill |
| BF16 gated-delta-net prefill | Linear attention | Family provider | Native L3/L1/L0; used by Qwen-Drive first prefill |
| D256 causal attention | Attention | Attention family candidate | Native L3/L1/L0; used by Qwen-Drive language attention |
| D256 split-batch attention | Attention | Attention family candidate | Native L3/L1/L0; used by Qwen-Drive action attention |
| Strided-group attention, 624 tokens | Attention | One shape-gated candidate | Native L3/L1/L0; used by Qwen-Drive vision attention |
| Strided-group attention, 2200 tokens | Attention | Same candidate as 624 | Shared native candidate used by Qwen-Drive |

The NVFP4 operator is exposed as a fixed GEMM-family API because it has a
distinct binding contract: packed NVFP4 operands, two scale outputs, and three
calibration scalars. The implementation accepts only its validated SM110 shape;
the model keeps the ordinary GEMM plus SwiGLU fallback for other row counts.

The other six translations are exposed through model-neutral GEMM, attention,
and linear-attention contracts. Their support guards retain the validated
SM110 geometry; misses fall back to the ordinary cuda-new operators. The
Qwen-Drive load path materializes one packed BF16 SwiGLU weight that is shared
by the fixed native prefill candidate and the generic variable-row route.

No offline AOT bundle, exporter, manifest, or runtime module loader remains.
`APXINF_CUDA_ARCH=sm_110` builds every translated provider from in-tree CUDA
and CUTLASS sources.
