# AOT-to-native operator migration

This note records where the seven translated CUDA operators belong in
`apxinf-cuda-new`. It follows the operator classification and L3-to-L0 layering
in `cuda-new-design.md`: public APIs describe mathematical semantics, while
model names and fixed production shapes stay inside candidate support checks.

| Translated operator | cuda-new family | Public L3 entry | Shape | Status in this change |
|---|---|---|---|---|
| NVFP4 GEMM + SwiGLU + requantization | GEMM | `ops::nvfp4_gemm_swiglu` | Fixed | L3/L2/ABI/L1/L0; used by Qwen3.8-27B |
| BF16 GEMM + SwiGLU | GEMM | `ops::gemm_swiglu_pair_interleaved_256` | GEMM family candidate | L3/L2/ABI/L1/L0; used by Qwen-Drive prefill |
| BF16 gated-delta-net prefill | Linear attention | `ops::gdn_first_prefill` | Complete composite semantic | Fixed SM110 launch with internal generic decomposition |
| D256 causal attention | Attention | `ops::attention` | Attention family candidate | Registered under dense attention |
| D256 split-batch attention | Attention | `ops::kv_cache_attention` | Attention family candidate | Registered under KV-cache attention |
| Strided-group attention, 624 tokens | Attention | `ops::segmented_attention` | One shape-gated candidate | Registered under segmented attention |
| Strided-group attention, 2200 tokens | Attention | `ops::segmented_attention` | Same candidate as 624 | Shared native candidate used by Qwen-Drive |

The NVFP4 operator is exposed as a fixed GEMM-family API because it has a
distinct binding contract: packed NVFP4 operands, two scale outputs, and three
calibration scalars. The implementation accepts only its validated SM110 shape;
the model keeps the ordinary GEMM plus SwiGLU fallback for other row counts.

The other six translations use model-neutral GEMM, Attention, RoPE, and GDN
contracts. Qwen-Drive does not inspect the GPU, call provider-specific
`try_*` launchers, or observe a candidate miss. Fixed shapes stay inside native
`supports` predicates. Attention/GEMM fallbacks are registry entries; the GDN
L3 semantic owns its generic decomposition. The Qwen-Drive load path
materializes one candidate-independent pair-interleaved BF16 SwiGLU weight,
which both the fixed kernel and vendor fallbacks consume.

No offline AOT bundle, exporter, manifest, or runtime module loader remains.
`APXINF_CUDA_ARCH=sm_110` builds every translated provider from in-tree CUDA
and CUTLASS sources.
