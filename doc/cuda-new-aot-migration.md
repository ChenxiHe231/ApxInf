# AOT-to-native operator migration

This note records where the seven translated CUDA operators belong in
`apxinf-cuda-new`. It follows the operator classification and L3-to-L0 layering
in `cuda-new-design.md`: public APIs describe mathematical semantics, while
model names and fixed production shapes stay inside candidate support checks.

| Translated operator | cuda-new family | Public L3 entry | Shape | Status in this change |
|---|---|---|---|---|
| NVFP4 GEMM + SwiGLU + requantization | GEMM | `ops::nvfp4_gemm_swiglu` | Fixed | L3/L2/ABI/L1/L0; used by Qwen3.8-27B |
| BF16 GEMM + SwiGLU | GEMM | `ops::bf16_swiglu_projection` | GEMM family candidate | Native L3/L1/L0; used by Qwen-Drive prefill |
| BF16 gated-delta-net prefill | Linear attention | `ops::fused_gdn_prefill_bf16` | Family provider | Native L3/L1/L0; used by Qwen-Drive first prefill |
| D256 causal attention | Attention | `ops::causal_gqa_bf16` | Attention family candidate | Native L3/L1/L0; used by Qwen-Drive language attention |
| D256 split-batch attention | Attention | `ops::noncausal_gqa_bf16` | Attention family candidate | Native L3/L1/L0; used by Qwen-Drive action attention |
| Strided-group attention, 624 tokens | Attention | `ops::vision_segmented_attention_bf16` | One shape-gated candidate | Native L3/L1/L0; used by Qwen-Drive vision attention |
| Strided-group attention, 2200 tokens | Attention | `ops::vision_segmented_attention_bf16` | Same candidate as 624 | Shared native candidate used by Qwen-Drive |

The NVFP4 operator is exposed as a fixed GEMM-family API because it has a
distinct binding contract: packed NVFP4 operands, two scale outputs, and three
calibration scalars. The implementation accepts only its validated SM110 shape;
the model keeps the ordinary GEMM plus SwiGLU fallback for other row counts.

The other six translations are exposed through model-neutral GEMM, attention,
and linear-attention contracts. Qwen-Drive calls only these `ops::*` entries;
it does not inspect the GPU or call provider-specific `try_*` launchers. Their
support guards retain the validated SM110 geometry inside cuda-new. Attention
and GEMM misses fall back to ordinary cuda-new operators in L3. Fused GDN
prefill returns `None` on a miss because its fallback is a larger stateful
composition owned by the model. The Qwen-Drive load path materializes one
packed BF16 SwiGLU weight that is shared by the fixed native prefill candidate
and the generic variable-row route.

No offline AOT bundle, exporter, manifest, or runtime module loader remains.
`APXINF_CUDA_ARCH=sm_110` builds every translated provider from in-tree CUDA
and CUTLASS sources.
