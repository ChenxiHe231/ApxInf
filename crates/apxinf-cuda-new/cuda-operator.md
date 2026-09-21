# CUDA L3 Operator Catalog

This document lists the model-independent L3 semantics currently exposed by `apxinf-cuda-new`. Model code must match the complete mathematical semantic and tensor contract, not only the operator name. If no interface matches completely, record an operator gap and follow [`doc/adding-new-kernels.md`](../../doc/adding-new-kernels.md).

This document describes only public contracts and does not promise a specific provider, candidate, or autotune winner. Every operator supports eager execution and CUDA Graph capture/replay. GEMM and Attention prepare reusable state through `ExecutionSession`; stateless families are captured directly and do not require a prepare cache entry.

## Shared Constraints

- Input, output, bias, and scale tensors must reside on the CUDA device of the current `CudaContext`.
- Tensors use contiguous row-major layout; output storage must not overlap read-only inputs.
- Shapes must be nonempty, and dimensions passed to the native layer must not exceed `i32::MAX`.
- Policy fields such as workspace, graph-safe, and deterministic affect only candidate eligibility and the recipe; they do not change the L3 mathematical semantic.
- `l3-operator` comments are machine-readable markers. A unit test compares them with the Rust semantic metadata registered by each operator family to ensure that every public semantic appears exactly once; a new family must be added to that set.

## Stateless Direct Families

Gather, Norm, Pointwise, Quantization, and RoPE each have one stateless, zero-workspace implementation per semantic. Their L3 call path is `normalize → native *_launch → asynchronous kernel submission`. They do not create a native heap execution, an `ExecutionKey`, a typed session-cache entry, or a prepared-sequence entry, and they do not synchronize. CUDA Graph captures their kernel launches directly.

Norm intentionally has no `NormArgs { semantic, Option<...>... }` union. Each mathematical contract has its own typed API, so required tensors are required fields and unrelated tensors cannot be supplied:

| Semantic | Rust API | Required bindings |
| --- | --- | --- |
| RMS norm | `rms_norm(ctx, RmsNormArgs)` | input, weight, normalized, eps |
| Layer norm | `layer_norm(ctx, LayerNormArgs)` | input, weight, bias, normalized, eps |
| Adaptive RMS norm | `adaptive_rms_norm(ctx, AdaptiveRmsNormArgs)` | input, norm style, normalized, eps |
| Bias + residual | `bias_residual(ctx, BiasResidualArgs)` | input, optional bias, residual, hidden |
| Bias + residual + RMS norm | `bias_residual_rms_norm(ctx, BiasResidualRmsNormArgs)` | input, optional bias, residual, weight, hidden, normalized, eps |
| Bias + residual + layer norm | `bias_residual_layer_norm(ctx, BiasResidualLayerNormArgs)` | input, optional bias, residual, weight, norm bias, hidden, normalized, eps |
| Adaptive gate + residual | `ada_gate_residual(ctx, AdaGateResidualArgs)` | input, residual, gate style, hidden |
| Adaptive gate + residual + RMS norm | `ada_gate_residual_rms_norm(ctx, AdaGateResidualRmsNormArgs)` | input, residual, norm style, gate style, hidden, normalized, eps |
| BF16 bias-then-residual | `bias_then_residual(ctx, BiasThenResidualArgs)` | input, optional bias, residual, hidden |

This lifecycle distinction changes dispatch and ownership only. L0 launch geometry, arithmetic, and rounding remain the migrated ApxInf CUDA implementation and must not be changed as part of a lifecycle refactor.

## Shared GEMM Contract

GEMM uses `A=[M,K]` and `B=[K,N]`. `alpha` applies to the projection, and the final result is divided by a finite positive `output_scale`. `projection(A,B)` interprets inputs according to the following quantization contract:

- `None`: A/B have the same dtype, which is neither E4M3 nor INT8.
- `Fp8UnitScale`: A/B are both E4M3 tensors with the expected scaling already applied.
- `Fp8`: A/B are both E4M3; FP32 `row_scales=[M]` and `channel_scales=[N]` dequantize rows of A and columns of B, respectively.
- `W8A8`: A/B are both INT8 and use FP32 row/channel scales with the same shapes; output is BF16, `K <= 131071`, and the mode applies only to `gemm` and `gemm_bias`.

Quantization occurs before the API call; the current L3 contract does not include dynamic quantization. At least one registered candidate must still support the concrete spec.

<!-- l3-operator:gemm -->
### `gemm`

| Item | Contract |
| --- | --- |
| Rust API | `ops::gemm(ctx, GemmArgs)` |
| Inputs | `A=[M,K]`, `B=[K,N]`; supports `None`, `Fp8UnitScale`, `Fp8`, and `W8A8` |
| Output | `Y=[M,N]` |
| Mathematical semantic | `Y = alpha * projection(A,B) / output_scale` |
| Constraints | W8A8 output must be BF16; `WeightVersion` may declare immutable weights and allow prepare to cache an internal prepacked copy |
| Reference test | `gemm_all_candidates_match_torch` |

<!-- l3-operator:gemm_bias -->
### `gemm_bias`

| Item | Contract |
| --- | --- |
| Rust API | `ops::gemm_bias(ctx, GemmBiasArgs { gemm, bias })` |
| Inputs | `A=[M,K]`, `B=[K,N]`, `bias=[N]`; supports all four GEMM quantization contracts |
| Output | `Y=[M,N]` |
| Mathematical semantic | `Y = (alpha * projection(A,B) + bias) / output_scale`, with bias broadcast across M |
| Constraints | bias dtype matches the projection dtype; W8A8 output must be BF16 |
| Reference test | `gemm_bias_all_candidates_match_torch` |

<!-- l3-operator:gemm_bias_gelu -->
### `gemm_bias_gelu`

| Item | Contract |
| --- | --- |
| Rust API | `ops::gemm_bias_gelu(ctx, GemmBiasGeluArgs { gemm, bias })` |
| Inputs | `A=[M,K]`, `B=[K,N]`, `bias=[N]`; supports `None`, `Fp8UnitScale`, and `Fp8` |
| Output | `Y=[M,N]` |
| Mathematical semantic | `Y = GELU_tanh(alpha * projection(A,B) + bias) / output_scale` |
| Constraints | bias dtype matches the projection dtype; GELU uses the Torch reference's tanh approximation; W8A8 is unsupported |
| Reference test | `gemm_bias_gelu_all_candidates_match_torch` |

<!-- l3-operator:gemm_geglu -->
### `gemm_geglu`

| Item | Contract |
| --- | --- |
| Rust API | `ops::gemm_geglu(ctx, GemmGegluArgs { gemm })` |
| Inputs | `A=[M,K]`, `B=[K,2N]`; the first N columns of B are `B_gate`, and the last N columns are `B_up`; supports `None` and `Fp8UnitScale` |
| Output | `Y=[M,N]` |
| Mathematical semantic | `Y = GELU_tanh(alpha*(A@B_gate)) * (alpha*(A@B_up)) / output_scale` |
| Constraints | The second dimension of B is even; FP8 with row/channel scales and W8A8 are unsupported; candidate-specific packing may occur only internally |
| Reference test | `gemm_geglu_all_candidates_match_torch` |

## Shared Attention Contract

Attention computes `softmax(mask(scale * (Q @ K^T))) @ V`. Q/K/V have the same dtype, and `scale` is finite and positive with a default of `1/sqrt(head_dim)`. Dense and KV-cache support MHA, GQA, and MQA and require `query_heads % kv_heads == 0`.

<!-- l3-operator:attention -->
### `attention`

| Item | Contract |
| --- | --- |
| Rust API | `ops::attention(ctx, AttentionArgs)` |
| Inputs | `Q=[B,Tq,Hq,D]`, `K/V=[B,Tk,Hkv,D]`; Q/K/V share F16 or BF16 dtype; mask is `None` or `Causal` |
| Output | `Y=[B,Tq,Hq,D]`; ordinary output uses the input dtype, and F16 input may also be written as E4M3 |
| Mathematical semantic | dense scaled dot-product attention |
| Constraints | Causal requires `Tk>=Tq`, with queries aligned to the final positions of the key sequence; ordinary output requires `output_scale=1`; E4M3 stores `round_to_e4m3(attention/output_scale)` |
| Reference test | `attention_all_candidates_match_reference` |

<!-- l3-operator:kv_cache_attention -->
### `kv_cache_attention`

| Item | Contract |
| --- | --- |
| Rust API | `ops::kv_cache_attention(ctx, KvCacheAttentionArgs)` |
| Inputs | `Q=[B,Tq,Hq,D]`, `K_cache/V_cache=[B,key_capacity,Hkv,D]`; all share F16 or BF16 dtype; mask is `None` or `Causal` |
| Output | `Y=[B,Tq,Hq,D]`, with the same dtype as the inputs |
| Mathematical semantic | scaled dot-product attention from the query to the first `valid_key_tokens` rows of the cache |
| Constraints | `0<valid_key_tokens<=key_capacity`; when causal, token i is at `query_start+i`, and `query_start+Tq<=valid_key_tokens` is required |
| Reference test | `kv_cache_attention_all_candidates_match_reference` |

<!-- l3-operator:segmented_attention -->
### `segmented_attention`

| Item | Contract |
| --- | --- |
| Rust API | `ops::segmented_attention(ctx, SegmentedAttentionArgs)` |
| Inputs | Q/K/V are all `[total_tokens,H,D]` with the same dtype (F16 or BF16); device U32 offsets match the contents of `host_offsets` |
| Output | `Y=[total_tokens,H,D]`, with the same dtype as the inputs |
| Mathematical semantic | independent non-causal self-attention for each segment of a packed token sequence |
| Constraints | offsets contain at least two elements, are monotonically nondecreasing, begin at 0, and end at `total_tokens`; empty segments are allowed; causal and differing Q/KV head counts are unsupported |
| Reference test | `segmented_attention_all_candidates_match_reference` |

## Testing Responsibilities

The catalog test ensures only that semantics are neither missing nor duplicated; it cannot validate the written contracts. A new L3 semantic must also add a semantic test to `src/ops/tests/l3_behavior.rs` and an independent-reference all-candidate numerical test to `src/ops/tests/precision/precision.rs`.

Tests must run through `crates/apxinf-cuda-new/test-new.sh`; a normal `cargo test` from the repository root does not automatically test this crate.
