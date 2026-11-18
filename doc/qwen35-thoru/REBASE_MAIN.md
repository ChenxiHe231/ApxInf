# Final branch on current main

The accepted source snapshot is `68b12f23be337d7334d9d7431598274aedfb7e5e`.
`Final` replays its 102 non-merge commits on upstream
`50d283e41abf1c081b2535194900aedbea0e33c6` (2026-11-18).
Historical benchmark evidence in this directory belongs to its original
revision and is not a performance measurement of the rebased binary.

## Compatibility changes

- Keep the current CPU, CUDA, MUSA, Metal, Core ML, image and speech contracts.
  The `cuda` feature includes the Qwen family; `qwen35-cuda` retains its
  independent, model-owned CUDA context and sampler.
- Keep partial `memset_async(value, bytes, stream)` distinct from the accepted
  whole-allocation `fill_async(byte, stream)` helper. Both retain stream ordering.
- Use the current block-quantization, packed-scale, argmax and output-gate
  argument types at Qwen call sites. Keep the accepted scalar GDN entry points
  in the private `qwen35_gdn_compat` module, alongside the public typed GDN API.
- Preserve the accepted direct-launch MLP and attention helpers required by
  Qwen. They do not replace upstream operator implementations.
- Separate Qwen kernel implementation/launch translation units under
  `native/kernels/qwen35/` from their existing C ABI adapters. The private
  implementation bodies, including all kernels and their constants, are
  unchanged from the accepted snapshot.
- Give public helpers semantic names (`fused_routed_moe`, `small_routed_moe`,
  `block_quantized_gemv`, `qwen35_attention_static_cache`) and retain precision
  in their argument and weight types. C ABI names and checkpoint formats stay
  unchanged.
- Keep INT8 SiLU quantization at semantic code 7; assign fixed-scale E4M3
  dequantization code 8 consistently in Rust and C.
- Partition shared cuBLAS handles by device, stream and numerics mode. Keep
  upstream's exact stable-attention recipe keys; flexible causal decode may
  reuse the Qwen 4K live-length buckets.

## Validation boundary

Use the repository architecture checks, CPU tests and a native CUDA build for
this revision. GPU output correctness and performance require a fresh model
run with the intended toolkit and frozen recipe. In particular, this rebase
is not evidence that the previously observed CUDA 13.2 NVFP4 output problem
has been fixed. The original CUDA 12 / SM101 admission rule for the Edge NVFP4
AOT kernels is preserved.

CPU verification on Thor-3 passed 286 tests across 17 suites, with six tests
ignored by their own settings and the two unrelated PI0.5 full-size synthetic
weight tests excluded. The benchmark-tool suite passed 88 tests and skipped
three. Both the CUDA architecture and model-family boundary checks passed.
