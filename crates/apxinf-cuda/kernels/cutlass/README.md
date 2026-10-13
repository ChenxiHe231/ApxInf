# CUTLASS and FMHA kernel sources

This directory contains ApxInf's static-inference CUTLASS C++ operators and the
pinned headers they compile against. Stable C ABI shims live separately under
`crates/apxinf-cuda/adapters/`; `crates/apxinf-cuda/build.rs` compiles those
adapters against only these repository-local operator sources and headers.

## Source snapshot

Except for the explicitly pinned FMHA headers below, the contents were imported
without modification from an upstream snapshot at commit
`7434950d5b3a73dcbf810a797e772cebdb869598`:

- `include/` from `3rdparty/cutlass/include/`
- `tools/util/include/` from
  `3rdparty/cutlass/tools/util/include/`
- `fmha/` from `csrc/kernels/nvidia/attention/fmha/`
- `extensions/` from
  `csrc/kernels/nvidia/cutlass/cutlass_extensions/`; only the three headers
  required by the SM80-family W8A8 scale epilogue are included

The CUTLASS version header identifies release 4.3.0. ApxInf intentionally pins
this complete header set because the SM101/SM110 FMHA templates and the local
operators must be compiled against a mutually compatible CUTLASS snapshot.

These files are unmodified NVIDIA CUTLASS `v4.4.2` example-77 sources:

- `fmha/collective/sm100_fmha_fwd_mainloop_tma_warpspecialized.hpp`, SHA-256
  `9cceff09555a9ac4b379cbfaa92fe6b76372b4b6c65f9bca0faabd699143891f`
- `fmha/kernel/sm100_fmha_fwd_kernel_tma_warpspecialized.hpp`, SHA-256
  `8d7289812f7513c527987291125c820bb1385233780fe2dc0cf6b6e369cbbdbc`

Keeping the digests here makes accidental local algorithm patches visible.

## Deliberate local patches

Exactly one header carries an ApxInf modification. It is marked in place with an
`APXINF local patch` comment so a diff against upstream explains itself.

### `include/cutlass/arch/reg_reconfig.h` — add SM110 to `CUDA_CTA_RECONFIG_ACTIVATED`

Upstream gates the macro on `__CUDA_ARCH__` 900/1000/1010/1030/1200/1210 and
omits 1100 (Thor). Without 1100 every `warpgroup_reg_alloc` /
`warpgroup_reg_dealloc` compiles to nothing on SM110, so a kernel that asks
ptxas for a different per-warpgroup register budget silently does not get one.
The added clause is:

```c
|| (__CUDA_ARCH__ == 1100 && defined(__CUDA_ARCH_FEAT_SM110_ALL))
```

**Hard prerequisite:** the consuming translation unit must be compiled with
`--generate-code arch=compute_110a,code=sm_110a`, never `-arch=sm_110a`. Under
`-arch=` nvcc lowers to an internal `*.compute_110.ptx` whose `.target` has lost
the `a`, and ptxas then rejects `setmaxnreg.inc` outright, so the patch becomes
a build failure rather than a no-op. `scripts/check_kernel_architecture.sh`
enforces the flag spelling in the build scripts.

The byte-identical copy at
`crates/apxinf-cuda-new/native/kernels/cutlass/include/cutlass/arch/reg_reconfig.h`
carries the same patch; keep the two in sync.

| | SHA-256 |
|---|---|
| upstream | `02d929c9a49456c2648c98a22b01e81f01e39e4ff436c4f55eb53f94f67a97f9` |
| patched (both copies) | `413d18285ab6d36f255ec37ee9c853e959814433bd4f3f6c9edf29325c015e0b` |

The third copy, `fa2/cutlass/include/cutlass/arch/reg_reconfig.h`, comes from an
older CUTLASS revision with a different gate and is deliberately **not**
patched: its only consumers are the vendored Flash-Attention 2 families, none of
which issue a register reconfiguration.

## Compile flags that are part of the contract

`-DNDEBUG` is applied to every translation unit that includes these headers.
`nvcc -O3` does not imply it, and without it each `CUTLASS_ASSERT` survives into
device code: measured 124 `BPT.TRAP` instructions in
`adapters/cutlass_fmha_adapter.cu`, down to 3 with the flag. The residual three
are `cute/config.hpp`'s `CUTE_INVALID_CONTROL_PATH`, whose `printf` and
`__brkpt()` are not assert-guarded; they sit on dead paths after an
unconditional `RET`/`BRA`.

`--use_fast_math` must never reach these translation units.


## Attention integration policy

Existing BF16 dispatch remains unchanged; configurations that select NVIDIA
FMHA now use its official mainloop. The production FP16 vision shape uses the
separately vendored, upstream Flash-Attention 2 `v2.7.4.post1` kernels
documented in `fa2/VENDOR.md`, including packed QKV through an ApxInf-owned
raw-pointer adapter. This avoids modifying the NVIDIA mainloop to rescale
softmax probabilities before its probability-storage conversion. The official
mainloop's FP16 narrowing has unacceptable error for that shape, while FA2
preserves accuracy and was faster in the Thor comparison.

## Licensing

The imported CUTLASS and FMHA headers retain their original NVIDIA copyright
and BSD-3-Clause notices. The full CUTLASS license is preserved under
`licenses/`. The imported scale-epilogue extension headers retain their SGLang
copyright and Apache-2.0 license text.

ApxInf-specific C++ operators are `fp8_gemm_sm100.cu` and
`fmha_sm100.cu` for SM100-family devices and
`w8a8_gemm_sm80.cu` for SM80-family devices. They do not export
`extern "C"`; the corresponding shims are in `adapters/`.

### The three FA4 attention operators

`fmha_vision_sm100.cu`, `fmha_d256_language_sm100.cu` and
`fmha_d256_action_sm100.cu` (sharing `fmha_d256_sm100.cuh`) are SM110-only
instantiations of the vendored NVIDIA FMHA collectives above, configured to
reproduce FlashAttention-4's numerics bit-for-bit. The numerics they reproduce
come from upstream FlashAttention's `flash_fwd_sm100.py` (vision, action) and
`sm100_hd256_2cta_fmha_forward.py` (language), which are **BSD-3-Clause**, the
same licence as the CUTLASS/FMHA headers whose notices they carry; the full
text is under `licenses/CUTLASS-BSD-3-Clause.txt`. No upstream FlashAttention
source file is vendored for these three -- only its arithmetic is reproduced,
behind `-DAPXINF_*` switches whose defaults are all UPSTREAM CUTLASS, which is
what keeps `fmha_sm100.cu` byte-identical to its unconfigured self.

`include/cutlass/gemm/collective/apxinf_tma_dummy_shape.hpp` is an additive
ApxInf trait whose primary template is upstream's own expression; it is
specialised only by `fmha_d256_action_sm100.cu`, for the hierarchical pack-GQA
M-mode stride that cannot be typed from an all-dynamic dummy shape.
