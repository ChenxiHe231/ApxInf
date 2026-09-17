# Vendored: Flash-Attention 2 forward kernels

ApxInf compiles the BF16 head-dimension 128 and 256 forward instantiations and
the FP16 head-dimension 128 non-causal instantiation used by vision attention.
The repository-local raw-pointer wrapper in `../fa2.cu` owns parameter
marshalling. Split planning, provider selection, and other execution policy
belong outside this vendored tree.

## Sources

### flash_attn/
- Upstream:  https://github.com/Dao-AILab/flash-attention
- Tag:       v2.7.4.post1
- License:   BSD-3-Clause (see `flash_attn/LICENSE`)
- Subset:    `csrc/flash_attn/src/` headers + BF16 forward kernel
             instantiations. bwd, CK (AMD) path, Hopper FA3,
             alibi/rotary/dropout runtime paths (headers kept for
             compile; p_dropout=0 in inference means inert), and the
             torch-coupled `flash_api.cpp` pybind wrapper are excluded
             — ApxInf ships its own `fa2.cu` wrapper. The kernels reuse the
             CUTLASS headers already vendored by `apxinf-cuda-new`; no second
             CUTLASS copy is carried under this directory.

## PyTorch-free build boundary

`flash.h`, `philox_unpack.cuh`, and `flash_fwd_launch_template.h` retain their
upstream `v2.7.4.post1` contents. ApxInf does not depend on libtorch, so the
narrow ATen/C10 surface referenced by those files is implemented under the
sibling `../fa2_compat/` include root. Keeping compatibility code outside this
directory makes upstream provenance auditable by byte-for-byte comparison.

The compatibility layer is inference-only: dropout is disabled, and it is not
intended to emulate PyTorch beyond the types and checks required to instantiate
the forward kernels.

## Direct E4M3 exception

The guarded `APXINF_FA2_DIRECT_E4M3` output conversion in
`flash_fwd_kernel.h` is the sole ApxInf operator extension in the vendor
headers. Its output scale is carried through the upstream `softcap` field by a
dedicated ApxInf operator translation unit; the vendor ABI is not extended.

## Backporting upstream bugfixes

FA2 main line is in maintenance (each FA2 release is 5–50 LoC of
toolchain fixes, no algorithmic changes; big work goes to FA3 which
is SM90+ only).

Procedure when we want a fix from upstream:

```bash
# In /tmp, fetch upstream + diff
git clone https://github.com/Dao-AILab/flash-attention.git /tmp/fa-upstream
cd /tmp/fa-upstream
git log v2.7.4.post1..HEAD -- csrc/flash_attn/src/

# For each relevant commit, generate a patch and store it under native/patches
git format-patch -1 <SHA> --stdout > /tmp/fa-fix.patch
# verify: rebuild + cos test
```

Do not edit the vendored file in place. Record the upstream revision and patch
purpose in `native/patches/README.md`, then re-run the all-candidate precision
test on every compiled CUDA architecture.
