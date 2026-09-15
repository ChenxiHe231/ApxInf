# Vendored: Flash-Attention 2 forward kernels

ApxInf compiles the BF16 head-dimension 128 and 256 forward instantiations.
The 128 specialization handles dimensions up to 128; the 256 specialization
handles larger dimensions. Both causal and non-causal execution use the
repository-local raw-pointer wrapper in `../fa2.cu`. FP16, split-KV, backward,
and other instantiations are intentionally omitted from this operator package.

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

## Local patches

Three inherited PyTorch-decoupling patches in `flash_attn/` allow FA2 to build
without a Torch installation:

1. `flash_fwd_launch_template.h`: replaced `#include <c10/cuda/CUDAException.h>`
   with inline CUDA-runtime stubs for `C10_CUDA_CHECK` and
   `C10_CUDA_KERNEL_LAUNCH_CHECK`.

2. `flash.h`: replaced `#include <ATen/cuda/CUDAGeneratorImpl.h>` with a
   minimal POD `at::PhiloxCudaState` struct (dropout RNG state is
   carried through but never read in inference because `p_dropout=0`).

3. `philox_unpack.cuh`: replaced `#include <ATen/cuda/detail/UnpackRaw.cuh>`
   with a ~10-line inline stub for `at::cuda::philox::unpack()`.

Total inherited patch footprint: approximately 30 lines.

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

# For each relevant commit, generate a patch and apply here
git format-patch -1 <SHA> --stdout > /tmp/fa-fix.patch
cd <apxinf>/crates/apxinf-cuda-new/native/kernels/fa2/flash_attn
patch -p4 < /tmp/fa-fix.patch   # strip leading csrc/flash_attn/src/
# verify: rebuild + cos test
```

Record each applied fix in the commit log on this path and re-run the
all-candidate precision test on every compiled CUDA architecture.
