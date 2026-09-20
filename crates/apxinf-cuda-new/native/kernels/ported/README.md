# Ported CUDA operators

This directory contains project-owned physical kernels moved from
`crates/apxinf-cuda/kernels/custom`. They are compiled only by cuda-new; the
old crate is not a build dependency.

The companion translation units under `native/adapters/ported` retain the
existing stable C symbols while Rust call sites move to cuda-new. GEMM and
Attention are intentionally excluded from provider selection here: their
cuda-new registry/autotune implementations remain authoritative. New
single-provider families (`norm`, `pointwise`, `rope`, and `gather`) also use
their dedicated adapters; the ported symbols exist only for old public API
coverage that has not yet been redirected.

Delete a ported symbol and its now-unused kernel when its Rust compatibility
wrapper has moved to a dedicated cuda-new family. Do not add new model-specific
policy to this directory.
