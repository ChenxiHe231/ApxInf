# Ported CUDA operators

This directory contains project-owned physical kernels retained during the
direct cuda-new migration. They are compiled only by the current CUDA crate;
the retired legacy crate has been removed from the repository.

The companion translation units under `native/adapters/ported` retain the
existing stable C symbols used by the current Rust APIs. GEMM and
Attention are intentionally excluded from provider selection here: their
cuda-new registry/autotune implementations remain authoritative. New
single-provider families (`norm`, `pointwise`, `rope`, and `gather`) also use
their dedicated adapters; remaining ported symbols provide direct coverage for
operator families that do not yet have a dedicated provider.

Delete a ported symbol and its now-unused kernel when its Rust compatibility
wrapper has moved to a dedicated cuda-new family. Do not add new model-specific
policy to this directory.
