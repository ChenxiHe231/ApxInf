# CUDA operator primitives

This directory contains project-owned physical kernels used by cuda-new's
model-oriented and direct-launch adapters. Kernel files are named for their
operation; their launch policy stays in `native/adapters`.

The aggregate model-oriented translation units are named by dtype or operation
(`bf16_ops.cu`, `int8_ops.cu`, `mixed_precision_ops.cu`, and
`tensor_ops.cu`). GEMM and Attention keep their cuda-new registry/autotune
implementations; single-implementation families use their dedicated flat
adapters.

Delete an aggregate C symbol and its now-unused primitive when its caller has
moved to a dedicated cuda-new family. Do not add model-specific selection
policy to this directory.
