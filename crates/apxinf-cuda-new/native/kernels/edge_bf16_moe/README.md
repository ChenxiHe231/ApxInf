# TensorRT-Edge-LLM BF16 grouped MoE provider

This provider adapts NVIDIA TensorRT-Edge-LLM v0.11.0's Apache-2.0
homogeneous-F16 grouped-GEMM kernel to BF16 for the Qwen3.5-35B-A3B MTP
layer on Jetson Thor.

Upstream source: `NVIDIA/TensorRT-Edge-LLM`, tag `v0.11.0`, commit
`95515c2f87fba8982db5a519f9022277667b3cc9`:

- `kernelSrcs/f16_moe_cutedsl/export_grouped_gemm.py`;
- `kernelSrcs/f16_moe_cutedsl/grouped_gemm_blackwell.py`.

The generator copies those files into a temporary directory and changes the
three operand pointer types and matching public type checks from FP16 to BF16.
The grouped kernel already uses FP32 accumulation and otherwise derives its
MMA, shared-memory, and epilogue types from those operands. This keeps the MTP
checkpoint in BF16 and preserves the existing GEMM rounding boundary.

The runtime specializes the general expert-major algorithm for decode's one
token and eight selected experts: one device kernel builds eight grouped-GEMM
descriptors directly from the token-major expert IDs, FC1 and FC2 run through
the AOT module, and device kernels apply SwiGLU plus the weighted reduction.
No routing metadata or activation crosses the host, and enqueue performs no
allocation or synchronization.

Run `generate_sm110_aot.sh` on thor-u with CuTeDSL 4.7 and the CUDA-12 backend.
Copy the generated header into `generated/` and the archive into
`aot/aarch64-sm110-cuda12/`, then record the SHA-256 here.

The checked-in artifact was generated on an aarch64 Jetson Thor with CuTeDSL
4.7's CUDA-12 backend and `CUTE_DSL_ARCH=sm_110a`.

SHA-256:
`f2fbcd381e968069ddcb1c4a45e0e3ecdd6a4f42bc296f0ff2d2c69ea8259ed5`.

