# CUDA operator primitives

This directory contains the small project-owned CUDA helpers shared by the
typed operators in `native/adapters`. GEMM and Attention keep their registry
and autotune implementations; single-implementation families launch directly
from their dedicated adapters.

Do not add model-specific selection policy or compatibility aggregates here.
