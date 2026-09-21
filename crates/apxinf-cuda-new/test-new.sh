#!/usr/bin/env bash
# Run the complete apxinf-cuda-new test suite from the repository root:
# bash crates/apxinf-cuda-new/test-new.sh \
#   test -p apxinf-cuda -- --nocapture --test-threads=1

set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd -- "$script_dir/../.." && pwd)"
cd "$repo_root"
export PATH="$HOME/.cargo/bin:/usr/local/cuda/bin:$PATH"
: "${CUDA_PATH:=/usr/local/cuda}"
: "${APXINF_CUDA_ARCH:=sm_110}"
: "${CARGO_TARGET_DIR:=$repo_root/target-gemm-pilot}"
export CUDA_PATH APXINF_CUDA_ARCH CARGO_TARGET_DIR
cargo "$@"
