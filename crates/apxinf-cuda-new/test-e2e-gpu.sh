#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

# These tests execute real CUDA kernels.  Run serially because CUDA stream
# capture is deliberately exercised and concurrent crate tests can interfere
# with capture diagnostics on the same device.
"$script_dir/test-new.sh" test -p apxinf-cuda \
  ops::tests::gpu_e2e_ -- --nocapture --test-threads=1
