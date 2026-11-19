#!/usr/bin/env bash
# Early Thor-U operator gate for shape-range Recipe reuse and service fallback.
# Passing is not a long-Prefill chunk, full-model, or AIOS benchmark.
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 CHECKPOINT_DIR RECIPE_DIR CARGO_TARGET_DIR" >&2
  exit 2
fi

checkpoint_dir=$(realpath "$1")
recipe_dir=$(realpath "$2")
target_dir=$(realpath -m "$3")
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
[[ -f "$checkpoint_dir/config.json" ]] || { echo "missing checkpoint config.json" >&2; exit 2; }
[[ -d "$recipe_dir" ]] || { echo "missing recipe directory" >&2; exit 2; }
mkdir -p "$target_dir"
export CARGO_TARGET_DIR="$target_dir"
export PATH="/usr/local/cuda/bin:$HOME/.cargo/bin:$PATH"
export APXINF_REQUIRE_RECIPES=0

command -v nvcc >/dev/null || { echo "nvcc not found" >&2; exit 2; }
command -v cargo >/dev/null || { echo "cargo not found" >&2; exit 2; }
command -v flock >/dev/null || { echo "flock not found" >&2; exit 2; }

log_dir="$target_dir/prefill-range-validation-$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$log_dir"
cd "$repo_root"
{
  printf 'HEAD='; git rev-parse HEAD
  printf 'source_status_begin\n'; git status --short; printf 'source_status_end\n'
  printf 'checkpoint=%s\nrecipe_dir=%s\ntarget_dir=%s\n' \
    "$checkpoint_dir" "$recipe_dir" "$target_dir"
  nvcc --version | tail -4
  if command -v nvidia-smi >/dev/null; then
    nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader
  else
    cat /proc/device-tree/model
  fi
} > "$log_dir/identity.txt"

# All GPU work, including test-controlled donor tuning, uses the shared lock.
exec 9>/tmp/apxinf-gpu.lock
flock -n 9 || { echo "Thor-U GPU is busy; retry after the current owner finishes" >&2; exit 75; }

run_logged() {
  local label=$1
  shift
  echo "[$label] $*"
  "$@" 2>&1 | tee "$log_dir/$label.log"
}

for test_name in \
  gpu_e2e_fp8_recipe_range_prepares_actual_odd_m \
  gpu_e2e_fp8_recipe_range_handles_reported_gdn_shape \
  gpu_e2e_fp8_serving_fallback_covers_length_boundaries \
  gpu_e2e_fp8_strict_mode_rejects_range_recipe \
  gpu_e2e_attention_prefill_range_reprepares_for_actual_q_and_k \
  gpu_e2e_attention_strict_mode_rejects_range_recipe \
  gpu_e2e_kv_suffix_causal_mask_matches_production_gqa_geometry; do
  run_logged "$test_name" cargo test --locked -p apxinf-cuda-new --lib "$test_name" -- \
    --nocapture --test-threads=1
done

echo "Recipe range/fallback operator gate passed; long Prefill chunk, full model, AIOS and formal performance remain unverified; logs: $log_dir"
