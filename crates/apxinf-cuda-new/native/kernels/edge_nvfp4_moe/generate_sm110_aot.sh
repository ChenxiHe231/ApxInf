#!/usr/bin/env bash
set -euo pipefail

# Generate the two TensorRT-Edge-LLM CuTeDSL modules used by ApxInf's
# Qwen3.5-35B-A3B routed MoE provider. Run this script on thor-u.
#
# Keep the process deliberately low-impact: CuTeDSL and all compiler children
# inherit one-core affinity and low scheduler/I/O priority.

edge_root=${EDGE_LLM_ROOT:-/opt/data/dev/TensorRT-Edge-LLM-main}
venv_root=${CUTE_DSL_VENV:-/opt/data/dev/cutedsl-4.7-cu12}
output_root=${AOT_OUTPUT_DIR:-/opt/data/dev/cutedsl-artifacts-qwen35-sm110}
kernel_root="$edge_root/kernelSrcs"
module_root="$kernel_root/nvfp4_moe_cutedsl"
python_bin="$venv_root/bin/python"
runtime_archive=$(find "$venv_root/lib" \
  -path '*/nvidia_cutlass_dsl/cu12/lib/libcuda_dialect_runtime_static.a' \
  -print -quit)

test -x "$python_bin"
test -f "$module_root/export_fc1_kernel.py"
test -f "$module_root/export_fc2_kernel.py"
test -f "$runtime_archive"

export CUTE_DSL_ARCH=sm_110a
export CUDA_MODULE_LOADING=LAZY
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export CMAKE_BUILD_PARALLEL_LEVEL=1
export MAX_JOBS=1
export NVCC_THREADS=1
export PYTHONPATH="$kernel_root${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$output_root/fc1" "$output_root/fc2" "$output_root/lib"

run_export() {
  label=$1
  shift
  log="$output_root/$label/generate.log"
  : >"$log"
  echo "[$(date -Is)] begin $label CUTE_DSL_ARCH=$CUTE_DSL_ARCH" | tee -a "$log"
  (
    cd "$kernel_root"
    exec taskset -c 0 nice -n 15 ionice -c 3 "$python_bin" "$@"
  ) 2>&1 | stdbuf -oL -eL tee -a "$log"
  sync "$log"
  echo "[$(date -Is)] done $label" | tee -a "$log"
  sync "$log"
}

run_export fc1 \
  nvfp4_moe_cutedsl/export_fc1_kernel.py \
  --activation swiglu \
  --mma_tiler_n 128 \
  --dummy-tokens 128 \
  --dummy-experts 256 \
  --dummy-top-k 8 \
  --dummy-hidden-size 2048 \
  --dummy-intermediate-size 512 \
  --output_dir "$output_root/fc1" \
  --file_name nvfp4_moe_sm110_fc1_swiglu_n128 \
  --function_prefix nvfp4_moe_sm110_fc1_swiglu_n128 \
  --export_only

run_export fc2 \
  nvfp4_moe_cutedsl/export_fc2_kernel.py \
  --mma_tiler_n 128 \
  --output_dtype bf16 \
  --dummy-tokens 128 \
  --dummy-experts 256 \
  --dummy-top-k 8 \
  --dummy-hidden-size 2048 \
  --dummy-intermediate-size 512 \
  --output_dir "$output_root/fc2" \
  --file_name nvfp4_moe_sm110_fc2_n128_bf16 \
  --function_prefix nvfp4_moe_sm110_fc2_n128_bf16 \
  --export_only

archive="$output_root/lib/libapxinf_edge_nvfp4_moe_sm110_cuda12.a"
rm -f "$archive"
# CuTeDSL AOT objects call the CUDA dialect shim (`_cudaLibraryLoadData`,
# `_cudaLaunchKernelEx`, ...). Seed the provider archive with the matching
# CUDA-12 shim so downstream consumers do not depend on the generator venv.
cp "$runtime_archive" "$archive"
ar rcs "$archive" \
  "$output_root/fc1/nvfp4_moe_sm110_fc1_swiglu_n128.o" \
  "$output_root/fc2/nvfp4_moe_sm110_fc2_n128_bf16.o"
sha256sum "$archive" | tee "$archive.sha256"
sync "$output_root"
