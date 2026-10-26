#!/usr/bin/env bash
set -euo pipefail

# Generate the BF16 grouped-GEMM module used by ApxInf's Qwen3.5 MTP MoE.
# Run on thor-u. The source is copied to a temporary tree and patched from the
# official TensorRT-Edge-LLM v0.11.0 homogeneous-F16 kernel to BF16; upstream
# files are never modified.

edge_root=${EDGE_LLM_ROOT:-/opt/data/dev/TensorRT-Edge-LLM-v0.11.0}
venv_root=${CUTE_DSL_VENV:-/opt/data/dev/cutedsl-4.7-cu12}
output_root=${AOT_OUTPUT_DIR:-/opt/data/dev/cutedsl-artifacts-qwen35-bf16-sm110}
python_bin="$venv_root/bin/python"
runtime_archive=$(find "$venv_root/lib" \
  -path '*/nvidia_cutlass_dsl/cu12/lib/libcuda_dialect_runtime_static.a' \
  -print -quit)
source_root="$edge_root/kernelSrcs/f16_moe_cutedsl"
staging_root=$(mktemp -d /opt/data/dev/qwen35-bf16-moe-aot.XXXXXX)
trap 'rm -rf "$staging_root"' EXIT

test -x "$python_bin"
test -f "$source_root/export_grouped_gemm.py"
test -f "$source_root/grouped_gemm_blackwell.py"
test -f "$runtime_archive"
cp -R "$source_root/." "$staging_root/"

# The Blackwell implementation is datatype-generic except for its three
# public contract checks. Export BF16 pointers and make those checks explicit.
sed -i 's/cutlass\.Float16/cutlass.BFloat16/g' \
  "$staging_root/export_grouped_gemm.py" \
  "$staging_root/grouped_gemm_blackwell.py"

export CUTE_DSL_ARCH=sm_110a
export CUDA_MODULE_LOADING=LAZY
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export CMAKE_BUILD_PARALLEL_LEVEL=1
export MAX_JOBS=1
export NVCC_THREADS=1
export PYTHONPATH="$staging_root:$edge_root/kernelSrcs${PYTHONPATH:+:$PYTHONPATH}"

mkdir -p "$output_root/module" "$output_root/lib"
log="$output_root/module/generate.log"
: >"$log"
echo "[$(date -Is)] begin BF16 grouped GEMM CUTE_DSL_ARCH=$CUTE_DSL_ARCH" | tee -a "$log"
(
  cd "$staging_root"
  exec taskset -c 0 nice -n 15 ionice -c 3 "$python_bin" export_grouped_gemm.py \
    --family blackwell \
    --output_dir "$output_root/module" \
    --file_name edge_bf16_moe_sm110_grouped \
    --function_prefix edge_bf16_moe_sm110_grouped
) 2>&1 | stdbuf -oL -eL tee -a "$log"

archive="$output_root/lib/libapxinf_edge_bf16_moe_sm110_cuda12.a"
rm -f "$archive"
cp "$runtime_archive" "$archive"
ar rcs "$archive" "$output_root/module/edge_bf16_moe_sm110_grouped.o"
sha256sum "$archive" | tee "$archive.sha256"
sync "$output_root"

