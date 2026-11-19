#!/usr/bin/env bash
# Serve the validated range-Recipe/fallback path with the original model flow.
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 RELEASE_APXINF CHECKPOINT_DIR RECIPE_DIR" >&2
  exit 2
fi

server=$(realpath "$1")
model=$(realpath "$2")
recipes=$(realpath "$3")
[[ -x "$server" && -f "$model/config.json" && -d "$recipes" ]] || {
  echo "missing server, checkpoint, or recipe directory" >&2
  exit 2
}
if ldd "$server" | grep -q 'not found'; then
  ldd "$server" >&2
  exit 2
fi

export APXINF_AIOS_TIMELINE=1 APXINF_TOOL_CALL_LENIENT=1
export APXINF_QWEN35_CUDA_GRAPHS=1 APXINF_QWEN35_GRAPH_GDN=full
export APXINF_NVFP4_A16_WMMA=1 APXINF_NVFP4_A16_WMMA_TILE_N=128
export APXINF_NVFP4_A16_WMMA_MIN_ROWS=4 APXINF_NVFP4_A16_M1_STAGE_ALL=1
export APXINF_FA2_PACKED_GQA=0 APXINF_QWEN35_W4A16_SHARED=0
export APXINF_QWEN35_MTP_FIXED_DEPTH=3 APXINF_NVTX=0
export APXINF_QWEN35_MTP_SPLIT_KV=1
export APXINF_QWEN35_MTP_DRAFT_KV_SUFFIX_SYNC=1
export APXINF_QWEN35_MTP_KV_ONLY_MATERIALIZE=1
# Whole-model token chunks failed the frozen long-output and state gates.
# Explicitly disable both experimental switches for this serving script.
export APXINF_QWEN35_CHUNK_LONG_PREFILL=0
export APXINF_QWEN35_GDN_SEGMENT_LONG_PREFILL=0
export APXINF_QWEN35_GDN_PROJECTION_CHUNKS=0
export APXINF_REQUIRE_RECIPES=0

# Hold the shared GPU lock for the lifetime of the service.
exec 9>/tmp/apxinf-gpu.lock
flock -n 9 || { echo "GPU busy; retry after the current test finishes" >&2; exit 75; }
export PATH=/usr/local/cuda/bin:$HOME/.cargo/bin:$PATH
exec "$server" serve --model "$model" --model-name qwen3_5_moe \
  --served-model-name qwen3.6-35b-a3b --device cuda:0 --dtype bf16 \
  --qwen35-moe-activation nvfp4 --qwen35-kv-cache fp8_dynamic \
  --host "${HOST_IP:-0.0.0.0}" --port "${PORT:-48119}" --queue-capacity 16 \
  --max-input-tokens "${MAX_INPUT_TOKENS:-20000}" \
  --max-output-tokens "${MAX_OUTPUT_TOKENS:-4096}" \
  --enable-context-reuse --tuning-dir "$recipes"
