# Qwen3.6-35B-A3B-NVFP4 on Thor-U

This guide covers the `Qwen3.6-35B-A3B-NVFP4` checkpoint installed on Thor-U.
Its Hugging Face architecture metadata remains `qwen3_5_moe`, so the internal
Rust module and registry name continue to use `qwen35`; that name must not be
interpreted as support for the older `txn545/Qwen3.5-35B-A3B-NVFP4` checkpoint.
The guide includes fast builds,
the fixed 4,096-token benchmark, precision comparisons, text generation and
the OpenAI-compatible server.

> **Historical observations; current acceptance uses the model-microbench gates.**
>
> Historical cache-warm 4,096-token “FP8 prefill” runs under the 1206 MHz
> cap recorded 1.02–1.06 s TTFT (3,874–4,008 input tokens/s). These are historical
> shape-specific observations with their own revision/configuration, not the
> current 15-case matrix or evidence that every previous host failure had one
> clock-related cause. Keep clock-policy observations separate from the OOM
> incident in [MODEL_MICROBENCH_RECOVERY.md](MODEL_MICROBENCH_RECOVERY.md).
> Reapply and read back the baseline clock policy after every reboot.
>
> Earlier scalar repeat divergence prompted investigation of attention split
> selection. Attention autotuning has not been established as its sole cause;
> current precision checks also retain unresolved repeatability evidence.
> See [MODEL_MICROBENCH_PRECISION.md](MODEL_MICROBENCH_PRECISION.md) and
> [MODEL_MICROBENCH_USAGE.md](MODEL_MICROBENCH_USAGE.md) for current acceptance.

## 1. Connect and prepare the environment

Connect to Thor-U directly and enter this branch's checkout:

```bash
ssh thor-u
cd /opt/data/dev/apxinf-qwen35-thoru-perf
```

### Required build/inference memory split

Thor-U uses system RAM for both host processes and CUDA allocations, while the
CUDA runtime draws from the reserved HugeTLB pool. Never compile CUDA code with
the inference-sized pool still reserved: parallel `cicc` processes can exhaust
the remaining ordinary memory and trigger the system OOM killer.

Use the guarded build phase described in
[MODEL_MICROBENCH_USAGE.md](MODEL_MICROBENCH_USAGE.md). It holds the shared GPU
lock, refuses to shrink an in-use pool, retains **2048 HugeTLB pages (4 GiB)**,
requires at least 24 GiB of ordinary available memory, and limits both Cargo
and the native CUDA worker pool to two jobs. **Do not set the pool to zero**:
on this DriveOS installation the CUDA context failed with zero pages and
initialized with 2048 pages in a controlled check.

Finish all native builds before allocating the inference pool. An optional
`--inference-pages 20480` on the last successful guarded build requests 40 GiB,
checks the actual allocation, and retains at least 8 GiB of ordinary available
memory. A failed or partial request rolls back to the small build pool; it does
not start inference. Recheck the GPU clock policy after every reboot.

The [recovery record](MODEL_MICROBENCH_RECOVERY.md) distinguishes the observed
OOM incident from earlier clock-related failures and documents recovery and
prevention. Do not stop another user's processes to reclaim memory.

Set the remaining build and checkpoint paths once per shell:

```bash
export PATH=/usr/local/cuda/bin:$HOME/.cargo/bin:$PATH
export CARGO_TARGET_DIR=/opt/data/dev/apxinf-qwen35-mtp-serving/target
export QWEN35_CHECKPOINT=/opt/data/models/Qwen3.6-35B-A3B-NVFP4
```

The branch caches native CUDA objects by command line and dependency
fingerprint, and the `qwen35-cuda` feature avoids compiling the unrelated
legacy CUDA runtime. Incremental builds therefore reuse unchanged CUDA objects;
a cold build uses at most two native workers. Do not raise either job limit on
Thor-U even when more CPU cores are idle.

The checkpoint directory name is not used for dispatch. Its `config.json`
must identify `qwen3_5_moe` / `qwen3_5_moe_text` and use ModelOpt
`MIXED_PRECISION`: static FP8 main-model attention/GDN projections, NVFP4
MoE and `lm_head`, and BF16 MTP tensors. The old all-NVFP4/BF16-mixer
`txn545/Qwen3.5-35B-A3B-NVFP4` layout is incompatible.

## 2. Build

Use the guarded release build for the current model matrix. It builds both
benchmark examples and records the actual ELF identities:

```bash
python3 scripts/build_qwen35_thor_benchmarks.py --plan --target-dir "$CARGO_TARGET_DIR"
python3 scripts/build_qwen35_thor_benchmarks.py --execute --target-dir "$CARGO_TARGET_DIR"
```

The guard targets these two examples. CLI/server/check snippets later in this
guide are reference argv outside that target set; queue any such compilation
separately in the same owned, locked small-pool phase with the same memory and
two-worker limits. Complete all builds before reserving the inference pool.

Always use `qwen35-cuda` for this model. The broader `cuda` feature enables all
CUDA model families and intentionally links the legacy `apxinf-cuda` runtime;
it is not required by Qwen3.6. A completely empty target directory must still
compile `apxinf-cuda-new` once. Later builds reuse native objects across Cargo
examples, checks and tests, so do not run `cargo clean` during normal iteration.

Confirm that the Qwen-only dependency graph does not contain the legacy crate:

```bash
cargo tree -p apxinf-model --features qwen35-cuda \
  --edges normal,build | grep apxinf-cuda
```

The output should contain `apxinf-cuda-new` only.

## 3. Standard benchmark: 4096 prompt + 256 output

> **Run the benchmark from the repository root.** The attention/GEMM tuning
> recipe cache directory is built as a *relative* path
> (`adapters/runtime.cpp:41` returns `configs/tuning/nvidia/<family>-sm<cc>/...`),
> so it resolves against the process working directory. Launching from
> elsewhere silently starts with a cold recipe cache, which changes both the
> first-iteration timings and which kernels get selected.

The benchmark generates a deterministic prompt of exactly 4,096 token IDs,
loads the model once, performs two warmups, then records five iterations. EOS
stopping is disabled, so every successful iteration emits exactly 256 tokens.

Development binary:

```bash
"$CARGO_TARGET_DIR/debug/examples/qwen35_mtp_bench" \
  "$QWEN35_CHECKPOINT" \
  --device 0 \
  --max-new 256 \
  --warmups 2 \
  --repeats 5 \
  --mode both \
  --qwen35-moe-activation nvfp4 \
  --qwen35-kv-cache bf16 \
  | tee qwen35-4096x256-nvfp4-bf16.jsonl
```

For the release binary, replace `debug` with `release` in the executable path.

`--mode both` runs the scalar target decoder and the MTP path in the same
process, then requires exact token equality between them. Use `--mode scalar`
or `--mode mtp` when measuring only one path.

The logical sequence starts at 4,096 positions and reaches approximately
4,351/4,352 positions while generating 256 tokens. The implementation admits
a capacity of 4,355 positions because MTP needs three temporary lookahead
slots; those slots are not additional emitted tokens.

Each output line is JSON. Summary records include median/P95 TTFT, prefill
tokens/s, generation tokens/s and MTP acceptance counters. Iteration records
also include an FNV-1a token digest for repeatability checks.

### Tune once before measuring or serving

Recipe misses must not trigger tuning inside the first business request. Use
the HTTP server itself for one explicit **sampling-only** warmup pass, so the
warmup exercises the same path as AIOS. Always supply an absolute recipe
directory; this also makes a copied binary independent of its working directory:

```bash
export APXINF_TUNING_DIR=/opt/data/apxinf/qwen35-recipes
mkdir -p "$APXINF_TUNING_DIR"

APXINF_QWEN35_CUDA_GRAPHS=1 \
APXINF_QWEN35_GRAPH_GDN=full \
"$CARGO_TARGET_DIR/release/apxinf" serve \
  --model "$QWEN35_CHECKPOINT" \
  --model-name qwen3_5_moe \
  --served-model-name qwen3.6-35b-a3b \
  --device cuda:0 --dtype bf16 \
  --qwen35-moe-activation nvfp4 --qwen35-kv-cache bf16 \
  --host 127.0.0.1 --port 8000 \
  --max-input-tokens 20000 --max-output-tokens 4096 \
  --enable-context-reuse \
  --autotune \
  --tuning-dir "$APXINF_TUNING_DIR"
```

After the server is healthy, send one representative AIOS request with its real
roughly 13K system/tool prefix and at least two output tokens. The request must
explicitly use `temperature=1.0`, `top_k=20`, and `top_p=0.95`. Stop this warmup
server, then start the production server **without** `--autotune`, using the
same `--tuning-dir`. Valid exact-key recipes are restored without benchmarking.
Unseen NVFP4 GEMM shapes use the one-CTA CUTLASS baseline; unit-scale FP8 GEMM
uses a zero-workspace CUTLASS baseline instead of expanding weights to BF16.
Eligible exact-view
KV-cache attention uses fixed-config FA2; neither requires online tuning.
Fallback executions are not written as tuned winners. Warm up representative
AIOS prompt/binding shapes with `--autotune` before measuring optimized latency,
and report whether a measured request used a recipe or a baseline.

Prepared cuBLAS GEMMs share a handle per preparation thread/device/stream while
retaining independent graph workspaces. Thor's library handle itself reserves
about 64 MiB; creating one per GEMM can exhaust the device pool while preparing
the MTP verifier, even though the reported operator workspaces are small.
Serving admission includes input, output budget and three MTP lookahead slots:
for a Harness input cap of 32,768, use `--max-input-tokens 32768` and explicitly
set the output budget (for example `--max-output-tokens 1024`). Reapply the GPU
clock cap above after reboot and reserve a sufficient HugeTLB pool before load.

## 4. Supported runtime precision choices

BF16 target KV remains the current precision baseline. Explicit FP8 KV is
implemented: the builtin loader and context benchmark read `hf_quant_config.json`;
`config.rs::parse_trt_fp8_kv_cache_scales` requires `MIXED_PRECISION` / `FP8`
manifest flags and returns unit K/V dequantization defaults `(1.0, 1.0)`.
Missing/invalid manifests or an FP8 runtime without scales fail validation.
These are deployment defaults from source, not serialized per-layer checkpoint
calibration or verified TRT engine-export parameters. Storage/attention and
long-Agent quality must pass their separate gates before FP8 results are accepted.

| Arguments | MoE execution | KV-cache storage |
| --- | --- | --- |
| `nvfp4` + `bf16` | W4A4, runtime NVFP4 activations | BF16 |
| `bf16` + `bf16` | W4A16, BF16 activations | BF16 |
| `nvfp4` + `fp8` | W4A4, runtime NVFP4 activations | FP8, unit defaults; acceptance pending |
| `bf16` + `fp8` | W4A16, BF16 activations | FP8, unit defaults; acceptance pending |

Do not compare runs with different `--mode`, warmup count, repeat count, power
mode or thermal state.

## 5. Text generation

Build the Qwen-only CLI:

```bash
cargo build --features qwen35-cuda
```

Run sampling generation with the model's recommended parameters:

```bash
"$CARGO_TARGET_DIR/debug/apxinf" generate \
  --model "$QWEN35_CHECKPOINT" \
  --model-name qwen3_5_moe \
  --device cuda:0 \
  --dtype bf16 \
  --qwen35-moe-activation nvfp4 \
  --qwen35-kv-cache bf16 \
  --sample \
  --temperature 1.0 \
  --top-k 20 \
  --top-p 0.95 \
  --seed 123 \
  --max-tokens 256 \
  --prompt "Explain the main trade-off between latency and numerical precision."
```

Use only a supported row from the table above.

## 6. OpenAI-compatible server

On the Thor CUDA 12 SM101 build, choose `--qwen35-kv-cache fp8_dynamic`
with NVFP4 Edge experts for the fixed long-context Decode configuration. The
model enables CUDA Graphs with full GDN capture and uses three drafts per MTP
round for eligible greedy requests. The Thor LM head selects Stage All for
one row and tile-128 WMMA for four rows at `2048 → 248320`. Edge NVFP4 MoE
uses ordered reduction. The fast path needs no performance environment
variables. Select `--enable-context-reuse` to retain request prefixes and pass
the recipe directory for the chosen KV mode.

Build and start the server without enabling the legacy CUDA runtime:

```bash
cargo build --release --features qwen35-cuda,server

export APXINF_TUNING_DIR=/opt/data/apxinf/qwen35-recipes

APXINF_QWEN35_CUDA_GRAPHS=1 \
APXINF_QWEN35_GRAPH_GDN=full \
"$CARGO_TARGET_DIR/release/apxinf" serve \
  --model "$QWEN35_CHECKPOINT" \
  --model-name qwen3_5_moe \
  --served-model-name qwen3.6-35b-a3b \
  --device cuda:0 \
  --dtype bf16 \
  --qwen35-moe-activation nvfp4 \
  --qwen35-kv-cache bf16 \
  --host 0.0.0.0 \
  --port 8000 \
  --max-input-tokens 20000 \
  --max-output-tokens 4096 \
  --tuning-dir "$APXINF_TUNING_DIR" \
  --enable-context-reuse
```

Test it from another shell:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.6-35b-a3b",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 64,
    "temperature": 1.0,
    "top_k": 20,
    "top_p": 0.95,
    "seed": 123
  }'
```

The sampling path keeps up to four distinct system/tool prefix snapshots in
most-recently-used order. Requests whose tool bindings render different
prefixes therefore do not overwrite one another immediately; a fifth distinct
prefix evicts the least recently used snapshot. `usage.prompt_tokens_details`
reports the restored prefix as `cached_tokens`.

## 7. Fast validation

After restoring HugeTLB, validate load, prefill and sampled decode through the
same OpenAI-compatible endpoint used by AIOS. Use at least two output tokens;
one output token is insufficient to measure post-TTFT decode:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen3.6-35b-a3b",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 2,
    "temperature": 1.0,
    "top_k": 20,
    "top_p": 0.95,
    "seed": 123
  }'
```

```bash
cargo check -p apxinf-model \
  --features qwen35-cuda \
  --example qwen35_mtp_bench

cargo test -p apxinf-model \
  --no-default-features \
  qwen35moe --lib
```

The focused model test currently runs 23 tests. Compiler warnings from other
model families do not indicate a Qwen3.6 test failure; use the final test
summary and process exit code as the result.
