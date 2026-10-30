# Qwen3.6-35B-A3B-NVFP4 on Thor-U

This guide covers the `Qwen3.6-35B-A3B-NVFP4` checkpoint installed on Thor-U.
Its Hugging Face architecture metadata remains `qwen3_5_moe`, so the internal
Rust module and registry name continue to use `qwen35`; that name must not be
interpreted as support for the older `txn545/Qwen3.5-35B-A3B-NVFP4` checkpoint.
The guide includes fast builds,
the fixed 4,096-token benchmark, precision comparisons, text generation and
the OpenAI-compatible server.

> **Current validation status (two separate things, do not conflate them):**
>
> *Long prefill is fixed.* The 4,096-token FP8 prefill is validated on Thor-U
> **subject to the 1206 MHz GPU cap** described in the root `README.md`
> section “Thor-U Qwen3.6 long-prefill stability”. With that cap applied,
> repeated 4,096-token prefills complete with the host fully responsive and no
> OOM, CUDA Xid, or kernel panic; a cache-warm run measures 1.02-1.06 s TTFT
> (3,874-4,008 prefill tokens/s). The earlier claim that this workload
> exhausts unified memory and renders the host unreachable was a
> clock-frequency problem, not a memory-layout problem, and is superseded.
> The cap is **not persistent across reboot** - re-apply it every session and
> read the values back before benchmarking.
>
> *The section 3 benchmark still does not complete,* for an unrelated reason:
> scalar decode is not reproducible across repeated identical generate calls,
> so `qwen35_mtp_bench`'s own token-equality assertion aborts the run at the
> second iteration. The divergence is always at generated token 1 (token 0,
> produced by the fixed-shape prefill, is stable). Root cause: the attention
> autotuner's recipe key includes the live KV length, so every decode step is
> a cache miss and re-selects `num_splits` by noisy timing, changing the
> floating-point reduction order and occasionally flipping the greedy argmax.
> Until that is fixed, treat the section 3 command as a target benchmark.

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

Stop every ApxInf/Edge-LLM inference process, release HugeTLB, and limit both
Cargo and the native CUDA worker pool to two jobs before compiling:

```bash
sudo sysctl -w vm.nr_hugepages=0
grep -E 'MemAvailable|HugePages_Total|HugePages_Free|Hugetlb' /proc/meminfo

export APXINF_NVCC_PARALLELISM=2
export CARGO_BUILD_JOBS=2
```

After the build finishes, restore the 40 GiB inference pool **before** loading
the model, and do not run Cargo/NVCC at the same time as inference:

```bash
sudo sysctl -w vm.nr_hugepages=20480
grep -E 'MemAvailable|HugePages_Total|HugePages_Free|Hugetlb' /proc/meminfo
```

`HugePages_Total` must report `20480`. If the kernel cannot reserve the full
pool, stop memory-heavy processes and retry before starting inference.

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

For daily development and quick benchmark iteration:

```bash
cargo build -p apxinf-model \
  --features qwen35-cuda \
  --example qwen35_mtp_bench
```

For final performance records, build the release binary:

```bash
cargo build --release -p apxinf-model \
  --features qwen35-cuda \
  --example qwen35_mtp_bench
```

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

Recipe misses trigger measured online tuning and must never be included in a
performance result. Build the context benchmark and run one short explicit
`--autotune` pass before starting the server. Match `--kv-capacity` to the
server admission capacity (`max_input_tokens + max_output_tokens + 3`) and use
an input length representative of the deployment. For the AIOS 20K/4096
configuration and its roughly 13K static prefix:

```bash
cargo build --release -p apxinf-model \
  --features qwen35-cuda \
  --example qwen35_context_bench

APXINF_QWEN35_CUDA_GRAPHS=1 \
APXINF_QWEN35_GRAPH_GDN=full \
"$CARGO_TARGET_DIR/release/examples/qwen35_context_bench" \
  "$QWEN35_CHECKPOINT" \
  --autotune \
  --prefix-tokens 0 \
  --input-tokens 13312 \
  --kv-capacity 24099 \
  --max-new 2 \
  --warmups 1 \
  --repeats 1 \
  --mode scalar
```

Run this command from the repository root. A later server run reuses the
persisted recipe and must not create a new file in the tuning directory during
the measured request.

## 4. Supported runtime precision choices

The checkpoint does not contain `kv_cache_scheme`, `k_scale`, or `v_scale`.
Use BF16 KV cache for correctness. A request for FP8 KV cache must fail rather
than fabricate a scale and silently change model accuracy.

| Arguments | MoE execution | KV-cache storage |
| --- | --- | --- |
| `nvfp4` + `bf16` | W4A4, runtime NVFP4 activations | BF16 |
| `bf16` + `bf16` | W4A16, BF16 activations | BF16 |

Do not compare runs with different `--mode`, warmup count, repeat count, power
mode or thermal state.

## 5. Text generation

Build the Qwen-only CLI:

```bash
cargo build --features qwen35-cuda
```

Run deterministic greedy generation:

```bash
"$CARGO_TARGET_DIR/debug/apxinf" generate \
  --model "$QWEN35_CHECKPOINT" \
  --model-name qwen3_5_moe \
  --device cuda:0 \
  --dtype bf16 \
  --qwen35-moe-activation nvfp4 \
  --qwen35-kv-cache bf16 \
  --greedy \
  --max-tokens 256 \
  --prompt "Explain the main trade-off between latency and numerical precision."
```

Use only a supported row from the table above.

## 6. OpenAI-compatible server

Build and start the server without enabling the legacy CUDA runtime:

```bash
cargo build --release --features qwen35-cuda,server

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
    "temperature": 0
  }'
```

## 7. Fast validation

After restoring HugeTLB, use a one-token prompt and two output tokens to
validate load, prefill and one actual decode step. One output token is
insufficient because generation TPS needs a post-TTFT interval:

```bash
"$CARGO_TARGET_DIR/debug/examples/qwen35_mtp_bench" \
  "$QWEN35_CHECKPOINT" \
  --device 0 --prompt-tokens 1 --max-new 2 \
  --warmups 0 --repeats 1 --mode scalar \
  --qwen35-moe-activation nvfp4 --qwen35-kv-cache bf16
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
