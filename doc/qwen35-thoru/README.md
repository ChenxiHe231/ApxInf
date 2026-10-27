# Qwen3.6-35B-A3B-NVFP4 on Thor-U

This guide covers the `Qwen3.6-35B-A3B-NVFP4` checkpoint installed on Thor-U.
Its Hugging Face architecture metadata remains `qwen3_5_moe`, so the internal
Rust module and registry name continue to use `qwen35`; that name must not be
interpreted as support for the older `txn545/Qwen3.5-35B-A3B-NVFP4` checkpoint.
The guide includes fast builds,
the fixed 4,096-token benchmark, precision comparisons, text generation and
the OpenAI-compatible server.

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
cargo build --features qwen35-cuda,server

"$CARGO_TARGET_DIR/debug/apxinf" serve \
  --model "$QWEN35_CHECKPOINT" \
  --model-name qwen3_5_moe \
  --served-model-name qwen3.6-35b-a3b \
  --device cuda:0 \
  --dtype bf16 \
  --qwen35-moe-activation nvfp4 \
  --qwen35-kv-cache bf16 \
  --host 0.0.0.0 \
  --port 8000 \
  --max-input-tokens 4096 \
  --max-output-tokens 256
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
