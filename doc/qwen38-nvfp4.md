# Qwen3.8-27B-NVFP4 on ApxInf

Qwen3.8-27B-NVFP4 is a text LLM with 64 layers — 48 Gated DeltaNet (linear
attention) plus 16 full-attention layers — quantized to mixed NVFP4/FP8 by
NVIDIA ModelOpt. ApxInf runs it on Jetson AGX Thor (sm_110) with batch 1,
BF16 KV cache, and CUDA-graph decode.

Module layout and ownership follow
[Model Layer Architecture](model-layer-architecture.md); the implementation
lives in `crates/apxinf-model/src/qwen38/`.

## Requirements

- A Jetson AGX Thor (or another sm_110 device) with the CUDA toolkit.
- The checkpoint directory as published: safetensors shards with
  `model.safetensors.index.json`, `config.json` (`model_type: qwen3_5`),
  `tokenizer.json`, `tokenizer_config.json` and `chat_template.jinja`.
  Nothing needs to be converted or renamed; `AutoModel` detects the model
  from `config.json` directly.
- About 20 GiB of free device memory for weights plus the prefill copies.

Build the CLI once:

```bash
cargo build --features cuda --release --bin apxinf
```

## Real inference with the CLI

`apxinf generate` runs the complete pipeline: the checkpoint's own chat
template (system + thinking frame), tokenization, prefill, streamed greedy
decode, EOS stop, and a latency report.

```bash
./target/release/apxinf generate \
    --model <path-to-Qwen3.8-27B-NVFP4> \
    --device cuda --dtype bf16 \
    --prompt "What is the capital of France? Answer in one sentence."
```

Example output on Thor:

```
<think>
The user asks for the capital of France and requests the answer in one
sentence. This is a straightforward factual question.
</think>

The capital of France is Paris.

=== Generation Profile ===
TTFT:             752.4 ms
TPOT:             77.4 ms/token
Generation TPS:   12.92 tok/s
=========================
```

The checkpoint is a thinking model: it reasons inside `<think>…</think>`
before answering. The template is rendered from the checkpoint's
`chat_template.jinja`, so the conversation format always matches what the
model was trained on.

Useful flags:

| Flag | Effect |
|---|---|
| `--max-tokens N` | generation budget (model default otherwise) |
| `--system TEXT` | override the system message |
| `--no-eos-stop` | fixed-length generation, no early stop |
| `--sample --temperature T --top-k K --top-p P --seed S` | categorical sampling instead of greedy |

Greedy decoding uses the model's device-side argmax path (4 bytes read back
per token); sampling flags route through the model-neutral GPU sampler.

### Latency expectations

- The first process on a machine (or after a rebuild) additionally pays
  one-time GEMM/attention autotuning; the tuned recipes persist under
  `/tmp/apxinf-qwen38-*-recipes` and later processes skip it.
- Every CLI invocation is a fresh process and pays weight loading plus one
  CUDA-graph capture (~0.2 s inside the first TTFT). A resident process
  serving repeated requests reaches the steady-state numbers below from its
  second generation on.

## Benchmarking

`qwen38_bench` measures steady-state latency through the same public entry
(`AutoModel` → `generate_streaming`) with a deterministic prompt. It prints
one JSON object per repeat and a summary; repeat 0 is the warm-up
(autotune + graph capture) and is excluded from the means. The run aborts if
any repeat generates different tokens, and the reported `token_md5` is
comparable across the kernel test harness, the acceptance test and this
benchmark.

```bash
cargo run -p apxinf-model --features cuda --release --example qwen38_bench -- \
    <path-to-Qwen3.8-27B-NVFP4> --prompt-len 2048 --max-new 128 --repeats 3
```

Measured on Jetson AGX Thor at locked clocks (1575 MHz GPC):

| Metric | Steady state |
|---|---:|
| TTFT, 2048-token prompt | 534 ms |
| Decode | 78.1 ms/token (12.8 tok/s) |

## Correctness gates

- The NVFP4/FP8 quantizers are bit-compatible with vLLM's encodings
  (RNE FP4 midpoints, SATFINITE E4M3 with subnormals, double-rounded fused
  SwiGLU), enforced by the `qwen38_fp8_quant_contract` and
  `qwen38_nvfp4_quant_contract` suites in `apxinf-cuda-new`.
- The acceptance test `crates/apxinf-model/tests/qwen38_generation.rs`
  drives the module through the shared generation loop and checks the
  generated-token digest against the kernel-harness baseline:

  ```bash
  APXINF_QWEN38_CHECKPOINT=<path> \
  cargo test -p apxinf-model --features cuda --release \
      --test qwen38_generation -- --ignored --nocapture --test-threads=1
  ```

## Current limitations

- Batch 1, text only. Prompt plus generation must fit the KV capacity fixed
  at load (`max_seq_len`, minimum 4096).
- Tuned for sm_110 (Thor). Other architectures build but the kernel
  selection was not validated there.
- The independent whole-model logits comparison against vLLM is close but
  not yet exact (mean KL ≈ 5e-3 over a 2048-token fixed prefix; top-1 match
  126–127/129 positions); the remaining divergence is being localized.
