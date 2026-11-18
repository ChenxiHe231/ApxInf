# CUDA 13.2 Thor NVFP4 acceptance

Rebase completed onto upstream `50d283e4`, followed by compatibility commit
`f534b034`. Executed code is frozen at `374e0911` (repair `bb994476`).
The original accepted `68b12f23` snapshot and checkpoint were not modified.
Subsequent documentation commits do not change the measured code.

## Repair

CUDA 13 renamed Thor SM101 to SM110. Build and runtime admission now accept
both exact Thor identifiers. The unchanged CUDA-12 compatibility AOT archive
loads and computes on CUDA 13.2 / SM110. Unrelated architectures are excluded.
The rebase also introduced a GEMM baseline registry conflict; the upstream
single-baseline invariant is restored while preserving packed-format selection.

The host-routed NVFP4 activation fallback has an incompatible scale layout.
It now fails before upload when a validated fused provider is unavailable;
this change does not claim to repair generic fallback mathematics. BF16
activation remains the separate W4A16 path.

## Conditions and evidence

Thor-3: 20 SM, CC 11.0 / SM110, 128 GB class; CUDA 13.2.86, driver 595.78.
Release binaries, NVFP4 MoE activation, existing Qwen3.6-35B-A3B checkpoint.
GPC min=max=1575 MHz only while holding `/tmp/apxinf-gpu.lock`, restored before
unlock; CPU/EMC and HugePages policy unchanged. Other tasks were not stopped.

Remote evidence: `/home/wwxq/devlocal/final-rebase-20261118/devlocal/cuda13-nvfp4`.
The remote snapshot commit `4f794c86` has the same source tree as executed
`374e0911`: `bba8ac8626d678e399aee0d590acd46802edc0ea`.
The snapshot is not the same commit identity. ELF/checkpoint/log hashes,
commands, effective environment, input and output digests, and all formal
samples are preserved in [curated evidence](model-microbench/evidence/cuda13-acceptance.json).
Raw logs, frozen recipe bytes, fixtures, and diagnostic scripts are supplied
with the accompanying acceptance report archive; curated JSON records each
raw log's SHA256 and relative path.

## Short-context comparison

Batch 1, five frozen MTBench first-turn questions, input lengths
60/61/62/61/65 (mean 61.8), maximum 512 output tokens, greedy sampling,
EOS 248046/248044. Warmup 10 per mode, three repeats per question (15 measured
requests per mode). Target KV `fp8_dynamic`, draft KV FP8 static scale 1;
CUDA Graphs and full GDN enabled, fixed draft depth 3 / verify 4. All three
accepted MTP KV optimizations enabled; verifier graphs disabled.

Decode uses actual output tokens minus one over first-to-last callback wall
time. Model E2E includes prefill/generation, excludes load, tokenization and
HTTP/AIOS overhead. Aggregate throughput is total tokens / total time.
Early EOS is retained: actual outputs 386 or 512 (Vanilla), 383 or 512 (MTP).

| Mode | Decode accepted tok/s | Model E2E tok/s | Mean acceptance length |
| --- | ---: | ---: | ---: |
| vanilla | 71.4011 | 69.6105 | — |
| mtp | 101.6421 | 97.9593 | 2.7789 |

NVIDIA publishes 86.6 Vanilla and 109.3 MTP tok/s (acceptance 2.93), in
[Edge-LLM v0.10.0 results](https://nvidia.github.io/TensorRT-Edge-LLM/0.11.0/user_guide/performance/performance-benchmarks.html#v0-10-0-results).
This is not a strict reproduction: GPU-stage versus callback-wall timing,
five-question subset versus official distribution, exact model artifacts,
sampling/template and power/clock equivalence are not fully established.
The measured gap cannot be attributed uniquely to CUDA or rebase.
Edge-LLM 0.10 [input defaults](https://github.com/NVIDIA/TensorRT-Edge-LLM/blob/v0.10.0/docs/source/user_guide/format/input-format.md)
set `enable_thinking=false`; our frozen input ends with an opening `<think>`.
This is a potential additional mismatch; the published run's actual request
JSON was not obtained.

An independent stage diagnostic also used warmup 10, then five requests / 873
rounds. Output hashes match formal MTP. Per round: draft 5.752921 ms,
verify 20.800407 ms, commit 1.132876 ms. These include host/GPU synchronize
wait and instrumentation overhead; commit includes acceptance/scheduling.
They are not GPU-event timings or the primary throughput denominator.

## Complete model performance matrix

All 42 formal scenarios passed strict repeat consistency, with unchanged
source, binary and recipes. Each case has two warmups and ten measured
iterations in a separate process with online tuning/fallback disabled.
Four cold-prefill cases, five append-4K cases and six decode cases are run
for each target KV precision; six scalar decode controls per precision.
All use the same fixed synthetic token seed, not MTBench. Context setup is
excluded from append timing. Decode generates 1024 tokens / 1023 intervals.
MTP acceptance varies with each generated trajectory; throughput need not
be monotonic with context length. P90 is a throughput percentile for decode.

### fp8_dynamic-mtp

| Case | Metric | Median | P90 | Formal N |
| --- | --- | ---: | ---: | ---: |
| cold-i2k | prefill_ttft_ms | 354.9951 | 363.6445 | 10 |
| cold-i4k | prefill_ttft_ms | 533.3826 | 536.5680 | 10 |
| cold-i8k | prefill_ttft_ms | 1058.7317 | 1068.8976 | 10 |
| cold-i16k | prefill_ttft_ms | 2435.7948 | 2445.3971 | 10 |
| append-kv4k-i4k | prefill_ttft_ms | 555.9235 | 558.6920 | 10 |
| append-kv8k-i4k | prefill_ttft_ms | 588.3557 | 590.4682 | 10 |
| append-kv16k-i4k | prefill_ttft_ms | 643.7494 | 646.8969 | 10 |
| append-kv32k-i4k | prefill_ttft_ms | 731.3332 | 735.9504 | 10 |
| append-kv48k-i4k | prefill_ttft_ms | 833.5075 | 842.0157 | 10 |
| decode-kv4k-o1k | decode_tokens_per_second | 101.4503 | 102.4380 | 10 |
| decode-kv8k-o1k | decode_tokens_per_second | 110.4940 | 112.6942 | 10 |
| decode-kv16k-o1k | decode_tokens_per_second | 107.7321 | 108.7488 | 10 |
| decode-kv24k-o1k | decode_tokens_per_second | 110.3631 | 112.5316 | 10 |
| decode-kv32k-o1k | decode_tokens_per_second | 90.1224 | 91.1712 | 10 |
| decode-kv48k-o1k | decode_tokens_per_second | 98.1944 | 100.0615 | 10 |

### fp8_dynamic-scalar

| Case | Metric | Median | P90 | Formal N |
| --- | --- | ---: | ---: | ---: |
| decode-kv4k-o1k | decode_tokens_per_second | 70.5115 | 71.0982 | 10 |
| decode-kv8k-o1k | decode_tokens_per_second | 69.7579 | 69.8946 | 10 |
| decode-kv16k-o1k | decode_tokens_per_second | 67.7234 | 68.1646 | 10 |
| decode-kv24k-o1k | decode_tokens_per_second | 65.6145 | 66.1078 | 10 |
| decode-kv32k-o1k | decode_tokens_per_second | 63.7815 | 64.6996 | 10 |
| decode-kv48k-o1k | decode_tokens_per_second | 61.1756 | 61.5589 | 10 |

### bf16-mtp

| Case | Metric | Median | P90 | Formal N |
| --- | --- | ---: | ---: | ---: |
| cold-i2k | prefill_ttft_ms | 340.4531 | 344.6114 | 10 |
| cold-i4k | prefill_ttft_ms | 583.3854 | 590.8834 | 10 |
| cold-i8k | prefill_ttft_ms | 1135.7790 | 1142.4410 | 10 |
| cold-i16k | prefill_ttft_ms | 2754.4508 | 2760.7762 | 10 |
| append-kv4k-i4k | prefill_ttft_ms | 643.8459 | 650.4126 | 10 |
| append-kv8k-i4k | prefill_ttft_ms | 710.6413 | 717.2630 | 10 |
| append-kv16k-i4k | prefill_ttft_ms | 832.6586 | 838.7702 | 10 |
| append-kv32k-i4k | prefill_ttft_ms | 1083.4944 | 1099.7803 | 10 |
| append-kv48k-i4k | prefill_ttft_ms | 1310.2757 | 1328.9679 | 10 |
| decode-kv4k-o1k | decode_tokens_per_second | 104.0415 | 104.1605 | 10 |
| decode-kv8k-o1k | decode_tokens_per_second | 100.2445 | 100.4000 | 10 |
| decode-kv16k-o1k | decode_tokens_per_second | 121.9503 | 122.3370 | 10 |
| decode-kv24k-o1k | decode_tokens_per_second | 88.4395 | 88.7372 | 10 |
| decode-kv32k-o1k | decode_tokens_per_second | 62.8154 | 63.3509 | 10 |
| decode-kv48k-o1k | decode_tokens_per_second | 101.4026 | 102.6032 | 10 |

### bf16-scalar

| Case | Metric | Median | P90 | Formal N |
| --- | --- | ---: | ---: | ---: |
| decode-kv4k-o1k | decode_tokens_per_second | 64.1527 | 64.2588 | 10 |
| decode-kv8k-o1k | decode_tokens_per_second | 60.6623 | 60.7571 | 10 |
| decode-kv16k-o1k | decode_tokens_per_second | 54.5448 | 54.6485 | 10 |
| decode-kv24k-o1k | decode_tokens_per_second | 49.7737 | 49.8375 | 10 |
| decode-kv32k-o1k | decode_tokens_per_second | 45.6323 | 45.6817 | 10 |
| decode-kv48k-o1k | decode_tokens_per_second | 38.8271 | 38.9908 | 10 |

## Correctness scope and limitations

Six unique independent Release GPU tests passed (zero ignored), including
the quantized NVFP4 constant reference, numerical GEMM registry regression,
and attention/cache/Graph checks. Additional unchanged NVFP4 constant
references at 2/3/4/128 rows passed. The target verify/rollback protocol and
MTP snapshot/different-continuation protocol each passed 180 rows using the
original 20 repetitions and strict recipes; these protocols use BF16 target
KV, MTP draft FP8 scale 1, and the accepted cache optimization flags.

All measured repeats are identical within their mode. Vanilla and MTP
outputs differ on all five natural questions (also observed in the old
CUDA-12 Thor-U run). A separate strict scalar/MTP fixture failed at token 6;
that failure is retained. The natural-prompt differences have not been
localized layer by layer. This acceptance restores meaningful CUDA-13
NVFP4 output and validates the stated gates; it does not establish lossless
MTP or a full model task-accuracy score.

BF16 point-recipe tuning hit the 30-minute resource limit at large contexts.
Every timeout log is retained. The owned child was reaped and clock policy
restored before retrying with genuine accumulated cache progress. All final
formal runs passed without relaxed thresholds, fabricated recipes or a
repeat-divergence waiver. An earlier non-best-flag tuning attempt was
explicitly cancelled and excluded from formal results.

Historical 16K=83.28 and 48K=67.54 tok/s remain historical references only.
Different hardware/precision/context and timing prevent direct comparison.

CUDA architecture naming reference:
https://docs.nvidia.com/cuda/archive/13.1.2/cuda-features-archive/index.html
