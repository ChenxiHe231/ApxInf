# Qwen3.5/3.6 image performance acceptance

## Scope and result

Compared with `429c649bfff135c33033423419c81ab11ad0f2a4`, this change accelerates
the existing ordered BF16 visual GEMM. It preserves the visual weights, image
resolution, numerical contract, language decoder and MTP policy.

Thor-3 / Jetson Thor SM110, CUDA 13.2, existing
`/home/wwxq/models/Qwen3.6-35B-A3B-NVFP4`; NVFP4 MoE activations, dynamic FP8
target KV, CUDA Graph, greedy generation of exactly 32 tokens with EOS stopping
disabled for fixed work. Each process loads the model once. Each case/mode has
two warmups and ten measurements. The table reports median model-call time to
the first streaming token, including visual encoding and language prefill.
Image download, CPU preprocessing, processor startup and model loading are
outside this interval.

| Input | Mode | Baseline TTFT (ms) | Updated TTFT (ms) | Reduction |
| --- | --- | ---: | ---: | ---: |
| One image, 256 patches, 87 prompt tokens | Scalar | 2173.946 | 889.658 | 59.08% |
| One image, 256 patches, 87 prompt tokens | Adaptive MTP | 2173.241 | 874.111 | 59.78% |
| Two images, 616 patches, 326 prompt tokens | Scalar | 5003.548 | 1927.132 | 61.48% |
| Two images, 616 patches, 326 prompt tokens | Adaptive MTP | 5001.560 | 1924.973 | 61.51% |

All measured output token sequences match their corresponding baseline.
These results establish a visual-prefill/TTFT improvement for these cases.
They do not establish a decode speedup, a sampling throughput result, or a
benefit from image MTP itself. Image MTP still verifies rows sequentially.
Callback TPOT varies in the retained diagnostics and is not attributed to this
visual-only change. GPU frequency policy was left unchanged (`nvhost_podgov`,
315–1575 MHz); GPU work used the shared locks, with no concurrent task GPU jobs.

## Change and numerical contract

The old shared weight tile had a 64-float row stride, putting simultaneous
column loads into the same bank. The new stride is 65. Larger matrices reuse
the tile across four independent output accumulators per thread; small matrices
retain a single output per thread. Every output keeps its original K order.

Explicit PTX `add.rz/rp/rm.ftz.f32` and `mul.rn.ftz.f32` replace repeated software
operand-flushing masks. Directed bounds still detect inexact sums, round-to-odd
is retained, and zero/subnormal results and NaNs keep their declared treatment.
The final FP32 bias addition and BF16 rounding are unchanged. Tensor Core or
reassociated reductions are not substituted for this contract.

A paired Nsight capture of the same prepared two-image visual traversal shows
111 ordered GEMMs taking 4527.746 ms before and 1459.925 ms after. The unchanged
27 attention kernels take 246.257 and 246.461 ms. This explains the TTFT result;
these kernel totals are profiler evidence, not another model benchmark.

## Precision and regression checks

The frozen CPU Hugging Face BF16 eager reference and original thresholds remain:
finite output, cosine >= 0.999 and relative L2 <= 0.02. Red, natural photo, OCR,
two non-square images and packed photo+OCR all remain elementwise equal to their
references (relative L2 = 0); packing preserves independent image outputs.

The arithmetic probe compares 1,312,320 input pairs, including arbitrary FP32
bits, every BF16 encoding in four pairings, subnormal boundaries, infinities and
NaNs. Both addition and multiplication have zero bit mismatches against the old
helpers. Matrix probes cover representative visual dimensions, partial tiles
and the maximum 4096-patch extent, with unchanged inputs and output guards.
Maintained GPU tests cover cancellation across tiles, subnormal products,
overflow/NaNs, prepared execution, CUDA Graph replay and rejected contracts.
With EOS stopping enabled, six single-/two-image controls across BF16, static
FP8 and dynamic FP8 KV produce identical baseline/updated scalar/MTP tokens
(two tokens for the short single-image answer and 32 for each two-image answer).

## Reproduction and artifacts

Development evidence lives under `devlocal/qwen35-image-input/performance/`.
It includes both static benchmark executables, harness sources, per-round JSONL,
precision results, source/binary identities, recipe audits and paired Nsight
captures. Frozen image inputs/references are reused from the existing image-input
acceptance artifacts; checkpoints are not copied.

Prepare recipes in a separate process. The GEMM source fingerprint changes with
this kernel. Before timing, restore the baseline provider/configuration choices
for already prepared matching candidate records, as recorded by
`pin_comparison_recipes.py`; the native strict loader still validates them.
This holds all unchanged GEMMs to the same choices. Do not time online tuning.
Both baseline and updated formal runs require exact recipes and retain identical
recipe-file hashes before/after their measurement intervals.

With the supplied harness and frozen inputs in place, build each revision's
benchmark binaries using `performance/build_benches.py` (`LABEL=baseline` or
`LABEL=candidate`) against that revision's release libraries. Run with:

```bash
export APXINF_IMAGE_TEST_ROOT="$PWD/devlocal/qwen35-image-input"
export APXINF_REQUIRE_RECIPES=1 APXINF_QWEN35_CUDA_GRAPHS=1
export APXINF_QWEN35_GRAPH_GDN=full BENCH_ITERS=10
# Acquire /tmp/apxinf-gpu.lock, then /tmp/apxinf_gpu.lock before running.
"$APXINF_IMAGE_TEST_ROOT/performance/model_bench-candidate"
```

The maintained backend regression command is:

```bash
APXINF_CUDA_ARCHS=110 cargo test --release --offline -p apxinf-cuda-new \
  --test vision_numerics --test segmented_staged_bf16 -- --test-threads=1
```
