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

## Subsequent attention scheduling acceptance

The next change is measured independently against `9f7aecd204073eafd25bfdb99c1c4ab586f95fad`,
which integrates the visual GEMM change with the four then-current `Final`
commits. Do not add its percentages to the historical GEMM result above.
Hardware, checkpoint, input files, precision, Graph and fixed 32-token work are
unchanged. Each case/mode again uses two warmups and ten measurements.

| Input | Mode | Baseline TTFT (ms) | Updated TTFT (ms) | Reduction |
| --- | --- | ---: | ---: | ---: |
| One image, 256 patches, 87 prompt tokens | Scalar | 885.213 | 864.203 | 2.37% |
| One image, 256 patches, 87 prompt tokens | Adaptive MTP | 874.566 | 871.174 | 0.39% |
| Two images, 616 patches, 326 prompt tokens | Scalar | 1931.720 | 1914.971 | 0.87% |
| Two images, 616 patches, 326 prompt tokens | Adaptive MTP | 1926.349 | 1915.323 | 0.57% |

An additional reverse-order scalar check (updated then baseline, two warmups and
five measurements per process) gives 887.702 -> 866.173 ms for one image and
1926.184 -> 1910.492 ms for two images, reductions of 2.43% and 0.81%.
The small MTP first-token deltas are not an image-MTP decode-throughput claim.
All output tokens in both timing orders match the corresponding baseline.

Only the independent SLEEF exponential evaluations and BF16 probability
conversions are distributed across CUDA threads. The maximum traversal,
four-lane summation order, reciprocal, QK/PV pair reductions and rounding remain
unchanged. Max segment lengths 32..4096 use this path; other lengths retain the
original serial element traversal. Two shared floats compile to 16 additional
static bytes on Thor, within the default 49152-byte limit for the <=16384-byte
score buffer. The serial instantiation retains zero user static bytes and
successfully accepts the original 232448-byte opt-in dynamic shared limit.
No persistent workspace is added.

Paired single-image captures show 27 attention kernels taking 85.693 -> 78.730 ms;
111 unchanged visual GEMMs take 607.447 -> 606.975 ms. For two images, the retained
identical-kernel baseline and new capture show attention 246.461 -> 226.201 ms and
GEMM 1459.925 -> 1459.785 ms. Kernel captures explain the attention improvement;
model TTFT also includes the other GPU work, launch gaps and host activity.
The synthetic operator screen's larger gains are not substituted for these
real-input measurements.

Precision remains unchanged: five frozen CPU visual cases are elementwise equal
(relative L2 = 0), packing remains independent, and six scalar/MTP token controls
across all three KV precisions match `9f7aecd2`. Nine direct attention cases cover
small/mixed segments, 16 images, 4096 patches and the generic 4097-length serial
path, with exact output bits, read-only inputs and guards. Eight maintained GPU
tests pass, including the new parallel Graph/guard/image-independence test.

The GEMM build ID and its selected configurations remain unchanged. Attention's
source fingerprint changes, so prepare the needed exact Attention records before
strict deployment. Preparation and formal timing are separate; matched records
select the same baseline configurations, and recipe hashes remain unchanged in
both formal timing orders and the final strict token checks.

Two attempts to reduce ordered-GEMM add instructions were rejected: despite exact
arithmetic, their major shapes regress approximately 23% and 58%. An additional
attention FTZ experiment was also left out after a small incremental gain and a
4096-length regression. None is part of the product implementation.

Raw per-round data, reverse-order checks, numerical gates, shared-memory checks,
recipe audits, source identities and profiler captures are retained in
`devlocal/qwen35-image-input/performance/round-2/`. Run
`round-2/summarize_round2.py` over that evidence for the offline acceptance gate.
The existing benchmark harness can build this revision with `LABEL=attention`;
use the same strict benchmark command and shared locks described above.
