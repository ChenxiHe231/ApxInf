# Qwen3.5/3.6 image context reuse

This change enables process-local image prompt reuse behind the existing
`--enable-context-reuse` serving option. It does not change image decoding,
resize rules, shared pixel budgets, visual arithmetic, model weights, MoE
activation precision or KV-cache precision.

## Cache identity and ownership

A placeholder token sequence is insufficient to identify multimodal state.
The family compares exact prefix token IDs, ordered image spans, SHA-256 of
actual processed pixel bytes plus dtype and `grid_thw`, and three-axis prefix
positions. An image's entire identity is required even when the checkpoint
ends partway through its span. Positions following the original prompt use
its compressed mRoPE continuation. Text-only lookups reject image-dependent
entries; prefixes strictly before the first image remain reusable.

Consequently, a changed URL with identical processed content can hit, whereas
changed pixels, image order, grid, dtype or prefix positions cannot falsely hit.
The current processor still divides a total pixel budget across all images.
Adding an image can therefore invalidate earlier image encodings and KV state.
There is deliberately no fixed-resolution workaround.

The runtime owns these bounded caches:

- Visual results: at most 64 independent BF16 tensors, totaling at most 128 MiB.
  Entries copy merger outputs; they never retain mutable vision scratch views.
  On a partial hit, only missing processed images enter the visual tower;
  duplicate misses are encoded once. Existing and newly encoded rows are
  assembled in the original request order. Hits retain owning tensors until
  assembly completes, independently of later LRU eviction. Input pixels keep
  their exact bytes and dtype on both CPU and CUDA. A complete miss retains
  the original packed forward. Ordered-grid execution profiles stay bounded
  at four.
- Scalar target checkpoints: the existing four-entry LRU holds KV/GDN state,
  modality identity and, for complete prompts, independently copied logits.
  Image admission reserves stable/complete endpoints while preserving leading
  text checkpoints for replacement/resized images.
- MTP prompt checkpoints: the existing four-entry LRU holds target KV/GDN,
  proposer KV/position state and an independent final target hidden row.
  Complete entries also retain the greedy first token. The existing separate
  system checkpoint remains available. Allocation uses the session capacity
  and recycles evicted storage instead of growing with each conversation turn.

Complete checkpoints serve exact retries without another prefill row. Stable
checkpoints cover empty-thinking generation suffixes and the existing 64-token
volatile-tail policy for long tool prompts. Scalar and MTP append prefill use
absolute multimodal positions and the unchanged new-token embeddings. MTP
teacher forcing preserves the shifted target-hidden/next-embedding alignment.
Only target-committed generated tokens can identify a live endpoint; scalar
fallback clears its MTP live identity. Restoring a complete MTP prompt creates
a fresh request-local schedule rather than resuming old generation bookkeeping.

When reuse is disabled, image inputs retain the original packed visual forward
and full prompt prefill, without content hashing or additional host downloads.
Four-row image prefix chunks use temporary decoder bindings and cannot borrow
the persistent text-verifier projection execution session. Decode preparation
and the image rowwise verification policy remain unchanged.

## Precision and validation boundaries

Image results copied from the feature cache must be elementwise identical to
fresh native embeddings, including after another request overwrites vision
scratch. Target snapshot restore/append is compared elementwise with fresh
native execution using identical chunk boundaries. Uncached 32-token controls,
32-token cold-cache/replay equality, and scalar/MTP parity remain separate
checks. Seeded sampling is reproducible on the same cached execution path.

Whole and chunked prefill already use different native batch shapes and can
have different floating accumulation/rounding. Bitwise identity across those
shapes is not claimed; a fixed sampling seed can select `Cat` versus `cat`.
Closed-answer response comparisons ignore letter case, while cache replay and
scalar/MTP 32-token comparisons remain exact. The existing visual acceptance
thresholds (finite, cosine >= 0.999, relative L2 <= 0.02) are unchanged. No
precision threshold is relaxed to accept a cache restoration error.

The maintained real-checkpoint checks are
[`qwen35_image_reuse.rs`](../../crates/apxinf-model/tests/qwen35_image_reuse.rs)
and the ignored native chunk/feature-cache test in
[`llm_image.rs`](../../crates/apxinf-model/src/qwen35moe/llm_image.rs).
CPU identity tests cover changed pixels/grids, image order, partial image spans,
continuation positions, readonly input and uncached-plan rejection.

## Measurement protocol

The comparison uses the same release binary, checkpoint, NVFP4 weights and
activations, dynamic FP8 KV, ordinary greedy generation, Graph policy and
unchanged hardware frequency policy. The baseline disables context reuse and
retains the existing prepared visual path. Image preprocessing and tokenization
are included in HTTP end-to-end latency. Model loading, first-request setup and
separate autotuning are excluded from warm measurements.

The constructed long conversation has 36,681 prompt tokens, two 256-patch
images and an 11,616-token leading system prefix. Its answer is `Red, Blue`
(four completion tokens). Follow-ups preserve the image history and add a new
run marker plus 541 new text tokens, preventing an exact-request
hit. Both arms receive identical marker sequences. Each final arm uses two
warmups and ten measurements. Strict runs require existing recipes, disable
autotuning, and compare recipe hashes before and after.

This is a constructed workload on Thor-3, not a remeasurement of the user's
original 0900c53 replay or a TensorRT comparison. Original replay inputs were
not available during this implementation.

Final HTTP end-to-end results (median, two warmups plus ten measurements):

| Workload | Reuse disabled | Reuse enabled | Speedup | Actual reused prompt tokens |
|---|---:|---:|---:|---:|
| Same 36,681-token/two-image request | 6.142 s | 0.261 s | 23.53x | 36,681 |
| 37,222-token follow-up, 541 newly evaluated tokens | 6.325 s | 0.668 s | 9.47x | 36,681 |

Both arms emitted exactly `Red, Blue`, four completion tokens. All ten matched
request pairs have identical prompt/completion counts and answers. Both strict
arms used the same binary and identical 481-record recipe set; hashes remained
unchanged during each arm. First-request initialization is recorded separately
and is not counted as a warm speedup.

Release binary SHA-256:
`ed017621881eca04d3026ebf9c92523ebdb04230d1bc380458504c865717a97c`.
Recipe-set SHA-256:
`42b0967d275762e52448fd70d98af588ea19612267960c1fda32f3fc831f82cc`.
The baseline source revision is `d1d48ba1`; both arms include this cache change,
with only the existing reuse flag changed. No-reuse image behavior retains the
original prepared packed encoder/full-prefill path.

BF16, calibrated FP8 and dynamic FP8 KV each pass real-checkpoint scalar,
seeded-sampling and MTP checks with Graphs enabled: exact uncached32 controls,
exact cold-cache/replay32 and scalar/MTP32, normal EOS-ended history, changed
pixels and text/image separation. The two-image native test additionally proves
that mixed single-image/packed visual cache entries, odd merged-row boundaries
and overwritten vision workspaces preserve embeddings elementwise. Partial-hit
assembly is checked against the full packed output for CPU/CUDA F32/BF16,
multiple missing images, duplicate misses and odd merger-row boundaries. Restored
append logits match fresh native execution with the same chunk boundaries.
All three CPU identity tests and ten serving unit tests pass. HTTP checks cover
whole-request retries, history growth, adding an image, replacement/order,
streaming, seeded sampling and recovery after malformed input. Replacement
images preserve the 11,616-token system checkpoint while invalidating image
state; observed counters are actual restored rows, not estimated token matches.

A separate paired Nsight Systems trace uses cat86/scalar and captures only
through first-token delivery. It shows 111 ordered visual GEMMs and 108 staged
QK/PV launches in the baseline and zero of these launches on a full hit. The
hit performs embedding replacement, snapshot copies and token selection, with
no transformer prefill kernels. These are kernel-launch observations from a
short diagnostic input, not a long-workload critical-path fraction or formal
kernel speedup. Decode lies outside the capture range.

Development evidence is retained under `devlocal/qwen35-image-input/image-cache/`:
`FORMAL_V3.json`, `PROFILE_V3.json`, `http-long-v3-final-{0,1}.json`, matching
server logs, `precision-*-strict.log`, native mixed-group checks and paired
`ttft-{baseline,repeat}-v3.nsys-rep` captures.

## Follow-up: partial visual-cache hits

A second change compares the previously accepted cache implementation
(`cb13f8ac`, v3) with selective missing-image encoding (v4). Both arms enable
context reuse. Each request preserves the two old images and appends a third
256x256 green frame with a distinct pixel; the frame changes every iteration.
Reseeding the same prior history occurs outside the timed request. This avoids
measuring an all-feature-cache hit as a partial-hit improvement. Both arms use
the same frame sequence, long conversation, ordinary greedy decoding, dynamic
FP8 KV, NVFP4 weights/activations and Graph policy described above.

| New-image workload | Original full-batch miss path | Selective missing-image path |
|---|---:|---:|
| Median HTTP latency, 2 warmups + 10 measurements | 2.235 s | 1.263 s |
| Prompt / actually reused / output tokens | 36,809 / 36,711 / 6 | 36,809 / 36,711 / 6 |

Latency decreases 43.5% (1.77x speedup). All ten paired outputs equal
`Red, Blue, Green`; prompt, output and restored-token counts also match.
Only the new image is encoded instead of all three. Cached pixels/grid
identity still decides validity; adding images that resize old images does
not qualify for this partial-hit shortcut.

All-hit controls in these runs have median HTTP latency 257.7 versus 268.6 ms,
while server model-call latency is 104.0 versus 103.1 ms. The additional HTTP
time is outside the model-call interval; preprocessing was not isolated, so
its cause is not assigned. No all-hit performance improvement is claimed.
The paired details are retained in `ALL_HIT_V4_AUDIT.json`.

The two strict arms share 504 recipes with unchanged hash
`b6f943a992778420a15739c48b30ca91e8de484826bdadabe66502c834832746`.
The original binary hash is the v3 hash above; the v4 binary SHA-256 is
`dbb1fc3ba4685c1da40e2453bd8cd5f542bd0f6f20a32143ddd6d4c4462fc0eb`.
Additional MTP shapes were prepared in a separate process after strict mode
correctly rejected a missing M=23 recipe. Strict HTTP qualification then passed
ordinary greedy, seeded sampling, MTP, retries, replacements/order, streaming,
and recovery after malformed images. No fallback or tuning occurs in timing.

Separate prepared feature-only traces use the two-image 616-patch fixture.
One cached image leaves 308 patches to encode. Both traces contain 111 visual
GEMM launches; the smaller batch reduces their CUDA kernel-time sum from
1178.929 to 600.753 ms. This explains the reduced work and is not a model
latency measurement or an estimate of a critical-path fraction. Exact feature
checks cover CPU/CUDA F32/BF16 and multiple or duplicate misses. Native
snapshot/append equivalence also remains exact. Unaffected v3 KV/mode checks
are reused; this change does not alter language arithmetic or checkpointing.

Evidence: `FORMAL_V4.json`, `PROFILE_V4.json`, `partial-{v3,v4}-final.json`,
`partial-v4-http-quality-strict2.json`, `partial-v4-{prepare,strict}.log`, and
`features-{full,partial}-v4.nsys-rep` in the same development evidence directory.

## Reproduction

Use both shared GPU locks in order (`/tmp/apxinf-gpu.lock`, then
`/tmp/apxinf_gpu.lock`). Autotune in a separate process before strict timing.
The processed fixture is the unchanged serving processor's binary frame:
four little-endian metadata-length bytes, JSON `tokens`/`shape`/`grids`, then
contiguous little-endian F32 pixel bytes.

```bash
export APXINF_QWEN35_CHECKPOINT=/path/to/Qwen3.6-35B-A3B-NVFP4
export APXINF_QWEN35_IMAGE_FIXTURE=/path/to/processed.bin
export APXINF_QWEN35_IMAGE_KV=fp8_dynamic
export APXINF_QWEN35_IMAGE_RECIPES=/absolute/prepared/recipe-directory
export APXINF_QWEN35_CUDA_GRAPHS=1 APXINF_QWEN35_GRAPH_GDN=full
export APXINF_REQUIRE_RECIPES=1
cargo test --release -p apxinf-model --features qwen35-cuda \
  --test qwen35_image_reuse -- --ignored --nocapture --test-threads=1
```

Use `bf16` and `fp8` for the other KV modes. For isolated paired TTFT traces,
set `APXINF_QWEN35_IMAGE_MODES=scalar` and select exactly one profiler label:
`scalar/baseline/2` or `scalar/repeat/2`. The test's CUDA profiler range ends at
the first callback, excluding subsequent decode. Profiling is separate from
formal timing.

For isolated visual-cache profiling, run the ignored library test with a
multi-image fixture and `APXINF_QWEN35_FEATURE_PROFILE=full` or `partial`.
Its range captures prepared full packed execution or a partial hit; the first
image is cached and all remaining unique images are encoded. Small boundary
checks lie outside the range. Both captures must use the same fixture, recipes
and precision. These diagnostics do not replace model end-to-end A/B timing.

```bash
cargo test --release -p apxinf-model --features qwen35-cuda --lib \
  image_snapshot_append_matches_uncached_chunk_execution \
  -- --ignored --nocapture --test-threads=1
```
