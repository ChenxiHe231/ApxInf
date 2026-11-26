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
  Ordered-grid vision execution profiles remain separately bounded at four.
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
and overwritten vision workspaces preserve embeddings elementwise. Restored
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
