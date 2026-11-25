# Routed input quantization reuse on Thor3

`routed-quant-02`, based on P1 `5ad29ed890f336f7c19f900de1d418cfb0a3c5dd`,
passes the affected append formal A/B. This records an independent affected-scope
performance change. It has not been pushed; full15 final validation is pending.

| Initial KV + appended input | P1 mean ms | Candidate02 mean ms | Latency reduction |
|---|---:|---:|---:|
|4K + 4K|504.272613|468.820808|7.0303%|
|32K + 4K|665.266624|629.434572|5.3861%|

These are incremental gains against P1 measured on the same Thor3,2 warmups and
5 measurements. NVFP4, target dynamic-FP8 KV, greedy and fixed MTP depth3 are
unchanged. All306 frequency samples show GPU1530MHz, CPU2601MHz and EMC4266MHz;
frequency policy restoration and post guards pass. Historical Thor2/P1 gains are
not added. Full15-case final2warm10 and both20% targets remain unaccepted.

## Change and scope

The previous route kernel quantized each BF16 token row eight times. The new
kernel compares the eight actual GPU expert-scale bit patterns; equal scales
reuse the original quantize_row result and write identical FP4/SF bytes to eight
original route rows. Unequal scales run the original per-route arithmetic.
There is no approximate scale normalization, new environment flag or user switch.
FC1/FC2 AOT, weights/alpha, quantization order, output layout and recipes do not change.

The token-major kernel is selected automatically for H2048/top8, tokens>4 and
tokens>=the original grid. That grid is min(8*tokens,8*SMcount), yielding threshold160
on the current20-SM Thor. M1/M4 fused setup, smaller batches and other widths retain
the original path. This excludes measured old01 small-domain regressions:
M5/M7 heterogeneous roughly2.3–2.7x slower, M32 up to1.59x, M64 up to1.11x.
We stopped that domain rather than tuning its threads. The160 threshold is derived
from the device, not imposed on every GPU. Only the two listed append cases have
formal model performance acceptance; source support is broader than tested gains.

## Correctness and independent review

Source SHA256:61e021daa96260320a1171f8cbd6d8e0b3b5260a49cd0afb952328158791e701.
Full warp masks are valid at block256. Each warp reads the same eight immutable
scale patterns and takes a consistent branch. Each thread writes its own packed
columns and SF bytes; there is no cross-thread output read. Copies use original
FP4 H/8-word and SF H/16-byte row strides. CTA token gridstride partitions all
outputs uniquely, including tails. Unequal scales preserve arithmetic while
changing scheduling, so their performance was checked separately.

Current61-source probe was freshly compiled on Thor3:60 checks pass, covering all
34 actual scale patterns from40 layers on the qualified M161 path, zero/midpoints,
heterogeneous/mixed scales, tails, complete FP4/SF equality, readonly/canaries and
changed-input Graph replay against fresh original execution. The actual native
archive changes only runner.o; stable rustc1.98.1 and the same approved two-example
benchmark overlay are used, recipes stay unchanged and live5ad source is restored.

Current-artifact strict tests pass the full1024-token4K Decode vector and identical
MTP work, plus append4K/32K outputs. Formal append raw stdout/receipts were reparsed:
all per-sample values, means, vectors and MTP work match. Append benchmarks emit
one first token, so they establish prefill latency rather than Decode throughput.
No old P1 ELF lifecycle receipt is relabelled as candidate acceptance. These are
affected-domain tests, not exhaustive all-input/hidden-state proof.

## Profiling and remaining work

Existing append profiles identified40 routed quantization calls totalling about45ms
of visible component time. This is not critical-path f. Focused current-checkpoint
NCU selects one actual target quantization kernel per arm at4K+4K,capacity8196,
with kernel replay and cache-control=all. The diagnostic uses scalar mode,2warm1,
profiling=true and performanceEligible=false; it is not the MTP formal setup.

| Selected ROI metric | Parent | Candidate02 |
|---|---:|---:|
|Instrumented duration|1.194464ms|241.6us|
|Grid / block|160 / 256|160 / 256|
|Requested registers/thread|34|40|
|Allocated registers/thread|40|40|
|Dynamic shared bytes|0|0|
|Theoretical CTA limit|6|6|
|Measured active-warps percentage|74.566499%|89.765093%|
|Available spilling counters|0|0|

This single instrumented ROI is4.944x faster, consistent with removing repeated
conversion while preserving eight output copies. Allocated registers and CTA
limits do not regress. Active-warps percentage rises while normalized SM
throughput falls; those percentages alone do not establish instruction counts or
a memory bottleneck. DRAM read/write/throughput and detailed warp stalls remain
MISSING. Do not multiply this duration by40 layers or call it a model gain.
The independent profiler review verifies50 receipt-listed artifact hashes plus
three receipt identities,279 locked-frequency samples and clock policy restoration.

Synthetic CUDA-event screens are neither pure-kernel nor model end-to-end
performance, and Thor2 scores remain separate from Thor3. Initial2warm3 model
screens do not add a second gain to the formal result. Curated evidence includes
raw report/CSV hashes and the independent profiler review; current full-MTP
critical-path f remains MISSING.

Next acceptance is the three-version complete15-case final2warm10 matrix. The
current earlier Final/P1 full15 strict preflight is not a three-version formal
result. Both20% goals and push remain separate decisions.

## Reproduction and evidence

The narrow affected controller uses immutable parent
`final15/artifacts/prefill04-validation` and candidate
`final15/artifacts/routed-quant02-validation-attempt02`. Run strict, then2warm3
append screens, then a separately invoked2warm5 formal only after positive screens.
No timing autotune or fallback is allowed. Root coordinates double locks and the
1530MHz lease. Full artifact identities, sample arrays, raw receipt hashes and
reproduction metadata are in [curated evidence](evidence/routed-input-quant-thor3-affected.json).
The formal archive SHA256 is5f43d53e00430e03e2c316b39e23dbcb3c710f1ccf6e04a186c808470a3bd362.
CLI-comment and architecture-probe workflow failures remain recorded separately
from actual small-domain regressions; original failed evidence is preserved.
