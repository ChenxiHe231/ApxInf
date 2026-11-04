# Optional eager verifier execution sessions

APXINF_QWEN35_VERIFY_EXECUTION_SESSIONS=1 enables persistent eager resources;
unset/0 keeps the original verifier path. Existing CUDA_GRAPHS=1 takes Graph
precedence. This does not implement or relabel the pre-existing Graph feature.
No public model Rust API, arithmetic, sampling, KV precision/scale or commit
publication semantics change. The switch does not alter the1024 output budget.

For each fixed verifier width2/3/4 and each physical committed/shadow bank,
GDN owns mixer+MoE; full attention runs its live KV/position traversal outside
the session and owns only the MoE tail. Existing phase/bank swaps and resets
remain. Eager preparation executes once, without CUDA Graph capture; subsequent
actual traversals reuse native ordered execution resources. Missing/reordered
bindings/recipes fail under existing strict contracts. There is no global cache.
Fixed arenas total40layers×2banks×3widths×4MiB=960MiB, plus unmeasured native
resources; construction preparation cost must be counted separately from Decode.

The fixed-traversal evidence applies to the tested Edge routed-MoE, stable
context/weights and capacity≥4. Reference per-expert routing is data-dependent
and is not certified. Enabling at capacity1..3 still prepares the existing
width4 profile and can reject construction; no generic small-capacity claim.

Original4K measurements(BF16target/NVFP4/greedy fixeddepth3,1206MHz,
2warm+5formal,1024tokens/1023 callback intervals) were9.568961578tok/s for the
older eager-control ELF and82.125252650tok/s for new eager sessions. Those two
runs have different source/ELF identities; this is a historical eager-path
overhead comparison, not a clean same-ELF ablation. The same new ELF with the
pre-existing Graph1 feature reached88.544841354tok/s, which is faster than the
new eager path. Do not claim82.125 is better than the best existing backend.
All seven output digests matched694f5386c486b392 in each run; that is repeat
conformance for this case, not independent mathematical/model-quality proof.

The curated evidence JSON binds actual dirty-source patch/ELF/model hashes,
commands, full measured samples, callback boundaries, fixed clocks, recipe
freeze and lock/reap receipts. Original binaries also contained disabled
projection/BF16/materialize candidates and P2 test hooks. This draft excludes
them; its two production postimages match the original patch with only its
P2 raw-test EOF removed. No isolated clean-draft Cargo/GPU, weather/E2E/full15,
all-depth quality or30%-per-case acceptance is claimed. Root must independently
compile/validate the isolated draft before publishing any stronger conclusion.
