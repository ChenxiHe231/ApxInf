# Decode callback interval timing v2

The official Decode denominator is the monotonic host interval from the first
to last token callback. For1024 emitted tokens it has1023 intervals. TTFT is
entry→first callback; post-last tail is last callback→generate return. All use
one Instant origin and four clock reads. Their sum equals entry→return.
The engine's old total-minus-TTFT includes a different boundary/finalization;
it remains explicitly under legacy_engine_profile and cannot supply the new
formal interval score. Inputs, output budget and sampling/model math do not change.

Decode JSON is explicitly versioned v2 and matrix.run.v2; old/mixed interval
schemas are rejected. Existing CLI arguments and Rust model APIs are unchanged.
The same timing correction is intentionally active by default; it is not a
performance feature flag or an optimization speedup.

Profiler output directories are created on the remote worker after taking the
shared GPU lock. Existing same-prefix artifacts are rejected, preserving failed
runs and preventing stale trace attribution. Formal/tune do not create profile
directories. Real nonempty report/hash and stats-export checks remain required.

Curated evidence(callback-v2-commit.json) binds the exact tool/helper sources,
original pure-std Rust9 CPU and Linux Python53 CPU receipts, and a fresh private
Python53 test run. These are CPU/fake-subprocess contracts, not CUDA/model/GPU
acceptance. Original model runs used dirty source overlays and separate ELFs;
this clean draft has not been compiled or run on GPU. No weather/E2E/full15
or throughput improvement is claimed by changing the measurement definition.
