# Optional final-proposal materialization execution session

APXINF_QWEN35_MTP_MATERIALIZE_SESSION=1 enables a lazy private execution tape
for the final proposal consumed at full accept plus bonus. Default is off,
allocating no new tape. The first real materialization calls prepare instead
of a second execution; later rounds call run with the same no-LMhead sequence.
Draft already has a different tape including LMhead and is not reused here.

The non-resizable MtpScratch owns one4MiB arena plus native resources. Binding
checks include device/stream, fixed weight pointers, Edge/Reference traversal
count and deterministic policy; native keys/order cover actual A/output views.
Changing bindings creates a new tape. On partial error there is no retry;
existing successful synchronization, validation and publish boundary remain.
One-row GEMM bindings support depths1/2/3; the original actual performance run
was fixeddepth3 only. There is no new public Rust API, model arithmetic, KV
precision/scale, sampling policy or budget change, and no Graph capture is added.

Historical same-ELF Graph1 control versus Graph1+materialize measurements on
4Kinput/1024output with2warm5formal gave88.544841354→90.328646037tok/s.
Median TPOT11.293712708→11.070685147ms is a1.974794% reduction. Each had seven
matching output digests694f5386c486b392. This is a local case result, not a30%
claim and not a full-model mathematical/acceptance oracle. Materialization is
conditional on full accept; partial/rejected rounds have no such work to remove.

The curated evidence JSON binds exact original dirty source/ELF/model identity,
flags, full samples,1023callback intervals, clocks, recipe freeze and lock/reap.
Those binaries also included other disabled candidate code and P2 test hooks;
this isolated draft adds only materialization production hunks and its docs,
excluding original P2 EOF. It does not depend on projection/BF16/prefill/new
protocol optimizations. No clean-draft Cargo/GPU, weather/E2E/full15 or other
depth/long-context acceptance has been rerun. Benchmark source/ELF receipts must
be regenerated before representing this clean draft as tested deployment code.
