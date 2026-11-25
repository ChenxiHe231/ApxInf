# Thor-2 one-pass text prefill session

This change retains native executions through one real long-prefill traversal,
then drains the stream before releasing the session. It reduces intermediate
resource teardown/synchronization; it does not replay the stateful prefill or
change a CUDA kernel, precision, MTP policy, or native recipe. The existing Thor
NVFP4/dynamic-FP8 profile selects it automatically for text requests above four
rows. Other profiles, image positions, caller sessions and capture retain their
existing boundaries. No new switch or environment variable is introduced.

## Affected-scope acceptance, 2026-11-25

Frozen Final `02f1e638ae76e350826473693e454e772f08ae0c`, Thor-2, GPU1530MHz
(hardware maximum1575MHz), CPU2601MHz, EMC4266MHz; Qwen3.6-35B-A3B-NVFP4,
dynamic-FP8 target KV, FP8-scale1 draft KV, greedy, fixed MTP depth3.
Each arm uses two warmups and five measured requests, without profiling or tuning.
4K pair order is baseline/candidate; 32K is candidate/baseline. Prefix setup is
excluded; appended-prefill time includes the first argmax token.

| Initial KV | Appended input | Final ms | This change ms | Latency reduction |
|---|---|---:|---:|---:|
| 4096 | 4096 | 522.9215044 | 509.3247362 | 2.6002% |
| 32768 | 4096 | 692.6070744 | 672.1537494 | 2.9531% |

[Curated samples and artifact identities](evidence/prefill-session-thor2-affected.json)
record the exact source patch, independent ELF identities, unchanged native archive,
all five samples and original remote evidence location. Existing recipes and model
identity were unchanged before/after. These are affected-scope results. Full15-case
2warm10, long-context1024-token continuation, Decode gain and both20% goals remain
unaccepted at this commit; these numbers must not be represented as a full-model
Decode speedup or added to earlier historical gains.

## Correctness and lifetime

Strict real-checkpoint gates cover8+8, 4096+4096 and32768+4096 text rows. They compare
all selected logits/hidden rows, GDN state, physical KV bytes and dynamic scales
exactly against eager, check committed/read-only state after an injected recoverable
backend error following a real enqueue, retry and four greedy continuation steps.
The short gate also checks caller-session bypass and capture admission/copy.
This does not claim recovery from fatal CUDA faults or full MTP proposer quality.
The operation error is preserved if a secondary cleanup synchronization also fails.

Full-process NSys memory allocation ledgers show dynamic logical peak +4MiB
(+0.01145%), balanced dynamic allocation/free at exit in both arms. This is not a
physical residency or per-native-object peak. Candidate04 was newly captured; the
unchanged frozen02 baseline capture was reused with its original raw hashes.

## Reproduction

Build both revisions separately with the same lockfile/native build and recipe
cache:

```sh
cargo build --locked --release -p apxinf-model --features qwen35-cuda --example qwen35_context_bench
cargo test --locked --release -p apxinf-model --features qwen35-cuda --lib --no-run
```

The context example arguments for both cases are retained verbatim in the evidence
JSON. Run each revision's immutable ELF under the same fixed-frequency lease,
using the same prequalified existing recipe/configuration map. Formal runs must
require existing recipes and forbid autotuning/fallback; do not use the tuning
process as a timed baseline. Run the exact ignored gates
`qwen35moe::mtp_runtime::gpu_raw_protocol::real_checkpoint_prefill_session_lifecycle`,
`...::real_checkpoint_prefill_session_lifecycle_4k_4k` and
`...::real_checkpoint_prefill_session_lifecycle_32k_4k` from the library-test ELF with
`--exact --ignored --nocapture`, absolute checkpoint and recipe paths. These gated
hardware tests need the established explicit GPU-test authorization/configuration;
never run the whole ignored-test set or change frozen precision tolerances.

NSys is diagnostic only and uses graph-level tracing; its timing never enters the
table. Earlier same-implementation traces showed unchanged launch signatures and
memory traffic with fewer intermediate synchronization calls, consistent with better
host enqueue overlap. Opaque graphs and overlapping work prevent interpreting
summed kernel times as removable critical-path time. No NCU kernel acceleration is
claimed because native CUDA code is unchanged.
