# GDN causal carry correction and independent reference

The fused GDN prefill preparation now reads the incoming convolution history
for the first three rows of a segment. Previously a negative segment-local
source index contributed zero, even when the caller supplied a nonzero history.
This lost the causal prologue on append-prefill and on any supplied nonzero
initial state. It is a correctness correction, with no performance claim.

The change is limited to the private `fused_conv_prepare_kernel` in
`crates/apxinf-cuda-new/native/kernels/custom/gdn_ops.cu`: pass the existing
window pointer and read `window[channel*4 + source_token + 4]` when
`source_token < 0`. For the first row the live old taps are slots1/2/3;
slot0 is discarded by the causal shift. Current-segment rows are unchanged.
The final-window update still runs after preparation on the same stream, so it
does not overwrite history while preparation reads it. Public ABI, the separate
prefill-only causal-convolution API, scan math and numerical thresholds do not
change. A zero incoming window retains the original mathematical contribution.

## Actual validation and limits

[Curated evidence](model-microbench/evidence/gdn-causal-carry.json) binds the
original/corrected native SHA, actual build and process receipts, test/support
sources, raw-log hashes and independently generated CPU assets. The actual GPU
build was base8278871 plus the recorded dirty source overlay; it was not an
isolated clean build of this proposed correctness commit. Its relevant native,
test and support postimages match the files in this change.

The original GPU literal test passed the cold-zero control, then failed the
nonzero-history case: Q/K/V had12,288/12,288/20,480 incorrect **bytes**. Initial
uploads, readonly inputs, alpha, final window and outer guards passed. After
the minimal fix both literal tests passed every output byte and guard.

The independent mathematical GPU gate then passed all seven provider cases:
scalar-BF16 and Cake-F16 at64/128/2048 rows, plus Cake64→64 append. Each compares
all gated output rows and complete recurrent/convolution state, then runs the
next scalar decode and checks again. It also verifies readonly inputs and a
reserved next-output row. The fixed literal and mathematical tests ran in
fresh strict processes under the common GPU lock; all six test processes exited0,
were reaped before unlocking, and left the recipe manifest unchanged.

These are synthetic GDN operator/state checks. They do not certify full model
quality, model-level append/session integration, MTP acceptance or snapshots,
independent mathematical prefill lengths beyond2048 tokens, end-to-end latency, or the
15-case30% performance target. Larger GDN scheduling comparisons against a common
GPU scalar path are a different type of evidence and cannot replace this
independent mathematical reference.

## Frozen numerical contract

| Quantity | Required check |
| --- | --- |
| Convolution state | all bytes exact |
| Literal Q/K/V, alpha, window and outer guards | all bytes exact |
| Complete recurrent state | cosine≥0.9998 and relativeL2≤0.025 |
| All gated outputs and every output token | cosine≥0.999 and relativeL2≤0.05 |
| Scalar normalized-convolved output and every row | cosine≥0.99999 and relativeL2≤0.01 |
| Decay/beta | max_abs<0.00002 |

Max-absolute errors are also reported for GDN outputs/state; no new absolute
threshold is substituted for the existing cosine/relative contracts. The
independent container's v2 metrics convert finite F32 values to F64 before
squaring, use explicit zero-norm cases and reject empty/nonfinite inputs. They
do not hide small-norm failures behind an epsilon floor. The separate actual
CPU suite passed12 parser/metric tests, including22 corrupt-container negatives
and single-bad-token/tiny/subnormal/zero-norm cases. No GPU threshold was relaxed.

## Reproduce the CPU assets from source

The std-only generator is
`crates/apxinf-cuda-new/tests/reference/qwen35_gdn_reference.rs`. It is the
previously frozen independent `oracle.rs`, changed only to import the maintained
v2 container module by its repository-relative path and remove one extra empty
line at EOF. The container's writer and
data format are unchanged; v2 corrected measurement, not recurrence math. The
standalone parser/metric tests likewise only change their module path.

The scalar reference explicitly models convolution history shift, SiLU/BF16
rounding, Q/K L2 normalization, decay/beta, rank-one delta recurrence, RMS gate
and all final states. It uses fixed nonzero finite BF16 inputs and weights,
nonzero convolution/recurrent initial states,16 key heads,32 value heads,
head dimension128, and convolution width4. It never calls a CUDA implementation,
model projection, checkpoint or vendor reference.

The original CPU source passed5 tests and generated the three frozen assets on
AArch64 Linux with Rust1.98.0. The promoted path/v2-import source was then actually
compiled and rerun on that CPU: generator5/5 and parser/metric12/12 passed, and
all three freshly generated assets matched the original SHA256 exactly. The
curated evidence binds the new source, generator/test ELFs, nine stage logs and
the new reproducibility receipt. This reproduction did not use a GPU. Compare
all three SHA values again for any new reproduction. F32 transcendental results can
depend on the CPU/runtime, so a mismatch on another platform must be investigated,
not accepted by rewriting the frozen hashes or broadening GPU tolerances.

From the repository root, using an available Rust compiler and a new ignored
artifact directory:

```sh
task_root="$PWD/devlocal/gdn-carry-reproduce"
mkdir -p "$task_root/assets" "$task_root/parser-tmp"
rustc --edition=2021 -O crates/apxinf-cuda-new/tests/reference/qwen35_gdn_reference.rs -o "$task_root/gdn-reference"
rustc --edition=2021 -O --test crates/apxinf-cuda-new/tests/reference/qwen35_gdn_reference.rs -o "$task_root/gdn-reference-tests"
"$task_root/gdn-reference-tests" --test-threads=1
"$task_root/gdn-reference" --steps 64 --out "$task_root/assets/gdn-math-64.bin"
"$task_root/gdn-reference" --steps 128 --out "$task_root/assets/gdn-math-128.bin"
"$task_root/gdn-reference" --steps 2048 --out "$task_root/assets/gdn-math-2048.bin"
```

The generator refuses to overwrite an existing output. The assets total
149,737,156 bytes and remain ignored; do not commit them or generated executables.
Expected assets:

| Rows | Bytes | SHA256 |
| --- | ---: | --- |
| 64 | 10,503,404 | `28de737072a6d493ab1f1e7745cac8019da8d4649a31b4cafdf604fe08ab5f18` |
| 128 | 14,197,996 | `82cb55dca1caca44710db9224c85ee0609499752548bc042be69b4a5ce1b8cee` |
| 2048 | 125,035,756 | `142a9c454a16c8adb9e8b0ef7fe475e1ac466a447160af94c7bb97dd6ddb4e90` |

Each bundle includes its indicated prefill rows **plus one next-decode row**,
19 typed sections, geometry, finite-value validation and per-section FNV checks.
Bind the complete SHA externally; FNV does not replace it. Verify with:

```sh
APXINF_GDN_MATH_ASSET_DIR="$task_root/assets" python3 - <<'PY'
import hashlib, json, os
from pathlib import Path
evidence = json.loads(Path('doc/qwen35-thoru/model-microbench/evidence/gdn-causal-carry.json').read_text())
root = Path(os.environ['APXINF_GDN_MATH_ASSET_DIR'])
for asset in evidence['cpu']['assets']:
    path = root / asset['file']
    assert path.stat().st_size == asset['bytes']
    assert hashlib.sha256(path.read_bytes()).hexdigest() == asset['sha256'], path
    print('SHA256 exact:', path.name)
PY
rustc --edition=2021 -O --test crates/apxinf-cuda-new/tests/reference/qwen35_gdn_asset_tests.rs -o "$task_root/gdn-asset-tests"
APXINF_GDN_MATH_ASSET_DIR="$task_root/assets" APXINF_GDN_MATH_TEST_TMP="$task_root/parser-tmp" "$task_root/gdn-asset-tests" --nocapture --test-threads=1
```

Expected CPU counts are5 generator tests and12 parser/metric tests. These CPU
commands do not use CUDA, sudo, a model or network access. Parser temporary files
are confined to its uniquely created directory below the supplied `devlocal`.

## GPU reproduction

Only compile during the safe small-pool build phase under the shared
`/tmp/apxinf-gpu.lock`, with jobs2/nvcc2. The Cargo payload is:

```sh
cargo test -p apxinf-cuda-new --release --test qwen35_gdn_causal_prologue --test qwen35_gdn_independent_math --no-run -j2 --message-format=json
```

The root runner selects actual ELFs from the Cargo artifacts and records their
hashes. After owned compilation jobs are reaped, run the two prologue tests
and four independent-math tests as six separate fresh processes, already holding
the common GPU lock. Use `APXINF_REQUIRE_RECIPES=1` and pass the verified absolute
asset directory as `APXINF_GDN_MATH_ASSET_DIR`. Each invocation uses
`--ignored --exact <test-name> --nocapture --test-threads=1`:

- `gdn_conv_prepare_cold_zero_window_exact`
- `gdn_conv_prepare_nonzero_causal_prologue_exact`
- `gdn_independent_math_64`
- `gdn_independent_math_128`
- `gdn_independent_math_2048`
- `gdn_independent_math_prefix64_append64`

Both test constructors disable online autotune. The raw GDN/Cake operators do
not use GEMM recipe preparation; the strict flag is not a claim that model
recipes or a full model were prepared. Expected provider PASS counts are2/2/2/1
for the four mathematical tests, all under the unchanged thresholds. Preserve
source/ELF/asset/log identities, return codes and lock/reaping receipts for each
reproduction. Test runtimes are not inference benchmark results.
