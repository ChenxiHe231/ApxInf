# MTP snapshot hidden-row ownership

`Qwen35MtpCoordinator::capture_context` now freezes an independent GPU copy of the current target hidden row. Previously it cloned the `Tensor` owner, which retained a view of persistent verifier scratch. A later verification using that scratch could overwrite the saved snapshot before restoration. For this model's one-row BF16 hidden state, the owned payload is4096bytes. Allocation and D2D copying occur on the existing CUDA context/stream and finish through the existing capture synchronization. Target/draft cache copying, committed positions, token acceptance and restore sequencing are unchanged.

The original real-checkpoint test failed after verifier reuse: `snapshot target_hidden must survive later verifier reuse`, first differing byte0 (`30` versus saved `180`). The frozen baseline ELF was `6d810f4b61524bea9ed860d3d3b2ce2eeed21247c4390eeec1e69efb9bafd372`. The original controller receipt left causal attribution false; a separately preserved later audit combined that actual failure with the matching shallow-clone source and confirmed this alias. Neither receipt is rewritten.

The added protocol tests exercise actual target/draft tensors and model weights. Test-only tensor observation methods compile under `cfg(test)`; no release getter or production API is added. Public tests retain20 repeats at each depth1/2/3. Raw-bank tests retain all nine `(depth, keep)` combinations per target and draft. Target tests cover verify/prepare keeping committed banks readonly, exact checkpoint-row commit copies, valid BF16 KV prefixes, full GDN conv/recurrent copy state, snapshot freeze/restore and physical future-tail canaries. Draft tests cover shifted physical `P-1` rows, copy/commit/snapshot counts, source readonly state and untouched tails. Logical truncation does not require clearing bytes outside the visible range. These are storage/protocol references based on frozen actual GPU source rows and independent CPU byte-range ledgers; they are not independent GDN/attention mathematical oracles.

## Actual short-checkpoint evidence

The receipts in [the curated evidence](evidence/mtp-snapshot-hidden-ownership.json) bind source patch/untracked hashes, ELF SHA, raw log SHA, actual graph configuration, immutable recipes, GPU clocks, shared-lock ownership and child reaping. All runs use the real Qwen3.6-35B-A3B NVFP4 checkpoint, BF16 target KV, FP8 draft KV with unit scale, strict pre-tuned recipes and disabled optional optimization flags. The checkpoint fixture uses a16-token prompt. The clock receipts report the actual `simple_ondemand`54..1206MHz configuration; these correctness runs are not fixed-clock performance benchmarks.

| Execution | Original public MTP snapshot protocol | Original public target protocol | Raw target bank / draft bank |
|---|---:|---:|---:|
| Graph0 |180 records, pass |180 records, pass |9 /9 records, pass |
| Graph1 with full GDN graph phases |180 records, pass |180 records, pass |9 /9 records, pass |

Graph0 public ELF: `e5d0f8c1c0941b1aa022e527bdbe457607f0485883e97d3967759cd3b35b5fc0`; unit-test ELF: `2c9146e6e6a3ae8eaa12844b1f4bb2136e4b115f3dd78df54b0138959de2742e`. Graph1 public ELF: `768b32db36822ddf3b4a770822934fce545a01b163291df02bff99a747d2f749`; unit-test ELF: `d5c3c3f884560fd8a80d6d1c29d0199026f922cbbe6cc12548aef24d645abc49`.

Graph1's first controller invocation incorrectly expected the old config prefix and reported a false negative after a passing GPU test. The corrected `graph1-gpu-02` first entry reuses that exact `graph1-gpu-01` log: it is one actual GPU execution, not two. Its public-target and raw-bank entries are fresh GPU processes. The evidence retains original failure/correction receipts and the independent audits. Graph0 used the earlier public source hash `10ca3421…`; Graph1 uses the maintained graph-parameterized `c2b69f56…`. Both original GPU protocol function bodies are byte-identical; their exact hashes are recorded. Invalid graph-mode values are rejected instead of silently selecting another mode.

These actual runs came from identified **dirty8278871 overlays**, with other optional code present but its controls disabled. This isolated commit based on published `5a15634b` has not been clean-built or GPU-rerun. Its production difference is only the hidden-row deep copy; the remaining Rust changes are test-only. It adds no BF16 fused attention, projection-session candidate, future long-context diagnostic, quantization policy or tolerance change.

## Reproduction scope

Build only the model unit target and this public integration target, through the existing Thor small-pool build/resource guard:

```bash
cargo test --release -p apxinf-model --no-default-features \
  --features qwen35-cuda --lib --test qwen35_real_state_protocol \
  --no-run --jobs 2
```

Before any GPU payload, own `/tmp/apxinf-gpu.lock` through loading, execution and child reaping. Use independent process/log paths, absolute checkpoint and recipe paths, and audit recipe inventories before/after. Supply `APXINF_QWEN35_CHECKPOINT`, `APXINF_QWEN35_PARITY_TUNING_DIR`, prompt length16, target BF16/NVFP4, and the disabled optional controls recorded in the evidence. Run these three exact entries serially in their matching built executables:

```text
PUBLIC_ELF --ignored --exact real_mtp_short_snapshot_different_continuation_protocol --nocapture --test-threads=1
PUBLIC_ELF --ignored --exact real_target_verify_rollback_short_snapshot_protocol --nocapture --test-threads=1
UNIT_ELF --ignored --exact qwen35moe::mtp_runtime::gpu_raw_protocol::real_checkpoint_target_and_draft_raw_bank_protocol --nocapture --test-threads=1
```

The public tests and raw fixture set strict recipes and disable online tuning. Set `APXINF_QWEN35_PROTOCOL_GRAPHS=0` for Graph0; use `=1` and `APXINF_QWEN35_GRAPH_GDN=full` for the Graph1/full gate. The separately ignored `qwen35moe::mtp_runtime::gpu_raw_protocol::prepare_real_checkpoint_protocol_recipes` is the **only** recipe-preparation entry allowed to enable tuning; it requires its exact full name, `--ignored --exact` and `--test-threads=1`. Run preparation separately, record writes and then start fresh strict processes. Never invoke all ignored tests as a shortcut. Preparation is not a numerical or performance pass.

## Remaining limits

These short protocol gates do not certify all P2, EOS handling, calibrated FP8 attention, long-context mathematics, all1024 generated-token scalar/MTP equality or all15-case performance. In particular, a separate same-process4K `--mode both` run with new optimizations disabled produced scalar token15841 versus MTP3691 at output index1; that unresolved long-context issue is not fixed or waived here. The public short protocol does not override it. This change has no claimed throughput benefit, AIOS/weather end-to-end result or30% per-case improvement.
