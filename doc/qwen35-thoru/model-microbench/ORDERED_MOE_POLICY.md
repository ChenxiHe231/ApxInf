# Experimental ordered NVFP4 MoE reduction

This opt-in policy makes the final eight-route reduction deterministic for the tested real layers and shapes. It changes arithmetic; it does not establish full-model quality or a performance improvement. Commit it after the owned context-stream H2D prerequisite.

## Behavior and workspace

Set `APXINF_EDGE_NVFP4_MOE_ORDERED_REDUCE=1` **before starting the process**. Unset or exact `0` keeps the existing reduction; any other string is rejected. A process-fixed snapshot is shared by workspace query and execution, so changing the environment after query cannot change the allocation representation.

The enabled workspace adds `[tokens,8,hidden]` BF16 route storage after the existing FC1 scale region and before the existing trailing 256 bytes. Additional bytes are `tokens * 8 * hidden * 2` for supported shapes (H=2048: 32 KiB per token, 64 MiB at T=2048). The original eight regions retain their offsets. Default workspace size and normal path remain unchanged for valid inputs. New size/count/grid calculations reject overflow.

FC1 clears the expanded route output, and unchanged FC2 AOT receives `num_tokens=tokens*8`, `top_k=1`. Each route output already includes its original routing weight. A new kernel on the same context stream sums slots 0 through 7 from explicit positive zero using `add.rn.f32` without FTZ, then rounds once with `cvt.rn.bf16.f32`. No second weighting, new host synchronization, new AOT, or public ABI is introduced. Small-batch setup, routing/layout order, quantization and FC1/FC2 weights/math are unchanged. No stable-layout performance candidate is included.

This is a **new fixed FP32 accumulation policy over already-rounded BF16 route values**, not bit equivalence with the original BF16 reduction. Fixed FP32 can still lose values through extreme cancellation or overflow. The CPU fixtures contain such counterexamples; the policy is not claimed to be uniformly more accurate or equivalent to exact real summation.

## Recorded correctness evidence

The accompanying [evidence index](evidence/ordered-moe-policy-evidence.json) binds actual run receipts, original log hashes and the three exact tested source postimages. Counts below were independently parsed from preserved logs; this documentation did not rerun GPU work.

| Gate | Actual coverage | Scope |
| --- | --- | --- |
| CPU strict sum oracle | 9/9 tests, 54 frozen finite-input fixtures | Rounding boundaries, subnormals, signed zero, cancellation/overflow; actual log available, no compiler/ELF receipt preserved for this item |
| Standalone reducer | Both regular and `--use_fast_math`: 15 small shapes + 8 slot masks ×20; 4 large shapes ×1 per build | Exact independent frozen FP32-stage bits and final BF16 bytes; readonly inputs, guards and tail words |
| Real-layer integration | Layers 0/1 × T=1/4/2048 ×20 =120 calls | All output bytes match CPU reduction of actual expanded BF16 route values; poison/finite, guards, readonly inputs/weights, workspace/output repeat consistency |
| Eager versus CUDA Graph | Same six cases ×20 eager/graph pairs =240 calls; 12 separate warmup calls | Full-output reference, byte stability, guards and readonly storage through replay |

The integration source hash `47622bed…` is an exact byte prefix of final test `0caea57…`; the final file appends the capture module and was built and run in the original 13-file H2D+ordered overlay. The native runner and reducer header hashes are unchanged across these gates. The old fixed-routing probe already present in the parent revision remains unchanged. No old NaN/FC2-only private diagnostic additions are part of this commit delta.

The real-layer CPU reference validates the **new reduction only**: it consumes actual FC2 BF16 route values, and therefore does not independently validate FC1, FC2, quantization, routing, every layer or model output quality. Fixed routing uses uniform weights 0.125. Full-model accuracy, MTP behavior and all 15 benchmark results require separate gates with unchanged thresholds; these tests do not complete those gates.

## Running the maintained gates

Build on Thor with the existing source and dependency prerequisites; keep the GPU lock through compilation/execution and use the coordinator's memory guard. The following are the test-target arguments, not permission to interrupt a running benchmark:

```bash
cargo test --release -p apxinf-model --no-default-features --features qwen35-cuda --test qwen35_real_moe_bench --no-run --jobs 2
```

Using the resulting test ELF as `$MOE_TEST_ELF`, run the literal contract without CUDA. Run the policy gate in **separate processes** with the policy unset, `0`, `1` and `2`; `2` is an expected rejected-policy case handled by the test.

```bash
"$MOE_TEST_ELF" --exact edge_ordered_reduce_tests::ordered_reference_literal_rounding_contract --nocapture --test-threads=1
"$MOE_TEST_ELF" --exact edge_ordered_reduce_tests::edge_ordered_reduce_workspace_policy_gate --ignored --nocapture --test-threads=1
```

Run both GPU exact tests separately for `APXINF_QWEN35_MOE_PROBE_LAYER=0` and `1`, with `APXINF_EDGE_NVFP4_MOE_ORDERED_REDUCE=1`, `APXINF_QWEN35_CHECKPOINT=/opt/data/models/Qwen3.6-35B-A3B-NVFP4`, and `APXINF_QWEN35_MOE_PROBE_TOKENS=1,4,2048`. These tests do not use online autotuning.

```bash
"$MOE_TEST_ELF" --exact edge_ordered_reduce_tests::real_checkpoint_edge_ordered_reduce_integration_gate --ignored --nocapture --test-threads=1
"$MOE_TEST_ELF" --exact edge_ordered_reduce_capture_tests::real_checkpoint_edge_ordered_reduce_capture_eager_gate --ignored --nocapture --test-threads=1
```

## Performance and remaining acceptance

A recovered diagnostic Cold2K profile contains 40 ordered-reducer kernels totaling **13.404384 ms**. Separately, a strict H2D+ordered Cold2K run recorded median **486.597620 ms** (two warmups, five measured iterations). They use different ELF/configuration identities and cannot form a before/after comparison. The original failed profiling status remains preserved; recovery only makes its actual trace available for analysis. Neither result establishes a speedup, full-model quality, or the 30% target.

Before considering this policy a complete correctness baseline, retain separate full-model repeat/output gates, independent model-quality evaluation, MTP rollback/acceptance coverage and broader real routing/weight coverage. Before claiming a performance gain, run an isolated matched-source/configuration A/B against the frozen per-case baseline. A failed quality gate must not be hidden by changing a threshold or benchmark timing definition.

The private header retains its original preparation-time comment that it was not wired into production; its body is now wired only by this experimental opt-in. Its exact GPU-tested bytes are preserved here for evidence identity.
