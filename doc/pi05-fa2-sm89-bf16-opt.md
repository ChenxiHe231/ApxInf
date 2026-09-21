# Pi0.5 RTX 4090 SM89 BF16 Split-KV Attention

## Summary

ApxInf now supports a BF16 FlashAttention-2 split-KV path for the Pi0.5
action decoder on RTX 4090 / SM89. The optimization keeps the same attention
math and only changes the FA2 kernel variant used for small-query, long-KV MQA
shapes.

The split-KV candidate follows the legacy FA2 eligibility domain rather than a
single model shape:

```text
Sq <= 64
Sk > Sq
num_q_heads > num_kv_heads
head_dim in {128, 256}
dtype == BF16
dense causal or non-causal attention
```

For the measured Pi0.5 H10 profile this corresponds to:

```text
Sq = 10
num_q_heads = 8
num_kv_heads = 1
head_dim = 256
```

Regular FA2 leaves SM occupancy low on this shape. Split-KV partitions the KV
axis, runs more parallel FA2 work, then combines the partial outputs. This adds
a combine kernel but reduces total attention time.

## Implementation

The implementation reuses the vendored FlashAttention-2 source already present
in ApxInf. It does not introduce a new attention algorithm.

The current cuda-new implementation is an independent L1 Attention candidate:

- L0 reuses the vendored FA2 split-KV wrapper and the upstream BF16 D128/D256,
  causal/non-causal instantiations.
- The descriptor has a stable provider/implementation/version identity,
  precise shape/dtype/semantic support, and 16-byte alignment requirements.
- Configurations enumerate legal split counts from the KV tile geometry.
  Online tuning compares those configurations with regular FA2 on the actual
  GPU instead of adding a model-specific dispatch branch.
- `resource_requirements` is configuration-specific. `prepare` allocates and
  owns `softmax_lse`, `softmax_lse_accum`, and `o_accum`; `enqueue` only launches
  on the bound stream and performs no allocation, tuning, or synchronization.
- The Attention build fingerprint includes the adapter and FA2 source set, so
  adding this candidate creates a new recipe namespace. Cold tuning persists
  the chosen implementation/configuration and warm preparation restores it.
- The PI0.5 model has no provider-, candidate-, or architecture-specific
  branch and does not reserve split-KV graph scratch. Provider state owns its
  resources before capture.

Regular FA2 remains a separate candidate for larger shapes and for any shape
where it wins tuning. There is no model-level split-KV switch.

## Historical pre-cuda-new measurements

The correctness and performance numbers in this section were collected before
the cuda-new L1 candidate migration. They explain the optimization target, but
they are not acceptance evidence for the current implementation. Current
acceptance requires independent-reference all-candidate eager and graph tests,
workspace/alignment boundary tests, recipe cold/warm restore, and end-to-end
PI0.5 comparison on the target GPU.

### Correctness

End-to-end BF16 CUDA graph integrity still passes with bitwise equality between
eager execution and graph replay:

```text
eager_vs_graph.bitwise_equal = true
max_abs = 0
relative_l2 = 0
```

Regular FA2 and split-KV FA2 are not expected to be bitwise identical because
split-KV changes the softmax/reduction partitioning order. Operator-level
comparison against regular FA2 showed BF16-scale numerical differences:

```text
2-view-like shape, Sq=10, Sk=522:
max_abs     = 6.1035e-5
mean_abs    = 3.287e-6
cosine      = 0.99999748
relative_l2 = 0.002245

3-view-like shape, Sq=10, Sk=788:
max_abs     = 3.0518e-5
mean_abs    = 2.855e-6
cosine      = 0.99999729
relative_l2 = 0.002327
```

These differences are expected for an equivalent BF16 FA2 split-KV execution
path.

### Performance

Measurements were taken on RTX 4090 / SM89 with Pi0.5 BF16, H=10, token=10,
10 flow steps, graph replay timing, and NHWC RGB input inside the captured
graph.

Using the retuned RTX4090 BF16 tactics file:

```text
2-view H10 token=10:
regular FA2   P50 = 35.35 ms
split-KV FA2  P50 = 31.38 ms
improvement       = 3.97 ms

3-view H10 token=10:
regular FA2   P50 = 47.02 ms
split-KV FA2  P50 = 40.93 ms
improvement       = 6.10 ms
```

The same runs passed the built-in BF16 eager-vs-graph integrity check:

```text
2-view split-KV: eager_vs_graph.bitwise_equal = true, max_abs = 0
3-view split-KV: eager_vs_graph.bitwise_equal = true, max_abs = 0
```

Historical workspace usage, including the old model-owned split-KV scratch:

```text
2-view H10 token=10: capacity = 3,615,987,712 bytes, used = 1,511,330,688 bytes
3-view H10 token=10: capacity = 5,311,271,936 bytes, used = 3,089,184,640 bytes
```

Without the tactics file, the same trend was observed:

```text
2-view H10 token=10:
regular FA2   P50 = 37.28 ms
split-KV FA2  P50 = 33.35 ms

3-view H10 token=10:
regular FA2   P50 = 50.45 ms
split-KV FA2  P50 = 44.12 ms
```

## Platform acceptance

The candidate is compiled with the ordinary FA2 source set for configured CUDA
targets. It is not selected by a PI0.5 architecture branch: target capability,
the exact Attention contract, workspace policy, alignment, and measured tuning
decide whether it is eligible and whether it wins. A result on RTX 4090 / SM89
does not by itself accept Orin or Thor.

For each target platform:

1. Compare regular attention vs split-KV attention on the exact action-stage
   shapes.
2. Check operator-level numerical error.
3. Check end-to-end eager-vs-graph integrity.
4. Prove graph capture/replay with provider resources prepared beforehand.
5. Prove workspace and alignment rejection boundaries and cold/warm recipe
   restoration.
6. Benchmark 2-view and 3-view graph replay latency with the target platform's
   complete, compatible tactics set.
