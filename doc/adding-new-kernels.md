# Adding or Extending CUDA Kernels in ApxInf

This document is the kernel development procedure for `crates/apxinf-cuda-new`. The old `crates/apxinf-cuda` is no longer the recommended path for new implementations.

There are two reviewed lifecycles. GEMM/Attention are prepared because they tune, own provider state, or use workspace. Gather/Norm/Pointwise/Quantization/RoPE are direct because each semantic has one stateless, zero-workspace implementation. Choose the lifecycle from concrete resource and selection requirements; do not copy the prepared framework into a simple operator for structural uniformity.

Before starting, read the [`apxinf-cuda-new` architecture](../crates/apxinf-cuda-new/README.md) and the [CUDA L3 operator catalog](../crates/apxinf-cuda-new/cuda-operator.md). This document does not repeat the architecture; it specifies only the files and interfaces that must change, prohibited practices, and acceptance criteria.

Temporary probes, logs, and benchmark results must be placed in `devlocal/<feat-name>/` according to [`AGENTS.md`](../AGENTS.md), not mixed into production source directories.

## 0. Identify the Layer to Change First

| What Changes | Layer | Main Directories |
| --- | --- | --- |
| Model-visible mathematical semantic or tensor contract | L3 Rust Semantic | public API in `src/ops/<operator>/`, `cuda-operator.md` |
| normalize, direct launch, execution cache, session, or graph lifecycle | L2 Rust | `src/ops/<operator>/`, `src/workspace.rs`, `src/graph.rs`, `src/ffi/abi/` |
| C ABI, recipe, kernel selection, autotune, fallback, candidate, or provider adaptation | L1 C++ Native operator | `native/include/`, `native/adapters/`, `native/framework/` |
| CUDA implementation, template instance, or vendor patch | L0 Kernel | `native/kernels/`, `native/patches/`, `build.rs` |

```text
model → L3 Rust → L2 Rust → C ABI → L1 C++ → L0 CUDA
```

A new L0 kernel must first be integrated into L1; L2 calls L1 only through the C ABI; models call only L3. See the [`apxinf-cuda-new` architecture](../crates/apxinf-cuda-new/README.md) for the core objects and keys.

## 1. Single Decision Table

First record the mathematical semantic, shape, layout, dtype, quantization, mask, scale, target GPU, workspace, CUDA Graph requirements, determinism, numerical tolerance, and performance target. Then classify the change only by the following table.

| Decision | Change Type | Section |
| --- | --- | --- |
| The catalog already has the same semantic, and an existing candidate meets correctness and performance requirements | Reuse directly | 2 |
| The semantic exists, and the same candidate can safely cover the new shape/dtype/device | Extend a candidate | 3 |
| The semantic exists, but a different implementation or technology stack is required | Add a candidate/provider | 4 |
| The public mathematical semantic or a contract that the caller must express differs | Add an L3 semantic | 5 |
| A new semantic cannot reasonably reuse an existing family's Spec, Bindings, or execution lifecycle | Create an operator family under Section 5 | 5 |

Before adding an execution cache, answer all four questions: does the operator have multiple candidates/configurations requiring selection; does it create descriptors/handles/prepacked state; does it own workspace beyond caller outputs; does capture require a resource created beforehand? If every answer is no, use `normalize → *_launch` and keep it out of `ExecutionSession`. If any answer is yes, use the prepared lifecycle and document the owned state.

Do not add a semantic in the following cases: only the provider changes; only a configuration, shape, dtype, alignment, or GPU is added; only packing, workspace, or epilogue changes; only the model function name differs; fallback is correct but too slow. If uncertain, do not add a semantic. First complete a comparison of the public mathematical contracts; if uncertainty remains, submit an interface review rather than letting the implementer choose the layer independently.

## 2. Reuse an Existing Semantic Directly

| Category | Requirement |
| --- | --- |
| Required files | Modify only the model caller and related model tests; do not modify the native registry/provider |
| Required interface | Call only the safe Rust L3 API; do not call raw FFI, a provider, or a CUDA kernel |
| Required tests | Correctness on real model shapes, target-GPU provider summary, and end-to-end profiling |
| MUST NOT | Do not duplicate a semantic; do not branch on GPU/shape/provider in the model layer; do not bypass the registry |

A provider-selection test cannot replace a performance test; results on one GPU architecture cannot replace acceptance on the target architecture.

## 3. Extend an Existing Candidate (Prepared Lifecycle Only)

Use this path only when the underlying implementation already has the capability. A `true` result from `supports()` is a correctness guarantee, not a performance hint.

This section applies to GEMM/Attention candidates. To extend one of the five direct families without changing its mathematical semantic, edit its flat `native/adapters/<operator>.cu` adapter and L0 kernel directly; do not introduce the files or callbacks below.

### Required Files

| File | When to Modify |
| --- | --- |
| `native/adapters/<operator>/candidates.cpp` | Extend supports/alignment/resource/configurations |
| `native/adapters/<operator>/providers/*` | The provider adapter must handle the new contract |
| `native/kernels/*` | A new kernel branch or template instance is required |
| `build.rs` | Add new sources, includes, macros, or target architectures to the build |
| `build_support/<operator>_fingerprint.rs` | New inputs are outside the fingerprint's current coverage |
| `src/ops/tests/precision/*` | Add an independent-reference all-candidate test for the expanded domain |
| `src/ops/tests/l3_behavior.rs` | The public contract boundary is affected |

### Required Interfaces

| Interface | MUST |
| --- | --- |
| `supports(spec)` | Cover the semantic, shape, dtype, layout, mask, scale predicate, and quantization exactly |
| `alignment_requirements(spec)` | Declare the minimum alignment for each binding |
| `resource_requirements(spec[, configuration])` | Report resource requirements knowable before provider construction so the workspace budget can prefilter candidates; Attention passes the enumerated configuration because its workspace may vary by tactic, while GEMM is currently spec-only |
| `enumerate_configs(spec, out)` | Return every stable configuration that requires independent benchmarking |
| `prepare(execution)` | Construct complete provider state for every enumerated configuration |
| `enqueue(execution)` | Be correct and asynchronous across the new domain |
| `destroy(execution)` | Cover every successful prepare path |

### Prohibited Practices and Tests

MUST NOT: broaden `supports()` first and depend on runtime failure; read input values or pointer contents to select a kernel; enumerate only configurations predicted to be fastest; change the meaning of an old configuration number; add a shape-specific API; omit the target SM or a fingerprint input.

Required tests: all-candidate reference for the new shape/dtype; alignment/workspace boundaries; prepare/enqueue/destroy for every new configuration; capture/replay; target-GPU compile and link; cold-cache tune and warm-cache hit; real-shape benchmark.

## 4. Add a Candidate or Provider (Prepared Lifecycle Only)

The semantic must already exist, and the provider name must not enter the public API.

This section applies to prepared families. A direct family must remain a single fixed implementation; if it genuinely acquires selection or persistent-resource requirements, its lifecycle change requires a separate architecture review rather than a transitional second path.

### Required Files

| File | MUST |
| --- | --- |
| `native/adapters/<operator>/internal.h` | Declare provider callbacks and Execution state |
| `native/adapters/<operator>/candidates.cpp` | Register the identity, attributes, and all callbacks |
| `native/adapters/<operator>/providers/<provider>.*` | Thin provider adapter |
| `native/kernels/<operator-or-provider>/` | Kernel, instance, or vendor source |
| `build.rs` | Compile and link for the target architecture |
| `build_support/<operator>_fingerprint.rs` | Cover candidate/ABI/kernel inputs |
| `src/ops/tests/precision/*` | Independent-reference all-candidate tests |
| `src/ops/tests/backend_framework.rs` | Selection, resource, and lifecycle regression tests; add corresponding regressions when recipe/fallback is used |

A third-party provider must also include a `README.md` or `VENDOR.md` recording the upstream revision, license, and local modifications; place patches in `native/patches/`. Prefer a compatibility layer or build-time patch over modifying the vendor snapshot directly.

### Stable Identity and Callbacks

Every candidate must have a unique and stable `(provider_id, implementation_id, implementation_version)`. Do not reuse an identity for another implementation; configuration number meanings must remain stable; update the implementation version when an old configuration cannot be restored with its original meaning; include every implementation input in the operator build fingerprint.

The current GEMM/Attention candidate descriptors use the following interfaces. When extending these two families, implement and register all of them:

1. `supports(spec)`;
2. `alignment_requirements(spec)`;
3. `resource_requirements(spec, configuration)` for Attention and
   `resource_requirements(spec)` for GEMM;
4. `enumerate_configs(spec, out)`;
5. `prepare(execution)`;
6. `enqueue(execution)`;
7. `destroy(execution)`.

Also set `required_device_features`, `graph_safe`, `deterministic`, `fallback`, and a diagnosable `name` accurately.

A new operator family may define a different descriptor, but its adapter must map it without loss to the framework-required `Problem::registry/supports/configurations/prepare/enqueue/stream/graph_safe` protocol and explicitly represent capabilities, resources, configuration, and destroy lifecycle. Do not modify the common framework only for naming preference; deviations from the current descriptor template must be justified in that operator's architecture review.

`prepare` may create handles, descriptors, algorithms, prepacks, and workspace, but must obey `resource_limit`. `enqueue` may only use prepared state to submit work to the bound stream; it must not tune, allocate long-lived resources, initialize lazily, or synchronize.

### Fallback, Prohibited Practices, and Tests

- Every semantic must have an explicit baseline fallback; high-performance specializations are not fallback by default.
- The fallback must pass complete semantic tests; fallback selection belongs only to the operator adapter.
- The registry must not store per-call state; the model layer must not select providers; the framework must not contain operator special cases.
- Do not treat "can launch" as numerical correctness, and do not modify vendor source without a record.

Required tests: every candidate/configuration against an independent reference; unsupported filtering; identity/version/configuration restoration; resource limits; fallback allow/deny; eager and graph consistency; target-GPU cold/warm cache and performance.

## 5. Add an L3 Semantic

Implement in the following order. Do not register a provider specialization before the public contract is settled.

### Step 1: L3 Rust Args + L2 normalize/lifecycle

Required files:

```text
src/ops/<operator>/contracts.rs (or the existing family contract)
src/ops/<operator>/<semantic>.rs
src/ops/<operator>/launch.rs (direct lifecycle)
src/ops/<operator>/execution.rs (prepared lifecycle only)
src/ops/<operator>/mod.rs
src/ops/mod.rs
src/lib.rs
```

| Required Interface | MUST |
| --- | --- |
| `<Semantic>Args` | Express only the public mathematical semantic; make invalid combinations unrepresentable where practical |
| `normalize(ctx, args)` | Validate device/dtype/shape/layout/alias/overflow and produce direct `Spec + Bindings` or prepared `Spec + Policy + Bindings` |
| `<semantic>(ctx, args)` | Perform only `normalize → launch` for direct families or `normalize → execute` for prepared families |
| public exports | Export correctly from the operator module and crate root |

Normalize MUST: Spec stores only structural information required for the semantic and legality; Bindings stores addresses, the stream, and dynamic values. Prepared families may additionally use Policy for resource and selection constraints and normalize addresses to alignment classes when candidate selection depends on them. Direct families expose no Policy. Keep storage alive through submission; equivalent calls produce the same Spec. A new operator does not need to copy every GEMM/Attention field.

### Step 2: Stable L2/L1 C ABI

Required files:

```text
native/include/apxinf_cuda/<operator>_types.h
native/include/apxinf_cuda/<operator>.h
src/ffi/abi/<operator>.rs
src/ffi/abi/mod.rs
```

Prepared/tunable ABI:

```c
*_prepare(runtime, spec, policy, bindings, &execution);
*_enqueue(execution);
*_destroy(execution);
```

Stateless direct ABI:

```c
*_launch(runtime, spec, bindings);
```

A direct launch must validate the complete ABI contract, submit its fixed implementation asynchronously, and remain legal during CUDA Graph capture. It must not expose Policy, create a candidate registry, allocate a heap execution, enter the Rust execution cache/prepared sequence, initialize lazily, tune, or synchronize. Do not provide both ABIs for one family as a transition layer.

MUST: Spec has an explicit version that changes when layout or semantics change; use only C-compatible fixed-width fields and opaque pointers; Rust and C declarations match exactly; exceptions become status/last-error through the common ABI boundary; no failure path leaks an execution or provider resource.

### Step 3: L1 C++ Operator Adapter

Direct lifecycle required file:

```text
native/adapters/<operator>.cu
```

Prepared lifecycle required files:

```text
native/adapters/<operator>/internal.h
native/adapters/<operator>/candidates.cpp
native/adapters/<operator>/execution.cpp
```

When persistent tuning is used by a prepared family, also add:

```text
native/adapters/<operator>/tuning_key.cpp
native/adapters/<operator>/autotune.cpp
```

Direct adapters have no `Implementation`, `ImplementationRegistry`, configuration, provider callback, or `Execution`: the flat adapter validates and calls the fixed L0 kernel. Prepared adapters use `Spec`, `Implementation`, `ImplementationRegistry`, and `Execution`. When persistent tuning is used, also add `TuningKeys` containing only one exact key.

| Required Function | MUST |
| --- | --- |
| `registry(spec.semantic)` | Return the immutable candidate set for the semantic |
| `tuning_keys(spec, policy, device)` | For a tunable operator, construct the single exact key |
| `tune(spec, policy, bindings, device, report)` | For a tunable operator, adapt `framework::autotune` |
| operator `prepare(...)` | Prepared lifecycle only: validate candidate/policy/resource and create an Execution |
| C ABI `*_prepare` | Prepared lifecycle only: validate the ABI, look up recipe/tune/fallback, and return the execution |
| C ABI `*_enqueue` | Prepared lifecycle only: enqueue the prepared execution |
| C ABI `*_destroy` | Prepared lifecycle only: safely release all state |
| C ABI `*_launch` | Direct lifecycle only: validate and asynchronously launch the fixed L0 implementation |

| L1 Prepare Strategy | Implementation |
| --- | --- |
| Single stateless, zero-workspace implementation | Flat adapter validates + direct launch; no Policy/candidate/Execution/cache |
| Single implementation with persistent state/resources | Validate + prepare directly |
| Stable heuristic | Select from Spec/Policy inside the adapter; add `selection.cpp` only when the logic becomes complex |
| Measurement required | Exact recipe lookup; call `framework::autotune` on a miss |

| MUST NOT |
| --- |
| Add semantic-specific shape/dtype/registry/fallback logic to `native/framework` |
| Add a compatible/hint key or runtime selection-mode metadata/dispatcher |
| Put pointers/streams into a persistent key |
| Perform first prepare inside capture; allocate/cache/synchronize in a direct launch |
| Put a direct family under `native/adapters/<operator>/` or add a registry/provider callback for its sole implementation |

### Step 4: L1 Provider Callbacks (Prepared Only) + L0 Kernel

Required files:

```text
native/adapters/<operator>/providers/ (prepared lifecycle only)
native/kernels/<operator-or-provider>/
build.rs
build_support/<operator>_fingerprint.rs
```

Prepared families register at least one fallback that fully implements the semantic. Existing GEMM/Attention candidates must implement the seven callbacks listed in Section 4; every candidate must have a stable identity, complete device/policy constraints, and build-fingerprint coverage. A direct family has no provider registry or fallback selection; its flat adapter calls its fixed L0 implementation.

### Step 5: Catalog and Tests

Required files:

```text
cuda-operator.md
src/ops/tests/operator_doc.rs
src/ops/tests/l3_behavior.rs
src/ops/tests/precision/precision.rs
src/ops/tests/precision/generate_torch_l3_fixtures.py (when a Torch golden is required)
src/ops/tests/backend_framework.rs (when new execution behavior is introduced)
tests/public_ops.rs (when public exports/session behavior changes)
```

MUST: add a unique catalog marker and Rust semantic metadata; add an independent semantic test; make the catalog test reject missing, unknown, and duplicate semantics. For a direct family, add an independent numerical reference plus eager/CUDA Graph capture/replay coverage proving no prepare/cache is needed. For a prepared family, independently test every candidate/configuration and add prepare/capture/replay tests for binding/resource behavior.

## 6. Recipe Invariants for Tunable Operators

An operator with only one fixed implementation and no persisted selection may omit recipes. Any operator that tunes among multiple candidates/configurations and persists the winner must follow this section.

| Scenario | MUST |
| --- | --- |
| exact hit | implementation/version/configuration still exists, and supports/device/alignment/policy/prepare all succeed; use it directly without benchmarking |
| miss | When `online_tune=true`, fully benchmark every legal candidate/configuration |
| invalid/corrupt | Treat as a miss; do not reuse approximately |
| winner | Store only provider, implementation, version, and configuration |
| real prepare fails after tuning | Fallback only when `allow_fallback=true` |
| tuning disabled, fallback enabled | Use only the explicitly marked fallback |
| tuning and fallback disabled | Return an explicit cache miss/unsupported error |
| fallback execution | Do not persist as a tuned winner |

The exact key must cover recipe schema, operator build fingerprint, target GPU, defined compatible versions of CUDA/critical libraries, Spec fields required for the tuning equivalence class, workspace, graph-safe, deterministic, and alignment class. Any critical change must miss; do not add a second-level compatible key.

The autotuner does not validate numerical correctness; prepared-family all-candidate tests must guarantee it.

## 7. ExecutionSession and CUDA Graph Invariants

```rust
let session = ExecutionSession::with_capacity(workspace_bytes, device)?;
prepare_with_session(&session, || forward())?;
let graph = capture(&ctx, || with_session(&session, || forward()))?;
graph.replay()?;
```

| Stage | MUST | MUST NOT |
| --- | --- | --- |
| prepare | lookup/tune, create executions, allocate resources, record order | Run inside capture |
| with_session | Hit the same Spec/bindings/device/stream/execution constraints/order | Temporarily prepare on a miss |
| capture | Capture only asynchronous enqueue and keep resources alive | Multiple streams, synchronization, lazy initialization |
| replay | Launch the graph directly | Reselect a kernel or modify a recipe |

`graph_safe=true` means only that the prepared enqueue can be captured. Allocation, algorithm selection, and long-lived state initialization must complete in `prepare()`.

## 8. Unified Acceptance

| Acceptance Item | Reuse | Extension | New Candidate/Provider | New Semantic |
| --- | --- | --- | --- | --- |
| Correctness on real model shapes | MUST | MUST | MUST | MUST |
| Independent-reference test | Existing | Every changed candidate, or the direct implementation | Every candidate/configuration | Direct: fixed implementation; prepared: every candidate/configuration |
| Catalog/semantic test | Existing | When affected | When affected | MUST |
| alignment/workspace | Existing | Prepared candidates: MUST; direct: contract boundaries | MUST | Prepared: MUST; direct: contract boundaries |
| hit/miss/invalid recipe | Existing | When the key changes | MUST when recipes are used | MUST when recipes are used |
| fallback allow/deny | Existing | When fallback changes | MUST | Prepared lifecycle only |
| lifecycle/CUDA Graph | MUST | Direct: eager + capture/replay with no prepare; prepared: prepare/capture/replay | prepare/capture/replay | Direct: eager + capture/replay with no prepare; prepared: prepare/capture/replay |
| target-SM compile and link | MUST | MUST | MUST | MUST |
| cold tune, warm hit, profiling | MUST when tuning is used | MUST when tuning is used | MUST when tuning is used | MUST when tuning is used |

Run the complete tests from the repository root:

```bash
bash crates/apxinf-cuda-new/test-new.sh \
  test -p apxinf-cuda -- --nocapture --test-threads=1
```

Finally, verify on the actual target GPU: correct `APXINF_CUDA_ARCH`; new sources are linked; semantic, graph, and applicable recipe tests pass; direct implementations pass independent numerical tests; prepared families pass all-candidate tests; the actual prepared provider/configuration is explainable; cold/warm cache behavior is correct when tuning is used; real-shape benchmarks and end-to-end model numerical/performance targets are met.

Do not mark the work complete if any required interface, required test, target-GPU result, or documentation update is missing.
