# `apxinf-cuda-new` 架构契约

`apxinf-cuda-new` 向模型层提供稳定的 CUDA L3 语义接口，在 native 层为同一语义选择并准备 provider kernel。GEMM 和 Attention 是当前已接入的两个 operator family，不是框架允许的全部类型。

- 当前公开接口及数学语义：[L3 operator catalog](cuda-operator.md)
- 新增或扩展 kernel 的工作流：[Adding New Kernels](../../doc/adding-new-kernels.md)

## 调用链

```text
Rust L3 API
  → normalize
  → execute / Rust execution cache
  → C ABI *_prepare
  → adapter registry(spec.semantic)
  → operator-owned selection
      ├── Fixed: 直接选择唯一合法实现
      ├── Heuristic: 用稳定规则选择，不测速
      └── Autotune: exact recipe lookup → tune or fallback
  → candidate prepare
  → C ABI *_enqueue
  → provider enqueue
```

CUDA Graph 路径先在 `prepare_with_session` 中走完整准备流程，再在 `capture + with_session` 中只复用 prepared execution 和 enqueue。

## 核心对象

| 对象 | 身份/内容 | 生命周期 |
| --- | --- | --- |
| Semantic | 公开数学与布局契约 | API 级稳定 |
| Provider | kernel 技术栈或实现家族，如 FA2、CUTLASS、cuBLAS | 构建级 |
| Implementation / candidate | `(provider_id, implementation_id, implementation_version)` 和一组 callbacks | registry 级 |
| Configuration | implementation 枚举的离散配置编号 | candidate 级 |
| Recipe | winner 的 implementation identity + configuration | 可持久化、可跨进程恢复 |
| Prepared execution | spec、bindings、device、configuration、provider state 和资源 | 当前进程；通常绑定地址和 stream |

约束：semantic 不等于 provider，provider 不等于 candidate，recipe 不等于 prepared execution。

## 分层契约

### 1. Rust L3 semantic 层

| 契约项 | 规定 |
| --- | --- |
| 上游调用者/输入 | 模型或算子组合层；输入为 `CudaContext`、tensor 和 operator 自己的公开 Args；当前实例包括 `GemmArgs`、`AttentionArgs` |
| 对上接口 | `ops::<semantic>(ctx, args)`；当前接口以 [operator catalog](cuda-operator.md) 为准 |
| 可调用下层 | operator-local `normalize(...)`，然后 `execution::execute(...)` |
| 拥有状态/决策 | 公共 semantic；dtype/shape/layout/device 校验；默认值归一化；Rust storage 生命周期 |
| MUST | 将调用归一化为 ABI `Spec + Policy + Bindings`；记录影响 candidate 合法性的稳定属性，例如需要时使用 alignment class 或语义标志 |
| MUST NOT | 暴露 provider、algorithm、tile 或 configuration；在模型代码中选择 FA2/CUTLASS/cuBLAS |
| 具体文件 | `src/ops/<operator>/`；当前实现为 `gemm/` 和 `attention/` |

`Spec` 是归一化问题描述，`Policy` 是 operator 支持的资源与选择约束（当前 family 包含 workspace/graph-safe/deterministic/tuning），`Bindings` 是本次执行的地址、stream 和动态值。不是所有 operator 都必须有相同字段，也不是所有 binding 值都属于 recipe key。

### 2. Rust execution 与 session 层

| 契约项 | 规定 |
| --- | --- |
| 上游调用者/输入 | Rust L3 API 的 normalized value；可选活动 `ExecutionSession` |
| 对上接口 | operator-local `execute(ctx, normalized)` 和 `prepare(ctx, normalized)`；公共 `prepare_with_session`、`with_session`、`capture` |
| 可调用下层 | ABI `apxinf_*_prepare`、`apxinf_*_enqueue`、`apxinf_*_destroy` |
| 拥有状态/决策 | typed execution cache；固定地址/device/stream key；operator traversal 顺序；资源保活 |
| MUST | capture 前创建 native execution；capture 时只命中已准备 execution；校验相同 operator 顺序、device 和 stream |
| MUST NOT | 在 capture 中 autotune、创建 provider state 或接受 execution cache miss；把 execution cache 当作 recipe cache |
| 具体文件 | `src/ops/*/execution.rs`、`src/workspace.rs`、`src/graph.rs` |

`GraphWorkspace` 是 session 持有的 Rust device arena；`Policy::workspace_limit` 限制 native provider-owned resource。二者不是同一个对象。

### 3. FFI / stable C ABI 层

| 契约项 | 规定 |
| --- | --- |
| 上游调用者/输入 | Rust execution 层；ABI `Spec`、`Policy`、`Bindings` 和 opaque runtime |
| 对上接口 | `apxinf_<operator>_prepare`、`*_enqueue`、`*_destroy`、可选 `*_summary` |
| 可调用下层 | 对应 operator adapter 的 prepare/selection/execution 逻辑 |
| 拥有状态/决策 | C-compatible 类型、opaque execution handle、status/error 边界 |
| MUST | 在 `abi_boundary` 内把 native 异常转换为 `apxinf_status_t`；保持 Rust 声明与 C header 一致 |
| MUST NOT | 固定 semantic 到某个 provider；泄漏 C++ 类型或 provider state 到 Rust |
| 具体文件 | `src/ffi/abi/`、`native/include/apxinf_cuda/`、`native/adapters/runtime.cpp` |

ABI 生命周期固定为：

```text
*_prepare(..., &execution) → *_enqueue(execution) [0..N 次] → *_destroy(execution)
```

### 4. Operator adapter 层

| 契约项 | 规定 |
| --- | --- |
| 上游调用者/输入 | C ABI `*_prepare`；已归一化的 `Spec + Policy + Bindings` |
| 对上接口 | ABI prepare/enqueue/destroy；内部 `semantic_registry` 和 candidate `prepare`；tunable operator 另有 `tuning_keys`、`tune` |
| 可调用下层 | framework `Registry` 和 candidate callbacks；tunable operator 可调用 `autotune`、`read_recipe`、`write_recipe` |
| 拥有状态/决策 | semantic registry 及其 `SelectionKind`；operator validation；selection/fallback；native `Execution`；需要调优时再拥有 exact key 和 recipe 验证 |
| MUST | 用 `registry(spec.semantic)` 隔离 selection domain；使用 recipe 时，hit 后重新验证 identity、device、`supports`、configuration 和 prepare |
| MUST NOT | 把 semantic-specific 筛选、key 字段或 fallback 放进通用 framework；把 recipe 直接当可执行对象 |
| 具体文件 | `native/adapters/<operator>/`；按需要包含 `execution.cpp`、`candidates.cpp`、`tuning_key.cpp`、`autotune.cpp`、`internal.h` |

当前 Attention 和 GEMM 各自拥有 `autotune.cpp`，因为它们要把不同的 registry、输出、bindings、合法性和 prepare/enqueue 适配成通用 `Problem`。未来 operator 只有在需要 autotune 时才提供该适配；测速算法仍只使用 `framework::autotune`。

每个 semantic 在 `candidates.cpp` 中声明一个 `SelectionKind`：`Fixed` 表示唯一合法实现，`Heuristic` 表示 adapter 用 Spec/Policy 的稳定规则直接选取，`Autotune` 表示 exact hit 复用、miss 完整测速。只有启发式规则复杂到需要独立维护时才增加 `selection.cpp`。当前 GEMM 和 Attention 的所有 semantic 都声明为 `Autotune`；本次只建立统一声明和误用保护，不改变现有运行路径。

### 5. 通用 framework 层

| 契约项 | 规定 |
| --- | --- |
| 上游调用者/输入 | Operator adapter 提供的 candidate descriptor 或 `Problem` |
| 对上接口 | `SelectionKind`、`Registry::find/begin/end`、`autotune(Problem&, report)`、`read_recipe`、`write_recipe`、`abi_boundary` |
| 可调用下层 | `Problem::registry/supports/configurations/prepare/enqueue/stream/graph_safe`；CUDA event/graph 和文件系统 |
| 拥有状态/决策 | 通用 registry 容器；warmup/计时/winner 循环；recipe 文件 I/O；通用错误转换 |
| MUST | 遍历所有合法 implementation/configuration；隔离 candidate prepare/launch 失败；返回 winner `Recipe` |
| MUST NOT | include 任一 operator-specific Spec；解释 semantic；访问 provider state；决定 fallback；构造 operator key |
| 具体文件 | `native/framework/registry.h`、`autotune.h`、`tuning_db.cpp`、`runtime_internal.h` |

### 6. Candidate 与 provider 层

| 契约项 | 规定 |
| --- | --- |
| 上游调用者/输入 | Operator adapter；已选 implementation/configuration 和该 operator 的 Spec/Policy/Bindings |
| 对上接口 | 由 operator 定义 candidate descriptor；当前 GEMM/Attention 使用 `supports`、`alignment_requirements`、`resource_requirements`、`enumerate_configs`、`prepare`、`enqueue`、`destroy` |
| 可调用下层 | provider API 或 `native/kernels/` 下的实际 kernel |
| 拥有状态/决策 | provider handle/descriptor、algorithm、workspace、预打包权重和其他 opaque `provider_state` |
| MUST | adapter 能将 descriptor 映射为 framework `Problem::registry/supports/configurations/prepare/enqueue/stream/graph_safe`；能力、资源和生命周期约束必须可在 prepare 前判断；enqueue 不分配、不调优；destroy 释放所有 provider state |
| MUST NOT | 接受未声明支持的输入；在 enqueue 改变 algorithm；把 provider state 放入 framework |
| 具体文件 | `native/adapters/<operator>/providers/`、`native/kernels/` |

## Tunable operator 的 exact recipe 状态机

当前 GEMM 和 Attention 都是 tunable operator，各自只有一个 exact key，没有 compatible/bucket 二级复用。未来只有单一固定实现、无需持久化选择的 operator 可以不创建 recipe；一旦持久化 winner，就必须遵守下面的 exact-key 状态机。

```text
lookup exact key
  ├── miss / parse failure ─────────────┐
  └── hit → find implementation         │
              ├── identity/support/config invalid ─┤
              └── prepare               │
                    ├── success → execute（不测速）
                    └── failure ─────────┤
                                        ↓
                              online_tune ?
                                ├── yes → 完整 autotune → prepare winner
                                │                         ├── success → persist recipe
                                │                         └── failure ─────┐
                                └── no ────────────────────────────────────┤
                                                                          ↓
                                                              allow_fallback ?
                                                                ├── yes → adapter fallback
                                                                └── no → error
```

exact key 必须覆盖 recipe schema、operator build ID、CUDA/GPU 兼容身份、从 normalized Spec 中抽取的调优等价类字段，以及会改变候选合法性的 workspace/alignment/graph-safe/deterministic policy。并非 Spec 的每个字段都必须机械进入 key；取舍由 operator adapter 定义并通过 hit/miss 测试固定。

`build_support/<operator>_fingerprint.rs` 根据该 operator 的 candidate/kernel/ABI/framework 构建输入、crate/target 和 CUDA architectures 生成 build ID。当前实例是 Attention 和 GEMM。build ID 是同一个 exact key namespace 的字段，不是第二个 key；按 operator 分开计算可避免无关 recipe 失效。

runtime 内存 map 和持久化文件保存的是同一种 exact recipe。持久化文件名是 key 的哈希，但文件内仍保存并比较完整 key。

fallback 由 adapter 负责，因为“哪个 candidate 是该 semantic 的正确性基线”是 operator contract。`fallback` 标记不阻止 candidate 参加 autotune，只定义 miss 且需要退路时的选择资格。
fallback 选择本身不作为完整调优结果持久化。

## Session / CUDA Graph 状态机

| 阶段 | 入口 | 允许 | 禁止/失败条件 | 产物 |
| --- | --- | --- | --- | --- |
| Prepare traversal | `prepare_with_session(&session, forward)` | recipe lookup、autotune、native prepare、provider resource 创建、执行并记录顺序 | nested session | session typed execution cache + prepared sequence |
| Capture traversal | `capture(&ctx, || with_session(&session, forward))` | 按相同顺序命中 execution 并 enqueue | cache miss、顺序变化、device/stream 变化、native prepare | `CapturedGraph` |
| Eager reuse | `with_session(&session, forward)` | capture 外复用同一批 execution | cache miss或顺序变化 | 异步 enqueue |
| Replay | `CapturedGraph::replay()` | 启动已实例化 graph | 改变已捕获结构 | GPU work |

graph 保留捕获时使用的 execution 和 storage。recipe key 不包含原始指针；固定地址和 stream 属于 Rust execution cache key。

## 目录职责

| 路径 | 唯一职责 |
| --- | --- |
| `cuda-operator.md` | 公开 L3 semantic catalog |
| `src/ops/` | Rust contract、normalize 和 execution wrapper |
| `src/ffi/abi/` | C ABI 的 Rust 声明 |
| `src/workspace.rs`、`src/graph.rs` | session execution cache 与 CUDA Graph 生命周期 |
| `native/include/apxinf_cuda/` | stable C ABI headers |
| `native/adapters/<operator>/` | semantic registry、selection、exact key、fallback、prepared execution |
| `native/framework/` | operator-independent registry/autotune/recipe/error primitives |
| `native/adapters/<operator>/providers/` | provider state 与 callback 实现 |
| `native/kernels/` | kernel 源码、实例和 vendor tree |
| `build_support/`、`build.rs` | 构建目标、build fingerprint 和 native 编译 |

## 变更矩阵

| 变更类型 | 必改 | 可能需要改 | 不应改 |
| --- | --- | --- | --- |
| 复用已有 semantic | 模型调用代码 | Args/policy | registry、framework、ABI |
| 扩展 candidate shape/dtype | candidate `supports`；provider/kernel；all-candidate 测试 | alignment/resource/config；build 输入 | 新增 semantic、framework |
| 新增 candidate | identity/version；全部 callbacks；正确 semantic registry；测试 | provider 文件、kernel、build.rs/fingerprint 输入 | Rust L3 API、通用 autotune |
| 新增 provider | provider callbacks/state；依赖和编译接入；candidate registration | vendor provenance/patch | framework provider 分支、模型 provider 分支 |
| 新增 semantic | Rust Args/normalize/export；ABI enum/Spec；独立 registry；fallback；catalog/测试 | 新字段、provider/kernel、ABI version；需要调优时增加 exact key/autotune | 复用其他 semantic 的 selection domain |
| 修改 candidate 行为/性能 | implementation version 或能覆盖变更的 build fingerprint；回归测试 | recipe schema/key version | 继续复用不再等价的旧 recipe |

任何 candidate 都必须测试其声明支持的输入，而不能只测试最终 autotune winner。完整测试与验收步骤见 [Adding New Kernels](../../doc/adding-new-kernels.md)。
