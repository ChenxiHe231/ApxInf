# `apxinf-cuda-new` 架构契约

`apxinf-cuda-new` 向模型层提供稳定的 CUDA L3 语义接口，在 native 层为同一语义选择并准备 provider kernel。GEMM 和 Attention 是当前已接入的两个 operator family，不是框架允许的全部类型。

- 当前公开接口及数学语义：[L3 operator catalog](cuda-operator.md)
- 新增或扩展 kernel 的工作流：[Adding New Kernels](../../doc/adding-new-kernels.md)

## 算子分层：L3 到 L0

这里的 L0–L3 只描述 `apxinf-cuda-new` 内部的 CUDA 算子，不是仓库根文档中的模型、policy、serving 分层。

| 层级 | 职责 | 输入 → 输出 | 主要接口和目录 |
| --- | --- | --- | --- |
| L3 Semantic（Rust） | 定义模型可见的完整数学语义和 tensor contract | `CudaContext + Args` → `Result<()>` | `ops::<semantic>`；`src/ops/<operator>/` |
| L2 Execution（Rust） | 校验并归一化 L3 调用；管理 execution cache、session、storage 和 graph 生命周期 | L3 Args → `Spec + Policy + Bindings` → opaque native handle | `normalize`、`prepare/execute`、Rust FFI declaration；`src/ops/`、`src/workspace.rs`、`src/graph.rs` |
| L1 Native operator（C++） | 实现 C ABI；完成 recipe lookup、选核、autotune、fallback、candidate/provider prepare 和 enqueue | `Spec + Policy + Bindings` → native `Execution` | `*_prepare/*_enqueue/*_destroy`、registry 和 candidate callbacks；`native/adapters/`、`native/framework/` |
| L0 Kernel | 执行实际 GPU 计算 | provider launch 参数 → GPU work | custom CUDA、FA2、CUTLASS、cuBLAS/cuBLASLt；`native/kernels/` 或 vendor API |

```text
L3 Rust → L2 Rust │ C ABI │ L1 C++ → L0 CUDA

L3: no provider       L2: no candidate selection
L1: owns selection    L0: no recipe/fallback/model logic
```

## 调用链

```text
L3  Rust semantic API
L2  normalize → ExecutionKey/session cache → Rust FFI call
                     ───── C ABI boundary ─────
L1  recipe/selection → candidate/provider prepare → native Execution
L0  CUDA kernel or vendor API
```

```text
prepare: L2 → L1 *_prepare → recipe/selection → native Execution
run:     L2 → L1 *_enqueue ───────────────────→ L0 launch
capture: prepare_with_session once → capture/with_session reuses Execution
```

## 六个核心对象

| 对象 | 所属层 | 由什么组成 | 与其他对象的关系 |
| --- | --- | --- | --- |
| Semantic | L3 | 公开数学语义、tensor contract 和 semantic ID | 一个 Semantic 对应一个独立 candidate registry；L3 对外暴露它 |
| Provider | L1（适配 L0） | 实现技术栈身份，例如 FA2、CUTLASS、cuBLAS 或 custom CUDA | 一个 Provider 可以提供多个 Candidates；Provider 名称不进入 L3 API |
| Candidate / Implementation | L1 | Provider ID + implementation ID/version + 能力属性 + callbacks | 注册到某个 Semantic；可以枚举多个 Configurations |
| Configuration | L1 | Candidate 自己定义的稳定配置编号 | 只有放在所属 Candidate 下才有含义 |
| Recipe | L1 | Candidate 的稳定 identity + 一个 Configuration | recipe cache 的 value；记录 winner，但不保存指针、handle 或 provider state |
| Prepared Execution | L1；L2 持有 opaque handle | 当前 `Spec + Bindings + device + Candidate + Configuration + policy-derived limits + provider state/resources` | 由 Recipe 或本次选核结果重新 prepare；可 enqueue 多次，只在当前进程有效 |

组合关系：

```text
Semantic → Registry<Candidate>
Provider + implementation identity + callbacks → Candidate
Candidate identity + Configuration → Recipe
Recipe key → Recipe
当前 Spec + Policy + Bindings + 已解析 Recipe → Prepared Execution
Prepared Execution → Candidate enqueue → Provider → L0 kernel
```

```text
Registry = Candidate container          # not a seventh core object
Recipe   = winner identity              # not executable
Recipe hit → resolve Candidate → validate → prepare → Execution
```

## 三类跨层数据

| 数据 | 内容 | 是否绑定本次调用 |
| --- | --- | --- |
| `Spec` | semantic、shape、dtype、mask、layout、alignment class 等归一化问题描述 | 否；等价调用可相同 |
| `Policy` | workspace、graph-safe、deterministic、是否允许 tune/fallback 等约束 | 部分字段影响选核或 prepared state |
| `Bindings` | 本次地址、stream 和实际动态数值，例如 `alpha`、attention scale | 是 |

## 关键接口

| 接口 | 层级 | 责任 |
| --- | --- | --- |
| `ops::<semantic>(ctx, args)` | L3 Rust | 唯一模型入口；表达完整数学语义，不暴露 provider |
| `normalize(ctx, args)` | L2 Rust | 校验 L3 contract，生成 `Spec + Policy + Bindings` 并保活 storage |
| Rust `prepare/execute` | L2 Rust | 查 execution cache；通过 FFI 创建或 enqueue opaque native execution |
| `*_prepare(..., &execution)` | L1 C++ | 在 capture 外完成 recipe/选核和所有 provider state/resource 创建 |
| `*_enqueue(execution)` | L1 C++ | 只向绑定 stream 提交已准备的工作；不得选核、分配或同步 |
| `*_destroy(execution)` | L1 C++ | 释放 provider state 和资源 |
| `registry(spec.semantic)` | L1 C++ | 返回该 semantic 可参与选择的 candidates |
| `supports(spec)` | L1 C++ | 声明 candidate 的正确性适用范围，不是性能提示 |
| `alignment_requirements/resource_requirements` | L1 C++ | 在 prepare 前声明地址和资源约束 |
| `enumerate_configs` | L1 C++ | 返回需要独立选择或测速的所有 configuration |
| `framework::autotune(problem, report)` | L1 C++ | 只负责遍历、计时和返回 winner，不解释 semantic 或 fallback |

```text
*_prepare → *_enqueue [0..N] → *_destroy
```

## Key 与 cache

| Cache | 层级 | Key → Value | 生命周期 |
| --- | --- | --- | --- |
| Execution cache | L2 Rust | `ExecutionKey → Rc<Execution wrapper>` | 一个 `ExecutionSession` |
| Recipe cache | L1 C++ | `exact recipe key → Recipe` | native runtime 内存；可选磁盘持久化 |

### Execution key

| 组成 | Attention | GEMM |
| --- | --- | --- |
| 问题 | 完整 normalized `Spec` | 完整 normalized `Spec` |
| 设备与执行位置 | device、Q/K/V/offsets/output 地址、stream | device、A/B/bias/scales/output 地址、stream |
| 实际动态值 | `scale`、`output_scale` 的 bit pattern | `alpha`、`output_scale` 的 bit pattern；B version/immutable 标志 |
| 影响 prepared state 的 policy | workspace limit、graph-safe、deterministic | workspace limit、graph-safe、deterministic |
| 不进入 key | `online_tune`、`allow_fallback`、`cache_dir` | `online_tune`、`allow_fallback`、`cache_dir` |

### Exact recipe key

| 组成 | 内容 |
| --- | --- |
| Cache/build namespace | recipe schema version、operator build fingerprint、编译 CUDA toolkit |
| 运行环境 | GPU compute capability、SM 数；CUDA runtime/driver 兼容版本；GEMM 还包含 cuBLASLt 版本 |
| Attention 等价类 | semantic、input/output dtype、mask、batch、Q/K tokens、key capacity、Q/KV heads、head dim、query start、segments/max segment、default-scale predicate |
| GEMM 等价类 | semantic、M/N/K、A/B/accumulation/output dtype、quantization、B immutable、unit-alpha/unit-output-scale predicates |
| Candidate 合法性 | 各 binding 的 alignment class、workspace limit、graph-safe、deterministic |
| 不进入 key | 原始指针、stream、输入内容、实际 scale、B version、device UUID、cache dir、tune/fallback 开关、Attention offsets 内容 |

### 其他 identity 不是第二个 recipe key

| 名称 | 作用 |
| --- | --- |
| Build fingerprint | recipe key 的一个字段；candidate/kernel/ABI/framework 构建输入变化时使旧 recipe miss |
| Recipe value | `(provider_id, implementation_id, implementation_version, configuration)`，即 winner，不是 key |
| `.recipe` 文件名 | exact recipe key 的 FNV-1a 哈希，只用于定位文件；文件内仍保存并比较完整 key |
| Typed cache `TypeId` | 隔离不同 Rust `ExecutionKey/Execution` 类型，不表达算子等价类 |
| Prepared sequence identity | 校验 prepare 与 capture 的 operator 顺序一致，不参与 recipe lookup |

### Recipe 命中规则

| 情况 | 行为 |
| --- | --- |
| Key 模式 | 仅 exact key；没有 compatible/bucket key |
| hit 且 identity、supports、configuration、policy、prepare 均有效 | 直接创建 execution，不测速 |
| miss、损坏或 hit 后验证失败，且 `online_tune=true` | 完整测速全部合法 candidate/configuration |
| 无可用 recipe 且允许 fallback | 使用该 semantic 明确标记的 baseline；不持久化为 tuned winner |
| 无可用 recipe且 tune/fallback 均不可用 | 返回 cache miss/unsupported |
| 持久化条件 | tune winner 的真实 prepare 成功 |

## Session / CUDA Graph 状态机

| 阶段 | 入口 | 允许 | 禁止/失败条件 | 产物 |
| --- | --- | --- | --- | --- |
| Prepare traversal | `prepare_with_session(&session, forward)` | recipe lookup、autotune、native prepare、provider resource 创建、执行并记录顺序 | nested session | session typed execution cache + prepared sequence |
| Capture traversal | `capture(&ctx, || with_session(&session, forward))` | 按相同顺序命中 execution 并 enqueue | cache miss、顺序变化、device/stream 变化、native prepare | `CapturedGraph` |
| Eager reuse | `with_session(&session, forward)` | capture 外复用同一批 execution | cache miss或顺序变化 | 异步 enqueue |
| Replay | `CapturedGraph::replay()` | 启动已实例化 graph | 改变已捕获结构 | GPU work |

graph 保留捕获时使用的 execution 和 storage。recipe key 不包含原始指针；固定地址和 stream 属于 Rust execution cache key。

## 目录职责

| 层级 | 路径 | 唯一职责 |
| --- | --- | --- |
| L3 Rust | `cuda-operator.md`、`src/ops/<operator>/` 的公开 API | semantic、Args 和模型可见 contract |
| L2 Rust | `src/ops/<operator>/` 的 normalize/execution、`src/workspace.rs`、`src/graph.rs` | ABI lowering、execution cache、session 与 CUDA Graph 生命周期 |
| L2/L1 ABI | `src/ffi/abi/`、`native/include/apxinf_cuda/` | Rust/C 稳定边界和 opaque execution handle |
| L1 C++ | `native/adapters/<operator>/` | recipe key、registry、选核、fallback、candidate/provider state 和 prepared execution |
| L1 Shared | `native/framework/` | operator-independent registry、autotune、recipe I/O 和错误边界 |
| L0 | `native/kernels/` | kernel 源码、模板实例和 vendor tree |
| Build | `build_support/`、`build.rs` | 构建目标、build fingerprint 和 native 编译 |

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
