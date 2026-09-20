# 在 ApxInf 中新增或扩展 CUDA Kernel

本文是 `crates/apxinf-cuda-new` 的 kernel 开发操作规程。旧
`crates/apxinf-cuda` 不再作为新增实现的推荐路径。

GEMM 和 Attention 是当前参考实现，不是允许的全部 operator family。未来算子可以定义自己的
Spec、Bindings 和 candidate descriptor，但必须遵守本文规定的层间边界和执行生命周期。

开始前必须阅读 [`apxinf-cuda-new` 架构说明](../crates/apxinf-cuda-new/README.md)
和 [CUDA L3 operator catalog](../crates/apxinf-cuda-new/cuda-operator.md)。本文不重复架构，
只规定必须修改的文件、接口、禁止事项和验收项。

临时 probe、日志和 benchmark 结果必须按 [`AGENTS.md`](../AGENTS.md) 放入
`devlocal/<feat-name>/`，不得混入正式源码目录。

## 1. 唯一决策表

先记录数学语义、shape、layout、dtype、量化、mask、scale、目标 GPU、workspace、
CUDA Graph、确定性、数值容差和性能目标，然后只按下表分类。

| 判断 | 改动类型 | 章节 |
| --- | --- | --- |
| catalog 已有相同 semantic，现有 candidate 满足正确性和性能 | 直接复用 | 2 |
| semantic 已有，同一 candidate 能安全覆盖新 shape/dtype/device | 扩展 candidate | 3 |
| semantic 已有，但需要不同实现或技术栈 | 新增 candidate/provider | 4 |
| 公共数学语义或调用方必须表达的契约不同 | 新增 L3 semantic | 5 |
| 新 semantic 无法合理复用现有 family 的 Spec、Bindings 或执行生命周期 | 在第 5 节中新建 operator family | 5 |

以下情况不得新增 semantic：仅更换 provider；仅新增 configuration、shape、dtype、alignment
或 GPU；仅改变 packing、workspace 或 epilogue；仅模型函数名不同；fallback 正确但性能不足。
无法确定时，不得新增 semantic。先完成公共数学契约对比；仍无法判断时必须提交接口评审，不能由实现者自行选择层次。

## 2. 直接复用现有 semantic

| 类别 | 要求 |
| --- | --- |
| Required files | 只修改模型调用方和相关模型测试；不得修改 native registry/provider |
| Required interface | 只调用安全 Rust L3 API；不得调用 raw FFI、provider 或 CUDA kernel |
| Required tests | 模型真实 shape 正确性、目标 GPU provider summary、端到端 profiling |
| MUST NOT | 不复制 semantic；不在模型层按 GPU/shape/provider 分支；不绕过 registry |

provider selection 测试不能替代性能测试；一种 GPU 架构上的结果不能替代目标架构验收。

## 3. 扩展已有 candidate

只有底层实现已经具备该能力时才能走此路径。`supports()` 返回 `true` 是正确性承诺，
不是性能提示。

### Required files

| 文件 | 何时修改 |
| --- | --- |
| `native/adapters/<operator>/candidates.cpp` | 扩展 supports/alignment/resource/configurations |
| `native/adapters/<operator>/providers/*` | provider adapter 需处理新契约 |
| `native/kernels/*` | 需新增 kernel 分支或模板实例 |
| `build.rs` | 新源码、include、宏或目标架构进入构建 |
| `build_support/<operator>_fingerprint.rs` | 新输入不在 fingerprint 范围 |
| `src/ops/tests/precision/*` | 新覆盖范围的 all-candidate 独立参考测试 |
| `src/ops/tests/l3_behavior.rs` | 公共契约边界受影响 |

### Required interfaces

| 接口 | MUST |
| --- | --- |
| `supports(spec)` | 精确覆盖 semantic、shape、dtype、layout、mask、scale predicate、quantization |
| `alignment_requirements(spec)` | 声明每个 binding 的最低对齐 |
| `resource_requirements(spec)` | 在 provider 构造前报告可确定的资源需求，供 workspace budget 预筛选 |
| `enumerate_configs(spec, out)` | 返回全部需独立测速的稳定 configuration |
| `prepare(execution)` | 能构造每个已枚举配置的完整 provider state |
| `enqueue(execution)` | 对新范围正确且异步 |
| `destroy(execution)` | 覆盖所有 prepare 成功路径 |

### 禁止事项与测试

MUST NOT：先放宽 `supports()` 再依赖运行失败；读取输入值或指针内容选核；只枚举预估最快配置；
改变旧 configuration 编号含义；为 shape 特化新增 API；漏掉目标 SM 或 fingerprint 输入。

Required tests：新 shape/dtype 的 all-candidate reference；alignment/workspace 边界；所有新配置的
prepare/enqueue/destroy；capture/replay；目标 GPU 编译链接；冷 cache tune、热 cache hit；真实
shape benchmark。

## 4. 新增 candidate 或 provider

semantic 必须已经存在，provider 名称不得进入公共 API。

### Required files

| 文件 | MUST |
| --- | --- |
| `native/adapters/<operator>/internal.h` | 声明 provider callbacks 和 Execution state |
| `native/adapters/<operator>/candidates.cpp` | 注册 identity、属性和全部 callbacks |
| `native/adapters/<operator>/providers/<provider>.*` | provider 薄适配 |
| `native/kernels/<operator-or-provider>/` | kernel、实例或 vendor 源码 |
| `build.rs` | 按目标架构编译和链接 |
| `build_support/<operator>_fingerprint.rs` | 覆盖 candidate/ABI/kernel 输入 |
| `src/ops/tests/precision/*` | all-candidate 独立参考测试 |
| `src/ops/tests/backend_framework.rs` | selection、resource 和生命周期回归；使用 recipe/fallback 时增加对应回归 |

第三方 provider 还必须提供 `README.md` 或 `VENDOR.md`，记录上游 revision、许可证和本地改动；
patch 放入 `native/patches/`。优先使用 compat layer 或构建时 patch，不直接污染 vendor 快照。

### Stable identity 与 callbacks

每个 candidate 必须有唯一且稳定的
`(provider_id, implementation_id, implementation_version)`。不得复用 identity 表示另一实现；
configuration 编号含义必须稳定；旧配置无法按原义恢复时必须更新 implementation version；
所有实现输入必须进入 operator build fingerprint。

当前 GEMM/Attention candidate descriptor 使用以下接口；扩展这两个 family 时必须完整实现并注册：

1. `supports(spec)`；
2. `alignment_requirements(spec)`；
3. `resource_requirements(spec)`；
4. `enumerate_configs(spec, out)`；
5. `prepare(execution)`；
6. `enqueue(execution)`；
7. `destroy(execution)`。

同时必须准确填写 `required_device_features`、`graph_safe`、`deterministic`、`fallback` 和
可诊断的 `name`。

新的 operator family 可以定义不同的 descriptor，但必须由 adapter 无损映射为 framework 要求的
`Problem::registry/supports/configurations/prepare/enqueue/stream/graph_safe` 协议，并明确表达能力、
资源、configuration 和 destroy 生命周期。不得仅因命名偏好改造通用 framework；偏离当前 descriptor
模板必须在该 operator 的架构评审中说明原因。

`prepare` 可创建 handle、descriptor、算法、prepack 和 workspace，但必须遵守
`resource_limit`。`enqueue` 只能使用 prepared state 向绑定 stream 提交工作，不得 tune、
分配长期资源、lazy initialize 或 synchronize。

### Fallback、禁止事项与测试

- 每个 semantic 必须有一个明确的 baseline fallback；高性能特化默认不是 fallback。
- fallback 必须通过完整 semantic 测试；fallback 选择只属于 operator adapter。
- registry 不得保存 per-call state；模型层不得选择 provider；framework 不得出现 operator 特例。
- 不得把“能 launch”当作数值正确，不得无记录地修改 vendor 源码。

Required tests：全部 candidate/configuration 对独立 reference；unsupported 过滤；identity/version/
configuration 恢复；resource limit；fallback allow/deny；eager 与 graph 一致；目标 GPU 冷/热
cache 和性能。

## 5. 新增 L3 semantic

必须按以下顺序实现。公共 contract 未确定前，不得先注册 provider 特化。

### Step 1：Rust Args、normalize、execute

Required files：

```text
src/ops/<operator>/contracts.rs（或现有 family contract）
src/ops/<operator>/<semantic>.rs
src/ops/<operator>/execution.rs（或现有 family execution 文件）
src/ops/<operator>/mod.rs
src/ops/mod.rs
src/lib.rs
```

| Required interface | MUST |
| --- | --- |
| `<Semantic>Args` | 只表达公共数学语义；非法组合尽量不可表达 |
| `normalize(ctx, args)` | 校验 device/dtype/shape/layout/alias/overflow，生成 Spec/Policy/Bindings |
| `<semantic>(ctx, args)` | 只执行 `normalize → execute` |
| public exports | 从 operator module 和 crate 根正确导出 |

Normalize MUST：Spec 只保存 semantic、合法性和选择等价类所需结构信息；Bindings 保存地址、
stream 和不影响选核的动态数值；Policy 保存该 operator 支持的资源与选择约束；需要根据地址选择
candidate 时，将地址归一化为 alignment class；storage 保活；等价调用产生相同 Spec。新的 operator
不必照抄 GEMM/Attention 的全部字段。

### Step 2：稳定 C ABI

Required files：

```text
native/include/apxinf_cuda/<operator>_types.h
native/include/apxinf_cuda/<operator>.h
src/ffi/abi/<operator>.rs
src/ffi/abi/mod.rs
```

Required ABI：

```c
*_prepare(runtime, spec, policy, bindings, &execution);
*_enqueue(execution);
*_destroy(execution);
```

MUST：Spec 有明确 version，布局或语义变化时更新；只用 C-compatible 固定宽度字段和 opaque
pointer；Rust 与 C 声明完全一致；异常经统一 ABI boundary 转成 status/last-error；任何失败路径
不得泄漏 execution 或 provider resource。

### Step 3：operator adapter

Required files：

```text
native/adapters/<operator>/internal.h
native/adapters/<operator>/candidates.cpp
native/adapters/<operator>/execution.cpp
```

使用持久化调优时还必须增加：

```text
native/adapters/<operator>/tuning_key.cpp
native/adapters/<operator>/autotune.cpp
```

Required types：`Spec`、`Implementation`、`Execution`、`ImplementationRegistry` 和每个 semantic 的
`SelectionKind`。使用持久化调优时，
再增加只含一个 exact key 的 `TuningKeys`。

| Required function | MUST |
| --- | --- |
| `semantic_registry(spec.semantic)` | 返回该 semantic 的不可变 candidate 集合和 `SelectionKind` |
| `tuning_keys(spec, policy, device)` | tunable operator 构造唯一 exact key |
| `tune(spec, policy, bindings, device, report)` | tunable operator 适配 `framework::autotune` |
| operator `prepare(...)` | 校验 candidate/policy/resource 并创建 Execution |
| C ABI `*_prepare` | 校验 ABI；tunable operator 查 recipe/tune/fallback，非调优 operator 直接选择合法 baseline；返回 execution |
| C ABI `*_enqueue` | 只 enqueue prepared execution |
| C ABI `*_destroy` | 安全释放全部状态 |

MUST NOT：在 `native/framework` 添加本 semantic 的 shape/dtype/registry/fallback；新增 compatible
或 hint key；把 pointer/stream 放入持久 key；在 capture 中首次 prepare。

`SelectionKind` 只能取三种值：`Fixed` 直接使用唯一合法实现；`Heuristic` 由 adapter 的稳定规则选择且
不测速；`Autotune` 先查 exact recipe，miss 时完整测速。该值在 `candidates.cpp` 的 semantic registry
中声明。只有复杂启发式才需要单独的 `selection.cpp`；`Fixed` 和 `Autotune` 不要求该文件。当前
GEMM/Attention semantic 均使用 `Autotune`。

### Step 4：provider callbacks

Required files：

```text
native/adapters/<operator>/providers/
native/kernels/<operator-or-provider>/
build.rs
build_support/<operator>_fingerprint.rs
```

至少注册一个完整实现 semantic 的 fallback。现有 GEMM/Attention candidate 必须实现第 4 节列出的
七个 callbacks；新 operator family 使用经过评审的等价 descriptor。所有 candidate 都必须具有稳定
identity、完整的 device/policy 约束和 build fingerprint 覆盖。

### Step 5：catalog 和 tests

Required files：

```text
cuda-operator.md
src/ops/tests/operator_doc.rs
src/ops/tests/l3_behavior.rs
src/ops/tests/precision/precision.rs
src/ops/tests/precision/generate_torch_l3_fixtures.py（需要 Torch golden 时）
src/ops/tests/backend_framework.rs（有新执行行为时）
tests/public_ops.rs（公共导出/session 行为变化时）
```

MUST：增加唯一 catalog marker 和 Rust semantic metadata；独立 semantic test；覆盖所有 candidate/
configuration 的独立 reference；新 binding/resource/capture 行为的 prepare/capture/replay 测试；
catalog test 同时拒绝缺失、未知和重复 semantic。

## 6. Tunable operator 的 Recipe 不变量

只有一个固定实现且不持久化选择的 operator 可以不使用 recipe。任何需要在多个
candidate/configuration 间调优并持久化 winner 的 operator，都必须遵守本节。

| 场景 | MUST |
| --- | --- |
| exact hit | implementation/version/configuration 仍存在，supports/device/alignment/policy/prepare 均成功；直接使用，不测速 |
| miss | `online_tune=true` 时完整测速全部合法 candidate/configuration |
| invalid/corrupt | 按 miss 处理，不得近似复用 |
| winner | 只保存 provider、implementation、version、configuration |
| tune 后真实 prepare 失败 | 仅 `allow_fallback=true` 时 fallback |
| tune 关闭、fallback 开启 | 只使用明确标记的 fallback |
| tune 和 fallback 关闭 | 返回明确 cache miss/unsupported |
| fallback execution | 不得持久化为 tuned winner |

Exact key 必须覆盖 recipe schema、operator build fingerprint、目标 GPU、定义的 CUDA/关键库兼容
版本、调优等价类所需 Spec 字段、workspace、graph-safe、deterministic 和 alignment class。任一关键项
变化必须 miss，不得增加第二级 compatible key。

autotuner 不验证数值；数值正确性必须由 all-candidate 测试保证。

## 7. ExecutionSession 与 CUDA Graph 不变量

```rust
let session = ExecutionSession::with_capacity(workspace_bytes, device)?;
prepare_with_session(&session, || forward())?;
let graph = capture(&ctx, || with_session(&session, || forward()))?;
graph.replay()?;
```

| 阶段 | MUST | MUST NOT |
| --- | --- | --- |
| prepare | lookup/tune、建 execution、分配资源、记录顺序 | 在 capture 中运行 |
| with_session | 命中相同 Spec/bindings/device/stream/执行约束/顺序 | miss 时临时 prepare |
| capture | 只捕获异步 enqueue，并保活资源 | 多 stream、同步、lazy init |
| replay | 直接启动 graph | 重新选核或修改 recipe |

`graph_safe=true` 只表示 prepared enqueue 可捕获。allocation、算法选择和长期状态初始化必须在
`prepare()` 完成。

## 8. 统一验收

| 验收项 | 复用 | 扩展 | 新 candidate/provider | 新 semantic |
| --- | --- | --- | --- | --- |
| 模型真实 shape 正确性 | MUST | MUST | MUST | MUST |
| all-candidate 独立 reference | 已有 | MUST | MUST | MUST |
| catalog/semantic test | 已有 | 受影响时 | 受影响时 | MUST |
| alignment/workspace | 已有 | MUST | MUST | MUST |
| hit/miss/invalid recipe | 已有 | key 变化时 | 使用 recipe 时 MUST | 使用 recipe 时 MUST |
| fallback allow/deny | 已有 | fallback 变化时 | MUST | MUST |
| prepare/capture/replay | MUST | MUST | MUST | MUST |
| 目标 SM 编译链接 | MUST | MUST | MUST | MUST |
| 冷 tune、热 hit、profiling | 有调优时 MUST | 有调优时 MUST | 有调优时 MUST | 有调优时 MUST |

完整测试从仓库根目录运行：

```bash
bash crates/apxinf-cuda-new/test-new.sh \
  test -p apxinf-cuda -- --nocapture --test-threads=1
```

最终必须在实际目标 GPU 上确认：正确 `APXINF_CUDA_ARCH`；新源码进入链接；semantic、
all-candidate、graph 以及适用的 recipe 测试通过；实际 provider/configuration 可解释；使用调优时
冷/热 cache 行为正确；
真实 shape benchmark 和模型端到端数值/性能达标。

任一 required interface、required test、目标 GPU 结果或文档更新缺失，均不得标记完成。
