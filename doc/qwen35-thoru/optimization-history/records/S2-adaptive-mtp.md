# S2 — Adaptive MTP、GPU acceptance 和 GDN verification

天气 case：`明天上海的天气怎么样`。已完成精确版本 CUDA 构建及真实天气测试。

把 MTP 最大 draft depth 从 2 改为 3；根据近期接受率和 round wall time 在 depth 1/2/3 间选择，预计慢于配置的 scalar TPOT 时回退。

各 depth 保留独立 verifier 资源；draft token 链留在 GPU，接受判断与结果打包在 GPU 完成，host 读取 8 个 I32（32 字节）。

GDN 卷积、head normalization 与短序列 recurrent verification 合并并保留逐 row rollback checkpoint；target/draft commit 仍有 host 控制，尚非完整 round CUDA Graph。

prompt snapshot LRU 保留 4 项，tool follow-up 的 volatile tail 保留 64 token；静态 system、exact prompt 和 live conversation cache 分别命中。

增加同轮 MTP profile/诊断路径，修正 serving 的 3-token lookahead 容量。



预期收益：Decode 减少同步和 launch；Prefix Cache 减少重复 Prefill。默认 19.42 ms scalar 阈值只是旧测量参考，需同场重新校准，不能当作当前硬件常量。

精确 shape 会改变 recipe，补测时必须先 autotune，再无 autotune 正式运行。固定输入Prefill对照与Nsight分析见 [profiler报告](../PROFILER.md)。未完成长上下文数值验证；历史局部7.1%不归因于此commit。

CPU 验证：独立使用 `rustc --edition=2021 --test mtp_schedule.rs`，6/6 通过；
包括 scalar 等价、zero/partial/full acceptance、EOS、输出预算和失败事务。模型边界检查通过。
这些 CPU 检查不建立 CUDA 数值正确性或端到端性能结论。


GDN Prefill还将非64对齐输入拆为完整64-token块及精确尾段，避免291/758等Agent输入整段退回逐token recurrence；没有补假token。

## 精确版本重建实测（2026-11-03 起）

Source: `118337ceb5ef9cdbd440f28da728314bd7cfbb13`；tracked 工作区干净。

Binary SHA256: `bcf0702de3388e751d890e87b8aa1119666fc1791a60433bdd3795777e8105bd`。

配置与逐轮记录：[JSON](../evidence/118337ce-clean-weather.json)。

正式 1 轮，回答校验通过 1 轮，完整有效阶段记录 1 轮。

端到端单轮实测（不是统计中位数）：**2.042260 s**。

以下为同一真实请求 index=2，回答校验=True；不拼接不同请求的阶段分位数。

| 阶段 | 时间 | 占比 |
|---|---:|---:|
| 模型调用1 Prefill/TTFT：新增 1 tokens | 0.127057 s | 6.22% |
| 模型调用1 Decode：37 个间隔 | 0.452682 s | 22.17% |
| AMap天气查询（AIOS工具计时） | 0.358800 s | 17.57% |
| 模型调用2 Prefill/TTFT：新增 64 tokens | 0.174526 s | 8.55% |
| 模型调用2 Decode：27 个间隔 | 0.433871 s | 21.24% |
| AIOS、模型服务、HTTP及其他余项 | 0.495325 s | 24.25% |
| **端到端总计** | **2.042260 s** | **100%** |

| 模型调用 | full tokens | reused tokens | computed tokens | output tokens | TPOT |
|---|---:|---:|---:|---:|---:|
| 1 | 8853 | 8852 | 1 | 38 | 12.235 ms |
| 2 | 9322 | 9258 | 64 | 28 | 16.069 ms |

TTFT 包括缓存恢复与首 token；余项含未归属的服务/调度/HTTP时间。工具时长为AIOS执行计时，不能声称等于纯网络时间。当前没有仅凭该单轮成立的单项优化提升百分比。

正式测试前后 recipe 数：1237 → 1237；清单内容未变。旧版默认在线调优可能进入正式请求时间，不能直接当作纯GPU kernel性能。

独立 Nsight capture：[kernel/API/memcpy 汇总](../evidence/118337ce-nsys-stats.csv)。采集时延不进入未插桩正式成绩。
