# S1 — FP8 attention tensor-core QK

天气 case：`明天上海的天气怎么样`。已完成精确版本 CUDA 构建及真实天气测试。

将 fused split attention 的 QK 从标量 FP32 FMA 改为 BF16 WMMA，保留 FP32 accumulator；增加 `APXINF_QWEN35_FP8_ATTENTION_MAX_SPLITS` 上限。

预期影响：Decode/MTP verification。数值归约顺序改变，必须做精度与工具选择回归；本提交已有天气与Nsight证据，详见下表及 [profiler分析](../PROFILER.md)；没有长上下文数值/带宽验收，不宣称全面收益。


## 精确版本重建实测（2026-11-03 起）

Source: `b452e715c716953261fcda73cc58c2764289b70a`；tracked 工作区干净。

Binary SHA256: `99d82484c4dfa6ce78a4c2f157f9db0d6405006406fa76358c2dcfab620aed4e`。

配置与逐轮记录：[JSON](../evidence/b452e715-clean-weather.json)。

正式 1 轮，回答校验通过 1 轮，完整有效阶段记录 1 轮。

端到端单轮实测（不是统计中位数）：**3.574020 s**。

以下为同一真实请求 index=2，回答校验=True；不拼接不同请求的阶段分位数。

| 阶段 | 时间 | 占比 |
|---|---:|---:|
| 模型调用1 Prefill/TTFT：新增 291 tokens | 0.632594 s | 17.70% |
| 模型调用1 Decode：38 个间隔 | 0.627347 s | 17.55% |
| AMap天气查询（AIOS工具计时） | 0.446100 s | 12.48% |
| 模型调用2 Prefill/TTFT：新增 432 tokens | 0.801486 s | 22.43% |
| 模型调用2 Decode：30 个间隔 | 0.558576 s | 15.63% |
| AIOS、模型服务、HTTP及其他余项 | 0.507917 s | 14.21% |
| **端到端总计** | **3.574020 s** | **100%** |

| 模型调用 | full tokens | reused tokens | computed tokens | output tokens | TPOT |
|---|---:|---:|---:|---:|---:|
| 1 | 8853 | 8562 | 291 | 39 | 16.509 ms |
| 2 | 9323 | 8891 | 432 | 31 | 18.619 ms |

TTFT 包括缓存恢复与首 token；余项含未归属的服务/调度/HTTP时间。工具时长为AIOS执行计时，不能声称等于纯网络时间。当前没有仅凭该单轮成立的单项优化提升百分比。

正式测试前后 recipe 数：1235 → 1235；清单内容未变。旧版默认在线调优可能进入正式请求时间，不能直接当作纯GPU kernel性能。

独立 Nsight capture：[kernel/API/memcpy 汇总](../evidence/b452e715-nsys-stats.csv)。采集时延不进入未插桩正式成绩。
