# 7e3bccbf — fix(bench): qualify historical weather evidence and target dates

天气 case：`明天上海的天气怎么样`。历史提交不改 hash。

## 精确版本重建实测（2026-11-03 起）

Source: `7e3bccbff669caa6df9efacc92e2debd259c9723`；tracked 工作区干净。

Binary SHA256: `29a3dcf892b5c9ea66ef431435381f3a5e9981d095ce291d571577e9dbf147dc`。

配置与逐轮记录：[JSON](../evidence/7e3bccbf-clean-weather.json)。

正式 1 轮，回答校验通过 1 轮，完整有效阶段记录 1 轮。

端到端单轮实测（不是统计中位数）：**2.441770 s**。

以下为同一真实请求 index=2，回答校验=True；不拼接不同请求的阶段分位数。

| 阶段 | 时间 | 占比 |
|---|---:|---:|
| 模型调用1 Prefill/TTFT：新增 1 tokens | 0.150068 s | 6.15% |
| 模型调用1 Decode：38 个间隔 | 0.614596 s | 25.17% |
| AMap天气查询（AIOS工具计时） | 0.436200 s | 17.86% |
| 模型调用2 Prefill/TTFT：新增 168 tokens | 0.601575 s | 24.64% |
| 模型调用2 Decode：19 个间隔 | 0.283159 s | 11.60% |
| AIOS、模型服务、HTTP及其他余项 | 0.356172 s | 14.59% |
| **端到端总计** | **2.441770 s** | **100%** |

| 模型调用 | full tokens | reused tokens | computed tokens | output tokens | TPOT |
|---|---:|---:|---:|---:|---:|
| 1 | 8853 | 8852 | 1 | 39 | 16.174 ms |
| 2 | 9059 | 8891 | 168 | 20 | 14.903 ms |

TTFT 包括缓存恢复与首 token；余项含未归属的服务/调度/HTTP时间。工具时长为AIOS执行计时，不能声称等于纯网络时间。当前没有仅凭该单轮成立的单项优化提升百分比。

正式测试前后 recipe 数：1224 → 1224；清单内容未变。旧版默认在线调优可能进入正式请求时间，不能直接当作纯GPU kernel性能。
