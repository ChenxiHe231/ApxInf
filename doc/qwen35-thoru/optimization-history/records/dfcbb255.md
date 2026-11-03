# dfcbb255 — perf(attention): bucket live decode recipes

天气 case：`明天上海的天气怎么样`。历史提交不改 hash。

## 精确版本重建实测（2026-11-03 起）

Source: `dfcbb25547d4cc1a2b376daa8a8bdd5ff05560bb`；tracked 工作区干净。

Binary SHA256: `c27b8829bd3a58aaf34e7beefd082a0fbfba7f2cc6c24edc4b97da26a2e5e87a`。

配置与逐轮记录：[JSON](../evidence/dfcbb255-clean-weather.json)。

正式 1 轮，回答校验通过 1 轮，完整有效阶段记录 1 轮。

端到端单轮实测（不是统计中位数）：**16.810070 s**。

以下为同一真实请求 index=0，回答校验=True；不拼接不同请求的阶段分位数。

| 阶段 | 时间 | 占比 |
|---|---:|---:|
| 模型调用1 Prefill/TTFT：新增 291 tokens | 1.049336 s | 6.24% |
| 模型调用1 Decode：38 个间隔 | 8.219985 s | 48.90% |
| AMap天气查询（AIOS工具计时） | 0.396000 s | 2.36% |
| 模型调用2 Prefill/TTFT：新增 757 tokens | 1.313967 s | 7.82% |
| 模型调用2 Decode：23 个间隔 | 5.299989 s | 31.53% |
| AIOS、模型服务、HTTP及其他余项 | 0.530793 s | 3.16% |
| **端到端总计** | **16.810070 s** | **100%** |

| 模型调用 | full tokens | reused tokens | computed tokens | output tokens | TPOT |
|---|---:|---:|---:|---:|---:|
| 1 | 8853 | 8562 | 291 | 39 | 216.315 ms |
| 2 | 9319 | 8562 | 757 | 24 | 230.434 ms |

TTFT 包括缓存恢复与首 token；余项含未归属的服务/调度/HTTP时间。工具时长为AIOS执行计时，不能声称等于纯网络时间。当前没有仅凭该单轮成立的单项优化提升百分比。

正式测试前后 recipe 数：1223 → 1223；清单内容未变。旧版默认在线调优可能进入正式请求时间，不能直接当作纯GPU kernel性能。
