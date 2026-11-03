# S3 — Serving 模板/tokenization cache 与 SSE 合并

天气 case：`明天上海的天气怎么样`。已完成精确版本 CUDA 构建及真实天气测试。

在服务端以消息与模板 options 为 key 缓存渲染后的 token IDs；system/tool prefix boundary 使用独立有界 LRU，避免热请求重复渲染/分词 8K+ Prompt。

SSE 首 delta 立即发送，随后每 8 个 delta 合并；worker/API 队列从 8 增至 128，降低推理线程被 HTTP 背压打断的概率。合并按 delta 数而非时间，低速生成时仍可能增加后续 chunk 等待，需流式延迟回归。

主要目标是 host 模板、tokenization、跨线程唤醒和发送开销。模型内部 TTFT 与 HTTP wall time 的边界不同，不能把差值直接写成 GPU 性能提升。

没有本 commit 与 parent 的严格消融数据；历史 AIOS 余项 1.411→0.181 s 不能作为本次优化收益。


## 精确版本重建实测（2026-11-03 起）

Source: `679322b04571a4b3553acb132cd44bfac67a8854`；tracked 工作区干净。

Binary SHA256: `29a3dcf892b5c9ea66ef431435381f3a5e9981d095ce291d571577e9dbf147dc`。

配置与逐轮记录：[JSON](../evidence/679322b0-clean-weather.json)。

正式 1 轮，回答校验通过 1 轮，完整有效阶段记录 1 轮。

端到端单轮实测（不是统计中位数）：**1.985230 s**。

以下为同一真实请求 index=2，回答校验=True；不拼接不同请求的阶段分位数。

| 阶段 | 时间 | 占比 |
|---|---:|---:|
| 模型调用1 Prefill/TTFT：新增 1 tokens | 0.205802 s | 10.37% |
| 模型调用1 Decode：37 个间隔 | 0.487777 s | 24.57% |
| AMap天气查询（AIOS工具计时） | 0.475100 s | 23.93% |
| 模型调用2 Prefill/TTFT：新增 64 tokens | 0.154333 s | 7.77% |
| 模型调用2 Decode：23 个间隔 | 0.480914 s | 24.22% |
| AIOS、模型服务、HTTP及其他余项 | 0.181304 s | 9.13% |
| **端到端总计** | **1.985230 s** | **100%** |

| 模型调用 | full tokens | reused tokens | computed tokens | output tokens | TPOT |
|---|---:|---:|---:|---:|---:|
| 1 | 8853 | 8852 | 1 | 38 | 13.183 ms |
| 2 | 9322 | 9258 | 64 | 24 | 20.909 ms |

TTFT 包括缓存恢复与首 token；余项含未归属的服务/调度/HTTP时间。工具时长为AIOS执行计时，不能声称等于纯网络时间。当前没有仅凭该单轮成立的单项优化提升百分比。

正式测试前后 recipe 数：1236 → 1236；清单内容未变。旧版默认在线调优可能进入正式请求时间，不能直接当作纯GPU kernel性能。

独立 Nsight capture：[kernel/API/memcpy 汇总](../evidence/679322b0-nsys-stats.csv)。采集时延不进入未插桩正式成绩。
