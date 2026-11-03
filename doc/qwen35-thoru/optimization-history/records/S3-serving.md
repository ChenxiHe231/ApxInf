# S3 — Serving 模板/tokenization cache 与 SSE 合并

天气 case：`明天上海的天气怎么样`。状态：**待 Thor-U 独占测试**。

在服务端以消息与模板 options 为 key 缓存渲染后的 token IDs；system/tool prefix boundary 使用独立有界 LRU，避免热请求重复渲染/分词 8K+ Prompt。

SSE 首 delta 立即发送，随后每 8 个 delta 合并；worker/API 队列从 8 增至 128，降低推理线程被 HTTP 背压打断的概率。合并按 delta 数而非时间，低速生成时仍可能增加后续 chunk 等待，需流式延迟回归。

主要目标是 host 模板、tokenization、跨线程唤醒和发送开销。模型内部 TTFT 与 HTTP wall time 的边界不同，不能把差值直接写成 GPU 性能提升。

没有本 commit 与 parent 的严格消融数据；历史 AIOS 余项 1.411→0.181 s 不能作为本次优化收益。

用户要求本轮保持其他任务的 GPU 服务和编译不变。本页是测试记录占位，不是已执行的性能结果；
不得将历史 dirty-worktree 的 1.520 s 结果归入此版本。

| 阶段 | 时间 | 占比 |
|---|---:|---:|
| 模型调用 1 Prefill/TTFT：full/reused/computed 待测 | 待测 | 待测 |
| 模型调用 1 Decode：输出 token 和间隔数待测 | 待测 | 待测 |
| AMap 查询及工具编排间隔 | 待测 | 待测 |
| 模型调用 2 Prefill/TTFT：full/reused/computed 待测 | 待测 | 待测 |
| 模型调用 2 Decode：输出 token 和间隔数待测 | 待测 | 待测 |
| AIOS、模型服务、HTTP 及其他余项 | 待测 | 待测 |
| **端到端总计** | **待测** | **待测** |

Prefill/TTFT 包含首 token 生成和缓存恢复；工具间隔不等于纯 AMap API 时间。
待测项目包括：该精确源码版本的 release 二进制哈希、recipe 清单、AIOS 请求参数、
两轮预热与十轮正式结果、同轮 server profile、工具正确性及 profiler 消融。
`temperature=0` 对应 greedy MTP；不得把非零 temperature 请求标签当成实际 sampling MTP。
`scripts/bench_qwen35_aios_weather.py` 在最终提交中提供记录工具；本页以后使用该工具补测。
record-path 与 commit 的映射由提交正文中的 `Weather-Record` trailer 固定。
