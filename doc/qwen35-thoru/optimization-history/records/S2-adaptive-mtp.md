# S2 — Adaptive MTP、GPU acceptance 和 GDN verification

天气 case：`明天上海的天气怎么样`。状态：**待 Thor-U 独占测试**。

把 MTP 最大 draft depth 从 2 改为 3；根据近期接受率和 round wall time 在 depth 1/2/3 间选择，预计慢于配置的 scalar TPOT 时回退。

各 depth 保留独立 verifier 资源；draft token 链留在 GPU，接受判断与结果打包在 GPU 完成，host 读取 8 个 I32（32 字节）。

GDN 卷积、head normalization 与短序列 recurrent verification 合并并保留逐 row rollback checkpoint；target/draft commit 仍有 host 控制，尚非完整 round CUDA Graph。

prompt snapshot LRU 保留 4 项，tool follow-up 的 volatile tail 保留 64 token；静态 system、exact prompt 和 live conversation cache 分别命中。

增加同轮 MTP profile/诊断路径，修正 serving 的 3-token lookahead 容量。



预期收益：Decode 减少同步和 launch；Prefix Cache 减少重复 Prefill。默认 19.42 ms scalar 阈值只是旧测量参考，需同场重新校准，不能当作当前硬件常量。

精确 shape 会改变 recipe，补测时必须先 autotune，再无 autotune 正式运行。独立性能消融和 CUDA 精度验证尚待进行；历史局部 7.1% 与此 commit 不建立归因。

CPU 验证：独立使用 `rustc --edition=2021 --test mtp_schedule.rs`，6/6 通过；
包括 scalar 等价、zero/partial/full acceptance、EOS、输出预算和失败事务。模型边界检查通过。
这些 CPU 检查不建立 CUDA 数值正确性或端到端性能结论。

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
