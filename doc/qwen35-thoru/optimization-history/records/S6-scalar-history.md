# S6 — 恢复 scalar 历史天气记录

本提交为证据归档，不修改推理实现。天气case为`明天上海的天气怎么样`。
已恢复的旧版本记录：[a50a058](a50a058c.md)（sampling scalar，有/无Prefix Cache）和[96344b1](96344b12.md)（greedy scalar）。
只将原始AIOS SSE/HTTP计时标为实测；模型拆分来源为旧PROFILE报告中的圆整数值，不当作原始CUDA trace。

CPU数据检查：3份历史切分逐项求和均等于各自E2E，所有占比合计100%；原始文件和二进制校验和已记录。
精确版本重测仍待独占Thor-U，未新跑天气case或Nsight。

| 阶段 | 时间 | 占比 |
|---|---:|---:|
| 模型调用1 Prefill/TTFT及tokens | 待测 | 待测 |
| 模型调用1 Decode及间隔数 | 待测 | 待测 |
| AMap查询及工具编排 | 待测 | 待测 |
| 模型调用2 Prefill/TTFT及tokens | 待测 | 待测 |
| 模型调用2 Decode及间隔数 | 待测 | 待测 |
| AIOS、模型服务、HTTP及其他 | 待测 | 待测 |
| **端到端总计** | **待测** | **待测** |

本提交不存在新的Prefill/Decode提升主张，不能拿3个历史单样本代替当前提交的测试。
