# Qwen3.6 Thor-U / AIOS optimization history

本目录是版本与证据台账。源码已整理并推送；用户于 2026-11-03 释放 GPU 后，已开始在独立干净 checkout 中逐 commit 编译、调优和实测。**补测正在进行，不能把尚待测的页面当成成绩。**

首个干净重建结果为 `975631ca`：10/10 轮最终天气回答通过次日工具数据校验，端到端中位数 **3.315455 s**，其中 7/10 轮重复查询天气，需要 3 次模型调用。完整同轮阶段表见 [S7](records/S7-live-measurement.md)，原始计时见 [JSON](evidence/975631ca-clean-weather.json)。这不复现旧未提交工作区的 1.519645 s，也不能据此声称最终整理版本比 TRT 快。

## 版本边界

- 历史起点：`fd6520a`，对应最初 AIOS 优化前的分支状态。
- 收集开始时远端：`2c24e07d67565ee9251fb71fc480ef9d86823665`。
- 并行任务随后已推送 `36d0d0f66141bce7944c5041fae9f61d9294a32b`；本序列保留它并在其后增量提交。
- 本序列在 `codex/qwen35-aios-optimization-series` 独立分支整理，不覆盖并行任务分支。
- 原工作区以及 thor-u 的额外未提交改动保留；来源不确定的 norm/精度测试、工具参数转换等不混入此序列。

## 新增提交顺序

| 提交主题 | 主要内容 | 天气记录 |
|---|---|---|
| FP8 attention tensor-core QK | BF16 WMMA QK 和 split 上限配置 | [S1](records/S1-fp8-attention.md) |
| Adaptive MTP + GPU acceptance + GDN verification | depth 1/2/3、独立 verifier shape、GPU token 链/接受打包、GDN checkpoint、精确 prompt cache | [S2](records/S2-adaptive-mtp.md) |
| Serving prompt/SSE cache | 缓存模板/tokenization、prefix boundary 和 SSE 合并/缓冲 | [S3](records/S3-serving.md) |
| Weather adapter and reproducible evidence tooling | opt-in 天气结果压缩、请求记录代理、阶段统计脚本和历史原始数据 | [S4](records/S4-weather-adapter.md) |
| Historical evidence and date-policy audit | 恢复2c24e07包的同轮profile；要求显式目标预报日期 | [S5](records/S5-date-audit.md) |
| Scalar historical evidence | 恢复a50a058和96344b1包的原始E2E/HTTP计时与报告精度限制 | [S6](records/S6-scalar-history.md) |
| Live phase collector | 读取 AIOS 工具计时；展开所有实际模型/工具调用，保留额外查询和失败 | [S7](records/S7-live-measurement.md) |

MTP depth、backend proposal 常量、verifier 资源、GDN checkpoint 与 host commit 合约耦合，作为一个完整提交；不能拆成中间运行时相互不匹配的版本。

每个新提交正文包含 `Weather-Record: <path>` 和 `Weather-Status: pending-exclusive-thor-u`。历史 commit 不改 hash，使用下面的台账关联测试页。

## 历史 commit 台账

| Commit | 改动 | 天气 case / profile 状态 |
|---|---|---|
| `fd6520a` | 原始起点 | [待测](records/fd6520a.md) |
| `ceba744b` | fix(qwen35): reuse AIOS system prefix for sampled requests | [待测](records/ceba744b.md) |
| `a6d50b55` | perf(cuda): accelerate small top-k sampling | [待测](records/a6d50b55.md) |
| `294eb1a9` | fix(qwen35): make context autotune bench match scalar serving | [待测](records/294eb1a9.md) |
| `c67fa795` | Revert "perf(cuda): accelerate small top-k sampling" | [待测](records/c67fa795.md) |
| `5db33206` | docs(qwen35): make deployment autotune explicit | [待测](records/5db33206.md) |
| `9fba8543` | perf(qwen35): reuse scalar benchmark argmax output | [待测](records/9fba8543.md) |
| `dcf5e497` | fix(qwen35): keep live decode attention on FA2 | [待测](records/dcf5e497.md) |
| `dfcbb255` | perf(attention): bucket live decode recipes | [待测](records/dfcbb255.md) |
| `8ad5b97c` | feat(serving): accept OpenAI top-k sampling | [待测](records/8ad5b97c.md) |
| `3423907f` | feat(qwen35): make sampling autotune explicit | [待测](records/3423907f.md) |
| `54e22691` | fix(qwen35): retain sampling prefix variants | [待测](records/54e22691.md) |
| `dd5d7374` | style(qwen35): format prefix cache LRU | [待测](records/dd5d7374.md) |
| `a50a058c` | docs(qwen35): require complete sampling recipes | [历史包记录已恢复；模型切分为报告估算，重测待执行](records/a50a058c.md) |
| `96344b12` | feat(serving): allow forcing scalar decode | [历史包记录已恢复；模型切分为报告估算，重测待执行](records/96344b12.md) |
| `c6f02352` | bench(qwen35): make MTP recipes explicit | [待测](records/c6f02352.md) |
| `fe372dfc` | perf(qwen35): annotate MTP phases for profiling | [待测](records/fe372dfc.md) |
| `8f7fd233` | perf(qwen35): prepare fixed-shape MTP verification | [待测](records/8f7fd233.md) |
| `4868a580` | fix(qwen35): keep live attention outside verifier session | [待测](records/4868a580.md) |
| `c3606422` | feat(qwen35): page KV cache in 128-token blocks | [待测](records/c3606422.md) |
| `11f638b8` | feat(qwen35): load TensorRT static FP8 KV scales | [待测](records/11f638b8.md) |
| `6265b8fa` | Optimize Qwen3.6 MTP2 and FP8 prefix reuse | [待测](records/6265b8fa.md) |
| `2c24e07d` | fix(qwen35): reuse live MTP context across tool turns | [历史包记录已恢复；重建重测待执行](records/2c24e07d.md) |
| `36d0d0f6` | fix(qwen35): recover missing recipes and share cuBLAS handles | [待测](records/36d0d0f6.md) |

## 历史成绩的适用范围

2026-11-03 的 10 轮天气 case 中位数 1.519645 s、最快 1.48636 s，是当时未提交工作区与特定 recipe/代理配置的运行记录。它证明该次运行的端到端结果，**无法证明本目录任一精确 commit 的性能**。原始输出整理在最终提交的 `evidence/` 中。
无法复原的二进制/源码哈希标为 unknown，不从文件名猜测。新序列继承并行任务的 36d0d0f，也不同于此前历史测试的源码组合。

另已恢复声明为 `2c24e07` 的历史部署包：二进制 SHA256 已核对，5轮 AIOS 输出、逐调用 profile 和 HTTP 时间戳可复查；源码归属只有包内声明，未建立干净重建证明。日期审计发现 UTC/上海跨午夜歧义，旧“5/5成功”不能当成5轮日期正确。详见该版本记录与 `evidence/2c24e07-package-weather.json`。

历史 AMap 记录为两次模型 HTTP 之间的间隔，包含调度与序列化，不等于纯 API 时间。新实测直接读取 AIOS 的工具执行时长；剩余编排间隔归入余项。模型时间以 server profile 为准；TTFT 包括缓存恢复及首 token。其余为差值，不能据此宣称 AIOS 或 AMap 被 kernel 优化加速。旧 3.92 s 表无可定位的完整 trace，不能用于分阶段增益归因。

## 补测要求

依序在干净 checkout 构建各 commit，记录源码树、二进制 SHA256、完整 argv/environment、checkpoint/recipe 哈希与时钟。每个历史版本至少保留 1 轮正式天气记录；性能中位数比较使用 10 轮，不能将单轮叫作中位数。支持显式调优的版本先单独 autotune，再关闭 autotune、预热 2 轮。较早版本没有 serving 调优开关，按原有默认在线调优行为运行，保留 3 轮调优/预热请求并明确注明差异；不补造旧版不支持的配置。
每轮输出用户要求的阶段表，标明 full/reused/computed tokens、completion tokens/间隔数；同一轮的所有阶段之和应等于该轮 E2E。不得拼接不同样本的分位数。
正确性须检查结构化 AMap tool call、非空 city、工具成功、次日日期和返回温度/天气一致；仅有天气文本不足以判定通过。

性能消融需固定真实请求/工具结果、采样语义、缓存热度；分别比较 prefill TTFT、decode TPOT/接受率、HTTP 和 E2E。Nsight 插桩会改变时延，profiler 与未插桩正式计时分开记录。
建议启用 `APXINF_QWEN35_MTP_TIMING=1` / `APXINF_QWEN35_VERIFY_LAYER_TIMING=1` 做诊断，配合 `nsys profile --trace=cuda,nvtx,osrt` 检查 kernel、graph launch、同步和 D2H；没有 trace 不填写带宽/算力利用率或单项性能提升百分比。
