# Qwen3.6 / Thor-U：优化提交与天气实测报告

最新源码 `4424f1b7` 已在干净 checkout 构建：**AIOS 天气正式 10/10 完整通过，端到端中位数 1.774990 s，P90 1.964870 s**。二进制 SHA256：`b50755001eab6818bfd1c922c1ed04f0010940cafd2dd36129a37130bec46b90`。这组请求在上海 2026-11-03 午夜前完成，目标预报日期为 11 月 4 日。

提交分支：`codex/qwen35-aios-optimization-series`。从原始 `fd6520a` 开始逐个干净重建，保留所有原有 commit hash；新增优化在 `36d0d0f6` 后增量整理。每个版本都有真实天气请求、二进制/模型/recipe 哈希与失败记录，不把旧未提交工作区的成绩填入新版本。原有两端 dirty 工作区未切换、未清理。

本次配置是 temperature=0 的 greedy MTP、NVFP4 MoE、静态 FP8 KV、Prefix Cache、部分阶段 CUDA Graph，代理启用按上海日期归一化、次日天气结果压缩及单工具响应早停。**这组成绩不代表 sampling MTP、冷缓存或未经代理处理的所有 AIOS 任务。** 没有同场重测 TRT，因此不能用旧 TRT 2.123 s 宣称已实现领先至少20%。

## 最新同轮阶段分布

下表是最接近中位数的一个真实请求，1.728600 s；10轮总体中位数为1.774990 s。偶数样本的中位数不是某个实际请求，不能把各阶段分位数拼成虚构的一轮。

| 阶段 | 时间 | 占比 |
|---|---:|---:|
| 模型调用1 Prefill/TTFT：新增 1 tokens | 0.138999 s | 8.04% |
| 模型调用1 Decode：36 个间隔 | 0.440207 s | 25.47% |
| AMap天气查询（AIOS工具计时） | 0.386200 s | 22.34% |
| 模型调用2 Prefill/TTFT：新增 64 tokens | 0.182843 s | 10.58% |
| 模型调用2 Decode：24 个间隔 | 0.395373 s | 22.87% |
| AIOS、模型服务、HTTP及其他余项 | 0.184978 s | 10.70% |
| **端到端总计** | **1.728600 s** | **100%** |

| 模型调用 | full tokens | reused tokens | computed tokens | output tokens | TPOT |
|---|---:|---:|---:|---:|---:|
| 1 | 8855 | 8854 | 1 | 37 | 12.228 ms |
| 2 | 9064 | 9000 | 64 | 25 | 16.474 ms |


完整请求、成功率及参数见 [最新版本记录](records/4424f1b7.md) / [脱敏逐轮 JSON](evidence/4424f1b7-clean-weather.json)。AMap列来自AIOS实际工具计时，余项是同轮E2E减去模型与工具时间；它不是单独测得的纯网络开销。TTFT包含缓存恢复和首token，不等于纯Prefill kernel时间。

## 改了什么，哪些收益已有证据

| 提交 | 改动与作用阶段 | 实测证据与结论 |
|---|---|---|
| `b452e715` | FP8-cache attention 的 QK 改用 BF16 WMMA、FP32 accumulator；影响 Decode/verification | 对照 trace 中 fused-split kernel 平均 740.9→546.3 µs（下降26.3%）；shape混合不同，只是观测，不等于E2E提升26.3% |
| `118337ce` | MTP depth 1/2/3、自适应选择、GPU draft链与接受打包、独立verifier/GDN checkpoint；GDN Prefill 的64-token整块加精确尾段；exact prompt/LRU复用 | Prefill实际新增量从291/432等降至1/64；完整trace的GDN累计时间显著减少，包含少算token与kernel融合两种因素。关闭复用的相同291-token输入，3次TTFT中位数580.3→292.3ms，下降49.6%；320-token控制组无收益。见profiler报告 |
| `679322b0` | 模板/tokenization LRU、SSE合并/缓冲 | 减少host准备和发送工作；单轮E2E 1.985230 s，不足以给出独立因果百分比 |
| `586edc9e` → `4424f1b7` | 次日天气结果压缩、保留forecasts数组、逐条处理重复结果、上海时区 | 减少Agent输入/歧义，影响回答长度和是否重复查天气；不是GPU算子加速。早期压缩单轮仍12.073310 s，保留不删 |
| `9596e6f8` | 显式parallel_tool_calls=false时，在原生单token闭合标记处终止并保留解析标记 | 该版本正式10/10成功，中位数1.775060 s；此前9d批次8/10成功，不能把其成功子集当成有效对照中位数 |
| `4ec67457` | 全流检查SSE末尾错误、日期/city校验 | 纠正原975批次误报：天气文字10/10正确，协议无错9/10；该提交单轮1.939220 s，不声称推理加速 |

详细 [Prefill/Decode profiler与归因边界](PROFILER.md)、[历史接入回归的代码解释](INTEGRATION_REGRESSIONS.md)、[复现与启动方法](REPRODUCE.md)。未完成的完整round CUDA Graph、动态/细粒度FP8校准、Ncu算力/带宽利用率与TRT同场比较明确留作后续；本报告不将它们写为已实现。

## 每个提交的天气记录

每个版本至少1轮；只有全部严格通过且有阶段计时才列有效值。单轮明确标注N=1，不作统计中位数。失败/缺少计时仍保留请求返回时间与错误，见链接；不能构造成功阶段表。

| Commit | 主题 | 正式有效/总轮数 | 有效E2E | 记录 |
|---|---|---:|---:|---|
| `fd6520a7` | chore(thor): serialize GPU work and refuse runs that cannot succeed | 0/1 | 回答正确；无内部阶段日志 | [实测](records/fd6520a.md) |
| `ceba744b` | fix(qwen35): reuse AIOS system prefix for sampled requests | 1/1 | N=1 18.746520 s | [实测](records/ceba744b.md) |
| `a6d50b55` | perf(cuda): accelerate small top-k sampling | 0/1 | 不作为有效成绩 | [实测](records/a6d50b55.md) |
| `294eb1a9` | fix(qwen35): make context autotune bench match scalar serving | 0/1 | 不作为有效成绩 | [实测](records/294eb1a9.md) |
| `c67fa795` | Revert "perf(cuda): accelerate small top-k sampling" | 1/1 | N=1 31.187710 s | [实测](records/c67fa795.md) |
| `5db33206` | docs(qwen35): make deployment autotune explicit | 1/1 | N=1 36.174720 s | [实测](records/5db33206.md) |
| `9fba8543` | perf(qwen35): reuse scalar benchmark argmax output | 0/1 | 不作为有效成绩 | [实测](records/9fba8543.md) |
| `dcf5e497` | fix(qwen35): keep live decode attention on FA2 | 1/1 | N=1 17.510840 s | [实测](records/dcf5e497.md) |
| `dfcbb255` | perf(attention): bucket live decode recipes | 1/1 | N=1 16.810070 s | [实测](records/dfcbb255.md) |
| `8ad5b97c` | feat(serving): accept OpenAI top-k sampling | 1/1 | N=1 20.568850 s | [实测](records/8ad5b97c.md) |
| `3423907f` | feat(qwen35): make sampling autotune explicit | 1/1 | N=1 18.667610 s | [实测](records/3423907f.md) |
| `54e22691` | fix(qwen35): retain sampling prefix variants | 1/1 | N=1 17.664020 s | [实测](records/54e22691.md) |
| `dd5d7374` | style(qwen35): format prefix cache LRU | 0/1 | 不作为有效成绩 | [实测](records/dd5d7374.md) |
| `a50a058c` | docs(qwen35): require complete sampling recipes | 0/1 | 不作为有效成绩 | [实测](records/a50a058c.md) |
| `96344b12` | feat(serving): allow forcing scalar decode | 0/1 | 不作为有效成绩 | [实测](records/96344b12.md) |
| `c6f02352` | bench(qwen35): make MTP recipes explicit | 0/1 | 不作为有效成绩 | [实测](records/c6f02352.md) |
| `fe372dfc` | perf(qwen35): annotate MTP phases for profiling | 0/1 | 不作为有效成绩 | [实测](records/fe372dfc.md) |
| `8f7fd233` | perf(qwen35): prepare fixed-shape MTP verification | 0/1 | 不作为有效成绩 | [实测](records/8f7fd233.md) |
| `4868a580` | fix(qwen35): keep live attention outside verifier session | 0/1 | 不作为有效成绩 | [实测](records/4868a580.md) |
| `c3606422` | feat(qwen35): page KV cache in 128-token blocks | 0/1 | 不作为有效成绩 | [实测](records/c3606422.md) |
| `11f638b8` | feat(qwen35): load TensorRT static FP8 KV scales | 0/1 | 不作为有效成绩 | [实测](records/11f638b8.md) |
| `6265b8fa` | Optimize Qwen3.6 MTP2 and FP8 prefix reuse | 1/1 | N=1 4.410280 s | [实测](records/6265b8fa.md) |
| `2c24e07d` | fix(qwen35): reuse live MTP context across tool turns | 1/1 | N=1 3.850490 s | [实测](records/2c24e07d.md) |
| `36d0d0f6` | fix(qwen35): recover missing recipes and share cuBLAS handles | 0/1 | 不作为有效成绩 | [实测](records/36d0d0f6.md) |
| `91c25d8e` | docs(qwen35): inventory optimization revisions and weather evidence gaps | 1/1 | N=1 3.638830 s | [实测](records/91c25d8e.md) |
| `b452e715` | perf(qwen35): use tensor cores for FP8-cache attention QK | 1/1 | N=1 3.574020 s | [实测](records/b452e715.md) |
| `118337ce` | perf(qwen35): add adaptive MTP and device-resident verification | 1/1 | N=1 2.042260 s | [实测](records/118337ce.md) |
| `679322b0` | perf(serving): cache rendered prompts and coalesce stream deltas | 1/1 | N=1 1.985230 s | [实测](records/679322b0.md) |
| `586edc9e` | feat(aios): track weather compression and per-revision timing evidence | 1/1 | N=1 12.073310 s | [实测](records/586edc9e.md) |
| `7e3bccbf` | fix(bench): qualify historical weather evidence and target dates | 1/1 | N=1 2.441770 s | [实测](records/7e3bccbf.md) |
| `975631ca` | docs(bench): recover scalar weather timings with provenance limits | 9/10 无错 | 不给有效总体中位数 | [S6/复核](records/S7-live-measurement.md) |
| `9d6642c5` | fix(bench): attribute every weather round and AIOS tool duration | 8/10 | 不作为有效成绩 | [实测](records/9d6642c5.md) |
| `4ec67457` | fix(bench): reject late model-stream errors in weather evidence | 1/1 | N=1 1.939220 s | [实测](records/4ec67457.md) |
| `9596e6f8` | perf(serving): stop single-tool responses at the native closing token | 10/10 | 中位数 1.775060 s | [实测](records/9596e6f8.md) |
| `4424f1b7` | fix(aios): preserve forecast lists and normalize the Shanghai date | 10/10 | 中位数 1.774990 s | [实测](records/4424f1b7.md) |

原始起点 `fd6520a` 没有内部profile日志：保留它自身的E2E，另建只加日志的 [41a5bbc7](records/41a5bbc7.md) 测量分支（单轮19.948270 s）。两轮总耗时不混拼。历史commit正文里的pending状态是当时真实状态；本目录与 `refs/notes/weather-e2e` 记录后续实测，不改写源码历史。

## 测量约束

每次编译前停止本任务服务并释放HugeTLB；无编译与推理并行。先3轮真实请求调优；支持显式开关的版本重启关闭autotune，预热2轮后正式测量。旧版无该开关，保留其默认在线调优行为，正式前后的recipe变化逐版本标明。实际CUDA为12.8，默认目录为SM101/CUDA12.8；曾误用目录的采集标为不可比较，表中已使用纠正后的重测。

GPU max1206MHz，simple_ondemand、min54MHz，未声称锁频。服务器墙钟落后Mac约25小时；模型和工具用各自monotonic时长，HTTP间隔仅用客户端时间，不跨机器相减。每份JSON保留时钟审计。Nsight是独立capture，额外同步会改变MTP选择，采集时延不混入正式成绩。

Prefix Cache 的1/64新增token是热态exact/live cache结果，第一次真正接入仍要建立静态前缀与模型状态。128-token物理KV分块不等于128-token量化分组；FP8当前使用checkpoint/TRT静态layer K/V scale，没有宣称动态校准。

## 历史成绩的适用范围

2026-11-03 的 10 轮天气 case 中位数 1.519645 s、最快 1.48636 s，是当时未提交工作区与特定 recipe/代理配置的运行记录。它证明该次运行的端到端结果，**无法证明本目录任一精确 commit 的性能**。原始输出整理在最终提交的 `evidence/` 中。
无法复原的二进制/源码哈希标为 unknown，不从文件名猜测。新序列继承并行任务的 36d0d0f，也不同于此前历史测试的源码组合。

另已恢复声明为 `2c24e07` 的历史部署包：二进制 SHA256 已核对，5轮 AIOS 输出、逐调用 profile 和 HTTP 时间戳可复查；源码归属只有包内声明，未建立干净重建证明。日期审计发现 UTC/上海跨午夜歧义，旧“5/5成功”不能当成5轮日期正确。详见该版本记录与 `evidence/2c24e07-package-weather.json`。

历史 AMap 记录为两次模型 HTTP 之间的间隔，包含调度与序列化，不等于纯 API 时间。新实测直接读取 AIOS 的工具执行时长；剩余编排间隔归入余项。模型时间以 server profile 为准；TTFT 包括缓存恢复及首 token。其余为差值，不能据此宣称 AIOS 或 AMap 被 kernel 优化加速。旧 3.92 s 表无可定位的完整 trace，不能用于分阶段增益归因。
