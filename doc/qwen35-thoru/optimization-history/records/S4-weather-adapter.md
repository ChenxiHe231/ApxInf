# S4 — 天气工具结果压缩与可复核记录工具

天气 case：`明天上海的天气怎么样`。已完成精确版本 CUDA 构建及真实天气测试。

将原 devlocal 请求代理整理为 `scripts/qwen35_aios_proxy.py`；`--compress-amap-weather` 显式启用，根据请求日期仅保留次日预报与天气必要字段。默认不变。

维护 `scripts/bench_qwen35_aios_weather.py`：不启动/停止任何服务，读取同轮 server profile 与代理时间戳，输出原始 JSON 和阶段 Markdown 表。记录失败/重试，不把成功子集的中位数当全组成绩；检查天气工具调用、city、工具返回和次日温度/天气。

CPU 验证：`python3 -m unittest discover -s scripts -p test_qwen35_aios_weather.py -v`，7/7 通过。



## 历史运行记录（非本 commit 实测）

`evidence/20261103-dirty-worktree-weather.json` 保存当时 10/10 正确的结果、代表样本同轮 HTTP 时间戳与内部 profile 切分。来源是未提交工作区；exact commit、binary SHA256 未保存，无法归因到此 commit。

| 历史样本阶段 | 实测/衍生时间 | 占比 |

|---|---:|---:|

| 模型调用1 Prefill/TTFT：full/reused/computed=8853/8852/1 | 0.047459 s | 3.11% |

| 模型调用1 Decode：38 tokens，37 个间隔 | 0.421494 s | 27.60% |

| AMap 查询及工具编排间隔 | 0.419903 s | 27.50% |

| 模型调用2 Prefill/TTFT：full/reused/computed=9058/8994/64 | 0.123388 s | 8.08% |

| 模型调用2 Decode：25 tokens，24 个间隔 | 0.334020 s | 21.87% |

| AIOS、模型服务、HTTP及其他余项 | 0.180697 s | 11.84% |

| **端到端总计（历史第6轮）** | **1.526960 s** | **100%** |



历史全组中位数 1.519645 s、P90 1.6299 s；上表不是全组各阶段中位数。

原始 Tool Result 902→196 字符、prompt 9322→9058 token 是历史观测。缓存热态新增 Prefill 均为64 token，因此不能把264个完整prompt token减少宣称为新增Prefill计算减少。

历史代表样本第二轮 Decode .642→.334 s 同时受回答长度和接受率影响，不能写成已证明的独立 kernel 提速；需要 parent/new 的成对消融。




## 精确版本重建实测（2026-11-03 起）

Source: `586edc9e3c19a81a978e25af13d07eb44b1a7cf5`；tracked 工作区干净。

Binary SHA256: `29a3dcf892b5c9ea66ef431435381f3a5e9981d095ce291d571577e9dbf147dc`。

配置与逐轮记录：[JSON](../evidence/586edc9e-clean-weather.json)。

正式 1 轮，回答校验通过 1 轮，完整有效阶段记录 1 轮。

端到端单轮实测（不是统计中位数）：**12.073310 s**。

以下为同一真实请求 index=2，回答校验=True；不拼接不同请求的阶段分位数。

| 阶段 | 时间 | 占比 |
|---|---:|---:|
| 模型调用1 Prefill/TTFT：新增 1 tokens | 0.132579 s | 1.10% |
| 模型调用1 Decode：37 个间隔 | 0.538150 s | 4.46% |
| AMap天气查询（AIOS工具计时） | 10.553400 s | 87.41% |
| 模型调用2 Prefill/TTFT：新增 65 tokens | 0.325601 s | 2.70% |
| 模型调用2 Decode：19 个间隔 | 0.331562 s | 2.75% |
| AIOS、模型服务、HTTP及其他余项 | 0.192017 s | 1.59% |
| **端到端总计** | **12.073310 s** | **100%** |

| 模型调用 | full tokens | reused tokens | computed tokens | output tokens | TPOT |
|---|---:|---:|---:|---:|---:|
| 1 | 8853 | 8852 | 1 | 38 | 14.545 ms |
| 2 | 9059 | 8994 | 65 | 20 | 17.451 ms |

TTFT 包括缓存恢复与首 token；余项含未归属的服务/调度/HTTP时间。工具时长为AIOS执行计时，不能声称等于纯网络时间。当前没有仅凭该单轮成立的单项优化提升百分比。

正式测试前后 recipe 数：1249 → 1249；清单内容未变。旧版默认在线调优可能进入正式请求时间，不能直接当作纯GPU kernel性能。

独立 Nsight capture：[kernel/API/memcpy 汇总](../evidence/586edc9e-nsys-stats.csv)。采集时延不进入未插桩正式成绩。
