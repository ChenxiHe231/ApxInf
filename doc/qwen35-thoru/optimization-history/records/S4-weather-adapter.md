# S4 — 天气工具结果压缩与可复核记录工具

天气 case：`明天上海的天气怎么样`。状态：**待 Thor-U 独占测试**。

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



## 本 commit 的天气测试

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
