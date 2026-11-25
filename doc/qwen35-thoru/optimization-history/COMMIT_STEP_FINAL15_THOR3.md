# Thor3：完整15项逐版本验收

已独立重解析全部原始输出、配置、摘要、工作量、文件哈希和频率。本节完整六版本表中，B6/Prod 是新鲜成对对照；前四版为已冻结的同机结果，百分比不可相加。完整协议通过不自动代表无回退、20%达标或已推送。

固定1530MHz（原最高1575MHz），2轮预热/10轮测量；NVFP4 MoE、静态FP8投影、动态FP8 KV、greedy、MTP depth3。Decode首token归TTFT，随后1023间隔归Decode。

## 主6项 Decode tok/s

| Case | Final02 | P1 | P2 | Safe397 | B6 | Prod |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| decode-4k | 110.489319 | 110.808477 | 111.363998 | 109.382706 | 109.646897 | 111.146379 |
| decode-8k | 119.657355 | 119.935274 | 120.240200 | 118.956465 | 120.203970 | 119.442625 |
| decode-16k | 115.810420 | 116.463567 | 116.491456 | 116.506295 | 116.329855 | 115.784831 |
| decode-24k | 120.154408 | 120.435407 | 120.933554 | 120.673517 | 120.087195 | 121.591100 |
| decode-32k | 98.275813 | 97.417475 | 98.155762 | 96.896805 | 96.184121 | 96.496709 |
| decode-48k | 105.929525 | 106.061205 | 106.040671 | 106.639944 | 105.687167 | 105.875666 |

| 版本 | 较 Final02 几何平均改善 | 较显示前版改善 |
| --- | ---: | ---: |
| Final02 | 0.0000% | 0.0000% |
| P1 | 0.0939% | 0.0939% |
| P2 | 0.4158% | 0.3216% |
| Safe397 | -0.2179% | -0.6311% |
| B6 | -0.3823% | -0.1648% |
| Prod | -0.0496% | 0.3340% |

## 5项追加 Prefill ms

| Case | Final02 | P1 | P2 | Safe397 | B6 | Prod |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| append-4k | 524.677949 | 504.363063 | 469.500387 | 460.873205 | 452.817035 | 414.764130 |
| append-8k | 553.025449 | 533.021489 | 497.420385 | 488.631562 | 483.322657 | 441.767887 |
| append-16k | 605.079646 | 585.517217 | 547.832813 | 539.313042 | 528.256091 | 495.320034 |
| append-32k | 694.178457 | 667.955943 | 633.052529 | 622.861662 | 616.251091 | 580.360706 |
| append-48k | 786.640156 | 760.381063 | 724.823054 | 717.411237 | 711.729016 | 674.937072 |

| 版本 | 较 Final02 几何平均改善 | 较显示前版改善 |
| --- | ---: | ---: |
| Final02 | 0.0000% | 0.0000% |
| P1 | 3.5679% | 3.5679% |
| P2 | 9.3440% | 5.9899% |
| Safe397 | 10.7572% | 1.5588% |
| B6 | 11.9609% | 1.3487% |
| Prod | 17.9971% | 6.8564% |

## 4项固定长文本 Decode tok/s（非真实QA验收）

| Case | Final02 | P1 | P2 | Safe397 | B6 | Prod |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| eligible-decode-4k | 131.657452 | 132.345359 | 132.359869 | 129.169447 | 129.887566 | 129.851734 |
| eligible-decode-16k | 125.633234 | 126.167381 | 125.961311 | 127.493109 | 127.723731 | 126.725239 |
| eligible-decode-32k | 98.485020 | 97.862069 | 98.068627 | 98.333442 | 98.332068 | 99.061572 |
| eligible-decode-48k | 113.079378 | 112.679766 | 113.126357 | 111.919220 | 111.430247 | 111.889836 |

| 版本 | 较 Final02 几何平均改善 | 较显示前版改善 |
| --- | ---: | ---: |
| Final02 | 0.0000% | 0.0000% |
| P1 | -0.0108% | -0.0108% |
| P2 | 0.1027% | 0.1135% |
| Safe397 | -0.4051% | -0.5073% |
| B6 | -0.3314% | 0.0740% |
| Prod | -0.2470% | 0.0846% |

## 所有负点（保留点估计）

- Final02 较Final02：无负点。
- Final02 较显示前版：无负点。
- P1 较Final02：decode-32k -0.8734%；eligible-decode-32k -0.6325%；eligible-decode-48k -0.3534%。
- P1 较显示前版：decode-32k -0.8734%；eligible-decode-32k -0.6325%；eligible-decode-48k -0.3534%。
- P2 较Final02：decode-32k -0.1222%；eligible-decode-32k -0.4228%。
- P2 较显示前版：decode-48k -0.0194%；eligible-decode-16k -0.1633%。
- Safe397 较Final02：decode-4k -1.0016%；decode-8k -0.5857%；decode-32k -1.4032%；eligible-decode-4k -1.8898%；eligible-decode-32k -0.1539%；eligible-decode-48k -1.0260%。
- Safe397 较显示前版：decode-4k -1.7791%；decode-8k -1.0676%；decode-24k -0.2150%；decode-32k -1.2826%；eligible-decode-4k -2.4104%；eligible-decode-48k -1.0671%。
- B6 较Final02：decode-4k -0.7624%；decode-24k -0.0559%；decode-32k -2.1284%；decode-48k -0.2288%；eligible-decode-4k -1.3443%；eligible-decode-32k -0.1553%；eligible-decode-48k -1.4584%。
- B6 较显示前版：decode-16k -0.1514%；decode-24k -0.4859%；decode-32k -0.7355%；decode-48k -0.8935%；eligible-decode-32k -0.0014%；eligible-decode-48k -0.4369%。
- Prod 较Final02：decode-8k -0.1795%；decode-16k -0.0221%；decode-32k -1.8103%；decode-48k -0.0508%；eligible-decode-4k -1.3715%；eligible-decode-48k -1.0520%。
- Prod 较显示前版：decode-8k -0.6334%；decode-16k -0.4685%；eligible-decode-4k -0.0276%；eligible-decode-16k -0.7818%。

## 产物与边界

版本代号、候选/正式交付提交、逐点样本、最大回退及原件 SHA 见[逐轮证据](evidence/commit-step-final15-thor3.json)。P1/P2 历史测量为冻结候选产物加共同验证 overlay，未移签为后来的整仓提交。Safe397 为集成版本，不能把其相邻表格差额当单一算子因果。
完整输出向量只存原始包，本报告保留摘要/长度、逐位一致及工作量验证结论，不向 Git 重复写入每轮1024 token。
20%观察：{"baseline": "Final02", "mainDecode6_geomean_gain_percent": -0.049644495186462745, "append5_geomean_latency_reduction_percent": 17.997147723595408, "Decode20": false, "append20": false, "both20": false, "Decode_points_passing": 0, "Decode_points": 10, "append_points_passing": 2, "append_points": 5, "automatically_accepted": false, "interpretation": "Individual points crossing20 do not establish aggregate goal. Current decode geometric mean is approximately flat, append geometric mean is below20. Final push/regression acceptance remains separate."}。当前Final的两代表点回退复核另列；两个整体20%目标仍未达。

正式交付：B6 对应 `81e2e942`，Prod 对应包含本文的独立 FC2 提交；代码与实验 b6/d875 源码逐字一致，文档整理没有重复 GPU 测量。完整标量表可[下载 CSV](evidence/commit-step-final15-thor3.csv)。当前远程 Final 两代表点的公平复核已通过，见[独立结果](CURRENT_FINAL_FAIR_THOR3.md)。

表中 Prod 对应本轮 ApxInf 4.0 测试组合，实际源码与产物身份见逐轮证据。

## 历史优化、本轮增量与尚未达到的目标

warp-prefix（`a5cca938`/`f25fe56d`）、FFN并行（`1111b413`/`9ae60d1b`）、GDN只读状态/必要快照组合（`0e7720be`）、draft词表裁剪（`7cc20e50`）、位置及复制Graph（`595a46da`）、视觉GEMM（`9f7aecd2`）已经在冻结Final02中，不能再次计入本轮20%目标。旧Thor2/Thor-U结果、视觉动态频率/32输出结果与本轮Thor3固定1530MHz/长上下文1024输出协议不同，均只作历史证据，不能移签或相加。

**verify-head half2没有进入Final。** 早期Thor-U两组13题曾有+0.37%/+0.43%，但最新最快基线上的260请求ABBA仅为−0.0109%/+0.0179%，未复现稳定收益，已停止，不能记为已合入成果。

| 本轮交付映射 | 正式受影响N5 | 完整N10的范围 |
| --- | --- | --- |
| P1 `f5ff8ddd`，实验5ad/prefill04 | 旧Thor2追加时延−2.60%/−2.95% | 新Thor3 P1/Final02五档追加几何平均−3.5679% |
| P2 `1c469d2c`，实验451b/routed-quant02 | Thor3相对P1，−7.03%/−5.39% | P2/P1追加−5.9899%；累计P2/Final02直接实测−9.3440% |
| 默认策略修正 `2e236bb2` | 自动选择已启用的三项MTP路径，**新增性能信用0%** | 不把旧FLAGS1已有能力重复算收益 |
| norm / Safe397 `39784163` | 单独norm相对实际父C，−2.97%/−2.55% | Safe397为上游/默认策略/recipe等集成版，其N10差额不能全部归因norm |
| 卷积 `81e2e942`，实验b6 | 相对Safe397，−0.91%/−1.12% | B6/Safe397追加−1.3487%，Safe分母为冻结同机结果 |
| FC2 store（包含本文的提交），实验d875 | 相对b6，−8.83%/−7.28% | 新鲜B6/Prod成对追加−6.8564%，各点−5.17%～−8.60% |

以上N5均为各自实际输入/产物的2warm5，N10为完整15项2warm10，百分比不能累加。P1/P2测量为冻结候选加共同benchmark验证overlay，后续产品提交映射不代表旧ELF被移签；只有B6/Prod是本次新鲜成对对照。

Prod/Final02的主六档Decode几何平均为**−0.0496%**，五档追加时延降低**17.9971%**，两个整体20%目标仍未达到。4K/8K追加单点已超过20%，不代表五档整体或逐项验收通过。仅按汇总门槛换算，当前追加时延还需再降约2.4424%，Decode吞吐还需提高约20.0596%；这不是下一候选收益预测。主表六档Decode、五档追加和四档固定文本分别保留，固定文本不替代真实QA。

全部N10输出/FNV/工作量/身份/recipe/频率及恢复守卫通过，但不能自动称“无明显回退”。Prod/Final02的32K Decode为**−1.8103%**，其他负点和全部样本波动仍见主表。最终Prod对当前Final3b最快FLAGS1的新鲜4K/32K公平复核已通过，独立结果另列；两个20%目标仍分别验收。

| 已停止路径 | 实际结论 |
| --- | --- |
| 原EGLops MoE | block8/GEMM/aggregate为unsupported，不能写成测得端到端0% |
| 原EGLops GDN | 旧T4完整适配约慢6.06倍；剥离接入成本后本体仍慢84.2%，不能全归咎配置或适配 |
| 新draft eligibility D | 完整范围24K Decode约−5.14%，已从Safe/交付产物剔除，早期N5正信号不再计默认交付收益 |
| Router warp-local | GPU逐位门禁通过，4096短筛AB/BA event区间反增约17.30%/17.84%，立即停止，不跑完整模型 |
| MR寄存器 / FC2双驻留 / 转置N8 | 分别因资源退化（NCU仪器时间+23.08%，非E2E）、实际occupancy仍1、全部代表短筛严重负向止损；未当模型成果 |
| 其他主要源门禁 | Lt query32仍仅原8算法、head已跨行复用、outerGraph gap太小且含真实FA工作；N256/旧F16 WMMA等旧否定路径不重开 |

本轮复用的是实际shape、布局与默认路径中的有效优化，并非将原EGLops包直接替换。已实际成功采集node级NSys；旧“Thor不用危险node trace”建议不能当作当前从未使用node的事实。新d875的363轮/14520目标图、80图ID、实际router grid4、源码及CUDA关联共同证明首个verify M4的FC1 skip40、FC2 skip0；后者因大Prefill store与M4旧核实名不同，不能照搬跳过40。

成对NSys已支持norm/卷积/store的组件变化；actual-model与合成算子NCU、cold与warm replay仍按各自原件身份区分。组件union、寄存器/occupancy或仪器耗时不直接等于模型收益；缺失DRAM/stall/关键路径f继续写MISSING。所有候选保持冻结精度、原最快prepared/Graph对照和累计预算；无信号止损，不为整理文档重跑基线。

按主表逐项门槛，48K 追加仍需在 Prod 基础上再降约 6.76%，32K Decode 仍需提高约 22.21%；这些是距目标的算术差额，不能作为候选收益预测。

旧 GDN 的 T4 指验证长度为4；本轮卷积 temporal4 是4096-token prefill内相邻4个token的复用，真实形状和状态布局不同。原验证算子负收益路径保持停止。

## Router 候选的止损诊断

旧GPU核逐位/NaN/Inf/同值ID/只读/canary检查通过，但4096行prepared Graph短筛AB/BA时延分别增加17.30% / 17.84%，停止性能推进，没有模型试验或性能提交。一组实际NSys确认每臂6次同形状调用；两边均35寄存器、32B stack、LOCAL0，不能归因于寄存器增多或占用率下降。额外全lane比较/shuffle指令与依赖开销只是尚未获stall计数验证的假设。原始包SHA256 `c991bbdd6a8d2a7510714fa2de0ef472f2087ae203294dced2261cd76e90ce24`；完整止损卡SHA256 `6487500a72aa3c748ddc9f93ae7410940219c903a4c07495a9817067bc0241e4`。
