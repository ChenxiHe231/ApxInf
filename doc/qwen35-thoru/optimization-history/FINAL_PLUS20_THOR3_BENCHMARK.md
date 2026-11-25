# Thor3：Final → Prefill Session → Routed Quant 完整 Benchmark

> 数据状态：完整 15-case 正式矩阵与独立数值、源码及协议审查通过；两个 20% 目标均未达到。默认路径修正已通过后续独立检查并提交，新增性能收益计为 0%。

完整 15 case 已在同一台 Thor3 完成，2 轮预热、10 轮正式测量。当前 C 相对冻结 Final02 的 LCG Decode 吞吐变化为 −0.12%～+0.79%，追加 Prefill 时延降低 7.86%～10.52%；两个 20% 目标均未达到。32K LCG Decode 的 P1 为 −0.87%，C 为 −0.12%，不能以完整矩阵 PASS 表示没有性能回退或目标完成。

作为描述性汇总，6 组 LCG Decode 的几何吞吐比：P1 较 Final +0.09%，C 较 Final +0.42%（C 较 P1 +0.32%）；5 组追加 Prefill 的几何时延比：P1 较 Final 降低 3.57%，C 较 Final 降低 9.34%（C 较 P1 降低 5.99%）。这不是重定义目标，也不代替逐 case 结果。

## 版本、硬件与口径

- Final：冻结 `02f1e638ae76e350826473693e454e772f08ae0c`；P1：`5ad29ed890f336f7c19f900de1d418cfb0a3c5dd`，复用追加 Prefill projection session；C：`451b8163230848666687c3271347901b46c06b24`，大 batch routed-input quant 相同 scale 时量化一次并复制 top8。每列是实际测得的累计版本，增量按相邻列计算。
- 三阶段计时均为新跑的 Thor3 数据。Final/P1 的 ELF 原产于 Thor2，再传至 Thor3；C 的 ELF 原产于 Thor3。ELF 来源不同已记录，不把 Thor2 历史计时移作本表，也不与旧 Thor2 百分比相加。
- 上述提交是产品版本对应关系，不给运输来的旧 ELF 重贴“在该提交重建”的标签。P1 冻结 core 到 5ad 仅多两份维护文档，C 冻结 core 到 451b 也仅多两份维护文档；实际运行源与提交的桥接须以 source/protocol 审查为准。计时期间 live 源仍为 5ad，不能称全体 live 源为 451b。三版使用同一 `def3e8bd…` 两份 benchmark-only example overlay，完整 token 向量在计时后输出。
- GPU 锁定 1530 MHz；硬件可用最高频率 1575 MHz，因此本次不是最高频测试。CPU 2601 MHz，EMC 4266 MHz；实测频率与恢复设置已独立核对。
- 同一原位 Qwen3.6-35B-A3B NVFP4 检查点、NVFP4 MoE 激活、target 动态 FP8 KV（draft FP8 scale1）、greedy、固定 MTP depth3，不复用 context。每个 case 三版本使用完全相同输入、容量和策略；正式计时禁 autotune/fallback，复用有效且不变的 1056 份 recipe。
- Decode 输出 1024 token：首 token 计入 TTFT，其后 1023 个间隔计入 Decode。表内 Decode tok/s 为 `1023 / mean(decode_interval_seconds)`，不是逐轮 tok/s 的算术平均；含 Prefill tok/s 为 `1024 / mean(total_latency_seconds)`。追加 Prefill 报追加 4096 token 的时延，越低越好。
- 主表采用冻结的 18 项环境配置，包含 MTP 的三项 legacy `=1` 及 fixed depth3。后续[默认 MTP 策略检查](DEFAULT_MTP_PROFILE_THOR3.md)已独立通过：单元、清空变量/仅三个旧变量设 0 的六组严格检查，以及同 ELF 反向 BAAB 检查。原三列仍是 FLAGS1 协议，不改称完整无变量 15-case 矩阵；该默认交付修正不增加本轮性能收益。

## 6 组 LCG 长上下文 Decode

每组输出 1024 token。变化为吞吐百分比，正值表示更快。

| KV | Final | P1 | C | P1/Final | C/Final | C/P1 |
|---|---:|---:|---:|---:|---:|---:|
| 4K | 110.49 | 110.81 | 111.36 | +0.29% | +0.79% | +0.50% |
| 8K | 119.66 | 119.94 | 120.24 | +0.23% | +0.49% | +0.25% |
| 16K | 115.81 | 116.46 | 116.49 | +0.56% | +0.59% | +0.02% |
| 24K | 120.15 | 120.44 | 120.93 | +0.23% | +0.65% | +0.41% |
| 32K | 98.28 | 97.42 | 98.16 | -0.87% | -0.12% | +0.76% |
| 48K | 105.93 | 106.06 | 106.04 | +0.12% | +0.10% | -0.02% |

## 5 组追加 Prefill

新增输入均为 4096 token，benchmark 仅输出 1 token。计时排除初始 prefix 的准备，包含追加 Prefill 与首 token argmax；不是“追加后再输出 1024 token”的指标。绝对值为毫秒；降低百分比为正表示更快。

| KV | Final | P1 | C | P1/Final | C/Final | C/P1 |
|---|---:|---:|---:|---:|---:|---:|
| 4K | 524.68 | 504.36 | 469.50 | +3.87% | +10.52% | +6.91% |
| 8K | 553.03 | 533.02 | 497.42 | +3.62% | +10.05% | +6.68% |
| 16K | 605.08 | 585.52 | 547.83 | +3.23% | +9.46% | +6.44% |
| 32K | 694.18 | 667.96 | 633.05 | +3.78% | +8.81% | +5.23% |
| 48K | 786.64 | 760.38 | 724.82 | +3.34% | +7.86% | +4.68% |

## 4 组 eligible 输入 Decode

这是独立的 cyclic-real13 eligible 输入集，不能与 LCG 主表混合平均或相加，也不是完整真实问答质量评估。每组输出 1024 token。

| KV | Final | P1 | C | P1/Final | C/Final | C/P1 |
|---|---:|---:|---:|---:|---:|---:|
| 4K | 131.66 | 132.35 | 132.36 | +0.52% | +0.53% | +0.01% |
| 16K | 125.63 | 126.17 | 125.96 | +0.43% | +0.26% | -0.16% |
| 32K | 98.49 | 97.86 | 98.07 | -0.63% | -0.42% | +0.21% |
| 48K | 113.08 | 112.68 | 113.13 | -0.35% | +0.04% | +0.40% |

## 含 Prefill 的吞吐

对应上方 10 个 Decode case，单位 tok/s。TTFT、逐轮离散度及全部未舍入指标保存在随附 curated JSON；它们不替代主表 Decode 或追加 Prefill 指标。

| Case | Final | P1 | C | P1/Final | C/Final | C/P1 |
|---|---:|---:|---:|---:|---:|---:|
| decode-4k | 104.92 | 105.42 | 106.34 | +0.48% | +1.35% | +0.87% |
| decode-8k | 107.14 | 107.59 | 108.70 | +0.42% | +1.45% | +1.03% |
| decode-16k | 91.36 | 92.21 | 93.47 | +0.92% | +2.30% | +1.37% |
| decode-24k | 84.10 | 84.62 | 86.41 | +0.61% | +2.75% | +2.12% |
| decode-32k | 66.24 | 66.03 | 67.68 | -0.31% | +2.18% | +2.49% |
| decode-48k | 57.75 | 58.11 | 59.60 | +0.61% | +3.19% | +2.56% |
| eligible-decode-4k | 123.70 | 124.64 | 125.16 | +0.76% | +1.19% | +0.42% |
| eligible-decode-16k | 97.07 | 97.93 | 99.22 | +0.88% | +2.21% | +1.32% |
| eligible-decode-32k | 66.21 | 66.11 | 67.51 | -0.14% | +1.97% | +2.11% |
| eligible-decode-48k | 59.76 | 59.98 | 61.69 | +0.36% | +3.23% | +2.86% |

## 验证与边界

本轮比较是 Final02、P1、C 在 Thor3 的同机新矩阵。独立统计复核覆盖 15 case、45 runs、540 轮（其中 450 轮正式、90 轮预热），368820 个输出 token ID 的完整向量、FNV 与 MTP work 一致；recipe 前后恒定，11232 个全局频率样本、最终硬件与 cohort guard 通过，foreign GPU events 为 0。独立来源审查另核对了源码全 map、三版构建/overlay/产物元数据链与原始记录运输字节。大 ELF/native archive 未再下载到本地重 hash，其身份依据实际运行时的产物 hash、原始构建 receipt 和最终 cohort guard；不以源码等价推断二进制相同。VALID 表示协议、精度与运行有效，不表示有显著收益。每版是一块顺序测量，没有据此作统计显著性结论。

仅 append8K 的 P1/C 两项派生样本标准差存在约 `2.22e-16 ms` 的跨平台舍入差；原始时延、均值、输出与 MTP work 没有放宽。表中的标准差与吞吐评分分开记录，不将标准差当作置信区间。

P1 对 32K LCG Decode 的 −0.87% 小退必须保留；C 相对 P1 恢复约 +0.76%，累计仍较 Final 低约 0.12%。eligible 32K 的 C 也较 Final 低约 0.42%。其他小幅负值同样按表保留，不截零、不跨版本累加百分比。

两项优化提交已形成，本记录编写时尚未推送远程 Final；完整矩阵通过不表示两个 20% 目标达到。默认路径修正已另外验收并提交，计零新增收益；draft eligibility、GDN norm 与 FC2 资源候选不在此三列性能成果中。与最新 Final 合并后的来源与 recipe 桥接仍需独立检查。旧 TRT 数字仅可作历史参考，不纳入本次比例或目标判定。

优化机制与既有局部 profiler 分别见 [Prefill Session 记录](PREFILL_SESSION_THOR2.md) 和 [Routed Input Quant 记录](ROUTED_INPUT_QUANT_THOR3.md)。局部 NCU 的 scalar/单 ROI 结果不能代替本矩阵 MTP 关键路径占比，`f` 仍为 MISSING；本表报告实际模型指标，不从算子倍数外推模型收益。

## 复现与证据

固定三臂入口为 `devlocal/final-plus20/final15/three-stage/scripts/matrix_thor3_three.py`；实际执行须由统一资源调度器持有双 GPU 锁并复用冻结 plan/bindings。不得仅凭文档启动另一次 GPU 测试。

- 正式原始记录：`devlocal/final-plus20/results/thor3-final15-complete01/extracted/final15/three-stage/runs/thor3-three-stage-formal-01/`。
- 独立统计：`devlocal/final-plus20/results/thor3-final15-complete01/collation-final/` 中的 `report.json`、`table.csv`、`benchmark-tables.txt`、`independent-review.json`。v3 仅修正旧 partial 标签，未改原始数据、阈值或样本，旧报告保留。
- 独立来源与协议：`devlocal/final-plus20/results/thor3-final15-source-protocol-review-final02.json`，补齐 Final/P1 四份 build/combined JSON 的本地链；原 review01 保留。
- [curated JSON](evidence/final-plus20-thor3-benchmark.json)：全部未舍入指标、离散度、源/产物身份、固定环境、独立报告与运输归档的实际 SHA。未嵌入全部 recipe 内容。
