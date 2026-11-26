# Final 默认 MTP 优化的长上下文逐提交验收

> **2026-11-26 完整 benchmark：** 最新 Final 与最新 TRT 的 prefill、decode 12-case 结果见[完整测试](#thoru-full-current-20261126)。

> **2026-11-26 最新 Final 已重编译：** 本次实际最新源码与最新 TRT 的 8K/16K 对照见[最新 Final 对照](#thoru-latest-final-20261126)。

> **2026-11-26 锁屏复测：** 相同产物/缓存约定下的 8K/16K 重测，以及 MTP 算量、API 传输和旧程序参照分析见[新复测](#thoru-locked-repeat-20261126)。

> **同日缓存协议复测：** 8K/16K、1530MHz、精确 N−1 缓存对照及 MTP 工作量核对见[缓存复测](#thoru-cache-repro-20261125)。

> **2026-11-25 Thor-U 增补：** 原表是 Thor-2 的历史逐提交验收。本机最新 TRT Engine 与远程 Final 的新测量见[文末对照](#thoru-final-trt-20261125)，两机成绩不混用。

完整矩阵通过，两个提交在各自声明的受影响配置组上均有端到端收益，且无确认的 ≥2% 回退。汇总吞吐按同组 token 总数/时间总数计算；两行对应不同输入组和父版本，不能相加百分比。逐项结果和小幅负收益均保留在后文。

| 独立增量及汇总输入组 | Decode 提升 | 含prefill端到端提升 |
|---|---:|---:|
| 词表裁剪：4档裁剪生效长文字 | +2.20% | +1.50% |
| 位置表/复制Graph：全部10档Decode | +0.93% | +0.78% |

本轮以最新远程 Final **`429c649b`** 为基线，分别测量两个独立提交：

1. **`7cc20e50`**：按文字上下文自动裁剪 draft 词表，原生 W4A16 权重视图覆盖前 131072 列及末 512 列，主 verifier 保持完整词表；越出集合时自动使用完整 draft head。
2. **`595a46da`**：在第一项之上复用 dynamic-FP8 短行文字位置表，并预捕获两个物理方向的 verifier 状态复制 Graph，随状态缓冲区交换并先于引用内存销毁。

新增路径默认执行，没有新增优化开关或环境变量。两项收益按实际父版本比较；组合收益直接比较组合版与 Final，不相加百分比。

本轮验收期间，远程 Final 新增图像支持提交 `429c649b`，因此在旧 Final `1f605865` 的六档 Decode 完成后停止旧矩阵余项，保留已有记录，并在保留该更新的基础上重新编译、验收完整矩阵。合并时保留图像请求的完整 draft head；图像上传使用独立可写位置缓冲，避免覆盖文字只读位置表。新增的图文切换及图像 rowwise 状态检查均通过。

## 配置、范围与验收

Thor-2；GPU 固定 **1530 MHz**（最高可用 1575 MHz），CPU 2601 MHz、EMC 4266 MHz。
原位使用 `Qwen3.6-35B-A3B-NVFP4` checkpoint；checkpoint-native mixed weights、MoE activation `nvfp4`、dynamic-FP8 target KV、scale1 FP8 draft KV、greedy、固定 MTP depth3。FP8 KV 与 FP8 权重/激活投影是不同配置。

完整 15-case × 3 版本，每组 **2 次预热＋10 次正式测量**，共 **450 条计分、90 条预热**。逐 case 交替版本顺序；有疑似回退时另做反向顺序复测。Decode 精确输出 1024 token，关闭 EOS 提前结束；首 token 归 TTFT，后续 1023 个间隔归 Decode。含 prefill 吞吐为 1024/完整 generate 时间。追加 prefill 的初始 KV 建立在计时之外，测量新增 4096 输入及首个 argmax 的完成时间。均不含加载、分词或 HTTP。

原主表保留确定性的 LCG 输入；四档补充输入循环拼接同一组 13 题文字 token，以覆盖裁剪生效域。两类输入分别呈现。原主表输入含裁剪集合之外的 token，自动使用完整 draft head，因此不据此宣称裁剪加速。输入文件 hash、输入/输出 digest、逐轮数据和执行配置均收录在维护证据中。

正式计时使用冻结的 **953 份 recipe**，保留旧记录，仅为最新原生内核身份补齐新键；计时中禁止调优和缺记录回退。12644 次频率采样通过，权重身份和 recipe 内容前后不变。每个版本均使用独立、不可变的实测可执行文件和完整源码清单，实际 ELF/CUDA image 身份已核对。

## 原主表：长上下文输出 1K

Decode，单位 tok/s，越大越好；百分比为吞吐提升。

| 初始 KV | Final | ＋词表裁剪 | ＋位置表/复制 Graph | 第一提交较 Final | 第二提交较父版本 | 组合较 Final |
|---|---:|---:|---:|---:|---:|---:|
| 4K | 109.65 | 109.50 | 109.60 | -0.14% | +0.09% | -0.05% |
| 8K | 119.05 | 118.39 | 120.24 | -0.56% | +1.56% | +1.00% |
| 16K | 115.41 | 114.68 | 116.06 | -0.63% | +1.20% | +0.57% |
| 24K | 119.02 | 118.40 | 119.26 | -0.52% | +0.73% | +0.20% |
| 32K | 96.35 | 95.59 | 97.43 | -0.79% | +1.93% | +1.13% |
| 48K | 67.16 | 66.76 | 67.71 | -0.60% | +1.41% | +0.81% |

含 prefill 的完整生成吞吐，单位 tok/s，越大越好。

| 初始 KV | Final | ＋词表裁剪 | ＋位置表/复制 Graph | 第一提交较 Final | 第二提交较父版本 | 组合较 Final |
|---|---:|---:|---:|---:|---:|---:|
| 4K | 104.22 | 104.09 | 104.19 | -0.12% | +0.09% | -0.03% |
| 8K | 106.73 | 106.20 | 107.70 | -0.49% | +1.41% | +0.91% |
| 16K | 91.19 | 90.67 | 91.62 | -0.58% | +1.05% | +0.47% |
| 24K | 83.48 | 83.09 | 83.50 | -0.47% | +0.50% | +0.03% |
| 32K | 65.08 | 64.67 | 65.64 | -0.64% | +1.51% | +0.86% |
| 48K | 43.94 | 43.71 | 44.23 | -0.53% | +1.21% | +0.66% |

## 原主表：追加 prefill 4K

单位毫秒，越小越好；百分比为时延降低，因此负数表示变慢。

| 初始 KV | Final | ＋词表裁剪 | ＋位置表/复制 Graph | 第一提交较 Final | 第二提交较父版本 | 组合较 Final |
|---|---:|---:|---:|---:|---:|---:|
| 4K | 518.26 | 521.92 | 520.36 | -0.71% | +0.30% | -0.41% |
| 8K | 550.36 | 549.06 | 551.28 | +0.23% | -0.40% | -0.17% |
| 16K | 604.18 | 601.21 | 603.27 | +0.49% | -0.34% | +0.15% |
| 32K | 693.81 | 694.17 | 695.36 | -0.05% | -0.17% | -0.22% |
| 48K | 796.75 | 796.90 | 795.42 | -0.02% | +0.19% | +0.17% |

## 补充：裁剪生效的长文字输入输出 1K

Decode，单位 tok/s，越大越好。

| 初始 KV | Final | ＋词表裁剪 | ＋位置表/复制 Graph | 第一提交较 Final | 第二提交较父版本 | 组合较 Final |
|---|---:|---:|---:|---:|---:|---:|
| 4K | 125.88 | 129.66 | 129.85 | +3.00% | +0.14% | +3.15% |
| 16K | 121.90 | 125.29 | 125.26 | +2.78% | -0.02% | +2.76% |
| 32K | 97.16 | 97.47 | 97.75 | +0.32% | +0.29% | +0.61% |
| 48K | 120.69 | 124.61 | 126.51 | +3.25% | +1.52% | +4.82% |

含 prefill 的完整生成吞吐，单位 tok/s，越大越好。

| 初始 KV | Final | ＋词表裁剪 | ＋位置表/复制 Graph | 第一提交较 Final | 第二提交较父版本 | 组合较 Final |
|---|---:|---:|---:|---:|---:|---:|
| 4K | 118.70 | 122.04 | 122.18 | +2.81% | +0.12% | +2.93% |
| 16K | 94.88 | 97.14 | 96.92 | +2.39% | -0.23% | +2.15% |
| 32K | 65.42 | 65.56 | 65.70 | +0.21% | +0.21% | +0.43% |
| 48K | 61.74 | 62.65 | 63.34 | +1.48% | +1.10% | +2.60% |

独立 Nsight 采样实际观察到以下调用；采样性能不混入正式结果。生成途中出现集合外 token 后可自动退回完整 draft head，因此不宣称整个 1K 输出始终裁剪。

| 输入 | 分段投影 kernel 调用 | remapped argmax 调用 |
|---|---:|---:|
| 4K | 816 | 408 |
| 16K | 1224 | 612 |
| 32K | 264 | 132 |
| 48K | 1626 | 813 |

裁剪可以改变 draft 提案和接收率。下表列出本轮正式计时的实际工作量；位置表/复制 Graph 增量保持完整 MTP 工作量一致。接受比例按已提交接受 token/提案 token 计算。

| 输入 | Final 验证轮数 | 裁剪版轮数 | 组合版轮数 | Final 接受比例 | 裁剪版接受比例 | 组合版接受比例 |
|---|---:|---:|---:|---:|---:|---:|
| 4K | 310 | 309 | 309 | 76.77% | 77.13% | 77.13% |
| 16K | 305 | 310 | 310 | 78.58% | 76.67% | 76.67% |
| 32K | 363 | 366 | 366 | 60.61% | 59.84% | 59.84% |
| 48K | 270 | 271 | 271 | 93.09% | 92.62% | 92.62% |

## 正确性、回退检查与资源

45 项预检查全部通过；正式请求无轮间输出分歧，三个版本主输出 digest 一致。第二提交与父版本的完整 MTP 工作量相同。组合产物通过裁剪投影数值/只读/guard/Graph/事务状态/回退/重绑定检查，最低同输入余弦为 1.0；位置表边界、同位置图文切换、普通及图像 rowwise 物理状态缓冲/部分接受/快照协议检查通过。当前三组 benchmark 的 CUDA device images 分别完全相同，未混入仅属于某个版本的原生 kernel 代码改动。

预设 ≥2% 的 Decode/完整生成吞吐下降或追加 prefill 时延上升，需要与 Final 和直接父版本分别复测。本轮正式计时未触发预设的 ≥2% 回退复测阈值。

测试后已恢复初始频率策略，确认无 GPU 进程、HugePages Total/Free 均为 0，checkpoint 与冻结 recipe 未变。仅移除私有 benchmark 输入绑定覆盖层；机器保留验收通过的产品运行源码。此轮不建立新的 TRT 对照或两个 20% 目标。

## 复现与证据

- [逐轮维护证据](model-microbench/evidence/mtp-long-context.json)：版本、完整数据、产物/hash、recipe、输入身份、正确性、时钟与清理记录。
- [原始资产离线核验器](model-microbench/verify_mtp_long_context.py)：在保留机器资产的前提下重算指标、校验二进制和原始输出、检查频率和执行口径。
- [统一 benchmark 输入绑定补丁](model-microbench/mtp-benchmark-input-binding.patch)：仅为既有 example 增加显式 token 输入，在计时区间之外绑定；复现时使用 `git apply --unidiff-zero`，未作为产品开关合入。
- [历史真实 13 题的裁剪收益](APXINF4_MTP_DRAFT_SHORTLIST.md)与[位置/复制增量收益](APXINF4_MTP_VERIFY_TRANSPORT.md)：旧 Final、最多 512 输出且保留 EOS 的独立证据，不能当作最新 Final 的新测量，也不能累加百分比。

原始资产保留在 Thor-2 checkout 的 `devlocal/mtp-final-validation/long-current-final/`；模型与 13 题 fixture 沿用原位资产，维护证据不包含私有 token 内容。源码清单明确声明统一的 example 输入绑定覆盖层，产品运行代码精确来自表列提交。

```sh
python3 doc/qwen35-thoru/model-microbench/verify_mtp_long_context.py \
  devlocal/mtp-final-validation/long-current-final --run long-formal-01

# 使用已冻结的三个源码/ELF 阶段和 recipe；重新运行会占用 GPU。
python3 devlocal/mtp-final-validation/long-current-final/scripts/matrix.py \
  formal --name long-formal-repro-01
```

表中吞吐按实际 token 总数/总时间重算，追加 prefill 为测量均值。逐轮波动和筛查记录均保留，本轮未触发追加复测；结论限于本机、这些输入和冻结配置。

<a id="thoru-final-trt-20261125"></a>

## Thor-U：最新 TRT Engine 与 Final（2026-11-25）

本次固定远程 Final `438d5940b6a9e197a6dd085c6c6740204081490b`，在 Thor-U 本机重新编译和测量。TRT 使用本机最新可用交付包中的 `Qwen3.6-35B-A3B-NVFP4-FP16KV-MTP-703-14sm-64k-latest-20261112`，配套运行时 `qwen36-703-viskv_20261105`，不是旧的 128K FP8-KV Engine。Engine、embedding、tokenizer 和配置的九项交付 MD5 校验全部通过。

Thor-U：SM101、14 SM；CUDA 12.8.93，TensorRT 10.10.10.3。GPU 固定在本机最高可用频率 **1530 MHz**，实际 GPC 时钟持续采样；CPU performance（约 2.6 GHz），EMC 4266 MHz。按用户要求，本轮**不测试 48K**，没有使用 1206 MHz。

Final 使用原位 `Qwen3.6-35B-A3B-NVFP4` checkpoint、checkpoint-native mixed weights、NVFP4 MoE activation、dynamic-FP8 target KV、scale1 FP8 draft KV；TRT 的 target/draft KV 均为 FP16，内部算子精度按交付 Engine 固定。两者都是 greedy、MTP depth3 / verify4、CUDA Graph，关闭 EOS 提前结束。FP8 KV 不等于 FP8 权重/激活投影；本表是两套交付配置对照，不是同精度的算子 A/B；不透明 Engine 未做逐张量同权重校验。

12 个 case × 2 后端，每项 **2 次预热＋10 次计分**，共 **240 条计分、48 条受测预热请求**；前缀建立请求另计且不计分。Decode 类严格输出 1024 token（追加类输出 1 token），首 token 计入 TTFT，Decode 用其后 1023 个首末回调间隔。吞吐按正式 token 总数/正式时间总数计算，追加 prefill 按正式时延均值；均排除加载和 Graph 捕获。

本机产物仅对两个既有 benchmark 例程增加了计时外的显式 token 文件绑定，产品运行代码原样来自表列 Final；补丁及 ELF SHA256 在证据中。两边输入 token 已用各自 tokenizer 校验一致。原历史 LCG token 不能直接无损送入 TRT 的纯文字接口，因此本轮主表复用本机冻结的可往返长文字，另测本次 13 题循环语料；输入 SHA256 和题目 ID 列在证据中。它们不替代前面的 Thor-2 LCG 测量。

**冷请求边界：** 本机 1530 MHz 下，TRT 的 16K 整段冷 prefill 导致系统无响应，Linux 串口无响应，最终通过控制器断电/上电恢复；根因未确认。该次没有计分。24K/32K 未继续整段冷请求。对 16K/24K/32K（含补充文字的 16K/32K），改用 2K 分段在计时外建立至少 `N−2K` 的缓存前缀，再测完整 1024 输出的 Decode。每轮实际复用量均核验；这些行**只比较 Decode，不比较缓存请求的总吞吐与 Final 冷生成吞吐**。

**接口边界：** TRT 经公开运行时文字请求入口，包含其内部分词、调度，以及适用时的缓存恢复；Final 模型入口接收相同的预分词 IDs。追加 prefill 的两边都在计时外建立初始 KV，TRT 用 2K 分段建立前缀，计时请求必须精确复用 N 个 token、只追加 4096 个输入及输出首 token。表中时间不是纯 kernel 时间，接口开销不能归为算子收益。

### 共享长文字：Decode（tok/s）

| 输入 / Decode 起始 KV | TRT | Final | Final 较 TRT | TRT 路径 |
|---|---:|---:|---:|---|
| 4K | 101.44 | 95.89 | -5.47% | 冷请求 |
| 8K | 97.85 | 92.68 | -5.28% | 冷请求 |
| 16K | 94.33 | 89.04 | -5.61% | 缓存前缀 |
| 24K | 88.34 | 82.25 | -6.90% | 缓存前缀 |
| 32K | 83.57 | 77.62 | -7.11% | 缓存前缀 |

### 共享长文字：含 prefill 的冷生成吞吐（输出 tok/s）

“—”表示无可比的 TRT 冷请求结果；不使用缓存请求吞吐填补。

| 输入 | TRT 冷生成 | Final 冷生成 | Final 较 TRT |
|---|---:|---:|---:|
| 4K | 94.18 | 90.25 | -4.18% |
| 8K | 85.53 | 82.62 | -3.40% |
| 16K | — | 70.30 | — |
| 24K | — | 59.66 | — |
| 32K | — | 51.99 | — |

### 已有 KV 追加 4K（ms，越低越好）

百分比为 Final 相对 TRT 的时延降低；负数表示更慢。两套入口范围见上文。

| 初始 KV | TRT 请求 | Final 模型 | Final 时延降低 |
|---|---:|---:|---:|
| 4K | 846.51 | 702.49 | +17.01% |
| 8K | 893.22 | 745.67 | +16.52% |
| 16K | 975.18 | 805.04 | +17.45% |
| 32K | 1139.93 | 930.69 | +18.36% |

### 补充：本次 13 题循环长文字（输出 1K）

| 输入 / KV | TRT Decode | Final Decode | Final 较 TRT | TRT 冷生成 | Final 冷生成 |
|---|---:|---:|---:|---:|---:|
| 4K | 111.20 | 99.14 | -10.85% | 102.42 | 93.08 |
| 16K | 92.45 | 78.85 | -14.71% | — | 63.79 |
| 32K | 100.75 | 76.74 | -23.84% | — | 51.57 |

16K/32K 的 TRT 为缓存前缀 Decode；两档冷生成留空。单位均为输出 tok/s。

### 运行检查、限制与复现

所有纳入成绩的请求都完成了输入长度、输出数量、初始 KV / 实际复用量和时钟检查；Final 还通过例程内的逐轮输出一致性检查。TRT 的 greedy 输出在重新 prefill 时存在轮间差异，逐 case 的正式输出种类数及波动范围保留在证据中；不同 KV 精度及输出/MTP 工作量不能作逐 token 质量等价或单一 kernel 因果结论。未新增自然任务质量验收。

Engine 反序列化时 TensorRT 提示 plan 设备型号差异警告；本表保留该交付包在本机的实际运行结果，不将其表述为重新原生构建 Engine 的验收。

准备阶段曾发生一次整模型在线调优中断，之后禁止该路径，缺项通过独立算子准备，新准备项的常量数值、guard 和只读检查通过。Final 正式计时严格使用冻结 recipe，不在线调优、无缺项回退；权重/Engine/运行时身份、相关 ELF 和 recipe 内容在三段有效采集之间一致。16K 冷请求挂起前已完成的五组完整 2＋10 数据经恢复后核对并复用，没有重跑或挑选更好的轮次。中断请求不计分。恢复后十二组完成，切换到 TRT 32K 追加前的内存预留检查安全退出；整理空闲内存后仅补余下七组，先前成绩均未重跑。

测试后恢复原 GPU 策略与 7380 页 HugeTLB，确认无遗留推理进程。仅维护 benchmark 证据与文档；没有把本次两套配置的差值当作新代码优化收益，也不与前面的 Thor-2 逐提交百分比累加。

- [逐轮证据及身份](model-microbench/evidence/thoru-final-trt-20261125.json)：240 条正式测量、预热、实际模式、输入/产物/recipe 身份、时钟与恢复记录。
- [离线核验器](model-microbench/verify_thoru_final_trt.py)：重新计算表格，可校验保留的原始文件与 ELF。

原始资产保留在 Thor-U 的 `/opt/data/devlocal/final-prefill-range-thoru/source/devlocal/thoru-final-trt/checkout/devlocal/thoru-final-trt/`。`runs/formal-02/` 复用五组完整记录，`runs/formal-03/` 复用十二组，`runs/formal-04/` 补齐最后七组。原始文件、失败尝试和控制器均保留，私有输入内容不写入维护证据。

```sh
python3 doc/qwen35-thoru/model-microbench/verify_thoru_final_trt.py \
  doc/qwen35-thoru/model-microbench/evidence/thoru-final-trt-20261125.json

# 在保留原始资产的 Thor-U 上追加 --raw-root，核对日志、输入、recipe 和 ELF。
```

<a id="thoru-cache-repro-20261125"></a>

## Thor-U：1530MHz、旧 N−1 缓存协议复测（2026-11-25）

本节专门核对用户旧表的 8K/16K Decode。GPU min=max **1530MHz**，同一份冻结文字/token IDs，按原来的 2048-token 分段边界建立前缀至 `N−1`；每个受测请求精确复用 `N−1`，只计算 1 个新输入 token，严格输出 1024 token。三套后端均 greedy、固定 MTP depth3/verify4、CUDA Graph、禁止 EOS 提前结束。**不测试 48K，不使用 1206MHz。**

统一使用原协议的父进程时钟：收到首个 `T+u32le` token 帧至最后一个帧的 1023 个间隔；前缀准备和首 token 在 Decode 计时外。此前一节使用的冷请求/2K 尾段协议及数值保留，不能混入本节汇总。

Final 使用远程 **`9f7aecd204073eafd25bfdb99c1c4ab586f95fad`** 的实际新构建（已保留新增视觉 GEMM 提交），动态 FP8 target KV、scale1 FP8 draft KV、NVFP4 MoE activation。仅增加计时外的 benchmark 缓存准备入口及协议适配例程，正式生成走原模型 API；8-token 直接继续/恢复检查点的一致性检查通过。构建后撤销缓存准备入口的源码覆盖层，补丁和实际 ELF 身份在证据中。

旧 TRT 指本机保留的 `/opt/data/thorU703_vlm_128k` 128K FP8 target/draft KV 交付；最新 TRT 指 `Qwen3.6-35B-A3B-NVFP4-FP16KV-MTP-703-14sm-64k-latest-20261112` 及 `qwen36-703-viskv_20261105` 运行时，target/draft KV 均为 FP16。这两套当前资产有明确身份，但尚未找到用户旧表 TRT 129.6/119.4 对应的完整收据，不能假定任一套与旧表产物逐字节相同。

每个前缀状态 **2 次预热＋10 次正式测量**。最新 TRT 每档独立进程建立两次前缀，分别保留 10 次成绩并按 token 总数/时间总数汇总；Final 和旧 TRT 每档各 10 次。共 **80 条正式测量、16 条纳入数据的预热、16 条独立 MTP 计数诊断**。诊断启用内部 profiling，只用于工作量核对，不进入性能汇总；正式计时关闭 profiling、使用冻结 recipe，无在线调优/缺项回退。

### 同缓存协议 Decode（tok/s）

百分比统一为 `Final / 最新TRT − 1`；最新 TRT 汇总两个独立前缀状态，未挑选较快状态。

| 初始 KV | 旧 TRT FP8 | 最新 TRT FP16（20 次） | Final（10 次） | Final 较最新 TRT |
|---|---:|---:|---:|---:|
| 8K | 83.19 | 97.64 | 93.44 | -4.30% |
| 16K | 82.62 | 90.36 | 86.83 | -3.91% |

### 独立前缀状态与 MTP 工作量

最新 TRT 的计数诊断只在完整输出 SHA 与正式请求匹配时关联；下表轮数来自这些匹配诊断。Final 轮数直接来自正式请求计数。新运行时仅公开验证轮数与生成 token 数，不虚填其未公开的提议/接受计数。Native 的生成计数为首 token 后的 1023，TRT 原生生成计数含首 token 为 1024；轮数按各自原生接口报告，不能视作逐指令工作量完全等价。

| KV | 最新 TRT 前缀 1 / 2 吞吐 | 最新 TRT 前缀 1 / 2 的 MTP 轮数 | Final MTP 轮数 | 最新 TRT 正式输出种类 / 20 | Final 输出种类 / 10 |
|---|---:|---:|---:|---:|---:|
| 8K | 99.36 / 95.97 | 369 / 382 | 369 | 2 | 1 |
| 16K | 89.49 / 91.25 | 384 / 377 | 385 | 2 | 1 |

两次独立前缀之间，最新 TRT 的完整输出均发生变化；8K 的 MTP 轮数由 369 增至 382，吞吐由第一状态的约 99.36 降至第二状态的约 95.97 tok/s，变化与轮数增加相符。不能只用某一次前缀状态代表该 Engine 的唯一吞吐。

跨后端完整输出也不相同：两次 TRT 前缀状态均在 8K 的第 2 个输出 token、16K 的第 9 个输出 token 与 Final 首次分叉。分叉位置和完整 SHA 已核对；两套精度/数值路径的质量优劣尚未验收。

旧 TRT 的正式输出仍有轮间变化：8K 为 10/10 种；16K 为 10/10 种。其诊断单独记录；与正式输出不匹配的诊断不冒充正式请求的 MTP 工作量。

### 与用户旧表核对

本次三后端完整输入及各段前缀均通过 tokenizer/token ID 对齐；它们也与已找到的旧 ApxInf 输入哈希一致。旧表 TRT 两个数字对应的输入哈希、Engine/运行时和逐轮输出尚缺直接证据，不能把这一事实扩大为“旧 TRT 的输入也已证明相同”。

| KV | 用户旧表 TRT（1530MHz） | 本次最新 TRT 同缓存汇总 | 数值差异 |
|---|---:|---:|---:|
| 8K | 129.6 | 97.64 | -24.66% |
| 16K | 119.4 | 90.36 | -24.32% |

按旧缓存协议测量后的实际数值如上；仅更换缓存协议不足以解释与旧表的全部差异。由于旧 TRT 原始产物/输出记录未对齐，这个跨记录数值差不能直接归为 Engine 版本性能回退，也不能据此宣称 Final 已追回旧表差距。

已定位的旧 ApxInf 3.0 为 `68b12f23`，2 次预热＋5 次计分的吞吐中位数为 8K 92.43、16K 90.23 tok/s，与用户表吻合。其实际工作量与本次 Final 对比如下；同输出长度不保证同 MTP 轮数。

| KV | 旧 ApxInf MTP 轮数 | 本次 Final MTP 轮数 | 完整输出 SHA 是否相同 |
|---|---:|---:|---|
| 8K | 363 | 369 | 不同 |
| 16K | 358 | 385 | 不同 |

用户旧表百分比以 ApxInf 为分母，本轮以 TRT 为分母。旧表 4K 的 −22.11% 按本轮公式为 −18.10%；分母变化只解释百分比，不解释 tok/s。历史中位数与本轮按总时间汇总的吞吐也分别标注，不相互替换。

### 证据与范围

本表仍是不同 KV 精度/缩放方案及不透明 TRT 权重的交付配置比较，未新增逐 token 质量等价验收，不把差值称为纯 kernel 收益。TRT 的跨设备 plan 警告继续保留。计时适配器曾在准备/首个预热阶段修正统计字段兼容及输入总量检查；失败请求另存且不计分，完整有效组沿用。新 Final 的缺项只通过独立算子准备，数值、guard、只读检查通过。

结束恢复原 GPU 策略与 7380 页 HugeTLB，确认无遗留推理进程，各有效采集阶段的检查点/Engine/recipe 身份前后不变；Final 切换新提交及补齐其 recipe 均在该版正式计时前完成。原始目录：`/opt/data/devlocal/final-prefill-range-thoru/source/devlocal/thoru-final-trt/checkout/devlocal/thoru-final-trt/cache-repro/`。

- [逐轮计时、输出身份、MTP 诊断及两次前缀证据](model-microbench/evidence/thoru-cache-repro-20261125.json)。
- [离线核验器](model-microbench/verify_thoru_cache_repro.py)。

```sh
python3 doc/qwen35-thoru/model-microbench/verify_thoru_cache_repro.py \
  doc/qwen35-thoru/model-microbench/evidence/thoru-cache-repro-20261125.json
```

<a id="thoru-locked-repeat-20261126"></a>

## Thor-U：锁屏状态复测与 TRT 差异分析（2026-11-26）

本轮按用户要求重测同一约定，目的是检查前一轮数据的可重复性。固定上一轮实际产物：Final **`9f7aecd2`**、ELF `b7c0ce53…`、同一份 recipe、同一检查点和两套 TRT Engine/运行时，未重新编译或调优。为隔离变量，没有把复测程序切换成随后新增的 Final 代码；本节不代表那些新提交的性能验收。

执行了 `loginctl lock-sessions`。设备实际处于 GDM 登录认证界面，没有已解锁的用户图形桌面；该 greeter 的 `LockedHint=no`，不能把它写成用户桌面的 `LockedHint=yes`。随后将 DPMS 显示输出关闭，**每个 case 前后和约每 5 秒检查一次 Monitor Off**，所有纳入数据的显示状态采样通过。保留 GUI 服务，没有强行登出用户或终止应用。

GPU min=max **1530MHz**，实际 GPC 时钟采样；输入仍为相同 8K/16K IDs，2K 分段准备至 N−1，每次精确复用 N−1、只计算 1 个新输入 token，greedy MTP depth3/verify4、Graph、忽略 EOS，输出 1024 token。Decode 仍按共同父进程首末 token 帧之间的 **1023 个间隔**，不含前缀准备和首 token。**不测 48K，不使用 1206MHz。**

Final/旧 TRT 每档各 2 预热＋10 计分；最新 TRT 每档独立建立两次前缀，每次 2＋10，共 **80 条正式测量、16 条预热、18 条独立 MTP 计数诊断**。另有旧 ApxInf 8K 参照和 API 传输配对诊断，均单列，不混入这 80 条结果。MTP 诊断启用 profiling；正式计时关闭 profiling 且冻结 recipe。

### 同约定复测（tok/s）

最新 TRT 每档仍按两个前缀状态的 token 总数/时间总数汇总；Final 取本轮 10 次。百分比为 Final / 本轮最新 TRT − 1。

| KV | 上轮最新 TRT | 本轮最新 TRT | 上轮 Final | 本轮 Final | 本轮 Final 较 TRT |
|---|---:|---:|---:|---:|---:|
| 8K | 97.64 | 98.32 | 93.44 | 94.14 | -4.26% |
| 16K | 90.36 | 95.31 | 86.83 | 86.70 | -9.03% |

旧 FP8-KV TRT 本轮为 8K 81.78、16K 84.99 tok/s；它与最新 FP16-KV 交付分开标注，不假定它就是用户旧表中的同一产物。

### 已确认：重建前缀后，输出与 MTP 工作量变化

每个最新 TRT 前缀内的 10 次正式输出一致；独立重建前缀后的完整输出变化，匹配这些正式输出的 MTP 诊断轮数也变化。表中每轮耗时是首末帧墙钟区间除以全请求的原生 MTP 轮数（后者含首 token），仅作关联归一化对照，包含执行供给，**不是纯 kernel 时间**。当前未定位首次数值分歧或具体缓存错误，不能据此直接断言某个缓存实现有缺陷。

| KV / 轮次 | 前缀 1：tok/s / MTP 轮数 / ms每轮 | 前缀 2：tok/s / MTP 轮数 / ms每轮 |
|---|---:|---:|
| 8K / 上轮 | 99.36 / 369 / 27.902 | 95.97 / 382 / 27.904 |
| 8K / 本轮 | 97.55 / 376 / 27.891 | 99.11 / 369 / 27.973 |
| 16K / 上轮 | 89.49 / 384 / 29.769 | 91.25 / 377 / 29.738 |
| 16K / 本轮 | 97.87 / 352 / 29.694 | 92.88 / 372 / 29.608 |

例如 16K 从上一轮首个前缀的 384 轮、89.49 tok/s，变为本轮首个前缀的 352 轮、约 97.87 tok/s；每轮约 29.7ms，吞吐变化与少做 32 轮验证吻合。它不能被表述为锁屏带来了约 9% 的算子加速。Final 本轮与上轮的完整输出及轮数核对结果保存在证据中。

**修正上一轮结论的适用范围：** −4.30%/−3.91% 只是当时各两个前缀状态的汇总，不能当成不随前缀准备/输出变化的固定性能差距。每个前缀的 10 次计时反映该状态内的执行波动，不能把它们当作 10 次独立的前缀/输出样本。不同后端输出也不完全相同，仍未新增质量等价验收。

### 独立检查：传输方式和旧程序参照

8K 同一已准备前缀中，直接 API 与逐 token 流式 API 做了 3 组配对，其中 **3 组完整输出 SHA 相同**。匹配组的直接 `handle_request` 均值 **10209.91ms**，流式 **10212.72ms**，流式相对差值 **+0.03%**；父进程总区间相对流式子进程 API 区间的额外时间约 **0.77ms**。这些均为 API 墙钟时间，含首 token/结束处理，不能冒充主表 Decode 吞吐。

旧 ApxInf 3.0 使用已归档的 `68b12f23` 程序，交付 SHA256 校验通过，原 3 个 MTP 开关和 1579 份 recipe 沿用。为匹配旧六档测试到达 8K 前的进程历史，先执行旧 4K 前置段（不计入成绩），再测 8K。当前 10 次中位数 **92.49 tok/s**，旧记录为 **92.43**，变化 **+0.06%**；输出与旧记录相同，本轮 MTP 轮数 **363**，旧记录 **363**。这项参照单列，未用新 Final 替代旧程序。

### 仍未证明的部分

本次关闭显示输出后的同约定结果仍未复现用户旧表 TRT 8K 129.6、16K 119.4 tok/s。已对本地和远程历史记录做有范围的检索，找到的近似数字属于其他 ApxInf case/设备或现有文档副本，没有定位这两项 TRT 的原始命令、产物和输出/轮数记录；这不能用来断言旧数字错误。

| KV | 用户旧 TRT tok/s | 按本轮每轮成本推算所需 MTP 轮数 | 本轮实际观察轮数 |
|---|---:|---:|---:|
| 8K | 129.6 | 约 283 | 376 / 369 |
| 16K | 119.4 | 约 289 | 352 / 372 |

上表的所需轮数是“假设旧记录每轮成本与本轮相同”的推算，**不是对旧 TRT 接受率的实测**。它说明仅记录输入长度和 1K 输出，仍不足以冻结 MTP 算量。旧记录是否用了不同输入/前缀历史、MTP 配置、真实输出数量/EOS 规则、Engine 或计时边界，仍需原始记录核对，不能单凭 tok/s 差值认定 Engine 回退约 25% 或 Final 已追回旧表差距。

结束恢复原 GPU 策略和 7380 页 HugeTLB，确认无遗留推理进程与自有 GPU 锁；显示继续保持关闭。每个阶段产物/recipe/源码身份前后不变。原始目录沿用 `.../devlocal/thoru-final-trt/cache-repro/`，本轮记录为 `runs/formal-06-locked-primary/`、`formal-07-locked-repeat/`、`diagnostic-08-transport/`、`control-09-legacy-apx/`。

- [锁屏复测、MTP、API 配对与旧程序参照证据](model-microbench/evidence/thoru-locked-repeat-20261126.json)。
- [独立离线核验器](model-microbench/verify_thoru_locked_repeat.py)。

```sh
python3 doc/qwen35-thoru/model-microbench/verify_thoru_locked_repeat.py \
  doc/qwen35-thoru/model-microbench/evidence/thoru-locked-repeat-20261126.json
```

<a id="thoru-latest-final-20261126"></a>

## Thor-U：拉取最新 Final 重新编译对照（2026-11-26）

本轮按用户新指令实际拉取、快进到测试开始时远程 Final 最新 **`ae909d2b0f992fbd3a2b33baa88b60057e96c479`**，在 Thor-U 重新编译；该提交包含最新代码 `b3292251`。实际新 ELF SHA256 为 **`87916cc1e86f2c659373ab50e99d4def2f83fe32efa939011f1272723c44991c`**，CUDA 静态归档为 `7731dd4e380af81b9871597917bdac54a5fa212123e5d677a1e4ed11820727b1`。上节固定 `9f7aecd2` 的复测不代表本轮源码。沿用同一 benchmark-only 前缀准备适配器，完整补丁、worker 源码和构建命令记入证据；适配器只添加测试入口，构建结束恢复原源码。

最新可用 TRT 交付仍为 `Qwen3.6-35B-A3B-NVFP4-FP16KV-MTP-703-14sm-64k-latest-20261112`，运行时 `qwen36-703-viskv_20261105`，target/draft KV 均为 FP16。本轮不运行旧 FP8-KV TRT。Final 沿用同一检查点、NVFP4 MoE 激活、动态 FP8 target KV / scale=1 FP8 draft KV；后端配置不同，不能称为等精度算子 A/B 或质量等价验收。

执行 `loginctl lock-sessions`，设备为 GDM 登录认证界面，无已解锁用户图形桌面；保持 DPMS Monitor Off。每个 case 前后及约每 5 秒验证认证/锁定状态与显示关闭，保留 GUI 服务。GPU min=max **1530MHz**，运行中检查实际 GPC 时钟、内存和独占推理进程。

与上一轮相同：8K/16K 输入 IDs；2K 分段准备至 N−1；每次 reused=N−1、computed=1；greedy MTP depth3/verify4、Graph、忽略 EOS；输出 1024 token，首 token 归 TTFT，其后 **1023 个首末 token 帧间隔**归 Decode，使用同一父进程墙钟，含 token 帧供给。前缀准备不计入 Decode。**不测 48K，不用 1206MHz。**

Final 每档 2 预热＋10 正式；最新 TRT 每档独立建立两个前缀，每个前缀 2＋10，再另做 3 条 profiling 请求核对 MTP 计数：共 **60 条正式、12 条预热、12 条独立计数诊断**。正式 profiling/autotune 关闭，强制已存在 recipe 且前后不变；新 CUDA 身份所需 recipe 在独立进程预先准备并通过常量参考、边界和只读检查。与上节旧 Final 的变化包含新二进制和相应 recipe，未把它归因为某个单独提交的收益。

### 最新产物对照（tok/s）

TRT 汇总两个前缀各 10 次的 token 总数/时间总数；Final 汇总本档 10 次。差值为 Final / TRT − 1。不同前缀的输出和 MTP 工作量单列，成绩是本次输入轨迹下的配置对比。

| 初始 KV | 最新 Final | 最新 TRT | Final 较 TRT | 上节旧 Final | Final 相对上节变化 |
|---|---:|---:|---:|---:|---:|
| 8K | 97.96 | 97.54 | +0.43% | 94.14 | +4.06% |
| 16K | 93.74 | 93.54 | +0.22% | 86.70 | +8.12% |

本轮两个 TRT 前缀汇总下，两档差距均小于 1%，可描述为本次接近持平；TRT 独立前缀间的吞吐差异大于该差值，不能宣称 Final 已稳定领先。

### 输出与 MTP 工作量

| KV | Final：轮数 / 不同正式输出数 | TRT 前缀 1：tok/s / 同输出诊断轮数 | TRT 前缀 2：tok/s / 同输出诊断轮数 | Final 与上节旧产物输出一致 |
|---|---:|---:|---:|---|
| 8K | 369 / 1 | 95.03 / 386 | 100.20 / 368 | 是 |
| 16K | 384 / 1 | 95.15 / 362 | 91.97 / 375 | 是 |

TRT 的计数来自独立 profiling 请求，只有完整输出 SHA 匹配的正式请求才与该轮数关联；逐条匹配关系在证据中。每个前缀的 10 次计时只是同一状态内重复，不能当成 10 个独立前缀样本。不同后端或不同前缀输出发生变化时，MTP 接受工作量也会变，因此上表不能单独证明底层算子加速比例；此前固定约 4% 的笼统判断继续撤回。

用户历史 TRT 8K 129.6 / 16K 119.4 的原始产物、命令和输出/MTP 记录仍未定位。本次新 Final 的对照也不能替代对那两项历史成绩的复现。

所有计分 case 完成后恢复原 GPU 策略及 7380 页 HugeTLB，无遗留推理进程，自有 GPU 锁释放；显示保持关闭。构建/资源准备不计作性能成绩，历史表格保留。

- [最新 Final 重编译与本轮原始测量摘要](model-microbench/evidence/thoru-current-final-20261126.json)。
- [独立离线核验器](model-microbench/verify_thoru_current_final.py)。

```sh
python3 doc/qwen35-thoru/model-microbench/verify_thoru_current_final.py \
  doc/qwen35-thoru/model-microbench/evidence/thoru-current-final-20261126.json
```

<a id="thoru-full-current-20261126"></a>

## Thor-U：最新 Final 完整 prefill 与 decode benchmark（2026-11-26）

按完整的既有 Thor-U 12-case 约定重测：5 档共享长文字 Decode、4 档已有 KV 追加 4K 的 prefill、3 档补充文本 Decode；Final 和最新 TRT 各测一遍，每项 2 次预热、10 次正式测量，**共 240 次正式、48 次预热**。未测 48K，也未使用 1206MHz。后续核对 GitHub `Final` 至 `d1d48ba1`：自本轮实测源码以来只有文档提交，模型代码及构建依赖未变，因此沿用本轮计分结果。

Final 使用测试开始时最新源码 `ae909d2b0f992fbd3a2b33baa88b60057e96c479`，重新编译两个原有 benchmark 程序；Decode ELF SHA256 `de7ad8d2193db4bd1855f019e7b9cb2123ff1ea33241a76eca70022e0885c46c`，追加 prefill ELF SHA256 `fafd22f95cd9e2b3bcc23e91b5732380761b5e81e5c71f3967267015a57fb32c`。计时外沿用已公开的显式 token 输入绑定补丁。TRT 使用最新可用交付 `Qwen3.6-35B-A3B-NVFP4-FP16KV-MTP-703-14sm-64k-latest-20261112` 与运行时 `qwen36-703-viskv_20261105`。

GPU 固定 1530MHz，实际时钟、内存和 GPU 占用持续采样；GDM 登录认证界面执行锁会话并关闭显示输出，每个 case 前后及运行中约每 5 秒检查 Monitor Off。正式测量关闭调优和 profiling，强制使用冻结 recipe。新 CUDA 身份缺少的 66 个 prefill 配置在独立进程预先准备并通过常量参考、边界和只读检查，不计入成绩。

Final 使用动态 FP8 target KV / scale1 FP8 draft KV，TRT target/draft 为 FP16 KV；这是两套交付配置的对照，未做等精度或质量等价验收。Final 模型入口接收预分词 IDs，TRT 经公开文字接口，含分词、调度和适用时的缓存恢复。

### 共享长文字：Decode（tok/s）

严格生成 1024 token，首 token 归 TTFT，后续 1023 个首末回调间隔归 Decode；按正式 token 总数/总时间汇总。

| 输入 / Decode 起始 KV | TRT | Final | Final 较 TRT | TRT 路径 |
|---|---:|---:|---:|---|
| 4K | 102.85 | 100.94 | -1.86% | 冷请求 |
| 8K | 101.28 | 96.63 | -4.59% | 冷请求 |
| 16K | 93.28 | 96.56 | +3.52% | 缓存前缀 |
| 24K | 89.22 | 89.58 | +0.40% | 缓存前缀 |
| 32K | 82.61 | 87.29 | +5.66% | 缓存前缀 |

<a id="thoru-full-current-prefill"></a>

### 冷请求：prefill 至首 token 时延（TTFT，ms，越低越好）

这是冷请求从进入各自接口到收到首 token 的均值，包含 prefill、首 token 计算及接口开销，**不是纯 prefill kernel 时间**。仅 4K/8K 有 TRT 冷请求实测；长输入 TRT 的缓存请求 TTFT 不填入本表。百分比为 Final 相对 TRT 的时延降低。

| 输入 | TRT 冷请求 | Final 冷请求 | Final 时延降低 |
|---|---:|---:|---:|
| 4K | 786.76 | 541.78 | +31.14% |
| 8K | 1520.83 | 1186.21 | +22.00% |
| 16K | — | 2754.32 | — |
| 24K | — | 4257.03 | — |
| 32K | — | 5898.63 | — |

### 共享长文字：含 prefill 的冷生成吞吐（输出 tok/s）

“—”表示无可比的 TRT 冷请求结果；不使用缓存请求吞吐填补。

| 输入 | TRT 冷生成 | Final 冷生成 | Final 较 TRT |
|---|---:|---:|---:|
| 4K | 95.40 | 95.91 | +0.54% |
| 8K | 88.10 | 86.97 | -1.28% |
| 16K | — | 76.71 | — |
| 24K | — | 65.32 | — |
| 32K | — | 58.12 | — |

### 已有 KV 追加 4K（ms，越低越好）

每轮在计时外建立初始 KV，请求复用该 KV、追加 4096 输入并生成首 token。表中是 10 次正式请求的平均时延，包含各自接口开销；百分比为 Final 相对 TRT 的时延降低。两套入口范围见上文。

| 初始 KV | TRT 请求 | Final 模型 | Final 时延降低 |
|---|---:|---:|---:|
| 4K | 846.26 | 566.31 | +33.08% |
| 8K | 891.96 | 608.11 | +31.82% |
| 16K | 974.96 | 668.46 | +31.44% |
| 32K | 1140.47 | 796.08 | +30.20% |

### 补充：本次 13 题循环长文字（输出 1K）

| 输入 / KV | TRT Decode | Final Decode | Final 较 TRT | TRT 冷生成 | Final 冷生成 |
|---|---:|---:|---:|---:|---:|
| 4K | 111.48 | 104.54 | -6.23% | 102.63 | 99.12 |
| 16K | 102.39 | 83.93 | -18.03% | — | 68.53 |
| 32K | 94.17 | 85.20 | -9.52% | — | 57.21 |

16K/32K 的补充文本 TRT 为缓存前缀 Decode；冷生成留空。TRT greedy 输出在重新 prefill 时存在轮间差异，各档正式输出种类数和波动范围保留在证据中，不能据此作单算子性能或质量等价结论。

### 运行检查与复现

16K/24K/32K 的 TRT 冷 prefill 曾使 Thor-U 无响应，本轮长输入 TRT 仅用 2K 分段在计时外建立前缀，并核验实际复用量；**不将缓存请求总吞吐填为冷生成**。追加 prefill 两边均在计时外建立初始 KV，计时请求追加 4096 输入并生成首 token；表中毫秒值含各自接口开销，不是纯 kernel 时间。TRT Engine 在本机仍提示跨设备型号 plan 警告。

逐 case 核验输入长度、输出数量、缓存复用、1530MHz、普通内存下限和屏幕关闭；结束恢复原 GPU 策略及 7380 页 HugeTLB，无遗留推理进程，自有 GPU 锁释放，显示保持关闭。原始记录存于 Thor-U `devlocal/thoru-final-trt/runs/formal-full-current-20261126-r2/`，准备阶段缺项失败的请求没有计分。旧表完整保留。

- [完整 benchmark 证据](model-microbench/evidence/thoru-full-current-20261126.json)。
- [离线核验器](model-microbench/verify_thoru_full_current.py)。

```sh
python3 doc/qwen35-thoru/model-microbench/verify_thoru_full_current.py \
  doc/qwen35-thoru/model-microbench/evidence/thoru-full-current-20261126.json
```
