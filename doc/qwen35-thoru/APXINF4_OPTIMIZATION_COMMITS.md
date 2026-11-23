# 已有模型端到端收益的改动与commit

> 本文保留11月23日实验版本的逐commit收益与原始身份。当前Final的默认落地提交、开关移除说明及11月24日默认路径110轮复测，见[Final默认性能路径交付](APXINF4_FINAL_DEFAULTS.md)。以下实验开关/分支只用于解释历史证据，当前三组优化无需设置这些开关。

本轮11组case、四版本的440轮正式计分，以及两组反向A/B的40轮计分均已核验。六组生成场景有端到端收益；追加prefill存在回退，尚未完成全场景性能验收。

## 统一口径

模型为Qwen3.6-35B-A3B-NVFP4，以Thor-2自身Final `9457b9bc`为基线；GPU按约定固定1530MHz（当前最高可用1575MHz），CPU2601MHz、EMC4266MHz固定最高可用档；NVFP4、动态FP8 KV、greedy、固定MTP depth3、Graph/full GDN。每组初始KV为4/8/16/24/32/48Ki，输出1024 token，每版本2轮预热＋10轮计分。

下面“端到端”主要指**含prompt prefill的模型生成吞吐**（1024输出/模型总时间），排除权重加载、分词和HTTP。Decode单独统计首token之后1023个间隔。每个相同case的版本间输出与完整MTP工作量相同，736份recipe冻结，无在线调优/缺失回退。

## 已确认的三组优化（四个优化点）

| 优化 | 交付commit | 相对上一阶段：含prefill吞吐 | 相对上一阶段：Decode |
|---|---|---:|---:|
| MoE warp-prefix | `31f6fe53` | +0.18%～+1.45% | +0.21%～+1.60% |
| FFN routed/shared并行 | `bc55919f ＋ ac57887e` | +0.31%～+2.34% | +0.86%～+2.60% |
| GDN状态搬运与快照组合 | `f670ad32` | +1.09%～+3.44% | +1.90%～+3.56% |

这些是六组case的实测范围，不是把不同版本的百分比累加。GDN两个点作为一组验收，未把组合收益分别记到两项上。

### MoE warp-prefix

仅在真实M1/M4小批量路径，用warp ballot/popcount/前缀扫描并行生成expert offsets、tile和scatter元数据，替换原串行元数据工作。量化和GEMM数学路径沿用Final。

此前同一算法的提交为`f7cbe12e`；移植/整理版本不重复计作新优化。

### FFN routed/shared并行

在同一个ExecutionSession和CUDA Graph中，从同一前沿捕获routed experts和shared expert两条独立分支，汇合后再combine。M1/M4启用，M2/M3保持串行；不增加额外模型精度策略。

此前同一算法的提交为`de86bb89`；移植/整理版本不重复计作新优化。

### GDN状态搬运与快照组合

两项合并验收：①验证核只读committed recurrent state并直接写shadow，省去每个GDN层每次验证前2MiB recurrent初始化复制；②只保留必要的前T−1个recurrent checkpoints，最终状态继续写live state，完整接受使用live state、部分接受使用保留的前缀。conv复制和快照保留。

此前同一算法的提交为`8b29ce6c`；移植/整理版本不重复计作新优化。

## 累计相对Final的收益

- Thor-U warp-prefix＋FFN（ApxInf4.0）：含prefill吞吐 **+0.88%～+2.67%**，Decode **+1.99%～+3.19%**。
- 再加GDN组合：含prefill吞吐 **+2.77%～+5.94%**，Decode **+4.54%～+6.21%**。

| 初始KV | Final | ＋warp-prefix | ＋FFN（4.0） | ＋GDN组合 |
|---|---:|---:|---:|---:|
| 4K | 97.80 | 98.61 | 100.17 | 103.61 |
| 8K | 101.27 | 101.46 | 103.83 | 105.61 |
| 16K | 87.01 | 88.27 | 89.33 | 90.74 |
| 24K | 80.57 | 80.74 | 82.03 | 83.42 |
| 32K | 62.76 | 63.35 | 63.80 | 64.50 |
| 48K | 55.92 | 56.23 | 56.41 | 57.61 |

上表单位为含prefill输出tok/s，所有值来自本轮每项10轮正式测量的总输出/总时间。

## 弱收益与回退限制

- 8K warp-prefix含prefill吞吐仅约+0.18%；48K FFN相对warp阶段仅约+0.31%，短测还曾轻微回退。这两项在这些case中增量较弱，不能概括为所有配置都稳定有同样收益。
- 单开GDN末尾快照策略的固定频率4K短测曾回退约0.38%；只读源状态单开在旧16K正式测试中，总时延略慢约0.15%。因此本总结将已通过本轮六组模型测量的GDN组合列为成果，不独立给两项各算一份收益。
- 追加prefill未证明有稳定新增收益。8K累计版在主矩阵比Final慢约1.70%，反向复测655.05ms对625.45ms，仍慢约4.73%，回退原因未定位。48K FFN主矩阵慢约2.18%，反向复测866.67ms对883.84ms却快约1.94%，该项进程间波动较明显。主表与复测分别保留。补充profiler被另一编译任务持有的共享GPU锁拦住，未启动GPU采集。
- 同一4K/1024规格下，Thor-U与Thor-2的输出digest及MTP工作量不同（329轮/1316验证行，对比363轮/1452验证行）。两机权重/配置13文件SHA256相同，但绝对tok/s不作直接硬件/commit归因。

## 兼容、正确性和诊断记录

`ac57887e`只修复CUDA12/13的Graph API名称兼容，属于平台修复，不独立宣称性能收益。QKV范围/溢出、别名、只读性、Graph生命周期等检查是GDN实现配套的正确性工作，不独立计作加速。原始CUDA13编译失败、首次mtime导致的缓存误用都被保留为诊断，均未产生性能成绩。

历史实测整理分支为`codex/apxinf4-thor2-verified`。`f670ad32`与实际计时的`f7c31521`完整Git树相同，树ID为`2159dfbe0955a2fe0968e868e11304c345e4286f`；沿用原位置的已计时ELF和其构建身份，整理没有产生新二进制。FFN和GDN各101份CUDA设备模块与之前验收版本逐字节一致；本轮新增2个Graph测试与1个真实FFN层测试通过。未push/merge。

各开关需在创建model/worker前设置。阶段四开关（MoE prefix、FFN fork、GDN prefix snapshots、GDN source state）为0000、1000、1100、1111。

生成时间：2026-11-24T00:02:23.039126+08:00。完整结果见项目文档doc/qwen35-thoru/APXINF4_THORU_THOR2_RETEST.md。
