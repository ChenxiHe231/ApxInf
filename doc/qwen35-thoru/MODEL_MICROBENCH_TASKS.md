# 模型推理基准任务拆解

用户于2026-11-04收敛任务：暂停AIOS端到端优化，先优化冷Prefill、已有KV追加Prefill、长上下文Decode共15个case。现有未验收R01改动保留隔离，不能默认并入新基线。

## 阶段与交付

| 阶段 | 工作 | 交付/完成条件 |
|---|---|---|
| P0 口径与工具 | 固定权重/token输入/recipe/配置；调优和正式测试分离；核对时钟、KV长度、MTP策略和容量；Prefill/TTFT与Decode分别计时 | 15case manifest、精确argv、源码/二进制/输入哈希；recipe miss/自动调优不混入正式结果 |
| P1 当前基线 | 冷Prefill4组、追加Prefill5组、Decode6组，先BF16 target KV | 两轮预热、五轮正式的逐轮数据；空KV/已有KV/输出长度符合定义；历史数据仅作参考 |
| P2 精度门禁 | BF16路径独立验收；GDN bulk/long/checkpoint、attention mask/页界/位置/scale、MTP接受/回滚/终止 | 独立reference、冻结阈值、shape完整、失败有记录；FP8单独验收，不把其reference问题直接套BF16 |
| P3 Prefill优化 | 分别定位GEMM/MoE、GDN整块、长KVattention、搬运与launch；每次只改变一项 | 冷Prefill与追加4K的无插桩收益+独立Nsight；代表shape探索后全9组回归 |
| P4 Decode优化 | 先拆scalar/固定MTP/自适应MTP；检查验证批处理、接受率、KV读取/复制、同步；再单列FP8 target KV | 6组tokens/s、TPOT、MTP接受与有效提交数；同精度和跨精度结果分别标注 |
| P5 合并验收 | 每项优化父版本/候选消融，保持同配置、输入和精度 | 15组精度与性能回归；最终10轮正式；有可重复收益再commit并附profiler与变更原因 |

## 15个case

- 冷Prefill：空KV + 输入2048/4096/8192/16384 token。
- 追加Prefill：初始4096/8192/16384/32768/49152 KV + 新输入4096 token；构建初始KV不计入追加时间。
- Decode：初始4096/8192/16384/24576/32768/49152 KV + 输出1024 token；首token计入TTFT，Decode按1023个生成间隔统计。

“冷”指空KV且权重已加载，进程加载/编译/调优另计。每个case记录原始KV/新增token/输出token、MTP深度策略、target与draft KV dtype/scale和物理block、功耗及频率策略。历史表来自用户，不等于本轮实测，不在缺失原始源码/配置时宣称精确复现。

首版保留BF16 target KV，沿用用户确认的temperature=0 greedy MTP口径；scalar作为消融。固定depth与自适应各自单列。FP8是后续独立配置，须另过校准/误差门禁。历史TRT Decode为FP8 KV，与ApxInf BF16不同精度，不能用该差值直接归因引擎；同精度对照另测。

所有Prefill输入长度都64对齐；之前291-token bulk+tail的收益不能直接归到这些case。先对4K/16K冷Prefill、48K+4K和4K/48K Decode取代表trace，按实际瓶颈优化，再回归全部15组。Roofline仅作用户给定假设下参考，功耗/频率、dtype/字节量或算法变化后需要核对，不能把它当作已测性能。

## 已确认的工具前置问题

P0已将autotune改为显式独立阶段，正式进程设置APXINF_REQUIRE_RECIPES=1并关闭online tune；native GEMM/attention在recipe缺失或不可恢复时直接拒绝，不允许隐式fallback。lookahead、分深度统计与容量已统一到实际depth3。runner的26项CPU合同检查及native GPU recipe子进程合同已通过；这不表示完整15组基线已经完成。不得启用AIOS proxy/weather压缩/按天归一化，这些不属于当前模型基准。

阶段内GPU任务串行；代码分析、reference/脚本准备可并行。每轮保留假设、parent/candidate、完整配置、精度检查、未插桩时延、Nsight和接受/否决结论。尚未通过GPU与完整输入验证的候选不作为已发布优化。

测量命令见 [使用说明](MODEL_MICROBENCH_USAGE.md)，P0验证摘要见 [验证记录](model-microbench/evidence/p0-validation.json)。正式测试发现的数值/状态不一致必须保留失败记录；显式诊断采集可以保留时间分布，但始终标记精度未解决，不升级为正式性能结论。
