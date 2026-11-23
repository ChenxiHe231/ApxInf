# Final 默认性能路径交付

远程基线为 `9ae60d1b`，原始性能对照为 `9457b9bc`。保留已有远程历史，以普通快进方式更新 `ChenxiHe231/ApxInf:Final`。

| 优化组 | Final 上的提交 | 内容 |
|---|---|---|
| MoE warp-prefix | `a5cca938`、`f25fe56d`（远程已有） | ballot/popcount/prefix 并行路由元数据；M1/M4 支持域自动选择，无运行开关 |
| FFN routed/shared 并行 | `1111b413`、`9ae60d1b`（远程已有），接口清理 `c2c827a0` | 同一 Session/Graph 中并发独立分支，combine 前汇合；删除最后的 enable 参数 |
| GDN 状态与快照组合 | `0e7720be` | 只读 committed 源状态直接写 shadow，删除 recurrent 初始化复制；只保存前 T-1 recurrent 快照，完整接受取 live，部分接受取前缀；删除策略开关 |
| CUDA 兼容 | `4d9d2a26` | 按 SDK 使用正确 Graph API 名称，不独立计算性能收益 |

GDN 两点仅组合通过性能验证，因此同一提交交付，未独立分配组合收益。完整 checkpoint/in-place 底层 API 保留其不同的存储语义和参考测试用途，Qwen target verifier 的默认路径固定使用新 API。

已移除的四个优化环境变量在运行源码中均无引用：MoE setup prefix、Qwen FFN fork、GDN prefix snapshots、GDN source state。新的 GDN API 也没有 bool 策略参数；Graph 分支帮助函数同样没有 enable 参数。形状、后端、精度及执行阶段的合法性判断继续执行；它们不是用户可切换的优化选项。

此前固定 1530MHz 的 480 轮正式测试中，六组生成场景相对原始 Final：warp-prefix+FFN 的含 prefill 吞吐 +0.88%～+2.67%；再加 GDN 为 +2.77%～+5.94%，Decode +4.54%～+6.21%。端到端指含输入 prefill 的模型生成总时间，排除权重加载、分词、HTTP；这些旧数据不能冒充新默认源码的重新计时。

限制：8K 追加 prefill 的累计版两轮回退 1.70%/4.73%，原因未定位；48K FFN 回退/收益符号随反向复测改变。收益声明仅适用于已验证的生成场景。

## 新默认路径的实际验证

源码 `c2c827a0` 独立重建，11组case均为2轮预热＋10轮测量，共110轮正式计分。GPU固定1530MHz（最高可用1575MHz），CPU2601MHz、EMC4266MHz。NVFP4、动态FP8 KV、greedy、固定MTP depth3；生成1024token，Decode计首token之后1023个间隔。

四项优化环境变量均未设置，Graph开启及GDN full Graph的两个强制环境变量也未设置。Thor动态FP8执行配置默认启用全Graph；因此本次没有借助这些变量启用FFN/GDN优化。其他原有计量配置保持冻结recipe约定，不是新增的性能开关。

每个case所有预热和测量的输出digest及完整MTP工作量均与冻结Final相同。736份recipe不变、无在线调优或缺失回退，14个原始模型文件身份未变。构建、CUDA分层检查、Graph拓扑/失败恢复、真实FFN层M1/M4与M2/M3边界、GDN两个物理缓冲区的部分/完整接受及快照、低层T1/2/3/4/8精确对照、只读性、guard、别名/溢出拒绝、Graph引用生命周期及memcheck均通过；memcheck零错误。时钟策略已恢复，无遗留本任务GPU进程。

以下是**新默认源码的实际计时**与同机冻结基线比较，不是重新交错采集的A/B。原始Final/FFN/旧GDN组合基线的权重、输入、精度、策略、硬件及有效recipe不变，按项目规则复用，未把源码相同当作新二进制相同。

| 初始KV | 冻结Final Decode | 新默认Decode | 含prefill吞吐：Final→新默认 | 含prefill收益 |
|---|---:|---:|---:|---:|
| 4K | 103.38 | 108.87 | 97.80→102.68 | +4.98% |
| 8K | 112.30 | 119.07 | 101.27→106.74 | +5.39% |
| 16K | 108.82 | 114.34 | 87.01→90.45 | +3.96% |
| 24K | 113.26 | 119.11 | 80.57→83.55 | +3.70% |
| 32K | 91.38 | 95.45 | 62.76→64.46 | +2.71% |
| 48K | 100.06 | 104.96 | 55.92→57.43 | +2.70% |

新默认相对冻结原始Final：Decode **+4.45%～+6.02%**，含prefill吞吐 **+2.70%～+5.39%**。相对旧已验收GDN组合的含prefill吞吐变化为-0.90%～+1.06%；与旧FFN阶段比较为+1.03%～+2.80%。不得跨版本累加这些百分比。

| 初始KV | 追加输入 | 冻结Final prefill ms | 新默认 prefill ms | 时延变化（正值回退） |
|---|---|---:|---:|---:|
| 4K | 4K | 589.37 | 588.81 | -0.09% |
| 8K | 4K | 628.72 | 635.22 | +1.03% |
| 16K | 4K | 675.38 | 680.17 | +0.71% |
| 32K | 4K | 785.46 | 777.59 | -1.00% |
| 48K | 4K | 873.50 | 888.39 | +1.70% |

这些结果继续按生成和追加prefill分别报告。追加prefill的波动/历史8K回退没有因移除开关而被解释，不能声称全场景加速。

## 复现与证据

Thor-2代码目录 `/home/wwxq/devlocal/eglops-eval/checkout`；权重继续使用 `/home/wwxq/models/Qwen3.6-35B-A3B-NVFP4`，recipe继续使用代码目录下 `devlocal/apxinf4-bench/recipes`。本次构建复用既有Cargo target，未复制权重或重做调优。

本次控制器/构建/原始日志位于代码目录下 `devlocal/final-defaults/`；正式结果位于 `runs/default-eleven-01/`。首次启动benchmark在取得锁之前被其他编译任务阻挡，没有计分；随后排队完成了唯一一轮本次默认源码矩阵，没有挑选替换失败成绩。

离线核验（不占GPU，不修改时钟/权重/recipe）：

```sh
python3 doc/qwen35-thoru/model-microbench/verify_apxinf4_defaults.py   devlocal/final-defaults --run default-eleven-01 --complete   --out devlocal/final-defaults/results/reverified.json
```

维护证据为 `model-microbench/evidence/apxinf4-final-defaults.json`，包含每轮计分/预热原始迭代记录、源码树、构建ELF SHA256、正确性结果、冻结基线数据、原始文件hash和清理核验。原始case receipt、stderr及频率采样保存在服务器上述目录，离线核验器会直接重新检查它们。
