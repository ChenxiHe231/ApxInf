# Thor3：长 prefill 默认采用既有四 warp GDN norm

> 正式受影响范围 A/B、strict、最小正确性和新候选 profiler 原件已独立核验，可作为独立变更提交。完整新版本矩阵与两个20%目标仍待验收，尚未推送。

本改动只调整 FP16 输入 norm 的 launch 策略：当 tokens≥4096、heads32、D128 时，默认使用已有四 warp 路径，并且该域不读取环境变量。每个 warp 仍处理原来的一行，lane 对应、归约和 BF16 舍入顺序、kernel 正文、ABI 及资源生命周期保持不变。其他 FP16 域保留既有策略，BF16 入口和 convolution 均未改。没有新增用户开关或环境变量。

这是复用既有四 warp 执行路径的默认行为优化，不是新 norm 数学算法。原最快有效 prepared/Graph 路径保持作为模型对照；没有换慢 eager 路径制造收益。

## 正式追加 prefill A/B

同一 Thor3、同一原位检查点和输入，NVFP4 MoE、target 动态 FP8 KV、draft FP8 scale1、greedy、固定 MTP depth3；GPU锁1530MHz，最高可用1575MHz，CPU2601MHz、EMC4266MHz。两 arm 均2次预热、5次测量，禁调优/fallback/profiling。初始prefix准备排除，时延包括新增4096输入及首argmax输出。

父 C 为已验证 routed-quant02 的实际 Thor3 产物，对应产品451b；没有将旧 ELF 改贴新源码标签。

| 初始KV | 新增输入 | 父 C ms | 候选 ms | 时延降低 | 顺序 |
|---|---|---:|---:|---:|---|
| 4K | 4K | 468.5065814 | 454.5941040 | 2.969537% | C→候选 |
| 32K | 4K | 630.4585094 | 614.3860930 | 2.549322% | 候选→C |

四个正式进程共28轮，完整首argmax向量/FNV与零提案 MTP work 相同，所有轮次稳定。另有候选4K初始KV输出1024的strict及32K+4K追加strict；加上方向筛选，共45轮、1068个输出ID重新解析核验。不得把追加prefill的一token输出说成1024输出测试。正式303个频率样本全部固定；strict99、方向筛选172、算子43和边界47个样本也符合1530/2601/4266。设置恢复、recipe、source、runtime和最终资源守卫通过。两个顺序块不构成统计显著性声明。

## 最小正确性与反例

实际harness03比较旧默认一warp、既有四warp和新自动四warp：4096真实shape完整BF16输出逐位相同，finite、readonly、canary及改变输入后的Graph replay对fresh eager通过。输入含正负零、half subnormal、BF16舍入边界与finite最大值。九个边界case覆盖4095/4097、较小M64/256、heads16/31/48及D64/256，均实际通过。BF16入口源码未变，本轮未冒称重跑其GPU精度。

算子2warm3正反顺序中，新默认相对旧一warp的event区间降低43.04%/41.64%；但相对既有四warp仅+0.258%/−2.263%的方向差异，不能声称新算法加速。CUDA event包含launch供给空隙，不是纯kernel或E2E。

原全局`APXINF_GDN_PREFILL_TILED=1`的模型诊断4K为453.525766ms，但它同时改变conv，不能隔离归因norm。本候选只改norm，其正式收益取上表的同配置新鲜成对N5；原全局开关数字和N3对旧N10筛选都不增加正式收益。早期harness02缺少uint16_t声明的编译错误在GPU计时前修复，仅属probe编译问题，原失败保留，不算算子负收益。

## 来源与 profiler 解释

实际测试源为451b8163加patch244a0636，仅`gdn_ops.cu`一文件；源码SHA f644ef64，core fb025603、combined f2789d25。实际新native f83a53de仅gdn_ops.o一成员变化，父native9a4fa1e4的其他成员不变；GEMM/attention指纹和recipe恒定。实际Cargo输出证明模型库fresh=false，由451b重新编译，避免复用之前default/draft的Rust缓存；source字节及元数据恢复。新context ELF e5a50e18、MTP ELF72c10552与build/run receipt绑定。大ELF/native archive未下载本地重hash，不将源码相同当作二进制相同。

复用既有postquant C的实际append4K trace，对比本次新ELF的同输入、同配置2warm1 NSys诊断；未为换收集器重跑旧profile。两个SQLite均readonly打开、quick_check通过，实际具名norm次数均为30，kernel符号仍为`gated_norm_seq_kernel<__half>`：

| 诊断项 | 父 C | 候选 |
|---|---:|---:|
| norm grid / block | 131072 / 32 | 32768 / 128 |
| norm kernel sum=union ms | 31.132224 | 17.638176 |
| registers/thread / dynamic SMEM | 33 / 0 | 33 / 0 |
| 全GPU kernel union ms | 443.479712 | 428.676192 |
| ROI代理区间 ms | 481.841453 | 455.520469 |

实际launch从每block一warp变成四warp，与源码的默认路径选择一致；每warp仍处理原来一行。norm component降低43.3443%，支持其减少该阶段成本的方向，但不是新的正式模型收益。新profile全部3次的首argmax和work与正式锚相同，19个receipt产物SHA/bytes及45个锁频样本/设置恢复均通过。

两个trace都缺少精确的measured NVTX范围；ROI明确采用`cuProfilerStart.end`到最后一条被捕获CUDA runtime API结束的代理，排除flush/stop，而不是将它称为正式E2E。profile开启后其他kernel/host供给和不同采集批次也可能变化，不能把component总和当关键路径f，不能用ROI降低或43.34%增加性能成绩；正式收益仅取上方N5成对结果。关键路径f仍为MISSING。本项未采NCU计数器，不据此声称occupancy、stall或带宽的因果变化。

本轮只正式验证4096新增输入及所列边界，未穷举所有更长输入或其他GPU。完整新版本15-case、与最新Final组合后的新ELF来源/recipe桥接和两个20%目标分别验收，不跨版本叠加百分比，候选family预算不因换包、编译修复或文档重置。

[精炼证据](evidence/gdn-prefill-norm-thor3.json)保存未舍入样本、work、配置/产物身份及原件SHA。实际raw归档为`results/gdn-norm-acceptance-evidence-01`，SHA f77d2b10c55307b6f692868b4fe892843339bcf77b17c0207b1214bf4289faef，116份原文件与归档逐字一致。新profile归档SHA为be0c0c7006906c1d84691d40dde5f993f45a6dc6d8a5c0f8ef8e050fc58052f0，20份文件与归档逐字一致。两份CPU独审记录留在开发草稿目录，不作为产品运行依赖。这两份维护文档在测试后新增，不改变已测代码或给ELF重签身份。
