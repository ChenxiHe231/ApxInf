# FA2 packed-GQA：默认关闭的小 Q attention 路径

本项为 BF16 target attention 增加独立 FA2 implementation4/version1。它把同一
KV head下的8个query heads打包为query行，使小Q1..4以2个packed heads、8T行
执行split-KV，再还原原O/LSE布局。原default provider、vendor FA2头文件和外部
Rust/C ABI不变。该项已通过独立raw算子门禁及synthetic operator A/B，尚未通过
actual checkpoint/model集成、全15case或逐case30%验收。

## 配置、数学与资源边界

- `APXINF_FA2_PACKED_GQA=1`显式启用；unset/0保留原选择，其他值报错。策略首次
  使用后进程固定，比较两臂必须使用fresh process。
- 仅BF16、B1/Hq16/Hkv2/D256、T1..4、K>64、exact-live causal dense/KV，
  query_start=K−T；其他几何保持原路径。新路径要求当前primary CUDA context。
- 新key后缀`|packed-gqa-q1to4-v1`及provider标识
  `flash-attention-2-packed-gqa-q1to4-v1`隔离原recipe，旧普通recipe不能假命中。
- 先bit-copy pack Q，再private split core、原FA2 FP32 split-combine，最后inverse
  O/LSE unpack。packed head对应KV group，packed行m的chronological位置为m/8。
  保留原P→BF16后Tensor Core PV、FP32 softmax denominator/partial O与K遍历；
  typed-P rounding门禁独立于ideal-F64门禁，不用FP32-PV替换原数值路径。
- 每个live prepared Execution独占65792B slab（Q32768/O32768/LSE256，256B对齐）；
  graph保留Execution owner，不让live graph与其他Execution复用同址。按device/
  primary-context的free pool最多保留32idle slabs=2105344B/device，保留primary
  context引用，属于process-lifetime资源。外部强制reset仍在现有live-owner合同外。
  释放失败的lease隔离，不能推断可安全复用；原三份FA2 scratch分配仍保留。
  enqueue只有pack/core/combine/unpack和错误检查，不在该阶段alloc、D2H或同步。

这次native源变更会改变attention全局build-id。因此，即使关闭packed flag，也
必须在新ELF下显式检查并独立prepare/retune原普通recipe；不能沿用旧build的调优
收据并声称免重做。prepare与严格formal分进程，formal始终REQUIRE_RECIPES=1、
autotune=false、不得fallback/写recipe。两种provider各有独立recipe namespace。

## 实际质量门禁

[完整curated证据](evidence/fa2-packed-gqa.json)绑定各原始source/ELF/log SHA、
recipe、共享GPU锁及reap。原CUDA build01/02因namespace问题失败并保留；root
两次修复后build03真实通过，最终CU SHA为
`a912c38d427f1d8032a16feccc338e77a9e975b80ca5c72ec184b55b2f85d22f`。
其余八个V2实现/测试postimage与原manifest完全一致。operator bench为第十个源码
路径。已有independent F64 reference在基线中存在且逐byte不变。

七阶段真实GPU流程全部通过，包括单独prepare、下面五组质量检查及strict-miss。
所有实际输出/head-row仍要求`max_abs < 0.01`，无阈值放宽：

|门禁|实际checks|worst max_abs|
|---|---:|---:|
|P65 typed-P BF16 literal|16|0.000000000000000|
|非零Q/K full-F64 A/B/A|160|0.004194498062134|
|40K/48K完整uniform analytic|32|0.001030921936035|
|40K/48K非零head/time beacon|32|0.001944065093994|
|4K双owner、drop后capture replay|48|0.000315397977829|

full-F64覆盖40个普通形状；long/beacon各覆盖40K/48K×T1..4。实际逐head-row
检查、全输出、只读Q/K/V及padding/future poison/guards、A→B→A不同binding、
graph owner存活均保留。uniform零Q/K本身不能认证组内head置换或P rounding，
因此另有非零beacon和typed-P witness；typed域不冒充ideal-F64域。fresh empty
strict-miss测试确认错误退出且没有fallback、在线tune、recipe或输出写入。

memcheck/racecheck在4K×T1..4 graph-owner gate实际各0errors/0hazards，
不扩为所有长形状的sanitizer认证。其身份范围为明确frozen target+
共享锁/exit前后SHA；sanitizer launcher可能在/proc采集前退出，不伪称直接
application/proc身份。Orin独立uniform CPU11测试及148个定向负例实际通过；
Thor benchmark contract实际1测试通过。这些CPU证据不代替CUDA或模型质量。

## 同ELF、fixed1206的独立operator结果

两臂使用同一个ELF：
`b523572928b10991b7f52d7fb1bbdd47f063e450f23173d8fa16c25b6581599b`。
6个prefix×Q1/4=12个distinct shapes，两臂共24份独立严格recipe；相同输入FNV
逐case匹配，各臂输出重复一致并通过完整F64 admission。每case准备一次，排除
prepare/CPU reference/readback验证，2个warmup batches后正式5个batches，
每batch50次ExecutionSession dispatch，最后一次stream sync。

下表是host-wall/batch÷50的正式中位数，包含dispatch与sync，**没有CUDA event
计时，也不是模型tok/s/TTFT**。S为实际调优后的split数，未强制两边相同：

|prefix|Q|S control/candidate|同S|control ms/call|candidate ms/call|降低%|
|---|---:|---:|---|---:|---:|---:|
|4K|1|5/13|否|0.203931860|0.049787780|75.586071|
|4K|4|5/13|否|0.205768320|0.051889800|74.782415|
|8K|1|7/13|否|0.379302960|0.073968700|80.498781|
|8K|4|7/13|否|0.379592400|0.075263700|80.172496|
|16K|1|7/7|是|0.708352580|0.151753140|78.576609|
|16K|4|7/7|是|0.708896660|0.154302980|78.233361|
|24K|1|7/7|是|1.042185560|0.214155740|79.451285|
|24K|4|7/7|是|1.043266480|0.216425180|79.255043|
|32K|1|7/13|否|1.386983160|0.278563340|79.915882|
|32K|4|7/7|是|1.387568160|0.278709640|79.913805|
|48K|1|7/7|是|2.043900000|0.400901860|80.385446|
|48K|4|7/7|是|2.044366840|0.403295000|80.272865|

12行降低74.7824%..80.4988%。7行两臂同S7；其余是各自retuned S的端实现对照，
不能把其中全部收益单独归给packing或声称matched-S因果实验。完整120个正式
sample、每case配置/inputFNV/recipe key+byte FNV/输出FNV及收据SHA在JSON中。
这些输入是synthetic full-F64 fixtures，未使用真实checkpoint Q/K/V trace。

AB01的control prepare真实77.79s通过，但controller漏识Rust harness前缀造成
false negative。AB02复用CPU contract与该control prepare原log各一次，并未重跑；
candidate prepare及两臂formal为fresh process。原failed receipt、复用标记和log
SHA均保留，没有将parser修正当成第二次GPU通过。

## 四份实际Nsight：计入所有额外kernel

下面是完整process的GPU核均时，单位ms/call；每个组成均351次=1次prepare+
2×50 warmup+5×50 formal。它不只包含正式区间，不能用profile结果替代quiet
host-wall分数；API/拷贝时间与GPU工作可重叠，不能相加作为wall savings。

|prefix/Q|flag|core|combine|pack Q|unpack O/LSE|全部kernel合计|
|---|---:|---:|---:|---:|---:|---:|
|4K/Q4|0|0.195008182|0.005167863|0.000000000|0.000000000|0.200176046|
|4K/Q4|1|0.031917402|0.006812444|0.001693356|0.001934222|0.042357425|
|48K/Q4|0|2.028856980|0.009149265|0.000000000|0.000000000|2.038006245|
|48K/Q4|1|0.383429652|0.006086929|0.002053835|0.001928387|0.393498803|

4K/Q4对照S5→13，包含重调优因素；48K/Q4两臂同S7，core约2.028857→
0.383430ms，仍须计入新pack/unpack和combine。四份真实非空report的SHA/size、
Nsight版本、stats导出命令/退出码、原CSV/日志/ELF/source/clock/lock/reap已绑定。
profiler身份仍是launcher，未伪称直接application/proc证明。本提交未纳入NCU数据，
不填写DRAM带宽、Tensor/warp利用率或把缺失值写0；表中0只表示原路径无pack/
unpack组成。

## 复现和未验收范围

native policy默认off。通过已有Thor小池build guard分别构建两个固定target：

```sh
cargo test --release -p apxinf-cuda-new --no-default-features   --test fa2_packed_gqa --no-run --jobs 2
cargo test --release -p apxinf-cuda-new --no-default-features   --test fa2_packed_gqa_bench --no-run --jobs 2
```

运行必须全程持`/tmp/apxinf-gpu.lock`直到child reaped，fresh `--exact`进程、
绝对隔离recipe目录。raw先`fa2_packed_gqa_recipe_prepare_only`（REQUIRE0），
再fresh REQUIRE1运行各quality/owner/missing-recipe exact entry。operator先
`fa2_packed_gqa_operator_prepare_only`，再fresh
`fa2_packed_gqa_operator_measure`，分别显式flag0/1，REPORT1，recipe origin
retuned/restored必须如实声明；保存actual split/provider tuple和原字节hash。不要
运行all-ignored，也不要在正式进程在线tune或静默替换recipe。

证据来自有明确source/ELF的dirty8278871 overlay。这个隔离clean提交尚未重新
编译/GPU运行；它仅带十个实现/测试文件及本doc/evidence。actualcheckpoint层、
完整模型、全15、逐case30%、天气AIOS/E2E及原4K scalar/verification质量差异
仍未验收；已有4K差异不能归因为这个新FA2。只做下一步实际模型gate/quiet A/B后
才可给模型收益结论。后续NCU或checkpoint实验若完成，应另绑定真实收据补证。
