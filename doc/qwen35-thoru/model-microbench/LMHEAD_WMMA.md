# LMhead：可选 BF16 WMMA W4A16 路径

本项只改变小行数 A16 GEMV 的执行方式，默认关闭。目标是用 Tensor Core
执行 target verifier 的4行 LMhead；保留 M1 的旧路径，避免已有测量中的退化。
权重仍是 BLACKWELL_N128_K64_V1：[N/128,K/64,128,32]FP4 bytes 和
[…,128,4]E4M3 scale；激活仍是原 BF16，不额外量化。有限FP4×有限E4M3
block scale 可精确存为BF16；WMMA使用FP32累加，完整dot后才乘原F32 global
alpha，最后BF16RNE。归约顺序与旧SIMT不同，不声明对所有输入逐位等价。

## 配置与范围

- `APXINF_NVFP4_A16_WMMA`：仅1启用；unset/0走旧实现，其他值拒绝。
- `APXINF_NVFP4_A16_WMMA_TILE_N`：启用时32/64/128，默认64。
- `APXINF_NVFP4_A16_WMMA_MIN_ROWS`：启用时1/2/3/4，默认4。
- 仅`MIN_ROWS <= M <= 4`进入WMMA；其余已支持M仍走原实现。
- 本次模型测试显式TileN128、MIN_ROWS4；不能把结果写成默认Tile64的结果。
- L1策略第一次调用后进程固定，切换实验须用fresh process。

模型隔离配置是：

```sh
export APXINF_NVFP4_A16_WMMA=1
export APXINF_NVFP4_A16_WMMA_TILE_N=128
export APXINF_NVFP4_A16_WMMA_MIN_ROWS=4
export APXINF_QWEN35_W4A16_SHARED=0
```

这不是公共Rust API或C ABI变更。新内部C++函数和stub同签名；L3保留原shape/
dtype/device检查，补充N/K整数溢出与output/input重叠拒绝。无新global workspace、
CUDA launch阶段、host/D2H转换或context缓存。原七个L0定义逐字保留；正确entry
SHA为410de669ef48d49c82ba4c61799776d1209351fb402d1676e86a449b5f95b218。
证据JSON还记录原manifest的函数名substring标签问题，未沿用错误entry哈希。

TileN32/64/128分别64/128/256线程、2/4/8warps，每warp独占16输出列；
shared bytes分别8960/15616/28928。A补零到16行但仅读真实M行激活；B按
col_major解释为[K,N]。shared基址32B对齐，BF16 leading dimension72；
staging、复用Ktile和读回输出边界均有CTA barrier。N%128、K%64检查使所有
N/Ktile完整。严格数学/guard/capture测试不因性能改变而放宽。

## quiet 同ELF4K结果

[完整curated证据](evidence/lmhead-wmma.json)逐项绑定原始路径/SHA、dirty source、
实际ELF、checkpoint、环境、recipe冻结、锁及reap。两臂同一decode ELF：
`84da8cdce02c63f11c7dfe74b354e9c6ac4a15a4a8ba7440059219462c51fe11`；
dirty源码base8278871、patch18511ae73e87ff5c4025424f0008e1546928b888d66ee592cf6c14c9d48989d8。
Graph1为已存在功能，其他新优化flag均0；BF16target、draftFP8unit1、NVFP4、
greedy固定depth3、1206MHz、4096输入、1024输出，每臂2warm+5formal。
两臂环境唯一差异为WMMA=0/1。结果不是把已有Graph1计为新贡献。

|arm|正式中位tok/s|正式中位TPOT ms|
|---|---:|---:|
|control|88.507417945429|11.298488004888|
|candidate|90.403204857840|11.061554748778|

TPOT降低**2.0970350723653275%**。下面是全部14轮的原始host callback时间点，
单位ns、起点为同一generate entry；1023间隔TPS直接由last−first计算。
全部14轮输出digest均为`694f5386c486b392`。JSON另存逐轮TTFT、finalize tail、
1024callback数、recipe manifest、严格退出/锁证据及完整共同MTP计数与逐轮SHA。

|arm|iteration|用途|tok/s|first ns|last ns|return ns|
|---|---:|---|---:|---:|---:|---:|
|control|0|warmup|88.492657847|964402268|12524683368|12525006933|
|control|1|warmup|88.524152267|935870658|12492038933|12492237285|
|control|2|formal|88.497166668|931858907|12491551025|12491756674|
|control|3|formal|88.507417945|938856167|12497209396|12497404211|
|control|4|formal|88.524381220|932841314|12488979701|12489180293|
|control|5|formal|88.590921926|930411333|12477869896|12478066637|
|control|6|formal|88.455477876|934811175|12499951340|12500148784|
|candidate|0|warmup|89.912399520|970001417|12347742479|12347938109|
|candidate|1|warmup|90.521206389|933094000|12234313248|12234512924|
|candidate|2|formal|90.439873967|934089694|12245472109|12245669747|
|candidate|3|formal|90.403204858|932934740|12248905248|12249101044|
|candidate|4|formal|90.408397194|930507046|12245827655|12246022933|
|candidate|5|formal|90.362941419|933127555|12254140173|12254339322|
|candidate|6|formal|90.372773129|932388231|12252169229|12252362655|

MTP计数14轮完全相同：324rounds、972proposals、1296target verified rows、
700accepted、158full-accept、133partial、33zero-accept、1023committed rows；
各depth接受/命中/提议数组和其他计数也逐项相同，未靠改变接受率或预算提速。
这证明本case的重复/对照一致性，不能代替所有输入或完整模型数学质量认证。

首轮90.40355tok/s与CPU disassembler重叠，仅保留为diagnostic；没有混入以上
quiet数据。本结果不含天气/AIOS、网络、8K结果、full15或逐case30%达标声明。

## 为什么 MIN_ROWS4：8份实际NCU

以下是同一合成operator的独立NCU capture；单位ms。tensor列是
`sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed`；warp列是
`sm__warps_active.avg.pct_of_peak_sustained_active`，均是该kernel级指标，
不能直接解释成整个模型的算力利用率。原CSV/rep和实际ELF/source/clock恢复
receipt的SHA全部在JSON。Unavailable的shared-conflict指标没有填0。

|case|duration ms|tensor %|warp active %|register/thread|
|---|---:|---:|---:|---:|
|wmma0-tile64-m1|1.469472|0|98.281586|39|
|wmma0-tile64-m4|3.292032|0|49.604623|69|
|wmma1-tile128-m1|2.325472|20.279420|65.818741|63|
|wmma1-tile128-m4|2.512960|18.745222|49.573651|71|
|wmma1-tile32-m1|2.407616|19.570098|57.515407|71|
|wmma1-tile32-m4|2.922848|16.117736|65.785609|64|
|wmma1-tile64-m1|2.215456|21.258317|65.736235|63|
|wmma1-tile64-m4|2.727136|17.266104|57.636788|71|

M1的WMMA实测慢于旧1.469472ms，所以保留原M1；M4的Tile128为2.512960ms，
旧M4为3.292032ms。Tensor Core并不意味着所有小M都更快。Tile128是本次
实测M4选择，不据此推断所有设备/shape的最优参数。

完整模型generation capture（包含prompt prefill+Decode）另记录：

|路径|旧M1次数|M4次数|M4 GPU合计s|
|---|---:|---:|---:|
|旧Graph control|973|324旧SIMT|1.071483360|
|WMMA MIN4|973|324新WMMA|0.825768992|

M4合计降低22.932168353972383%，M1仍是旧kernel。此表是完整capture的聚合，
不是精确NVTX时间窗SQL归因，也不是quiet正式计时；不能将算子22.93%当成模型
或端到端22.93%。模型TPOT改善使用上面的quiet2.0970%。

24个WMMA函数的实际SASS已保存并绑定ELF/SHA；可见宽shared store，未另行添加
或归功于“pair32”源码优化。静态SASS不是运行次数/吞吐证明，未认证所有shared
store都已向量化；本提交的L0kernel保持原已验证4c39…字节不变。

## 正确性、工具证据与复现边界

- 原policy16个raw GPU进程和最终policy显式MIN_ROWS1的16个raw GPU进程全部通过；
  涵盖M1..4、TileN32/64/128、Ktile边界、解析beacon、4064有限产品、F32global
  stage、NaN poison、完整readonly/guards、不同buffer A→B→A和capture/eager。
- raw测试必须显式`APXINF_NVFP4_A16_WMMA_MIN_ROWS=1`才强制所有M进入WMMA。
  test日志requested_wmma不能单独作为实际路由证明；本模型另有真实kernel-count审计。
- 两个actual sanitizer进程（memcheck/racecheck）为0errors/0hazards，目标是原
  dispatch policy、未改的同一L0kernel；记录明确是frozen target+lock/exit SHA，
  launcher可能在/proc采集前退出，不伪称直接application/proc身份。最终MIN1
  raw16进程另绑定新ELF5b7248ac…，不把旧sanitizer ELF101dd090…改名为新ELF。
- Orin纯数学CPU5测试通过；其receipt有source/compiler/stdout成功，但没有CPU
  ELF SHA，未补造。Mac clang精确最终policy/dispatch shim36cases通过，source/
  binary/policy SHA齐全；这些是CPU证据，不是CUDA native编译证明。
- 每项CUDA构建的dirty source、实际artifact SHA和cleanup/共享锁记录保留。原测试
  数学门限仍`max_abs < 0.01`且逐行argmax一致；有限乘积字面测试要求exactbytes。
  合成真实vocab几何不等于真实checkpoint数学oracle，NaN/极端overflow范围也未扩大。

实际证据来自含其他disabled候选和P2测试文件的原dirty overlay。本项隔离提交
只加入WMMA实现/测试7条路径和本说明/证据2条路径，折叠最终MIN_ROWS策略；没有
带入BF16/projection/P2/ref-v4其他源码。这份**隔离clean提交尚未重新编译/GPU运行**，
不能用历史dirty-source结果代替其fresh source/ELF验收。原大日志、NCU/SASS、
模型数据和binary保持外部ignored，Git只保存curated绑定与必要逐轮数据。
