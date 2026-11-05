# LM Head M1：一次装入全部激活

新增默认关闭的 `APXINF_NVFP4_A16_M1_STAGE_ALL=1`，仅适用于原 A16 GEMV 路径的
M=1、N=248320、K=2048。目标是降低 MTP draft 的 LM Head 开销。

## 改动

原实现循环处理32个K64块，每块装入激活并执行两次CTA barrier。
新实现由256线程一次装入2048个BF16激活，转为F32保存在32×68的shared数组，
保留每个K64的原四元素padding，只执行一次CTA barrier。权重布局、FP4/E4M3解码、
每个lane的K块与pair累加顺序、XOR归约、完整dot后乘F32 global scale、BF16 RNE保持。
激活精度和KV精度均未改变。

```sh
export APXINF_NVFP4_A16_WMMA=1
export APXINF_NVFP4_A16_WMMA_TILE_N=128
export APXINF_NVFP4_A16_WMMA_MIN_ROWS=4
export APXINF_NVFP4_A16_M1_STAGE_ALL=1
```

WMMA的MIN_ROWS=1会使M1绕过此GEMV分支，因此对照显式使用MIN_ROWS=4。
新flag在进程第一次有效M1 GEMV调用时缓存；unset/0关闭，非法值返回provider错误。
切换配置需要新进程。公共API和权重/recipe格式保持原样。

## 实测

|范围|原实现|新实现|时延下降|
|---|---:|---:|---:|
|算子CUDA Event中位，每臂2warm5，每batch100次|1.460531 ms|1.253964 ms|14.14%|
|Nsight选中单个kernel|1.464384 ms|1.255904 ms|14.24%|
|模型4K KV、输出1K，Decode中位|87.1620 tok/s|88.6760 tok/s|TPOT下降1.7073%|

模型每臂2轮预热、5轮正式测量，BF16 KV、NVFP4 MoE、固定MTP depth3、Graph1、
WMMA M4/Tile128、Packed GQA开启、GPU固定1206MHz。757个已有recipe保持字节不变，
正式阶段严格recipe-only、无autotune。计时使用1024个host callback中的首末1023个间隔；
TTFT和最后callback之后的尾部另计。全部14轮digest为`a79ed7c8bf839729`，
每轮351个MTP round及全部接受/回滚统计一致。

这次模型对照是独立的同ELF测量，不将此前Packed GQA、WMMA的其它构建收益直接相加。
Event包含可能的host enqueue间隙；NCU两点用于解释算子机制，不作为正式样本。

## Profiler解释与精度

实际kernel名证明两臂分别进入原GEMV和stage-all实现，grid1940、block256相同。
寄存器39→42，static shared272B→8704B，实测active warps47.18→38.23。
barrier-stall ratio从1.867234降到0.214520，支持减少同步开销这一解释；该ratio不是百分比。
本项没有测得物理DRAM带宽，不能据此宣称带宽利用率。

冻结的独立F64参考和严格max_abs<0.01阈值保持不变。真实词表几何的合成权重/输入，
全部248320输出、16个稀疏K地址锚点、只读输入和256B guards、3次Graph重放和3次eager检查通过；
两臂496640B输出逐字节一致。另4项既有回归通过，其中real-vocab的M1 A/B/A实际覆盖新增分支。
memcheck为0 errors，racecheck为0 hazards、0 errors、0 warnings。
16个地址锚点是正向检查，未宣称为坏kernel注入测试。

[完整证据](evidence/lmhead-m1-stage-all.json)绑定实际源码overlay、ELF/native archive、
checkpoint、recipe清单、环境、逐轮callback、输出摘要、profiler与sanitizer日志SHA，及锁/子进程回收/频率恢复。
本提交发布的两个源码postimage与测试文件一致；模型记录保留其dirty实验overlay身份。

长上下文scalar/MTP首步精度验收仍未通过；此项保持默认关闭，不代表完整P2/P5通过。
6组Decode、5组追加Prefill、4组冷Prefill的完整回归与30%目标仍未完成。
