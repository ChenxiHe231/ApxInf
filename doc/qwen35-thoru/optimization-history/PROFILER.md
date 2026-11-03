# Prefill、Decode 与端到端收益：证据及边界

本报告区分固定输入实验、完整天气trace和正式E2E。kernel累计时间是所有kernel duration之和，不是GPU忙碌区间或SM利用率。Nsight及MTP phase timing会加入同步，改变自适应决策；诊断时延不进入正式成绩。

## 固定输入 Prefill 对照

使用已核对SHA256的 `b452e715` / `118337ce` 二进制、同一NVFP4模型与静态FP8 KV。同一JSON payload，关闭context reuse、max_tokens=1、temperature=0；两种长度各单独调优、重启关闭autotune、预热1次、正式3次。输入是公开的重复 `a ` 合成文本，无AIOS或天气网络。实际profile确认full=computed、reused=0、completion=1；正式前后recipe内容未变。

| 输入 | b452e715 TTFT中位数 | 118337ce TTFT中位数 | 新版变化 |
|---|---:|---:|---:|
| 291 tokens | 580.278 ms | 292.310 ms | -49.6% |
| 320 tokens | 189.626 ms | 220.208 ms | +16.1% |

非64对齐的291-token输入TTFT下降49.6%；320-token对齐控制组没有加速，反而慢16.1%。每组仅3轮、GPU未锁频，不能把这组结果泛化到全部长度，也不能将TTFT全部当成GDN kernel时间。

代码原因位于 [`qwen35_gdn_prefill`](../../../crates/apxinf-cuda-new/src/ops/qwen35_mixers.rs)：旧版仅在整段长度可被64整除时用FlashInfer快速路径；291-token输入整段逐token recurrence。新版先处理完整64-token块，最后35个token走精确尾段，不补假token。该实验比较的是完整118提交，仍含其他MTP准备变化；对齐控制组与源码路径支持上述解释，但不是单个函数开关的纯消融。

[完整payload、每轮profile、argv、二进制/recipe哈希](evidence/prefill-same-payload.json)。测的是无复用的合成Prefill，不是天气E2E，也不是8K静态前缀命中后的1-token请求。

## 独立天气 Nsight captures

每个capture前先在新服务上预热2轮，仅采1轮真实天气。公开kernel/API/memcpy汇总CSV、报告SHA256和MTP诊断JSON；CSV仅去除行尾空白，JSON分别保留原始和归档CSV哈希；完整nsys-rep保留Thor-U `devlocal/commit-audit/nsys-<sha>.nsys-rep`。工作负载会随工具查询数、返回内容、缓存命中与输出长度变化，下面不作严格E2E消融排名。

| 版本 | kernel累计 | kernel个数 | GDN累计 | attention累计 | 证据 |
|---|---:|---:|---:|---:|---|
| `36d0d0f6` | 2151.052 ms | 176000 | 989.036 ms | 309.369 ms | [CSV](evidence/36d0d0f6-nsys-stats.csv) / [MTP](evidence/36d0d0f6-mtp-diagnostic.json) |
| `b452e715` | 2119.312 ms | 179364 | 972.900 ms | 257.152 ms | [CSV](evidence/b452e715-nsys-stats.csv) / [MTP](evidence/b452e715-mtp-diagnostic.json) |
| `118337ce` | 1389.826 ms | 42572 | 85.403 ms | 276.552 ms | [CSV](evidence/118337ce-nsys-stats.csv) / [MTP](evidence/118337ce-mtp-diagnostic.json) |
| `679322b0` | 1418.578 ms | 34281 | 83.935 ms | 309.524 ms | [CSV](evidence/679322b0-nsys-stats.csv) / [MTP](evidence/679322b0-mtp-diagnostic.json) |
| `586edc9e` | 8101.127 ms | 248923 | 942.495 ms | 1735.819 ms | [CSV](evidence/586edc9e-nsys-stats.csv) / [MTP](evidence/586edc9e-mtp-diagnostic.json) |
| `9d6642c5` | 1449.866 ms | 58756 | 166.328 ms | 282.008 ms | [CSV](evidence/9d6642c5-nsys-stats.csv) / [MTP](evidence/9d6642c5-mtp-diagnostic.json) |
| `9596e6f8` | 1266.875 ms | 35967 | 168.213 ms | 251.617 ms | [CSV](evidence/9596e6f8-nsys-stats.csv) / [MTP](evidence/9596e6f8-mtp-diagnostic.json) |
| `4424f1b7` | 885.819 ms | 24614 | 55.046 ms | 194.412 ms | [CSV](evidence/4424f1b7-nsys-stats.csv) / [MTP](evidence/4424f1b7-mtp-diagnostic.json) |

GDN统计包含原生gdn_ops与FlashInfer GDN（按kernel名匹配）；attention按attention/flash_fwd命名归类，不声称覆盖全部隐式相关工作。完整CSV保留原始名称便于复核。586edc9e capture耗时较大，伴随额外工作，不删除它以塑造单调加速曲线。

### FP8 attention QK 的局部变化

`36d0d0f6`→`b452e715`：`attention_fused_split_kernel` 从scalar FP32 QK改用BF16 WMMA、FP32累加。两段trace平均kernel时间740.914→546.337 µs，下降26.3%；调用320→342，累计237.092→186.847 ms。整段kernel累计2151.052→2119.312 ms仅下降1.5%。两次shape/grid混合和输出长度不同，因此只报告观测，不宣称它独立带来26.3%的Decode或E2E收益。相关代码在 [`qwen35_attention.cu`](../../../crates/apxinf-cuda-new/native/adapters/qwen35_attention.cu)。

### MTP、GDN和缓存减少了哪些工作

118包含depth1/2/3 verifier、GPU token链和接受判断/打包、GDN短序列verification/checkpoint融合、GDN Prefill bulk+tail，以及exact prompt snapshot/LRU。host仍参与commit，尚未把完整MTP round捕获成一个CUDA Graph。

从b452到118的capture，GDN累计972.900→85.403 ms，kernel总数179364→42572；但实际新增Prefill从291/432等变成1/64，且输出长度有差异。因此这不是“GDN算子快91.2%”的证明，其中很大部分是根本不再执行重复Prefill。固定长度控制的49.6%下降才是独立于Prefix Cache的Prefill实测。

| API/搬运 | b452e715 capture | 118337ce capture | 4424f1b7 capture |
|---|---:|---:|---:|
| cudaLaunchKernel | 167256 次 / 821.435 ms | 6387 次 / 27.131 ms | 4698 次 / 20.212 ms |
| cudaGraphLaunch_v10000 | 0 | 1200 次 / 87.714 ms | 640 次 / 48.731 ms |
| cudaStreamSynchronize | 2171 次 / 1328.124 ms | 1745 次 / 1191.891 ms | 1664 次 / 726.127 ms |
| cudaDeviceSynchronize | 692 次 / 2.681 ms | 466 次 / 4.614 ms | 422 次 / 4.099 ms |
| Device-to-Host | 56 次 / 0.157 ms | 32 次 / 0.047 ms | 18 次 / 0.033 ms |
| Device-to-Device | 8111 次 / 83.554 ms | 2441 次 / 71.916 ms | 2179 次 / 68.569 ms |

这些是完整capture的计数，不能全部归到一个MTP round。StreamSynchronize的API duration包含等待GPU完成，不能当作可直接删除的CPU空转。D2D仍有大量snapshot/状态复制，完整round graph尚未完成。旧NVTX宏受legacy cuda feature门控，本次qwen35-cuda构建不把缺失NVTX范围当证据；使用实际CUDA kernel/API和独立phase日志。

最新4424的diagnostic round中，draft/verify/commit共16轮累计约158.450/591.321/60.430 ms；这些额外同步计时用于定位verification仍占大头，不能与无插桩E2E相加。该capture第1次调用接受26/30个proposal，第2次调用见原始MTP行；单轮接受率不代表全组平均。

最新GPU kernel累计885.819 ms中，fused-split attention为169.584 ms，两个主要MoE gather/finalize融合kernel共194.897 ms，NVFP4 GEMV等也有明显占比。当前不支持“唯一瓶颈是KV带宽”的结论。没有Ncu SM/DRAM计数，本报告不填写算力或带宽利用率。

## 端到端提升能说到什么程度

最新4424正式10/10、中位数1.774990 s，9596正式10/10、中位数1.775060 s；两者几乎持平，日期/结构修正没有可证明的GPU加速。9d正式8/10成功、975全流复核9/10，均不能给出有效总体中位数，不能从成功子集算早停的端到端提速百分比。

早停把显式单工具响应停在原生闭合token，减少闭合后继续生成/再次工具调用的机会；压缩影响Agent输入、回答与重复查询。这类行为优化与网络波动共同影响E2E，必须与纯kernel收益区分。旧3.92s估算表、旧ApxInf2.148s/TRT2.123s缺少本批同条件trace，不能用于新提交的逐项归因。

完整性能回归仍需：同一冻结Agent trace逐项ablation、同场TRT、采样语义下正确性、4K/8K/16K/32K/48K长上下文、动态/细粒度FP8的校准与精度验收。本次完成的是逐commit真实天气记录和有边界的profiler解释，不把未完成目标写成成绩。
