# MTP 位置准备与状态复制 Graph 的增量验收

本报告记录旧 Final 的真实请求验收；最新 Final 的完整长上下文逐提交复测见[长上下文验收](APXINF4_MTP_LONG_CONTEXT.md)。

本提交的性能父版本是 **`8270500a`（draft 词表裁剪版）**，
不是原始 Final。位置表和复制 Graph 作为一组验收，不为内部两点分别分配收益，
也不将本组百分比与词表裁剪的百分比相加。

动态 FP8 的 1～4 行位置数据在 scratch 准备时生成，执行时选择只读视图；
BF16/static-FP8 和较大行数保留原有准备。Session 准备时捕获两个物理方向的
verify 状态复制 Graph，发布时与物理状态缓冲区一起交换，销毁时先于引用的
状态内存释放。保留 Final 已有的只读 GDN recurrent 源绑定，不重新引入 recurrent
初始化复制；其他阶段的完整复制语义保持原状。新增选择不读取环境开关。

## 正式端到端结果

Thor-2，GPU 固定 1530 MHz（可选最高 1575），CPU 2601 MHz、EMC 4266 MHz。
NVFP4、动态 FP8 target KV、scale1 FP8 draft KV、greedy、固定 MTP depth3。
与父提交相同的 13 题、最多 512 输出 token、保留 EOS、无前缀复用。
每题每个进程预热 2 次、测量 5 次，A→B→B→A：260 条正式请求、104 条预热。
吞吐按实际 token 总数/总时间统计，Decode 为首 token 之后的 N−1 个间隔。

| 问题组 | 父版本→本提交：含 prefill tok/s | 提升 | Decode tok/s | 提升 |
|---|---:|---:|---:|---:|
| 原 5 题 | 109.63→110.27 | +0.59% | 112.39→113.06 | +0.60% |
| 另外 8 题 | 123.70→124.39 | +0.56% | 127.35→128.08 | +0.57% |

原 5 题的正向/反向端到端提升为 +0.21% / +0.97%，另外 8 题为 +0.09% / +1.04%。
全部计分及预热的输出 digest、完整 MTP 工作量、接受率和验证轮数均与父版本一致。
829 份有效 recipe 和 checkpoint 身份与词表裁剪验收相同，无计时中调优或
recipe 缺失回退。3227 次频率采样核验通过，正式 A/B 结束后时钟策略恢复。
原有 MTP 配置两侧同样冻结，见[父提交配置说明](APXINF4_MTP_DRAFT_SHORTLIST.md#配置与正确性)。

为区分小收益与计时噪声，使用预先记录的固定 seed=20261124、10000 次成对
整组重复块重采样；两个 A/B 时间窗口内各重采样 5 块。条件性的 95% 区间为：

- 原 5 题：端到端 +0.19%～+0.98%。
- 另外 8 题：端到端 +0.25%～+0.86%。

此区间只衡量固定问题、顺序和时间窗口内的重复计时变化，不保证其他设备或
部署负载的收益。两组的正反方向和区间下界均为正才接受本次增量。

## 正确性与解释

组合产物独立编译。位置表在容量 128/2200/8200、行数 1～4 的首尾/变化位置上
与原始位置值精确一致，表只读，越界被拒绝。完整 target 状态协议覆盖物理
缓冲区交换、部分/完整接受、快照、恢复和拒绝边界；词表裁剪的完整数值/Graph/
状态测试也在组合产物上通过。相关 target kernel 与 CUDA API 内存/流生命周期
检查的 scoped memcheck 为零错误；不是整个 35B 模型所有 kernel 的插桩声明。

另有父版本/组合版的成对 Nsight 采样，实际调用计数收录在维护证据中。
本次父版本/组合版采样中，两侧均为39轮验证；组合版减少1209次CPU发起的异步复制和
39 次流等待，增加 39 次复制 Graph 调用。此计数说明调度机制变化，不能直接
折算为关键路径比例或模型提升。运行 ELF 的 CUDA fatbin 与父版本相同，
收益来自元数据准备及复制调度，未夹带 verify-head half2 改动。

## 逐题含 prefill 吞吐

| case | 父版本 tok/s | 本提交 tok/s | 提升 |
|---|---:|---:|---:|
| 82 | 122.84 | 123.62 | +0.63% |
| 92 | 98.46 | 98.81 | +0.36% |
| 96 | 101.07 | 101.54 | +0.46% |
| 100 | 97.45 | 98.02 | +0.58% |
| 102 | 127.65 | 128.62 | +0.76% |
| 109 | 131.53 | 132.33 | +0.61% |
| 112 | 138.40 | 139.18 | +0.56% |
| 115 | 141.88 | 142.40 | +0.36% |
| 123 | 141.79 | 142.63 | +0.59% |
| 131 | 133.19 | 134.13 | +0.71% |
| 145 | 126.82 | 127.33 | +0.40% |
| 154 | 101.15 | 101.99 | +0.83% |
| 155 | 107.41 | 108.03 | +0.58% |

## 复现与限制

维护证据：[mtp-final-transport.json](model-microbench/evidence/mtp-final-transport.json)。
两次提交使用的13份输入文件hash均收录在维护证据的共享输入清单中。
原始资产沿用 Thor-2 checkout 下的 `devlocal/mtp-final-validation/`，模型不复制。
父版本实际 product 源码与已提交源码 hash 相同；私有 example/test 覆盖层独立声明。
计时包含公共生成调用的 prefill、生成和回调，不包含权重加载、分词或 HTTP。

```sh
python3 devlocal/mtp-final-validation/scripts/bench_fast_requests.py \
  --name transport-increment-repro-01 \
  --stages shortlist-final,shortlist-transport-final,shortlist-transport-final,shortlist-final \
  --cases 82,92,96,100,102,109,112,115,123,131,145,154,155 \
  --warmups 26 --repeats 5 --budget 512

python3 doc/qwen35-thoru/model-microbench/verify_mtp_final_retest.py \
  devlocal/mtp-final-validation --run transport-increment-formal-01 --complete --fast-baseline
python3 doc/qwen35-thoru/model-microbench/paired_mtp_uncertainty.py \
  devlocal/mtp-final-validation --run transport-increment-formal-01
```

本组没有直接重新交错测量原始 Final→组合版的累计收益。长上下文 1K 输出、
追加 prefill、最终 2 轮预热＋10 轮合并矩阵及两个 20% 目标仍未由本轮结果建立。
