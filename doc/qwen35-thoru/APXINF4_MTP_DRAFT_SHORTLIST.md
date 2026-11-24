# Final 上的 MTP draft 词表裁剪验收

本报告记录旧 Final 的真实请求验收；最新 Final 的完整长上下文逐提交复测见[长上下文验收](APXINF4_MTP_LONG_CONTEXT.md)。

本提交在 `1f605865` 上复用已有 W4A16 算子，自动将符合条件的 draft head
从 248320 列缩小到 `[0, 131072)` 和 `[247808, 248320)` 两个视图，共 131584 列。
主模型始终执行完整词表验证。权重不复制、不重新量化；缓存绑定设备和权重、
块 scale、全局 scale 的地址。输入或已提交 token 超出集合后，该上下文使用完整
draft head；reset、append、部分接受、失败重试、深快照恢复和 A16 权重重新绑定
均参与验证。新增路径自动选择，没有用户开关或优化环境变量。

## 端到端正式结果

Thor-2，GPU 固定 1530 MHz（可选最高 1575），CPU 2601 MHz、EMC 4266 MHz。
NVFP4、动态 FP8 target KV、固定 scale1 FP8 draft KV、greedy、固定 MTP depth3。
13 道历史真实问题，最大输出 512 token，保留 EOS；其中一道实际生成 383 token。
每题每个进程预热 2 次、测量 5 次，以 A→B→B→A 运行：260 条正式请求、104 条预热。
以实际 token 总数除以总时间，不平均单请求吞吐。

| 问题组 | Final→本提交：含 prefill tok/s | 含 prefill 提升 | Decode tok/s | Decode 提升 | 接受率变化 |
|---|---:|---:|---:|---:|---:|
| 原 5 题 | 106.01→109.80 | +3.57% | 108.59→112.57 | +3.67% | +0.2132 个百分点 |
| 另外 8 题 | 120.27→123.95 | +3.06% | 123.73→127.65 | +3.17% | −0.4387 个百分点 |

正向/反向的端到端提升分别为原 5 题 +3.50% / +3.63%，另外 8 题 +2.88% / +3.24%。
两组分别统计，不把百分比相加。draft 策略可以改变接受率和工作量；全部计分及预热
的主输出 digest 与 Final 一致。逐题均值中 12 题提升，155 题 −0.09%，其两轮为
−0.41% / +0.23%，不宣称全输入加速。

这里的端到端时间包含输入 prefill、生成和回调，排除权重加载、分词、HTTP；
Decode 使用首 token 之后的 N−1 个回调间隔。没有前缀复用。

## 配置与正确性

两侧固定原有最快有效配置：`APXINF_QWEN35_MTP_SPLIT_KV=1`、
`APXINF_QWEN35_MTP_DRAFT_KV_SUFFIX_SYNC=1`、
`APXINF_QWEN35_MTP_KV_ONLY_MATERIALIZE=1`、
`APXINF_QWEN35_MTP_MATERIALIZE_SESSION=0`，以及 `APXINF_FA2_PACKED_GQA=0`。
这些是已有基线设置，不是本提交的开关。本提交没有读取旧 shortlist 环境变量。
早期清空这些配置得到的 head 默认配置计时仅作诊断，没有混入此处正式结果。

829 份 recipe 在正式测量前冻结，计时期间禁止在线调优和缺失 recipe 回退；
原 Final 的 736 份 recipe 字节全部保留，新增缺失输入/测试键在独立进程准备。
原有 14 个 checkpoint/tokenizer 文件和既有任务标记身份未变。3282 次频率采样
核验通过，正式 A/B 结束后频率策略恢复。

严格 GPU 测试的同输入裁剪投影余弦为 1.0，有限性、输入与权重只读、guard、
变输入 Graph 重放、集合边界、退出集合后的回退、事务状态和权重绑定检查通过。
相关投影与 remapped argmax 的 scoped memcheck 为零错误；同一完整状态测试在
过滤插桩下也通过。这里没有将余弦 1.0 称为所有 logits 逐位一致。
模型模块边界及 diff 检查通过。

## 收益解释

独立的成对 Nsight 采样使用同一题的 128-token 生成，两侧均为 39 次验证、117 次
draft 投影，输出和接受工作量相同。剔除一次目标 prefill 的单行 head 后，
每次逻辑 draft head 的可见 GPU kernel 时间由 1129.61 μs 降至 626.21 μs（−44.56%）；
新路径每次包含 prefix 和 tail 两个 kernel。此数是诊断，不是模型端到端提升，
也不把 kernel 耗时和当作关键路径占比。

实际运行的两侧 ELF 中 CUDA fatbin 完全一致，证明复用了现有算子，没有夹带
verify-head half2 修改。正式模型收益同时包含真实投影成本和 draft 接受变化的影响。

## 逐题端到端吞吐

| case | Final tok/s | 本提交 tok/s | 提升 |
|---|---:|---:|---:|
| 82 | 119.40 | 122.68 | +2.75% |
| 92 | 97.36 | 98.09 | +0.76% |
| 96 | 99.13 | 101.14 | +2.03% |
| 100 | 96.08 | 97.84 | +1.83% |
| 102 | 119.99 | 128.02 | +6.69% |
| 109 | 125.21 | 132.16 | +5.55% |
| 112 | 133.31 | 138.86 | +4.16% |
| 115 | 137.98 | 142.08 | +2.98% |
| 123 | 135.56 | 143.05 | +5.53% |
| 131 | 129.79 | 134.28 | +3.46% |
| 145 | 123.92 | 126.80 | +2.33% |
| 154 | 95.16 | 101.26 | +6.42% |
| 155 | 107.26 | 107.16 | -0.09% |

## 复现与边界

维护证据：[mtp-final-shortlist.json](model-microbench/evidence/mtp-final-shortlist.json)，
含每轮指标与输出 digest、产物/源码身份、recipe、精度/内存结果及成对采样。
原始输入和运行资产在 Thor-2 的 `/home/wwxq/devlocal/eglops-eval/checkout/devlocal/mtp-final-validation/`；
模型仍使用 `/home/wwxq/models/Qwen3.6-35B-A3B-NVFP4`，没有复制 checkpoint。
构建时私有 benchmark example 调用正式公共生成 API；提交不包含私有入口或测试准备覆盖层。

在保有上述冻结资产的 Thor-2 checkout 内复现正式 A/B（串行锁和时钟恢复由控制器管理）：

```sh
python3 devlocal/mtp-final-validation/scripts/bench_fast_requests.py \
  --name shortlist-repro-01 \
  --stages head-baseline,shortlist-final,shortlist-final,head-baseline \
  --cases 82,92,96,100,102,109,112,115,123,131,145,154,155 \
  --warmups 26 --repeats 5 --budget 512 --proposal-change
```

离线重新核验原始正式结果（不运行 GPU）：

```sh
python3 doc/qwen35-thoru/model-microbench/verify_mtp_final_retest.py \
  devlocal/mtp-final-validation --run shortlist-fast-formal-01 \
  --complete --fast-baseline
```

收益仅验收上述 13 题生成配置。长上下文 1K 输出、追加 prefill、最终 2 轮预热＋
10 轮合并矩阵及两个 20% 目标不由本轮结果建立。verify-head half2 尚无可重复增益，
没有合入；位置表/复制 Graph 也不在本提交中，需要单独的增量验收。
