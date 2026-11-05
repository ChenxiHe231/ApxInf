# Qwen3.6-35B-A3B NVFP4：动态 FP8 KV v1（thor-u decode）

## 版本内容

- 分支：`wangchaofan/dynamic-fp8-mr-decode`，基线 `2ea382e`
  （动态 FP8 KV v3 + `codex/qwen35-model-perf-matrix` 合并）。
- v1 = v3 动态 FP8 KV（per-token、per-KV-head，FP32 scale，K 在 RoPE 之后量化）
  **加上新的多行 verifier 注意力 kernel**：
  - `dynamic_mr_kernel<kSplit,kRows>`（`qwen35_attention.cu`），MTP verifier 的
    1–4 行 causal query 共用一次 K/V 读取；
  - 动态 decode spec 升到 version 3，新增 `rows` 字段，`APXINF_QWEN35_DYNAMIC_DECODE_MAX_ROWS = 4`；
  - `attention_backend.rs`：verifier 的 tokens ≤ DRAFT+1 时走直接动态 decode；
  - 单元测试 `dynamic_multi_row_matches_single_row_decode`。
- 不变的部分：静态 FP8 路径、draft 层（静态 fp8，scale 1.0，旧的单行 kernel）、
  非注意力部分、`APXINF_QWEN35_ATTENTION_MAX_SPLITS = 32`。

## 测试配置

- 机器 thor-u，GPU 最低频率锁在 1206 MHz（2.0 基线是在 54 MHz 下测的，
  所以和 2.0 只能比量级）；测完把 `min_freq` 恢复成 54000000。
- fixture：decode-{4,8,16,24,32,48}k，生成 1024 token，greedy，MTP 固定深度 3
  （`qwen3.6_moe_adaptive_mtp`，4 行 verifier）。
- 2 次 warmup + 5 次重复，取中位数；`--allow-repeat-divergence`；
  调优和测量分成不同进程，测量时不 autotune；recipes 1542 条，改动 0 条。
- 静态和动态用同一个二进制（sha256 前缀 `d7da9b2e31d5`）、同样的环境变量；
  命令行只有 KV 模式和 recipe 目录不同：

```
APXINF_QWEN35_CUDA_GRAPHS=1 APXINF_QWEN35_GRAPH_GDN=full APXINF_NVTX=0
APXINF_NVFP4_A16_WMMA=1 APXINF_NVFP4_A16_WMMA_TILE_N=128
APXINF_NVFP4_A16_WMMA_MIN_ROWS=4 APXINF_NVFP4_A16_M1_STAGE_ALL=1
APXINF_FA2_PACKED_GQA=0 APXINF_QWEN35_W4A16_SHARED=0
APXINF_QWEN35_MTP_FIXED_DEPTH=3
```

## 结果（tok/s，中位数）

| 上下文 | 2.0 静态 | 静态 fp8 | 动态 fp8 v1 | 动态/静态 | 静态 ms/轮 | 动态 ms/轮 | 静态 tok/轮 | 动态 tok/轮 |
|---|---|---|---|---|---|---|---|---|
| 4K  | 74.8 | 72.1 | 83.3 | 1.16 | 36.6 | 33.2 | 2.64 | 2.77 |
| 8K  | 65.7 | 66.7 | 78.9 | 1.18 | 40.3 | 34.4 | 2.69 | 2.71 |
| 16K | 57.9 | 57.9 | 74.0 | 1.28 | 47.6 | 36.6 | 2.76 | 2.71 |
| 24K | 48.8 | 49.0 | 71.3 | 1.45 | 55.9 | 38.8 | 2.74 | 2.77 |
| 32K | 41.9 | 42.7 | 66.3 | 1.55 | 66.2 | 41.0 | 2.83 | 2.71 |
| 48K | 33.1 | 35.1 | 61.0 | 1.74 | 83.5 | 45.4 | 2.93 | 2.77 |

每轮接受的 token 数两边接近（2.64–2.93 对 2.71–2.77；动态在 16K、32K、48K 反而略低），所以提速来自每轮耗时下降，
不是 MTP 接受得更多。

## 为什么更快（nsys，每轮 GPU 忙时，ms）

`APXINF_QWEN35_CUDA_GRAPHS=0` 下采的，只用来看 kernel 耗时，不是性能数据。

| 上下文 | 非注意力 | 静态注意力 | 动态注意力 | 静态合计 | 动态合计 |
|---|---|---|---|---|---|
| 4K  | ≈28 | 5.3  | 1.7  | 33.6 | 29.9 |
| 8K  | ≈28 | 8.9  | 2.8  | 37.5 | 31.2 |
| 16K | ≈28 | 16.1 | 4.8  | 44.3 | 33.2 |
| 24K | ≈28 | 24.5 | 6.9  | 52.6 | 35.2 |
| 32K | ≈28 | 34.4 | 9.1  | 62.5 | 37.4 |
| 48K | ≈28 | 51.4 | 13.5 | 79.4 | 42.0 |

- 非注意力部分两边一样（约 28 ms/轮），差别全部在注意力。
- 48K：静态 verifier 约 4.49 ms/次（推算值），`dynamic_mr_kernel` 0.73 ms/次，每轮 10 次；
  draft 层两边一样，1.59 ms/次 × 3.4 次 ≈ 5.5 ms/轮；`dynamic_kv_prepare_kernel` 0.03 ms/轮。
- 动态量化本身只会增加开销（写入时算 scale、每个 token-head 多读 8 字节 scale、
  每个 score/权重多一次 FP32 乘法）。提速来自新 kernel：
  4 行共用一次 K/V 读取（静态每行各读一遍）；48K 时 192 个 split，静态最多 32 个；
  16 字节向量加载，静态是 1 字节；e4m3→bf16 批量转换；寄存器预取；
  QK 阶段 8 个 warp 全用上，不做 Q padding；softmax 按 warp 并行；PV 用 Tensor Core。
  这几项各自贡献了多少没有拆开测。
- 长上下文提升更大：非注意力约 28 ms 固定，注意力随长度线性增长。
  静态注意力占比 16% → 65%，动态 6% → 32%。

## 验证情况

- `qwen35_dynamic_attention`：7 个通过，1 个 ignored（含多行和单行结果一致的测试）。
- 输出合理性：fixture 是真实的对话 prompt，动态模式日志里的 8-token 窗口是通顺的英文。
  **这不能代替精度验证。**
- **精度没有验证**：AIOS / token 一致性 / GPQA 都还没测。

- 二进制：编译不是逐字节可复现的（同一份源码连续编两次 sha 也不同），
  所以 v1 源码编出来的 sha 和 A/B 用的 `d7da9b2e31d5` 不一样；源码就是 A/B 时那份。
- `qwen35_attention_fp8`（静态路径）有 1 个测试失败：
  `qwen35_prefill_writes_fp8_cache_and_matches_scalar_attention`，输出 -0.785，期望 -0.973。
  v1 的 diff 没有碰静态 prefill 路径，但没有在基线 `2ea382e` 上重跑确认。

## 已知限制 / 待办

- 精度验证：在 bf16 / fp8 / fp8_dynamic 下跑 token 一致性和 AIOS mock。
- 这次 A/B 测的是 v3 加多行 kernel；纯 v3 没有在这套框架下测过。
- 多行 kernel 是 kernel 层面的优化，本身和量化方式无关。
- 还没研究的方向：draft 层注意力（约 5.5 ms/轮）、LM head（约 6.4 ms/轮）。
