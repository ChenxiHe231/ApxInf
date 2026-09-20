# CUDA L3 算子目录

本文档列出 `apxinf-cuda-new` 当前公开的模型无关 L3 semantic。模型代码必须匹配完整的
数学语义和张量契约，不能只按算子名称匹配；没有完全匹配的接口时，应记录 operator gap，
并按照 [`doc/adding-new-kernels.md`](../../doc/adding-new-kernels.md) 处理。

本文档只描述公共契约，不承诺具体 provider、candidate 或 autotune winner。所有算子都支持
eager execution 和 `prepare_with_session` → capture → replay。

## 共享约束

- 输入、输出、bias 和 scale tensor 必须位于当前 `CudaContext` 的 CUDA device。
- tensor 使用连续 row-major layout；输出 storage 不能与只读输入重叠。
- shape 非空，传入 native 层的维度不能超过 `i32::MAX`。
- policy 中的 workspace、graph-safe、deterministic 等字段只影响 candidate 合法性和
  recipe，不改变 L3 数学语义。
- `l3-operator` 注释是机器可读标记。单元测试会将其与各 operator family 注册的 Rust
  semantic metadata 比较，保证每个公开 semantic 恰好出现一次；新增 family 必须加入该集合。

## GEMM 共享契约

GEMM 使用 `A=[M,K]`、`B=[K,N]`。`alpha` 作用于 projection，最终结果除以有限正数
`output_scale`。`projection(A,B)` 按以下量化契约解释输入：

- `None`：A/B dtype 相同，且不是 E4M3 或 INT8。
- `Fp8UnitScale`：A/B 均为已包含预期缩放的 E4M3 tensor。
- `Fp8`：A/B 均为 E4M3；FP32 `row_scales=[M]` 和 `channel_scales=[N]`
  分别按 A 的行和 B 的列反量化。
- `W8A8`：A/B 均为 INT8，并使用相同形状的 FP32 row/channel scales；输出为 BF16，
  `K <= 131071`，且只适用于 `gemm` 和 `gemm_bias`。

量化发生在 API 调用之前；当前 L3 契约不包含动态量化。具体 spec 仍需至少一个已注册
candidate 支持。

<!-- l3-operator:gemm -->
### `gemm`

| 项目 | 契约 |
| --- | --- |
| Rust API | `ops::gemm(ctx, GemmArgs)` |
| 输入 | `A=[M,K]`，`B=[K,N]`；支持 `None`、`Fp8UnitScale`、`Fp8`、`W8A8` |
| 输出 | `Y=[M,N]` |
| 数学语义 | `Y = alpha * projection(A,B) / output_scale` |
| 限制 | W8A8 输出只能为 BF16；`WeightVersion` 可声明不可变权重并允许 prepare 阶段缓存内部预打包副本 |
| Reference test | `gemm_all_candidates_match_torch` |

<!-- l3-operator:gemm_bias -->
### `gemm_bias`

| 项目 | 契约 |
| --- | --- |
| Rust API | `ops::gemm_bias(ctx, GemmBiasArgs { gemm, bias })` |
| 输入 | `A=[M,K]`，`B=[K,N]`，`bias=[N]`；支持全部四种 GEMM 量化契约 |
| 输出 | `Y=[M,N]` |
| 数学语义 | `Y = (alpha * projection(A,B) + bias) / output_scale`，bias 按 M 维广播 |
| 限制 | bias dtype 与 projection dtype 相同；W8A8 输出只能为 BF16 |
| Reference test | `gemm_bias_all_candidates_match_torch` |

<!-- l3-operator:gemm_bias_gelu -->
### `gemm_bias_gelu`

| 项目 | 契约 |
| --- | --- |
| Rust API | `ops::gemm_bias_gelu(ctx, GemmBiasGeluArgs { gemm, bias })` |
| 输入 | `A=[M,K]`，`B=[K,N]`，`bias=[N]`；支持 `None`、`Fp8UnitScale`、`Fp8` |
| 输出 | `Y=[M,N]` |
| 数学语义 | `Y = GELU_tanh(alpha * projection(A,B) + bias) / output_scale` |
| 限制 | bias dtype 与 projection dtype 相同；GELU 使用 Torch reference 的 tanh approximation；不支持 W8A8 |
| Reference test | `gemm_bias_gelu_all_candidates_match_torch` |

<!-- l3-operator:gemm_geglu -->
### `gemm_geglu`

| 项目 | 契约 |
| --- | --- |
| Rust API | `ops::gemm_geglu(ctx, GemmGegluArgs { gemm })` |
| 输入 | `A=[M,K]`，`B=[K,2N]`；B 的前 N 列为 `B_gate`，后 N 列为 `B_up`；支持 `None`、`Fp8UnitScale` |
| 输出 | `Y=[M,N]` |
| 数学语义 | `Y = GELU_tanh(alpha*(A@B_gate)) * (alpha*(A@B_up)) / output_scale` |
| 限制 | B 的第二维为偶数；不支持带 row/channel scales 的 FP8 或 W8A8；candidate-specific packing 只能在内部完成 |
| Reference test | `gemm_geglu_all_candidates_match_torch` |

## Attention 共享契约

Attention 计算 `softmax(mask(scale * (Q @ K^T))) @ V`。Q/K/V dtype 相同，`scale`
是有限正数，默认值为 `1/sqrt(head_dim)`。Dense 和 KV-cache 支持 MHA、GQA 和 MQA，
且要求 `query_heads % kv_heads == 0`。

<!-- l3-operator:attention -->
### `attention`

| 项目 | 契约 |
| --- | --- |
| Rust API | `ops::attention(ctx, AttentionArgs)` |
| 输入 | `Q=[B,Tq,Hq,D]`，`K/V=[B,Tk,Hkv,D]`；Q/K/V 为相同的 F16 或 BF16；mask 为 `None` 或 `Causal` |
| 输出 | `Y=[B,Tq,Hq,D]`；普通输出与输入 dtype 相同，也支持 F16 输入写入 E4M3 |
| 数学语义 | dense scaled dot-product attention |
| 限制 | causal 要求 `Tk>=Tq`，query 对应 key 序列末尾位置；普通输出要求 `output_scale=1`；E4M3 存储 `round_to_e4m3(attention/output_scale)` |
| Reference test | `attention_all_candidates_match_reference` |

<!-- l3-operator:kv_cache_attention -->
### `kv_cache_attention`

| 项目 | 契约 |
| --- | --- |
| Rust API | `ops::kv_cache_attention(ctx, KvCacheAttentionArgs)` |
| 输入 | `Q=[B,Tq,Hq,D]`，`K_cache/V_cache=[B,key_capacity,Hkv,D]`；全部为相同的 F16 或 BF16；mask 为 `None` 或 `Causal` |
| 输出 | `Y=[B,Tq,Hq,D]`，dtype 与输入相同 |
| 数学语义 | query 对 cache 的前 `valid_key_tokens` 行执行 scaled dot-product attention |
| 限制 | `0<valid_key_tokens<=key_capacity`；causal 时 token i 位于 `query_start+i`，并要求 `query_start+Tq<=valid_key_tokens` |
| Reference test | `kv_cache_attention_all_candidates_match_reference` |

<!-- l3-operator:segmented_attention -->
### `segmented_attention`

| 项目 | 契约 |
| --- | --- |
| Rust API | `ops::segmented_attention(ctx, SegmentedAttentionArgs)` |
| 输入 | Q/K/V 均为 `[total_tokens,H,D]` 且 dtype 相同（F16 或 BF16）；device U32 offsets 与 `host_offsets` 内容一致 |
| 输出 | `Y=[total_tokens,H,D]`，dtype 与输入相同 |
| 数学语义 | 对 packed token 序列的每个 segment 独立执行 non-causal self-attention |
| 限制 | offsets 至少两个元素、单调不减、首项为 0、末项为 `total_tokens`；允许空 segment；不支持 causal 或不同 Q/KV head 数 |
| Reference test | `segmented_attention_all_candidates_match_reference` |

## 测试责任

catalog 测试只保证 semantic 不缺失、不重复，不能验证文字契约。新增 L3 semantic 还必须在
`src/ops/tests/l3_behavior.rs` 增加语义测试，并在
`src/ops/tests/precision/precision.rs` 增加独立 reference 的 all-candidate 数值测试。
测试需通过 `crates/apxinf-cuda-new/test-new.sh` 运行；普通根目录 `cargo test` 不会自动测试
本 crate。
