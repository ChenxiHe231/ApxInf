# 逐版本天气回放发现的接入回归

这些是历史提交的实际失败，不把失败返回耗时当成有效天气性能；对应修复保留在原有后续提交中，不重写历史。

## `8f7fd233`：capture 中的 attention execution cache miss

天气调优与正式请求都得到空回答。正式请求内 6 次模型尝试均返回：

`generation failed: Qwen3.5 layer 3 attention: Attention execution cache miss during capture; prepare the same bindings first`

这里的 execution cache 绑定当前 Q/K/V 等设备资源，不是按 prompt token 身份匹配的 Prefix KV Cache，也不等同于磁盘 `.recipe` 算法选择记录。磁盘 recipe 已存在，不能证明当前 CUDA Graph 捕获所需的设备 bindings 已准备好。

后续 [`4868a580`](https://github.com/ChenxiHe231/ApxInf/commit/4868a580dbbe539b5a4122cebeb3812e2e67e672) 有两处直接相关的改动：`projection_session` 的 key 从 `(device, layer)` 改为 `(device, layer, tokens)`；在 `prepare_verify_phases` 中，live prefix 会变化的 full-attention mixer 放在外层 verifier session 之外执行，只捕获其固定地址的 MoE tail。GDN 仍可保留在对应的固定阶段中。

该修复版本已在调优阶段恢复天气回答；是否通过每一轮目标日期校验，仍以对应版本的正式记录为准。修复空输出不等于证明天气语义正确。

`4868a580` 的正式请求在输出部分正确天气文字后，第二次模型调用又返回 `Attention Recipe miss and fallback is disabled`，因此没有完整第二次 request profile。它仍不能算端到端成功；统计器同时检查所有模型调用的 SSE 错误，不能只根据最终文字中的温度判定通过。

原始错误、response SHA256 和重试次数见 [8f7fd233 实测记录](records/8f7fd233.md)。当前构建使用 `qwen35-cuda,server`；legacy `cuda` feature 下的 NVTX 标记不作为本批 GPU 阶段证据。

## `c3606422`：Prefix snapshot 与 live cache 的 capacity 不相同

新增 128-token 物理分块时，`validate_cache_pair` 增加了两个相等性检查：

```rust
source.capacity != destination.capacity
source.blocks != destination.blocks
```

但 snapshot 按可复用前缀长度分配，live session 按 prompt/output admission 分配，两者容量与物理块数本来就可能不同。实际天气请求返回 `cache copy requires matching dtype, layer, geometry and metadata`，无法发布可复用的前缀状态；多次重试仍为空回答。

[`6265b8f`](https://github.com/ChenxiHe231/ApxInf/commit/6265b8f) 删除了上述两条相等性约束，保留 layer、geometry、dtype、scale 的匹配，并检查被复制的 token range 同时位于两个缓存容量内。每行的连续布局相同，因此块数不同不阻止合法的前缀范围复制。

该代码差异直接解释为什么“支持 128-token 分块”与“Prefix Cache 接入成功”不能合并当作同一个验收项。对应正式请求和后续修复的实际结果分别记录，不用修复后成绩填回失败版本。
