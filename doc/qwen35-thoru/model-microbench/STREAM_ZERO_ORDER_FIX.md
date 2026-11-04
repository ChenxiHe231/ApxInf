# Qwen35 初始化与计算的 stream 顺序修复

四个 `zeros(ctx, ...)` helper原先调用 `CudaBuffer::alloc_zeros`：它在legacy default stream上执行 `cudaMemset`；模型计算stream通过 `CUDA_STREAM_NON_BLOCKING` 创建。两条stream之间没有这里所需的先行依赖，初始化清零可能覆盖已经计算的输出。

本改动仅将attention、target decoder、GDN/MoE scratch、MTP scratch初始化改为 `alloc_zeros_async(..., ctx.stream())`。同一计算stream按顺序完成初始化和后续消费；GDN辅助投影stream继续通过既有 `normalized_ready` / `projections_ready` 事件依赖主stream。没有更改dtype、scale、数学、内核、全局CudaBuffer API或精度阈值，也没有新增device-wide barrier。

## 实测证据

| 检查 | 修复前 | 修复后 |
|---|---|---|
| GDN短独立CPU参考及状态suite，5个独立进程 | 通过/失败/通过/失败/通过；失败输出出现随机连续清零区域 | 连续5轮通过 |
| 2K Prefill：2轮预热+5轮正式 | 首轮正式输出变化而中止，无有效中位数 | 7轮输出一致；正式中位数442.700 ms |
| BF16 attention short/long | 单独数值检查可通过，初始化存在同类写法 | 40个几何×padded/exact-live两种绑定通过；页界与future poison检查通过 |
| 完整Prefill attention shape | 先建立非退化的固定V fixture | 9个完整GPU shape通过；仅预声明查询行有独立f64参考，不称全输出数学验收 |
| GDN long状态与后续Decode | 原阈值未变 | 9组GPU逐token调度对照通过；不是独立长序列数学参考 |

2K正式五轮为441.792、442.700、457.256、439.328、449.979 ms，target BF16、draft FP8/scale1、NVFP4 MoE、固定depth3、Graph关闭、native strict recipe且无autotune。首token计入Prefill，输出长度1，权重加载和初始KV构建均不计入。

这是正确性修复，**没有可计算的修复前正式中位数，因此不报告加速比例**。本结果也不证明完整15组、Graph或MTP rollback已经验收。历史用户数据的468 ms不能在缺少相同原始配置时直接作为本改动的父版本。

源码/ELF/模型/recipe身份、逐轮值和测试日志哈希见 [验证记录](evidence/stream-zero-order.json)。P0工具状态中的 `repeat_consistent` 仅表示输出重复一致，不等于完整P2数值认证。原始日志和Nsight产物保留在checkout的 `devlocal/model-microbench/` 中。
