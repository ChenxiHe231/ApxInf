# Thor3：无需环境变量的 MTP 默认执行策略

本改动属于默认路径交付修正，新增性能收益计为 **0%**。此前完整三阶段
Benchmark 显式开启的三个 MTP 策略，现在由合格的 Thor/NVFP4/动态 FP8 KV
模型工厂自动启用：split-KV、draft KV 后缀同步、KV-only materialization。
该配置不读取三个旧选择变量，显式设为 0 也不能关闭这些策略；没有新增开关、
环境变量、近似计算或精度阈值。公开底层构造器及其他配置保留原有兼容行为。

## 实际验证

基于 `451b8163230848666687c3271347901b46c06b24` 的独立候选，实际重新构建
examples 和 libtest，沿用 native `9a4fa1e4…`。构建、验证输入和 ELF 的完整
身份在随附 JSON 中；不是把旧 ELF 改名为新版本。初次 libtest 构建在严格
GPU owner 守卫退出，原 builder 重试通过，原失败记录保留；这不是数值失败。

- 3 项纯单元检查通过：合格配置忽略旧变量、其他配置惰性读取、SM 边界。
- 实际模型工厂运行 LCG 4K、LCG 48K、eligible 4K，各输出 1024 token。
  每项分别清除全部 `APXINF_*`/`NVFP4_*`，及清除后仅设置三个旧变量为 0。
  6 个独立进程的完整输出和全部 MTP work 与冻结 C 版本一致。
- 清除的变量包含 fixed-depth；实际配置仍为 greedy、固定 depth3、NVFP4 MoE、
  target 动态 FP8 KV、draft FP8 scale1。模型使用同一有效 recipe，禁止调优。
- 同机 GPU 1530 MHz、CPU 2601 MHz、EMC 4266 MHz，运行后守卫与频率恢复通过。

## 性能检查与保留的负信号

初次顺序筛选为同一新 ELF 的 FLAGS1 → 清空变量，各 2 warm / 3 measure。
Decode 为 111.96858 → 109.80022 tok/s，约 **−1.94%**；含 prefill 为
106.89496 → 104.86458 tok/s。该负信号保留，未直接据此验收或调整代码。

随后使用同一 ELF、输入和资源，执行清空变量 → FLAGS1 → FLAGS1 → 清空变量，
四个独立进程各 2 warm / 5 measure。

| 对照顺序 | 清空变量较 FLAGS1 的 Decode | 含 prefill |
|---|---:|---:|
| 清空 → FLAGS1 | +0.0466% | +0.0284% |
| FLAGS1 → 清空 | +0.2567% | +0.2929% |
| 每环境 10 次测量合并 | +0.1522% | +0.1613% |

合并 Decode：FLAGS1 110.75436、清空变量 110.92293 tok/s；TTFT 分别
445.24218、443.78171 ms。吞吐按 `1023 / mean(Decode 秒数)`、含 prefill 按
`1024 / mean(请求秒数)` 计算，未平均逐轮吞吐。每环境只有两个独立进程块，
不据此作统计显著性或新增加速结论。

两种相反顺序未复现初次回退，且源码选择策略、完整输出与 work 检查通过，
据此接收本次默认策略交付修正。初次差异的原因仍未确定。原三阶段 15-case
N10 表保持其 FLAGS1 协议，不改称完整无变量矩阵；本项不增加 20% 目标进度。

## 证据与复现

[精炼证据 JSON](evidence/default-mtp-profile-thor3.json) 保留实际构建身份、
原负信号、两组反向对照、未舍入样本和原始归档 SHA。

实际入口为 `devlocal/final-plus20/mtp-next/gate_mtp_candidate.py` 的
`default02-units-01`、`default02-strict-01`、`default02-screen-01`，以及
`devlocal/final-plus20/scripts/compare_default_env_baab_thor3.py` 的
`default02-baab-01`。GPU 作业只由统一调度器持有双锁执行。
raw 归档分别为 `default02-delivery-evidence-01.tar.xz` 与
`default02-baab-evidence-01.tar.xz`，位于 `devlocal/final-plus20/results/`。

与最新远程 Final 合并后的新 native/recipe 身份和必要模型桥接仍需单独检查；
本记录不替代该集成验证。
