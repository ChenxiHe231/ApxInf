# Reproduce a revision's weather record

以下命令用于 GPU 空闲后的补测。本轮只提交工具与历史记录，没有执行以下服务切换/测试。

## Build and identity

在干净的该 commit checkout 构建 release；Thor-U 不得同时运行推理与 CUDA 编译，使用已有部署文档的 HugeTLB 释放/恢复和双 job 限制。保留原服务 argv/environment，测完后恢复。

```sh
export PATH=/usr/local/cuda/bin:$HOME/.cargo/bin:$PATH
export APXINF_NVCC_PARALLELISM=2 CARGO_BUILD_JOBS=2
cargo build --release --bin apxinf --features qwen35-cuda,server
git rev-parse HEAD
git status --porcelain
sha256sum target/release/apxinf
```

先针对真实天气请求启用 --autotune 补齐 recipe，保存 recipe 文件列表与 SHA256，再关闭 autotune 正式启动。不得以 recipe miss/重试请求作为性能成绩。启动参数遵循各 commit 的 `--help`；早期版本尚不支持 FP8 时记录 BF16，并将配置差异列明。

## Proxy and collector

在 Mac 创建独立 run 目录。端口及 SSH alias 使用当前环境已验证的配置；脚本不自动修改 AIOS endpoint 或 SSH 隧道。

```sh
mkdir -p devlocal/commit-audit/run-<revision>
python3 scripts/qwen35_aios_proxy.py \
  --listen-host 0.0.0.0 --listen-port 18001 \
  --upstream-host 127.0.0.1 --upstream-port 18000 \
  --force-temperature 0 --normalize-time-context \
  --capture-dir devlocal/commit-audit/run-<revision>/captures \
  --log devlocal/commit-audit/run-<revision>/proxy.jsonl
```

此命令固定为 greedy MTP 对照；仅在 S4 后的压缩消融组增加 `--compress-amap-weather`。时间归一化是缓存实验条件，应同时记录真实动态时间的部署表现，不能把规范化测试等同于生产所有请求。
原始捕获可能有私有 Prompt/tool 数据，保留 devlocal，不上传鉴权信息、工具 key、完整业务 Prompt 或大体积二进制。维护台账仅保留脱敏计时、profile、摘要与哈希。

创建 `metadata.json`，所有值从实际运行现场核验：

```json
{
  "source_commit": "<40位完整SHA>",
  "binary_sha256": "<实际进程二进制64位SHA256>",
  "source_dirty": false,
  "case_target_date": "2026-11-04",
  "configuration": {"argv": [], "environment": {}, "temperature": 0},
  "recipes": {"path": "<绝对路径>", "manifest_sha256": "<清单SHA256>"},
  "hardware": {"host": "thor-u", "power_mode": "<实测>", "clock_policy": "<实测>"},
  "aios": {"image_digest": "<实测>", "endpoint": "<实测>"}
}
```

collector 会校验 SHA 字段格式，身份内容仍需操作者对应运行进程核验；不要只对某个磁盘文件计算 hash 后假定它就是运行中的 executable。

`case_target_date` 必须按测试用户时区明确给定。例如上海用户在 2026-11-03 询问“明天”，目标为 2026-11-04；UTC 2026-11-03 16:00 后上海已是 11 月 4 日，目标应为 11 月 5 日。跨本地午夜需新建 run 和 manifest，不能从被归一化的 UTC 日期静默推导目标日期。

```sh
python3 scripts/bench_qwen35_aios_weather.py \
  --aios-url http://127.0.0.1:8081 \
  --proxy-log devlocal/commit-audit/run-<revision>/proxy.jsonl \
  --capture-dir devlocal/commit-audit/run-<revision>/captures \
  --ssh-host thor-u --server-log /opt/data/<该版本服务日志> \
  --metadata devlocal/commit-audit/run-<revision>/metadata.json \
  --output devlocal/commit-audit/run-<revision>/result \
  --warmups 2 --repeats 10
```

单一独占顺序请求下按 log 增量对应 profile；出现并发、重试、缺少两次调用或日志轮转时拒绝自动生成完整阶段表。采样中文数字/同义天气描述可能被保守校验器判失败，应人工复核并保存理由，不能删除失败样本。
所有正式轮正确才输出正式分布表；失败时 summary 里的 successful_subset 指标只供诊断。

## Prefill/Decode profiler

正式 E2E 在不插桩的 release 服务上测量。诊断另启相同版本：记录 MTP draft/verify/commit 时长与接受率、D2H/同步次数、CUDA Graph replay 和 launch gap。已有环境变量：

- `APXINF_QWEN35_MTP_TIMING=1`
- `APXINF_QWEN35_VERIFY_LAYER_TIMING=1`
- `APXINF_QWEN35_DECODE_TIMING=1`
- `APXINF_QWEN35_ATTENTION_TIMING=1`

使用 `nsys profile --trace=cuda,nvtx,osrt --output=<run>/weather <相同服务启动命令>` 收集诊断 trace，重放同一个脱敏保留请求；用 `nsys stats` 对比 CUDA kernel、API 同步、memcpy 和 NVTX。使用 ncu 的带宽/算力指标需另跑，插桩耗时不用于正式 E2E。
单项归因必须 parent/new 同权重、同请求、同工具结果、同缓存热度和采样语义；报告原始/新增/复用 tokens 和 completion长度。不能把外部 AMap 波动或余项下降计为 Prefill/Decode kernel 收益。

CPU 工具回归：`python3 -m unittest discover -s scripts -p test_qwen35_aios_weather.py -v`。CUDA 全量编译、数值精度和各版本硬件验收仍需补测。
