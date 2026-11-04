# Qwen3.6 模型基准运行说明

本工具只测驻留权重后的15个模型case，不运行AIOS。完整矩阵、精度与消融要求见 [任务拆解](MODEL_MICROBENCH_TASKS.md)。历史表仅作参考，不替代本次原始数据。

在Mac的本checkout根目录运行以下命令。下列构建目录是本任务独立runner；不要与其他任务共用输出二进制或recipe目录。

```bash
matrix_args=(
  --ssh-host thor-u
  --workdir /opt/data/dev/apxinf-qwen35-thoru-perf/devlocal/commit-audit/runner
  --context-binary /opt/data/dev/apxinf-qwen35-thoru-perf/target/release/examples/qwen35_context_bench
  --decode-binary /opt/data/dev/apxinf-qwen35-thoru-perf/target/release/examples/qwen35_mtp_bench
  --checkpoint /opt/data/models/Qwen3.6-35B-A3B-NVFP4
  --tuning-dir /opt/data/dev/apxinf-qwen35-thoru-perf/devlocal/commit-audit/runner/devlocal/model-microbench/recipes-bf16
  --env APXINF_QWEN35_CUDA_GRAPHS=0
  --env APXINF_QWEN35_VERIFY_GRAPHS=0
)
mkdir -p devlocal/model-microbench/runs
python3 scripts/bench_qwen35_model_matrix.py "${matrix_args[@]}" --plan
python3 scripts/bench_qwen35_model_matrix.py "${matrix_args[@]}" --inspect \
  > devlocal/model-microbench/runs/identity.json
```

故障与恢复原因、重启后核对流程见 [恢复记录](MODEL_MICROBENCH_RECOVERY.md)。

先在 Thor 本机完成全部编译，再回到上述 matrix 流程。使用专用编译 launcher，
不要交错 native 编译和大 HugeTLB 池下的推理，也不要在外层重复获取同一公共锁：

```bash
python3 scripts/build_qwen35_thor_benchmarks.py --plan \
  --target-dir /opt/data/dev/apxinf-qwen35-thoru-perf/target
python3 scripts/build_qwen35_thor_benchmarks.py --execute \
  --target-dir /opt/data/dev/apxinf-qwen35-thoru-perf/target
```

默认仅 plan：不取锁、不 sudo、不改池、不编译、不写 receipt。`--execute` 只允许
Thor Linux aarch64 普通用户；持有固定 `/tmp/apxinf-gpu.lock` 后，若
`HugePages_Total - HugePages_Free + HugePages_Rsvd > 2048` 则拒绝缩池。
仅用已缓存授权的 `sudo -n` 设置并读回2048页（绝不清零），核对 Total/Free/Rsvd
与 `MemAvailable >= 24 GiB`，才以当前普通用户启动 release、`qwen35-cuda`、
Cargo jobs2 / NVCC parallelism2，一次编译两个 benchmark。

默认编译结束保留小池；仅最后一次成功编译可显式加 `--inference-pages N` 请求
后续推理池。该请求只在 owned PGID 的 Cargo 和孙进程已全部结束/reap 后执行，
增长量必须不超过编译后 `MemAvailable - 8 GiB`；root pool writer 还会在增长前后
再次检查至少8GiB普通内存余量，避免扩池吞掉全部普通内存。
读回失败在同一授权操作中回到2048；权限过期则失败并保留小池，不自动恢复大池。
构建失败不触发扩池。不能证明终止时 guard 落盘失败 receipt 并保持进程持锁，
不显式 unlock；先检查 receipt 中的 owned PGID，确认其结束后再处理该 guard。
正常 TERM/INT 会延后至固定 pool 操作记录完毕。外部 SIGKILL 或 OOM 强杀 guard
不在这项保证内：sudo pool helper 可能继续，而 sudo 不保证继承 Cargo 的锁 FD；
此时必须人工核实 pool 与残留进程，不能把不完整 receipt 当作成功/清理证明。

receipt 默认在 `devlocal/model-microbench/safe-build/<run>/`，记录源码、实际编译
argv、前后内存/池、退出码、清理证明及 Cargo JSON 返回的实际 aarch64 ELF SHA256。
源码、构建日志和实际ELF哈希必须对应；inspect自身不能证明旧二进制来自当前源码。

先独立调优，再启动新的正式进程：

```bash
python3 scripts/bench_qwen35_model_matrix.py "${matrix_args[@]}" \
  --metadata devlocal/model-microbench/runs/identity.json \
  --run-mode tune --output devlocal/model-microbench/runs/tune-01
python3 scripts/bench_qwen35_model_matrix.py "${matrix_args[@]}" \
  --metadata devlocal/model-microbench/runs/identity.json \
  --run-mode formal --warmups 2 --repeats 5 \
  --output devlocal/model-microbench/runs/formal-01
```

调优时延永远不是正式成绩。正式运行要求已有可恢复recipe、native strict guard生效且recipe前后不变；缺失时保存失败、另行调优，然后用新输出目录重新开始。固定直接内核没有recipe概念，不伪造命中记录。默认target BF16、draft FP8/scale1、NVFP4 MoE、greedy固定depth3；这些是当前基准身份，并不证明其他sampling/MTP路径等价。

可用 `--cases cold-i2k,append-kv48k-i4k,decode-kv48k-o1k` 选代表场景。只选部分case不会获得“完整15组通过”。Cold为空KV且权重已加载；append只计新增4096输入；Decode必须输出1024 token，首token归TTFT，其余1023间隔归Decode。

所有加载、调优、预热、正式测试和profiler统一使用 `/tmp/apxinf-gpu.lock`。runner等待其他持锁任务，不停止其进程。其子进程完全退出前不释放锁；不要在父层再开同一路径的独立flock然后让runner二次取锁。HugeTLB、功耗和频率策略由机器负责人在持锁时准备，runner只记录。该Thor平台不能将HugeTLB池清零后假定CUDA仍可初始化。

独立profiler采集示例：

```bash
python3 scripts/bench_qwen35_model_matrix.py "${matrix_args[@]}" \
  --metadata devlocal/model-microbench/runs/identity.json --case cold-i2k \
  --run-mode profile --env APXINF_NVTX=1 \
  --profile-dir /opt/data/dev/apxinf-qwen35-thoru-perf/devlocal/commit-audit/runner/devlocal/model-microbench/nsys-01 \
  --output devlocal/model-microbench/runs/profile-01
```

profiler使用CUDA profiler API圈定测量，不包含权重加载、warmup或初始KV构建；生成真实非空`.nsys-rep`后保存SHA256并独立导出stats CSV。NVTX仅标注实际有range的路径：没有Prefill内部标记时仍可按kernel/API分析，不能伪造阶段拆分。默认正式运行不启用NVTX/REPORT。profile/audit的日志和工具开销不作为正式性能。

若默认重复输出一致性检查失败，原始失败必须保留。只有明确的探索运行才添加 `--allow-repeat-divergence`：它保留全部轮次、token digest和首差异窗口，结果始终为 `accuracy_status=unresolved`、`performance_accepted=false`；表中只能称诊断中位数。不能用该选项绕过算子精度门禁或MTP正确性验收。

每次只变一个候选，采用相同测量policy、输入、权重、精度及调优方案做父版本/候选对照；源码或ELF在运行中改变会使结果失效。GPU锁是协作锁，不能约束不遵守该锁的外部进程，因此还须核对机器实际占用。


## Decode策略消融

显式选择Decode子集，例如 `--cases decode-kv4k-o1k,decode-kv48k-o1k`。使用上述common参数、metadata与新的output目录，分别运行：

- `--mode scalar`：禁用MTP，取得每case的TPOT成本。
- `--mode mtp --mtp-strategy fixed --mtp-depth 1`，以及depth2、depth3。
- `--mode mtp --mtp-strategy adaptive --scalar-calibration /ABS/scalar-run/summary.json`。

Adaptive要求保留scalar的summary、metadata和逐case原始log；每组至少2轮预热、5轮测量，完整1024输出/1023间隔。工具重验原始时间与摘要哈希、源码/实际ELF/权重、精度及运行环境，再从各case逐轮TPOT取中位数，不使用运行时内置19.42ms默认成本。

校准时使用的所有recipe必须仍存在且字节不变；可以增加新的verifier recipe。目标snapshot预检与每case取得GPU锁后的清单都会复验该子集。源码、二进制、已有recipe或精度方案改变后需要重新校准。

Strict和显式diagnostic只能各自使用同policy的scalar校准，不能互相升级。diagnostic adaptive始终保留完整输出差异证据并标记准确性未解决、正式性能未验收。profile/audit可以在NVTX0的scalar成本上启用NVTX1，但记录两侧observer环境；formal仍严格匹配NVTX。额外GPU同步计时、kernel和精度环境差异不会被忽略。

这是测量工具能力；42项CPU合同检查通过，GPU策略矩阵尚待执行，见[验证记录](model-microbench/evidence/decode-policy-runner.json)。profiler插桩会改变adaptive的观测成本，其吞吐不替代正式结果。
