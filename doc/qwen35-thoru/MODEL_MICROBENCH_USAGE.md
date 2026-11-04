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

构建命令为 `cargo build --release -p apxinf-model --features qwen35-cuda --example qwen35_context_bench --example qwen35_mtp_bench`。在Thor执行时先设置CUDA/Cargo的PATH，并使用公共GPU锁。源码、构建日志和实际ELF哈希必须对应；inspect自身不能证明一个旧二进制来自当前源码。

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
