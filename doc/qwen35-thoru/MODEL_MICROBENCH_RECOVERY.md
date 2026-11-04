# Thor-U 模型基准故障与恢复记录

## 2026-11-04：编译期间普通内存不足

**已观察到系统 OOM，重启后 SSH 恢复。** 这是系统内存故障记录，尚无证据将其称为 GPU kernel hang 或 CUDA Xid。串口记录的 kernel 时间约为启动后 7684/7688 秒：

```text
Out of memory: Killed process 2419 (wireplumber)
Out of memory: Killed process 2417 (pipewire)
```

故障前在保留 21814 个 2 MiB HugeTLB 页（约 42.6 GiB）的情况下启动了候选版本完整 CUDA native 编译，普通可用内存此前约 13 GiB。HugeTLB 空闲页不能直接供普通编译进程使用；这次操作违反了编译与推理分阶段的约定，是本轮最明确的可避免诱因。没有完整 OOM victim/进程 RSS 时间线，不能据此断言某一个编译子进程的峰值内存，或将全部症状归因于某一 GPU 算子。

SSH 握手与串口登录随后卡顿，尝试通过 SSH 缩池没有执行成功的证明。不能将命令已发出等同于内存池已恢复。原基线和候选编译均中断，不补写为成功测试，不计入正式性能。

用户批准重启后，通过上海 Orin 的 `/dev/ttyUSB2` MCU 控制台，先读取本机 `soc` 帮助，再执行一次 `soc reset`。Linux 串口显示重新启动并进入登录提示；随后 `ssh thor-u` 返回新 uptime 约 25 秒，可用内存 43840676 KiB。复核时旧 Cargo/NVCC/基准进程均不存在。源码、已冻结 ELF 和此前完成的记录位于 `/opt/data`，继续保留。

重启后的默认 GPU 策略为 performance、上限 1530 MHz。测试前已在公共锁内恢复原基线的 simple_ondemand、最低 54 MHz、最高 1206 MHz，并读回验证。这个恢复操作不算性能优化。旧版本文档中的“将 HugeTLB 清零编译”已更正；本机曾做零页与 2048 页 CUDA context 对照，零页不能创建 context，2048 页可以。

机器时钟与本地记录可能存在差异：事故日期按工作站记录，服务端日志中的 wall clock 不单独作为事件先后顺序依据；同时保留 boot uptime 和启动输出。

## 后续固定流程

1. 所有编译、调优、推理和 profiler 共用 `/tmp/apxinf-gpu.lock`。核对实际进程，不能只依赖协作锁排除外部任务。
2. 使用 `scripts/build_qwen35_thor_benchmarks.py --execute`：编译池固定 2048 页，已用加保留页不得超过 2048，普通可用内存至少 24 GiB，Cargo/NVCC 并发均为 2。检测不满足则退出，不能继续强行编译。
3. 先编译全部版本并分别冻结 ELF/源码哈希。确认自有 Cargo 进程组和孙进程全部退出、被回收后，才扩推理池；始终保留至少 8 GiB 普通可用内存。不得穿插完整 native 编译与大池推理。
4. 每次重启后重读内存池、GPU governor/频率、磁盘空间及实际设备标识，按基线恢复并保留 readback。
5. 调优独立运行；正式测试只使用已有且可恢复的 recipe。中断批次归档，新批次使用新输出目录和 metadata。
6. 如果再次失联，先查物理链路和本机 SSH 实际配置，再检查代理/串口。用户已授权 GPU 故障时重启恢复，但仍需先记录症状、现存任务、执行方式以及恢复后的验证；不反复发送未经验证的复位命令。

Guard 正常 INT/TERM 会完成自有进程清理再解锁；外部 SIGKILL/OOM 在特权缩池操作期间不在完整保证内，必须人工核实内存池与残留进程。自动恢复不能代替故障证据。

原始证据在本任务 ignored `devlocal/model-microbench/recovery-current/`：`serial-mem-cont.log`、`authorized-mcu-reboot.log`、`post-mcu-ssh.log`；内存和时钟 readback、编译 receipt 位于 Thor runner 的 `devlocal/model-microbench/recovery-build/`。这些是本轮运行位置，不要求其他机器上存在相同绝对路径。
