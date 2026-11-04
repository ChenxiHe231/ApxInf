# Qwen3.5/3.6 CPU→CUDA 上传顺序与生命周期

Qwen35已有context使用non-blocking stream。旧`copy_cpu_to_cuda`调用默认stream的`cudaMemcpy`，而scratch初始化/后续kernel在context stream；两条stream之间没有所需依赖。受控Thor探针在context stream加入约50ms的有界host callback及其后的旧zero，再调用旧H2D并由同context stream的D2D冻结consumer值，实际得到0而非123；显式前后fence的control得到123。旧H2D在27µs返回时callback仍未完成。它复现的是上传与已有context写入的次序缺口，不是模型数值或性能测试。

CUDA12.8官方合同说明，同步形式的memcpy使用默认stream，pageable H2D返回时最终DMA可能尚未完成；legacy默认stream的隐式同步不包括non-blocking stream。因此API名中的“同步”不能代替跨stream依赖。[API同步合同](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-runtime-api/api-sync-behavior.html)、[stream同步合同](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-runtime-api/stream-sync-behavior.html)。

## 修复边界

新增model-neutral safe helper `transfers::to_cuda_on_stream(ctx, source: Tensor)`及`copy_cpu_to_cuda_on_stream(ctx, source: Tensor, destination)`。CPU Tensor所有权移入helper；同context stream排入`cudaMemcpyAsync`，成功返回前`ctx.synchronize()`。旧zero、上传和随后consumer因同stream而有序，临时CPU来源在完成前存活。固定destination保留shape/dtype/CPU-source检查，另要求device与context一致，并验证字节长度。提交或完成错误不能证明DMA已停止，错误路径保守保留source和destination owners，可能保留内存；错误不会被当作成功。

这是一条保守的host同步边界，会等待context已有工作；没有承诺无同步或性能提升。旧默认stream API保持原语义，其它模型不被迁移。Qwen35更新范围为token IDs、attention positions、Cake cu_seqlens、target verifier初始化、MTP shifted prefill/append/draft/commit输入及device weight上传；计算、精度、scale、MTP接受策略和KV/GDN协议不变。

## CUDA Graph范围

helper在GPU分配/提交之前拒绝capture：既检查Rust capture-retention边界，也查询context stream的`cudaStreamIsCapturing`。它不捕获临时host指针、不跳过copy、不静默回退。调用者应在capture外完成host上传，再捕获device消费者。

旧模型图路径若仍在capture中发起这类host上传，会显式返回unsupported；本次不认证全部旧model graphs或graph replay。实际capture回归覆盖公开Rust `capture`包装下的拒绝、destination不变及退出capture后恢复上传；raw CUDA自行开启capture的非零status分支未单独执行。当前模型基准保持graphs0，完整图适配/性能另行验收。

## 已执行证据与限制

[H2D维护证据](model-microbench/evidence/h2d-owned-stream-gate.json)绑定原始源码、ELF、receipt及日志SHA：

- controlled race旧路径0、双fence control123；旧无pending-zero复用60轮未见差异仅是该fixture观测，不证明旧API对任意流安全。
- 原始guard编译的13文件源码归档已逐SHA核验，抽取的10个H2D代码postimage与归档逐byte相同；实际下载的两个测试ELF同时匹配Cargo输出、build receipt和执行receipt，源码→ELF→测试证据链完整。
- 新helper三个显式GPU测试通过：延迟initial-zero后必须为123；临时source与1/16/4096元素各20轮复用精确相等；capture拒绝及恢复。一个CPU验证测试通过，仅使用合法空CUDA metadata检查shape/dtype/device/source条件，不调用CUDA。
- 原13文件组合source下的两个模型benchmark已实际guard构建成功；其中还含独立MoE ordered改动。这里提取的H2D-only补丁尚未单独重编译，组合构建记录不被扩写成单独H2D的完整模型精度或速度验收。

本变更不包含MoE ordered、后加P2 EOF协议测试、P0计时或其它优化。真实模型、长上下文、图与性能仍需各自门禁；本证据只认证已列出的上传次序、临时来源完成和边界拒绝合同。
