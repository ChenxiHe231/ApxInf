# ApxInf `Final` 源码交付与部署

这次交付的是 [ApxInf `Final` 分支](https://github.com/ChenxiHe231/ApxInf/tree/Final)的**完整源码**。接收方在自己的 Jetson Thor 上编译程序、为本机生成 AutoTune Recipe，再连接自己的 AIOS 镜像。旧 `68b12f23` 交付包的二进制和 Recipe 不属于这次交付，不要混用。

模型权重和 AIOS 镜像由接收方提供。以下命令针对 `Qwen3.6-35B-A3B-NVFP4`，服务使用 NVFP4 MoE、动态 FP8 KV cache、MTP 和 CUDA Graph。请记录实际 Git 提交、模型文件、CUDA／驱动、GPU 型号和 Recipe 目录；不同机器重新编译与调优后的性能不能直接视为旧 Thor-U 的测量结果。

## 1. 拉取源码与检查环境

在接收方的 Jetson Thor 上执行。工作目录应放在空间充足的数据盘；下面以 `/opt/data` 为例，可改成该机器实际可写的数据盘。本指南**不修改 HugeTLB 页数，也不锁定 GPU 频率**；使用该机器已有的资源配置。

```bash
git clone --branch Final --single-branch https://github.com/ChenxiHe231/ApxInf.git
cd ApxInf
git rev-parse HEAD
git status --short

export PATH=/usr/local/cuda/bin:$HOME/.cargo/bin:$PATH
export CUDA_HOME=/usr/local/cuda
export WORK_DIR=/opt/data/apxinf-final-work
export MODEL_DIR=/opt/data/models/Qwen3.6-35B-A3B-NVFP4
export RECIPE_DIR=$WORK_DIR/recipes
export CARGO_TARGET_DIR=$WORK_DIR/target
mkdir -p "$RECIPE_DIR" "$CARGO_TARGET_DIR"

uname -m                 # aarch64
rustc --version
cargo --version
nvcc --version
test -f "$MODEL_DIR/config.json"
test -f "$MODEL_DIR/hf_quant_config.json"
```

还需安装可用的 CUDA 驱动、C++17 编译器、Python 3 和 `flock`。模型目录必须是上述 **Qwen3.6** 权重；不要换成名字相近的 Qwen3.5 权重。Recipe 按本机 GPU 架构、CUDA／cuBLAS 版本及构建身份匹配：换机器或工具链时，先用独立的空目录重新调优，不要把旧包的 Recipe 直接复制过来。

## 2. 从源码编译

先停止本机其他 ApxInf 推理或调优进程。下面两个编译目标分别是模型 HTTP 服务和 Prefill／Decode 基准程序：

```bash
export CARGO_BUILD_JOBS=2 APXINF_NVCC_PARALLELISM=2
cargo build --locked --release -p apxinf --no-default-features \
  --features qwen35-cuda,server --bin apxinf
cargo build --locked --release -p apxinf-model --no-default-features \
  --features qwen35-cuda --example qwen35_mtp_bench

test -x "$CARGO_TARGET_DIR/release/apxinf"
test -x "$CARGO_TARGET_DIR/release/examples/qwen35_mtp_bench"
ldd "$CARGO_TARGET_DIR/release/apxinf" | grep 'not found' && exit 1 || true
sha256sum "$CARGO_TARGET_DIR/release/apxinf" \
  "$CARGO_TARGET_DIR/release/examples/qwen35_mtp_bench"
```

保留 `git rev-parse HEAD`、两个 SHA256 和编译日志，后续复测必须指向同一源码与产物。首次 CUDA 编译可能较久；不要在同一 GPU 上并行运行调优或基准。

## 3. 在本机 AutoTune 并保存 Recipe

**调优是独立的一轮准备工作，不计入正式推理时延。** 先用同一服务脚本启动调优模式；Recipe 会写入绝对路径 `$RECIPE_DIR`：

```bash
AUTOTUNE=1 HOST_IP=127.0.0.1 PORT=48119 \
  bash scripts/run_qwen35_prefill_range_server.sh \
  "$CARGO_TARGET_DIR/release/apxinf" "$MODEL_DIR" "$RECIPE_DIR"
```

在另一终端等待 `curl -f http://127.0.0.1:48119/health` 成功，然后通过 HTTP 发送代表性的**真实 AIOS 请求**。至少覆盖实际 system prompt、工具 schema、多轮会话、短／中／长输入和预期输出长度；仅调优一句短问答无法覆盖真实业务形状。可先用下面的请求检查链路，再回放自己的 AIOS 请求：

```bash
curl -sS http://127.0.0.1:48119/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.6-35b-a3b","messages":[{"role":"user","content":"上海明天适合散步吗？"}],"temperature":0,"max_tokens":32}'
```

回放结束后停止调优服务，确认 Recipe 已生成，并为目录保存校验清单：

```bash
find "$RECIPE_DIR" -maxdepth 1 -name '*.recipe' | wc -l
find "$RECIPE_DIR" -maxdepth 1 -name '*.recipe' -print0 \
  | sort -z | xargs -0 -r sha256sum > "$WORK_DIR/recipe-sha256.txt"
python3 scripts/build_gemm_range_index.py "$RECIPE_DIR" --operator gemm
python3 scripts/build_gemm_range_index.py "$RECIPE_DIR" --operator attention
```

索引只加快查找，实际使用时仍会重新验证 Recipe 的完整 key 和配置。`Final` 对兼容的 Prefill 形状可复用已有 Recipe；新形状在正式服务中可以使用安全回退路径。因此不必为 1 到 20K 的每一个精确 token 数都单独调优，但**回退路径的时延可能不同于已调优路径**。若要比较某个形状的最佳性能，应先用该形状的真实请求调优，再在关闭 AutoTune 后测量。

## 4. 关闭 AutoTune，部署模型服务

用刚才的同一个 Recipe 目录重新启动；**正式服务不带 `--autotune`**：

```bash
AUTOTUNE=0 HOST_IP=0.0.0.0 PORT=48119 \
  bash scripts/run_qwen35_prefill_range_server.sh \
  "$CARGO_TARGET_DIR/release/apxinf" "$MODEL_DIR" "$RECIPE_DIR"
```

检查服务和一次实际生成：

```bash
curl -f http://127.0.0.1:48119/health
curl -f http://127.0.0.1:48119/v1/models
curl -sS http://127.0.0.1:48119/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.6-35b-a3b","messages":[{"role":"user","content":"用一句话介绍上海。"}],"temperature":0,"max_tokens":32}'
```

在接收方自己的 AIOS 镜像中，把模型地址配置成 `http://<Thor-IP>:48119/v1/chat/completions`，模型名为 `qwen3.6-35b-a3b`，并确保 AIOS 所在机器能访问 Thor 的端口。AIOS 的 Harness、MCP、ASR、TTS 和网络耗时不属于模型基准时间。启动脚本使用 `APXINF_REQUIRE_RECIPES=0`：正式请求不在线调优；未命中 Recipe 时走受支持的回退路径，避免直接以 Recipe 缺失终止业务请求。不要把回退结果作为已调优性能成绩。

## 5. 自测与性能复现

先用不同内容、不同长度的真实请求检查 HTTP 200、非空生成及服务日志，再测时延。`Final` 的一轮功能检查曾在 Thor-3 上让 20 条 41–19,938 input tokens 的请求全部生成成功；这只证明当时的短输出功能，不保证另一台机器的速度或长输出质量。

需要测 **4K 输入、256 token 输出**时，先用基准程序对相同形状做一次独立调优，然后关闭调优执行 2 次预热、5 次计时：

```bash
export APXINF_QWEN35_CUDA_GRAPHS=1 APXINF_QWEN35_GRAPH_GDN=full
export APXINF_NVFP4_A16_WMMA=1 APXINF_NVFP4_A16_WMMA_TILE_N=128
export APXINF_NVFP4_A16_WMMA_MIN_ROWS=4 APXINF_NVFP4_A16_M1_STAGE_ALL=1
export APXINF_FA2_PACKED_GQA=0 APXINF_QWEN35_W4A16_SHARED=0
export APXINF_QWEN35_MTP_FIXED_DEPTH=3 APXINF_QWEN35_MTP_SPLIT_KV=1
export APXINF_QWEN35_MTP_DRAFT_KV_SUFFIX_SYNC=1
export APXINF_QWEN35_MTP_KV_ONLY_MATERIALIZE=1
export APXINF_QWEN35_CHUNK_LONG_PREFILL=0
export APXINF_QWEN35_GDN_SEGMENT_LONG_PREFILL=0
export APXINF_QWEN35_GDN_PROJECTION_CHUNKS=0

BENCH=$CARGO_TARGET_DIR/release/examples/qwen35_mtp_bench
"$BENCH" "$MODEL_DIR" --device 0 --prompt-tokens 4096 --max-new 256 \
  --warmups 1 --repeats 1 --mode mtp \
  --qwen35-moe-activation nvfp4 --qwen35-kv-cache fp8_dynamic \
  --tuning-dir "$RECIPE_DIR" --autotune

"$BENCH" "$MODEL_DIR" --device 0 --prompt-tokens 4096 --max-new 256 \
  --warmups 2 --repeats 5 --mode mtp \
  --qwen35-moe-activation nvfp4 --qwen35-kv-cache fp8_dynamic \
  --tuning-dir "$RECIPE_DIR" | tee "$WORK_DIR/bench-4k-256.jsonl"
```

基准程序在非调优模式下要求已有 Recipe，缺失时应先补齐对应形状，而不是把调优或失败重试时间混入成绩。若测试 16K／48K 等长上下文，使用仓库中的 `qwen35_context_bench` 并分别对**相同**初始 KV、输入、输出和精度配置进行调优及正式计时；不要把不同形状、KV 精度或 MTP 配置的数字合并。正式报告至少附 Git HEAD、二进制 SHA256、Recipe 清单、权重身份、CUDA／驱动版本、GPU 型号、预热／计时次数与实际 token 数。
