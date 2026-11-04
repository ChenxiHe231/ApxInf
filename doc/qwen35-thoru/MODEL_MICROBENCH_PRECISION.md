# 15 个模型基准的精度与测量门禁

状态：起点为 `f267f5ae`。现已在同stream初始化修复后执行prepared-BF16短/长attention、9个完整Prefill shape的选定行独立参考、短GDN独立数学参考及9组长GDN调度/状态对照；见[实际验收记录](model-microbench/evidence/precision-gates.json)。另已通过 same-prepared BF16→FP8 存储门禁，见[FP8 存储证据](model-microbench/evidence/fp8-prepared-storage-gate.json)；这不认证 FP8 attention 或模型质量。完整P2仍未完成。P1 可采集“当前版本表征”，但未过本门禁的数值路径不宣称完成 P2、也不凭吞吐发布新的算子优化。所有新增阈值必须在候选之前声明、独立评审并冻结；不能在候选失败后静默放宽。

## 先区分实际执行路径

- Target KV 为 BF16 时，`attention_backend.rs` 走分解的 Q/K norm → partial RoPE → BF16 KV append → FA2 → sigmoid gate；不走 `qwen35_attention_fp8` 的短行 fused 分支。当前 target 几何为 Q-heads=16、KV-heads=2、head_dim=256，rotary_dim=64，物理 KV page=128 tokens。
- Target FP8 是后续独立配置。短行 1–4 rows 使用 native FP8 fused attention；大 Prefill 仍可先反量化到 BF16 再 FA2。不能拿一种 provider 的通过结果认证另一种。
- Draft MTP 是独立层，`QWEN35_MTP_KV_DTYPE=F8E4M3`、scale=1.0；因此“BF16 KV 基线”应准确标为 **target BF16 / draft FP8**，不称所有 KV 都 BF16。
- 模型 GDN 几何为 key/query heads=16、value heads=32、head_dim=128、conv width=4、scan chunk=64。新 15-case 中所有大 Prefill 长度为64对齐，291-token尾段的旧结果不能代替它们。

旧 review 工作区的 R01 增量复制、短 GDN 门禁、FP8 三个 K byte 差异均不属于本分支已经验收的结果。该 FP8 差异混合了 CPU/GPU QKNorm/RoPE 与量化：不能直接判量化器缺陷，也不能当同一 prequant 输入的 bitwise 测试。它不直接阻断 BF16 FA2 的独立表征，但 FP8 上线前必须隔离 reference 范围并重新验收。

## 必须冻结的证据

每次 gate 记录源码/tree/patch、实际 test executable SHA256、reference源码与输入数据 SHA256、dtype/scale/layout/positions/mask、provider/config、功耗/时钟策略、完整 stdout/stderr/退出码。数学 reference 身份与报告检查器身份分开。没有完成的 shape、资源 OOM、被 ignore/feature gate 排除的测试均是未完成，不能由相邻短 case 推断通过。

所有输入、输出和状态须有限；整数、token、cache offset、有效长度、未写区 canary、直接 KV memcpy 要求逐位一致。浮点至少报告 max-absolute、relative L2、cosine、最差逐 row relative L2；不能只看全局 cosine。候选与基线使用相同 reference/input 和冻结误差预算；dtype、accumulator、scale、mask、RNG 或接受算法的变更作为独立精度方案，不混入“纯性能”优化。

## 最小执行矩阵

| 项目 | 形状/状态覆盖 | 独立证据与判定 |
|---|---|---|
| BF16 FA2 attention core | cold Q=2K/4K/8K/16K、K=Q；append Q=4096、prefix=4K/8K/16K/32K/48K；decode/verify Q=1/2/3/4、prefix=4K/8K/16K/24K/32K/48K，并覆盖生成过程中实际增长的长度 | 同一 prepared BF16 Q/K/V 的 FP32 scalar/独立 math reference；causal key ≤ query_start+row；使用非零、非恒定输入；同一 actual K/V，不让 reference 重新执行另一套投影后冒充同-cache比较 |
| BF16 KV 生命周期 | 127/128/129、2047/2048/2049、4095/4096/4097、容量末端；prefix48K+append4K及decode1K的实际capacity | 检查被写范围、未写未来区域、旧prefix、逐row位置、截断后有效长度；target committed/shadow交换与snapshot restore后相同range逐byte比对；逻辑有效长度不能与物理padding混淆 |
| GDN scalar与整块 Prefill | 64、128、2048、2049，新增输入4096/8192/16384；零/非零conv与recurrent初态；大prefix后追加4096 | 独立 CPU 单步 recurrence；GPU逐token recurrence可作长序列调度对照，但不能独自证明共同数学实现正确；检查完整final recurrent/conv状态、输出和随后的decode；非零状态要来自明确固定的prefix/reference，不能当作空state重置 |
| MTP verifier/commit | depth1/2/3，0/partial/full accept；EOS在各row、预算1/2/3/4/1024、容量边界、失败或stale transaction、较短snapshot恢复后不同续写 | 纯状态机精确判定 + 真实GPU每个可提交row的checkpoint/rollback续跑；已接受范围逐byte正确，未接受row不可见；只能在成功同步后提交target+draft状态 |
| 完整模型 | 每个cold/append/decode输入清单；固定MTP与scalar、后续adaptive分别检查 | 相同权重与语义下输出实际token IDs/digest、首个差异、MTP计数与停止原因；decode必须真的产出1024 token，不能用提前EOS、截断或错误返回获得更快成绩 |
| 后续FP8 | 相同输入的BF16/FP8对照，真实checkpoint scale，page128/长context与1–4 query rows | 存储量化与attention算术分开；同一prequant BF16 bit-pattern/scale的RNE量化才要求相同bytes；Q/K norm/RoPE另作数值检查；模型量化影响另验，不称FP8无损 |

大 attention 可分离 operator 进程后用不共享 FA2/fused kernel 的 FP32 math reference 分块计算，避免与模型权重共同占用显存。如果 CPU oracle 只算预先声明的选中行（例如0/63/64/127/128/2047/2048/last），报告必须称“全shape执行、选定row参考验证”，不能称全输出逐元素独立验收；未验证的范围仍列出。mask负例应给未来/无效 KV 填非零有限 canary，避免全零输入把错误mask掩盖。

## 现有合同能证明什么

| 现有源码/测试 | 既有合同/限制 | P2使用方式 |
|---|---|---|
| `qwen35_mixers::decode_matches_independent_cpu_reference_at_qwen35_geometry` | conv逐位；convolved/recurrent cosine≥0.99999、relL2≤0.01；gated cosine≥0.9999、relL2≤0.025；decay/beta maxabs<2e-5 | 是独立CPU短step reference的起点，不能替代长state累积 |
| `chunk_prefill_and_decode_share_exact_qwen35_state_contract` | 64-token；conv逐位；state cosine≥0.9998、relL2≤0.025；output cosine≥0.999、relL2≤0.05，并检查下一token | 冻结为既有上限，不理解为候选可新增2.5%/5%误差 |
| `long_prefill_output_window_and_state_match_canonical_decode` 与 `long_prefill_{4096,8192,16384}_matches_canonical_decode`、`long_prefix_{4096,8192,16384,32768,49152}_append_4096_matches_canonical_decode` | 全部 ignored；各段 full output、首64输出、完整conv/recurrent状态及下一步decode；conv无条件逐位，数值上限保持state cosine≥0.9998/relL2≤0.025、output cosine≥0.999/relL2≤0.05；bitwise flag仅追加诊断要求 | 已在同stream初始化修复后显式执行9个长case并通过，源码与日志身份见[验收记录](model-microbench/evidence/precision-gates.json)。长reference是相同CUDA逐token recurrence的调度/状态对照；独立数学覆盖仍是CPU单步reference加CPU纯解析四步decay sanity，不能把长GPU调度对照称作16K独立数学参考 |
| `cake_matches_chunk_scan_at_qwen35_head_ratio` | ignored，覆盖64/128/1024/2048/4096和非零初态，但比较包含BF16/F16路径且相对L2仅报告 | 可辅助诊断，不代替same-representation独立reference，也不能把仅cosine过线当本P2全部指标合格 |
| `attention_bf16_qwen35_causal_cache_uses_fa2` | 仅全零Q/K/V，(Q,K)=(128,128)/(1,129)，只查provider名称 | 仅路由测试，**不是48K BF16数值证明** |
| native attention candidate validator | 对其给定reference执行逐元素 finite 与 `abs_error ≤0.03+0.02×abs(ref)` | 既有library admission上限；它不是35B模型官方无损承诺，不代替逐row/冻结baseline更严格的case合同；新增BF16专用上限需在候选前明确冻结 |
| `mtp_schedule.rs` tests | 精确0/partial/full、EOS/budget、失败/stale不可提交 | 无GPU数学证明；必须与GPU实际checkpoint/状态测试相接 |
| `qwen35_real_checkpoint::neutral_mtp_matches_forced_scalar_greedy` | 文件顶层现为`cfg(feature="qwen35-cuda")`，ignored；默认prompt16、输出32；实际断言token IDs、rollback统计 | 仍须先`--list`确认存在并显式执行ignored测试；feature修正不代表已完成GPU验收。默认短prompt不认证48K/1K输出 |

算子改写若宣称“仅调度/复制改变”，同输入的原始KV/state应逐位一致。不同CPU/GPU的FP32归约或trig发生合法舍入差异时，应先隔离该stage；不能对不同prequant输入的FP8 bytes机械要求相等，也不能因此放宽真正同输入量化或数据复制的要求。

NVFP4 MoE可能存在非逐位可重现的expert reduction。重复输出不一致不能静默删除：记录reference自身重复分布、首个差异和可取得的logit margin；若不能区分数值噪声与MTP协议错误，`accuracy_status=unresolved`，不得写token-equivalent=true。`--allow-repeat-divergence`只可用于明确标注的性能表征，不把它转换成精度通过；固定token输入、禁提前停止和完整1024输出要求仍保留。

## 正式计时必须已有 recipe

`CudaContext::new_with_options(..., autotune=false, ...)`会把native runtime的`allow_online_tune`关闭。native GEMM及attention只有在 `runtime.allow_online_tune && policy.online_tune`时进入tune分支，当前源码未发现另一个绕过该许可的recover_missing_recipe调优入口。

普通生产路径的BF16 FA2边界显式 `allow_fallback=true`。没有strict guard时，无recipe可使用固定配置FA2 fallback，且不会写recipe，所以 **recipe目录hash不变不足以证明全程命中recipe**。当前正式runner额外启用下述strict guard。正式前的独立调优必须覆盖所有实际KV长度/bucket、scalar与MTP verifier形状；调优不自动等价于数值P2验收。

native strict-recipe guard现已实现：正式新进程设置 `APXINF_REQUIRE_RECIPES=1`，GEMM/attention recipe缺失或不可恢复时在在线调优/fallback之前返回cache-miss；该flag按进程读取，不复用调优进程的prepared executions。独立audit/profile进程另启用 `APXINF_GEMM_REPORT=1` 与 `APXINF_ATTENTION_REPORT=1`，只接受 `source=recipe` / `source=memory-recipe`。正式无插桩进程关闭这些REPORT变量，依靠native strict guard拒绝miss。发现 `fallback`、`fallback-after-tune`、fresh tune或unknown source时保留失败，补全独立调优后开新正式进程重测。日志开销须前后同配置；不要用有日志的baseline比较无日志的candidate。直接固定kernel/GDN没有recipe概念，列入执行路径清单，不伪造“recipe命中”。

初始KV预建、weight load、autotune和graph/resource预备不计入测量；但它们必须成功且不能在正式decode循环里重新发生。调优/正式进程的recipe清单、实际argv及`source`审计都归档。P0代码改动只能修测量，不顺便接受旧未验证R01或新的算术变化。

## MTP策略的测量身份

当前Decode矩阵使用 ordinary generation 路径的 `qwen35_mtp_bench`：可分开测 scalar、固定depth1/2/3及adaptive；`APXINF_QWEN35_MTP_FIXED_DEPTH`控制ordinary controller的固定深度，adaptive需显式绑定同配置scalar校准，不能把内置默认成本当实测校准。策略身份及完整输出必须写入记录，GPU策略矩阵尚未完成。`qwen35_context_bench`调用的 `benchmark_mtp_decode` hook仍直接传入 `QWEN35_MTP_DRAFT_STEPS=3`，未调用 `AdaptiveMtpController`；仅设该变量不能把此hook标成depth1/2或adaptive。不同路径分别建立同scope消融，不能互换成绩。容量至少覆盖实际prompt +1024输出 +最大lookahead，并记录物理padding；不能通过减少可验证row或scope缩短Decode。

本文保留未完成门禁，不把已执行的局部测试扩写为完整模型认证。P0已实现native strict-recipe guard；后续正式矩阵、MTP协议、FP8 attention与模型质量分别验收。固定路由的真实MoE诊断观察到小幅非逐位重复输出，原始差异保留，不能推断为某个单一kernel根因或放宽阈值。

## Same-prepared BF16 → FP8 KV 独立门禁

`qwen35_prepared_fp8_cache_precision` 只调用公开
`ops::quantization(FixedScaleE4m3)`，它也是分解式 target KV append 使用的边界。
输入直接由固定 BF16 bit-pattern 准备，不包含 projection、QKNorm、RoPE 或
attention。CPU oracle 以数值公式枚举127个非负有限 E4M3 编码（含零），用
nearest/ties-even 选取，不复刻 GPU converter 的位运算实现。

实际 native 阶段是 **host F32 `1/scale` → device F32 乘法 → clamp ±448 → E4M3**。
融合式短行 attention 的 device `x/scale` 是另一个阶段合同，此测试不认证该路径。
power2 scales 0.25/0.5/1/2/4 要求全部65,280个有限 BF16、正负零、subnormal、
ties 和 saturation 逐 byte 一致；synthetic nonpower2 scales 同样要求 GPU 精确
匹配 reciprocal-product 阶段，F64 division / direct F32 division 的边界差异单独报告，
不能让 GPU 在两个相邻编码之间任选，也不能据此放宽阈值。

缓存写入门禁覆盖 page128、prefix127/128/129及跨页 append，逐 byte 检查旧 prefix、
写入范围和未来 canary。ignored 长门禁覆盖 prefix4K/8K/16K/32K/48K追加4K。
这认证的是静态存储量化/写入，不是 FP8 attention、完整模型质量、动态量化或 MTP。

当前 checkpoint 的初始 header 查询没有找到所查询命名的序列化 K/V scale tensor；
扩展命名查询因网络中断未完成，不能记为零匹配。源码 deployment manifest loader
使用 K/V=1.0 unit default；这与 TRT engine 的 serialized calibration 参数、真实 trace
校准得到的 per-layer scales 是不同的证据，`kv_cache_quant_algo=FP8` 也不是校准证明。

实际 GPU executable 已执行：`--list` 为10项；默认套件9通过、1 ignored（含6项CPU oracle与3项GPU测试）；ignored长门禁另行精确执行1通过、9 filtered。14条scale/domain记录、14条短K/V范围与10条长K/V范围均完整，逐byte断言全部通过。零差异结论来自冻结断言之后的 `byte_exact=PASS`，不是日志另行打印的数值误差计数。离线解析器的8种缺项/重复/错误退出码/字节错误负例均拒绝；实际源码、ELF、日志及解析产物哈希见[维护证据](model-microbench/evidence/fp8-prepared-storage-gate.json)。

该通过结论仅限同一prepared BF16输入的静态E4M3量化与指定cache view的写入范围；没有执行模型权重、工具选择、长文理解或生成一致性验收，不认证 fused FP8 attention、动态量化、checkpoint/TRT scale校准、模型质量、MTP、Prefix Cache复用或性能。
