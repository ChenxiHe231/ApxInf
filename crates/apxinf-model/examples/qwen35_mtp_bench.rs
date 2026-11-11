//! Load-once Qwen3.5-35B-A3B NVFP4 scalar-versus-MTP benchmark.
//!
//! The default prompt is exactly 4,096 deterministic token IDs; smoke tests
//! may override it with `--prompt-tokens`. Each mode runs two
//! warmups followed by five measured iterations and emits exactly 256 tokens
//! unless a larger `--max-new` value is requested. EOS stopping is disabled.
//! The model and its CUDA allocations are loaded once for the whole process.

use std::path::PathBuf;
use std::time::Instant;

use apxinf_core::Device;
use apxinf_model::qwen35moe::{Qwen35KvCacheDType, Qwen35MoeActivation};
use apxinf_model::{
    AutoModel, GenerationOptions, LlmInput, LoadOptions, LoadedModel, SamplingMode,
    SpeculativeGenerationStats,
};
use serde_json::{json, Value};

#[path = "support/qwen35_benchmark.rs"]
mod benchmark;
#[path = "support/qwen35_cuda_profile.rs"]
mod cuda_profile;
#[path = "support/qwen35_decode_profile.rs"]
mod decode_profile;
#[path = "support/qwen35_callback_timing.rs"]
mod callback_clock;

struct NvtxGenerationRanges;
impl decode_profile::RangeSink for NvtxGenerationRanges {
    type Guard = apxinf_model::profiling::trace::Range;
    fn push(&mut self, label: &'static str) -> Self::Guard {
        apxinf_model::profiling::trace::range(label)
    }
}
use benchmark::{quantile, token_digest, INPUT_SEED};
const PROMPT_TOKENS: usize = 4_096;
const DEFAULT_OUTPUT_TOKENS: usize = 256;
const DEFAULT_WARMUPS: usize = 2;
const DEFAULT_REPEATS: usize = 5;
const MTP_LOOKAHEAD: usize = apxinf_model::qwen35moe::mtp_schedule::QWEN35_MTP_DRAFT_STEPS;

fn report_repeat_divergence(
    mode: &str,
    iteration: usize,
    warmup: bool,
    allow_repeat_divergence: bool,
    expected: &[u32],
    actual: &[u32],
) {
    let first_diff = expected.iter().zip(actual)
        .position(|(a, b)| a != b)
        .unwrap_or(expected.len().min(actual.len()));
    let start = first_diff.saturating_sub(3);
    let expected_start = start.min(expected.len());
    let actual_start = start.min(actual.len());
    eprintln!("{}", json!({
        "schema": "apxinf.qwen35.benchmark.repeat_divergence.v1",
        "benchmark": "decode",
        "mode": mode,
        "iteration": iteration,
        "warmup": warmup,
        "allow_repeat_divergence": allow_repeat_divergence,
        "first_diff": first_diff,
        "expected_output_tokens": expected.len(),
        "actual_output_tokens": actual.len(),
        "expected_digest_fnv1a64": format!("{:016x}", token_digest(expected)),
        "actual_digest_fnv1a64": format!("{:016x}", token_digest(actual)),
        "expected_first_diff_token": expected.get(first_diff),
        "actual_first_diff_token": actual.get(first_diff),
        "expected_window": {"start": expected_start, "tokens": &expected[expected_start..expected_start.saturating_add(8).min(expected.len())]},
        "actual_window": {"start": actual_start, "tokens": &actual[actual_start..actual_start.saturating_add(8).min(actual.len())]},
    }));
}

#[derive(Clone, Copy)]
enum Mode {
    Scalar,
    Mtp,
}

#[derive(Clone, Copy)]
enum BenchmarkMode {
    Scalar,
    Mtp,
    Both,
}

impl BenchmarkMode {
    fn parse(args: &[String]) -> Result<Self, Box<dyn std::error::Error>> {
        let value = args
            .iter()
            .position(|arg| arg == "--mode")
            .and_then(|index| args.get(index + 1))
            .map(String::as_str)
            .unwrap_or("both");
        match value {
            "scalar" => Ok(Self::Scalar),
            "mtp" => Ok(Self::Mtp),
            "both" => Ok(Self::Both),
            _ => Err(format!("--mode must be scalar, mtp, or both; got {value}").into()),
        }
    }

    fn name(self) -> &'static str {
        match self {
            Self::Scalar => "scalar",
            Self::Mtp => "mtp",
            Self::Both => "both",
        }
    }
}

impl Mode {
    fn name(self) -> &'static str {
        match self {
            Self::Scalar => "scalar",
            Self::Mtp => "mtp",
        }
    }

    fn options(self, max_new_tokens: usize) -> GenerationOptions {
        match self {
            Self::Mtp => GenerationOptions::greedy(max_new_tokens, None),
            Self::Scalar => GenerationOptions {
                speculative_decoding: Some(false),
                sampling_mode: Some(SamplingMode::Greedy),
                ..GenerationOptions::greedy(max_new_tokens, None)
            },
        }
    }
}

#[derive(Default)]
struct AggregateSpeculation {
    rounds: usize,
    zero_accept_rounds: usize,
    partial_accept_rounds: usize,
    full_accept_rounds: usize,
    proposed: usize,
    verified_accepted: usize,
    committed_accepted: usize,
    target_verified_rows: usize,
    target_committed_rows: usize,
    target_rolled_back_rows: usize,
    proposed_by_depth: [usize; MTP_LOOKAHEAD],
    matched_by_depth: [usize; MTP_LOOKAHEAD],
    accepted_prefix_by_depth: [usize; MTP_LOOKAHEAD],
}

impl AggregateSpeculation {
    fn add(&mut self, stats: &SpeculativeGenerationStats) {
        self.rounds += stats.rounds;
        self.zero_accept_rounds += stats.zero_accept_rounds;
        self.partial_accept_rounds += stats.partial_accept_rounds;
        self.full_accept_rounds += stats.full_accept_rounds;
        self.proposed += stats.proposed_tokens;
        self.verified_accepted += stats.verified_accepted_tokens;
        self.committed_accepted += stats.committed_accepted_tokens;
        self.target_verified_rows += stats.target_verified_rows;
        self.target_committed_rows += stats.target_committed_rows;
        self.target_rolled_back_rows += stats.target_rolled_back_rows;
        for depth in 0..MTP_LOOKAHEAD {
            self.proposed_by_depth[depth] += stats.proposed_by_depth[depth];
            self.matched_by_depth[depth] += stats.matched_by_depth[depth];
            self.accepted_prefix_by_depth[depth] += stats.accepted_prefix_by_depth[depth];
        }
    }

    fn to_json(&self) -> Value {
        let rate = |value: usize, total: usize| (total != 0).then_some(value as f64 / total as f64);
        json!({
            "rounds": self.rounds,
            "zero_accept_rounds": self.zero_accept_rounds,
            "partial_accept_rounds": self.partial_accept_rounds,
            "full_accept_rounds": self.full_accept_rounds,
            "proposed_tokens": self.proposed,
            "verified_accepted_tokens": self.verified_accepted,
            "committed_accepted_tokens": self.committed_accepted,
            "verified_acceptance_rate": rate(self.verified_accepted, self.proposed),
            "committed_acceptance_rate": rate(self.committed_accepted, self.proposed),
            "target_verified_rows": self.target_verified_rows,
            "target_committed_rows": self.target_committed_rows,
            "target_rolled_back_rows": self.target_rolled_back_rows,
            "proposed_by_depth": self.proposed_by_depth,
            "matched_by_depth": self.matched_by_depth,
            "accepted_prefix_by_depth": self.accepted_prefix_by_depth,
        })
    }
}

struct ModeResult {
    canonical_tokens: Vec<u32>,
    summary: Value,
}

/// Record how many measured repeats diverged from the first. Non-zero is
/// expected on the NVFP4 MoE path and is reported, not treated as failure.
fn note_divergence(summary: &mut Value, divergent_repeats: usize) {
    if let Some(map) = summary.as_object_mut() {
        map.insert(
            "divergent_repeats".to_string(),
            Value::from(divergent_repeats),
        );
    }
}

fn exact_prompt(prompt_tokens: usize) -> Vec<u32> {
    benchmark::exact_tokens(prompt_tokens)
}

fn run_mode(
    model: &mut LoadedModel,
    prompt: &[u32],
    max_new_tokens: usize,
    mode: Mode,
    warmups: usize,
    repeats: usize,
    allow_repeat_divergence: bool,
    autotune: bool,
    profiling: bool,
) -> Result<ModeResult, Box<dyn std::error::Error>> {
    let mut divergent_repeats = 0usize;
    let mut divergent_warmups = 0usize;
    let options = mode.options(max_new_tokens);
    let mut measured_ttft_ms = Vec::with_capacity(repeats);
    let mut measured_prefill_tps = Vec::with_capacity(repeats);
    let mut measured_generation_tps = Vec::with_capacity(repeats);
    let mut measured_decode_ms = Vec::with_capacity(repeats);
    let mut measured_post_last_ms = Vec::with_capacity(repeats);
    let mut measured_total_ms = Vec::with_capacity(repeats);
    let mut measured_legacy_generation_tps = Vec::with_capacity(repeats);
    let mut aggregate = AggregateSpeculation::default();
    let mut canonical_tokens: Option<Vec<u32>> = None;

    for iteration in 0..warmups + repeats {
        let warmup = iteration < warmups;
        let profile_range = cuda_profile::CudaProfileRange::start(profiling && !warmup)?;
        let (output, callback_timing) = if profiling && !warmup {
            // Callback endpoints are host availability boundaries. They do not
            // introduce GPU synchronization or split an MTP round into tokens.
            let mut phases = decode_profile::GenerationRanges::new(
                NvtxGenerationRanges, max_new_tokens,
            )?;
            let mut timing = callback_clock::CallbackTimer::start(max_new_tokens)?;
            let output = model.generate_streaming_with_options(
                LlmInput::text(prompt), &options, |_token| {
                    timing.on_token();
                    phases.on_token();
                },
            )?;
            let timing = timing.finish_after_generate(output.tokens.len())?;
            phases.finish(output.tokens.len())?;
            (output, timing)
        } else {
            // The same timing helper checks warmups and formal output budgets.
            // Only first/final callbacks read the clock; intermediate callbacks
            // count delivery without token timestamps, GPU work, or logging.
            let mut timing = callback_clock::CallbackTimer::start(max_new_tokens)?;
            let output = model.generate_streaming_with_options(
                LlmInput::text(prompt), &options, |_token| timing.on_token(),
            )?;
            let timing = timing.finish_after_generate(output.tokens.len())?;
            (output, timing)
        };
        profile_range.finish()?;
        if profiling && !warmup {
            println!("{}", json!({
                "schema": "apxinf.qwen35.benchmark.profile_scope.v2",
                "mode": mode.name(), "iteration": iteration,
                "performance_eligible": false,
                "capture_scope": "full_generate_including_prompt_prefill",
                "nvtx_requested": std::env::var("APXINF_NVTX").as_deref() == Ok("1"),
                "generate_range": decode_profile::GENERATE,
                "entry_to_first_callback_range": decode_profile::PREFILL,
                "first_to_last_callback_range": (max_new_tokens > 1).then_some(decode_profile::DECODE),
                "last_callback_to_return_range": decode_profile::FINALIZE,
                "callback_tokens": output.tokens.len(),
                "callback_intervals": output.tokens.len().saturating_sub(1),
                "attribution_requires_actual_nvtx_events_and_cuda_correlation": true,
            }));
        }
        let token_ids = output.token_ids();
        if let Some(expected) = &canonical_tokens {
            if expected != &token_ids {
                // Emit before either length or repeat gates return an error.
                report_repeat_divergence(
                    mode.name(), iteration, warmup, allow_repeat_divergence, expected, &token_ids,
                );
            }
        }
        if token_ids.len() != max_new_tokens {
            return Err(format!(
                "{} iteration {iteration} produced {} tokens, expected {max_new_tokens}",
                mode.name(),
                token_ids.len()
            )
            .into());
        }
        let mut repeat_diverged = false;
        if let Some(expected) = &canonical_tokens {
            if expected != &token_ids {
                let mismatch = expected
                    .iter()
                    .zip(&token_ids)
                    .position(|(left, right)| left != right)
                    .unwrap_or(expected.len().min(token_ids.len()));
                let detail = format!(
                    "{} output changed across repeats at token {mismatch}: expected={:?}, actual={:?}, expected_digest={:016x}, actual_digest={:016x}",
                    mode.name(),
                    expected.get(mismatch),
                    token_ids.get(mismatch),
                    token_digest(expected),
                    token_digest(&token_ids),
                );
                // The NVFP4 MoE combines experts with a relaxed bf16 atomic
                // add, so the per-token sum over experts depends on CTA
                // arrival order and is not bit-reproducible. That is a
                // property of the vendored CuTe DSL kernel, not of this
                // engine: the TensorRT Edge-LLM build of the same model
                // returns four different completions for four identical
                // greedy requests. Asserting bit-equality across repeats
                // therefore asserts something the kernel never promised, and
                // it makes the benchmark unrunnable on its own default path.
                //
                // `--allow-repeat-divergence` downgrades it to a counted
                // observation so throughput can be measured the same way the
                // reference engine is measured. The `bf16` MoE activation is
                // a genuinely deterministic path and should still be run
                // without this flag as the correctness gate.
                if allow_repeat_divergence {
                    repeat_diverged = true;
                    if warmup {
                        divergent_warmups += 1;
                    } else {
                        divergent_repeats += 1;
                    }
                    eprintln!("[qwen35-bench] WARNING {detail}");
                } else {
                    return Err(detail.into());
                }
            }
        } else {
            canonical_tokens = Some(token_ids.clone());
        }

        let legacy_ttft_ms = output.profile.ttft_ms()
            .ok_or("legacy generation profile did not record TTFT")?;
        let legacy_generation_tps = output.profile.generation_tps();
        let legacy_total_ms = output.profile.total_latency_ms()
            .ok_or("legacy generation profile did not record total latency")?;
        let legacy_decode_ms = (max_new_tokens > 1).then_some(legacy_total_ms - legacy_ttft_ms);
        let ttft_ms = callback_clock::ms(callback_timing.first_callback);
        let total_latency_ms = callback_clock::ms(callback_timing.generate_return);
        let decode_ms = callback_timing.decode_ms();
        let post_last_callback_ms = callback_clock::ms(callback_timing.post_last_callback);
        let generation_tps = callback_timing.decode_tps();
        if max_new_tokens > 1 && (generation_tps.is_none() || legacy_generation_tps.is_none()) {
            return Err("callback or legacy generation profile did not record Decode TPS".into());
        }
        if output.profile.reused_input_tokens() != 0
            || output.profile.input_tokens() != prompt.len()
        {
            return Err(
                "decode benchmark requires the entire prompt with zero context reuse".into(),
            );
        }
        let prefill_tps = prompt.len() as f64 / (ttft_ms / 1_000.0);
        let speculation = model.speculative_generation_stats()?;
        match (mode, speculation.as_ref()) {
            (Mode::Mtp, Some(stats)) => {
                if !warmup {
                    aggregate.add(stats);
                }
            }
            (Mode::Mtp, None) => {
                return Err("MTP benchmark request exposed no speculative counters".into())
            }
            (Mode::Scalar, Some(_)) => {
                return Err("forced-scalar benchmark request unexpectedly used MTP".into())
            }
            (Mode::Scalar, None) => {}
        }

        println!(
            "{}",
            json!({
                "schema": "apxinf.qwen35.benchmark.iteration.v2",
                "mode": mode.name(),
                "iteration": iteration,
                "warmup": warmup,
                "autotune": autotune,
                "profiling": profiling,
                "performance_eligible": !autotune && !profiling,
                "strict_recipe_required": !autotune,
                "recipe_audit_required": true,
                "computed_prompt_tokens": output.profile.input_tokens(),
                "reused_prompt_tokens": output.profile.reused_input_tokens(),
                "decode_intervals": token_ids.len().saturating_sub(1),
                "total_latency_ms": total_latency_ms,
                "timing_method": callback_clock::METHOD,
                "decode_interval_ms": decode_ms,
                "post_last_callback_ms": post_last_callback_ms,
                "callback_timing": {
                    "clock": callback_clock::CLOCK,
                    "origin": callback_clock::ORIGIN,
                    "clock_reads": callback_timing.clock_reads,
                    "callback_count": callback_timing.callback_count,
                    "decode_intervals": callback_timing.intervals,
                    "first_callback_elapsed_ns": callback_timing.first_callback.as_nanos(),
                    "last_callback_elapsed_ns": callback_timing.last_callback.as_nanos(),
                    "generate_return_elapsed_ns": callback_timing.generate_return.as_nanos(),
                },
                "legacy_engine_profile": {
                    "schema": "apxinf.qwen35.legacy-engine-profile.v1",
                    "scope": "first_model_token_ready_to_engine_finalize_callbacks_excluded",
                    "ttft_ms": legacy_ttft_ms,
                    "total_latency_ms": legacy_total_ms,
                    "decode_including_finalize_ms": legacy_decode_ms,
                    "generation_tokens_per_second": legacy_generation_tps,
                },
                "prompt_tokens": prompt.len(),
                "output_tokens": token_ids.len(),
                "ttft_ms": ttft_ms,
                "prefill_tokens_per_second": prefill_tps,
                "decode_tokens_per_second": generation_tps,
                "repeat_diverged": repeat_diverged,
                "token_digest_fnv1a64": format!("{:016x}", token_digest(&token_ids)),
                "speculation": speculation,
            })
        );

        if !warmup {
            measured_ttft_ms.push(ttft_ms);
            measured_prefill_tps.push(prefill_tps);
            measured_post_last_ms.push(post_last_callback_ms);
            measured_total_ms.push(total_latency_ms);
            if let Some(ms) = decode_ms { measured_decode_ms.push(ms); }
            if let Some(tps) = legacy_generation_tps { measured_legacy_generation_tps.push(tps); }
            if let Some(tps) = generation_tps {
                measured_generation_tps.push(tps);
            }
        }
    }

    let mut summary = json!({
        "schema": "apxinf.qwen35.benchmark.summary.v2",
        "mode": mode.name(),
        "autotune": autotune,
        "profiling": profiling,
        "performance_eligible": !autotune && !profiling,
        "strict_recipe_required": !autotune,
        "recipe_audit_required": true,
        "decode_intervals": max_new_tokens - 1,
        "prompt_tokens": prompt.len(),
        "output_tokens": max_new_tokens,
        "warmups": warmups,
        "repeats": repeats,
        "ttft_ms": {
            "median": quantile(&measured_ttft_ms, 0.5),
            "p95": quantile(&measured_ttft_ms, 0.95),
        },
        "prefill_tokens_per_second": {
            "median": quantile(&measured_prefill_tps, 0.5),
            "p95": quantile(&measured_prefill_tps, 0.95),
        },
        "timing_method": callback_clock::METHOD,
        "post_last_callback_ms": {
            "median": quantile(&measured_post_last_ms, 0.5),
            "p95": quantile(&measured_post_last_ms, 0.95),
        },
        "total_latency_ms": {
            "median": quantile(&measured_total_ms, 0.5),
            "p95": quantile(&measured_total_ms, 0.95),
        },
        "decode_interval_ms": (!measured_decode_ms.is_empty()).then(|| json!({
            "median": quantile(&measured_decode_ms, 0.5),
            "p95": quantile(&measured_decode_ms, 0.95),
        })),
        "legacy_engine_profile": {
            "scope": "first_model_token_ready_to_engine_finalize_callbacks_excluded",
            "generation_tokens_per_second": (!measured_legacy_generation_tps.is_empty()).then(|| json!({
                "median": quantile(&measured_legacy_generation_tps, 0.5),
                "p95": quantile(&measured_legacy_generation_tps, 0.95),
            })),
        },
        "decode_tokens_per_second": (!measured_generation_tps.is_empty()).then(|| json!({
            "median": quantile(&measured_generation_tps, 0.5),
            "p95": quantile(&measured_generation_tps, 0.95),
        })),
        "speculation": match mode {
            Mode::Mtp => aggregate.to_json(),
            Mode::Scalar => Value::Null,
        },
    });
    note_divergence(&mut summary, divergent_repeats);
    summary["divergent_warmups"] = Value::from(divergent_warmups);
    Ok(ModeResult {
        canonical_tokens: canonical_tokens.expect("at least one iteration"),
        summary,
    })
}

fn arg_value(args: &[String], name: &str, default: usize) -> usize {
    args.iter()
        .position(|arg| arg == name)
        .and_then(|index| args.get(index + 1))
        .map(|value| value.parse().expect(name))
        .unwrap_or(default)
}

fn optional_path_arg(
    args: &[String],
    name: &str,
) -> Result<Option<PathBuf>, Box<dyn std::error::Error>> {
    args.iter()
        .position(|arg| arg == name)
        .map(|index| {
            args.get(index + 1)
                .filter(|value| !value.starts_with("--"))
                .map(PathBuf::from)
                .ok_or_else(|| format!("{name} requires a path").into())
        })
        .transpose()
}

fn precision_arg<T>(args: &[String], name: &str) -> Result<T, Box<dyn std::error::Error>>
where
    T: std::str::FromStr<Err = String> + Default,
{
    args.iter()
        .position(|arg| arg == name)
        .map(|index| {
            args.get(index + 1)
                .ok_or_else(|| format!("{name} requires a value"))?
                .parse::<T>()
        })
        .transpose()
        .map(|value| value.unwrap_or_default())
        .map_err(Into::into)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let checkpoint = args.get(1).filter(|arg| !arg.starts_with("--")).ok_or(
        "usage: qwen35_mtp_bench <checkpoint-dir> [--device N] [--max-new N] \
             [--prompt-tokens N] [--warmups N] [--repeats N] [--mode scalar|mtp|both] \
             [--qwen35-moe-activation nvfp4|bf16] [--qwen35-kv-cache bf16|fp8|fp8_dynamic] \
             [--tuning-dir ABSOLUTE_PATH] [--autotune]",
    )?;
    let device = arg_value(&args, "--device", 0);
    let prompt_tokens = arg_value(&args, "--prompt-tokens", PROMPT_TOKENS);
    let max_new_tokens = arg_value(&args, "--max-new", DEFAULT_OUTPUT_TOKENS);
    let warmups = arg_value(&args, "--warmups", DEFAULT_WARMUPS);
    let repeats = arg_value(&args, "--repeats", DEFAULT_REPEATS);
    // The NVFP4 MoE expert combine is not bit-reproducible; see run_mode.
    let allow_repeat_divergence = args.iter().any(|arg| arg == "--allow-repeat-divergence");
    let benchmark_mode = BenchmarkMode::parse(&args)?;
    let moe_activation = precision_arg::<Qwen35MoeActivation>(&args, "--qwen35-moe-activation")?;
    let kv_cache = precision_arg::<Qwen35KvCacheDType>(&args, "--qwen35-kv-cache")?;
    let tuning_path = optional_path_arg(&args, "--tuning-dir")?;
    let autotune = args.iter().any(|arg| arg == "--autotune");
    let profiling = args.iter().any(|arg| arg == "--profile-measured");
    if profiling && (autotune || repeats != 1 || matches!(benchmark_mode, BenchmarkMode::Both)) {
        return Err("--profile-measured requires one non-autotune repeat and one --mode".into());
    }
    let fa2_packed_gqa = benchmark::pin_fa2_packed_gqa()?;
    std::env::set_var("APXINF_REQUIRE_RECIPES", if autotune { "0" } else { "1" });
    benchmark::validate_tuning_path(&tuning_path, autotune)?;
    if tuning_path.as_ref().is_some_and(|path| !path.is_absolute()) {
        return Err("--tuning-dir must be an absolute path".into());
    }
    if autotune && tuning_path.is_none() {
        return Err("--autotune requires --tuning-dir".into());
    }
    if prompt_tokens == 0 {
        return Err("--prompt-tokens must be at least 1".into());
    }
    if max_new_tokens == 0 {
        return Err("--max-new must be at least 1".into());
    }
    if repeats == 0 {
        return Err("--repeats must be at least 1".into());
    }

    let prompt = exact_prompt(prompt_tokens);
    let capacity = prompt_tokens
        .checked_add(max_new_tokens)
        .and_then(|value| value.checked_add(MTP_LOOKAHEAD))
        .ok_or("KV capacity overflow")?;
    let execution = apxinf_model::qwen35moe::Qwen35ExecutionOptions {
        moe_activation,
        kv_cache,
    };
    let requested_depth = std::env::var("APXINF_QWEN35_MTP_FIXED_DEPTH")
        .ok()
        .map(|value| value.parse::<usize>())
        .transpose()?;
    if requested_depth.is_some_and(|depth| !(1..=MTP_LOOKAHEAD).contains(&depth)) {
        return Err("APXINF_QWEN35_MTP_FIXED_DEPTH must be within supported draft depth".into());
    }
    let fixed_depth = if execution.uses_thor_dynamic_fp8_decode(device, capacity) {
        if requested_depth.is_some_and(|depth| depth != MTP_LOOKAHEAD) {
            return Err("Thor dynamic FP8 Decode uses fixed MTP depth 3".into());
        }
        Some(MTP_LOOKAHEAD)
    } else {
        requested_depth
    };
    let options = LoadOptions {
        text_kv_capacity: Some(capacity),
        qwen35_moe_activation: Some(moe_activation),
        qwen35_kv_cache: Some(kv_cache),
        tuning_path: tuning_path.clone(),
        autotune,
        ..LoadOptions::default()
    };
    let load_start = Instant::now();
    let mut model = AutoModel::load_model(Device::Cuda(device), checkpoint, &options)?;
    model.set_context_reuse(false)?;
    println!(
        "{}",
        json!({
            "schema": "apxinf.qwen35.benchmark.config.v2",
            "fa2_packed_gqa": fa2_packed_gqa,
            "checkpoint": checkpoint,
            "device": device,
            "prompt_tokens": prompt_tokens,
            "output_tokens": max_new_tokens,
            "warmups": warmups,
            "repeats": repeats,
            "mode": benchmark_mode.name(),
            "qwen35_moe_activation": moe_activation.as_str(),
            "qwen35_kv_cache": kv_cache.as_str(),
            "tuning_dir": tuning_path,
            "autotune": autotune,
            "profiling": profiling,
            "performance_eligible": !autotune && !profiling,
            "strict_recipe_required": !autotune,
            "recipe_audit_required": true,
            "context_reuse": false,
            "timing_method": callback_clock::METHOD,
            "timing_clock": callback_clock::CLOCK,
            "timing_origin": callback_clock::ORIGIN,
            "timing_clock_reads_per_generate": if max_new_tokens > 1 { 4 } else { 3 },
            "ttft_scope": "generate_entry_to_first_callback",
            "decode_scope": "first_to_last_callback_intervals",
            "post_last_callback_scope": "last_callback_to_generate_return",
            "temperature": 0.0,
            "sampling_mode": "greedy",
            "kv_capacity": capacity,
            "input_seed_hex": format!("{:016x}", INPUT_SEED),
            "input_digest_fnv1a64": format!("{:016x}", token_digest(&prompt)),
            "mtp_enabled": !matches!(benchmark_mode, BenchmarkMode::Scalar),
            "mtp_strategy": if matches!(benchmark_mode, BenchmarkMode::Scalar) { "disabled" } else if fixed_depth.is_some() { "fixed" } else { "adaptive" },
            "fixed_depth": if matches!(benchmark_mode, BenchmarkMode::Scalar) { None } else { fixed_depth },
            "mtp_depth": if matches!(benchmark_mode, BenchmarkMode::Scalar) { None } else { fixed_depth },
            "draft_kv_dtype": "fp8",
            "draft_kv_scale": 1.0,
            "allow_repeat_divergence": allow_repeat_divergence,
            "load_seconds": load_start.elapsed().as_secs_f64(),
        })
    );

    match benchmark_mode {
        BenchmarkMode::Scalar => {
            let scalar = run_mode(
                &mut model,
                &prompt,
                max_new_tokens,
                Mode::Scalar,
                warmups,
                repeats,
                allow_repeat_divergence,
                autotune,
                profiling,
            )?;
            println!("{}", scalar.summary);
        }
        BenchmarkMode::Mtp => {
            let mtp = run_mode(
                &mut model,
                &prompt,
                max_new_tokens,
                Mode::Mtp,
                warmups,
                repeats,
                allow_repeat_divergence,
                autotune,
                profiling,
            )?;
            println!("{}", mtp.summary);
        }
        BenchmarkMode::Both => {
            let scalar = run_mode(
                &mut model,
                &prompt,
                max_new_tokens,
                Mode::Scalar,
                warmups,
                repeats,
                allow_repeat_divergence,
                autotune,
                profiling,
            )?;
            let mtp = run_mode(
                &mut model,
                &prompt,
                max_new_tokens,
                Mode::Mtp,
                warmups,
                repeats,
                allow_repeat_divergence,
                autotune,
                profiling,
            )?;
            if scalar.canonical_tokens != mtp.canonical_tokens {
                let mismatch = scalar
                    .canonical_tokens
                    .iter()
                    .zip(&mtp.canonical_tokens)
                    .position(|(scalar, mtp)| scalar != mtp)
                    .unwrap_or(max_new_tokens);
                return Err(format!(
                    "MTP token IDs differ from forced-scalar greedy token IDs at {mismatch}: scalar={:?}, mtp={:?}",
                    scalar.canonical_tokens.get(mismatch),
                    mtp.canonical_tokens.get(mismatch),
                )
                .into());
            }
            println!("{}", scalar.summary);
            println!("{}", mtp.summary);
            println!(
                "{}",
                json!({
                    "schema": "apxinf.qwen35.benchmark.parity.v1",
                    "token_ids_equal": true,
                    "compared_tokens": max_new_tokens,
                })
            );
        }
    }
    Ok(())
}
