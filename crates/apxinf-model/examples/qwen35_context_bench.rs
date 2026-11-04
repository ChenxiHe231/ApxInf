//! Qwen3.5-35B-A3B NVFP4 chunk-prefill benchmark with an existing KV prefix.

use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use apxinf_core::{Backend, CpuBackend, Device};
use apxinf_model::qwen35moe::{
    load_trt_fp8_kv_cache_scales, Qwen35ExecutionOptions, Qwen35KvCacheDType, Qwen35Moe,
    Qwen35MoeActivation, Qwen35MoeConfig,
};
use apxinf_model::{GenerationOptions, LlmTrait, SamplingMode};
use serde_json::{json, Value};

#[path = "support/qwen35_benchmark.rs"]
mod benchmark;
#[path = "support/qwen35_cuda_profile.rs"]
mod cuda_profile;
use benchmark::{exact_tokens, quantile, token_digest, usize_arg, INPUT_SEED};
const DEFAULT_OUTPUT_TOKENS: usize = 1;
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
) -> usize {
    let first_diff = expected.iter().zip(actual)
        .position(|(a, b)| a != b)
        .unwrap_or(expected.len().min(actual.len()));
    let start = first_diff.saturating_sub(3);
    let expected_start = start.min(expected.len());
    let actual_start = start.min(actual.len());
    eprintln!("{}", json!({
        "schema": "apxinf.qwen35.benchmark.repeat_divergence.v1",
        "benchmark": "context",
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
    first_diff
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

impl BenchmarkMode {
    fn parse(args: &[String]) -> Result<Self, Box<dyn std::error::Error>> {
        match string_arg(args, "--mode", "both").as_str() {
            "scalar" => Ok(Self::Scalar),
            "mtp" => Ok(Self::Mtp),
            "both" => Ok(Self::Both),
            value => Err(format!("--mode must be scalar, mtp, or both; got {value}").into()),
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

struct ModeResult {
    canonical_tokens: Vec<u32>,
    summary: Value,
}

fn run_mode(
    model: &mut Qwen35Moe,
    prefix: &[u32],
    chunk: &[u32],
    max_new_tokens: usize,
    mode: Mode,
    warmups: usize,
    repeats: usize,
    autotune: bool,
    profiling: bool,
    allow_repeat_divergence: bool,
) -> Result<ModeResult, Box<dyn std::error::Error>> {
    let options = mode.options(max_new_tokens);
    let mut prefill_ms = Vec::with_capacity(repeats);
    let mut prefill_tps = Vec::with_capacity(repeats);
    let mut decode_ms = Vec::with_capacity(repeats);
    let mut decode_tps = Vec::with_capacity(repeats);
    let mut canonical_tokens: Option<Vec<u32>> = None;
    let mut speculation = Vec::new();
    let mut divergent_repeats = 0;
    let mut divergent_warmups = 0;

    for iteration in 0..warmups + repeats {
        let warmup = iteration < warmups;
        match mode {
            Mode::Scalar => model.benchmark_prepare_scalar_prefix(prefix)?,
            Mode::Mtp => model.benchmark_prepare_mtp_prefix(prefix, &options)?,
        }

        let prepared_prefix_tokens = model.benchmark_context_position();
        if prepared_prefix_tokens != prefix.len() {
            return Err("prepared KV length does not match requested prefix".into());
        }
        let profile_range = cuda_profile::CudaProfileRange::start(profiling && !warmup)?;
        let prefill_start = Instant::now();
        let first_token = match mode {
            Mode::Scalar => model.benchmark_scalar_prefill_chunk(chunk)?,
            Mode::Mtp => model.benchmark_mtp_prefill_chunk(chunk, &options)?,
        };
        let current_prefill_ms = prefill_start.elapsed().as_secs_f64() * 1_000.0;
        let committed_context_tokens = model.benchmark_context_position();
        if committed_context_tokens != prefix.len() + chunk.len() {
            return Err("prefill did not commit exactly the new input tokens".into());
        }

        let decode_start = Instant::now();
        let tokens = if max_new_tokens == 1 {
            vec![first_token]
        } else {
            match mode {
                Mode::Scalar => model.benchmark_scalar_decode(first_token, max_new_tokens)?,
                Mode::Mtp => model.benchmark_mtp_decode(first_token, max_new_tokens)?,
            }
        };
        let current_decode_ms =
            (max_new_tokens > 1).then(|| decode_start.elapsed().as_secs_f64() * 1_000.0);
        profile_range.finish()?;
        let current_prefill_tps = chunk.len() as f64 / (current_prefill_ms / 1_000.0);
        let current_decode_tps =
            current_decode_ms.map(|ms| (max_new_tokens - 1) as f64 / (ms / 1_000.0));
        let current_speculation = match mode {
            Mode::Scalar => None,
            Mode::Mtp => model.speculative_generation_stats(),
        };

        let mut repeat_diverged = false;
        if let Some(expected) = &canonical_tokens {
            if expected != &tokens {
                let first_diff = report_repeat_divergence(
                    mode.name(), iteration, warmup, allow_repeat_divergence, expected, &tokens,
                );
                if !allow_repeat_divergence {
                    return Err(format!("{} output changed across repeats at token {first_diff}", mode.name()).into());
                }
                repeat_diverged = true;
                if warmup {
                    divergent_warmups += 1;
                } else {
                    divergent_repeats += 1;
                }
            }
        } else {
            canonical_tokens = Some(tokens.clone());
        }

        println!(
            "{}",
            json!({
                "schema": "apxinf.qwen35.context_benchmark.iteration.v1",
                "mode": mode.name(),
                "iteration": iteration,
                "warmup": warmup,
                "autotune": autotune,
                "performance_eligible": !autotune && !profiling,
                "profiling": profiling,
                "strict_recipe_required": !autotune,
                "recipe_audit_required": true,
                "prefix_tokens": prefix.len(),
                "initial_kv_tokens": prepared_prefix_tokens,
                "computed_input_tokens": committed_context_tokens - prepared_prefix_tokens,
                "committed_context_tokens_after_prefill": committed_context_tokens,
                "input_tokens": chunk.len(),
                "total_context_tokens": prefix.len() + chunk.len(),
                "output_tokens": tokens.len(),
                "decode_intervals": tokens.len().saturating_sub(1),
                "repeat_diverged": repeat_diverged,
                "token_digest_fnv1a64": format!("{:016x}", token_digest(&tokens)),
                "prefill_ms": current_prefill_ms,
                "prefill_tokens_per_second": current_prefill_tps,
                "decode_ms": current_decode_ms,
                "decode_tokens_per_second": current_decode_tps,
                "speculation": current_speculation,
            })
        );

        if !warmup {
            prefill_ms.push(current_prefill_ms);
            prefill_tps.push(current_prefill_tps);
            if let Some(ms) = current_decode_ms {
                decode_ms.push(ms);
            }
            if let Some(tps) = current_decode_tps {
                decode_tps.push(tps);
            }
            if let Some(stats) = current_speculation {
                speculation.push(stats);
            }
        }
    }

    let summary = json!({
        "schema": "apxinf.qwen35.context_benchmark.summary.v1",
        "mode": mode.name(),
        "autotune": autotune,
        "performance_eligible": !autotune && !profiling,
                "profiling": profiling,
                "strict_recipe_required": !autotune,
        "recipe_audit_required": true,
        "allow_repeat_divergence": allow_repeat_divergence,
        "divergent_repeats": divergent_repeats,
        "divergent_warmups": divergent_warmups,
        "decode_intervals": max_new_tokens - 1,
        "prefix_tokens": prefix.len(),
        "input_tokens": chunk.len(),
        "total_context_tokens": prefix.len() + chunk.len(),
        "output_tokens": max_new_tokens,
        "warmups": warmups,
        "repeats": repeats,
        "prefill_ms": {"median": quantile(&prefill_ms, 0.5), "p95": quantile(&prefill_ms, 0.95)},
        "prefill_tokens_per_second": {"median": quantile(&prefill_tps, 0.5), "p95": quantile(&prefill_tps, 0.95)},
        "decode_ms": (!decode_ms.is_empty()).then(|| json!({"median": quantile(&decode_ms, 0.5), "p95": quantile(&decode_ms, 0.95)})),
        "decode_tokens_per_second": (!decode_tps.is_empty()).then(|| json!({"median": quantile(&decode_tps, 0.5), "p95": quantile(&decode_tps, 0.95)})),
        "speculation": speculation,
    });
    Ok(ModeResult {
        canonical_tokens: canonical_tokens.expect("at least one iteration"),
        summary,
    })
}

fn string_arg(args: &[String], name: &str, default: &str) -> String {
    args.iter()
        .position(|arg| arg == name)
        .and_then(|index| args.get(index + 1))
        .map_or_else(|| default.to_owned(), Clone::clone)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let checkpoint = args.get(1).filter(|arg| !arg.starts_with("--")).ok_or(
        "usage: qwen35_context_bench <checkpoint-dir> [--autotune] --tuning-dir ABSOLUTE_PATH \
         --prefix-tokens N --input-tokens N [--kv-capacity N] [--max-new N] [--warmups N] \
         [--repeats N] [--mode scalar|mtp|both]",
    )?;
    let autotune = args.iter().any(|arg| arg == "--autotune");
    let profiling = args.iter().any(|arg| arg == "--profile-measured");
    // Set before constructing any runtime: formal preparation must not silently
    // fall back when the tuning pass omitted a needed shape.
    std::env::set_var("APXINF_REQUIRE_RECIPES", if autotune { "0" } else { "1" });
    let allow_repeat_divergence = args.iter().any(|arg| arg == "--allow-repeat-divergence");
    let tuning_path = benchmark::optional_path(&args, "--tuning-dir")?;
    benchmark::validate_tuning_path(&tuning_path, autotune)?;
    let moe_activation: Qwen35MoeActivation =
        string_arg(&args, "--qwen35-moe-activation", "nvfp4").parse()?;
    let kv_cache: Qwen35KvCacheDType = string_arg(&args, "--qwen35-kv-cache", "bf16").parse()?;
    let prefix_tokens = usize_arg(&args, "--prefix-tokens", 0)?;
    let input_tokens = usize_arg(&args, "--input-tokens", 2_048)?;
    let max_new_tokens = usize_arg(&args, "--max-new", DEFAULT_OUTPUT_TOKENS)?;
    let warmups = usize_arg(&args, "--warmups", DEFAULT_WARMUPS)?;
    let repeats = usize_arg(&args, "--repeats", DEFAULT_REPEATS)?;
    let device = usize_arg(&args, "--device", 0)?;
    let benchmark_mode = BenchmarkMode::parse(&args)?;
    if profiling && (autotune || repeats != 1 || matches!(benchmark_mode, BenchmarkMode::Both)) {
        return Err("--profile-measured requires one non-autotune repeat and one --mode".into());
    }
    if input_tokens == 0 || max_new_tokens == 0 || repeats == 0 {
        return Err("input tokens must be positive, max-new >= 1, and repeats >= 1".into());
    }
    let total_context = prefix_tokens
        .checked_add(input_tokens)
        .ok_or("context length overflow")?;
    let minimum_capacity = total_context
        .checked_add(max_new_tokens)
        .and_then(|value| value.checked_add(MTP_LOOKAHEAD))
        .ok_or("KV capacity overflow")?;
    let capacity = usize_arg(&args, "--kv-capacity", minimum_capacity)?;
    if capacity < minimum_capacity {
        return Err(format!(
            "--kv-capacity must be at least {minimum_capacity} for this prompt/output shape"
        )
        .into());
    }

    let prompt = exact_tokens(total_context);
    let (prefix, chunk) = prompt.split_at(prefix_tokens);
    // Qwen3.6 owns its cuda-new execution context internally. The trait
    // backend is retained only for the generic LLM surface, so the benchmark
    // must not pull the unrelated legacy CUDA runtime into a qwen35-cuda build.
    let backend: Arc<dyn Backend> = Arc::new(CpuBackend);
    let config = Qwen35MoeConfig::from_json_file(&Path::new(checkpoint).join("config.json"))?;
    let load_start = Instant::now();
    let scales = if kv_cache == Qwen35KvCacheDType::Fp8 {
        Some(load_trt_fp8_kv_cache_scales(
            &Path::new(checkpoint).join("hf_quant_config.json"),
        )?)
    } else {
        None
    };
    let mut model = Qwen35Moe::from_checkpoint_path_with_runtime_options(
        config,
        Path::new(checkpoint),
        Device::Cuda(device),
        backend,
        capacity,
        Qwen35ExecutionOptions {
            moe_activation,
            kv_cache,
        },
        autotune,
        tuning_path.clone(),
        scales,
    )?;
    model.set_context_reuse(false)?;
    println!(
        "{}",
        json!({
            "schema": "apxinf.qwen35.context_benchmark.config.v1",
            "checkpoint": checkpoint,
            "device": device,
            "autotune": autotune,
            "performance_eligible": !autotune && !profiling,
                "profiling": profiling,
                "strict_recipe_required": !autotune,
            "recipe_audit_required": true,
            "tuning_dir": tuning_path,
            "qwen35_moe_activation": moe_activation.as_str(),
            "qwen35_kv_cache": kv_cache.as_str(),
            "temperature": 0.0,
            "sampling_mode": "greedy",
            "context_reuse": false,
            "mtp_enabled": !matches!(benchmark_mode, BenchmarkMode::Scalar),
            "mtp_strategy": if matches!(benchmark_mode, BenchmarkMode::Scalar) { "disabled" } else { "fixed" },
            "mtp_depth": (!matches!(benchmark_mode, BenchmarkMode::Scalar)).then_some(MTP_LOOKAHEAD),
            "draft_kv_dtype": "fp8",
            "draft_kv_scale": 1.0,
            "fixed_depth": (!matches!(benchmark_mode, BenchmarkMode::Scalar)).then_some(MTP_LOOKAHEAD),
            "allow_repeat_divergence": allow_repeat_divergence,
            "input_seed_hex": format!("{:016x}", INPUT_SEED),
            "input_digest_fnv1a64": format!("{:016x}", token_digest(&prompt)),
            "prefix_digest_fnv1a64": format!("{:016x}", token_digest(prefix)),
            "chunk_digest_fnv1a64": format!("{:016x}", token_digest(chunk)),
            "timing_scope": "prefix setup excluded; prefill includes first argmax token; decode excludes first token",
            "prefix_tokens": prefix_tokens,
            "input_tokens": input_tokens,
            "total_context_tokens": total_context,
            "kv_capacity": capacity,
            "output_tokens": max_new_tokens,
            "warmups": warmups,
            "repeats": repeats,
            "mode": benchmark_mode.name(),
            "load_seconds": load_start.elapsed().as_secs_f64(),
        })
    );

    match benchmark_mode {
        BenchmarkMode::Scalar => {
            let scalar = run_mode(
                &mut model,
                prefix,
                chunk,
                max_new_tokens,
                Mode::Scalar,
                warmups,
                repeats,
                autotune,
                profiling,
                allow_repeat_divergence,
            )?;
            println!("{}", scalar.summary);
        }
        BenchmarkMode::Mtp => {
            let mtp = run_mode(
                &mut model,
                prefix,
                chunk,
                max_new_tokens,
                Mode::Mtp,
                warmups,
                repeats,
                autotune,
                profiling,
                allow_repeat_divergence,
            )?;
            println!("{}", mtp.summary);
        }
        BenchmarkMode::Both => {
            let scalar = run_mode(
                &mut model,
                prefix,
                chunk,
                max_new_tokens,
                Mode::Scalar,
                warmups,
                repeats,
                autotune,
                profiling,
                allow_repeat_divergence,
            )?;
            let mtp = run_mode(
                &mut model,
                prefix,
                chunk,
                max_new_tokens,
                Mode::Mtp,
                warmups,
                repeats,
                autotune,
                profiling,
                allow_repeat_divergence,
            )?;
            if scalar.canonical_tokens != mtp.canonical_tokens {
                return Err("MTP token IDs differ from forced-scalar greedy token IDs".into());
            }
            println!("{}", scalar.summary);
            println!("{}", mtp.summary);
            println!(
                "{}",
                json!({
                    "schema": "apxinf.qwen35.context_benchmark.parity.v1",
                    "token_ids_equal": true,
                    "compared_tokens": max_new_tokens,
                })
            );
        }
    }
    Ok(())
}
