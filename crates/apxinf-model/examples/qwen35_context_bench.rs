//! Qwen3.5-35B-A3B NVFP4 chunk-prefill benchmark with an existing KV prefix.

use std::cmp::Ordering;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use apxinf_core::{Backend, CpuBackend, Device};
use apxinf_model::qwen35moe::{Qwen35Moe, Qwen35MoeConfig};
use apxinf_model::{GenerationOptions, LlmTrait, SamplingMode};
use serde_json::{json, Value};

const VOCAB_SIZE: usize = 248_320;
const DEFAULT_OUTPUT_TOKENS: usize = 32;
const DEFAULT_WARMUPS: usize = 1;
const DEFAULT_REPEATS: usize = 3;
const MTP_LOOKAHEAD: usize = 3;

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

fn exact_tokens(total: usize) -> Vec<u32> {
    let mut state = 0x35A3_BF04_4096_0001u64;
    let mut tokens = Vec::with_capacity(total);
    for _ in 0..total {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        tokens.push(((state >> 17) as usize % (VOCAB_SIZE - 1_000)) as u32);
    }
    tokens
}

fn quantile(values: &[f64], percentile: f64) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(|left, right| left.partial_cmp(right).unwrap_or(Ordering::Equal));
    let index = ((percentile * sorted.len() as f64).ceil() as usize)
        .saturating_sub(1)
        .min(sorted.len() - 1);
    sorted[index]
}

fn run_mode(
    model: &mut Qwen35Moe,
    prefix: &[u32],
    chunk: &[u32],
    max_new_tokens: usize,
    mode: Mode,
    warmups: usize,
    repeats: usize,
) -> Result<ModeResult, Box<dyn std::error::Error>> {
    let options = mode.options(max_new_tokens);
    let mut prefill_ms = Vec::with_capacity(repeats);
    let mut prefill_tps = Vec::with_capacity(repeats);
    let mut decode_ms = Vec::with_capacity(repeats);
    let mut decode_tps = Vec::with_capacity(repeats);
    let mut canonical_tokens: Option<Vec<u32>> = None;
    let mut speculation = Vec::new();

    for iteration in 0..warmups + repeats {
        let warmup = iteration < warmups;
        match mode {
            Mode::Scalar => model.benchmark_prepare_scalar_prefix(prefix)?,
            Mode::Mtp => model.benchmark_prepare_mtp_prefix(prefix, &options)?,
        }

        let prefill_start = Instant::now();
        let first_token = match mode {
            Mode::Scalar => model.benchmark_scalar_prefill_chunk(chunk)?,
            Mode::Mtp => model.benchmark_mtp_prefill_chunk(chunk, &options)?,
        };
        let current_prefill_ms = prefill_start.elapsed().as_secs_f64() * 1_000.0;

        let decode_start = Instant::now();
        let tokens = match mode {
            Mode::Scalar => model.benchmark_scalar_decode(first_token, max_new_tokens)?,
            Mode::Mtp => model.benchmark_mtp_decode(first_token, max_new_tokens)?,
        };
        let current_decode_ms = decode_start.elapsed().as_secs_f64() * 1_000.0;
        let current_prefill_tps = chunk.len() as f64 / (current_prefill_ms / 1_000.0);
        let current_decode_tps = (max_new_tokens - 1) as f64 / (current_decode_ms / 1_000.0);
        let current_speculation = match mode {
            Mode::Scalar => None,
            Mode::Mtp => model.speculative_generation_stats(),
        };

        if let Some(expected) = &canonical_tokens {
            if expected != &tokens {
                return Err(format!("{} output changed across repeats", mode.name()).into());
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
                "prefix_tokens": prefix.len(),
                "input_tokens": chunk.len(),
                "total_context_tokens": prefix.len() + chunk.len(),
                "output_tokens": max_new_tokens,
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
            decode_ms.push(current_decode_ms);
            decode_tps.push(current_decode_tps);
            if let Some(stats) = current_speculation {
                speculation.push(stats);
            }
        }
    }

    let summary = json!({
        "schema": "apxinf.qwen35.context_benchmark.summary.v1",
        "mode": mode.name(),
        "prefix_tokens": prefix.len(),
        "input_tokens": chunk.len(),
        "total_context_tokens": prefix.len() + chunk.len(),
        "output_tokens": max_new_tokens,
        "warmups": warmups,
        "repeats": repeats,
        "prefill_ms": {"median": quantile(&prefill_ms, 0.5), "p95": quantile(&prefill_ms, 0.95)},
        "prefill_tokens_per_second": {"median": quantile(&prefill_tps, 0.5), "p95": quantile(&prefill_tps, 0.95)},
        "decode_ms": {"median": quantile(&decode_ms, 0.5), "p95": quantile(&decode_ms, 0.95)},
        "decode_tokens_per_second": {"median": quantile(&decode_tps, 0.5), "p95": quantile(&decode_tps, 0.95)},
        "speculation": speculation,
    });
    Ok(ModeResult {
        canonical_tokens: canonical_tokens.expect("at least one iteration"),
        summary,
    })
}

fn usize_arg(args: &[String], name: &str, default: usize) -> usize {
    args.iter()
        .position(|arg| arg == name)
        .and_then(|index| args.get(index + 1))
        .map(|value| value.parse().expect(name))
        .unwrap_or(default)
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
        "usage: qwen35_context_bench <checkpoint-dir> --autotune \
         --prefix-tokens N --input-tokens N [--kv-capacity N] [--max-new N] [--warmups N] \
         [--repeats N] [--mode scalar|mtp|both]",
    )?;
    if !args.iter().any(|arg| arg == "--autotune") {
        return Err("qwen35_context_bench requires --autotune".into());
    }
    let prefix_tokens = usize_arg(&args, "--prefix-tokens", 0);
    let input_tokens = usize_arg(&args, "--input-tokens", 2_048);
    let max_new_tokens = usize_arg(&args, "--max-new", DEFAULT_OUTPUT_TOKENS);
    let warmups = usize_arg(&args, "--warmups", DEFAULT_WARMUPS);
    let repeats = usize_arg(&args, "--repeats", DEFAULT_REPEATS);
    let device = usize_arg(&args, "--device", 0);
    let benchmark_mode = BenchmarkMode::parse(&args)?;
    if input_tokens == 0 || max_new_tokens < 2 || repeats == 0 {
        return Err("input tokens must be positive, max-new >= 2, and repeats >= 1".into());
    }
    let total_context = prefix_tokens
        .checked_add(input_tokens)
        .ok_or("context length overflow")?;
    let minimum_capacity = total_context
        .checked_add(max_new_tokens)
        .and_then(|value| value.checked_add(MTP_LOOKAHEAD))
        .ok_or("KV capacity overflow")?;
    let capacity = usize_arg(&args, "--kv-capacity", minimum_capacity);
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
    let mut model = Qwen35Moe::from_checkpoint_path(
        config,
        Path::new(checkpoint),
        Device::Cuda(device),
        backend,
        capacity,
    )?;
    println!(
        "{}",
        json!({
            "schema": "apxinf.qwen35.context_benchmark.config.v1",
            "checkpoint": checkpoint,
            "device": device,
            "autotune": true,
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
            )?;
            let mtp = run_mode(
                &mut model,
                prefix,
                chunk,
                max_new_tokens,
                Mode::Mtp,
                warmups,
                repeats,
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
