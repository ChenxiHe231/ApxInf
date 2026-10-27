//! Load-once Qwen3.5-35B-A3B NVFP4 scalar-versus-MTP benchmark.
//!
//! The prompt is exactly 4,096 deterministic token IDs. Each mode runs two
//! warmups followed by five measured iterations and emits exactly 256 tokens
//! unless a larger `--max-new` value is requested. EOS stopping is disabled.
//! The model and its CUDA allocations are loaded once for the whole process.

use std::cmp::Ordering;
use std::time::Instant;

use apxinf_core::Device;
use apxinf_model::qwen35moe::{Qwen35KvCacheDType, Qwen35MoeActivation};
use apxinf_model::{
    AutoModel, GenerationOptions, LlmInput, LoadOptions, LoadedModel, SamplingMode,
    SpeculativeGenerationStats,
};
use serde_json::{json, Value};

const VOCAB_SIZE: usize = 248_320;
const PROMPT_TOKENS: usize = 4_096;
const DEFAULT_OUTPUT_TOKENS: usize = 256;
const DEFAULT_WARMUPS: usize = 2;
const DEFAULT_REPEATS: usize = 5;
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
                // Requesting log probabilities changes only the returned
                // metadata, not greedy argmax token selection. MTP does not
                // expose log probabilities, so this forces the exact scalar
                // path while preserving token parity with MTP.
                return_logprob: Some(true),
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

fn exact_prompt() -> Vec<u32> {
    let mut state = 0x35A3_BF04_4096_0001u64;
    let mut tokens = Vec::with_capacity(PROMPT_TOKENS);
    for _ in 0..PROMPT_TOKENS {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        // Avoid the checkpoint's high special-token range. These are already
        // token IDs, so tokenizer version cannot change the measured length.
        tokens.push(((state >> 17) as usize % (VOCAB_SIZE - 1_000)) as u32);
    }
    tokens
}

fn quantile(values: &[f64], percentile: f64) -> f64 {
    assert!(!values.is_empty());
    let mut sorted = values.to_vec();
    sorted.sort_by(|left, right| left.partial_cmp(right).unwrap_or(Ordering::Equal));
    let index = ((percentile * sorted.len() as f64).ceil() as usize)
        .saturating_sub(1)
        .min(sorted.len() - 1);
    sorted[index]
}

fn run_mode(
    model: &mut LoadedModel,
    prompt: &[u32],
    max_new_tokens: usize,
    mode: Mode,
    warmups: usize,
    repeats: usize,
) -> Result<ModeResult, Box<dyn std::error::Error>> {
    let options = mode.options(max_new_tokens);
    let mut measured_ttft_ms = Vec::with_capacity(repeats);
    let mut measured_prefill_tps = Vec::with_capacity(repeats);
    let mut measured_generation_tps = Vec::with_capacity(repeats);
    let mut aggregate = AggregateSpeculation::default();
    let mut canonical_tokens: Option<Vec<u32>> = None;

    for iteration in 0..warmups + repeats {
        let warmup = iteration < warmups;
        let output =
            model.generate_streaming_with_options(LlmInput::text(prompt), &options, |_token| {})?;
        let token_ids = output.token_ids();
        if token_ids.len() != max_new_tokens {
            return Err(format!(
                "{} iteration {iteration} produced {} tokens, expected {max_new_tokens}",
                mode.name(),
                token_ids.len()
            )
            .into());
        }
        if let Some(expected) = &canonical_tokens {
            if expected != &token_ids {
                let mismatch = expected
                    .iter()
                    .zip(&token_ids)
                    .position(|(left, right)| left != right)
                    .unwrap_or(expected.len().min(token_ids.len()));
                return Err(format!(
                    "{} output changed across repeats at token {mismatch}: expected={:?}, actual={:?}, expected_digest={:016x}, actual_digest={:016x}",
                    mode.name(),
                    expected.get(mismatch),
                    token_ids.get(mismatch),
                    token_digest(expected),
                    token_digest(&token_ids),
                )
                .into());
            }
        } else {
            canonical_tokens = Some(token_ids.clone());
        }

        let ttft_ms = output
            .profile
            .ttft_ms()
            .ok_or("generation profile did not record TTFT")?;
        let generation_tps = output
            .profile
            .generation_tps()
            .ok_or("generation profile did not record generation TPS")?;
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
                "schema": "apxinf.qwen35.benchmark.iteration.v1",
                "mode": mode.name(),
                "iteration": iteration,
                "warmup": warmup,
                "prompt_tokens": prompt.len(),
                "output_tokens": token_ids.len(),
                "ttft_ms": ttft_ms,
                "prefill_tokens_per_second": prefill_tps,
                "generation_tokens_per_second": generation_tps,
                "token_digest_fnv1a64": format!("{:016x}", token_digest(&token_ids)),
                "speculation": speculation,
            })
        );

        if !warmup {
            measured_ttft_ms.push(ttft_ms);
            measured_prefill_tps.push(prefill_tps);
            measured_generation_tps.push(generation_tps);
        }
    }

    let summary = json!({
        "schema": "apxinf.qwen35.benchmark.summary.v1",
        "mode": mode.name(),
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
        "generation_tokens_per_second": {
            "median": quantile(&measured_generation_tps, 0.5),
            "p95": quantile(&measured_generation_tps, 0.95),
        },
        "speculation": match mode {
            Mode::Mtp => aggregate.to_json(),
            Mode::Scalar => Value::Null,
        },
    });
    Ok(ModeResult {
        canonical_tokens: canonical_tokens.expect("at least one iteration"),
        summary,
    })
}

fn token_digest(tokens: &[u32]) -> u64 {
    tokens.iter().fold(0xcbf29ce484222325u64, |hash, token| {
        token.to_le_bytes().into_iter().fold(hash, |hash, byte| {
            (hash ^ u64::from(byte)).wrapping_mul(0x100000001b3)
        })
    })
}

fn arg_value(args: &[String], name: &str, default: usize) -> usize {
    args.iter()
        .position(|arg| arg == name)
        .and_then(|index| args.get(index + 1))
        .map(|value| value.parse().expect(name))
        .unwrap_or(default)
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
             [--warmups N] [--repeats N] [--mode scalar|mtp|both] \
             [--qwen35-moe-activation nvfp4|bf16] [--qwen35-kv-cache bf16|fp8]",
    )?;
    let device = arg_value(&args, "--device", 0);
    let max_new_tokens = arg_value(&args, "--max-new", DEFAULT_OUTPUT_TOKENS);
    let warmups = arg_value(&args, "--warmups", DEFAULT_WARMUPS);
    let repeats = arg_value(&args, "--repeats", DEFAULT_REPEATS);
    let benchmark_mode = BenchmarkMode::parse(&args)?;
    let moe_activation = precision_arg::<Qwen35MoeActivation>(&args, "--qwen35-moe-activation")?;
    let kv_cache = precision_arg::<Qwen35KvCacheDType>(&args, "--qwen35-kv-cache")?;
    if max_new_tokens == 0 {
        return Err("--max-new must be at least 1".into());
    }
    if repeats == 0 {
        return Err("--repeats must be at least 1".into());
    }

    let prompt = exact_prompt();
    assert_eq!(prompt.len(), PROMPT_TOKENS);
    let options = LoadOptions {
        text_kv_capacity: Some(PROMPT_TOKENS + max_new_tokens + MTP_LOOKAHEAD),
        qwen35_moe_activation: Some(moe_activation),
        qwen35_kv_cache: Some(kv_cache),
        ..LoadOptions::default()
    };
    let load_start = Instant::now();
    let mut model = AutoModel::load_model(Device::Cuda(device), checkpoint, &options)?;
    println!(
        "{}",
        json!({
            "schema": "apxinf.qwen35.benchmark.config.v1",
            "checkpoint": checkpoint,
            "device": device,
            "prompt_tokens": PROMPT_TOKENS,
            "output_tokens": max_new_tokens,
            "warmups": warmups,
            "repeats": repeats,
            "mode": benchmark_mode.name(),
            "qwen35_moe_activation": moe_activation.as_str(),
            "qwen35_kv_cache": kv_cache.as_str(),
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
            )?;
            let mtp = run_mode(
                &mut model,
                &prompt,
                max_new_tokens,
                Mode::Mtp,
                warmups,
                repeats,
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
