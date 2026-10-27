//! Case B of the AIOS matrix: a KV cache already holds `--prefix` tokens and a
//! further `--input` tokens must be prefilled.
//!
//! Measured through the ordinary generate path so the number is comparable to
//! the TTFT reported for the other cases. Iteration 0 populates the reusable
//! prefix and is reported separately as the cold cost; later iterations are the
//! steady-state incremental cost, which is the thing Case B is about.
//!
//! `--no-reuse` measures the same shape with context reuse disabled. That is
//! the apples-to-apples number against the shipped TensorRT engine, whose
//! server reports `context_reuse=False`.

use std::time::Instant;

use apxinf_core::Device;
use apxinf_model::qwen35moe::{Qwen35KvCacheDType, Qwen35MoeActivation};
use apxinf_model::{AutoModel, GenerationOptions, LlmInput, LoadOptions, SamplingMode};
use serde_json::json;

const VOCAB_SIZE: usize = 248_320;

fn tokens(count: usize, seed: u64) -> Vec<u32> {
    let mut state = seed;
    let mut out = Vec::with_capacity(count);
    for _ in 0..count {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        out.push(((state >> 17) as usize % (VOCAB_SIZE - 1_000)) as u32);
    }
    out
}

fn arg(args: &[String], name: &str, default: usize) -> usize {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let checkpoint = args
        .get(1)
        .filter(|a| !a.starts_with("--"))
        .ok_or("usage: qwen35_aios_ctx_prefill <checkpoint> --prefix N --input N [--repeats N] [--no-reuse]")?;
    let prefix_tokens = arg(&args, "--prefix", 4096);
    let input_tokens = arg(&args, "--input", 4096);
    let repeats = arg(&args, "--repeats", 3).max(1);
    let reuse = !args.iter().any(|a| a == "--no-reuse");

    let total = prefix_tokens + input_tokens;
    let options = LoadOptions {
        text_kv_capacity: Some(total + 8),
        qwen35_moe_activation: Some(Qwen35MoeActivation::Nvfp4),
        qwen35_kv_cache: Some(Qwen35KvCacheDType::Bf16),
        ..LoadOptions::default()
    };
    let load_start = Instant::now();
    let mut model = AutoModel::load_model(Device::Cuda(0), checkpoint, &options)?;
    model.set_context_reuse(reuse)?;
    println!(
        "{}",
        json!({
            "schema": "apxinf.qwen35.aios.ctx_prefill.config.v1",
            "prefix_tokens": prefix_tokens, "input_tokens": input_tokens,
            "total_prompt_tokens": total, "context_reuse": reuse,
            "repeats": repeats,
            "load_seconds": load_start.elapsed().as_secs_f64(),
        })
    );

    // The prefix is byte-identical across iterations; only the tail differs,
    // which is exactly the shape context reuse is designed for.
    let mut prompt = tokens(prefix_tokens, 0x5EED_0001);
    prompt.extend(tokens(input_tokens, 0x5EED_0002));

    let gen = GenerationOptions {
        max_new_tokens: Some(2),
        sampling_mode: Some(SamplingMode::Greedy),
        ..GenerationOptions::default()
    };

    for iteration in 0..repeats {
        let t = Instant::now();
        let out = model.generate_streaming_with_options(
            LlmInput::text_with_reusable_prefix(&prompt, prefix_tokens),
            &gen,
            |_| {},
        )?;
        let wall = t.elapsed().as_secs_f64();
        let ttft = out.profile.ttft_ms();
        println!(
            "{}",
            json!({
                "schema": "apxinf.qwen35.aios.ctx_prefill.iteration.v1",
                "iteration": iteration,
                "cold": iteration == 0,
                "prefix_tokens": prefix_tokens,
                "input_tokens": input_tokens,
                "context_reuse": reuse,
                "ttft_ms": ttft,
                "wall_seconds": wall,
                // tokens/s attributed to the newly prefilled tail only
                "incremental_prefill_tok_s": ttft.map(|ms| input_tokens as f64 / (ms / 1000.0)),
            })
        );
    }
    Ok(())
}
