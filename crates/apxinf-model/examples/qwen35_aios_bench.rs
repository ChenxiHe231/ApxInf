//! AIOS scenario matrix for Qwen3.6-35B-A3B on Thor-U.
//!
//! Three cases, each reported as one JSON line so a run can be resumed or
//! diffed:
//!
//!   cold-prefill   no KV cache, prefill N from zero       (N = 2k/4k/8k/16k)
//!   ctx-prefill    KV cache already holds P, prefill N    (P = 4k..48k, N = 4k)
//!   ctx-decode     KV cache already holds P, decode N     (P = 4k..48k, N = 512)
//!
//! The prefix is established with one `forward(prompt, 0)` and is deliberately
//! excluded from the measurement; only the second phase is timed. That is what
//! separates "already have a cache" from "pay for the cache now".
//!
//! Token IDs are generated directly, so no tokenizer version can change the
//! measured length.

use std::time::Instant;

use apxinf_core::Device;
use apxinf_model::qwen35moe::{Qwen35KvCacheDType, Qwen35MoeActivation};
use apxinf_model::{AutoModel, LoadOptions, LoadedModel};
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

fn arg<T: std::str::FromStr + Default>(args: &[String], name: &str) -> T
where
    T::Err: std::fmt::Debug,
{
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .map(|v| v.parse::<T>().expect("bad numeric argument"))
        .unwrap_or_default()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let checkpoint = args
        .get(1)
        .filter(|a| !a.starts_with("--"))
        .ok_or("usage: qwen35_aios_bench <checkpoint> --case cold|ctx-prefill|ctx-decode --prefix N --input N --repeats N")?;
    let case = args
        .iter()
        .position(|a| a == "--case")
        .and_then(|i| args.get(i + 1))
        .cloned()
        .ok_or("--case is required")?;
    let prefix_tokens: usize = arg(&args, "--prefix");
    let input_tokens: usize = arg(&args, "--input");
    let mut repeats: usize = arg(&args, "--repeats");
    if repeats == 0 {
        repeats = 3;
    }

    // Capacity must cover the prefix plus whatever the measured phase adds.
    let capacity = prefix_tokens + input_tokens + 8;
    let options = LoadOptions {
        text_kv_capacity: Some(capacity),
        qwen35_moe_activation: Some(Qwen35MoeActivation::Nvfp4),
        qwen35_kv_cache: Some(Qwen35KvCacheDType::Bf16),
        ..LoadOptions::default()
    };
    let load_start = Instant::now();
    let mut model = AutoModel::load_model(Device::Cuda(0), checkpoint, &options)?;
    println!(
        "{}",
        json!({
            "schema": "apxinf.qwen35.aios.config.v1",
            "case": case, "prefix_tokens": prefix_tokens,
            "input_tokens": input_tokens, "capacity": capacity,
            "repeats": repeats,
            "load_seconds": load_start.elapsed().as_secs_f64(),
        })
    );
    run(&mut model, &case, prefix_tokens, input_tokens, repeats)
}

fn run(
    model: &mut LoadedModel,
    case: &str,
    prefix_tokens: usize,
    input_tokens: usize,
    repeats: usize,
) -> Result<(), Box<dyn std::error::Error>> {
    let prefix = tokens(prefix_tokens.max(1), 0x5EED_0001);
    let input = tokens(input_tokens.max(1), 0x5EED_0002);

    for iteration in 0..repeats {
        let mut record = json!({
            "schema": "apxinf.qwen35.aios.iteration.v1",
            "case": case, "iteration": iteration,
            "prefix_tokens": prefix_tokens, "input_tokens": input_tokens,
        });
        let map = record.as_object_mut().unwrap();

        match case {
            "cold" => {
                // No prefix. Time one prefill of `input_tokens` from an empty cache.
                model.reset()?;
                let t = Instant::now();
                model.forward(&input, 0)?;
                let seconds = t.elapsed().as_secs_f64();
                map.insert("prefill_seconds".into(), json!(seconds));
                map.insert(
                    "prefill_tokens_per_second".into(),
                    json!(input_tokens as f64 / seconds),
                );
            }
            "ctx-prefill" => {
                // Establish the prefix untimed, then time the incremental prefill.
                model.reset()?;
                model.forward(&prefix, 0)?;
                let t = Instant::now();
                model.forward(&input, prefix_tokens as u32)?;
                let seconds = t.elapsed().as_secs_f64();
                map.insert("prefill_seconds".into(), json!(seconds));
                map.insert(
                    "prefill_tokens_per_second".into(),
                    json!(input_tokens as f64 / seconds),
                );
            }
            "ctx-decode" => {
                // Establish the prefix untimed, then time `input_tokens` single-token steps.
                model.reset()?;
                model.forward(&prefix, 0)?;
                let mut position = prefix_tokens as u32;
                let mut token = input[0];
                let t = Instant::now();
                for _ in 0..input_tokens {
                    model.forward(&[token], position)?;
                    position += 1;
                    token = input[(position as usize) % input.len()];
                }
                let seconds = t.elapsed().as_secs_f64();
                map.insert("decode_seconds".into(), json!(seconds));
                map.insert(
                    "decode_tokens_per_second".into(),
                    json!(input_tokens as f64 / seconds),
                );
            }
            other => return Err(format!("unknown --case {other}").into()),
        }
        println!("{record}");
    }
    Ok(())
}
