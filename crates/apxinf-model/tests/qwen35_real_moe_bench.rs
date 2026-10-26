#![cfg(feature = "cuda")]

use std::path::PathBuf;
use std::time::Instant;

use apxinf_core::{DType, Tensor};
use apxinf_cuda_new::{transfers, CudaContext};
use apxinf_model::qwen35moe::config::Qwen35MoeConfig;
use apxinf_model::qwen35moe::device_weights::{
    load_qwen35_moe_layer_from_path, DeviceNvfp4RoutedExperts,
};
use apxinf_model::qwen35moe::forward::BaseMoeScratch;
use half::bf16;

const TOKENS_ENV: &str = "APXINF_QWEN35_MOE_BENCH_TOKENS";
const ITERATIONS_ENV: &str = "APXINF_QWEN35_MOE_BENCH_ITERS";

fn positive_env(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .map(|value| {
            value
                .parse::<usize>()
                .unwrap_or_else(|_| panic!("{name} must be a positive integer"))
        })
        .unwrap_or(default)
        .max(1)
}

fn checkpoint() -> PathBuf {
    std::env::var_os("APXINF_QWEN35_CHECKPOINT")
        .map(PathBuf::from)
        .expect("set APXINF_QWEN35_CHECKPOINT to Qwen3.5-35B-A3B-NVFP4")
}

#[test]
#[ignore = "benchmark; loads the complete Qwen3.5-35B-A3B NVFP4 checkpoint"]
fn real_checkpoint_first_moe_layer() -> Result<(), Box<dyn std::error::Error>> {
    let checkpoint = checkpoint();
    let config = Qwen35MoeConfig::from_json_file(&checkpoint.join("config.json"))?;
    let ctx = CudaContext::new(0)?;
    let load_started = Instant::now();
    let moe = load_qwen35_moe_layer_from_path(&ctx, &config, &checkpoint, 0)?;
    ctx.synchronize()?;
    eprintln!("Qwen3.5 layer-0 MoE load: {:?}", load_started.elapsed());

    assert!(
        matches!(&moe.routed_experts, DeviceNvfp4RoutedExperts::Edge(_)),
        "real checkpoint did not select the Edge NVFP4 routed-expert provider"
    );
    let tokens = positive_env(TOKENS_ENV, 1024);
    let iterations = positive_env(ITERATIONS_ENV, 10);
    let hidden = config.text_config.hidden_size;
    let input_values = (0..tokens * hidden)
        .map(|index| bf16::from_f32(((index % 31) as f32 - 15.0) / 64.0))
        .collect::<Vec<_>>();
    let input = transfers::to_cuda(
        &Tensor::from_bf16(vec![tokens, hidden], &input_values)?,
        ctx.device_id(),
    )?;
    let mut scratch = BaseMoeScratch::new(
        &ctx,
        tokens,
        hidden,
        config.text_config.num_experts,
        config.text_config.num_experts_per_tok,
        config.text_config.moe_intermediate_size,
    )?;

    scratch.run(&ctx, &moe, &input)?;
    ctx.synchronize()?;
    let started = Instant::now();
    for _ in 0..iterations {
        scratch.run(&ctx, &moe, &input)?;
    }
    ctx.synchronize()?;
    let elapsed = started.elapsed();
    let per_iteration = elapsed / iterations as u32;
    eprintln!(
        "Qwen3.5 real layer-0 MoE: tokens={tokens}, iterations={iterations}, {per_iteration:?}/iteration, {:.1} input tok/s",
        tokens as f64 / per_iteration.as_secs_f64()
    );

    let output = transfers::to_cpu(scratch.output())?;
    let output = output.as_bf16()?;
    assert_eq!(output.len(), tokens * hidden);
    assert!(output.iter().all(|value| value.to_f32().is_finite()));
    assert!(output.iter().any(|value| value.to_bits() & 0x7fff != 0));
    assert_eq!(scratch.output().dtype(), DType::BF16);
    Ok(())
}
