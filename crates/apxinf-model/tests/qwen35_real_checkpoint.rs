#![cfg(feature = "cuda")]

use std::path::{Path, PathBuf};

use apxinf_core::Device;
use apxinf_model::{
    AutoModel, GenerationOptions, LlmInput, LoadOptions, SamplingMode, SpeculativeGenerationStats,
};

const DEFAULT_OUTPUT_TOKENS: usize = 32;
const OUTPUT_TOKENS_ENV: &str = "APXINF_QWEN35_PARITY_TOKENS";
const PROMPT_TOKENS_ENV: &str = "APXINF_QWEN35_PARITY_PROMPT_TOKENS";
const DIAGNOSTICS_ENV: &str = "APXINF_QWEN35_PARITY_DIAGNOSTICS";

fn checkpoint() -> PathBuf {
    std::env::var_os("APXINF_QWEN35_CHECKPOINT")
        .map(PathBuf::from)
        .expect("set APXINF_QWEN35_CHECKPOINT to Qwen3.5-35B-A3B-NVFP4")
}

fn cuda_device() -> usize {
    std::env::var("APXINF_CUDA_DEVICE")
        .ok()
        .map(|value| {
            value
                .parse()
                .expect("APXINF_CUDA_DEVICE must be an integer")
        })
        .unwrap_or(0)
}

fn prompt() -> Vec<u32> {
    // Token IDs stay below the exact checkpoint vocabulary (248,320) and do
    // not depend on a tokenizer implementation. A non-trivial prompt also
    // exercises shifted MTP teacher forcing.
    let base = vec![
        151_644, 8948, 198, 9707, 16_357, 220, 19_687, 25, 198, 151_645, 198, 151_644, 77091, 198,
        35_064, 16_933,
    ];
    let Some(target) = std::env::var(PROMPT_TOKENS_ENV).ok().map(|value| {
        value
            .parse::<usize>()
            .unwrap_or_else(|_| panic!("{PROMPT_TOKENS_ENV} must be a positive integer"))
    }) else {
        return base;
    };
    assert!(target > 0, "{PROMPT_TOKENS_ENV} must be positive");
    // Repeat the pattern to the requested length. Length selects the prefill
    // path: a multiple of the GDN scan chunk takes the chunked kernel, anything
    // else takes the exact per-token recurrence, so this is how the two are
    // compared against the same MTP protocol.
    base.into_iter().cycle().take(target).collect()
}

fn output_tokens() -> usize {
    let output_tokens = std::env::var(OUTPUT_TOKENS_ENV)
        .ok()
        .map(|value| {
            value
                .parse::<usize>()
                .unwrap_or_else(|_| panic!("{OUTPUT_TOKENS_ENV} must be a positive integer"))
        })
        .unwrap_or(DEFAULT_OUTPUT_TOKENS);
    assert!(
        output_tokens >= 2,
        "{OUTPUT_TOKENS_ENV} must be at least 2 to exercise one speculative round"
    );
    output_tokens
}

fn mtp_options(output_tokens: usize) -> GenerationOptions {
    GenerationOptions::greedy(output_tokens, None)
}

fn forced_scalar_options(output_tokens: usize) -> GenerationOptions {
    GenerationOptions {
        speculative_decoding: Some(false),
        sampling_mode: Some(SamplingMode::Greedy),
        ..GenerationOptions::greedy(output_tokens, None)
    }
}

fn print_stats(stats: &SpeculativeGenerationStats) {
    eprintln!(
        "Qwen3.5 MTP diagnostics: {}",
        serde_json::to_string(stats).expect("serialize MTP diagnostics")
    );
}

fn write_diagnostics(
    path: &Path,
    checkpoint: &Path,
    output_tokens: usize,
    stats: &SpeculativeGenerationStats,
    mtp_token_ids: &[u32],
    scalar_token_ids: &[u32],
) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        std::fs::create_dir_all(parent)?;
    }
    let diagnostics = serde_json::json!({
        "schema": "apxinf.qwen35.real_checkpoint_parity.v1",
        "checkpoint": checkpoint,
        "output_tokens": output_tokens,
        "token_ids_equal": mtp_token_ids == scalar_token_ids,
        "mtp_token_ids": mtp_token_ids,
        "scalar_token_ids": scalar_token_ids,
        "acceptance_rounds": {
            "zero": stats.zero_accept_rounds,
            "partial": stats.partial_accept_rounds,
            "full": stats.full_accept_rounds,
            "total": stats.rounds,
        },
        "rollback": {
            "target_verified_rows": stats.target_verified_rows,
            "target_committed_rows": stats.target_committed_rows,
            "target_rolled_back_rows": stats.target_rolled_back_rows,
        },
        "speculation": stats,
    });
    std::fs::write(path, serde_json::to_vec_pretty(&diagnostics)?)?;
    eprintln!("saved Qwen3.5 parity diagnostics to {}", path.display());
    Ok(())
}

/// Real-checkpoint token parity gate. This is ignored because it loads the
/// complete 35B NVFP4 checkpoint and requires a CUDA device with enough
/// visible memory. The model is loaded exactly once; both requests reuse it.
#[test]
#[ignore = "requires Qwen3.5-35B-A3B-NVFP4 and a large-memory CUDA device"]
fn neutral_mtp_matches_forced_scalar_greedy() -> Result<(), Box<dyn std::error::Error>> {
    let prompt = prompt();
    let output_tokens = output_tokens();
    let options = LoadOptions {
        text_kv_capacity: Some(prompt.len() + output_tokens + 3),
        ..LoadOptions::default()
    };
    let checkpoint = checkpoint();
    let mut model =
        AutoModel::load_model(Device::Cuda(cuda_device()), checkpoint.clone(), &options)?;

    let mtp = model.generate_streaming_with_options(
        LlmInput::text(&prompt),
        &mtp_options(output_tokens),
        |_token| {},
    )?;
    eprintln!("Qwen3.5 MTP profile:\n{}", mtp.profile.summary());
    let stats = model
        .speculative_generation_stats()?
        .expect("neutral greedy request must expose real MTP counters");
    print_stats(&stats);
    assert!(
        stats.rounds > 0,
        "MTP request did not execute a draft round"
    );
    assert!(stats.proposed_tokens > 0, "MTP request proposed no tokens");
    assert_eq!(
        stats.zero_accept_rounds + stats.partial_accept_rounds + stats.full_accept_rounds,
        stats.rounds,
        "zero/partial/full acceptance diagnostics must partition all rounds"
    );
    assert!(
        stats.target_rolled_back_rows > 0,
        "the output budget must force a truncated or rejected verification window"
    );

    let scalar = model.generate_streaming_with_options(
        LlmInput::text(&prompt),
        &forced_scalar_options(output_tokens),
        |_token| {},
    )?;
    eprintln!("Qwen3.5 scalar profile:\n{}", scalar.profile.summary());
    assert!(
        model.speculative_generation_stats()?.is_none(),
        "forced-scalar request unexpectedly used MTP"
    );

    let mtp_token_ids = mtp.token_ids();
    let scalar_token_ids = scalar.token_ids();
    if let Some(path) = std::env::var_os(DIAGNOSTICS_ENV) {
        write_diagnostics(
            Path::new(&path),
            &checkpoint,
            output_tokens,
            &stats,
            &mtp_token_ids,
            &scalar_token_ids,
        )?;
    }
    assert_eq!(mtp.tokens.len(), output_tokens);
    assert_eq!(scalar.tokens.len(), output_tokens);
    // Report where the two sequences part company. Speculative decoding is
    // exact, so any divergence is a bug, and its position says which round of
    // the accept/rollback protocol produced it.
    if mtp_token_ids != scalar_token_ids {
        let first = mtp_token_ids
            .iter()
            .zip(scalar_token_ids.iter())
            .position(|(a, b)| a != b)
            .expect("sequences differ but no differing position found");
        let window = first.saturating_sub(3)..(first + 4).min(mtp_token_ids.len());
        let mtp_window = &mtp_token_ids[window.clone()];
        let scalar_window = &scalar_token_ids[window.clone()];
        panic!(
            "MTP diverged from forced-scalar greedy at output index {first} of {output_tokens} \
(prompt {} tokens, prompt % 64 = {}).\n  window {window:?}\n  mtp    = {mtp_window:?}\n  \
scalar = {scalar_window:?}\n  rounds={} zero={} partial={} full={} proposed={} committed={} \
rolled_back_rows={}",
            prompt.len(),
            prompt.len() % 64,
            stats.rounds,
            stats.zero_accept_rounds,
            stats.partial_accept_rounds,
            stats.full_accept_rounds,
            stats.proposed_tokens,
            stats.committed_accepted_tokens,
            stats.target_rolled_back_rows,
        );
    }
    Ok(())
}
