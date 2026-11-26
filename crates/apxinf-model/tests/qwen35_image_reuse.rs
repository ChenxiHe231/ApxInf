#![cfg(feature = "qwen35-cuda")]
//! Real-checkpoint regression for media identity and target/GDN/MTP restoration.
//! Run explicitly with both shared GPU locks, prepared recipes and Graphs.
use apxinf_core::{Device, Tensor};
use apxinf_model::qwen35moe::{Qwen35KvCacheDType, Qwen35MoeActivation};
use apxinf_model::{
    AutoModel, GenerationConfigSource, GenerationOptions, ImageInput, LlmInput, LoadOptions,
    LoadedModel, SamplingMode,
};
use std::{fs, path::PathBuf};
#[path = "../examples/support/qwen35_cuda_profile.rs"]
mod cuda_profile;

fn env(name: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| panic!("set {name}"))
}

fn response(tokens: &[u32]) -> &[u32] {
    &tokens[..tokens
        .iter()
        .position(|&t| t == 248046)
        .map_or(tokens.len(), |i| i + 1)]
}

fn generate(
    model: &mut LoadedModel,
    tokens: &[u32],
    pixels: &Tensor,
    grids: &[[u32; 3]],
    prefix: usize,
    options: &GenerationOptions,
    label: &str,
) -> (Vec<u32>, usize) {
    // Isolated diagnostics only. Stop at the first token so decode work is
    // not misattributed to image encoding or prompt restoration.
    let mut capture = (std::env::var("APXINF_QWEN35_IMAGE_PROFILE_LABEL").as_deref() == Ok(label))
        .then(|| cuda_profile::CudaProfileRange::start(true).unwrap());
    let output = model
        .generate_streaming_with_options(
            LlmInput::with_image_and_reusable_prefix(
                tokens,
                ImageInput::new(pixels, grids),
                prefix,
            ),
            options,
            |_| {
                if let Some(range) = capture.take() {
                    range.finish().unwrap();
                }
            },
        )
        .unwrap();
    let reused = output.profile.reused_input_tokens();
    println!(
        "{}",
        serde_json::json!({"case": label, "prompt_tokens":tokens.len(),
        "reused_tokens":reused, "ttft_ms":output.profile.ttft_ms(),
        "total_ms":output.profile.total_latency_ms(), "tokens":output.token_ids()})
    );
    (output.token_ids(), reused)
}

#[test]
#[ignore = "requires CUDA checkpoint, processed image fixture and prepared recipes"]
fn image_context_reuse_preserves_generation_and_rejects_changed_pixels() {
    let fixture = fs::read(env("APXINF_QWEN35_IMAGE_FIXTURE")).unwrap();
    let end = 4 + u32::from_le_bytes(fixture[..4].try_into().unwrap()) as usize;
    let metadata: serde_json::Value = serde_json::from_slice(&fixture[4..end]).unwrap();
    let original: Vec<u32> = serde_json::from_value(metadata["tokens"].clone()).unwrap();
    let grids: Vec<[u32; 3]> = serde_json::from_value(metadata["grids"].clone()).unwrap();
    let shape: Vec<usize> = serde_json::from_value(metadata["shape"].clone()).unwrap();
    let values: Vec<f32> = fixture[end..]
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect();
    let pixels = Tensor::from_f32(shape.clone(), &values).unwrap();
    let changed = Tensor::from_f32(
        shape,
        &values.iter().map(|v| -*v + 0.25).collect::<Vec<_>>(),
    )
    .unwrap();
    let prefix: usize = std::env::var("APXINF_QWEN35_IMAGE_PREFIX")
        .unwrap_or("0".into())
        .parse()
        .unwrap();
    let tokens = original.clone();
    let kv: Qwen35KvCacheDType = env("APXINF_QWEN35_IMAGE_KV").parse().unwrap();
    let prepare = std::env::var_os("APXINF_QWEN35_IMAGE_PREPARE").is_some();
    if !prepare {
        std::env::set_var("APXINF_REQUIRE_RECIPES", "1");
    }
    let options = LoadOptions {
        model_name: Some("qwen3_5_moe".into()),
        text_kv_capacity: Some(tokens.len() + 128),
        qwen35_moe_activation: Some(Qwen35MoeActivation::Nvfp4),
        qwen35_kv_cache: Some(kv),
        tuning_path: Some(PathBuf::from(env("APXINF_QWEN35_IMAGE_RECIPES"))),
        autotune: prepare,
        generation_config: GenerationConfigSource::from_cli_value("apxinf"),
        ..LoadOptions::default()
    };
    let mut model = AutoModel::load_model(
        Device::Cuda(0),
        &PathBuf::from(env("APXINF_QWEN35_CHECKPOINT")),
        &options,
    )
    .unwrap();
    let tokenizer = apxinf_tokenizer::Tokenizer::from_file(
        PathBuf::from(env("APXINF_QWEN35_CHECKPOINT")).join("tokenizer.json"),
    )
    .unwrap();
    let normalized_response = |tokens: &[u32]| {
        tokenizer
            .decode(response(tokens))
            .unwrap()
            .trim()
            .to_lowercase()
    };
    let repeats: usize = std::env::var("APXINF_QWEN35_IMAGE_REPEATS")
        .unwrap_or("3".into())
        .parse()
        .unwrap();
    let modes = std::env::var("APXINF_QWEN35_IMAGE_MODES").unwrap_or("scalar,sampling,mtp".into());
    let mut scalar_whole = None;
    let mut scalar_cached = None;
    for mode in modes.split(',') {
        let eos = std::env::var_os("APXINF_QWEN35_IMAGE_STOP_EOS").map(|_| 248046);
        let mut generation = GenerationOptions::greedy(32, eos);
        generation.speculative_decoding = Some(mode == "mtp");
        if mode == "sampling" {
            generation.sampling_mode = Some(SamplingMode::Random);
            generation.temperature = Some(1.0);
            generation.top_k = Some(20);
            generation.top_p = Some(0.95);
            generation.seed = Some(123);
        }
        model.set_context_reuse(false).unwrap();
        let (expected, _) = generate(
            &mut model,
            &tokens,
            &pixels,
            &grids,
            prefix,
            &generation,
            &format!("{mode}/oracle"),
        );
        if mode == "scalar" {
            scalar_whole = Some(expected.clone());
        }
        if mode == "mtp" {
            if let Some(scalar) = &scalar_whole {
                assert_eq!(&expected, scalar, "uncached scalar/MTP32 differs");
            }
        }
        for iteration in 0..2 + repeats {
            let (actual, reused) = generate(
                &mut model,
                &tokens,
                &pixels,
                &grids,
                prefix,
                &generation,
                &format!("{mode}/baseline/{iteration}"),
            );
            assert_eq!(actual, expected, "uncached generation is not repeatable");
            assert_eq!(reused, 0);
        }
        let mut followup = if tokens.ends_with(&[248068, 271, 248069, 271]) {
            tokens[..tokens.len() - 4].to_vec()
        } else {
            tokens.clone()
        };
        let response_len = expected
            .iter()
            .position(|&t| t == 248046)
            .map_or(expected.len(), |i| i + 1);
        followup.extend_from_slice(&expected[..response_len]);
        if expected.last() != Some(&248046) && response_len == expected.len() {
            followup.push(248046);
        }
        followup.extend([198, 248045, 846, 198]);
        // Render another user turn with the original question, followed by
        // the original assistant header. Historical image rows stay intact.
        let query = original.iter().rposition(|&t| t == 248054).unwrap() + 1;
        followup.extend_from_slice(&original[query..]);
        let mut followup_options = generation.clone();
        followup_options.eos_token_ids = Some(vec![248046]);
        let (expected_followup, _) = generate(
            &mut model,
            &followup,
            &pixels,
            &grids,
            prefix,
            &followup_options,
            &format!("{mode}/followup-oracle"),
        );
        let (expected_changed, _) = generate(
            &mut model,
            &tokens,
            &changed,
            &grids,
            prefix,
            &generation,
            &format!("{mode}/changed-oracle"),
        );
        model.set_context_reuse(true).unwrap();
        // Seed a text-only system checkpoint, then introduce an image.
        let mut text = tokens[..prefix].to_vec();
        text.extend([15, 16]);
        model
            .generate_streaming_with_options(
                LlmInput::text_with_reusable_prefix(&text, prefix),
                &generation,
                |_| {},
            )
            .unwrap();
        let (actual, reused) = generate(
            &mut model,
            &tokens,
            &pixels,
            &grids,
            prefix,
            &generation,
            &format!("{mode}/text-to-image"),
        );
        // Chunked prefill uses the existing target arithmetic with different
        // batch shapes. Compare the real response through EOS against whole
        // prefill, and require all32 tokens to match fresh chunk execution on
        // each snapshot replay. Uncached32-token controls above stay exact.
        assert_eq!(
            normalized_response(&actual),
            normalized_response(&expected),
            "image response differs beyond letter case across prefill batch shapes"
        );
        let cached_expected = actual;
        if mode == "scalar" {
            scalar_cached = Some(cached_expected.clone());
        }
        if mode == "mtp" {
            if let Some(scalar) = &scalar_cached {
                assert_eq!(&cached_expected, scalar, "cached scalar/MTP32 differs");
            }
        }
        assert_eq!(
            reused, prefix,
            "image request must reuse its text system checkpoint"
        );
        for iteration in 0..2 + repeats {
            let (actual, reused) = generate(
                &mut model,
                &tokens,
                &pixels,
                &grids,
                prefix,
                &generation,
                &format!("{mode}/repeat/{iteration}"),
            );
            assert_eq!(actual, cached_expected, "cached {mode} generation differs");
            assert_eq!(reused, tokens.len(), "full image history was not restored");
        }
        let (actual, reused) = generate(
            &mut model,
            &followup,
            &pixels,
            &grids,
            prefix,
            &followup_options,
            &format!("{mode}/followup"),
        );
        assert_eq!(
            normalized_response(&actual),
            normalized_response(&expected_followup),
            "follow-up response differs beyond letter case across prefill batch shapes"
        );
        assert!(reused >= tokens.len() - 64);
        let (actual, reused) = generate(
            &mut model,
            &tokens,
            &changed,
            &grids,
            prefix,
            &generation,
            &format!("{mode}/changed"),
        );
        assert_eq!(
            normalized_response(&actual),
            normalized_response(&expected_changed),
            "changed-image response differs beyond letter case across prefill batch shapes"
        );
        assert!(
            reused <= prefix,
            "changed pixels reused image-dependent state"
        );
        // A text-only request with the same raw token IDs must not identify
        // the multimodal snapshot. The image placeholders alone are ambiguous.
        if let LoadedModel::Text { model, .. } = &mut model {
            model.reset();
            model
                .prefill(LlmInput::text_with_reusable_prefix(&tokens, prefix))
                .unwrap();
            assert!(model.reused_input_tokens() <= prefix);
        }
    }
}
