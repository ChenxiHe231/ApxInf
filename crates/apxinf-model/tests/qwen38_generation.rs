//! Qwen3.8-27B-NVFP4 module acceptance: drive the `apxinf_model::qwen38`
//! module through the shared generation loop against the real checkpoint.
//!
//!   APXINF_QWEN38_CHECKPOINT=<dir> \
//!   cargo test -p apxinf-model --features cuda --release \
//!     --test qwen38_generation -- --ignored --nocapture --test-threads=1
//!
//! The 2048-token deterministic prompt and 128-token greedy budget match the
//! kernel harness in `apxinf-cuda-new/tests/qwen38_end_to_end.rs`, so the
//! generated-token md5 printed here must equal the PR-A baseline for the same
//! commit of the kernel crate.

#![cfg(feature = "cuda")]

use std::path::PathBuf;
use std::time::Instant;

use apxinf_core::Device;
use apxinf_model::llm_trait::{LlmInput, LlmTrait};
use apxinf_model::qwen38::{Qwen38, Qwen38Config, VOCAB};

const PROMPT_LEN: usize = 2048;
const MAX_NEW: usize = 128;

/// The same fixed LCG prompt as the kernel harness's
/// APXINF_QWEN38_PROMPT_LEN path: reproducible, spread across the vocab,
/// clear of the special tokens near the top of the range.
fn deterministic_prompt(n: usize) -> Vec<u32> {
    let mut state = 0x2545_F491_4F6C_DD1Du64;
    (0..n)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 16) as usize % (VOCAB - 1000)) as u32
        })
        .collect()
}

#[test]
#[ignore = "requires the 20 GiB Qwen3.8-27B-NVFP4 checkpoint and a GPU"]
fn generates_the_baseline_tokens_through_the_module() {
    let ckpt: PathBuf = std::env::var_os("APXINF_QWEN38_CHECKPOINT")
        .expect("set APXINF_QWEN38_CHECKPOINT")
        .into();

    let (weights, _metadata) =
        apxinf_loader::safetensors::load_native_path(&ckpt).expect("load checkpoint");

    let load_start = Instant::now();
    let mut model = Qwen38::with_config(
        weights,
        Device::Cuda(0),
        PROMPT_LEN + MAX_NEW + 8,
        Qwen38Config::default(),
    )
    .expect("construct Qwen38");
    println!("model load: {:.1} s", load_start.elapsed().as_secs_f64());

    let prompt = deterministic_prompt(PROMPT_LEN);

    // Warm up autotuners and graph capture outside the timed window, then
    // reset so the measured generation starts from clean state.
    model.reset();
    let _ = model
        .forward(&prompt, 0)
        .expect("warm-up prefill");

    let mut tokens = Vec::with_capacity(MAX_NEW);
    let prefill_start = Instant::now();
    let mut prefill_ms = 0.0f64;
    let (generated, _profile) = apxinf_model::llm_trait::generate_streaming(
        &mut model,
        LlmInput::text(&prompt),
        MAX_NEW,
        |token| {
            if tokens.is_empty() {
                prefill_ms = prefill_start.elapsed().as_secs_f64() * 1e3;
            }
            tokens.push(token);
        },
        None, // fixed-length run: EOS must not stop a benchmark decode
    )
    .expect("generate");
    let total_secs = prefill_start.elapsed().as_secs_f64();

    assert_eq!(generated.len(), MAX_NEW, "fixed decode budget");
    assert_eq!(tokens, generated);
    let decode_ms = (total_secs * 1e3 - prefill_ms) / (MAX_NEW - 1) as f64;
    println!("prefill[module]: {prefill_ms:7.2} ms  ({PROMPT_LEN} tokens)  TTFT {prefill_ms:7.2} ms");
    println!("decode: {decode_ms:7.3} ms/token   {:6.2} tok/s   ({} steps)", 1e3 / decode_ms, MAX_NEW - 1);

    // The PR-A evidence logs hash the printed line itself:
    //   grep -oE "generated token ids.*" run.log | md5sum
    // Reproduce that exact byte stream (line + trailing newline) so the digest
    // is directly comparable.
    let line = format!("generated token ids ({}): {generated:?}", generated.len());
    println!("{line}");
    let digest = md5_hex(format!("{line}\n").as_bytes());
    println!("token md5: {digest}");

    if let Ok(expected) = std::env::var("APXINF_QWEN38_EXPECTED_MD5") {
        assert_eq!(digest, expected, "generated tokens diverge from the PR-A baseline");
    }
}

// A dependency-free md5 so the acceptance number is comparable with the
// `md5sum` invocations in the kernel-harness evidence logs.
fn md5_hex(data: &[u8]) -> String {
    const S: [u32; 64] = [
        7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 7, 12, 17, 22, 5, 9, 14, 20, 5, 9, 14, 20, 5,
        9, 14, 20, 5, 9, 14, 20, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 4, 11, 16, 23, 6,
        10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21, 6, 10, 15, 21,
    ];
    const K: [u32; 64] = [
        0xd76aa478, 0xe8c7b756, 0x242070db, 0xc1bdceee, 0xf57c0faf, 0x4787c62a, 0xa8304613,
        0xfd469501, 0x698098d8, 0x8b44f7af, 0xffff5bb1, 0x895cd7be, 0x6b901122, 0xfd987193,
        0xa679438e, 0x49b40821, 0xf61e2562, 0xc040b340, 0x265e5a51, 0xe9b6c7aa, 0xd62f105d,
        0x02441453, 0xd8a1e681, 0xe7d3fbc8, 0x21e1cde6, 0xc33707d6, 0xf4d50d87, 0x455a14ed,
        0xa9e3e905, 0xfcefa3f8, 0x676f02d9, 0x8d2a4c8a, 0xfffa3942, 0x8771f681, 0x6d9d6122,
        0xfde5380c, 0xa4beea44, 0x4bdecfa9, 0xf6bb4b60, 0xbebfbc70, 0x289b7ec6, 0xeaa127fa,
        0xd4ef3085, 0x04881d05, 0xd9d4d039, 0xe6db99e5, 0x1fa27cf8, 0xc4ac5665, 0xf4292244,
        0x432aff97, 0xab9423a7, 0xfc93a039, 0x655b59c3, 0x8f0ccc92, 0xffeff47d, 0x85845dd1,
        0x6fa87e4f, 0xfe2ce6e0, 0xa3014314, 0x4e0811a1, 0xf7537e82, 0xbd3af235, 0x2ad7d2bb,
        0xeb86d391,
    ];
    let mut message = data.to_vec();
    let bit_len = (data.len() as u64).wrapping_mul(8);
    message.push(0x80);
    while message.len() % 64 != 56 {
        message.push(0);
    }
    message.extend_from_slice(&bit_len.to_le_bytes());

    let (mut a0, mut b0, mut c0, mut d0) =
        (0x67452301u32, 0xefcdab89u32, 0x98badcfeu32, 0x10325476u32);
    for chunk in message.chunks_exact(64) {
        let mut m = [0u32; 16];
        for (i, word) in m.iter_mut().enumerate() {
            *word = u32::from_le_bytes(chunk[i * 4..i * 4 + 4].try_into().unwrap());
        }
        let (mut a, mut b, mut c, mut d) = (a0, b0, c0, d0);
        for i in 0..64 {
            let (f, g) = match i / 16 {
                0 => ((b & c) | (!b & d), i),
                1 => ((d & b) | (!d & c), (5 * i + 1) % 16),
                2 => (b ^ c ^ d, (3 * i + 5) % 16),
                _ => (c ^ (b | !d), (7 * i) % 16),
            };
            let temp = d;
            d = c;
            c = b;
            b = b.wrapping_add(
                a.wrapping_add(f)
                    .wrapping_add(K[i])
                    .wrapping_add(m[g])
                    .rotate_left(S[i]),
            );
            a = temp;
        }
        a0 = a0.wrapping_add(a);
        b0 = b0.wrapping_add(b);
        c0 = c0.wrapping_add(c);
        d0 = d0.wrapping_add(d);
    }
    let mut out = String::with_capacity(32);
    for word in [a0, b0, c0, d0] {
        for byte in word.to_le_bytes() {
            out.push_str(&format!("{byte:02x}"));
        }
    }
    out
}
