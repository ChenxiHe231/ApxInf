//! Qwen3.8-27B-NVFP4 end-to-end latency benchmark through the public entry.
//!
//! Loads through `AutoModel` (registry detection from the checkpoint's
//! config.json), generates greedily through `LoadedModel::generate_streaming`,
//! and reports JSON per repeat plus a summary. The deterministic LCG prompt
//! matches the kernel harness and the vLLM comparison scripts, so `token_md5`
//! is directly comparable across all three.
//!
//! ```text
//! cargo run -p apxinf-model --features cuda --release --example qwen38_bench -- \
//!     <checkpoint-dir> [--prompt-len N] [--max-new N] [--repeats N]
//! ```
//!
//! The first generation warms autotuners and captures the decode graphs; it is
//! reported with `"warmup": true` and excluded from the summary statistics.

use std::time::Instant;

use apxinf_core::Device;
use apxinf_model::llm_trait::LlmInput;
use apxinf_model::{AutoModel, LoadOptions};

const VOCAB: usize = 248_320;

/// The fixed LCG prompt of the kernel harness (`APXINF_QWEN38_PROMPT_LEN`
/// path): reproducible, spread across the vocabulary, clear of the special
/// tokens near the top of the id range.
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

fn arg_value(args: &[String], name: &str, default: usize) -> usize {
    args.iter()
        .position(|arg| arg == name)
        .and_then(|index| args.get(index + 1))
        .map(|value| value.parse().expect(name))
        .unwrap_or(default)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let checkpoint = args
        .get(1)
        .filter(|arg| !arg.starts_with("--"))
        .expect("usage: qwen38_bench <checkpoint-dir> [--prompt-len N] [--max-new N] [--repeats N]");
    let prompt_len = arg_value(&args, "--prompt-len", 2048);
    let max_new = arg_value(&args, "--max-new", 128);
    let repeats = arg_value(&args, "--repeats", 3);

    let load_start = Instant::now();
    let mut model = AutoModel::load_model(Device::Cuda(0), checkpoint, &LoadOptions::default())?;
    let load_secs = load_start.elapsed().as_secs_f64();

    let prompt = deterministic_prompt(prompt_len);
    println!(
        "{{\"schema\": \"apxinf.qwen38.benchmark.v1\", \"checkpoint\": {checkpoint:?}, \
         \"prompt_len\": {prompt_len}, \"max_new\": {max_new}, \"repeats\": {repeats}, \
         \"load_s\": {load_secs:.1}}}"
    );

    let mut ttfts = Vec::new();
    let mut tpots = Vec::new();
    let mut digests = Vec::new();
    // repeat 0 is the warm-up: autotune, graph capture, prefill session prepare.
    for repeat in 0..=repeats {
        let (tokens, profile) = model.generate_streaming(
            LlmInput::text(&prompt),
            max_new,
            |_token| {},
            None, // fixed-length benchmark: EOS must not stop decode early
        )?;
        assert_eq!(tokens.len(), max_new, "fixed decode budget");

        // The exact byte stream the harness evidence hashes:
        //   grep -oE "generated token ids.*" run.log | md5sum
        let line = format!("generated token ids ({}): {tokens:?}\n", tokens.len());
        let digest = md5_hex(line.as_bytes());
        let ttft = profile.ttft_ms().expect("profile records first token");
        let tpot = profile.tpot_ms().expect("profile records decode");
        let warmup = repeat == 0;
        println!(
            "{{\"repeat\": {repeat}, \"warmup\": {warmup}, \"ttft_ms\": {ttft:.2}, \
             \"decode_ms_per_token\": {tpot:.3}, \"decode_tokens_per_second\": {:.2}, \
             \"token_md5\": \"{digest}\"}}",
            1e3 / tpot
        );
        if !warmup {
            ttfts.push(ttft);
            tpots.push(tpot);
        }
        digests.push(digest);
    }

    assert!(
        digests.windows(2).all(|pair| pair[0] == pair[1]),
        "generation is not deterministic across repeats: {digests:?}"
    );
    let mean = |values: &[f64]| values.iter().sum::<f64>() / values.len() as f64;
    println!(
        "{{\"summary\": {{\"ttft_ms_mean\": {:.2}, \"decode_ms_per_token_mean\": {:.3}, \
         \"decode_tokens_per_second_mean\": {:.2}, \"token_md5\": \"{}\"}}}}",
        mean(&ttfts),
        mean(&tpots),
        1e3 / mean(&tpots),
        digests[0]
    );
    Ok(())
}

// Dependency-free md5 so the digest matches `md5sum` over the kernel-harness
// log line.
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
