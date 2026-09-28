//! Interactive text generation for Qwen3.8-27B-NVFP4 through the public entry.
//!
//! Loads with `AutoModel` (registry detection from config.json), encodes the
//! prompt with the checkpoint's tokenizer (chat template when available),
//! streams tokens as they decode, and prints latency at the end.
//!
//! ```text
//! # one-shot
//! cargo run -p apxinf-model --features cuda --release --example qwen38_chat -- \
//!     <checkpoint-dir> --prompt "What is the capital of France?" [--max-new 256] [--raw]
//!
//! # interactive REPL (reads one prompt per line from stdin)
//! cargo run -p apxinf-model --features cuda --release --example qwen38_chat -- <checkpoint-dir>
//! ```
//!
//! `--raw` skips the chat template and completes the literal prompt text.

use std::io::{BufRead, Write};
use std::time::Instant;

use apxinf_core::Device;
use apxinf_model::llm_trait::LlmInput;
use apxinf_model::{AutoModel, LoadOptions};
use apxinf_tokenizer::Tokenizer;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let checkpoint = args
        .get(1)
        .filter(|arg| !arg.starts_with("--"))
        .expect("usage: qwen38_chat <checkpoint-dir> [--prompt TEXT] [--max-new N] [--raw]")
        .clone();
    let max_new: usize = args
        .iter()
        .position(|arg| arg == "--max-new")
        .and_then(|index| args.get(index + 1))
        .map(|value| value.parse().expect("--max-new"))
        .unwrap_or(256);
    let raw = args.iter().any(|arg| arg == "--raw");
    let one_shot = args
        .iter()
        .position(|arg| arg == "--prompt")
        .and_then(|index| args.get(index + 1))
        .cloned();

    let tokenizer = Tokenizer::from_file(format!("{checkpoint}/tokenizer.json"))?;
    let eos = tokenizer.eos_token_id();

    eprint!("loading {checkpoint} ... ");
    let load_start = Instant::now();
    let mut model = AutoModel::load_model(Device::Cuda(0), &checkpoint, &LoadOptions::default())?;
    eprintln!("{:.1}s", load_start.elapsed().as_secs_f64());

    let mut generate = |prompt_text: &str| -> Result<(), Box<dyn std::error::Error>> {
        // The checkpoint is instruction-tuned in the Qwen conversation format.
        // Its template ships as a standalone chat_template.jinja (with vision
        // branches), which the tokenizer does not load; text-only chat needs
        // exactly the im_start/im_end framing, so build it directly.
        // --raw bypasses it for plain text completion.
        let encoded = if raw {
            tokenizer.encode(prompt_text)?
        } else {
            tokenizer.encode(&format!(
                "<|im_start|>user\n{prompt_text}<|im_end|>\n<|im_start|>assistant\n"
            ))?
        };

        // Stream: decode incrementally, printing only the newly stable text so
        // multi-token characters (CJK, emoji) come out intact.
        let mut generated: Vec<u32> = Vec::new();
        let mut printed = 0usize;
        let (tokens, profile) = model.generate_streaming(
            LlmInput::text(&encoded),
            max_new,
            |token| {
                generated.push(token);
                if let Ok(text) = tokenizer.decode(&generated) {
                    // Hold back a partially decoded character (U+FFFD tail).
                    let stable = text.strip_suffix('\u{FFFD}').unwrap_or(&text);
                    if stable.len() > printed {
                        print!("{}", &stable[printed..]);
                        std::io::stdout().flush().ok();
                        printed = stable.len();
                    }
                }
            },
            eos,
        )?;
        // Flush any held-back tail.
        let text = tokenizer.decode(&tokens)?;
        if text.len() > printed {
            print!("{}", &text[printed..]);
        }
        println!();
        eprintln!(
            "[{} prompt + {} generated | ttft {:.0} ms | {:.1} tok/s]",
            encoded.len(),
            tokens.len(),
            profile.ttft_ms().unwrap_or(0.0),
            profile.generation_tps().unwrap_or(0.0),
        );
        Ok(())
    };

    match one_shot {
        Some(prompt_text) => generate(&prompt_text)?,
        None => {
            eprintln!("interactive mode; one prompt per line, Ctrl-D to exit");
            let stdin = std::io::stdin();
            for line in stdin.lock().lines() {
                let line = line?;
                if line.trim().is_empty() {
                    continue;
                }
                generate(&line)?;
            }
        }
    }
    Ok(())
}
