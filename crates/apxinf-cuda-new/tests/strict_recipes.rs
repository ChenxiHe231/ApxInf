//! Subprocess contracts for recipe-only benchmark execution. No model weights.
//! A fresh process is required because the native option is immutable there.
use std::path::{Path, PathBuf};
use std::process::Command;

use apxinf_core::{DType, Shape, Tensor};
use apxinf_cuda_new::{ops, CudaBuffer, CudaContext};

fn tensor(ctx: &CudaContext, shape: Vec<usize>, one: bool) -> Tensor {
    let n = shape.iter().product::<usize>();
    let values = if one {
        vec![0x80, 0x3f].repeat(n)
    } else {
        vec![0; n * 2]
    };
    let buffer = CudaBuffer::alloc(n * 2, ctx.device_id()).unwrap();
    buffer.copy_from_host(&values).unwrap();
    buffer.as_tensor(Shape::new(shape), DType::BF16).unwrap()
}

fn bytes(value: &Tensor) -> Vec<u8> {
    let buffer = CudaBuffer::from_tensor(value).unwrap();
    let mut result = vec![0; buffer.len()];
    buffer.copy_to_host(&mut result).unwrap();
    result
}

fn run_child(stage: &str, operator: &str, directory: &Path) {
    let status = Command::new(std::env::current_exe().unwrap())
        .args([
            "--exact",
            "strict_recipe_child",
            "--nocapture",
            "--test-threads=1",
        ])
        .env("APXINF_STRICT_RECIPE_TEST_STAGE", stage)
        .env("APXINF_STRICT_RECIPE_TEST_OPERATOR", operator)
        .env("APXINF_STRICT_RECIPE_TEST_DIRECTORY", directory)
        .env(
            "APXINF_REQUIRE_RECIPES",
            if stage == "seed" { "0" } else { "1" },
        )
        .status()
        .unwrap();
    assert!(status.success(), "{stage}/{operator} failed");
}

#[test]
fn strict_recipes_preserve_outputs_and_require_restorable_files() {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../devlocal/model-microbench/strict-recipes")
        .join(std::process::id().to_string());
    for operator in ["gemm", "attention"] {
        let directory = root.join(operator);
        std::fs::create_dir_all(&directory).unwrap();
        run_child("missing", operator, &directory);
        assert_eq!(std::fs::read_dir(&directory).unwrap().count(), 0);
        run_child("seed", operator, &directory);
        let recipes: Vec<_> = std::fs::read_dir(&directory)
            .unwrap()
            .map(|entry| entry.unwrap().path())
            .collect();
        assert!(!recipes.is_empty(), "tuning did not persist a recipe");
        let original: Vec<_> = recipes.iter().map(|p| std::fs::read(p).unwrap()).collect();
        run_child("restore", operator, &directory);
        for (path, expected) in recipes.iter().zip(original) {
            assert_eq!(
                std::fs::read(path).unwrap(),
                expected,
                "formal run changed recipe"
            );
            std::fs::write(path, b"invalid recipe\n").unwrap();
        }
        run_child("corrupt", operator, &directory);
        for path in recipes {
            assert_eq!(std::fs::read(path).unwrap(), b"invalid recipe\n");
        }
    }
}

#[test]
fn strict_recipe_child() {
    let Ok(stage) = std::env::var("APXINF_STRICT_RECIPE_TEST_STAGE") else {
        return;
    };
    let operator = std::env::var("APXINF_STRICT_RECIPE_TEST_OPERATOR").unwrap();
    let directory = PathBuf::from(std::env::var_os("APXINF_STRICT_RECIPE_TEST_DIRECTORY").unwrap());
    // Missing/corrupt cases deliberately permit online tuning and fallback in
    // the normal policy: the process-wide guard must override both choices.
    let ctx = CudaContext::new_with_options(0, stage != "restore", Some(&directory)).unwrap();
    let (output, before, outcome, expected) = if operator == "gemm" {
        let a = tensor(&ctx, vec![64, 64], true);
        let b = tensor(&ctx, vec![64, 64], true);
        let mut output = tensor(&ctx, vec![64, 64], false);
        let before = bytes(&output);
        let outcome = ops::gemm(&ctx, ops::GemmArgs::new(&a, &b, &mut output));
        (output, before, outcome, 64.0f32)
    } else {
        let q = tensor(&ctx, vec![1, 1, 16, 256], false);
        let k = tensor(&ctx, vec![1, 129, 2, 256], false);
        let v = tensor(&ctx, vec![1, 129, 2, 256], true);
        let mut output = tensor(&ctx, vec![1, 1, 16, 256], false);
        let before = bytes(&output);
        let outcome = ops::kv_cache_attention(
            &ctx,
            ops::KvCacheAttentionArgs::new(&q, &k, &v, &mut output),
        );
        (output, before, outcome, 1.0f32)
    };
    if stage == "missing" || stage == "corrupt" {
        let error = outcome.unwrap_err().to_string();
        assert!(
            error.contains("strict recipe mode"),
            "unexpected error: {error}"
        );
        ctx.synchronize().unwrap();
        assert_eq!(
            bytes(&output),
            before,
            "failed recipe prepare changed output"
        );
    } else {
        outcome.unwrap();
        ctx.synchronize().unwrap();
        for raw in bytes(&output).chunks_exact(2) {
            let actual = half::bf16::from_bits(u16::from_le_bytes([raw[0], raw[1]])).to_f32();
            assert!((actual - expected).abs() <= 0.01, "{actual} != {expected}");
        }
    }
}
