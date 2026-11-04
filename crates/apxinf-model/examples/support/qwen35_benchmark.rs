//! Pure measurement helpers shared by the model-only Qwen benchmark examples.
//! Can be checked with `rustc --test` without linking CUDA or loading weights.
use std::cmp::Ordering;
use std::path::PathBuf;

pub const INPUT_SEED: u64 = 0x35A3_BF04_4096_0001;
pub const VOCAB_SIZE: usize = 248_320;

pub fn exact_tokens(count: usize) -> Vec<u32> {
    let mut state = INPUT_SEED;
    (0..count)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((state >> 17) as usize % (VOCAB_SIZE - 1_000)) as u32
        })
        .collect()
}

pub fn token_digest(tokens: &[u32]) -> u64 {
    tokens.iter().fold(0xcbf29ce484222325u64, |hash, token| {
        token.to_le_bytes().into_iter().fold(hash, |hash, byte| {
            (hash ^ u64::from(byte)).wrapping_mul(0x100000001b3)
        })
    })
}

pub fn quantile(values: &[f64], percentile: f64) -> f64 {
    assert!(!values.is_empty());
    assert!(values.iter().all(|value| value.is_finite()));
    assert!((0.0..=1.0).contains(&percentile));
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    // Statistical median for an even-sized final validation batch. Other
    // percentiles use nearest-rank and keep the observed extrema.
    if percentile == 0.5 && sorted.len() % 2 == 0 {
        let middle = sorted.len() / 2;
        return sorted[middle - 1] / 2.0 + sorted[middle] / 2.0;
    }
    let index = ((percentile * sorted.len() as f64).ceil() as usize)
        .saturating_sub(1)
        .min(sorted.len() - 1);
    sorted[index]
}

pub fn optional_path(args: &[String], name: &str) -> Result<Option<PathBuf>, String> {
    args.iter()
        .position(|arg| arg == name)
        .map(|index| {
            args.get(index + 1)
                .filter(|value| !value.starts_with("--"))
                .map(PathBuf::from)
                .ok_or_else(|| format!("{name} requires a path"))
        })
        .transpose()
}

pub fn usize_arg(args: &[String], name: &str, default: usize) -> Result<usize, String> {
    args.iter()
        .position(|arg| arg == name)
        .map(|index| {
            args.get(index + 1)
                .ok_or_else(|| format!("{name} requires a number"))?
                .parse::<usize>()
                .map_err(|_| format!("{name} requires a nonnegative integer"))
        })
        .transpose()
        .map(|value| value.unwrap_or(default))
}

pub fn validate_tuning_path(path: &Option<PathBuf>, autotune: bool) -> Result<(), String> {
    let path = path
        .as_ref()
        .ok_or("--tuning-dir is required for both tuning and formal runs")?;
    if !path.is_absolute() {
        return Err("--tuning-dir must be an absolute path".into());
    }
    if !autotune {
        let existing = std::fs::read_dir(path)
            .map_err(|error| format!("formal recipe directory {}: {error}", path.display()))?
            .filter_map(Result::ok)
            .any(|entry| {
                entry
                    .path()
                    .extension()
                    .is_some_and(|extension| extension == "recipe")
            });
        if !existing {
            return Err("formal run requires existing .recipe files; run the separate --autotune pass first".into());
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn median_and_percentiles() {
        assert_eq!(quantile(&[4., 1., 3., 2.], 0.5), 2.5);
        assert_eq!(quantile(&[3., 1., 2.], 0.5), 2.);
        assert_eq!(quantile(&[4., 1., 3., 2.], 0.95), 4.);
    }
    #[test]
    fn token_corpus_is_prefix_stable_and_avoids_special_tokens() {
        let a = exact_tokens(2048);
        let b = exact_tokens(4096);
        assert_eq!(a, b[..2048]);
        assert!(b.iter().all(|id| (*id as usize) < VOCAB_SIZE - 1000));
        assert_ne!(token_digest(&a), token_digest(&b));
    }
    #[test]
    fn digest_is_little_endian_and_order_sensitive() {
        assert_eq!(token_digest(&[]), 0xcbf29ce484222325);
        assert_ne!(token_digest(&[1, 2]), token_digest(&[2, 1]));
    }
    #[test]
    fn missing_or_bad_numeric_arguments_are_rejected() {
        assert!(usize_arg(&["--x".into()], "--x", 5).is_err());
        assert!(usize_arg(&["--x".into(), "bad".into()], "--x", 5).is_err());
        assert_eq!(usize_arg(&[], "--x", 5).unwrap(), 5);
    }
    #[test]
    fn formal_requires_absolute_recipe_path() {
        assert!(validate_tuning_path(&None, false).is_err());
        assert!(validate_tuning_path(&Some("relative".into()), false).is_err());
        assert!(validate_tuning_path(
            &Some(PathBuf::from("/nonexistent-qwen-benchmark-recipes")),
            false
        )
        .is_err());
        assert!(validate_tuning_path(
            &Some(PathBuf::from("/nonexistent-qwen-benchmark-recipes")),
            true
        )
        .is_ok());
    }
}
