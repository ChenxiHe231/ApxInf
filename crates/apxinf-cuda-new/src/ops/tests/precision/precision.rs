//! Torch-golden precision tests for every public L3 operator.
//!
//! Every new L3 operator must add `<op>_all_candidates_match_torch` here,
//! extend `generate_torch_l3_fixtures.py`, and regenerate
//! `torch_l3_fixtures.rs`. Tests compare final L3 outputs only; they must not
//! depend on or prescribe a candidate's internal implementation.

use super::framework::{bf16_bits_tensor, bytes_tensor, f32_tensor, scales, tensor, zeros_tensor};
use super::*;
use crate::CudaContext;
use apxinf_core::{DType, Tensor};

#[path = "torch_l3_fixtures.rs"]
mod torch_fixture;

const MAX_SCALED_ERROR: f64 = 0.08;
const MAX_RELATIVE_L2: f64 = 0.05;
const MIN_COSINE: f64 = 0.9999;

fn check_precision(reference: &[f32], actual: &[f32]) -> Result<(), String> {
    if reference.len() != actual.len() {
        return Err(format!(
            "output length mismatch: expected {}, got {}",
            reference.len(),
            actual.len()
        ));
    }
    if reference.is_empty() {
        return Err("precision comparison requires a non-empty output".into());
    }
    if reference
        .iter()
        .chain(actual)
        .any(|value| !value.is_finite())
    {
        return Err("precision comparison encountered NaN or Inf".into());
    }

    let mut max_scaled_error = 0.0f64;
    let mut squared_error = 0.0f64;
    let mut reference_norm = 0.0f64;
    let mut actual_norm = 0.0f64;
    let mut dot = 0.0f64;
    for (&expected, &observed) in reference.iter().zip(actual) {
        let expected = expected as f64;
        let observed = observed as f64;
        let error = observed - expected;
        max_scaled_error = max_scaled_error.max(error.abs() / expected.abs().max(1.0));
        squared_error += error * error;
        reference_norm += expected * expected;
        actual_norm += observed * observed;
        dot += expected * observed;
    }
    let relative_l2 = (squared_error / reference_norm.max(1e-20)).sqrt();
    let cosine = if reference_norm == 0.0 && actual_norm == 0.0 {
        1.0
    } else if reference_norm == 0.0 || actual_norm == 0.0 {
        0.0
    } else {
        dot / (reference_norm * actual_norm).sqrt()
    };

    if max_scaled_error > MAX_SCALED_ERROR || relative_l2 > MAX_RELATIVE_L2 || cosine < MIN_COSINE {
        return Err(format!(
            "precision failed: max_scaled_error={max_scaled_error}, relative_l2={relative_l2}, cosine={cosine}"
        ));
    }
    Ok(())
}

fn check_candidate_coverage(expected: &[&str], visited: &[&str]) -> Result<(), String> {
    if expected.is_empty() {
        return Err("acceptance case declared no expected candidate capability".into());
    }
    if visited.is_empty() {
        return Err("no candidate was exercised".into());
    }
    let missing: Vec<_> = expected
        .iter()
        .copied()
        .filter(|candidate| !visited.contains(candidate))
        .collect();
    if !missing.is_empty() {
        return Err(format!(
            "expected candidates were not exercised: {missing:?}"
        ));
    }
    Ok(())
}

/// P05 validates the acceptance harness itself. In particular, cosine alone
/// must not accept a uniformly scaled wrong result, and an empty/partial
/// candidate traversal must never become a vacuous pass.
#[test]
fn precision_checks_reject_bad_outputs_and_missing_candidates() {
    assert!(check_precision(&[1.0, 2.0, 3.0], &[1.0, 2.0, 3.0]).is_ok());
    assert!(check_precision(&[1.0, 2.0, 3.0], &[2.0, 4.0, 6.0]).is_err());
    assert!(check_precision(&[1.0, 2.0, 3.0], &[1.0, f32::NAN, 3.0]).is_err());
    assert!(check_precision(&[1.0, 2.0, 3.0], &[1.0, f32::INFINITY, 3.0]).is_err());
    assert!(check_precision(&[1.0, 2.0, 3.0], &[1.0, 2.0]).is_err());
    assert!(check_precision(&[0.0, 0.0], &[0.0, 1.0]).is_err());

    assert!(check_candidate_coverage(&["baseline", "specialized"], &[]).is_err());
    assert!(check_candidate_coverage(&["baseline", "specialized"], &["baseline"]).is_err());
    assert!(
        check_candidate_coverage(&["baseline", "specialized"], &["baseline", "specialized"])
            .is_ok()
    );
}

fn assert_all_applicable_candidates_checked(summary: &str, expected_backends: &[&str]) {
    assert!(summary.contains("reference=torch"), "{summary}");
    assert!(summary.contains("max_element="), "{summary}");
    assert!(summary.contains("rel_l2="), "{summary}");
    assert!(summary.contains("cosine="), "{summary}");
    assert!(
        !summary.contains("reject(numeric:"),
        "an applicable candidate failed the Torch reference: {summary}"
    );
    for backend in expected_backends {
        let attempted = format!("{backend}#");
        assert!(
            summary.contains(&attempted),
            "candidate was not visited: {backend}; {summary}"
        );
    }
}

fn prepare_with_torch_reference<'a>(
    ctx: &CudaContext,
    args: GemmArgs<'a>,
    semantic: super::contracts::Semantic,
    api: super::execution::PlanApi,
    bias: Option<&'a Tensor>,
    expected: &'a [f32],
) -> apxinf_core::Result<super::execution::PreparedExecution> {
    let normalized = super::contracts::normalize(ctx, args, semantic, api, bias)?;
    let normalized = super::contracts::with_validation_reference(
        normalized,
        super::contracts::ValidationReference::torch(expected),
    )?;
    super::execution::prepare(ctx, normalized)
}

fn configure_torch_case(args: &mut GemmArgs<'_>, alpha: f32, output_scale: f32) {
    args.alpha = alpha;
    args.output_scale = output_scale;
    args.policy.allow_fallback = false;
    args.policy.graph_safe = false;
}

#[test]
fn torch_validation_requires_the_l3_output_shape() {
    let ctx = CudaContext::new(0).unwrap();
    let a = tensor(0, vec![2, 3], &[1.0; 6]);
    let b = tensor(0, vec![3, 4], &[1.0; 12]);
    let mut out = tensor(0, vec![2, 4], &[0.0; 8]);
    let args = GemmArgs::new(&a, &b, &mut out);
    let normalized = super::contracts::normalize(
        &ctx,
        args,
        super::contracts::Semantic::Gemm,
        super::execution::PlanApi::gemm(),
        None,
    )
    .unwrap();
    let error = match super::contracts::with_validation_reference(
        normalized,
        super::contracts::ValidationReference::torch(&[1.0; 7]),
    ) {
        Ok(_) => panic!("invalid Torch output was accepted"),
        Err(error) => error,
    };
    assert!(error
        .to_string()
        .contains("Torch validation output does not match"));
}

#[test]
fn gemm_all_candidates_match_torch() {
    use torch_fixture as f;
    let vendor = ["cublas+custom-epilogue", "cublasLt+custom-epilogue"];

    let ctx = CudaContext::new(0).unwrap();
    let a = bf16_bits_tensor(0, vec![f::M, f::K], f::BF16_A);
    let b = bf16_bits_tensor(0, vec![f::K, f::N], f::BF16_B);
    let mut out = zeros_tensor(0, vec![f::M, f::N], DType::F32);
    let mut args = GemmArgs::new(&a, &b, &mut out);
    configure_torch_case(&mut args, f::BF16_ALPHA, f::BF16_OUTPUT_SCALE);
    let prepared = prepare_with_torch_reference(
        &ctx,
        args,
        super::contracts::Semantic::Gemm,
        super::execution::PlanApi::gemm(),
        None,
        f::BF16_GEMM,
    )
    .unwrap();
    assert_all_applicable_candidates_checked(prepared.summary(), &vendor);

    let ctx = CudaContext::new(0).unwrap();
    let a = bytes_tensor(0, vec![f::M, f::K], DType::F8E4M3, f::FP8_A);
    let b = bytes_tensor(0, vec![f::K, f::N], DType::F8E4M3, f::FP8_B);
    let mut out = zeros_tensor(0, vec![f::M, f::N], DType::F16);
    let mut args = GemmArgs::new(&a, &b, &mut out);
    args.quantization = GemmQuantization::Fp8UnitScale;
    configure_torch_case(&mut args, f::FP8_UNIT_ALPHA, f::FP8_UNIT_OUTPUT_SCALE);
    let prepared = prepare_with_torch_reference(
        &ctx,
        args,
        super::contracts::Semantic::Gemm,
        super::execution::PlanApi::gemm(),
        None,
        f::FP8_UNIT_GEMM,
    )
    .unwrap();
    assert_all_applicable_candidates_checked(
        prepared.summary(),
        &[
            "cublas+custom-epilogue",
            "cublasLt+custom-epilogue",
            "cublasLt-native-fp8+custom-epilogue",
            "cutlass-fp8",
        ],
    );

    let ctx = CudaContext::new(0).unwrap();
    let a = bytes_tensor(0, vec![f::M, f::K], DType::F8E4M3, f::FP8_A);
    let b = bytes_tensor(0, vec![f::K, f::N], DType::F8E4M3, f::FP8_B);
    let row_scales = scales(0, f::ROW_SCALES);
    let channel_scales = scales(0, f::CHANNEL_SCALES);
    let mut out = zeros_tensor(0, vec![f::M, f::N], DType::F32);
    let mut args = GemmArgs::fp8(&a, &row_scales, &b, &channel_scales, &mut out);
    configure_torch_case(&mut args, f::FP8_SCALED_ALPHA, f::FP8_SCALED_OUTPUT_SCALE);
    let prepared = prepare_with_torch_reference(
        &ctx,
        args,
        super::contracts::Semantic::Gemm,
        super::execution::PlanApi::gemm(),
        None,
        f::FP8_SCALED_GEMM,
    )
    .unwrap();
    assert_all_applicable_candidates_checked(
        prepared.summary(),
        &[
            "cublas+custom-epilogue",
            "cublasLt+custom-epilogue",
            "cublasLt-native-fp8+custom-epilogue",
        ],
    );
}

#[test]
fn gemm_bias_all_candidates_match_torch() {
    use torch_fixture as f;
    let vendor = ["cublas+custom-epilogue", "cublasLt+custom-epilogue"];

    let ctx = CudaContext::new(0).unwrap();
    let a = bf16_bits_tensor(0, vec![f::M, f::K], f::BF16_A);
    let b = bf16_bits_tensor(0, vec![f::K, f::N], f::BF16_B);
    let bias = bf16_bits_tensor(0, vec![f::N], f::BF16_BIAS);
    let mut out = zeros_tensor(0, vec![f::M, f::N], DType::F32);
    let mut args = GemmArgs::new(&a, &b, &mut out);
    configure_torch_case(&mut args, f::BF16_ALPHA, f::BF16_OUTPUT_SCALE);
    let prepared = prepare_with_torch_reference(
        &ctx,
        args,
        super::contracts::Semantic::GemmBias,
        super::execution::PlanApi::gemm_bias(),
        Some(&bias),
        f::BF16_GEMM_BIAS,
    )
    .unwrap();
    assert_all_applicable_candidates_checked(prepared.summary(), &vendor);

    let ctx = CudaContext::new(0).unwrap();
    let a = bytes_tensor(0, vec![f::M, f::K], DType::F8E4M3, f::FP8_A);
    let b = bytes_tensor(0, vec![f::K, f::N], DType::F8E4M3, f::FP8_B);
    let bias = f32_tensor(0, vec![f::N], f::FP8_BIAS);
    let row_scales = scales(0, f::ROW_SCALES);
    let channel_scales = scales(0, f::CHANNEL_SCALES);
    let mut out = zeros_tensor(0, vec![f::M, f::N], DType::F32);
    let mut args = GemmArgs::fp8(&a, &row_scales, &b, &channel_scales, &mut out);
    configure_torch_case(&mut args, f::FP8_SCALED_ALPHA, f::FP8_SCALED_OUTPUT_SCALE);
    let prepared = prepare_with_torch_reference(
        &ctx,
        args,
        super::contracts::Semantic::GemmBias,
        super::execution::PlanApi::gemm_bias(),
        Some(&bias),
        f::FP8_SCALED_GEMM_BIAS,
    )
    .unwrap();
    assert_all_applicable_candidates_checked(prepared.summary(), &vendor);
}

#[test]
fn gemm_bias_gelu_all_candidates_match_torch() {
    use torch_fixture as f;
    let vendor = ["cublas+custom-epilogue", "cublasLt+custom-epilogue"];

    let ctx = CudaContext::new(0).unwrap();
    let a = bf16_bits_tensor(0, vec![f::M, f::K], f::BF16_A);
    let b = bf16_bits_tensor(0, vec![f::K, f::N], f::BF16_B);
    let bias = bf16_bits_tensor(0, vec![f::N], f::BF16_BIAS);
    let mut out = zeros_tensor(0, vec![f::M, f::N], DType::F32);
    let mut args = GemmArgs::new(&a, &b, &mut out);
    configure_torch_case(&mut args, f::BF16_ALPHA, f::BF16_OUTPUT_SCALE);
    let prepared = prepare_with_torch_reference(
        &ctx,
        args,
        super::contracts::Semantic::GemmBiasGelu,
        super::execution::PlanApi::gemm_bias_gelu(),
        Some(&bias),
        f::BF16_GEMM_BIAS_GELU,
    )
    .unwrap();
    assert_all_applicable_candidates_checked(prepared.summary(), &vendor);

    let ctx = CudaContext::new(0).unwrap();
    let a = bytes_tensor(0, vec![f::M, f::K], DType::F8E4M3, f::FP8_A);
    let b = bytes_tensor(0, vec![f::K, f::N], DType::F8E4M3, f::FP8_B);
    let bias = f32_tensor(0, vec![f::N], f::FP8_BIAS);
    let row_scales = scales(0, f::ROW_SCALES);
    let channel_scales = scales(0, f::CHANNEL_SCALES);
    let mut out = zeros_tensor(0, vec![f::M, f::N], DType::F32);
    let mut args = GemmArgs::fp8(&a, &row_scales, &b, &channel_scales, &mut out);
    configure_torch_case(&mut args, f::FP8_SCALED_ALPHA, f::FP8_SCALED_OUTPUT_SCALE);
    let prepared = prepare_with_torch_reference(
        &ctx,
        args,
        super::contracts::Semantic::GemmBiasGelu,
        super::execution::PlanApi::gemm_bias_gelu(),
        Some(&bias),
        f::FP8_SCALED_GEMM_BIAS_GELU,
    )
    .unwrap();
    assert_all_applicable_candidates_checked(prepared.summary(), &vendor);
}

#[test]
fn gemm_geglu_all_candidates_match_torch() {
    use torch_fixture as f;
    let vendor = ["cublas+custom-epilogue", "cublasLt+custom-epilogue"];

    let ctx = CudaContext::new(0).unwrap();
    let a = bf16_bits_tensor(0, vec![f::M, f::K], f::BF16_A);
    let b = bf16_bits_tensor(0, vec![f::K, 2 * f::N], f::BF16_GEGLU_B);
    let mut out = zeros_tensor(0, vec![f::M, f::N], DType::F32);
    let mut args = GemmArgs::new(&a, &b, &mut out);
    configure_torch_case(&mut args, f::BF16_ALPHA, f::BF16_OUTPUT_SCALE);
    let prepared = prepare_with_torch_reference(
        &ctx,
        args,
        super::contracts::Semantic::GemmGeglu,
        super::execution::PlanApi::gemm_geglu(),
        None,
        f::BF16_GEMM_GEGLU,
    )
    .unwrap();
    assert_all_applicable_candidates_checked(prepared.summary(), &vendor);

    let ctx = CudaContext::new(0).unwrap();
    let a = bytes_tensor(0, vec![f::M, f::K], DType::F8E4M3, f::FP8_A);
    let b = bytes_tensor(0, vec![f::K, 2 * f::N], DType::F8E4M3, f::FP8_GEGLU_B);
    let mut out = zeros_tensor(0, vec![f::M, f::N], DType::F32);
    let mut args = GemmArgs::new(&a, &b, &mut out);
    args.quantization = GemmQuantization::Fp8UnitScale;
    configure_torch_case(&mut args, f::FP8_UNIT_ALPHA, f::FP8_UNIT_OUTPUT_SCALE);
    let prepared = prepare_with_torch_reference(
        &ctx,
        args,
        super::contracts::Semantic::GemmGeglu,
        super::execution::PlanApi::gemm_geglu(),
        None,
        f::FP8_UNIT_GEMM_GEGLU,
    )
    .unwrap();
    assert_all_applicable_candidates_checked(prepared.summary(), &vendor);
}
