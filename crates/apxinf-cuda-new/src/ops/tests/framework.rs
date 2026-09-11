//! Shared operator-framework regression tests.
//!
//! Adding an L3 operator normally does not require editing this file. Extend it
//! only when the operator introduces or changes shared behavior such as keys,
//! candidate selection, alignment, caching, resources, or lifetimes.

use super::*;
use crate::{CudaBuffer, CudaContext};
use apxinf_core::{DType, Shape, Tensor};
use half::bf16;
use std::ffi::CStr;
use std::os::raw::{c_char, c_int};

unsafe extern "C" {
    fn apxinf_gemm_test_hardware_fingerprint(
        uuid: *const u8,
        uuid_size: usize,
        multiprocessor_count: i32,
        total_global_memory: u64,
        performance: c_int,
        output: *mut c_char,
        capacity: usize,
    ) -> usize;
    fn apxinf_gemm_test_resource_prefilter(device: c_int) -> c_int;
    fn apxinf_gemm_test_reset_tuning_observers();
    fn apxinf_gemm_test_tuning_count() -> u64;
    fn apxinf_gemm_test_configuration_evaluation_count() -> u64;
}

fn reset_tuning_observers() {
    unsafe { apxinf_gemm_test_reset_tuning_observers() }
}

fn tuning_observers() -> (u64, u64) {
    unsafe {
        (
            apxinf_gemm_test_tuning_count(),
            apxinf_gemm_test_configuration_evaluation_count(),
        )
    }
}

fn hardware_fingerprint(uuid: [u8; 16], sms: i32, memory: u64, performance: bool) -> String {
    let mut output = vec![0 as c_char; 512];
    let length = unsafe {
        apxinf_gemm_test_hardware_fingerprint(
            uuid.as_ptr(),
            uuid.len(),
            sms,
            memory,
            if performance { 1 } else { 0 },
            output.as_mut_ptr(),
            output.len(),
        )
    };
    assert!(length < output.len());
    unsafe { CStr::from_ptr(output.as_ptr()) }
        .to_str()
        .unwrap()
        .to_owned()
}

#[test]
fn uuid_does_not_partition_persistent_tuning_cache() {
    let first = [0x11; 16];
    let second = [0xee; 16];
    assert_eq!(
        hardware_fingerprint(first, 14, 32 << 30, true),
        hardware_fingerprint(second, 14, 32 << 30, true),
        "equivalent GPUs with different UUIDs must share tuned recipes"
    );
    assert_eq!(
        hardware_fingerprint(first, 14, 32 << 30, false),
        hardware_fingerprint(second, 14, 32 << 30, false)
    );
}

#[test]
fn performance_mismatch_is_compatible_but_not_fully_tuned() {
    let uuid = [0x42; 16];
    let full = hardware_fingerprint(uuid, 14, 32 << 30, true);
    let reduced = hardware_fingerprint(uuid, 7, 16 << 30, true);
    assert_ne!(
        full, reduced,
        "different performance profiles need retuning"
    );
    assert_eq!(
        hardware_fingerprint(uuid, 14, 32 << 30, false),
        hardware_fingerprint(uuid, 7, 16 << 30, false),
        "the previous winner remains an execution-compatible tuning hint"
    );
}

#[test]
fn workspace_budget_rejects_before_provider_create() {
    assert_eq!(unsafe { apxinf_gemm_test_resource_prefilter(0) }, 1);
}

pub(super) fn tensor(device: usize, shape: Vec<usize>, values: &[f32]) -> Tensor {
    let host: Vec<_> = values.iter().map(|value| bf16::from_f32(*value)).collect();
    let bytes = unsafe {
        std::slice::from_raw_parts(
            host.as_ptr().cast::<u8>(),
            host.len() * std::mem::size_of::<bf16>(),
        )
    };
    let buffer = CudaBuffer::alloc(bytes.len(), device).unwrap();
    buffer.copy_from_host(bytes).unwrap();
    buffer.as_tensor(Shape::new(shape), DType::BF16).unwrap()
}

pub(super) fn bf16_bits_tensor(device: usize, shape: Vec<usize>, values: &[u16]) -> Tensor {
    let bytes: Vec<_> = values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect();
    bytes_tensor(device, shape, DType::BF16, &bytes)
}

pub(super) fn f32_tensor(device: usize, shape: Vec<usize>, values: &[f32]) -> Tensor {
    let bytes: Vec<_> = values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect();
    bytes_tensor(device, shape, DType::F32, &bytes)
}

pub(super) fn zeros_tensor(device: usize, shape: Vec<usize>, dtype: DType) -> Tensor {
    let bytes = vec![0; shape.iter().product::<usize>() * dtype.size_in_bytes()];
    let buffer = CudaBuffer::alloc(bytes.len(), device).unwrap();
    buffer.copy_from_host(&bytes).unwrap();
    buffer.as_tensor(Shape::new(shape), dtype).unwrap()
}

pub(super) fn bytes_tensor(device: usize, shape: Vec<usize>, dtype: DType, bytes: &[u8]) -> Tensor {
    assert_eq!(
        bytes.len(),
        shape.iter().product::<usize>() * dtype.size_in_bytes()
    );
    let buffer = CudaBuffer::alloc(bytes.len(), device).unwrap();
    buffer.copy_from_host(bytes).unwrap();
    buffer.as_tensor(Shape::new(shape), dtype).unwrap()
}

pub(super) fn scales(device: usize, values: &[f32]) -> Tensor {
    let bytes: Vec<_> = values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect();
    let buffer = CudaBuffer::alloc(bytes.len(), device).unwrap();
    buffer.copy_from_host(&bytes).unwrap();
    buffer
        .as_tensor(Shape::new(vec![values.len()]), DType::F32)
        .unwrap()
}

fn prepare_test_gemm(
    ctx: &CudaContext,
    args: GemmArgs<'_>,
) -> apxinf_core::Result<super::execution::PreparedExecution> {
    super::execution::prepare(
        ctx,
        super::contracts::normalize(
            ctx,
            args,
            super::contracts::Semantic::Gemm,
            super::execution::PlanApi::gemm(),
            None,
        )?,
    )
}

fn prepare_test_geglu(
    ctx: &CudaContext,
    args: GemmGegluArgs<'_>,
) -> apxinf_core::Result<super::execution::PreparedExecution> {
    super::execution::prepare(
        ctx,
        super::contracts::normalize(
            ctx,
            args.gemm,
            super::contracts::Semantic::GemmGeglu,
            super::execution::PlanApi::gemm_geglu(),
            None,
        )?,
    )
}

#[test]
fn gemm_rejects_partially_overlapping_output_storage() {
    let ctx = CudaContext::new(0).unwrap();
    let backing = CudaBuffer::alloc(12, 0).unwrap();
    let a = backing
        .view(0, 8)
        .unwrap()
        .as_tensor(Shape::new(vec![2, 2]), DType::BF16)
        .unwrap();
    let b = tensor(0, vec![2, 2], &[1.0; 4]);
    let mut out = backing
        .view(4, 8)
        .unwrap()
        .as_tensor(Shape::new(vec![2, 2]), DType::BF16)
        .unwrap();

    let error = gemm(&ctx, GemmArgs::new(&a, &b, &mut out)).unwrap_err();
    assert!(error.to_string().contains("overlaps read-only A storage"));
}

#[test]
fn gemm_rejects_storage_misaligned_for_its_dtype() {
    let ctx = CudaContext::new(0).unwrap();
    let backing = CudaBuffer::alloc(9, 0).unwrap();
    let a = backing
        .view(1, 8)
        .unwrap()
        .as_tensor(Shape::new(vec![2, 2]), DType::BF16)
        .unwrap();
    let b = tensor(0, vec![2, 2], &[1.0; 4]);
    let mut out = tensor(0, vec![2, 2], &[0.0; 4]);

    let error = gemm(&ctx, GemmArgs::new(&a, &b, &mut out)).unwrap_err();
    assert!(error.to_string().contains("2-byte dtype"), "{error}");
}

#[test]
fn gpu_e2e_candidate_alignment_is_selected_and_keyed_from_actual_bindings() {
    let ctx = CudaContext::new(0).unwrap();
    let (m, k, n) = (64, 64, 64);
    let a = bytes_tensor(0, vec![m, k], DType::F8E4M3, &vec![0x38; m * k]);
    let b = bytes_tensor(0, vec![k, n], DType::F8E4M3, &vec![0x38; k * n]);

    // Seed the in-memory plan cache with the naturally aligned binding class.
    let mut aligned_out = zeros_tensor(0, vec![m, n], DType::F16);
    let mut aligned_args = GemmArgs::new(&a, &b, &mut aligned_out);
    aligned_args.quantization = GemmQuantization::Fp8UnitScale;
    aligned_args.policy.allow_fallback = false;
    let aligned = super::contracts::normalize(
        &ctx,
        aligned_args,
        super::contracts::Semantic::Gemm,
        super::execution::PlanApi::gemm(),
        None,
    )
    .unwrap();
    assert!(aligned.spec.output_alignment >= 16);
    let aligned_instance = super::execution::prepare(&ctx, aligned).unwrap();
    eprintln!("GPU_ALIGNMENT_ALIGNED {}", aligned_instance.summary());

    // Offset by two bytes: valid for F16, but below CUTLASS's 16-byte output
    // requirement.  This must form a different plan key and skip CUTLASS.
    let output_bytes = m * n * DType::F16.size_in_bytes();
    let backing = CudaBuffer::alloc(output_bytes + 2, 0).unwrap();
    let mut misaligned_out = backing
        .view(2, output_bytes)
        .unwrap()
        .as_tensor(Shape::new(vec![m, n]), DType::F16)
        .unwrap();
    let mut misaligned_args = GemmArgs::new(&a, &b, &mut misaligned_out);
    misaligned_args.quantization = GemmQuantization::Fp8UnitScale;
    misaligned_args.policy.allow_fallback = false;
    let misaligned = super::contracts::normalize(
        &ctx,
        misaligned_args,
        super::contracts::Semantic::Gemm,
        super::execution::PlanApi::gemm(),
        None,
    )
    .unwrap();
    assert_eq!(misaligned.spec.output_alignment, 2);
    let mut instance = super::execution::prepare(&ctx, misaligned).unwrap();
    let summary = instance.summary().to_owned();
    eprintln!("GPU_ALIGNMENT_MISALIGNED {summary}");
    assert!(summary.contains("cutlass-fp8=skip(alignment)"), "{summary}");
    assert!(
        summary.contains("cublasLt+custom-epilogue=skip(alignment)"),
        "{summary}"
    );
    assert!(
        summary.contains("cublasLt-native-fp8+custom-epilogue=skip(alignment)"),
        "{summary}"
    );
    assert!(
        !summary.contains("source=memory"),
        "alignment key aliased: {summary}"
    );

    instance.enqueue().unwrap();
    ctx.synchronize().unwrap();
    assert!(f16_values(&misaligned_out)
        .iter()
        .all(|&value| value == k as f32));
}

pub(super) fn values(tensor: &Tensor) -> Vec<f32> {
    let buffer = CudaBuffer::from_tensor(tensor).unwrap();
    let mut bytes = vec![0; buffer.len()];
    buffer.copy_to_host(&mut bytes).unwrap();
    bytes
        .chunks_exact(2)
        .map(|value| bf16::from_bits(u16::from_ne_bytes([value[0], value[1]])).to_f32())
        .collect()
}

fn f16_values(tensor: &Tensor) -> Vec<f32> {
    assert_eq!(tensor.dtype(), DType::F16);
    let buffer = CudaBuffer::from_tensor(tensor).unwrap();
    let mut bytes = vec![0; buffer.len()];
    buffer.copy_to_host(&mut bytes).unwrap();
    bytes
        .chunks_exact(2)
        .map(|value| half::f16::from_bits(u16::from_ne_bytes([value[0], value[1]])).to_f32())
        .collect()
}

fn f32_values(tensor: &Tensor) -> Vec<f32> {
    assert_eq!(tensor.dtype(), DType::F32);
    let buffer = CudaBuffer::from_tensor(tensor).unwrap();
    let mut bytes = vec![0; buffer.len()];
    buffer.copy_to_host(&mut bytes).unwrap();
    bytes
        .chunks_exact(4)
        .map(|value| f32::from_ne_bytes(value.try_into().unwrap()))
        .collect()
}

#[test]
fn fp8_unit_scale_f32_output_does_not_round_through_f16() {
    let ctx = CudaContext::new(0).unwrap();
    let (m, k, n) = (1, 16, 16);
    // E4M3 448 is 0x7e. The dot product is finite in F32 but overflows F16.
    let a = bytes_tensor(0, vec![m, k], DType::F8E4M3, &vec![0x7e; m * k]);
    let b = bytes_tensor(0, vec![k, n], DType::F8E4M3, &vec![0x7e; k * n]);
    let mut out = zeros_tensor(0, vec![m, n], DType::F32);
    let mut args = GemmArgs::new(&a, &b, &mut out);
    args.quantization = GemmQuantization::Fp8UnitScale;
    args.policy.online_tune = false;

    gemm(&ctx, args).unwrap();

    let expected = 448.0 * 448.0 * k as f32;
    assert_eq!(expected, 3_211_264.0);
    assert!(f32_values(&out).iter().all(|&value| value == expected));
}

#[test]
fn fp8_unit_scale_bf16_output_does_not_round_through_f16() {
    let ctx = CudaContext::new(0).unwrap();
    let (m, k, n) = (1, 16, 16);
    // BF16 can represent this dot product exactly, while F16 overflows it.
    let a = bytes_tensor(0, vec![m, k], DType::F8E4M3, &vec![0x7e; m * k]);
    let b = bytes_tensor(0, vec![k, n], DType::F8E4M3, &vec![0x7e; k * n]);
    let mut out = zeros_tensor(0, vec![m, n], DType::BF16);
    let mut args = GemmArgs::new(&a, &b, &mut out);
    args.quantization = GemmQuantization::Fp8UnitScale;
    args.policy.online_tune = true;
    args.policy.allow_fallback = false;

    let normalized = super::contracts::normalize(
        &ctx,
        args,
        super::contracts::Semantic::Gemm,
        super::execution::PlanApi::gemm(),
        None,
    )
    .unwrap();
    let mut instance = super::execution::prepare(&ctx, normalized).unwrap();
    let summary = instance.summary();
    assert!(
        summary.contains("cublas+custom-epilogue#0=pass"),
        "{summary}"
    );
    assert!(
        summary
            .split(',')
            .any(|entry| entry.contains("cublasLt+custom-epilogue") && entry.contains("=pass")),
        "{summary}"
    );

    instance.enqueue().unwrap();
    ctx.synchronize().unwrap();

    let expected = 448.0 * 448.0 * k as f32;
    assert_eq!(expected, 3_211_264.0);
    assert!(values(&out).iter().all(|&value| value == expected));
}

#[test]
fn gpu_e2e_registered_backends_execute_and_report_verdicts() {
    let ctx = CudaContext::new(0).unwrap();
    let (m, k, n) = (64, 64, 64);
    // E4M3 1.0 is 0x38.  This contract is supported by cuBLAS, cuBLASLt,
    // cuBLASLt native FP8, and the SM100-family CUTLASS implementation.
    let a = bytes_tensor(0, vec![m, k], DType::F8E4M3, &vec![0x38; m * k]);
    let b = bytes_tensor(0, vec![k, n], DType::F8E4M3, &vec![0x38; k * n]);
    let mut out = zeros_tensor(0, vec![m, n], DType::F16);
    let mut args = GemmArgs::new(&a, &b, &mut out);
    args.quantization = GemmQuantization::Fp8UnitScale;
    args.policy.online_tune = true;
    args.policy.allow_fallback = false;
    args.policy.graph_safe = true;

    let normalized = super::contracts::normalize(
        &ctx,
        args,
        super::contracts::Semantic::Gemm,
        super::execution::PlanApi::gemm(),
        None,
    )
    .unwrap();
    let mut instance = super::execution::prepare(&ctx, normalized).unwrap();
    let summary = instance.summary().to_owned();
    eprintln!("GPU_E2E_CANDIDATES {summary}");
    assert!(
        summary.contains("winner-graph=pass"),
        "the eager winner was not validated through CUDA Graph replay: {summary}"
    );
    for backend in [
        "cublas+custom-epilogue#0=pass",
        "cublasLt+custom-epilogue#",
        "cublasLt-native-fp8+custom-epilogue#",
        "cutlass-fp8#",
    ] {
        assert!(
            summary.contains(backend),
            "registered GPU backend was not explicitly exercised: {backend}; {summary}"
        );
    }
    // cuBLASLt/CUTLASS may reject individual tactics, but each registered
    // backend must have at least one numerically validated execution.
    for backend in [
        "cublasLt+custom-epilogue",
        "cublasLt-native-fp8+custom-epilogue",
        "cutlass-fp8",
    ] {
        let passed = summary
            .split(',')
            .any(|entry| entry.contains(backend) && entry.contains("=pass"));
        assert!(
            passed,
            "GPU backend has no passing tactic: {backend}; {summary}"
        );
    }

    instance.enqueue().unwrap();
    ctx.synchronize().unwrap();
    assert!(f16_values(&out).iter().all(|&value| value == k as f32));
}

#[test]
fn gpu_e2e_graph_replay_overwrites_sentinel_and_matches_numeric_result() {
    let ctx = CudaContext::new(0).unwrap();
    let (m, k, n) = (7, 19, 22);
    let a = tensor(0, vec![m, k], &vec![1.0; m * k]);
    let b = tensor(0, vec![k, n], &vec![1.0; k * n]);
    let mut out = tensor(0, vec![m, n], &vec![0.0; m * n]);
    let observed = CudaBuffer::from_tensor(&out).unwrap();
    let mut args = GemmArgs::new(&a, &b, &mut out);
    args.policy.online_tune = false;

    let normalized = super::contracts::normalize(
        &ctx,
        args,
        super::contracts::Semantic::Gemm,
        super::execution::PlanApi::gemm(),
        None,
    )
    .unwrap();
    let mut instance = super::execution::prepare(&ctx, normalized).unwrap();
    let graph = crate::capture(&ctx, || instance.enqueue()).unwrap();

    let sentinel: Vec<u8> = (0..m * n)
        .flat_map(|_| bf16::from_f32(-123.0).to_bits().to_ne_bytes())
        .collect();
    observed.copy_from_host(&sentinel).unwrap();
    assert!(values(&out).iter().all(|&value| value == -123.0));

    graph.replay().unwrap();
    ctx.synchronize().unwrap();
    let actual = values(&out);
    assert!(
        actual.iter().all(|&value| value == k as f32),
        "captured graph replay did not overwrite the sentinel: {actual:?}"
    );
    eprintln!("GPU_E2E_GRAPH replay overwrote sentinel and matched {k}");
}

#[test]
fn bf16_gemm_numeric_recipe_and_graph() {
    let ctx = CudaContext::new(0).unwrap();
    let (m, k, n) = (7, 19, 22);
    let a_values: Vec<_> = (0..m * k)
        .map(|i| ((i * 7 % 17) as f32 - 8.0) / 16.0)
        .collect();
    let b_values: Vec<_> = (0..k * n)
        .map(|i| ((i * 3 % 13) as f32 - 6.0) / 16.0)
        .collect();
    let a = tensor(0, vec![m, k], &a_values);
    let b = tensor(0, vec![k, n], &b_values);
    let mut out = tensor(0, vec![m, n], &vec![0.0; m * n]);
    let mut args = GemmArgs::new(&a, &b, &mut out);
    args.policy.cache_dir = Some("/tmp/apxinf-gemm-design-tests".into());
    let mut instance = super::execution::prepare(
        &ctx,
        super::contracts::normalize(
            &ctx,
            args,
            super::contracts::Semantic::Gemm,
            super::execution::PlanApi::gemm(),
            None,
        )
        .unwrap(),
    )
    .unwrap();
    eprintln!("GEMM {}", instance.summary());
    instance.enqueue().unwrap();
    ctx.synchronize().unwrap();
    let actual = values(&out);
    for row in 0..m {
        for column in 0..n {
            let expected: f32 = (0..k)
                .map(|inner| a_values[row * k + inner] * b_values[inner * n + column])
                .sum();
            assert!((actual[row * n + column] - expected).abs() < 0.01);
        }
    }

    let mut cached_args = GemmArgs::new(&a, &b, &mut out);
    cached_args.policy.cache_dir = Some("/tmp/apxinf-gemm-design-tests".into());
    cached_args.policy.online_tune = false;
    cached_args.policy.allow_fallback = false;
    gemm(&ctx, cached_args).unwrap();

    let graph = crate::capture(&ctx, || instance.enqueue()).unwrap();
    graph.replay().unwrap();
    ctx.synchronize().unwrap();
    assert_eq!(values(&out), actual);
    drop(a);
    drop(b);
    graph.replay().unwrap();
    ctx.synchronize().unwrap();
    assert_eq!(values(&out), actual);
}

#[test]
fn compatible_recipe_is_only_a_retuning_hint() {
    let cache_dir = std::env::temp_dir().join(format!(
        "apxinf-compatible-recipe-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir(&cache_dir).unwrap();
    let cache = cache_dir.to_string_lossy().into_owned();

    let run = |ctx: &CudaContext, online_tune: bool| {
        let a = tensor(0, vec![2, 3], &[1.0; 6]);
        let b = tensor(0, vec![3, 4], &[1.0; 12]);
        let mut out = tensor(0, vec![2, 4], &[0.0; 8]);
        let mut args = GemmArgs::new(&a, &b, &mut out);
        args.policy.cache_dir = Some(cache.clone());
        args.policy.online_tune = online_tune;
        args.policy.allow_fallback = false;
        super::execution::prepare(
            ctx,
            super::contracts::normalize(
                ctx,
                args,
                super::contracts::Semantic::Gemm,
                super::execution::PlanApi::gemm(),
                None,
            )?,
        )
    };
    let remove_performance_recipe = || {
        for entry in std::fs::read_dir(&cache_dir).unwrap() {
            let path = entry.unwrap().path();
            let contents = std::fs::read_to_string(&path).unwrap();
            if contents
                .lines()
                .next()
                .unwrap_or_default()
                .contains("|performance|")
            {
                std::fs::remove_file(path).unwrap();
            }
        }
    };

    let first = run(&CudaContext::new(0).unwrap(), true).unwrap();
    assert!(first.summary().contains("tuned preferred=0"));
    drop(first);
    remove_performance_recipe();

    let retuned = run(&CudaContext::new(0).unwrap(), true).unwrap();
    assert!(
        retuned.summary().contains("tuned preferred=1"),
        "compatible winner must be tried first but still fully tuned: {}",
        retuned.summary()
    );
    drop(retuned);
    remove_performance_recipe();

    for entry in std::fs::read_dir(&cache_dir).unwrap() {
        let path = entry.unwrap().path();
        let contents = std::fs::read_to_string(&path).unwrap();
        let key = contents.lines().next().unwrap_or_default();
        if key.contains("|compatible-hint") {
            std::fs::write(path, format!("{key}\n999 999 999 999\n")).unwrap();
        }
    }
    let invalid_hint = run(&CudaContext::new(0).unwrap(), true).unwrap();
    assert!(
        invalid_hint.summary().contains("tuned preferred=0"),
        "missing or version-incompatible candidates must be filtered: {}",
        invalid_hint.summary()
    );
    drop(invalid_hint);
    remove_performance_recipe();

    let miss = match run(&CudaContext::new(0).unwrap(), false) {
        Ok(_) => panic!("compatible hint was incorrectly accepted as fully tuned"),
        Err(error) => error,
    };
    assert!(miss.to_string().contains("recipe miss"));
    std::fs::remove_dir_all(cache_dir).unwrap();
}

fn scratch_cache_dir(label: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "apxinf-{}-{}-{}",
        label,
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// F03: a tuned choice belongs to the semantic request and constraints, not
/// to one allocation address. The test observes the real native tuning path;
/// it does not inspect a cache key or plan summary.
#[test]
fn tuning_reuse_is_independent_of_tensor_addresses() {
    let cache_dir = scratch_cache_dir("acceptance-f03-address-independent-tuning");
    let cache = cache_dir.to_string_lossy().into_owned();
    let ctx = CudaContext::new(0).unwrap();

    let a1 = tensor(0, vec![2, 3], &[1.0; 6]);
    let b1 = tensor(0, vec![3, 4], &[1.0; 12]);
    let mut out1 = tensor(0, vec![2, 4], &[0.0; 8]);
    let a2 = tensor(0, vec![2, 3], &[2.0; 6]);
    let b2 = tensor(0, vec![3, 4], &[1.0; 12]);
    let mut out2 = tensor(0, vec![2, 4], &[0.0; 8]);

    let run = |a: &Tensor, b: &Tensor, out: &mut Tensor| {
        let mut args = GemmArgs::new(a, b, out);
        args.policy.cache_dir = Some(cache.clone());
        args.policy.online_tune = true;
        args.policy.allow_fallback = false;
        gemm(&ctx, args).unwrap();
    };

    reset_tuning_observers();
    run(&a1, &b1, &mut out1);
    let after_first = tuning_observers();
    assert_eq!(after_first.0, 1, "the cold request did not actually tune");
    assert!(
        after_first.1 > 0,
        "the cold request did not evaluate any candidate configurations"
    );
    assert!(values(&out1).iter().all(|&value| value == 3.0));

    run(&a2, &b2, &mut out2);
    let after_second = tuning_observers();
    assert_eq!(
        after_second, after_first,
        "changing only tensor addresses repeated tuning or configuration timing"
    );
    assert!(values(&out2).iter().all(|&value| value == 6.0));

    run(&a1, &b1, &mut out1);
    assert_eq!(tuning_observers(), after_first);
    assert!(values(&out1).iter().all(|&value| value == 3.0));

    std::fs::remove_dir_all(cache_dir).unwrap();
}

/// F10: after one ordinary eager request has prepared and warmed the fixed
/// bindings, the next 100 submissions must be pure enqueues. This assertion
/// intentionally fails on implementations that rebuild an instance or force
/// a stream synchronization on every public L3 call.
#[test]
fn prepared_eager_execution_has_no_repeated_setup() {
    let ctx = CudaContext::new(0).unwrap();
    let a = tensor(0, vec![8, 16], &[1.0; 8 * 16]);
    let b = tensor(0, vec![16, 8], &[1.0; 16 * 8]);
    let mut out = tensor(0, vec![8, 8], &[0.0; 8 * 8]);

    let submit = |out: &mut Tensor| {
        let mut args = GemmArgs::new(&a, &b, out);
        args.policy.online_tune = false;
        gemm(&ctx, args).unwrap();
    };

    // Establish the choice and all execution resources before the measured
    // steady-state window.
    submit(&mut out);

    reset_tuning_observers();
    super::execution::reset_prepared_execution_create_count();
    crate::buffer::reset_device_allocation_count();
    crate::stream::reset_synchronize_count();

    for _ in 0..100 {
        submit(&mut out);
    }

    // Snapshot before the test's one explicit final wait and D2H read.
    let (tuning, configuration_evaluations) = tuning_observers();
    let provider_prepares = super::execution::prepared_execution_create_count();
    let framework_allocations = crate::buffer::device_allocation_count();
    let framework_synchronizations = crate::stream::synchronize_count();

    ctx.synchronize().unwrap();
    assert!(values(&out).iter().all(|&value| value == 16.0));

    let mut violations = Vec::new();
    if tuning != 0 {
        violations.push(format!("tuning={tuning}, expected 0"));
    }
    if configuration_evaluations != 0 {
        violations.push(format!(
            "configuration_evaluations={configuration_evaluations}, expected 0"
        ));
    }
    if provider_prepares != 0 {
        violations.push(format!("provider_prepares={provider_prepares}, expected 0"));
    }
    if framework_allocations != 0 {
        violations.push(format!(
            "framework_device_allocations={framework_allocations}, expected 0"
        ));
    }
    if framework_synchronizations != 0 {
        violations.push(format!(
            "framework_stream_synchronizations={framework_synchronizations}, expected 0"
        ));
    }
    assert!(
        violations.is_empty(),
        "eager steady state performed forbidden setup:\n{}",
        violations.join("\n")
    );
}

/// A plan is shared by every caller whose Spec matches, and alpha is no longer
/// part of that Spec. Executing the same shape twice with different alphas
/// must therefore still scale each result independently: if alpha were cached
/// inside the plan the second execution would silently reuse the first one.
#[test]
fn gpu_e2e_alpha_is_bound_per_execution_not_baked_into_the_plan() {
    let cache_dir = scratch_cache_dir("gemm-alpha-binding");
    let cache = cache_dir.to_string_lossy().into_owned();
    let ctx = CudaContext::new(0).unwrap();
    let (m, k, n) = (5, 11, 8);
    let a_values: Vec<f32> = (0..m * k)
        .map(|i| ((i * 5 % 13) as f32 - 6.0) / 8.0)
        .collect();
    let b_values: Vec<f32> = (0..k * n)
        .map(|i| ((i * 3 % 11) as f32 - 5.0) / 8.0)
        .collect();
    let a = tensor(0, vec![m, k], &a_values);
    let b = tensor(0, vec![k, n], &b_values);

    let run = |alpha: f32| {
        let mut out = tensor(0, vec![m, n], &vec![0.0; m * n]);
        let mut args = GemmArgs::new(&a, &b, &mut out);
        args.alpha = alpha;
        args.policy.cache_dir = Some(cache.clone());
        let mut instance = super::execution::prepare(
            &ctx,
            super::contracts::normalize(
                &ctx,
                args,
                super::contracts::Semantic::Gemm,
                super::execution::PlanApi::gemm(),
                None,
            )
            .unwrap(),
        )
        .unwrap();
        let summary = instance.summary().to_owned();
        instance.enqueue().unwrap();
        ctx.synchronize().unwrap();
        (values(&out), summary)
    };
    let expected = |alpha: f32| -> Vec<f32> {
        let mut values = vec![0.0f32; m * n];
        for row in 0..m {
            for column in 0..n {
                values[row * n + column] = alpha
                    * (0..k)
                        .map(|inner| a_values[row * k + inner] * b_values[inner * n + column])
                        .sum::<f32>();
            }
        }
        values
    };

    let (doubled, first) = run(2.0);
    let (tripled, second) = run(3.0);
    eprintln!("GPU_E2E_ALPHA first=[{first}] second=[{second}]");
    for (index, want) in expected(2.0).into_iter().enumerate() {
        assert!(
            (doubled[index] - want).abs() < 0.02,
            "alpha=2 element {index} is {} but should be {want}",
            doubled[index]
        );
    }
    for (index, want) in expected(3.0).into_iter().enumerate() {
        assert!(
            (tripled[index] - want).abs() < 0.02,
            "alpha=3 element {index} is {} but should be {want} \
             (a shared plan reused the previous alpha)",
            tripled[index]
        );
    }
    assert_ne!(
        doubled, tripled,
        "alpha must change the numeric result even when the plan is shared"
    );
    std::fs::remove_dir_all(cache_dir).unwrap();
}

/// Only the unit/non-unit scale predicate belongs to the tuning key, so calls
/// that differ solely by alpha must land on one recipe instead of tuning once
/// per calibration scale.
#[test]
fn gpu_e2e_scale_values_do_not_fragment_the_tuning_cache() {
    let cache_dir = scratch_cache_dir("gemm-scale-fragmentation");
    let cache = cache_dir.to_string_lossy().into_owned();
    let (m, k, n) = (6, 9, 12);
    let run = |ctx: &CudaContext, alpha: f32, online_tune: bool| -> String {
        let a = tensor(0, vec![m, k], &vec![0.5; m * k]);
        let b = tensor(0, vec![k, n], &vec![0.25; k * n]);
        let mut out = tensor(0, vec![m, n], &vec![0.0; m * n]);
        let mut args = GemmArgs::new(&a, &b, &mut out);
        args.alpha = alpha;
        args.policy.cache_dir = Some(cache.clone());
        args.policy.online_tune = online_tune;
        args.policy.allow_fallback = false;
        let instance = super::execution::prepare(
            ctx,
            super::contracts::normalize(
                ctx,
                args,
                super::contracts::Semantic::Gemm,
                super::execution::PlanApi::gemm(),
                None,
            )
            .unwrap(),
        )
        .unwrap();
        instance.summary().to_owned()
    };

    let ctx = CudaContext::new(0).unwrap();
    let tuned = run(&ctx, 2.0, true);
    assert!(
        tuned.contains("source=tuned"),
        "the first call must actually tune: {tuned}"
    );
    let reused = run(&ctx, 3.0, true);
    assert!(
        reused.contains("source=memory"),
        "a different alpha must hit the same tuned plan: {reused}"
    );
    // A fresh runtime has no in-memory plan, so this proves the persisted
    // recipe is shared too and that tuning is not repeated.
    let restored = run(&CudaContext::new(0).unwrap(), 4.0, false);
    assert!(
        restored.contains("source=recipe"),
        "a different alpha must restore the persisted recipe: {restored}"
    );
    std::fs::remove_dir_all(cache_dir).unwrap();
}

#[test]
fn captured_graph_retains_workspace_instance_and_tensor_storage() {
    let ctx = CudaContext::new(0).unwrap();
    let a = tensor(0, vec![2, 3], &[1.0; 6]);
    let b = tensor(0, vec![3, 4], &[1.0; 12]);
    let mut out = tensor(0, vec![2, 4], &[0.0; 8]);
    let observed = out.clone();
    let workspace = GraphWorkspace::new(4096, 0).unwrap();

    prepare_with_workspace(&workspace, || gemm(&ctx, GemmArgs::new(&a, &b, &mut out))).unwrap();

    let graph = crate::capture(&ctx, || {
        with_workspace(&workspace, || gemm(&ctx, GemmArgs::new(&a, &b, &mut out)))
    })
    .unwrap();

    drop(workspace);
    drop(a);
    drop(b);
    drop(out);
    graph.replay().unwrap();
    ctx.synchronize().unwrap();
    assert!(values(&observed).iter().all(|&value| value == 3.0));
}

#[test]
fn quantization_contract_becomes_part_of_the_gemm_key() {
    let ctx = CudaContext::new(0).unwrap();
    let (m, k, n) = (2, 16, 8);
    let row_scales = scales(0, &[1.0; 2]);
    let channel_scales = scales(0, &[1.0; 8]);

    let fp8_a = zeros_tensor(0, vec![m, k], DType::F8E4M3);
    let fp8_b = zeros_tensor(0, vec![k, n], DType::F8E4M3);
    let mut fp8_out = zeros_tensor(0, vec![m, n], DType::BF16);
    let fp8 = super::contracts::normalize(
        &ctx,
        GemmArgs::fp8(&fp8_a, &row_scales, &fp8_b, &channel_scales, &mut fp8_out),
        super::contracts::Semantic::Gemm,
        super::execution::PlanApi::gemm(),
        None,
    )
    .unwrap();
    assert_eq!(fp8.spec.quantization, 2);
    assert_eq!(fp8.spec.accumulation_dtype, 0);

    let int8_a = zeros_tensor(0, vec![m, k], DType::I8);
    let int8_b = zeros_tensor(0, vec![k, n], DType::I8);
    let mut int8_out = zeros_tensor(0, vec![m, n], DType::BF16);
    let w8a8 = super::contracts::normalize(
        &ctx,
        GemmArgs::w8a8(
            &int8_a,
            &row_scales,
            &int8_b,
            &channel_scales,
            &mut int8_out,
        ),
        super::contracts::Semantic::Gemm,
        super::execution::PlanApi::gemm(),
        None,
    )
    .unwrap();
    assert_eq!(w8a8.spec.quantization, 3);
    assert_eq!(w8a8.spec.accumulation_dtype, 5);
}

#[test]
fn scaled_fp8_and_w8a8_use_the_canonical_kn_weight_contract() {
    let ctx = CudaContext::new(0).unwrap();
    let (m, k, n) = (2, 16, 8);
    let row_scales = scales(0, &[0.5, 0.25]);
    let channel_scales = scales(0, &[2.0; 8]);

    // E4M3 1.0 is 0x38. Both rows therefore produce K * row * channel.
    let fp8_a = bytes_tensor(0, vec![m, k], DType::F8E4M3, &vec![0x38; m * k]);
    let fp8_b = bytes_tensor(0, vec![k, n], DType::F8E4M3, &vec![0x38; k * n]);
    let mut fp8_out = zeros_tensor(0, vec![m, n], DType::BF16);
    let mut fp8_args = GemmArgs::fp8(&fp8_a, &row_scales, &fp8_b, &channel_scales, &mut fp8_out);
    fp8_args.policy.online_tune = false;
    gemm(&ctx, fp8_args).unwrap();
    let fp8_values = values(&fp8_out);
    assert!(fp8_values[..n].iter().all(|&value| value == 16.0));
    assert!(fp8_values[n..].iter().all(|&value| value == 8.0));

    let int8_a = bytes_tensor(0, vec![m, k], DType::I8, &vec![1; m * k]);
    let int8_b = bytes_tensor(0, vec![k, n], DType::I8, &vec![1; k * n]);
    let mut int8_out = zeros_tensor(0, vec![m, n], DType::BF16);
    let mut int8_args = GemmArgs::w8a8(
        &int8_a,
        &row_scales,
        &int8_b,
        &channel_scales,
        &mut int8_out,
    );
    int8_args.policy.online_tune = false;
    gemm(&ctx, int8_args).unwrap();
    let int8_values = values(&int8_out);
    assert!(int8_values[..n].iter().all(|&value| value == 16.0));
    assert!(int8_values[n..].iter().all(|&value| value == 8.0));
}

#[test]
fn graph_workspace_reuses_one_native_instance_across_forward_traversals() {
    let ctx = CudaContext::new(0).unwrap();
    let a = tensor(0, vec![2, 3], &[1.0; 6]);
    let b = tensor(0, vec![3, 4], &[1.0; 12]);
    let mut out = tensor(0, vec![2, 4], &[0.0; 8]);
    let workspace = GraphWorkspace::new(4096, 0).unwrap();

    super::execution::reset_prepared_execution_create_count();
    prepare_with_workspace(&workspace, || {
        let mut args = GemmArgs::new(&a, &b, &mut out);
        args.policy.online_tune = false;
        gemm(&ctx, args)
    })
    .unwrap();
    assert_eq!(super::execution::prepared_execution_create_count(), 1);

    for _ in 0..2 {
        with_workspace(&workspace, || {
            let mut args = GemmArgs::new(&a, &b, &mut out);
            args.policy.online_tune = false;
            gemm(&ctx, args)
        })
        .unwrap();
    }

    // Native instance creation includes provider resource creation and warmup.
    // Repeating the same public L3 traversal must hit the workspace instance.
    assert_eq!(super::execution::prepared_execution_create_count(), 1);
    assert!(values(&out).iter().all(|&value| value == 3.0));
}

#[test]
fn gpu_e2e_cached_plan_rebuilds_and_drops_provider_private_instances() {
    let ctx = CudaContext::new(0).unwrap();
    let a = tensor(0, vec![8, 16], &[1.0; 8 * 16]);
    let b = tensor(0, vec![16, 8], &[1.0; 16 * 8]);

    for iteration in 0..4 {
        let mut out = tensor(0, vec![8, 8], &[0.0; 8 * 8]);
        let mut args = GemmArgs::new(&a, &b, &mut out);
        args.policy.online_tune = true;
        args.policy.allow_fallback = false;

        let mut prepared = prepare_test_gemm(&ctx, args).unwrap();
        if iteration != 0 {
            assert!(
                prepared.summary().contains("source=memory"),
                "provider recipe was not rebuilt from the compacted plan: {}",
                prepared.summary()
            );
        }
        prepared.enqueue().unwrap();
        ctx.synchronize().unwrap();
        drop(prepared);

        assert!(values(&out).iter().all(|&value| value == 16.0));
    }
}

#[test]
fn gpu_e2e_cutlass_geglu_prepack_is_bound_to_allocation_and_version() {
    let ctx = CudaContext::new(0).unwrap();
    let (m, k, n) = (522, 2048, 32768);
    let a = zeros_tensor(0, vec![m, k], DType::F8E4M3);
    let b = zeros_tensor(0, vec![k, n], DType::F8E4M3);
    let mut out = zeros_tensor(0, vec![m, n / 2], DType::F8E4M3);
    let cache_dir = std::env::temp_dir().join(format!(
        "apxinf-geglu-prepack-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&cache_dir).unwrap();
    let cache_dir_string = cache_dir.to_string_lossy().into_owned();

    let normalized = {
        let mut args = GemmArgs::new(&a, &b, &mut out).with_immutable_weight(WeightVersion::new(7));
        args.quantization = GemmQuantization::Fp8UnitScale;
        args.policy.online_tune = false;
        args.policy.allow_fallback = false;
        args.policy.cache_dir = Some(cache_dir_string.clone());
        super::contracts::normalize(
            &ctx,
            args,
            super::contracts::Semantic::GemmGeglu,
            super::execution::PlanApi::gemm_geglu(),
            None,
        )
        .unwrap()
    };
    // Provider 3, implementation 2, version 2 is the FP8 CUTLASS GeGLU
    // candidate. Seeding avoids an enormous CPU-reference autotune while
    // keeping instance construction and public enqueue/capture paths real.
    super::execution::seed_recipe(&ctx, &normalized, 3, 2, 2, 0).unwrap();
    drop(normalized);

    let workspace = GraphWorkspace::new(1, 0).unwrap();
    super::execution::reset_prepared_execution_create_count();
    let (mut first, same_version, changed_version) = prepare_with_workspace(&workspace, || {
        let first = {
            let mut args =
                GemmArgs::new(&a, &b, &mut out).with_immutable_weight(WeightVersion::new(7));
            args.quantization = GemmQuantization::Fp8UnitScale;
            args.policy.online_tune = false;
            args.policy.allow_fallback = false;
            args.policy.cache_dir = Some(cache_dir_string.clone());
            prepare_test_geglu(&ctx, GemmGegluArgs { gemm: args })?
        };

        let same = {
            let mut args =
                GemmArgs::new(&a, &b, &mut out).with_immutable_weight(WeightVersion::new(7));
            args.quantization = GemmQuantization::Fp8UnitScale;
            args.policy.online_tune = false;
            args.policy.allow_fallback = false;
            args.policy.cache_dir = Some(cache_dir_string.clone());
            prepare_test_geglu(&ctx, GemmGegluArgs { gemm: args })?
        };
        let changed = {
            let mut args =
                GemmArgs::new(&a, &b, &mut out).with_immutable_weight(WeightVersion::new(8));
            args.quantization = GemmQuantization::Fp8UnitScale;
            args.policy.online_tune = false;
            args.policy.allow_fallback = false;
            args.policy.cache_dir = Some(cache_dir_string.clone());
            prepare_test_geglu(&ctx, GemmGegluArgs { gemm: args })?
        };
        Ok((first, same, changed))
    })
    .unwrap();

    assert_eq!(super::execution::prepared_execution_create_count(), 2);
    assert_eq!(first.weight_prepack_count(), 1);
    assert_eq!(same_version.weight_prepack_count(), 1);
    assert_eq!(changed_version.weight_prepack_count(), 1);

    first.enqueue().unwrap();
    first.enqueue().unwrap();
    ctx.synchronize().unwrap();
    assert_eq!(first.weight_prepack_count(), 1);

    let graph = crate::capture(&ctx, || first.enqueue()).unwrap();
    graph.replay().unwrap();
    graph.replay().unwrap();
    ctx.synchronize().unwrap();
    assert_eq!(first.weight_prepack_count(), 1);

    drop((graph, first, same_version, changed_version));
    drop(workspace);

    // Without the explicit immutability/version contract, each launch must
    // conservatively repack because the caller may have changed B in place.
    let mut mutable_args = GemmArgs::new(&a, &b, &mut out);
    mutable_args.quantization = GemmQuantization::Fp8UnitScale;
    mutable_args.policy.online_tune = false;
    mutable_args.policy.allow_fallback = false;
    mutable_args.policy.cache_dir = Some(cache_dir_string);
    let mut mutable = prepare_test_geglu(&ctx, GemmGegluArgs { gemm: mutable_args }).unwrap();
    assert_eq!(mutable.weight_prepack_count(), 1); // instance warmup
    mutable.enqueue().unwrap();
    ctx.synchronize().unwrap();
    assert_eq!(mutable.weight_prepack_count(), 2);

    drop(mutable);
    std::fs::remove_dir_all(cache_dir).unwrap();
}
