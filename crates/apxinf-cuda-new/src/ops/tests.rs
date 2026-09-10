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

fn tensor(device: usize, shape: Vec<usize>, values: &[f32]) -> Tensor {
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

fn f32_tensor(device: usize, shape: Vec<usize>, values: &[f32]) -> Tensor {
    let bytes: Vec<_> = values
        .iter()
        .flat_map(|value| value.to_ne_bytes())
        .collect();
    bytes_tensor(device, shape, DType::F32, &bytes)
}

fn zeros_tensor(device: usize, shape: Vec<usize>, dtype: DType) -> Tensor {
    let bytes = vec![0; shape.iter().product::<usize>() * dtype.size_in_bytes()];
    let buffer = CudaBuffer::alloc(bytes.len(), device).unwrap();
    buffer.copy_from_host(&bytes).unwrap();
    buffer.as_tensor(Shape::new(shape), dtype).unwrap()
}

fn bytes_tensor(device: usize, shape: Vec<usize>, dtype: DType, bytes: &[u8]) -> Tensor {
    assert_eq!(
        bytes.len(),
        shape.iter().product::<usize>() * dtype.size_in_bytes()
    );
    let buffer = CudaBuffer::alloc(bytes.len(), device).unwrap();
    buffer.copy_from_host(bytes).unwrap();
    buffer.as_tensor(Shape::new(shape), dtype).unwrap()
}

fn scales(device: usize, values: &[f32]) -> Tensor {
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

fn decode_e4m3(bits: u8) -> Option<f32> {
    let sign = if bits & 0x80 == 0 { 1.0 } else { -1.0 };
    let exponent = (bits >> 3) & 0x0f;
    let mantissa = bits & 0x07;
    if exponent == 0x0f && mantissa == 0x07 {
        return None;
    }
    let magnitude = if exponent == 0 {
        (mantissa as f32) * 2.0_f32.powi(-9)
    } else {
        (1.0 + mantissa as f32 / 8.0) * 2.0_f32.powi(exponent as i32 - 7)
    };
    Some(sign * magnitude)
}

fn quantize_e4m3(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .map(|&value| {
            let mut best = 0_u8;
            let mut best_error = f32::INFINITY;
            for candidate in 0_u16..=u8::MAX as u16 {
                let bits = candidate as u8;
                let Some(decoded) = decode_e4m3(bits) else {
                    continue;
                };
                let error = (decoded - value).abs();
                if error < best_error {
                    best = bits;
                    best_error = error;
                }
            }
            best
        })
        .collect()
}

fn next_random(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

/// A deterministic mixture of zero, ordinary random, dynamic-range,
/// cancellation, and FP8-boundary values. Boundary positions are staggered
/// between A and B so their dot products remain finite in an F16 output.
fn validation_matrix(
    rows: usize,
    columns: usize,
    seed: u64,
    boundary_on_rows: bool,
    boundary_phase: usize,
) -> Vec<f32> {
    const RANDOM_VALUES: [f32; 9] = [-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0];
    const DYNAMIC_VALUES: [f32; 9] = [-16.0, -4.0, -1.0, -0.25, 0.0, 0.25, 1.0, 4.0, 16.0];
    let mut state = seed;
    let mut values = Vec::with_capacity(rows * columns);
    for row in 0..rows {
        for column in 0..columns {
            let random = next_random(&mut state) as usize;
            let selector = (row * columns + column) % 5;
            let boundary_index = if boundary_on_rows { row } else { column };
            let value = if boundary_index % 16 == boundary_phase {
                if column % 2 == 0 {
                    448.0
                } else {
                    -448.0
                }
            } else {
                match selector {
                    0 => 0.0,
                    1 => RANDOM_VALUES[random % RANDOM_VALUES.len()],
                    2 => DYNAMIC_VALUES[random % DYNAMIC_VALUES.len()],
                    3 => {
                        let magnitude = RANDOM_VALUES[(random / 7) % RANDOM_VALUES.len()].abs();
                        if column % 2 == 0 {
                            magnitude
                        } else {
                            -magnitude
                        }
                    }
                    _ => {
                        if random & 1 == 0 {
                            2.0
                        } else {
                            -2.0
                        }
                    }
                }
            };
            values.push(value);
        }
    }
    values
}

fn validation_bias(columns: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    (0..columns)
        .map(|column| {
            let value = (next_random(&mut state) % 9) as f32 / 4.0;
            if column % 3 == 0 {
                0.0
            } else if column % 2 == 0 {
                value
            } else {
                -value
            }
        })
        .collect()
}

fn assert_all_applicable_candidates_checked(summary: &str, expected_backends: &[&str]) {
    assert!(summary.contains("reference=original-fp32"), "{summary}");
    assert!(summary.contains("max_element="), "{summary}");
    assert!(summary.contains("rel_l2="), "{summary}");
    assert!(summary.contains("cosine="), "{summary}");
    assert!(
        !summary.contains("reject(numeric:"),
        "an applicable candidate failed the shared FP32 reference: {summary}"
    );
    for backend in expected_backends {
        assert!(
            summary.contains(backend),
            "candidate was not visited: {backend}; {summary}"
        );
    }
}

fn run_deterministic_reference_corpus(fp8: bool) {
    let ctx = CudaContext::new(0).unwrap();
    let (m, k, n) = (8, 16, 16);
    let raw_a = validation_matrix(m, k, 0x5eed_a11c_e001, false, 0);
    let raw_b = validation_matrix(k, n, 0x5eed_b22d_e002, true, 1);
    let raw_geglu_b = validation_matrix(k, 2 * n, 0x5eed_b33e_e003, true, 1);
    let raw_bias = validation_bias(n, 0x5eed_b1a5_e004);

    let a = if fp8 {
        bytes_tensor(0, vec![m, k], DType::F8E4M3, &quantize_e4m3(&raw_a))
    } else {
        tensor(0, vec![m, k], &raw_a)
    };
    let b = if fp8 {
        bytes_tensor(0, vec![k, n], DType::F8E4M3, &quantize_e4m3(&raw_b))
    } else {
        tensor(0, vec![k, n], &raw_b)
    };
    let geglu_b = if fp8 {
        bytes_tensor(
            0,
            vec![k, 2 * n],
            DType::F8E4M3,
            &quantize_e4m3(&raw_geglu_b),
        )
    } else {
        tensor(0, vec![k, 2 * n], &raw_geglu_b)
    };
    let bias = if fp8 {
        f32_tensor(0, vec![n], &raw_bias)
    } else {
        tensor(0, vec![n], &raw_bias)
    };

    let configure = |args: &mut GemmArgs<'_>| {
        if fp8 {
            args.quantization = GemmQuantization::Fp8UnitScale;
        }
        args.policy.allow_fallback = false;
        args.policy.graph_safe = false;
    };
    let vendor_backends = ["cublas+custom-epilogue", "cublasLt+custom-epilogue"];

    let mut gemm_out = zeros_tensor(0, vec![m, n], if fp8 { DType::F16 } else { DType::BF16 });
    let mut args = GemmArgs::new(&a, &b, &mut gemm_out);
    configure(&mut args);
    let prepared = prepare_with_original_fp32(
        &ctx,
        args,
        super::contracts::Semantic::Gemm,
        super::execution::PlanApi::gemm(),
        None,
        super::contracts::ValidationReference::new(&raw_a, &raw_b),
    )
    .unwrap();
    let expected = if fp8 {
        vec![
            "cublas+custom-epilogue",
            "cublasLt+custom-epilogue",
            "cublasLt-native-fp8+custom-epilogue",
            "cutlass-fp8",
        ]
    } else {
        vendor_backends.to_vec()
    };
    assert_all_applicable_candidates_checked(prepared.summary(), &expected);

    for semantic in [
        super::contracts::Semantic::GemmBias,
        super::contracts::Semantic::GemmBiasGelu,
    ] {
        let mut out = zeros_tensor(0, vec![m, n], if fp8 { DType::F32 } else { DType::BF16 });
        let mut args = GemmArgs::new(&a, &b, &mut out);
        configure(&mut args);
        let api = if semantic == super::contracts::Semantic::GemmBias {
            super::execution::PlanApi::gemm_bias()
        } else {
            super::execution::PlanApi::gemm_bias_gelu()
        };
        let prepared = prepare_with_original_fp32(
            &ctx,
            args,
            semantic,
            api,
            Some(&bias),
            super::contracts::ValidationReference::with_bias(&raw_a, &raw_b, &raw_bias),
        )
        .unwrap();
        assert_all_applicable_candidates_checked(prepared.summary(), &vendor_backends);
    }

    let mut geglu_out = zeros_tensor(0, vec![m, n], if fp8 { DType::F32 } else { DType::BF16 });
    let mut args = GemmArgs::new(&a, &geglu_b, &mut geglu_out);
    configure(&mut args);
    let prepared = prepare_with_original_fp32(
        &ctx,
        args,
        super::contracts::Semantic::GemmGeglu,
        super::execution::PlanApi::gemm_geglu(),
        None,
        super::contracts::ValidationReference::new(&raw_a, &raw_geglu_b),
    )
    .unwrap();
    assert_all_applicable_candidates_checked(prepared.summary(), &vendor_backends);
}

fn prepare_with_original_fp32<'a>(
    ctx: &CudaContext,
    args: GemmArgs<'a>,
    semantic: super::contracts::Semantic,
    api: super::execution::PlanApi,
    bias: Option<&'a Tensor>,
    reference: super::contracts::ValidationReference<'a>,
) -> apxinf_core::Result<PreparedExecution> {
    let normalized = super::contracts::normalize(ctx, args, semantic, api, bias)?;
    let normalized = super::contracts::with_validation_reference(normalized, reference)?;
    super::execution::prepare(ctx, normalized)
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

fn values(tensor: &Tensor) -> Vec<f32> {
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
fn gemm_geglu_is_a_separate_semantic_domain() {
    let ctx = CudaContext::new(0).unwrap();
    let (m, k, n) = (3, 5, 1024);
    let a_values = vec![0.25; m * k];
    let b_values: Vec<_> = (0..k * n).map(|i| ((i % 11) as f32 - 5.0) / 16.0).collect();
    let a = tensor(0, vec![m, k], &a_values);
    let b = tensor(0, vec![k, n], &b_values);
    let mut out = tensor(0, vec![m, n / 2], &vec![0.0; m * n / 2]);
    let mut args = GemmArgs::new(&a, &b, &mut out);
    args.policy.online_tune = false;
    gemm_geglu(&ctx, GemmGegluArgs { gemm: args }).unwrap();
    let actual = values(&out);
    assert_eq!(actual.len(), m * n / 2);
    for row in 0..m {
        for column in 0..n / 2 {
            let dot = |target| {
                bf16::from_f32(
                    (0..k)
                        .map(|inner| a_values[row * k + inner] * b_values[inner * n + target])
                        .sum(),
                )
                .to_f32()
            };
            let gate = dot(column);
            let expected = 0.5
                * gate
                * (1.0 + (0.79788456 * (gate + 0.044715 * gate.powi(3))).tanh())
                * dot(column + n / 2);
            assert!((actual[row * n / 2 + column] - expected).abs() < 0.01);
        }
    }
}

#[test]
fn gemm_bias_gelu_has_its_own_semantic_api() {
    let ctx = CudaContext::new(0).unwrap();
    let a = tensor(0, vec![2, 3], &[1.0; 6]);
    let b = tensor(0, vec![3, 4], &[1.0; 12]);
    let bias = tensor(0, vec![4], &[0.5; 4]);
    let mut out = tensor(0, vec![2, 4], &[0.0; 8]);
    let mut args = GemmArgs::new(&a, &b, &mut out);
    args.policy.online_tune = false;
    gemm_bias_gelu(
        &ctx,
        GemmBiasGeluArgs {
            gemm: args,
            bias: &bias,
        },
    )
    .unwrap();
    let output = values(&out);
    let x: f32 = 3.5;
    let expected = 0.5 * x * (1.0 + (0.79788456 * (x + 0.044715 * x.powi(3))).tanh());
    assert!(output.iter().all(|value| (*value - expected).abs() < 0.02));
}

#[test]
fn gemm_bias_has_its_own_semantic_api() {
    let ctx = CudaContext::new(0).unwrap();
    let a = tensor(0, vec![2, 3], &[1.0; 6]);
    let b = tensor(0, vec![3, 4], &[1.0; 12]);
    let bias = tensor(0, vec![4], &[0.5, -0.5, 1.0, -1.0]);
    let mut out = tensor(0, vec![2, 4], &[0.0; 8]);
    let mut args = GemmArgs::new(&a, &b, &mut out);
    args.policy.online_tune = false;
    gemm_bias(
        &ctx,
        GemmBiasArgs {
            gemm: args,
            bias: &bias,
        },
    )
    .unwrap();
    let expected = [3.5, 2.5, 4.0, 2.0, 3.5, 2.5, 4.0, 2.0];
    for (actual, expected) in values(&out).iter().zip(expected) {
        assert!((*actual - expected).abs() < 0.01);
    }
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
fn original_fp32_validation_requires_the_l3_semantic_operands() {
    let ctx = CudaContext::new(0).unwrap();
    let a = tensor(0, vec![2, 3], &[1.0; 6]);
    let b = tensor(0, vec![3, 4], &[1.0; 12]);
    let bias = tensor(0, vec![4], &[0.0; 4]);
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
        super::contracts::ValidationReference::new(&[1.0; 5], &[1.0; 12]),
    ) {
        Ok(_) => panic!("invalid validation operands were accepted"),
        Err(error) => error,
    };
    assert!(error
        .to_string()
        .contains("validation operands do not match"));

    let args = GemmArgs::new(&a, &b, &mut out);
    let normalized = super::contracts::normalize(
        &ctx,
        args,
        super::contracts::Semantic::GemmBias,
        super::execution::PlanApi::gemm_bias(),
        Some(&bias),
    )
    .unwrap();
    let error = match super::contracts::with_validation_reference(
        normalized,
        super::contracts::ValidationReference::new(&[1.0; 6], &[1.0; 12]),
    ) {
        Ok(_) => panic!("missing validation bias was accepted"),
        Err(error) => error,
    };
    assert!(error
        .to_string()
        .contains("validation operands do not match"));
}

#[test]
fn cpu_fp32_reference_covers_every_l3_semantic() {
    let ctx = CudaContext::new(0).unwrap();
    let raw_a = vec![1.0; 6];
    let raw_b = vec![1.0; 12];
    let raw_bias = vec![0.25, -0.25, 0.5, -0.5];
    let a = tensor(0, vec![2, 3], &raw_a);
    let b = tensor(0, vec![3, 4], &raw_b);
    let bias = tensor(0, vec![4], &raw_bias);

    let mut gemm_out = tensor(0, vec![2, 4], &[0.0; 8]);
    let mut args = GemmArgs::new(&a, &b, &mut gemm_out);
    args.policy.allow_fallback = false;
    args.policy.graph_safe = false;
    let prepared = prepare_with_original_fp32(
        &ctx,
        args,
        super::contracts::Semantic::Gemm,
        super::execution::PlanApi::gemm(),
        None,
        super::contracts::ValidationReference::new(&raw_a, &raw_b),
    )
    .unwrap();
    assert!(prepared.summary().contains("reference=original-fp32"));
    assert!(prepared.summary().contains("max_element="));
    assert!(prepared.summary().contains("rel_l2="));
    assert!(prepared.summary().contains("cosine="));

    let mut bias_out = tensor(0, vec![2, 4], &[0.0; 8]);
    let mut args = GemmArgs::new(&a, &b, &mut bias_out);
    args.policy.allow_fallback = false;
    args.policy.graph_safe = false;
    let prepared = prepare_with_original_fp32(
        &ctx,
        args,
        super::contracts::Semantic::GemmBias,
        super::execution::PlanApi::gemm_bias(),
        Some(&bias),
        super::contracts::ValidationReference::with_bias(&raw_a, &raw_b, &raw_bias),
    )
    .unwrap();
    assert!(prepared.summary().contains("reference=original-fp32"));

    let mut gelu_out = tensor(0, vec![2, 4], &[0.0; 8]);
    let mut args = GemmArgs::new(&a, &b, &mut gelu_out);
    args.policy.allow_fallback = false;
    args.policy.graph_safe = false;
    let prepared = prepare_with_original_fp32(
        &ctx,
        args,
        super::contracts::Semantic::GemmBiasGelu,
        super::execution::PlanApi::gemm_bias_gelu(),
        Some(&bias),
        super::contracts::ValidationReference::with_bias(&raw_a, &raw_b, &raw_bias),
    )
    .unwrap();
    assert!(prepared.summary().contains("reference=original-fp32"));

    let mut geglu_out = tensor(0, vec![2, 2], &[0.0; 4]);
    let mut args = GemmArgs::new(&a, &b, &mut geglu_out);
    args.policy.allow_fallback = false;
    args.policy.graph_safe = false;
    let prepared = prepare_with_original_fp32(
        &ctx,
        args,
        super::contracts::Semantic::GemmGeglu,
        super::execution::PlanApi::gemm_geglu(),
        None,
        super::contracts::ValidationReference::new(&raw_a, &raw_b),
    )
    .unwrap();
    assert!(prepared.summary().contains("reference=original-fp32"));
}

#[test]
fn deterministic_bf16_reference_corpus_validates_all_l3_candidate_domains() {
    run_deterministic_reference_corpus(false);
}

#[test]
fn deterministic_fp8_reference_corpus_quantizes_and_validates_all_l3_candidate_domains() {
    run_deterministic_reference_corpus(true);
}

#[test]
fn fp8_reference_distinguishes_full_quantization_error() {
    let (m, k, n) = (2, 16, 8);
    let fp8_bytes_a = vec![0x38; m * k]; // E4M3 1.0
    let fp8_bytes_b = vec![0x38; k * n];

    let ctx = CudaContext::new(0).unwrap();
    let a = bytes_tensor(0, vec![m, k], DType::F8E4M3, &fp8_bytes_a);
    let b = bytes_tensor(0, vec![k, n], DType::F8E4M3, &fp8_bytes_b);
    let mut out = zeros_tensor(0, vec![m, n], DType::BF16);
    let mut args = GemmArgs::new(&a, &b, &mut out);
    args.quantization = GemmQuantization::Fp8UnitScale;
    let original_a = vec![1.2; m * k];
    let original_b = vec![1.0; k * n];
    args.policy.allow_fallback = false;
    args.policy.graph_safe = false;
    let error = match prepare_with_original_fp32(
        &ctx,
        args,
        super::contracts::Semantic::Gemm,
        super::execution::PlanApi::gemm(),
        None,
        super::contracts::ValidationReference::new(&original_a, &original_b),
    ) {
        Ok(_) => panic!("quantization loss unexpectedly passed the fixed thresholds"),
        Err(error) => error,
    };
    assert!(error.to_string().contains("fixed FP32 reference contract"));

    let ctx = CudaContext::new(0).unwrap();
    let a = bytes_tensor(0, vec![m, k], DType::F8E4M3, &fp8_bytes_a);
    let b = bytes_tensor(0, vec![k, n], DType::F8E4M3, &fp8_bytes_b);
    let mut out = zeros_tensor(0, vec![m, n], DType::BF16);
    let mut args = GemmArgs::new(&a, &b, &mut out);
    args.quantization = GemmQuantization::Fp8UnitScale;
    args.policy.allow_fallback = false;
    args.policy.graph_safe = false;
    let prepared = prepare_gemm(&ctx, args).unwrap();
    assert!(prepared.summary().contains("reference=dequantized-input"));
}

#[test]
fn bf16_and_fp8_share_the_original_fp32_reference_contract() {
    let (m, k, n) = (2, 16, 8);
    let raw_a = vec![1.0; m * k];
    let raw_b = vec![1.0; k * n];

    let bf16_ctx = CudaContext::new(0).unwrap();
    let bf16_a = tensor(0, vec![m, k], &raw_a);
    let bf16_b = tensor(0, vec![k, n], &raw_b);
    let mut bf16_out = zeros_tensor(0, vec![m, n], DType::BF16);
    let mut bf16_args = GemmArgs::new(&bf16_a, &bf16_b, &mut bf16_out);
    bf16_args.policy.allow_fallback = false;
    bf16_args.policy.graph_safe = false;
    let mut bf16 = prepare_with_original_fp32(
        &bf16_ctx,
        bf16_args,
        super::contracts::Semantic::Gemm,
        super::execution::PlanApi::gemm(),
        None,
        super::contracts::ValidationReference::new(&raw_a, &raw_b),
    )
    .unwrap();
    assert!(bf16.summary().contains("reference=original-fp32"));
    bf16.enqueue().unwrap();
    bf16_ctx.synchronize().unwrap();

    let fp8_ctx = CudaContext::new(0).unwrap();
    let fp8_a = bytes_tensor(0, vec![m, k], DType::F8E4M3, &vec![0x38; m * k]);
    let fp8_b = bytes_tensor(0, vec![k, n], DType::F8E4M3, &vec![0x38; k * n]);
    let mut fp8_out = zeros_tensor(0, vec![m, n], DType::BF16);
    let mut fp8_args = GemmArgs::new(&fp8_a, &fp8_b, &mut fp8_out);
    fp8_args.quantization = GemmQuantization::Fp8UnitScale;
    fp8_args.policy.allow_fallback = false;
    fp8_args.policy.graph_safe = false;
    let mut fp8 = prepare_with_original_fp32(
        &fp8_ctx,
        fp8_args,
        super::contracts::Semantic::Gemm,
        super::execution::PlanApi::gemm(),
        None,
        super::contracts::ValidationReference::new(&raw_a, &raw_b),
    )
    .unwrap();
    assert!(fp8.summary().contains("reference=original-fp32"));
    fp8.enqueue().unwrap();
    fp8_ctx.synchronize().unwrap();

    assert_eq!(values(&bf16_out), values(&fp8_out));
    assert!(values(&bf16_out).iter().all(|&value| value == k as f32));
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
fn prepared_execution_reuses_one_native_instance_across_enqueues() {
    let ctx = CudaContext::new(0).unwrap();
    let a = tensor(0, vec![2, 3], &[1.0; 6]);
    let b = tensor(0, vec![3, 4], &[1.0; 12]);
    let mut out = tensor(0, vec![2, 4], &[0.0; 8]);
    let mut args = GemmArgs::new(&a, &b, &mut out);
    args.policy.online_tune = false;

    super::execution::reset_prepared_execution_create_count();
    let mut prepared = prepare_gemm(&ctx, args).unwrap();
    assert_eq!(super::execution::prepared_execution_create_count(), 1);

    prepared.enqueue().unwrap();
    prepared.enqueue().unwrap();
    ctx.synchronize().unwrap();

    // Native instance creation includes provider resource creation and warmup.
    // Repeated hot-path submissions must not run that preparation again.
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

        let mut prepared = prepare_gemm(&ctx, args).unwrap();
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
            prepare_gemm_geglu(&ctx, GemmGegluArgs { gemm: args })?
        };

        let same = {
            let mut args =
                GemmArgs::new(&a, &b, &mut out).with_immutable_weight(WeightVersion::new(7));
            args.quantization = GemmQuantization::Fp8UnitScale;
            args.policy.online_tune = false;
            args.policy.allow_fallback = false;
            args.policy.cache_dir = Some(cache_dir_string.clone());
            prepare_gemm_geglu(&ctx, GemmGegluArgs { gemm: args })?
        };
        let changed = {
            let mut args =
                GemmArgs::new(&a, &b, &mut out).with_immutable_weight(WeightVersion::new(8));
            args.quantization = GemmQuantization::Fp8UnitScale;
            args.policy.online_tune = false;
            args.policy.allow_fallback = false;
            args.policy.cache_dir = Some(cache_dir_string.clone());
            prepare_gemm_geglu(&ctx, GemmGegluArgs { gemm: args })?
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
    let mut mutable = prepare_gemm_geglu(&ctx, GemmGegluArgs { gemm: mutable_args }).unwrap();
    assert_eq!(mutable.weight_prepack_count(), 1); // instance warmup
    mutable.enqueue().unwrap();
    ctx.synchronize().unwrap();
    assert_eq!(mutable.weight_prepack_count(), 2);

    drop(mutable);
    std::fs::remove_dir_all(cache_dir).unwrap();
}
