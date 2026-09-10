use super::*;
use crate::{CudaBuffer, CudaContext};
use apxinf_core::{DType, Shape, Tensor};
use half::bf16;

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
        assert!(passed, "GPU backend has no passing tactic: {backend}; {summary}");
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
    let instance = super::execution::prepare(&ctx, normalized).unwrap();
    let mut graph = instance.capture().unwrap();

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
