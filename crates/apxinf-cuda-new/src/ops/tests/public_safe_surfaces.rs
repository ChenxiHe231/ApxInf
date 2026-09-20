//! Black-box coverage for public CUDA-safe utility surfaces.
//!
//! The expected values here come from public model-neutral contracts: exact
//! tensor byte round trips, counter-based RNG keys, standard E4M3 encoding,
//! and the documented prefix-cache layout.

use crate::kernels::quantization::{cast_f16_bf16, quantize_bf16_e4m3, quantize_f16_e4m3};
use crate::sampling::{create_normal_generator, create_token_sampler};
use crate::transfers::{copy_cpu_to_cuda, to_cpu, to_cuda};
use crate::{reserve_prefix, CudaContext};
use apxinf_core::{
    standard_normal_f32, DType, NextTokenLogits, RngKey, Tensor, TokenPenalties, TokenSamplingInit,
    TokenSamplingParams, TokenSamplingSpec, TokenSelection,
};
use half::{bf16, f16};

fn cpu_tensor(dtype: DType, shape: Vec<usize>, values: &[f32]) -> Tensor {
    match dtype {
        DType::F32 => Tensor::from_f32(shape, values).unwrap(),
        DType::F16 => {
            let values: Vec<_> = values.iter().copied().map(f16::from_f32).collect();
            Tensor::from_f16(shape, &values).unwrap()
        }
        DType::BF16 => {
            let values: Vec<_> = values.iter().copied().map(bf16::from_f32).collect();
            Tensor::from_bf16(shape, &values).unwrap()
        }
        other => panic!("unsupported test dtype {other}"),
    }
}

fn cpu_bytes(tensor: &Tensor) -> Vec<u8> {
    tensor.storage().as_cpu().unwrap().to_vec()
}

fn cpu_f32_values(tensor: &Tensor) -> Vec<f32> {
    match tensor.dtype() {
        DType::F32 => tensor.as_f32().unwrap().to_vec(),
        DType::F16 => tensor
            .as_f16()
            .unwrap()
            .iter()
            .map(|value| value.to_f32())
            .collect(),
        DType::BF16 => tensor
            .as_bf16()
            .unwrap()
            .iter()
            .map(|value| value.to_f32())
            .collect(),
        other => panic!("unsupported test dtype {other}"),
    }
}

#[test]
fn transfers_round_trip_two_dimensional_tensors_exactly() {
    let values = [1.0, -2.0, 0.5, 4.0, -0.25, 8.0];
    for dtype in [DType::F32, DType::F16, DType::BF16] {
        let host = cpu_tensor(dtype, vec![2, 3], &values);
        let device = to_cuda(&host, 0).unwrap();
        let round_trip = to_cpu(&device).unwrap();
        assert_eq!(round_trip.shape(), host.shape());
        assert_eq!(round_trip.dtype(), dtype);
        assert_eq!(cpu_bytes(&round_trip), cpu_bytes(&host));
    }
}

#[test]
fn fixed_transfer_preserves_address_and_rejects_shape_or_dtype_mismatch() {
    let initial = cpu_tensor(DType::BF16, vec![2, 3], &[0.0; 6]);
    let destination = to_cuda(&initial, 0).unwrap();
    let address_before = destination.storage().as_gpu().unwrap().ptr;

    let replacement = cpu_tensor(DType::BF16, vec![2, 3], &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    copy_cpu_to_cuda(&replacement, &destination).unwrap();
    assert_eq!(destination.storage().as_gpu().unwrap().ptr, address_before);
    assert_eq!(
        cpu_bytes(&to_cpu(&destination).unwrap()),
        cpu_bytes(&replacement)
    );

    let wrong_shape = cpu_tensor(DType::BF16, vec![3, 2], &[1.0; 6]);
    assert!(copy_cpu_to_cuda(&wrong_shape, &destination).is_err());
    let wrong_dtype = cpu_tensor(DType::F16, vec![2, 3], &[1.0; 6]);
    assert!(copy_cpu_to_cuda(&wrong_dtype, &destination).is_err());
    assert!(to_cuda(&destination, 0).is_err());
    assert!(to_cpu(&replacement).is_err());
}

#[test]
fn reserve_prefix_preserves_rows_and_zeroes_the_tail_for_f16_and_bf16() {
    let ctx = CudaContext::new(0).unwrap();
    let prefix_values = [1.0, -2.0, 3.0, 0.5, -0.25, 4.0];

    for dtype in [DType::F16, DType::BF16] {
        let prefix_host = cpu_tensor(dtype, vec![2, 3], &prefix_values);
        let prefix = to_cuda(&prefix_host, 0).unwrap();
        let reserved = reserve_prefix(&ctx, &prefix, 5).unwrap();
        let reserved = to_cpu(&reserved).unwrap();

        assert_eq!(reserved.shape().dims(), &[5, 3]);
        assert_eq!(reserved.dtype(), dtype);
        let mut expected = prefix_values.to_vec();
        expected.extend([0.0; 9]);
        assert_eq!(cpu_f32_values(&reserved), expected);

        assert!(reserve_prefix(&ctx, &prefix, 1).is_err());
    }
}

#[test]
fn greedy_and_random_token_sampling_obey_public_determinism_contracts() {
    let ctx = CudaContext::new(0).unwrap();
    let spec = TokenSamplingSpec {
        vocab_size: 4,
        max_sequence_len: 8,
    };
    let logits_host = cpu_tensor(
        DType::F32,
        vec![2, 4],
        &[99.0, 0.0, 0.0, 0.0, 1.0, 5.0, 5.0, -1.0],
    );
    let logits = to_cuda(&logits_host, 0).unwrap();

    let greedy = TokenSamplingParams::greedy();
    let mut sampler = create_token_sampler(&ctx, spec).unwrap();
    sampler
        .begin(TokenSamplingInit {
            prompt_token_ids: &[3, 2],
            params: &greedy,
            rng: RngKey::new(7, 11, 13),
        })
        .unwrap();
    let sample = sampler
        .sample(NextTokenLogits::last(&logits, spec.vocab_size).unwrap())
        .unwrap();
    assert_eq!(sample.token_id, 1, "greedy ties choose the lowest token ID");
    assert_eq!(sample.logprob, None);

    let random = TokenSamplingParams {
        selection: TokenSelection::Random {
            temperature: 0.75,
            top_k: None,
            top_p: 1.0,
        },
        penalties: TokenPenalties::default(),
        return_logprob: true,
    };
    let key = RngKey::new(0x1234_5678, 17, 9);
    let sample_once = || {
        let mut sampler = create_token_sampler(&ctx, spec).unwrap();
        sampler
            .begin(TokenSamplingInit {
                prompt_token_ids: &[],
                params: &random,
                rng: key,
            })
            .unwrap();
        sampler
            .sample(NextTokenLogits::last(&logits, spec.vocab_size).unwrap())
            .unwrap()
    };
    assert_eq!(sample_once(), sample_once());
}

#[test]
fn normal_generator_matches_public_counter_based_cpu_reference() {
    let ctx = CudaContext::new(0).unwrap();
    let output = to_cuda(&cpu_tensor(DType::F32, vec![17], &[0.0; 17]), 0).unwrap();
    let mut generator = create_normal_generator(&ctx, output).unwrap();
    let key = RngKey::new(0xdecafbad, 29, 5);

    let first = to_cpu(generator.generate(key).unwrap()).unwrap();
    let second = to_cpu(generator.generate(key).unwrap()).unwrap();
    let expected = standard_normal_f32(17, key);
    assert_eq!(first.as_f32().unwrap(), second.as_f32().unwrap());
    for (index, (&actual, &expected)) in first
        .as_f32()
        .unwrap()
        .iter()
        .zip(expected.iter())
        .enumerate()
    {
        assert!(
            (actual - expected).abs() <= 2.0e-6,
            "normal element {index}: got {actual}, expected {expected}"
        );
    }
}

#[test]
fn compatibility_quantizers_match_standard_e4m3_at_unit_scale() {
    let ctx = CudaContext::new(0).unwrap();
    let values = [0.0, 0.5, 1.0, -1.0, 2.0];
    // Standard finite E4M3 encodings at scale 1.
    let expected = [0x00, 0x30, 0x38, 0xb8, 0x40];

    let f16_input = to_cuda(&cpu_tensor(DType::F16, vec![5], &values), 0).unwrap();
    let f16_output = quantize_f16_e4m3(&ctx, &f16_input, 1.0).unwrap();
    assert_eq!(f16_output.dtype(), DType::F8E4M3);
    assert_eq!(cpu_bytes(&to_cpu(&f16_output).unwrap()), expected);

    let bf16_input = to_cuda(&cpu_tensor(DType::BF16, vec![5], &values), 0).unwrap();
    let bf16_output = quantize_bf16_e4m3(&ctx, &bf16_input, 1.0).unwrap();
    assert_eq!(bf16_output.dtype(), DType::F8E4M3);
    assert_eq!(cpu_bytes(&to_cpu(&bf16_output).unwrap()), expected);

    assert!(quantize_f16_e4m3(&ctx, &f16_input, 0.0).is_err());
    assert!(quantize_bf16_e4m3(&ctx, &bf16_input, f32::NAN).is_err());
}

#[test]
fn compatibility_cast_matches_independent_f16_to_bf16_rounding() {
    let ctx = CudaContext::new(0).unwrap();
    let values = [0.0, 0.1, -0.25, 1.5, 1000.0, -7.75];
    let input_host = cpu_tensor(DType::F16, vec![2, 3], &values);
    let input = to_cuda(&input_host, 0).unwrap();
    let output = cast_f16_bf16(&ctx, &input).unwrap();
    let output = to_cpu(&output).unwrap();
    let expected: Vec<_> = input_host
        .as_f16()
        .unwrap()
        .iter()
        .map(|value| bf16::from_f32(value.to_f32()).to_bits())
        .collect();
    let actual: Vec<_> = output
        .as_bf16()
        .unwrap()
        .iter()
        .map(|value| value.to_bits())
        .collect();
    assert_eq!(actual, expected);
}
