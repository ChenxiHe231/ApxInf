//! Black-box Quantization L3 tests whose expectations are derived from the
//! public numeric contract, independently of the CUDA launch implementation.

use super::framework::{bytes_tensor, f16_tensor, tensor, zeros_tensor};
use super::*;
use crate::transfers::to_cpu;
use crate::CudaContext;
use apxinf_core::DType;
use half::{bf16, f16};

fn cpu_bytes(tensor: &apxinf_core::Tensor) -> Vec<u8> {
    to_cpu(tensor).unwrap().storage().as_cpu().unwrap().to_vec()
}

fn cpu_f32(tensor: &apxinf_core::Tensor) -> Vec<f32> {
    to_cpu(tensor).unwrap().as_f32().unwrap().to_vec()
}

fn cpu_bf16_bits(tensor: &apxinf_core::Tensor) -> Vec<u16> {
    to_cpu(tensor)
        .unwrap()
        .as_bf16()
        .unwrap()
        .iter()
        .map(|value| value.to_bits())
        .collect()
}

#[test]
fn fixed_e4m3_matches_standard_encoding_for_both_input_types() {
    let ctx = CudaContext::new(0).unwrap();
    let values = [0.0, 1.0, -2.0, 4.0];
    // With scale=2 the E4M3-domain values are [0, .5, -1, 2].
    let expected = [0x00, 0x30, 0xb8, 0x40];

    for dtype in [DType::F16, DType::BF16] {
        let input = if dtype == DType::F16 {
            f16_tensor(0, vec![2, 2], &values)
        } else {
            tensor(0, vec![2, 2], &values)
        };
        let mut output = zeros_tensor(0, vec![2, 2], DType::F8E4M3);
        let mut args =
            QuantizationArgs::new(QuantizationSemantic::FixedScaleE4m3, &input, &mut output);
        args.scale = 2.0;
        quantization(&ctx, args).unwrap();
        assert_eq!(cpu_bytes(&output), expected);
    }
}

#[test]
fn rowwise_e4m3_computes_scales_and_zero_fills_padding() {
    let ctx = CudaContext::new(0).unwrap();
    let input = tensor(0, vec![2, 3], &[0.0, 1.0, -2.0, 4.0, -1.0, 2.0]);
    let mut output = zeros_tensor(0, vec![2, 5], DType::F8E4M3);
    let mut scales = zeros_tensor(0, vec![2], DType::F32);
    let mut args = QuantizationArgs::new(QuantizationSemantic::RowwiseE4m3, &input, &mut output);
    args.scales = Some(&mut scales);
    quantization(&ctx, args).unwrap();

    let scales = cpu_f32(&scales);
    assert!((scales[0] - 2.0 / 448.0).abs() <= 1.0e-7);
    assert!((scales[1] - 4.0 / 448.0).abs() <= 1.0e-7);
    let bytes = cpu_bytes(&output);
    assert_eq!(&bytes[3..5], &[0, 0]);
    assert_eq!(&bytes[8..10], &[0, 0]);
    // Each row's maximum maps to the finite E4M3 value 448 (0x7e), with sign.
    assert_eq!(bytes[2], 0xfe);
    assert_eq!(bytes[5], 0x7e);
}

#[test]
fn cast_and_slice_match_independent_bf16_rounding_and_indexing() {
    let ctx = CudaContext::new(0).unwrap();
    let source_values = [0.1, -0.25, 1.5, 1000.0, -7.75, 3.0];
    let input = f16_tensor(0, vec![2, 3], &source_values);
    let mut cast = zeros_tensor(0, vec![2, 3], DType::BF16);
    quantization(
        &ctx,
        QuantizationArgs::new(QuantizationSemantic::CastF16ToBf16, &input, &mut cast),
    )
    .unwrap();
    let expected: Vec<_> = source_values
        .iter()
        .map(|&value| bf16::from_f32(f16::from_f32(value).to_f32()).to_bits())
        .collect();
    assert_eq!(cpu_bf16_bits(&cast), expected);

    let mut sliced = zeros_tensor(0, vec![2, 2], DType::BF16);
    quantization(
        &ctx,
        QuantizationArgs::new(QuantizationSemantic::SliceColumnsBf16, &cast, &mut sliced),
    )
    .unwrap();
    assert_eq!(
        cpu_bf16_bits(&sliced),
        vec![expected[0], expected[1], expected[3], expected[4]]
    );
}

#[test]
fn rowwise_i8_matches_cpu_scale_and_rounding_reference() {
    let ctx = CudaContext::new(0).unwrap();
    let input = tensor(0, vec![2, 4], &[0.0, 1.0, -2.0, 4.0, 0.0, 0.0, 0.0, 0.0]);
    let mut output = bytes_tensor(0, vec![2, 4], DType::I8, &[0; 8]);
    let mut scales = zeros_tensor(0, vec![2], DType::F32);
    let mut args = QuantizationArgs::new(QuantizationSemantic::RowwiseI8, &input, &mut output);
    args.scales = Some(&mut scales);
    quantization(&ctx, args).unwrap();

    assert_eq!(cpu_bytes(&output), &[0, 32, 192, 127, 0, 0, 0, 0]);
    let scales = cpu_f32(&scales);
    assert!((scales[0] - 4.0 / 127.0).abs() <= 1.0e-7);
    assert_eq!(scales[1], 1.0e-12);
}

#[test]
fn semantic_contract_rejects_wrong_auxiliaries_and_shapes() {
    let ctx = CudaContext::new(0).unwrap();
    let input = tensor(0, vec![2, 3], &[1.0; 6]);
    let mut fp8 = zeros_tensor(0, vec![2, 3], DType::F8E4M3);
    let mut scales = zeros_tensor(0, vec![2], DType::F32);

    let mut fixed = QuantizationArgs::new(QuantizationSemantic::FixedScaleE4m3, &input, &mut fp8);
    fixed.scales = Some(&mut scales);
    assert!(quantization(&ctx, fixed).is_err());

    let mut too_narrow = zeros_tensor(0, vec![2, 2], DType::F8E4M3);
    let mut rowwise =
        QuantizationArgs::new(QuantizationSemantic::RowwiseE4m3, &input, &mut too_narrow);
    rowwise.scales = Some(&mut scales);
    assert!(quantization(&ctx, rowwise).is_err());
}

#[test]
fn rowwise_i8_prepares_and_replays_without_native_work_during_capture() {
    let ctx = CudaContext::new(0).unwrap();
    let input = tensor(0, vec![1, 4], &[0.0, 1.0, -2.0, 4.0]);
    let mut output = bytes_tensor(0, vec![1, 4], DType::I8, &[0; 4]);
    let mut scales = zeros_tensor(0, vec![1], DType::F32);
    let session = ExecutionSession::with_capacity(256, 0).unwrap();

    prepare_with_session(&session, || {
        let mut args = QuantizationArgs::new(QuantizationSemantic::RowwiseI8, &input, &mut output);
        args.scales = Some(&mut scales);
        quantization(&ctx, args)
    })
    .unwrap();
    let expected = cpu_bytes(&output);

    let graph = crate::capture(&ctx, || {
        with_session(&session, || {
            let mut args =
                QuantizationArgs::new(QuantizationSemantic::RowwiseI8, &input, &mut output);
            args.scales = Some(&mut scales);
            quantization(&ctx, args)
        })
    })
    .unwrap();
    crate::CudaBuffer::from_tensor(&output)
        .unwrap()
        .copy_from_host(&[0xaa; 4])
        .unwrap();
    graph.replay().unwrap();
    ctx.synchronize().unwrap();
    assert_eq!(cpu_bytes(&output), expected);
}
