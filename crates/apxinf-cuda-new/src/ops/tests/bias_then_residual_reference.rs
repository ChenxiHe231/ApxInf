//! Independent checks for the BF16 intermediate-rounding contract.
//!
//! The distinguishing input below sits exactly on two BF16 rounding
//! boundaries. A single fused `projection + bias + residual` round produces
//! 1.0078125, while the required two-stage contract produces exactly 1.0.

use super::framework::{f16_tensor, tensor};
use super::*;
use crate::{CudaBuffer, CudaContext};
use apxinf_core::{DType, Tensor};
use half::bf16;

fn bf16_bits(tensor: &Tensor) -> Vec<u16> {
    // Direct pointwise operators enqueue on the context stream.  Synchronize
    // only at this host-observation boundary; production launches remain
    // asynchronous and graph-capturable.
    unsafe {
        crate::ffi::check_cuda(crate::ffi::cudaDeviceSynchronize()).unwrap();
    }
    let buffer = CudaBuffer::from_tensor(tensor).unwrap();
    let mut bytes = vec![0; buffer.len()];
    buffer.copy_to_host(&mut bytes).unwrap();
    bytes
        .chunks_exact(2)
        .map(|chunk| u16::from_ne_bytes([chunk[0], chunk[1]]))
        .collect()
}

fn two_stage_reference(projection: f32, bias: f32, residual: f32) -> bf16 {
    let projection = bf16::from_f32(projection);
    let bias = bf16::from_f32(bias);
    let residual = bf16::from_f32(residual);
    let biased = bf16::from_f32(projection.to_f32() + bias.to_f32());
    bf16::from_f32(biased.to_f32() + residual.to_f32())
}

#[test]
fn bias_then_residual_preserves_the_bf16_intermediate_rounding() {
    let ctx = CudaContext::new(0).unwrap();
    let half_ulp = 0.00390625;
    let projection = tensor(0, vec![1, 1], &[1.0]);
    let bias = tensor(0, vec![1], &[half_ulp]);
    let residual = tensor(0, vec![1, 1], &[half_ulp]);

    let mut two_stage = tensor(0, vec![1, 1], &[0.0]);
    bias_then_residual(
        &ctx,
        BiasThenResidualArgs::new(&projection, Some(&bias), &residual, &mut two_stage),
    )
    .unwrap();

    let expected = two_stage_reference(1.0, half_ulp, half_ulp);
    assert_eq!(bf16_bits(&two_stage), vec![expected.to_bits()]);
    assert_eq!(expected, bf16::from_f32(1.0));

    // Demonstrate that this is not merely an alias for the ordinary
    // bias-residual semantic, which rounds only after summing all operands.
    let mut one_stage = tensor(0, vec![1, 1], &[0.0]);
    bias_residual(
        &ctx,
        BiasResidualArgs::new(&projection, Some(&bias), &residual, &mut one_stage),
    )
    .unwrap();
    assert_eq!(
        bf16_bits(&one_stage),
        vec![bf16::from_f32(1.0078125).to_bits()]
    );
    assert_ne!(bf16_bits(&two_stage), bf16_bits(&one_stage));
}

#[test]
fn bias_then_residual_rejects_f16_instead_of_changing_its_contract() {
    let ctx = CudaContext::new(0).unwrap();
    let projection = f16_tensor(0, vec![1, 1], &[1.0]);
    let residual = f16_tensor(0, vec![1, 1], &[0.5]);
    let mut output = f16_tensor(0, vec![1, 1], &[0.0]);
    let error = bias_then_residual(
        &ctx,
        BiasThenResidualArgs::new(&projection, None, &residual, &mut output),
    )
    .expect_err("F16 must not enter a BF16-only semantic");
    assert!(
        error.to_string().contains("BF16-only semantic"),
        "unexpected error: {error}"
    );
    assert_eq!(output.dtype(), DType::F16);
}
