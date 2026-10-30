//! TensorRT-Edge-LLM BLACKWELL_N128_K64_V1 synthetic parity.

use apxinf_core::{DType, Shape, Tensor};
use apxinf_cuda_new::{encode_e4m3, ops, transfers, CudaBuffer, CudaContext};
use half::bf16;

fn upload(ctx: &CudaContext, tensor: &Tensor) -> Tensor {
    transfers::to_cuda(tensor, ctx.device_id()).unwrap()
}

#[test]
fn constant_weight_matches_bf16_reference_for_every_supported_row_count() {
    let ctx = CudaContext::new(0).unwrap();
    const N: usize = 128;
    const K: usize = 64;

    // E2M1 code 1 is +0.5. Both nibbles therefore use 0x1.
    let qweight = upload(
        &ctx,
        &Tensor::from_e2m1_pairs(vec![1, 1, 128, 32], &vec![0x11; N * K / 2]).unwrap(),
    );
    let block_scales = upload(
        &ctx,
        &Tensor::from_f8_e4m3(vec![1, 1, 128, 4], &vec![encode_e4m3(1.0); N * K / 16]).unwrap(),
    );
    let global_scale = upload(&ctx, &Tensor::from_f32(vec![1], &[1.0]).unwrap());
    for rows in [1, 2, 3, 4, 8, 16] {
        let activation = upload(
            &ctx,
            &Tensor::from_bf16(vec![rows, K], &vec![bf16::from_f32(1.0); rows * K]).unwrap(),
        );
        let output =
            CudaBuffer::alloc_zeros(rows * N * DType::BF16.size_in_bytes(), ctx.device_id())
                .unwrap()
                .as_tensor(Shape::new(vec![rows, N]), DType::BF16)
                .unwrap();

        ops::nvfp4_a16_blackwell_gemv(
            &ctx,
            &activation,
            &qweight,
            &block_scales,
            &global_scale,
            &output,
        )
        .unwrap();
        ctx.synchronize().unwrap();
        let actual = transfers::to_cpu(&output).unwrap();
        for value in actual.as_bf16().unwrap() {
            assert_eq!(value.to_f32(), 32.0);
        }
    }
}

#[test]
fn split_swiglu_matches_row_interleaved_fused_layout() {
    let ctx = CudaContext::new(0).unwrap();
    let gate_values = [0.5, -1.0, 2.0, 3.0, -0.25, 1.5];
    let up_values = [2.0, 0.5, -1.0, 0.25, 4.0, -2.0];
    let gate = upload(
        &ctx,
        &Tensor::from_bf16(vec![2, 3], &gate_values.map(bf16::from_f32)).unwrap(),
    );
    let up = upload(
        &ctx,
        &Tensor::from_bf16(vec![2, 3], &up_values.map(bf16::from_f32)).unwrap(),
    );
    let fused_values = [
        gate_values[0],
        gate_values[1],
        gate_values[2],
        up_values[0],
        up_values[1],
        up_values[2],
        gate_values[3],
        gate_values[4],
        gate_values[5],
        up_values[3],
        up_values[4],
        up_values[5],
    ];
    let fused = upload(
        &ctx,
        &Tensor::from_bf16(vec![2, 6], &fused_values.map(bf16::from_f32)).unwrap(),
    );
    let split_output = CudaBuffer::alloc_zeros(2 * 3 * DType::BF16.size_in_bytes(), 0)
        .unwrap()
        .as_tensor(Shape::new(vec![2, 3]), DType::BF16)
        .unwrap();
    let fused_output = CudaBuffer::alloc_zeros(2 * 3 * DType::BF16.size_in_bytes(), 0)
        .unwrap()
        .as_tensor(Shape::new(vec![2, 3]), DType::BF16)
        .unwrap();

    ops::swiglu_split(&ctx, &gate, &up, &split_output).unwrap();
    ops::swiglu(&ctx, &fused, &fused_output).unwrap();
    ctx.synchronize().unwrap();

    let split = transfers::to_cpu(&split_output).unwrap();
    let fused = transfers::to_cpu(&fused_output).unwrap();
    assert_eq!(split.as_bf16().unwrap(), fused.as_bf16().unwrap());
}
