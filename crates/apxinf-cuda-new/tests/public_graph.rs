use apxinf_core::{DType, Shape, Tensor};
use apxinf_cuda::{
    capture,
    ops::{prepare_gemm, GemmArgs},
    CudaBuffer, CudaContext,
};
use half::bf16;

fn bf16_tensor(device: usize, shape: Vec<usize>, values: &[f32]) -> Tensor {
    let words: Vec<u16> = values
        .iter()
        .map(|&value| bf16::from_f32(value).to_bits())
        .collect();
    let bytes: Vec<u8> = words.iter().flat_map(|word| word.to_ne_bytes()).collect();
    let buffer = CudaBuffer::alloc(bytes.len(), device).unwrap();
    buffer.copy_from_host(&bytes).unwrap();
    buffer.as_tensor(Shape::new(shape), DType::BF16).unwrap()
}

fn values(tensor: &Tensor) -> Vec<f32> {
    let buffer = CudaBuffer::from_tensor(tensor).unwrap();
    let mut bytes = vec![0; buffer.len()];
    buffer.copy_to_host(&mut bytes).unwrap();
    bytes
        .chunks_exact(2)
        .map(|bytes| bf16::from_bits(u16::from_ne_bytes([bytes[0], bytes[1]])).to_f32())
        .collect()
}

#[test]
fn public_prepared_execution_can_be_captured_and_replayed() {
    let ctx = CudaContext::new(0).unwrap();
    let a = bf16_tensor(0, vec![2, 3], &[1.0; 6]);
    let b = bf16_tensor(0, vec![3, 4], &[1.0; 12]);
    let mut out = bf16_tensor(0, vec![2, 4], &[0.0; 8]);
    let observed = CudaBuffer::from_tensor(&out).unwrap();

    let mut args = GemmArgs::new(&a, &b, &mut out);
    args.policy.online_tune = false;
    let mut prepared = prepare_gemm(&ctx, args).unwrap();
    let graph = capture(&ctx, || prepared.enqueue()).unwrap();

    let sentinel: Vec<u8> = (0..8)
        .flat_map(|_| bf16::from_f32(-123.0).to_bits().to_ne_bytes())
        .collect();
    observed.copy_from_host(&sentinel).unwrap();
    graph.replay().unwrap();
    ctx.synchronize().unwrap();

    assert!(values(&out).iter().all(|&value| value == 3.0));
}
