//! A model-independent smoke gate for the compiled GEMM provider registry.
use apxinf_core::{DType, Shape, Tensor};
use apxinf_cuda_new::{ops, transfers, CudaBuffer, CudaContext};
use half::bf16;

#[test]
fn plain_gemm_runs_with_a_unique_portable_baseline() {
    let ctx = CudaContext::new_with_autotune(0, false).unwrap();
    let a = transfers::to_cuda_on_stream(&ctx, Tensor::from_bf16(vec![4, 8], &vec![bf16::from_f32(0.5); 32]).unwrap()).unwrap();
    let b = transfers::to_cuda_on_stream(&ctx, Tensor::from_bf16(vec![8, 6], &vec![bf16::from_f32(0.25); 48]).unwrap()).unwrap();
    let mut out = CudaBuffer::alloc_zeros(4 * 6 * 2, 0).unwrap().as_tensor(Shape::new(vec![4, 6]), DType::BF16).unwrap();
    let mut args = ops::GemmArgs::new(&a, &b, &mut out);
    args.policy.online_tune = false;
    args.policy.cache_dir = Some(String::new());
    ops::gemm(&ctx, args).unwrap();
    ctx.synchronize().unwrap();
    let out = transfers::to_cpu(&out).unwrap();
    assert!(out.as_bf16().unwrap().iter().all(|x| x.to_f32() == 1.0));
}
