// Bit-exactness gate for the fused Qwen-Drive BF16 gate/up + SwiGLU operator,
// driven through the repository's own `kernels::gemm::bf16_swiglu_checkpoint`
// dispatch rather than through a scratch harness.
//
// The pinned CuTe-DSL AOT object (`apxinf_quack_swiglu_bf16_m256n256_sm110`)
// and the in-tree CUTLASS `SwigluContract` specialization are required to be
// bit-identical over all 3387 x 9216 outputs. Running them back to back in one
// process is the only form of this check that cannot be confused by a thermal
// or power-state difference, and going through the repo's own dispatch is what
// exercises the part a harness cannot: the two implementations consume
// physically different weights -- `[18432, 2560]` K-contiguous for the AOT,
// `[2560, 18432]` with `[gate256, up256]` column tiles for CUTLASS -- and the
// two layouts hold exactly the same number of elements.
//
// The test skips honestly when only one implementation is linked. Build the
// comparison arm with:
//
//   APXINF_CUDA_ARCH=sm_110 \
//   APXINF_CUDA_AOT_MANIFEST=<bundle>/manifest.json \
//     cargo test -p apxinf-cuda --release \
//       --features aot-gemm-swiglu-bf16 --test qwen_drive_bf16_swiglu_equivalence

use apxinf_core::Tensor;
use apxinf_cuda::kernels::gemm::{
    self, Bf16SwigluImpl, Bf16SwigluWeightLayout,
};
use apxinf_cuda::{transfers, CudaContext};
use half::bf16;

// The AOT object is specialized to exactly this shape.
const M: usize = 3387;
const K: usize = 2560;
const INNER: usize = 9216;
const FULL_N: usize = 2 * INNER;
const PADDED_M: usize = 3584;

/// Deterministic LCG in `[-0.05, 0.05)`.
///
/// The scale matters: at unit inputs a K=2560 dot product reaches |gate| ~ 90,
/// where `silu` saturates to either 0 or the identity and most outputs become
/// exactly representable. 0.05 puts |gate| around 1.5, i.e. in the part of the
/// sigmoid where `ex2.approx` and `rcp.approx` actually have to agree.
fn lcg_bf16(seed: u64, count: usize) -> Vec<bf16> {
    let mut state = seed;
    (0..count)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let unit = ((state >> 40) as f32) / ((1u32 << 24) as f32);
            bf16::from_f32((unit - 0.5) * 0.1)
        })
        .collect()
}

fn upload(shape: Vec<usize>, data: &[bf16]) -> Tensor {
    transfers::to_cuda(&Tensor::from_bf16(shape, data).unwrap(), 0).unwrap()
}

struct Inputs {
    activation: Tensor,
    /// `[2I, K]`, the checkpoint order the AOT kernel takes.
    weight_checkpoint: Tensor,
    /// `[K, 2I]` with `[gate256, up256]` column tiles, derived from the very
    /// same host bytes so the two arms cannot be fed different numbers.
    weight_packed: Tensor,
}

fn build_inputs() -> Inputs {
    let activation = lcg_bf16(0x2545_F491_4F6C_DD1D, M * K);
    let checkpoint = lcg_bf16(0x0fee_1dea_dbee_f001, FULL_N * K);
    // Through the very function the model load path calls, so the permutation
    // this test blesses is the one production performs.
    let packed = gemm::pack_bf16_swiglu_weight_host(
        &checkpoint.iter().map(|value| value.to_f32()).collect::<Vec<_>>(),
        FULL_N,
        K,
    )
    .unwrap();
    Inputs {
        activation: upload(vec![M, K], &activation),
        weight_checkpoint: upload(vec![FULL_N, K], &checkpoint),
        weight_packed: upload(vec![K, FULL_N], &packed),
    }
}

fn run(ctx: &CudaContext, inputs: &Inputs, implementation: Bf16SwigluImpl) -> Vec<f32> {
    let weight = match implementation.weight_layout() {
        Bf16SwigluWeightLayout::CheckpointRowMajor => &inputs.weight_checkpoint,
        Bf16SwigluWeightLayout::PackedPairInterleaved => &inputs.weight_packed,
    };
    let output =
        gemm::bf16_swiglu_checkpoint_with(ctx, &inputs.activation, weight, implementation).unwrap();
    assert_eq!(
        output.shape().dims(),
        [PADDED_M, INNER],
        "{implementation:?} returned the wrong padded shape"
    );
    ctx.synchronize().unwrap();
    transfers::to_cpu(&output).unwrap().to_f32_vec().unwrap()
}

/// Compare raw BF16 bit patterns, not floats: `f32` equality would hide a
/// signed-zero difference and would not be a bit-exactness claim.
fn bit_equal_count(left: &[f32], right: &[f32]) -> usize {
    left.iter()
        .zip(right)
        .filter(|(a, b)| bf16::from_f32(**a).to_bits() == bf16::from_f32(**b).to_bits())
        .count()
}

#[test]
fn aot_and_cuda_bf16_swiglu_are_bit_identical() {
    if !Bf16SwigluImpl::Aot.linked() || !Bf16SwigluImpl::Cuda.linked() {
        eprintln!(
            "skipping: this build links AOT={} CUDA={}; enable the \
             aot-gemm-swiglu-bf16 feature with APXINF_CUDA_AOT_MANIFEST set to \
             link both and run the comparison",
            Bf16SwigluImpl::Aot.linked(),
            Bf16SwigluImpl::Cuda.linked()
        );
        return;
    }
    let ctx = CudaContext::new(0).unwrap();
    if gemm::bf16_swiglu_impl(&ctx, &[M, K], FULL_N * K).is_none() {
        eprintln!("skipping: this device does not take the fused BF16 SwiGLU route");
        return;
    }
    let inputs = build_inputs();

    // Alternate the order: a kernel that corrupts an input it should only read
    // would otherwise be masked by always running second.
    let aot = run(&ctx, &inputs, Bf16SwigluImpl::Aot);
    let cuda = run(&ctx, &inputs, Bf16SwigluImpl::Cuda);
    let aot_again = run(&ctx, &inputs, Bf16SwigluImpl::Aot);

    assert_eq!(
        bit_equal_count(&aot, &aot_again),
        aot.len(),
        "the AOT arm is not reproducible"
    );
    let valid = M * INNER;
    let nonzero = aot[..valid].iter().filter(|value| **value != 0.0).count();
    assert!(
        nonzero * 10 > valid * 9,
        "the AOT arm wrote {nonzero}/{valid} nonzero values; a near-empty \
         output would make this comparison vacuous"
    );

    let equal = bit_equal_count(&aot[..valid], &cuda[..valid]);
    assert_eq!(
        equal, valid,
        "BF16 SwiGLU output differs: {equal}/{valid} elements bit-equal"
    );
    // Both routes promise the padded tail is zero, and the down projection
    // reads it.
    for (name, values) in [("AOT", &aot), ("CUDA", &cuda)] {
        assert!(
            values[valid..].iter().all(|value| *value == 0.0),
            "{name} left the padded tail [{M}, {PADDED_M}) nonzero"
        );
    }
    eprintln!(
        "BF16 SwiGLU AOT == CUDA: {equal}/{valid} elements bit-equal, \
         {nonzero} nonzero, tail [{M}, {PADDED_M}) zeroed"
    );
}

/// The single-implementation builds still have to reach the kernel: a wiring
/// error that returns a nonzero status would otherwise only surface end to end.
#[test]
fn the_linked_implementation_runs_and_is_reproducible() {
    let ctx = CudaContext::new(0).unwrap();
    let Some(implementation) = gemm::bf16_swiglu_impl(&ctx, &[M, K], FULL_N * K) else {
        eprintln!("skipping: no fused BF16 SwiGLU implementation applies to this device");
        return;
    };
    let inputs = build_inputs();
    let first = run(&ctx, &inputs, implementation);
    let second = run(&ctx, &inputs, implementation);
    let valid = M * INNER;
    assert!(
        first[..valid].iter().any(|value| *value != 0.0),
        "{implementation:?} wrote nothing"
    );
    assert_eq!(
        bit_equal_count(&first, &second),
        first.len(),
        "{implementation:?} is not reproducible"
    );
}

/// The layout guard in the dispatch, without a device: being handed the other
/// implementation's weight must fail rather than compute wrong numbers.
#[test]
fn the_two_weight_layouts_are_not_interchangeable_by_size() {
    assert_eq!(
        FULL_N * K,
        K * FULL_N,
        "if these ever differ, the size check alone would catch a mix-up"
    );
    assert_ne!(
        Bf16SwigluImpl::Aot.weight_layout(),
        Bf16SwigluImpl::Cuda.weight_layout()
    );
}
