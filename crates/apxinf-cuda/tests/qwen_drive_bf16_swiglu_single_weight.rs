// One resident gate/up weight, both routes.
//
// Report 19 landed the fused BF16 SwiGLU with the packed `[K, 2I]` weight the
// CUTLASS kernel's measured `StrideB = (_1, 18432, 0)` requires, and kept the
// checkpoint's `[2I, K]` copy resident as well, because decode (`seq == 1`)
// does not take the fused route: +94.4 MB per backbone layer, +3.02 GB at the
// shipping 32 layers.
//
// This gate is the evidence that the second copy is unnecessary: the generic
// cuBLAS route reads the *packed* buffer directly, with `transb = N` and
// `ldb = 2I` instead of `transb = T` and `ldb = K`, followed by
// `swiglu_bf16_rounded_pair_interleaved`, which undoes the `[gate256, up256]`
// column order. If that produces the same bits as the old route on the
// checkpoint layout, the checkpoint copy can be dropped outright.
//
//   APXINF_CUDA_ARCH=sm_110 cargo test -p apxinf-cuda --release \
//     --test qwen_drive_bf16_swiglu_single_weight -- --nocapture --test-threads=1

use apxinf_core::Tensor;
use apxinf_cuda::kernels::gemm::{self, Bf16SwigluImpl, Bf16SwigluWeightLayout};
use apxinf_cuda::{transfers, CudaContext};
use half::bf16;

const K: usize = 2560;
const INNER: usize = 9216;
const FULL_N: usize = 2 * INNER;
/// `kPairColumns` of the fused kernel's `[gate, up]` column tiling.
const PAIR: usize = 256;

/// Deterministic LCG in `[-0.05, 0.05)`; same reasoning as the oracle gate's,
/// i.e. keep `|gate|` near 1.5 where SiLU is neither saturated nor identity.
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

fn download(ctx: &CudaContext, tensor: &Tensor) -> Vec<u16> {
    ctx.synchronize().unwrap();
    transfers::to_cpu(tensor)
        .unwrap()
        .to_f32_vec()
        .unwrap()
        .iter()
        .map(|value| bf16::from_f32(*value).to_bits())
        .collect()
}

/// The generic (non-fused) route must compute the same bits from the packed
/// weight as it does from the checkpoint weight, at every `M` that is not the
/// fused kernel's pinned one.
///
/// This is the whole argument for dropping the second copy. Two things could
/// break it and both are checked here rather than argued:
///   * cuBLAS picks a different kernel for `N`-contiguous B than for
///     `K`-contiguous B, and a different accumulation order over K would move
///     the low bits of the projection;
///   * the interleaved activation's column map and the host packer's column
///     map could disagree -- then gate would multiply the wrong up.
#[test]
fn the_generic_route_is_bit_identical_on_both_weight_layouts() {
    let ctx = CudaContext::new(0).unwrap();
    let checkpoint = lcg_bf16(0x0fee_1dea_dbee_f001, FULL_N * K);
    let packed = gemm::pack_bf16_swiglu_weight_host(
        &checkpoint
            .iter()
            .map(|value| value.to_f32())
            .collect::<Vec<_>>(),
        FULL_N,
        K,
    )
    .unwrap();
    let w_checkpoint = upload(vec![FULL_N, K], &checkpoint);
    let w_packed = upload(vec![K, FULL_N], &packed);

    // Not 3387: at the pinned M the fused kernel takes over and its numerics
    // are deliberately different (one rounding, ex2.approx). This test is
    // about the generic route, which is the one decode uses.
    for rows in [1usize, 2, 7, 16, 64, 337] {
        let activation = upload(
            vec![rows, K],
            &lcg_bf16(0x2545_f491_4f6c_dd1d ^ rows as u64, rows * K),
        );
        assert!(
            gemm::bf16_swiglu_impl(&ctx, &[rows, K], FULL_N * K).is_none(),
            "M={rows} must take the generic route for this test to mean anything"
        );
        let from_checkpoint =
            gemm::bf16_swiglu_checkpoint(&ctx, &activation, &w_checkpoint).unwrap();
        assert_eq!(from_checkpoint.shape().dims(), [rows, INNER]);
        let reference = download(&ctx, &from_checkpoint);
        let from_packed = gemm::bf16_swiglu_checkpoint(&ctx, &activation, &w_packed).unwrap();
        assert_eq!(from_packed.shape().dims(), [rows, INNER]);
        let candidate = download(&ctx, &from_packed);

        let nonzero = reference.iter().filter(|bits| **bits != 0).count();
        assert!(
            nonzero * 10 > reference.len() * 9,
            "M={rows}: only {nonzero}/{} outputs are nonzero; the comparison \
             would be vacuous",
            reference.len()
        );
        let equal = reference
            .iter()
            .zip(&candidate)
            .filter(|(a, b)| a == b)
            .count();
        assert_eq!(
            equal,
            reference.len(),
            "M={rows}: packed-weight generic route differs from the checkpoint \
             one: {equal}/{} elements bit-equal",
            reference.len()
        );
        eprintln!(
            "generic SwiGLU M={rows}: packed == checkpoint on {equal}/{} \
             elements ({nonzero} nonzero)",
            reference.len()
        );
    }
}

/// With the packed weight being the only copy, the *same* tensor has to serve
/// the fused route at the pinned M and the generic route at decode's M=1. A
/// build that links only the AOT arm keeps the checkpoint layout instead, and
/// then the same property must hold for that buffer.
#[test]
fn one_weight_serves_both_the_fused_and_the_generic_route() {
    let ctx = CudaContext::new(0).unwrap();
    let layout = gemm::bf16_swiglu_weight_layout(&ctx, FULL_N * K);
    let checkpoint = lcg_bf16(0x5bd1_e995_cafe_0007, FULL_N * K);
    let weight = match layout {
        Bf16SwigluWeightLayout::CheckpointRowMajor => upload(vec![FULL_N, K], &checkpoint),
        Bf16SwigluWeightLayout::PackedPairInterleaved => upload(
            vec![K, FULL_N],
            &gemm::pack_bf16_swiglu_weight_host(
                &checkpoint
                    .iter()
                    .map(|value| value.to_f32())
                    .collect::<Vec<_>>(),
                FULL_N,
                K,
            )
            .unwrap(),
        ),
    };
    let Some(implementation) = gemm::bf16_swiglu_impl(&ctx, &[3387, K], FULL_N * K) else {
        eprintln!("skipping: no fused BF16 SwiGLU implementation applies to this device");
        return;
    };
    assert_eq!(implementation.weight_layout(), layout);

    // Prefill M, fused.
    let prefill = upload(vec![3387, K], &lcg_bf16(0x1234_5678_9abc_def0, 3387 * K));
    let fused = gemm::bf16_swiglu_checkpoint(&ctx, &prefill, &weight).unwrap();
    assert_eq!(fused.shape().dims(), [3584, INNER]);
    let fused_bits = download(&ctx, &fused);
    assert!(
        fused_bits[..3387 * INNER].iter().any(|bits| *bits != 0),
        "the fused route wrote nothing"
    );

    // Decode M, generic, same buffer.
    let decode = upload(vec![1, K], &lcg_bf16(0x0f0f_0f0f_0f0f_0f0f, K));
    let generic = gemm::bf16_swiglu_checkpoint(&ctx, &decode, &weight).unwrap();
    assert_eq!(generic.shape().dims(), [1, INNER]);
    let generic_bits = download(&ctx, &generic);
    assert!(
        generic_bits.iter().any(|bits| *bits != 0),
        "the generic route wrote nothing"
    );
    eprintln!("one {layout:?} buffer served the fused M=3387 and the generic M=1 route");
}

/// What the second copy cost, in bytes, pinned so the report's figure and the
/// code cannot drift apart. The shipping Qwen-Drive text backbone is 32 layers
/// of K=2560, I=9216 (checkpoint `config.json`: `num_hidden_layers` 32,
/// `hidden_size` 2560, `intermediate_size` 9216, 32 `layer_types` entries all
/// of which carry an MLP).
#[test]
fn the_dropped_copy_is_the_reported_size() {
    const LAYERS: usize = 32;
    let per_layer = FULL_N * K * std::mem::size_of::<bf16>();
    assert_eq!(per_layer, 94_371_840);
    assert_eq!(per_layer * LAYERS, 3_019_898_880);
    // The two layouts are the same number of bytes, which is why holding the
    // wrong one is not a size error -- the property the dispatch's shape check
    // exists for.
    assert_eq!(FULL_N * K, K * FULL_N);
    assert_ne!(K, FULL_N, "shape must disambiguate the two layouts");
    assert_eq!(INNER % PAIR, 0, "the column tiling must divide I");
    assert_ne!(
        Bf16SwigluImpl::Aot.weight_layout(),
        Bf16SwigluImpl::Cuda.weight_layout()
    );
}
