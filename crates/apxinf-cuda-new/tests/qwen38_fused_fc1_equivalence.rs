// Bit-exactness gate for the fused Qwen3.8 NVFP4 FC1 + SwiGLU operator,
// driven through the repository's own `ops::nvfp4_dense_swiglu` call path.
//
// The CuTe-DSL AOT object and the in-tree CUDA/CUTLASS translation are
// required to be byte-identical in *both* outputs: the packed E2M1 activation
// and the E4M3 block scales. Running them in one process, back to back, on the
// same device buffers is the only form of this check that cannot be confused
// by a thermal or power-state difference, and it is the form that catches a
// wiring bug the scratch harness cannot: the two implementations take
// different activation block-scale layouts (row-major for the AOT, the GEMM
// atom layout for CUTLASS) and the two layouts have the same byte count.
//
// The test skips honestly when only one implementation is linked. Build the
// comparison arm with:
//
//   APXINF_CUDA_ARCH=sm_110 \
//   APXINF_CUDA_AOT_MANIFEST=<bundle>/manifest.json \
//     cargo test -p apxinf-cuda-new --release \
//       --features aot-qwen38-dense-swiglu --test qwen38_fused_fc1_equivalence

use apxinf_core::{DType, Shape, Tensor};
use apxinf_cuda_new::ops::DenseSwigluImpl;
use apxinf_cuda_new::{ops, CudaBuffer, CudaContext};

// The AOT object is specialized to exactly this shape.
const ROWS: usize = 2048;
const K: usize = 5120;
const INTERM: usize = 17408;
const BLOCK: u32 = 16;

fn upload(ctx: &CudaContext, bytes: &[u8], dims: Vec<usize>, dtype: DType) -> Tensor {
    let buffer = CudaBuffer::alloc(bytes.len().max(1), ctx.device_id()).unwrap();
    buffer.copy_from_host(bytes).unwrap();
    buffer.as_tensor(Shape::new(dims), dtype).unwrap()
}

fn zeros(ctx: &CudaContext, bytes: usize, dims: Vec<usize>, dtype: DType) -> Tensor {
    upload(ctx, &vec![0u8; bytes], dims, dtype)
}

fn download(tensor: &Tensor, count: usize) -> Vec<u8> {
    let mut bytes = vec![0; count];
    CudaBuffer::from_tensor(tensor).unwrap().copy_to_host(&mut bytes).unwrap();
    bytes
}

/// Deterministic byte stream. An LCG rather than a constant pattern so every
/// E2M1 nibble and a wide span of E4M3 scale codes are exercised; a fixed seed
/// so the two arms see byte-identical input.
fn lcg_bytes(seed: u64, count: usize) -> Vec<u8> {
    let mut state = seed;
    (0..count)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 33) as u8
        })
        .collect()
}

/// E4M3 codes restricted to a benign exponent range: `0x7f` is NaN and would
/// poison every comparison, and extreme exponents overflow the accumulator so
/// both arms would agree on `inf` without proving anything.
fn benign_e4m3(seed: u64, count: usize) -> Vec<u8> {
    lcg_bytes(seed, count)
        .into_iter()
        .map(|byte| 0x30 | (byte & 0x0f))
        .collect()
}

fn f32_bytes(value: f32) -> Vec<u8> {
    value.to_le_bytes().to_vec()
}

struct Inputs {
    activation: Tensor,
    activation_scales_row_major: Tensor,
    activation_scales_atom: Tensor,
    weight: Tensor,
    weight_scales: Tensor,
    alpha: Tensor,
    input_global_scale: Tensor,
    down_inverse_global_scale: Tensor,
    tile_groups: Tensor,
    tile_limits: Tensor,
    token_map: Tensor,
    tile_count: Tensor,
}

fn build_inputs(ctx: &CudaContext) -> Inputs {
    let scale_blocks = K / BLOCK as usize;
    let activation = upload(
        ctx,
        &lcg_bytes(0x2545_F491_4F6C_DD1D, ROWS * K / 2),
        vec![ROWS, K / 2],
        DType::E2M1Pair,
    );
    // The AOT wants row-major activation scales; CUTLASS wants the atom
    // layout. Build the row-major buffer and let the production packer produce
    // the atom one from the same bytes, so the two arms really do see the same
    // numbers and not merely the same byte count.
    let row_major_bytes = benign_e4m3(0x1234_5678_9abc_def0, ROWS * scale_blocks);
    let activation_scales_row_major = upload(
        ctx,
        &row_major_bytes,
        vec![ROWS, scale_blocks],
        DType::F8E4M3,
    );
    let atom_bytes = ops::nvfp4_scale_buffer_bytes(ROWS, K, BLOCK).unwrap();
    let activation_scales_atom = zeros(ctx, atom_bytes, vec![atom_bytes], DType::F8E4M3);
    ops::nvfp4_pack_block_scales(
        ctx,
        &activation_scales_row_major,
        &activation_scales_atom,
        ROWS,
        K,
        BLOCK,
    )
    .unwrap();

    let weight = upload(
        ctx,
        &lcg_bytes(0x0fee_1dea_dbee_f001, 2 * INTERM * K / 2),
        vec![2 * INTERM, K / 2],
        DType::E2M1Pair,
    );
    // Weight scales are the atom layout for both implementations (report 10:
    // only SFA differs), so one buffer serves both arms.
    let weight_atom_bytes = ops::nvfp4_scale_buffer_bytes(2 * INTERM, K, BLOCK).unwrap();
    let weight_scales_row_major = upload(
        ctx,
        &benign_e4m3(0x00c0_ffee_0bad_f00d, 2 * INTERM * scale_blocks),
        vec![2 * INTERM, scale_blocks],
        DType::F8E4M3,
    );
    let weight_scales = zeros(ctx, weight_atom_bytes, vec![weight_atom_bytes], DType::F8E4M3);
    ops::nvfp4_pack_block_scales(
        ctx,
        &weight_scales_row_major,
        &weight_scales,
        2 * INTERM,
        K,
        BLOCK,
    )
    .unwrap();

    let tiles = ROWS / 128;
    let i32_bytes = |values: &[i32]| -> Vec<u8> {
        values.iter().flat_map(|value| value.to_le_bytes()).collect()
    };
    Inputs {
        activation,
        activation_scales_row_major,
        activation_scales_atom,
        weight,
        weight_scales,
        // `alpha` and the two global scales reproduce the oracle's constants.
        alpha: upload(ctx, &f32_bytes(0.013), vec![1], DType::F32),
        input_global_scale: upload(ctx, &f32_bytes(1.25), vec![1], DType::F32),
        down_inverse_global_scale: upload(ctx, &f32_bytes(0.02), vec![1], DType::F32),
        tile_groups: upload(ctx, &i32_bytes(&vec![0; tiles]), vec![tiles], DType::I32),
        tile_limits: upload(
            ctx,
            &i32_bytes(&vec![ROWS as i32; tiles]),
            vec![tiles],
            DType::I32,
        ),
        token_map: upload(
            ctx,
            &i32_bytes(&(0..ROWS as i32).collect::<Vec<_>>()),
            vec![ROWS],
            DType::I32,
        ),
        tile_count: upload(ctx, &i32_bytes(&[tiles as i32]), vec![1], DType::I32),
    }
}

/// Run one implementation into freshly poisoned output buffers and return
/// (packed E2M1, E4M3 scales). Poisoning with `0xff` means a kernel that
/// silently writes nothing cannot pass by leaving two zeroed buffers equal.
fn run(ctx: &CudaContext, inputs: &Inputs, implementation: DenseSwigluImpl) -> (Vec<u8>, Vec<u8>) {
    let output_bytes = ROWS * INTERM / 2;
    let scale_bytes = ops::nvfp4_scale_buffer_bytes(ROWS, INTERM, BLOCK).unwrap();
    let output = upload(
        ctx,
        &vec![0xff; output_bytes],
        vec![ROWS, INTERM / 2],
        DType::E2M1Pair,
    );
    let output_scales =
        upload(ctx, &vec![0xff; scale_bytes], vec![scale_bytes], DType::F8E4M3);
    let activation_scales = match implementation.activation_scale_layout() {
        ops::ScaleLayout::RowMajor => &inputs.activation_scales_row_major,
        ops::ScaleLayout::GemmAtom => &inputs.activation_scales_atom,
    };
    ops::nvfp4_dense_swiglu(
        ctx,
        implementation,
        &inputs.activation,
        activation_scales,
        &inputs.weight,
        &inputs.weight_scales,
        &output,
        &output_scales,
        &inputs.alpha,
        &inputs.input_global_scale,
        &inputs.down_inverse_global_scale,
        &inputs.tile_groups,
        &inputs.tile_limits,
        &inputs.token_map,
        &inputs.tile_count,
    )
    .unwrap();
    ctx.synchronize().unwrap();
    (
        download(&output, output_bytes),
        download(&output_scales, scale_bytes),
    )
}

fn equal_count(a: &[u8], b: &[u8]) -> usize {
    a.iter().zip(b).filter(|(left, right)| left == right).count()
}

#[test]
fn aot_and_cuda_fused_fc1_are_byte_identical() {
    if !apxinf_cuda_new::QWEN38_DENSE_SWIGLU_AOT || !apxinf_cuda_new::QWEN38_DENSE_SWIGLU_CUDA {
        eprintln!(
            "skipping: this build links only {:?}; enable the \
             aot-qwen38-dense-swiglu feature with APXINF_CUDA_AOT_MANIFEST set \
             to link both and run the comparison",
            apxinf_cuda_new::QWEN38_DENSE_SWIGLU
        );
        return;
    }
    let ctx = CudaContext::new(0).unwrap();
    let inputs = build_inputs(&ctx);

    // Alternate the order: a kernel that corrupts an input it should only read
    // would otherwise be masked by always running second.
    let (aot_packed, aot_scales) = run(&ctx, &inputs, DenseSwigluImpl::Aot);
    let (cuda_packed, cuda_scales) = run(&ctx, &inputs, DenseSwigluImpl::Cuda);
    let (aot_packed_2, aot_scales_2) = run(&ctx, &inputs, DenseSwigluImpl::Aot);

    assert_eq!(aot_packed, aot_packed_2, "the AOT arm is not reproducible");
    assert_eq!(aot_scales, aot_scales_2, "the AOT arm is not reproducible");
    assert!(
        aot_packed.iter().any(|byte| *byte != 0xff),
        "the AOT arm wrote nothing"
    );

    let packed_equal = equal_count(&aot_packed, &cuda_packed);
    let scales_equal = equal_count(&aot_scales, &cuda_scales);
    assert_eq!(
        packed_equal,
        aot_packed.len(),
        "packed E2M1 output differs: {packed_equal}/{} bytes equal",
        aot_packed.len()
    );
    assert_eq!(
        scales_equal,
        aot_scales.len(),
        "E4M3 block scales differ: {scales_equal}/{} bytes equal",
        aot_scales.len()
    );
    eprintln!(
        "fused FC1 AOT == CUDA: {}/{} packed bytes, {}/{} scale bytes",
        packed_equal,
        aot_packed.len(),
        scales_equal,
        aot_scales.len()
    );
}

/// The single-implementation builds still have to reach the kernel: a wiring
/// error that returns `APXINF_STATUS_UNSUPPORTED` would otherwise only show up
/// at end-to-end time.
#[test]
fn the_linked_implementation_runs_and_is_reproducible() {
    let Some(implementation) = apxinf_cuda_new::QWEN38_DENSE_SWIGLU else {
        eprintln!("skipping: no fused FC1 implementation is linked");
        return;
    };
    let ctx = CudaContext::new(0).unwrap();
    let inputs = build_inputs(&ctx);
    let (first_packed, first_scales) = run(&ctx, &inputs, implementation);
    let (second_packed, second_scales) = run(&ctx, &inputs, implementation);
    assert!(
        first_packed.iter().any(|byte| *byte != 0xff),
        "{implementation:?} wrote nothing"
    );
    assert_eq!(first_packed, second_packed, "{implementation:?} is not reproducible");
    assert_eq!(first_scales, second_scales, "{implementation:?} is not reproducible");
}
