// Bit-exactness gate for the fused BF16 first-prefill gated delta rule,
// driven through the repository's own
// `kernels::linear_attention::causal_conv_gdn_prefill_bf16_with` dispatch.
//
// The pinned CuTe-DSL AOT object (`flashinfer_gdn_bf16_t64_sm110_real3392`)
// and the vendored upstream FlashInfer Cake prefill in
// `kernels/flashinfer_gdn/` (BF16 I/O, FP32 state, full-DV,
// `USE_INITIAL_STATE=0`) are required to be bit-identical on BOTH outputs: the
// layer output `o` and the final recurrent state. The state is the one that
// matters: a wrong schedule once scored cosine 0.999988 on it while the bit
// count was 0.001%.
//
// Both arms go through the identical conv/pack/gate prologue and state
// transpose (`adapters/gdn_bf16_prefill_common.cuh`), so a difference can only
// come from the GDN launch itself.
//
//   APXINF_CUDA_ARCH=sm_110 \
//   APXINF_CUDA_AOT_MANIFEST=<bundle>/manifest.json \
//     cargo test -p apxinf-cuda --release \
//       --features aot-gdn-prefill-bf16 --test qwen_drive_gdn_prefill_equivalence

use apxinf_core::{DType, Shape, Tensor};
use apxinf_cuda::kernels::linear_attention::{self as la, GdnPrefillImpl};
use apxinf_cuda::{transfers, CudaBuffer, CudaContext};
use half::bf16;

const SEQ: usize = 3387;
const ZBA_WIDTH: usize = 12352;
const CONV_DIM: usize = 8192;
const CONV_KERNEL: usize = 4;
const HV: usize = 32;
const D: usize = 128;
const GEOMETRY: [usize; 4] = [16, HV, D, D];
const STATE_ELEMENTS: usize = HV * D * D;

/// Deterministic LCG in `[-scale, scale)`.
fn lcg_bf16(seed: u64, count: usize, scale: f32) -> Vec<bf16> {
    let mut state = seed;
    (0..count)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let unit = ((state >> 40) as f32) / ((1u32 << 24) as f32);
            bf16::from_f32((unit - 0.5) * 2.0 * scale)
        })
        .collect()
}

fn lcg_f32(seed: u64, count: usize, low: f32, high: f32) -> Vec<f32> {
    let mut state = seed;
    (0..count)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let unit = ((state >> 40) as f32) / ((1u32 << 24) as f32);
            low + unit * (high - low)
        })
        .collect()
}

fn upload_bf16(shape: Vec<usize>, data: &[bf16]) -> Tensor {
    transfers::to_cuda(&Tensor::from_bf16(shape, data).unwrap(), 0).unwrap()
}

struct Inputs {
    zba: Tensor,
    conv_weight: Tensor,
    a_log: Tensor,
    dt_bias: Tensor,
}

fn build_inputs() -> Inputs {
    // `zba` carries the post-projection q|k|v|z|b|a stream. 0.05 keeps the
    // convolution output in the range the delta rule is actually used over;
    // unit inputs drive `softplus` into its linear tail where `alpha`
    // underflows to 0 and the recurrence degenerates.
    let zba = lcg_bf16(0x2545_F491_4F6C_DD1D, SEQ * ZBA_WIDTH, 0.05);
    let conv_weight = lcg_bf16(0x0fee_1dea_dbee_f001, CONV_DIM * CONV_KERNEL, 0.25);
    // `A_log` is consumed as `-exp(A_log)`; the checkpoint's values sit near
    // this range. `dt_bias` shifts `softplus`.
    let a_log = lcg_f32(0x00c0_ffee_0bad_f00d, HV, -3.0, 0.5);
    let dt_bias = lcg_bf16(0xdead_beef_1234_5678, HV, 1.0);
    Inputs {
        zba: upload_bf16(vec![SEQ, ZBA_WIDTH], &zba),
        conv_weight: upload_bf16(vec![CONV_DIM, CONV_KERNEL], &conv_weight),
        a_log: transfers::to_cuda(&Tensor::from_f32(vec![HV], &a_log).unwrap(), 0).unwrap(),
        dt_bias: upload_bf16(vec![HV], &dt_bias),
    }
}

struct Outputs {
    o: Vec<u16>,
    state: Vec<f32>,
    conv_state: Vec<u16>,
}

fn bits(values: &[f32]) -> Vec<u16> {
    values.iter().map(|value| bf16::from_f32(*value).to_bits()).collect()
}

/// One run into freshly poisoned output buffers. `0xff` everywhere means a
/// kernel that silently writes nothing cannot pass by leaving two zeroed
/// buffers equal; for the f32 state that pattern is a NaN, which also fails
/// the finiteness check below.
fn run(ctx: &CudaContext, inputs: &Inputs, implementation: GdnPrefillImpl) -> Outputs {
    let final_state = CudaBuffer::alloc(STATE_ELEMENTS * 4, ctx.device_id()).unwrap();
    final_state.copy_from_host(&vec![0xffu8; STATE_ELEMENTS * 4]).unwrap();
    let new_conv_state = {
        let bytes = CONV_DIM * CONV_KERNEL * 2;
        let buffer = CudaBuffer::alloc(bytes, ctx.device_id()).unwrap();
        buffer.copy_from_host(&vec![0xffu8; bytes]).unwrap();
        buffer
            .as_tensor(Shape::new(vec![CONV_DIM, CONV_KERNEL]), DType::BF16)
            .unwrap()
    };
    let output = la::causal_conv_gdn_prefill_bf16_with(
        ctx,
        &inputs.zba,
        &inputs.conv_weight,
        &new_conv_state,
        &inputs.a_log,
        &inputs.dt_bias,
        &final_state,
        implementation,
    )
    .unwrap();
    assert_eq!(output.shape().dims(), [SEQ, HV * D]);
    ctx.synchronize().unwrap();

    let mut state = vec![0f32; STATE_ELEMENTS];
    final_state
        .copy_to_host(bytemuck::cast_slice_mut(&mut state))
        .unwrap();
    Outputs {
        o: bits(&transfers::to_cpu(&output).unwrap().to_f32_vec().unwrap()),
        state,
        conv_state: bits(
            &transfers::to_cpu(&new_conv_state)
                .unwrap()
                .to_f32_vec()
                .unwrap(),
        ),
    }
}

fn equal_count<T: PartialEq>(left: &[T], right: &[T]) -> usize {
    left.iter().zip(right).filter(|(a, b)| a == b).count()
}

fn absmax(left: &[f32], right: &[f32]) -> f32 {
    left.iter()
        .zip(right)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max)
}

fn rms(values: &[f32]) -> f32 {
    (values.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>() / values.len() as f64).sqrt()
        as f32
}

#[test]
fn aot_and_cuda_gdn_prefill_are_bit_identical() {
    if !GdnPrefillImpl::Aot.linked() || !GdnPrefillImpl::Cuda.linked() {
        eprintln!(
            "skipping: this build links AOT={} CUDA={}; enable the \
             aot-gdn-prefill-bf16 feature with APXINF_CUDA_AOT_MANIFEST set to \
             link both and run the comparison",
            GdnPrefillImpl::Aot.linked(),
            GdnPrefillImpl::Cuda.linked()
        );
        return;
    }
    let ctx = CudaContext::new(0).unwrap();
    if la::gdn_prefill_impl(&ctx, GEOMETRY).is_none() {
        eprintln!("skipping: this device does not take the fused GDN prefill route");
        return;
    }
    let inputs = build_inputs();

    // Alternate the order: a kernel that corrupts an input it should only read
    // would otherwise be masked by always running second.
    let aot = run(&ctx, &inputs, GdnPrefillImpl::Aot);
    let cuda = run(&ctx, &inputs, GdnPrefillImpl::Cuda);
    let aot_again = run(&ctx, &inputs, GdnPrefillImpl::Aot);

    assert_eq!(aot.o, aot_again.o, "the AOT arm is not reproducible on o");
    assert_eq!(
        aot.state, aot_again.state,
        "the AOT arm is not reproducible on the state"
    );

    // Magnitude, not just cosine: a ruined GDN state points the right
    // direction (cosine 0.999988 was measured on one) and only an absolute
    // check catches it.
    let state_rms = rms(&aot.state);
    assert!(
        state_rms.is_finite() && state_rms > 1e-6 && state_rms < 1e3,
        "the AOT state RMS is {state_rms}, outside the plausible range; the \
         comparison would be vacuous"
    );
    assert!(
        aot.state.iter().all(|value| value.is_finite()),
        "the AOT state has non-finite entries"
    );
    assert!(
        cuda.state.iter().all(|value| value.is_finite()),
        "the CUDA state has non-finite entries"
    );

    let o_equal = equal_count(&aot.o, &cuda.o);
    let state_equal = equal_count(&aot.state, &cuda.state);
    let conv_equal = equal_count(&aot.conv_state, &cuda.conv_state);
    eprintln!(
        "GDN prefill AOT == CUDA: o {o_equal}/{} bit-equal, state \
         {state_equal}/{} bit-equal, conv_state {conv_equal}/{} bit-equal, \
         state absmax {:e}, state RMS {state_rms:e}",
        aot.o.len(),
        aot.state.len(),
        aot.conv_state.len(),
        absmax(&aot.state, &cuda.state)
    );
    assert_eq!(
        conv_equal,
        aot.conv_state.len(),
        "the shared conv prologue differs between arms, so the comparison is \
         not isolated to the GDN launch"
    );
    assert_eq!(o_equal, aot.o.len(), "the layer output differs");
    assert_eq!(state_equal, aot.state.len(), "the final state differs");
}

/// The single-implementation builds still have to reach the kernel: a wiring
/// error that returns a nonzero status would otherwise only surface end to end.
#[test]
fn the_linked_implementation_runs_and_is_reproducible() {
    let ctx = CudaContext::new(0).unwrap();
    let Some(implementation) = la::gdn_prefill_impl(&ctx, GEOMETRY) else {
        eprintln!("skipping: no fused GDN prefill implementation applies to this device");
        return;
    };
    let inputs = build_inputs();
    let first = run(&ctx, &inputs, implementation);
    let second = run(&ctx, &inputs, implementation);
    let state_rms = rms(&first.state);
    assert!(
        state_rms.is_finite() && state_rms > 1e-6 && state_rms < 1e3,
        "{implementation:?} produced a state RMS of {state_rms}"
    );
    assert_eq!(first.o, second.o, "{implementation:?} is not reproducible on o");
    assert_eq!(
        first.state, second.state,
        "{implementation:?} is not reproducible on the state"
    );
}
