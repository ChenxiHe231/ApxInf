// Bit-exactness gates for the three FA4 attention operators, driven through
// the repository's own `kernels::attention` dispatch rather than through any
// `devlocal/` harness code.
//
// Each operator's pinned CuTe-DSL AOT object and the in-tree CUTLASS
// translation (`kernels/cutlass/fmha_vision_sm100.cu`,
// `fmha_d256_language_sm100.cu`, `fmha_d256_action_sm100.cu`) are required to
// be bit-identical. The numeric contract of each is its `-DAPXINF_*` flag line
// in `crates/apxinf-cuda/build.rs`; applying one operator's contract to
// another does not error, it silently drops bit-exactness to roughly 50%,
// which is exactly what these tests exist to catch.
//
// Shared between the arms, so a difference can only come from the attention
// launch itself:
//   * vision   `apxinf_static_vision_qk_rope_bf16_no_v` (the RoPE), and V read
//              in place out of the retained interleaved QKV buffer
//   * action   `apxinf_split_batch_duplicate_q` and `apxinf_split_batch_merge`
//
// EVERY assertion here is on integer ULP distance of the bit patterns, never
// on `absmax`. `absmax` is an absolute metric and is therefore blind to a
// large relative error on a small element; asserting it is precisely what let
// the action merge ship with a 12,112-ULP sign flip at `key_tokens = 3436`
// (reports 26 and 27). The same reasoning applies to any future gate in this
// file: `absmax` is for the transcript, ULP is for the assertion.
//
//   APXINF_CUDA_ARCH=sm_110 \
//   APXINF_CUDA_AOT_MANIFEST=<bundle>/manifest.json \
//     cargo test -p apxinf-cuda --release \
//       --features aot-vision-attention,aot-action-attention,aot-language-attention \
//       --test qwen_drive_fa4_attention_equivalence -- --test-threads=1

use apxinf_core::{DType, Shape, Tensor};
use apxinf_cuda::kernels::attention::{self as attn, Fa4AttentionImpl, Fa4Operator};
use apxinf_cuda::{transfers, CudaBuffer, CudaContext};
use half::bf16;

const SCENE_TOKENS: usize = 3387;
const VISION_TOKENS: usize = 12216;
const VISION_HEADS: usize = 16;
const VISION_HEAD_DIM: usize = 64;
const VISION_OFFSETS: [u32; 13] = [
    0, 624, 1248, 1872, 4072, 4696, 5320, 5944, 8144, 8768, 9392, 10016, 12216,
];

/// Deterministic LCG in `[-scale, scale)`.
///
/// `scale` is small on purpose. At unit inputs a head_dim-256 dot product
/// reaches `|score| ~ 16`, where the online softmax saturates: one KV tile
/// wins every row, `ex2` returns exactly 0 for everything else, and the
/// polynomial emulation and the rescale threshold -- the two things these
/// operators' contracts actually differ in -- never have to agree on anything.
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

fn upload_bf16(shape: Vec<usize>, data: &[bf16]) -> Tensor {
    transfers::to_cuda(&Tensor::from_bf16(shape, data).unwrap(), 0).unwrap()
}

fn bits(tensor: &Tensor) -> Vec<u16> {
    transfers::to_cpu(tensor)
        .unwrap()
        .to_f32_vec()
        .unwrap()
        .iter()
        .map(|value| bf16::from_f32(*value).to_bits())
        .collect()
}

/// FP32 bit patterns, for the action LSE. Compared as bits so a signed zero
/// difference cannot pass, the same rule the BF16 comparison follows.
fn f32_bits(tensor: &Tensor) -> Vec<u32> {
    transfers::to_cpu(tensor)
        .unwrap()
        .to_f32_vec()
        .unwrap()
        .iter()
        .map(|value| value.to_bits())
        .collect()
}

fn equal_count_u32(left: &[u32], right: &[u32]) -> usize {
    left.iter().zip(right).filter(|(a, b)| a == b).count()
}

fn max_ulp_u32(left: &[u32], right: &[u32]) -> i64 {
    let ordered = |bits: u32| {
        if bits & 0x8000_0000 != 0 {
            -i64::from(bits & 0x7fff_ffff)
        } else {
            i64::from(bits)
        }
    };
    left.iter()
        .zip(right)
        .map(|(a, b)| (ordered(*a) - ordered(*b)).abs())
        .max()
        .unwrap_or(0)
}

fn equal_count(left: &[u16], right: &[u16]) -> usize {
    left.iter().zip(right).filter(|(a, b)| a == b).count()
}

fn absmax(left: &[u16], right: &[u16]) -> f32 {
    left.iter()
        .zip(right)
        .map(|(a, b)| (bf16::from_bits(*a).to_f32() - bf16::from_bits(*b).to_f32()).abs())
        .fold(0.0f32, f32::max)
}

/// Monotone integer index of a BF16 bit pattern: positive patterns keep their
/// value, negative ones are reflected, so `ordered(b) - ordered(a)` counts the
/// representable BF16 values between the two and crossing zero costs what it
/// should.
///
/// **`absmax` is not a substitute for this and this gate used to assert only
/// `absmax`.** `absmax` is an ABSOLUTE metric, so it is blind by construction
/// to a large RELATIVE error on a small-magnitude element: report 26 found the
/// action merge off by 44,832 raw-`u16` / 12,112 ordered BF16 ULP at
/// `key_tokens = 3436` -- a sign flip across zero -- while `absmax` read
/// 7.6e-6, numerically indistinguishable from the other four shapes and four
/// orders of magnitude *smaller* than the 1-ULP rounding census they carry.
/// Report 27 found and fixed the cause. Assert ULP, print `absmax`.
fn ordered(bits: u16) -> i32 {
    if bits & 0x8000 != 0 {
        -(i32::from(bits & 0x7fff))
    } else {
        i32::from(bits)
    }
}

/// Worst elementwise integer ULP distance, with the index and both bit
/// patterns so a failure names the element.
fn max_ulp(left: &[u16], right: &[u16]) -> (i32, usize) {
    left.iter()
        .zip(right)
        .enumerate()
        .map(|(index, (a, b))| ((ordered(*a) - ordered(*b)).abs(), index))
        .max()
        .unwrap_or((0, 0))
}

/// Every mismatching element is within `limit` integer ULP, named individually.
fn assert_within_ulp(label: &str, left: &[u16], right: &[u16], limit: i32) {
    for (index, (a, b)) in left.iter().zip(right).enumerate() {
        let distance = (ordered(*a) - ordered(*b)).abs();
        assert!(
            distance <= limit,
            "{label}: element {index} differs by {distance} BF16 ULP, more than the \
             permitted {limit} (aot {a:#06x} = {}, cuda {b:#06x} = {}). A large ULP \
             distance at a small absolute difference is exactly what `absmax` cannot \
             see -- see report 27.",
            bf16::from_bits(*a).to_f32(),
            bf16::from_bits(*b).to_f32()
        );
    }
}

fn rows_with_mismatch(left: &[u16], right: &[u16], row: usize) -> usize {
    left.chunks(row)
        .zip(right.chunks(row))
        .filter(|(a, b)| a != b)
        .count()
}

fn report(label: &str, aot: &[u16], cuda: &[u16]) -> usize {
    let equal = equal_count(aot, cuda);
    let (ulp, index) = max_ulp(aot, cuda);
    // Only name an element when there IS one: `max_ulp` breaks ties by index,
    // so a perfect match would otherwise print the last element and read like
    // a finding.
    let worst = if ulp == 0 {
        String::new()
    } else {
        format!(" (element {index})")
    };
    println!(
        "  {label}: {equal}/{} bit-equal ({:.6}%), max {ulp} BF16 ULP{worst}, absmax {:e}",
        aot.len(),
        100.0 * equal as f64 / aot.len() as f64,
        absmax(aot, cuda)
    );
    equal
}

/// A build that links only one arm cannot compare anything. Say so and return
/// rather than passing silently.
fn both_arms(operator: Fa4Operator, feature: &str) -> bool {
    let aot = Fa4AttentionImpl::Aot.linked(operator);
    let cuda = Fa4AttentionImpl::Cuda.linked(operator);
    if aot && cuda {
        return true;
    }
    eprintln!(
        "skipping: this build links AOT={aot} CUDA={cuda} for {operator:?}; enable the \
         {feature} feature with APXINF_CUDA_AOT_MANIFEST set to link both and run \
         the comparison"
    );
    false
}

fn context() -> Option<CudaContext> {
    let ctx = CudaContext::new(0).ok()?;
    if ctx.caps().sm != 110 || ctx.caps().multiprocessor_count != 20 {
        eprintln!("skipping: these operators are specialised for SM110 with 20 SMs");
        return None;
    }
    Some(ctx)
}

// ---------------------------------------------------------------------------
// vision-attention (hd64, 12 segments over 3 batches of 4072 padded tokens)
// ---------------------------------------------------------------------------

struct VisionInputs {
    qkv: Tensor,
    /// Retained so a test can compute the uniform average of V on the host and
    /// prove its own stimulus is not in the plain-averaging regime. See
    /// `uniform_average_ratio`.
    host_qkv: Vec<bf16>,
    position_ids: CudaBuffer,
    offsets: CudaBuffer,
}

fn vision_inputs(ctx: &CudaContext) -> VisionInputs {
    vision_inputs_at_scale(ctx, 0.05)
}

fn vision_inputs_at_scale(ctx: &CudaContext, scale: f32) -> VisionInputs {
    vision_inputs_from(ctx, 0x9E37_79B9_7F4A_7C15, scale, 1.0, 1)
}

/// Deterministic LCG in `[-scale, scale)` with a sparse multiplicative tail:
/// one element in `one_in` is scaled by `gain`.
///
/// 🔴 The tail is the whole point, and it is calibrated against a capture, not
/// guessed. Vision block 1's real QKV, dumped out of the running model this
/// round (report 31 §3), has **std 1.1012 and absmax 10.25**, i.e.
/// `|max|/sigma = 9.3`. A uniform `+-s` stream has `|max|/sigma = sqrt(3) =
/// 1.73` whatever `s` is, so matching the model's variance by raising `s` --
/// which is what `VISION_SCALES` does -- still cannot produce the score
/// outliers that separate the two arms. `scale = 1.28, gain = 8, one_in = 52`
/// gives std 1.10 and absmax 10.24: both of the model's moments and its tail.
fn lcg_bf16_tailed(seed: u64, count: usize, scale: f32, gain: f32, one_in: u64) -> Vec<bf16> {
    let mut state = seed;
    (0..count)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let unit = ((state >> 40) as f32) / ((1u32 << 24) as f32);
            let tail = if one_in > 1 && (state >> 8) % one_in == 0 {
                gain
            } else {
                1.0
            };
            bf16::from_f32((unit - 0.5) * 2.0 * scale * tail)
        })
        .collect()
}

fn vision_inputs_from(
    ctx: &CudaContext,
    seed: u64,
    scale: f32,
    gain: f32,
    one_in: u64,
) -> VisionInputs {
    let width = 3 * VISION_HEADS * VISION_HEAD_DIM;
    let qkv = lcg_bf16_tailed(seed, VISION_TOKENS * width, scale, gain, one_in);
    // Two u32 per token. The model's are the merge-block-major (row, col) pairs
    // of 12 grids -- `[1,26,24] x3` then `[1,50,44]`, repeated three times,
    // captured off the fixture this round -- so the maxima are 25/23 and 49/43.
    // The synthetic pattern below spans a wider range; RoPE is a rotation, so
    // this changes which angles are used, not the magnitudes.
    let mut positions = vec![0u32; VISION_TOKENS * 2];
    for (index, slot) in positions.chunks_mut(2).enumerate() {
        slot[0] = (index % 64) as u32;
        slot[1] = (index / 64) as u32;
    }
    let position_ids = CudaBuffer::alloc(positions.len() * 4, ctx.device_id()).unwrap();
    position_ids
        .copy_from_host(bytemuck::cast_slice(&positions))
        .unwrap();
    let offsets_i32: Vec<i32> = VISION_OFFSETS.iter().map(|value| *value as i32).collect();
    let offsets = CudaBuffer::alloc(offsets_i32.len() * 4, ctx.device_id()).unwrap();
    offsets
        .copy_from_host(bytemuck::cast_slice(&offsets_i32))
        .unwrap();
    VisionInputs {
        qkv: upload_bf16(vec![VISION_TOKENS, width], &qkv),
        host_qkv: qkv,
        position_ids,
        offsets,
    }
}

/// How far the kernel's own output gets from the **uniform average of V** over
/// the same segment, as a fraction of V's spread about that average.
///
/// 🔴 This is the stimulus census, and it is the reason this file's original
/// `+-0.05` vision gate could report `12509184/12509184 absmax 0` while the
/// same operator moved the Qwen-Drive trajectory (report 31). At `+-0.05` a
/// head_dim-64 BF16 dot product has `|score| ~ 8e-4`, so every softmax weight
/// is within 0.43 % of `1/L`, `exp2` is only ever evaluated on `[-0.007, 0]`,
/// the online-softmax rescale (`rescale_threshold = 8.0`) can never trigger,
/// and the masked-tile ordering cannot matter because all the weights are
/// equal. Two kernels agreeing bit-for-bit on that stimulus have agreed on
/// `mean(V)`, not on attention. A gate over a non-discriminative stimulus is
/// the same class of defect as the `APXINF_EMIT_LSE=0` harness that compared
/// two buffers the kernel never wrote (SHARED-FINDINGS): it reads 100 % by
/// construction.
///
/// `~0` means plain averaging. The returned value is asserted to be large, so
/// a future change that quietly flattens the stimulus fails the gate instead
/// of silently making it vacuous.
fn uniform_average_ratio(out: &[u16], host_qkv: &[bf16], start: usize, len: usize) -> f32 {
    const HEAD: usize = 0;
    let width = 3 * VISION_HEADS * VISION_HEAD_DIM;
    let v_base = 2 * VISION_HEADS * VISION_HEAD_DIM + HEAD * VISION_HEAD_DIM;
    let v = |token: usize, d: usize| -> f64 {
        f64::from(host_qkv[token * width + v_base + d].to_f32())
    };
    let mut mean = [0f64; VISION_HEAD_DIM];
    for token in start..start + len {
        for (d, slot) in mean.iter_mut().enumerate() {
            *slot += v(token, d);
        }
    }
    for slot in mean.iter_mut() {
        *slot /= len as f64;
    }
    let mut spread = 0f64;
    for token in start..start + len {
        for (d, slot) in mean.iter().enumerate() {
            spread = spread.max((v(token, d) - slot).abs());
        }
    }
    if spread == 0.0 {
        return 0.0;
    }
    let mut worst = 0f64;
    for query in start..start + len {
        let row = (query * VISION_HEADS + HEAD) * VISION_HEAD_DIM;
        for (d, slot) in mean.iter().enumerate() {
            let value = f64::from(bf16::from_bits(out[row + d]).to_f32());
            worst = worst.max((value - slot).abs());
        }
    }
    (worst / spread) as f32
}

/// Bit-equality broken out per fixed group, so a failure says WHICH of the two
/// AOT objects disagrees instead of leaving it to be guessed. The fixed-group
/// call is `3 x (batch-3, L=624)` plus `1 x (batch-3, L=2200)` launches over a
/// 4072-token batch stride, so group membership is `(token % 4072)` against
/// `FA4_VFIXED_OFFSETS`' first four entries.
fn vision_group_census(aot: &[u16], cuda: &[u16]) -> [(usize, usize, i32); 4] {
    const STARTS: [usize; 4] = [0, 624, 1248, 1872];
    const LENGTHS: [usize; 4] = [624, 624, 624, 2200];
    let row = VISION_HEADS * VISION_HEAD_DIM;
    let mut census = [(0usize, 0usize, 0i32); 4];
    for token in 0..VISION_TOKENS {
        let within = token % 4072;
        let group = STARTS
            .iter()
            .zip(LENGTHS)
            .position(|(start, len)| within >= *start && within < start + len)
            .expect("every token belongs to exactly one fixed group");
        for element in token * row..(token + 1) * row {
            let distance = (ordered(aot[element]) - ordered(cuda[element])).abs();
            census[group].1 += 1;
            if distance == 0 {
                census[group].0 += 1;
            }
            census[group].2 = census[group].2.max(distance);
        }
    }
    census
}

fn run_vision(
    ctx: &CudaContext,
    inputs: &VisionInputs,
    implementation: Fa4AttentionImpl,
) -> Vec<u16> {
    let output = attn::vision_qkv_rope_segmented_fa4_skip_v_with(
        ctx,
        &inputs.qkv,
        &inputs.position_ids,
        VISION_HEADS,
        VISION_HEAD_DIM,
        10000.0,
        &inputs.offsets,
        VISION_OFFSETS.as_slice(),
        12,
        true,
        implementation,
    )
    .unwrap()
    .expect("the fixed-group vision route rejected its own geometry");
    assert_eq!(
        output.shape().dims(),
        [VISION_TOKENS, VISION_HEADS, VISION_HEAD_DIM]
    );
    ctx.synchronize().unwrap();
    bits(&output)
}

#[test]
fn aot_and_cuda_vision_attention_are_bit_identical() {
    if !both_arms(Fa4Operator::Vision, "aot-vision-attention") {
        return;
    }
    let Some(ctx) = context() else { return };
    let inputs = vision_inputs(&ctx);
    // Alternate the order: a kernel that corrupts an input it should only read
    // would otherwise be masked by always running second.
    let aot = run_vision(&ctx, &inputs, Fa4AttentionImpl::Aot);
    let cuda = run_vision(&ctx, &inputs, Fa4AttentionImpl::Cuda);
    let aot_again = run_vision(&ctx, &inputs, Fa4AttentionImpl::Aot);
    assert_eq!(aot, aot_again, "the vision AOT arm is not reproducible");
    assert!(
        aot.iter().any(|value| *value != 0),
        "both arms wrote nothing: the comparison would be vacuous"
    );
    let equal = report("vision AOT == CUDA", &aot, &cuda);
    assert_eq!(
        equal,
        aot.len(),
        "vision attention is not bit-identical between the arms"
    );
}

/// The input scales the vision gate is run at. `0.05` is the legacy stimulus
/// and is kept as the continuity control; the rest leave the plain-averaging
/// regime that one sits in. `2.0` is the one that matches the model: modelled
/// from this checkpoint's own `visual.blocks.0.norm1` and `attn.qkv` weights
/// with unit-variance pre-norm activations, the head-0 score row range is
/// 7.43 (median) / 9.79 (max) and the winning key carries 58x the uniform
/// weight, against row range 5.1e-3 and 1.004x at `+-0.05` (report 31 S2).
const VISION_SCALES: [f32; 5] = [0.05, 0.25, 0.5, 1.0, 2.0];

/// 🔴 The gate report 31 added, and the one that fails.
///
/// `aot_and_cuda_vision_attention_are_bit_identical` above drives the same
/// public entry over the same 12 segments and the same two AOT objects, and
/// passes `12509184/12509184 absmax 0`. It passes because its stimulus makes
/// the attention a plain mean of V (see `uniform_average_ratio`): the two
/// things the arms' numeric contracts actually differ in -- the ex2 polynomial
/// emulation's argument range and the `rescale_threshold = 8.0` online-softmax
/// rescale -- are both unreachable at `+-0.05`, so the comparison cannot see
/// them. Report 29 then measured the same operator moving the Qwen-Drive
/// trajectory on all four fixture scenes.
///
/// This test keeps everything else identical and varies only the input scale,
/// and it asserts that the stimulus is discriminative before it asserts
/// anything about the arms, so it cannot decay into the vacuous form.
#[test]
fn aot_and_cuda_vision_attention_are_bit_identical_across_input_regimes() {
    if !both_arms(Fa4Operator::Vision, "aot-vision-attention") {
        return;
    }
    let Some(ctx) = context() else { return };
    let mut best_ratio = 0.0f32;
    let mut failures: Vec<String> = Vec::new();
    for scale in VISION_SCALES {
        let inputs = vision_inputs_at_scale(&ctx, scale);
        let aot = run_vision(&ctx, &inputs, Fa4AttentionImpl::Aot);
        let cuda = run_vision(&ctx, &inputs, Fa4AttentionImpl::Cuda);
        let aot_again = run_vision(&ctx, &inputs, Fa4AttentionImpl::Aot);
        assert_eq!(
            aot, aot_again,
            "the vision AOT arm is not reproducible at input scale {scale}"
        );
        assert!(
            aot.iter().any(|value| *value != 0),
            "both arms wrote nothing at input scale {scale}: the comparison would be vacuous"
        );
        // The L=624 group 0 and the L=2200 group 3 of the first batch, i.e. one
        // of each AOT object, measured on the kernel's own output.
        let ratio_624 = uniform_average_ratio(&aot, &inputs.host_qkv, 0, 624);
        let ratio_2200 = uniform_average_ratio(&aot, &inputs.host_qkv, 1872, 2200);
        best_ratio = best_ratio.max(ratio_624.max(ratio_2200));
        let equal = report(&format!("vision scale +-{scale}"), &aot, &cuda);
        println!(
            "      stimulus/uniform-average ratio: L=624 {ratio_624:.4}, L=2200 {ratio_2200:.4} \
             (~0 == plain mean of V, so a bit-exact result there means nothing)"
        );
        let census = vision_group_census(&aot, &cuda);
        for (group, (equal_in_group, total, worst)) in census.iter().enumerate() {
            let length = if group == 3 { 2200 } else { 624 };
            println!(
                "      group {group} (L={length}, AOT object vision-attention-{length}): \
                 {equal_in_group}/{total} bit-equal, max {worst} BF16 ULP"
            );
        }
        if equal != aot.len() {
            let (worst, index) = max_ulp(&aot, &cuda);
            failures.push(format!(
                "scale +-{scale}: {}/{} elements differ, worst {worst} BF16 ULP at element \
                 {index} (token {}, head {}, dim {}); per-group bit-equal {:?}",
                aot.len() - equal,
                aot.len(),
                index / (VISION_HEADS * VISION_HEAD_DIM),
                (index / VISION_HEAD_DIM) % VISION_HEADS,
                index % VISION_HEAD_DIM,
                census.map(|(e, t, _)| (e, t)),
            ));
        }
    }
    // Stimulus census FIRST: a gate over a non-discriminative stimulus reads
    // 100% by construction, which is exactly how the `+-0.05` gate passed.
    assert!(
        best_ratio > 0.25,
        "every input scale in VISION_SCALES left the vision attention in the \
         plain-averaging regime (best stimulus/uniform-average ratio {best_ratio:.4} <= 0.25). \
         This gate would read 100% whatever the kernels did; raise the scales."
    );
    assert!(
        failures.is_empty(),
        "vision attention is NOT bit-identical between the arms once the stimulus leaves \
         the plain-averaging regime:\n  {}\n\
         The `+-0.05` gate in this file passes on the same call path, the same 12 segments \
         and the same two AOT objects -- so the difference is DATA-DEPENDENT, in the ex2 \
         emulation range or the rescale_threshold path, not in object selection, offsets, \
         strides or coverage. See report 31.",
        failures.join("\n  ")
    );
}

/// 🔴 **The gate report 31 delivers.** It is the one that fails on the current
/// tree and the one that would have blocked `aot-vision-attention`.
///
/// Measured in-model this round (report 31 §3): at vision block 1, with
/// *bit-identical* QKV input, the AOT and in-tree arms' attention outputs
/// differ in **exactly one element of 12 509 184, by exactly one BF16 ULP**,
/// in fixed group 3 (the `vision-attention-2200` object). One element in
/// 1.25e7. That is why every stimulus above misses it: a defect at a ~1e-7
/// per-element rate needs a stimulus with the model's TAIL (to make the
/// offending score configuration reachable at all) and enough independent
/// draws for it to appear.
///
/// So this test does both: `lcg_bf16_tailed` matches the captured QKV's std
/// (1.10) **and** its absmax (10.25), and it draws
/// `VISION_TAIL_SEEDS.len()` independent streams, i.e.
/// `len x 12 509 184` compared elements.
///
/// Every mismatch is required to be ZERO ULP -- the two arms are specified
/// bit-identical, and §1 of report 31 shows a single 1-ULP seed here reaches
/// the output as 2 ULP of a waypoint. There is no tolerance to spend.
const VISION_TAIL_SEEDS: [u64; 8] = [
    0x9E37_79B9_7F4A_7C15,
    0x0123_4567_89AB_CDEF,
    0xDEAD_BEEF_CAFE_F00D,
    0x5151_5151_A2A2_A2A2,
    0x1111_2222_3333_4444,
    0xFEDC_BA98_7654_3210,
    0x7F4A_7C15_9E37_79B9,
    0xA5A5_A5A5_5A5A_5A5A,
];

#[test]
fn aot_and_cuda_vision_attention_are_bit_identical_on_a_model_shaped_stimulus() {
    if !both_arms(Fa4Operator::Vision, "aot-vision-attention") {
        return;
    }
    let Some(ctx) = context() else { return };
    // Captured from the running model, vision block 1, fixture scene s3
    // (`key_tokens = 3436`): the geometry where report 29 measured the
    // end-to-end deviation. See report 31 §3.
    const MODEL_STD: f32 = 1.1012;
    const MODEL_ABSMAX: f32 = 10.25;
    const SCALE: f32 = 1.28;
    const GAIN: f32 = 8.0;
    const ONE_IN: u64 = 52;

    let mut failures: Vec<String> = Vec::new();
    let mut total_mismatch = 0usize;
    let mut compared = 0usize;
    for (index, seed) in VISION_TAIL_SEEDS.into_iter().enumerate() {
        let inputs = vision_inputs_from(&ctx, seed, SCALE, GAIN, ONE_IN);
        // The stimulus census, on the stimulus itself this time: both moments
        // and the tail must land near the capture, or this test is measuring
        // something other than the model.
        let count = inputs.host_qkv.len() as f64;
        let mean = inputs.host_qkv.iter().map(|v| f64::from(v.to_f32())).sum::<f64>() / count;
        let variance = inputs
            .host_qkv
            .iter()
            .map(|v| (f64::from(v.to_f32()) - mean).powi(2))
            .sum::<f64>()
            / count;
        let std = variance.sqrt() as f32;
        let absmax = inputs
            .host_qkv
            .iter()
            .fold(0f32, |acc, v| acc.max(v.to_f32().abs()));
        if index == 0 {
            println!(
                "  stimulus: std {std:.4} (model {MODEL_STD}), absmax {absmax:.4} \
                 (model {MODEL_ABSMAX}), |max|/sigma {:.2} (model 9.31, uniform 1.73)",
                absmax / std
            );
            assert!(
                (std / MODEL_STD - 1.0).abs() < 0.15 && (absmax / MODEL_ABSMAX - 1.0).abs() < 0.15,
                "the stimulus no longer matches the captured model QKV moments \
                 (std {std} vs {MODEL_STD}, absmax {absmax} vs {MODEL_ABSMAX}); \
                 recalibrate SCALE/GAIN/ONE_IN against a fresh capture rather than \
                 letting the gate drift off the model's regime"
            );
        }
        let aot = run_vision(&ctx, &inputs, Fa4AttentionImpl::Aot);
        let cuda = run_vision(&ctx, &inputs, Fa4AttentionImpl::Cuda);
        let aot_again = run_vision(&ctx, &inputs, Fa4AttentionImpl::Aot);
        assert_eq!(
            aot, aot_again,
            "the vision AOT arm is not reproducible on stimulus {index}"
        );
        let equal = equal_count(&aot, &cuda);
        compared += aot.len();
        total_mismatch += aot.len() - equal;
        let (worst, element) = max_ulp(&aot, &cuda);
        let census = vision_group_census(&aot, &cuda);
        println!(
            "  stimulus {index} (seed {seed:#018x}): {}/{} differ, worst {worst} BF16 ULP, \
             per-group differing {:?}",
            aot.len() - equal,
            aot.len(),
            census.map(|(e, t, _)| t - e),
        );
        if equal != aot.len() {
            let row = VISION_HEADS * VISION_HEAD_DIM;
            let token = element / row;
            failures.push(format!(
                "stimulus {index} (seed {seed:#018x}): {} of {} elements differ, worst \
                 {worst} BF16 ULP at element {element} = token {token} (within-batch {}, \
                 fixed group {}), head {}, dim {}; aot {:#06x} cuda {:#06x}",
                aot.len() - equal,
                aot.len(),
                token % 4072,
                [0usize, 624, 1248, 1872]
                    .iter()
                    .rposition(|start| token % 4072 >= *start)
                    .unwrap(),
                (element / VISION_HEAD_DIM) % VISION_HEADS,
                element % VISION_HEAD_DIM,
                aot[element],
                cuda[element],
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "vision attention is NOT bit-identical between the AOT and in-tree arms on a \
         stimulus with the model's own moments AND tail: {total_mismatch} of {compared} \
         elements differ over {} independent streams.\n  {}\n\
         The `+-0.05` gate and the `VISION_SCALES` sweep both read \
         12509184/12509184 absmax 0 on the SAME call path, the SAME 12 segments and the \
         SAME two AOT objects, because a uniform stream has |max|/sigma = 1.73 at every \
         scale and never reaches the score configuration that separates the arms. \
         Report 31.",
        VISION_TAIL_SEEDS.len(),
        failures.join("\n  ")
    );
}

/// One-off localiser, `#[ignore]`d because it needs captured tensors that are
/// too large to track. Point it at the dumps report 31 §3 took out of the
/// running model and it drives the identical repo path on the identical data:
///
/// ```sh
/// APXINF_R31_QKV=/path/to/vis01qkv.bf16 \
/// APXINF_R31_POS=/path/to/pos_ids-s3.u32 \
///   cargo test ... -- --ignored --nocapture vision_attention_on_a_captured_model_tensor
/// ```
#[test]
#[ignore = "needs captured model tensors; see report 31 S3"]
fn vision_attention_on_a_captured_model_tensor() {
    if !both_arms(Fa4Operator::Vision, "aot-vision-attention") {
        return;
    }
    let Some(ctx) = context() else { return };
    let qkv_path = std::env::var("APXINF_R31_QKV").expect("set APXINF_R31_QKV");
    let pos_path = std::env::var("APXINF_R31_POS").expect("set APXINF_R31_POS");
    let width = 3 * VISION_HEADS * VISION_HEAD_DIM;
    let raw = std::fs::read(&qkv_path).unwrap();
    assert_eq!(
        raw.len(),
        VISION_TOKENS * width * 2,
        "{qkv_path} is not a [12216, 3072] BF16 bit-pattern dump"
    );
    let host_qkv: Vec<bf16> = raw
        .chunks_exact(2)
        .map(|pair| bf16::from_bits(u16::from_le_bytes([pair[0], pair[1]])))
        .collect();
    let pos_raw = std::fs::read(&pos_path).unwrap();
    assert_eq!(pos_raw.len(), VISION_TOKENS * 2 * 4, "{pos_path} is not 12216 (h,w) u32 pairs");
    let position_ids = CudaBuffer::alloc(pos_raw.len(), ctx.device_id()).unwrap();
    position_ids.copy_from_host(&pos_raw).unwrap();
    let offsets_i32: Vec<i32> = VISION_OFFSETS.iter().map(|value| *value as i32).collect();
    let offsets = CudaBuffer::alloc(offsets_i32.len() * 4, ctx.device_id()).unwrap();
    offsets
        .copy_from_host(bytemuck::cast_slice(&offsets_i32))
        .unwrap();
    let inputs = VisionInputs {
        qkv: upload_bf16(vec![VISION_TOKENS, width], &host_qkv),
        host_qkv,
        position_ids,
        offsets,
    };
    let aot = run_vision(&ctx, &inputs, Fa4AttentionImpl::Aot);
    let cuda = run_vision(&ctx, &inputs, Fa4AttentionImpl::Cuda);
    let equal = report(&format!("captured {qkv_path}"), &aot, &cuda);
    let census = vision_group_census(&aot, &cuda);
    for (group, (equal_in_group, total, worst)) in census.iter().enumerate() {
        println!(
            "      group {group} (L={}): {equal_in_group}/{total} bit-equal, max {worst} ULP",
            if group == 3 { 2200 } else { 624 }
        );
    }
    let row = VISION_HEADS * VISION_HEAD_DIM;
    for (element, (a, b)) in aot.iter().zip(&cuda).enumerate() {
        if a != b {
            println!(
                "      element {element}: token {} (within-batch {}), head {}, dim {}, \
                 aot {a:#06x} cuda {b:#06x}, {} ULP",
                element / row,
                (element / row) % 4072,
                (element / VISION_HEAD_DIM) % VISION_HEADS,
                element % VISION_HEAD_DIM,
                (ordered(*a) - ordered(*b)).abs()
            );
        }
    }
    assert_eq!(equal, aot.len(), "the captured model tensor separates the arms");
}

// ---------------------------------------------------------------------------
// language-attention (hd256 causal, one 3387-token sequence)
// ---------------------------------------------------------------------------
#[test]
fn aot_and_cuda_language_attention_match_their_known_precision() {
    if !both_arms(Fa4Operator::Language, "aot-language-attention") {
        return;
    }
    let Some(ctx) = context() else { return };
    let q = upload_bf16(
        vec![SCENE_TOKENS, 16, 256],
        &lcg_bf16(0x1234_5678_9abc_def0, SCENE_TOKENS * 16 * 256, 0.05),
    );
    let k = upload_bf16(
        vec![SCENE_TOKENS, 4, 256],
        &lcg_bf16(0x0f1e_2d3c_4b5a_6978, SCENE_TOKENS * 4 * 256, 0.05),
    );
    let v = upload_bf16(
        vec![SCENE_TOKENS, 4, 256],
        &lcg_bf16(0xfeed_face_cafe_b0ba, SCENE_TOKENS * 4 * 256, 0.05),
    );

    let run = |implementation| {
        let output =
            attn::gqa_bf16_fa4_d256_sm110_with(&ctx, &q, &k, &v, SCENE_TOKENS, implementation)
                .unwrap()
                .expect("the language route rejected its own geometry");
        ctx.synchronize().unwrap();
        bits(&output)
    };
    let cuda = run(Fa4AttentionImpl::Cuda);
    // Run CUDA first so a partial write cannot inherit a fully populated AOT
    // buffer from the allocator's device-memory cache and produce a false
    // bit-identical result.
    let aot = run(Fa4AttentionImpl::Aot);
    let aot_again = run(Fa4AttentionImpl::Aot);
    assert_eq!(aot, aot_again, "the language AOT arm is not reproducible");
    assert!(
        aot.iter().any(|value| *value != 0),
        "both arms wrote nothing: the comparison would be vacuous"
    );

    let equal = report("language AOT == CUDA", &aot, &cuda);
    let rows = rows_with_mismatch(&aot, &cuda, 16 * 256);
    println!(
        "  language rows with any mismatch: {rows} of {}",
        aot.len() / (16 * 256)
    );
    // The 33 known +-1-ULP elements are an OPEN item (report 21 S21.6 has the
    // live `kLen`-sweep lead). They are stable, so the gate is written as a
    // bound on the same shape of residual rather than as equality: every
    // mismatch within one BF16 ULP of ORDERED integer distance -- not of raw
    // `u16` difference, which would score a +0/-0 or a sign flip on a tiny
    // element as 32,768 or as 1 depending on which side of zero it lands --
    // and essentially all elements equal.
    let mismatches = aot.len() - equal;
    assert!(
        mismatches * 100_000 <= aot.len(),
        "language attention: {mismatches} mismatching elements of {} is worse than the \
         known 33; the operator's numeric contract has moved",
        aot.len()
    );
    assert_within_ulp("language attention", &aot, &cuda, 1);
}

/// The same stimulus-regime sweep for language. The 31 known +-1-ULP elements
/// were censused at `+-0.05`, where a head_dim-256 BF16 dot product gives
/// `|score| ~ 8e-4` and the causal softmax is uniform over the prefix to within
/// 0.4 % -- so "31 of 13 873 152, every one exactly 1 ULP" is a statement about
/// a plain prefix mean, not about attention. This sweep is what decides whether
/// that bound survives a real activation magnitude.
#[test]
fn aot_and_cuda_language_attention_hold_their_ulp_bound_across_input_regimes() {
    if !both_arms(Fa4Operator::Language, "aot-language-attention") {
        return;
    }
    let Some(ctx) = context() else { return };
    let mut failures: Vec<String> = Vec::new();
    for scale in VISION_SCALES {
        let q = upload_bf16(
            vec![SCENE_TOKENS, 16, 256],
            &lcg_bf16(0x1234_5678_9abc_def0, SCENE_TOKENS * 16 * 256, scale),
        );
        let k = upload_bf16(
            vec![SCENE_TOKENS, 4, 256],
            &lcg_bf16(0x0f1e_2d3c_4b5a_6978, SCENE_TOKENS * 4 * 256, scale),
        );
        let v = upload_bf16(
            vec![SCENE_TOKENS, 4, 256],
            &lcg_bf16(0xfeed_face_cafe_b0ba, SCENE_TOKENS * 4 * 256, scale),
        );
        let run = |implementation| {
            let output =
                attn::gqa_bf16_fa4_d256_sm110_with(&ctx, &q, &k, &v, SCENE_TOKENS, implementation)
                    .unwrap()
                    .expect("the language route rejected its own geometry");
            ctx.synchronize().unwrap();
            bits(&output)
        };
        let aot = run(Fa4AttentionImpl::Aot);
        let cuda = run(Fa4AttentionImpl::Cuda);
        let aot_again = run(Fa4AttentionImpl::Aot);
        assert_eq!(
            aot, aot_again,
            "the language AOT arm is not reproducible at input scale {scale}"
        );
        let equal = report(&format!("language scale +-{scale}"), &aot, &cuda);
        let (worst, index) = max_ulp(&aot, &cuda);
        let rows = rows_with_mismatch(&aot, &cuda, 16 * 256);
        println!(
            "      {} mismatching elements, {rows} of {} rows, worst {worst} BF16 ULP",
            aot.len() - equal,
            aot.len() / (16 * 256)
        );
        if worst > 1 || (aot.len() - equal) * 100_000 > aot.len() {
            failures.push(format!(
                "scale +-{scale}: {}/{} elements differ, {rows} rows, worst {worst} BF16 ULP \
                 at element {index}",
                aot.len() - equal,
                aot.len()
            ));
        }
    }
    assert!(
        failures.is_empty(),
        "language attention's `<=1 ULP on <=1e-5 of elements` bound does not survive a \
         non-degenerate input magnitude:\n  {}\n\
         The `+-0.05` census (31 elements, all 1 ULP) is taken where the causal softmax is \
         a uniform prefix mean. See report 31.",
        failures.join("\n  ")
    );
}

/// 🔴 **The language half of report 31's gate.** It fails on the current tree,
/// and it is a one-line change of intent rather than of stimulus: the test
/// above *censuses* the 31-element 1-ULP residual and tolerates it; this one
/// asserts it away.
///
/// The sweep above shows the census is honest — 25 to 40 elements, never more
/// than 1 ULP, at every input regime from `+-0.05` to `+-2.0` — and the
/// in-model capture agrees: at full-attention layer 3 of the real fixture,
/// `only-language` differs from `cuda` in **24 of 13 873 152 elements, all
/// exactly 1 BF16 ULP, in 24 of 3387 rows** (report 31 §4). So the language
/// gate was never blind. It simply declared the residual acceptable.
///
/// It is not acceptable, and report 31 §1 is why: 24 one-ULP flips at layer 3
/// of 32 reach the model's output as **2 BF16 ULP of a waypoint**, which the
/// `trajectory_scale[0] = 165` denormalisation turns into the 0.322 m report 29
/// measured. The vision operator independently supplies the control: **one**
/// element of 12 509 184 differing by **one** ULP at vision block 1 produces
/// the same 0.322 m. A 1-ULP tolerance on an attention output inside a
/// 32-layer stack feeding a 10-step sampler is not a tolerance; it is the
/// whole defect.
#[test]
fn aot_and_cuda_language_attention_are_bit_identical() {
    if !both_arms(Fa4Operator::Language, "aot-language-attention") {
        return;
    }
    let Some(ctx) = context() else { return };
    let scene_tokens = std::env::var("APX_LANGUAGE_TEST_TOKENS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(SCENE_TOKENS);
    let q = upload_bf16(
        vec![scene_tokens, 16, 256],
        &lcg_bf16(0x1234_5678_9abc_def0, scene_tokens * 16 * 256, 0.05),
    );
    let k = upload_bf16(
        vec![scene_tokens, 4, 256],
        &lcg_bf16(0x0f1e_2d3c_4b5a_6978, scene_tokens * 4 * 256, 0.05),
    );
    let v = upload_bf16(
        vec![scene_tokens, 4, 256],
        &lcg_bf16(0xfeed_face_cafe_b0ba, scene_tokens * 4 * 256, 0.05),
    );
    let run = |implementation| {
        let output =
            attn::gqa_bf16_fa4_d256_sm110_with(&ctx, &q, &k, &v, scene_tokens, implementation)
                .unwrap()
                .expect("the language route rejected its own geometry");
        ctx.synchronize().unwrap();
        bits(&output)
    };
    if scene_tokens != SCENE_TOKENS {
        let cuda = run(Fa4AttentionImpl::Cuda);
        if std::env::var_os("APX_DUMP_LANGUAGE").is_some() {
            std::fs::write("/tmp/apx-language-cuda.bf16", bytemuck::cast_slice(&cuda)).unwrap();
        }
        assert!(cuda.iter().any(|value| *value != 0));
        return;
    }
    let cuda = run(Fa4AttentionImpl::Cuda);
    // Keep this ordering: CUDA must populate a fresh/reused output before the
    // AOT arm can leave the allocator cache containing the expected result.
    let aot = run(Fa4AttentionImpl::Aot);
    if let Some(repeats) = std::env::var("APX_LANGUAGE_PERF_REPEATS")
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
    {
        for i in 0..repeats {
            if i % 2 == 0 {
                drop(run(Fa4AttentionImpl::Aot));
                drop(run(Fa4AttentionImpl::Cuda));
            } else {
                drop(run(Fa4AttentionImpl::Cuda));
                drop(run(Fa4AttentionImpl::Aot));
            }
        }
    }
    if std::env::var_os("APX_DUMP_LANGUAGE").is_some() {
        std::fs::write("/tmp/apx-language-aot.bf16", bytemuck::cast_slice(&aot)).unwrap();
        std::fs::write("/tmp/apx-language-cuda.bf16", bytemuck::cast_slice(&cuda)).unwrap();
    }
    let equal = report("language AOT == CUDA (strict)", &aot, &cuda);
    let rows = rows_with_mismatch(&aot, &cuda, 16 * 256);
    assert_eq!(
        equal,
        aot.len(),
        "language attention is not bit-identical between the arms: {} of {} elements \
         differ in {rows} of {} rows. Every one is 1 BF16 ULP and that is exactly the \
         problem -- 24 such flips at full-attention layer 3 move the Qwen-Drive \
         trajectory by 0.322 m (= 2 BF16 ULP of one waypoint x trajectory_scale 165). \
         Report 21 S21.6's `kLen` sweep is the open lead on the cause. Report 31.",
        aot.len() - equal,
        aot.len(),
        aot.len() / (16 * 256),
    );
}

// ---------------------------------------------------------------------------
// action-attention (hd256 non-causal, packed GQA over a split-batch KV view)
// ---------------------------------------------------------------------------

#[test]
fn aot_and_cuda_action_attention_are_bit_identical() {
    if !both_arms(Fa4Operator::Action, "aot-action-attention") {
        return;
    }
    let Some(ctx) = context() else { return };
    let q = upload_bf16(
        vec![50, 16, 256],
        &lcg_bf16(0xa5a5_5a5a_3c3c_c3c3, 50 * 16 * 256, 0.05),
    );
    let k = upload_bf16(
        vec![3438, 4, 256],
        &lcg_bf16(0x1f2e_3d4c_5b6a_7988, 3438 * 4 * 256, 0.05),
    );
    let v = upload_bf16(
        vec![3438, 4, 256],
        &lcg_bf16(0x8899_aabb_ccdd_eeff, 3438 * 4 * 256, 0.05),
    );

    let run = |key_tokens, implementation| {
        let outputs = attn::gqa_bf16_fa4_d256_splitbatch_sm110_with(
            &ctx, &q, &k, &v, key_tokens, implementation,
        )
        .unwrap()
        .expect("the action route rejected its own geometry");
        ctx.synchronize().unwrap();
        assert_eq!(outputs.output.shape().dims(), [50, 16, 256]);
        assert_eq!(outputs.partial.shape().dims(), [2, 50, 16, 256]);
        assert_eq!(outputs.lse.shape().dims(), [2, 16, 50]);
        (
            bits(&outputs.partial),
            bits(&outputs.output),
            f32_bits(&outputs.lse),
        )
    };

    // Each `key_tokens` is its own shape: it changes `seqused_k` for the
    // SECOND batch only, so a mismatched pairing is a real, different problem
    // and not a relabelling. 3436 is the one whose two batches have EXACTLY
    // equal KV length, and it is the shape that matters -- see the module
    // comment.
    for key_tokens in 3433..=3437 {
        let (aot_partial, aot_merged, aot_lse) = run(key_tokens, Fa4AttentionImpl::Aot);
        let (cuda_partial, cuda_merged, cuda_lse) = run(key_tokens, Fa4AttentionImpl::Cuda);
        let (aot_partial_again, aot_merged_again, aot_lse_again) =
            run(key_tokens, Fa4AttentionImpl::Aot);
        assert_eq!(
            (&aot_partial, &aot_merged, &aot_lse),
            (&aot_partial_again, &aot_merged_again, &aot_lse_again),
            "the action AOT arm is not reproducible at key_tokens {key_tokens}"
        );
        assert!(
            aot_partial.iter().any(|value| *value != 0),
            "both arms wrote nothing: the comparison would be vacuous"
        );
        println!(
            "  key_tokens {key_tokens} (seqused_k = {{1718, {}}}):",
            key_tokens - 1718
        );

        // 1. the attention launch, which was the only asserted surface before
        //    report 27.
        let equal = report("    partial (pre-merge)", &aot_partial, &cuda_partial);
        assert_eq!(
            equal,
            aot_partial.len(),
            "action attention's pre-merge output is not bit-identical between the arms \
             at key_tokens {key_tokens}"
        );

        // 2. the LSE, the actual cause surface. The merge is the SAME kernel in
        //    both arms, so the merged output can only differ if this does, and
        //    an LSE assertion localises a future regression in one line
        //    instead of leaving it to be inferred from the output.
        let lse_equal = equal_count_u32(&aot_lse, &cuda_lse);
        println!(
            "    lse: {lse_equal}/{} bit-equal, max {} fp32 ULP",
            aot_lse.len(),
            max_ulp_u32(&aot_lse, &cuda_lse)
        );
        assert_eq!(
            lse_equal,
            aot_lse.len(),
            "action attention's LSE is not bit-identical between the arms at \
             key_tokens {key_tokens}; the epilogue must compute FA4's \
             `(row_max*scale_log2 + lg2.approx(row_sum)) * ln2` \
             (-DAPXINF_LSE_FA4=1), not `cutlass::fast_log`"
        );

        // 3. the MERGED output, by integer ULP. Asserting `absmax` here is what
        //    let report 26's 3436 defect through: a sign flip on a 5e-4
        //    element is 12,112 ULP and 1.9e-10 of absolute error.
        let merged_equal = report("    merged", &aot_merged, &cuda_merged);
        assert_within_ulp(
            &format!("action merged output at key_tokens {key_tokens}"),
            &aot_merged,
            &cuda_merged,
            0,
        );
        assert_eq!(
            merged_equal,
            aot_merged.len(),
            "action attention's merged output is not bit-identical between the arms at \
             key_tokens {key_tokens}"
        );
    }
}

/// Replays a Qwen-Drive production call captured with
/// `APXINF_DEBUG_CAPTURE_ACTION`. The fixture is intentionally kept outside
/// the repository because K and V are about 6.8 MiB each; setting
/// `APX_ACTION_CAPTURE_DIR` opts into this model-shaped regression gate.
#[test]
fn captured_qwen_drive_action_attention_is_bit_identical() {
    let Some(directory) = std::env::var_os("APX_ACTION_CAPTURE_DIR") else {
        eprintln!("skipping: APX_ACTION_CAPTURE_DIR is not set");
        return;
    };
    if !both_arms(Fa4Operator::Action, "aot-action-attention") {
        return;
    }
    let Some(ctx) = context() else { return };
    let directory = std::path::PathBuf::from(directory);
    let read_bf16 = |name: &str| {
        let bytes = std::fs::read(directory.join(name)).unwrap();
        assert_eq!(bytes.len() % 2, 0, "{name} has an odd byte count");
        bytes
            .chunks_exact(2)
            .map(|pair| bf16::from_bits(u16::from_le_bytes([pair[0], pair[1]])))
            .collect::<Vec<_>>()
    };
    let metadata = std::fs::read_to_string(directory.join("metadata.txt")).unwrap();
    let key_tokens = metadata
        .trim()
        .strip_prefix("key_tokens=")
        .unwrap()
        .parse::<usize>()
        .unwrap();
    let q = upload_bf16(vec![50, 16, 256], &read_bf16("q.bf16"));
    let k = upload_bf16(vec![3438, 4, 256], &read_bf16("k.bf16"));
    let v = upload_bf16(vec![3438, 4, 256], &read_bf16("v.bf16"));
    let run = |implementation| {
        let outputs = attn::gqa_bf16_fa4_d256_splitbatch_sm110_with(
            &ctx,
            &q,
            &k,
            &v,
            key_tokens,
            implementation,
        )
        .unwrap()
        .expect("the captured action geometry was rejected");
        ctx.synchronize().unwrap();
        (
            bits(&outputs.partial),
            bits(&outputs.output),
            f32_bits(&outputs.lse),
        )
    };
    let (aot_partial, aot_merged, aot_lse) = run(Fa4AttentionImpl::Aot);
    let (cuda_partial, cuda_merged, cuda_lse) = run(Fa4AttentionImpl::Cuda);
    let partial_equal = report("captured action partial", &aot_partial, &cuda_partial);
    let merged_equal = report("captured action merged", &aot_merged, &cuda_merged);
    let lse_equal = equal_count_u32(&aot_lse, &cuda_lse);
    println!(
        "  captured action lse: {lse_equal}/{} bit-equal, max {} fp32 ULP",
        aot_lse.len(),
        max_ulp_u32(&aot_lse, &cuda_lse)
    );
    assert_eq!(partial_equal, aot_partial.len());
    assert_eq!(merged_equal, aot_merged.len());
    assert_eq!(lse_equal, aot_lse.len());
}

// ---------------------------------------------------------------------------
// Properties that hold in a single-arm build too, so arm A exercises them.
// ---------------------------------------------------------------------------

#[test]
fn the_linked_implementations_run_and_are_reproducible() {
    let Some(ctx) = context() else { return };
    let Some(language) = attn::language_attention_impl(&ctx) else {
        eprintln!("skipping: no language-attention implementation is linked");
        return;
    };
    let q = upload_bf16(
        vec![SCENE_TOKENS, 16, 256],
        &lcg_bf16(0x5555_aaaa_3333_cccc, SCENE_TOKENS * 16 * 256, 0.05),
    );
    let k = upload_bf16(
        vec![SCENE_TOKENS, 4, 256],
        &lcg_bf16(0x2222_7777_dddd_4444, SCENE_TOKENS * 4 * 256, 0.05),
    );
    let v = upload_bf16(
        vec![SCENE_TOKENS, 4, 256],
        &lcg_bf16(0x6666_1111_9999_eeee, SCENE_TOKENS * 4 * 256, 0.05),
    );
    let first =
        attn::gqa_bf16_fa4_d256_sm110_with(&ctx, &q, &k, &v, SCENE_TOKENS, language).unwrap();
    let second =
        attn::gqa_bf16_fa4_d256_sm110_with(&ctx, &q, &k, &v, SCENE_TOKENS, language).unwrap();
    ctx.synchronize().unwrap();
    let first = bits(&first.unwrap());
    let second = bits(&second.unwrap());
    assert!(first.iter().any(|value| *value != 0), "nothing was written");
    assert_eq!(first, second, "{language:?} language attention is not reproducible");
}

#[test]
fn the_action_partial_is_a_distinct_buffer_from_the_merged_output() {
    // Same element count would be a same-size-different-layout hazard; these
    // two differ by a factor of two, and the shapes say so. Pinned because the
    // merge reads `partial` as two batches and writes one.
    let merged = Shape::new(vec![50, 16, 256]);
    let partial = Shape::new(vec![2, 50, 16, 256]);
    assert_eq!(partial.numel(), 2 * merged.numel());
    assert_eq!(DType::BF16.size_in_bytes(), 2);
}
