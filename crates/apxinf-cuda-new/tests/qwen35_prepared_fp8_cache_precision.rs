//! Prepared BF16 -> static FP8 cache quantization only: no QKNorm, RoPE,
//! attention arithmetic, model projection or FP8 quality claim is included.

#[path = "support/e4m3_nearest_reference.rs"]
mod reference;

use apxinf_core::{DType, Shape, Tensor};
use apxinf_cuda_new::{ops, CudaBuffer, CudaContext};
use reference::{bf16_value, Oracle};

const KV_WIDTH: usize = 2 * 256;
const PAGE: usize = 128;

fn upload(ctx: &CudaContext, bytes: &[u8], shape: Vec<usize>, dtype: DType) -> Tensor {
    let buffer = CudaBuffer::alloc(bytes.len(), ctx.device_id()).unwrap();
    buffer.copy_from_host(bytes).unwrap();
    buffer.as_tensor(Shape::new(shape), dtype).unwrap()
}

fn read(tensor: &Tensor) -> Vec<u8> {
    let b = CudaBuffer::from_tensor(tensor).unwrap();
    let mut bytes = vec![0; b.len()];
    b.copy_to_host(&mut bytes).unwrap();
    bytes
}

fn view(tensor: &Tensor, byte_start: usize, bytes: usize, shape: Vec<usize>, dtype: DType) -> Tensor {
    CudaBuffer::from_tensor(tensor).unwrap().view(byte_start, bytes).unwrap()
        .as_tensor(Shape::new(shape), dtype).unwrap()
}

fn assert_bytes(label: &str, got: &[u8], want: &[u8]) {
    assert_eq!(got.len(), want.len());
    if let Some(i) = got.iter().zip(want).position(|(a, b)| a != b) {
        let count = got.iter().zip(want).filter(|(a, b)| a != b).count();
        panic!("{label}: {count} byte errors; first at {i}: actual={:02x} expected={:02x}", got[i], want[i]);
    }
}

fn quantize(ctx: &CudaContext, source: &Tensor, destination: &mut Tensor, scale: f32) {
    let mut args = ops::QuantizationArgs::new(ops::QuantizationSemantic::FixedScaleE4m3, source, destination);
    args.scale = scale;
    ops::quantization(ctx, args).unwrap();
    ctx.synchronize().unwrap();
}

fn finite_domain() -> Vec<u16> {
    (0..=u16::MAX).filter(|x| x & 0x7f80 != 0x7f80).collect()
}

fn domain_case(ctx: &CudaContext, oracle: &Oracle, bits: &[u16], scale: f32, power2: bool) {
    let input_bytes: Vec<u8> = bits.iter().flat_map(|x| x.to_le_bytes()).collect();
    let input = upload(ctx, &input_bytes, vec![bits.len()], DType::BF16);
    // Offset17 also tests an unaligned FP8 output binding and short tails.
    let guard = 17;
    let mut expected = vec![0x55; bits.len() + guard * 2];
    let storage = upload(ctx, &expected, vec![expected.len()], DType::F8E4M3);
    let mut output = view(&storage, guard, bits.len(), vec![bits.len()], DType::F8E4M3);
    // Drain upload completion before the test. This is precision validation,
    // not a benchmark of host/device transfer or an unsequenced producer.
    assert_bytes("prepared source bytes", &read(&input), &input_bytes);
    let mut differences = Vec::new();
    for (i, &bf16) in bits.iter().enumerate() {
        let stage = oracle.stages(bf16_value(bf16), scale);
        if power2 { assert_eq!(stage.reciprocal_product_code, stage.mathematical_code); }
        if differences.len() < 8 && (stage.reciprocal_product_code != stage.mathematical_code || stage.reciprocal_product_code != stage.direct_division_code) {
            differences.push((bf16, stage));
        }
        expected[guard + i] = stage.reciprocal_product_code;
    }
    quantize(ctx, &input, &mut output, scale);
    assert_bytes("FP8 quantization and canaries", &read(&storage), &expected);
    assert_bytes("source unchanged", &read(&input), &input_bytes);
    println!("prepared-fp8 scale={scale:?} scale_bits={:08x} count={} oracle=enumerated-RNE expected_stage=host-F32-reciprocal/device-F32-product power2={power2} stage_boundary_examples={differences:?} byte_exact=PASS", scale.to_bits(), bits.len());
}

#[test]
fn prepared_bf16_to_fp8_power2_is_byte_exact_over_finite_domain() {
    let ctx = CudaContext::new_with_autotune(0, false).unwrap();
    let o = Oracle::new();
    let full = finite_domain();
    assert_eq!(full.len(), 65280);
    // Zero/subnormal/normal transition, midpoints, finite overflow and signs;
    // the exhaustive set supplies all other representable BF16 boundaries.
    let tail = [0, 0x8000, 0x3a80, 0xba80, 0x3b00, 0xbb00, 0x3c70,
                0xbc70, 0x3f88, 0xbf88, 0x43d8, 0x7f7f, 0xff7f];
    for scale in [0.25, 0.5, 1.0, 2.0, 4.0] {
        domain_case(&ctx, &o, &full, scale, true);
        domain_case(&ctx, &o, &tail, scale, true);
    }
}

#[test]
fn prepared_bf16_to_fp8_nonpower_scales_require_exact_reciprocal_stage_bytes() {
    let ctx = CudaContext::new_with_autotune(0, false).unwrap();
    let o = Oracle::new();
    let full = finite_domain();
    // Deliberately synthetic stress scales, not purported checkpoint scales.
    // GPU must exactly match the actual reciprocal/product stage; a boundary
    // difference from ideal division is classified, never silently tolerated.
    for scale in [0.1, 0.3, 0.11216518, 0.0322265625] {
        domain_case(&ctx, &o, &full, scale, false);
    }
}

fn append_case(ctx: &CudaContext, prefix: usize, append: usize) {
    let capacity = (prefix + append + 1).div_ceil(PAGE) * PAGE;
    let total = capacity * KV_WIDTH;
    let o = Oracle::new();
    // A repeating full BF16 domain creates different values on every K/V
    // channel and across pages, including signed zeros and saturation cases.
    let full = finite_domain();
    // Current deployment's unit K/V defaults; other power2 scales and
    // synthetic nonpower2 scales are covered independently above.
    for (kind, scale, shift) in [("K", 1.0f32, 0usize), ("V", 1.0f32, 7919usize)] {
        let bits: Vec<u16> = (0..append * KV_WIDTH).map(|i| full[(i * 13 + shift) % full.len()]).collect();
        let source: Vec<u8> = bits.iter().flat_map(|x| x.to_le_bytes()).collect();
        let input = upload(ctx, &source, vec![append, KV_WIDTH], DType::BF16);
        let mut expected: Vec<u8> = (0..total).map(|i| if i < prefix * KV_WIDTH { 0x28 + (i % 7) as u8 } else { 0x55 }).collect();
        let storage = upload(ctx, &expected, vec![capacity / PAGE, PAGE, 2, 256], DType::F8E4M3);
        let mut destination = view(&storage, prefix * KV_WIDTH, append * KV_WIDTH,
                                   vec![append, KV_WIDTH], DType::F8E4M3);
        assert_bytes("prepared cache source", &read(&input), &source);
        for (i, &x) in bits.iter().enumerate() {
            let stage = o.stages(bf16_value(x), scale);
            assert_eq!(stage.reciprocal_product_code, stage.mathematical_code);
            expected[prefix * KV_WIDTH + i] = stage.mathematical_code;
        }
        quantize(ctx, &input, &mut destination, scale);
        assert_bytes("append bytes / prior prefix / future canary", &read(&storage), &expected);
        assert_bytes("append source unchanged", &read(&input), &source);
        println!("prepared-fp8-cache kind={kind} prefix={prefix} append={append} capacity={capacity} page=128 byte_exact=PASS");
    }
}

#[test]
fn prepared_fp8_cache_append_preserves_prefix_and_future_at_page_boundaries() {
    let ctx = CudaContext::new_with_autotune(0, false).unwrap();
    for (prefix, append) in [(0, 1), (127, 1), (128, 1), (128, 2), (129, 3), (127, 4), (4095, 4)] {
        append_case(&ctx, prefix, append);
    }
}

#[test]
#[ignore = "48K prefix / 4K append storage-only precision gate; main task owns GPU"]
fn prepared_fp8_cache_append_preserves_long_prefix_and_future() {
    let ctx = CudaContext::new_with_autotune(0, false).unwrap();
    for prefix in [4096, 8192, 16384, 32768, 49152] { append_case(&ctx, prefix, 4096); }
}
