#![cfg(feature = "qwen35-cuda")]

use std::path::PathBuf;
use std::time::Instant;

use apxinf_core::{DType, Shape, Tensor};
use apxinf_cuda_new::{
    ops, transfers, CudaBuffer, CudaContext, EdgeNvfp4MoeArgs, EdgeNvfp4MoeWeights,
    RoutedMoeConfig,
};
use apxinf_model::qwen35moe::config::Qwen35MoeConfig;
use apxinf_model::qwen35moe::device_weights::{
    load_qwen35_moe_layer_from_path, DeviceEdgeNvfp4Experts, DeviceNvfp4RoutedExperts,
};
use apxinf_model::qwen35moe::forward::BaseMoeScratch;
use half::bf16;

const TOKENS_ENV: &str = "APXINF_QWEN35_MOE_BENCH_TOKENS";
const ITERATIONS_ENV: &str = "APXINF_QWEN35_MOE_BENCH_ITERS";
const PROBE_TOKENS_ENV: &str = "APXINF_QWEN35_MOE_PROBE_TOKENS";
const PROBE_LAYER_ENV: &str = "APXINF_QWEN35_MOE_PROBE_LAYER";
const PROBE_REPEATS: usize = 20;

fn positive_env(name: &str, default: usize) -> usize {
    std::env::var(name)
        .ok()
        .map(|value| {
            value
                .parse::<usize>()
                .unwrap_or_else(|_| panic!("{name} must be a positive integer"))
        })
        .unwrap_or(default)
        .max(1)
}

fn checkpoint() -> PathBuf {
    std::env::var_os("APXINF_QWEN35_CHECKPOINT")
        .map(PathBuf::from)
        .expect("set APXINF_QWEN35_CHECKPOINT to Qwen3.5-35B-A3B-NVFP4")
}

#[test]
#[ignore = "benchmark; loads the complete Qwen3.5-35B-A3B NVFP4 checkpoint"]
fn real_checkpoint_first_moe_layer() -> Result<(), Box<dyn std::error::Error>> {
    let checkpoint = checkpoint();
    let config = Qwen35MoeConfig::from_json_file(&checkpoint.join("config.json"))?;
    let ctx = CudaContext::new(0)?;
    let load_started = Instant::now();
    let moe = load_qwen35_moe_layer_from_path(&ctx, &config, &checkpoint, 0)?;
    ctx.synchronize()?;
    eprintln!("Qwen3.5 layer-0 MoE load: {:?}", load_started.elapsed());

    assert!(
        matches!(&moe.routed_experts, DeviceNvfp4RoutedExperts::Edge(_)),
        "real checkpoint did not select the Edge NVFP4 routed-expert provider"
    );
    let tokens = positive_env(TOKENS_ENV, 1024);
    let iterations = positive_env(ITERATIONS_ENV, 10);
    let hidden = config.text_config.hidden_size;
    let input_values = (0..tokens * hidden)
        .map(|index| bf16::from_f32(((index % 31) as f32 - 15.0) / 64.0))
        .collect::<Vec<_>>();
    let input = transfers::to_cuda(
        &Tensor::from_bf16(vec![tokens, hidden], &input_values)?,
        ctx.device_id(),
    )?;
    let mut scratch = BaseMoeScratch::new(
        &ctx,
        tokens,
        hidden,
        config.text_config.num_experts,
        config.text_config.num_experts_per_tok,
        config.text_config.moe_intermediate_size,
    )?;

    scratch.run(&ctx, &moe, &input)?;
    ctx.synchronize()?;
    let started = Instant::now();
    for _ in 0..iterations {
        scratch.run(&ctx, &moe, &input)?;
    }
    ctx.synchronize()?;
    let elapsed = started.elapsed();
    let per_iteration = elapsed / iterations as u32;
    eprintln!(
        "Qwen3.5 real layer-0 MoE: tokens={tokens}, iterations={iterations}, {per_iteration:?}/iteration, {:.1} input tok/s",
        tokens as f64 / per_iteration.as_secs_f64()
    );

    let output = transfers::to_cpu(scratch.output())?;
    let output = output.as_bf16()?;
    assert_eq!(output.len(), tokens * hidden);
    assert!(output.iter().all(|value| value.to_f32().is_finite()));
    assert!(output.iter().any(|value| value.to_bits() & 0x7fff != 0));
    assert_eq!(scratch.output().dtype(), DType::BF16);
    Ok(())
}

fn probe_token_cases() -> Vec<usize> {
    let Some(value) = std::env::var(PROBE_TOKENS_ENV).ok() else {
        return vec![1, 4, 2048];
    };
    let cases = value
        .split(',')
        .map(|item| {
            let tokens = item.trim().parse::<usize>()
                .unwrap_or_else(|_| panic!("{PROBE_TOKENS_ENV} must be a comma-separated positive token count list"));
            assert!(tokens > 0, "{PROBE_TOKENS_ENV} counts must be positive");
            tokens
        })
        .collect::<Vec<_>>();
    let unique = cases.iter().copied().collect::<std::collections::BTreeSet<_>>();
    assert_eq!(unique.len(), cases.len(), "{PROBE_TOKENS_ENV} contains duplicate cases");
    cases
}

fn raw_fnv1a64(bytes: &[u8]) -> String {
    let digest = bytes.iter().fold(0xcbf2_9ce4_8422_2325u64, |hash, byte| {
        (hash ^ u64::from(*byte)).wrapping_mul(0x100_0000_01b3)
    });
    format!("{digest:016x}")
}

fn read_gpu_bytes(tensor: &Tensor) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let buffer = CudaBuffer::from_tensor(tensor)?;
    assert_eq!(buffer.len(), tensor.size_in_bytes(), "probe requires exact tensor storage");
    let mut bytes = vec![0u8; tensor.size_in_bytes()];
    buffer.copy_to_host(&mut bytes)?;
    Ok(bytes)
}

fn expert_fingerprints(edge: &DeviceEdgeNvfp4Experts) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
    let tensors = [
        ("fc1_weight", &edge.fc1_weight), ("fc1_scales", &edge.fc1_scales),
        ("fc1_alpha", &edge.fc1_alpha), ("input_global_scale", &edge.input_global_scale),
        ("fc2_weight", &edge.fc2_weight), ("fc2_scales", &edge.fc2_scales),
        ("fc2_alpha", &edge.fc2_alpha), ("down_input_scale", &edge.down_input_scale),
    ];
    let mut fingerprints = Vec::with_capacity(tensors.len());
    for (name, tensor) in tensors {
        let bytes = read_gpu_bytes(tensor)?;
        fingerprints.push(serde_json::json!({
            "name": name, "dtype": format!("{:?}", tensor.dtype()),
            "shape": tensor.shape().dims(), "size_bytes": bytes.len(),
            "raw_fnv1a64": raw_fnv1a64(&bytes),
        }));
    }
    Ok(serde_json::json!(fingerprints))
}

/// Diagnostic completion is not an accuracy pass: numerical differences are
/// measured without a tolerance waiver. Launch under /tmp/apxinf-gpu.lock:
///
/// flock /tmp/apxinf-gpu.lock env APXINF_QWEN35_CHECKPOINT=/ABS/model cargo test -p apxinf-model \
///   --no-default-features --features qwen35-cuda --test qwen35_real_moe_bench \
///   real_checkpoint_edge_fixed_routing_repeat_probe -- --exact --ignored --nocapture --test-threads=1
///
/// Defaults: layer27, tokens1/4/2048, 20 repeats per shape. Override layer with
/// APXINF_QWEN35_MOE_PROBE_LAYER (zero allowed), or select explicit shapes with
/// APXINF_QWEN35_MOE_PROBE_TOKENS=1,4,2048. This bypasses router/shared experts
/// and uses a synthetic frozen BF16 input, not a captured model hidden row.
/// Workspace starts with stream-ordered zeros, then persists without clearing;
/// this does not test dependence on arbitrary initial padding. Weight/source
/// hashes are explicitly FNV1a64 fingerprints, not cryptographic attestations.
#[test]
#[ignore = "diagnostic only; requires real NVFP4 checkpoint and SM110 CUDA; no accuracy pass"]
fn real_checkpoint_edge_fixed_routing_repeat_probe() -> Result<(), Box<dyn std::error::Error>> {
    assert!(ops::edge_nvfp4_moe_available(), "fixed-routing probe requires the real Edge provider");
    let checkpoint = checkpoint();
    let config_path = checkpoint.join("config.json");
    let config = Qwen35MoeConfig::from_json_file(&config_path)?;
    let layer = std::env::var(PROBE_LAYER_ENV).ok().map(|value| {
        value.parse::<usize>().unwrap_or_else(|_| panic!("{PROBE_LAYER_ENV} must be a nonnegative integer"))
    }).unwrap_or(27);
    let device = std::env::var("APXINF_CUDA_DEVICE").ok().map(|value| {
        value.parse::<usize>().expect("APXINF_CUDA_DEVICE must be a nonnegative integer")
    }).unwrap_or(0);
    let cases = probe_token_cases();
    let ctx = CudaContext::new(device)?;
    let moe = load_qwen35_moe_layer_from_path(&ctx, &config, &checkpoint, layer)?;
    let DeviceNvfp4RoutedExperts::Edge(edge) = &moe.routed_experts else {
        return Err("real checkpoint did not select Edge; routing-isolated probe was not executed".into());
    };
    assert_eq!(config.text_config.num_experts_per_tok, 8);
    assert_eq!(edge.experts, 256);
    let route_config = RoutedMoeConfig::new(edge.experts, 8);
    ctx.synchronize()?;
    // Hash actual prepared/repacked device weights once before and once after
    // the entire probe. This is distinct from the per-call exact byte checks
    // on the input, expert IDs, and F32 routing weights below.
    let weights_before = expert_fingerprints(edge)?;
    println!("{}", serde_json::json!({
        "schema": "apxinf.qwen35.fixed_routing_moe_probe.config.v1",
        "checkpoint": checkpoint, "checkpoint_config_fnv1a64": raw_fnv1a64(&std::fs::read(&config_path)?),
        "probe_source_fnv1a64": raw_fnv1a64(include_bytes!("qwen35_real_moe_bench.rs")),
        "native_wrapper_source_fnv1a64": raw_fnv1a64(include_bytes!("../../apxinf-cuda-new/native/kernels/edge_nvfp4_moe/runner.cu")),
        "hash_algorithm": "fnv1a64-raw-bytes", "layer": layer, "device": device,
        "requested_token_cases": cases, "default_full_case_set": cases == [1, 4, 2048],
        "repeats_per_case": PROBE_REPEATS, "fixed_routing": true, "router_executed": false,
        "execution_mode": "direct_eager_no_capture",
        "shared_expert_executed": false, "top_k": 8, "routing_weight": 0.125,
        "input_pattern": "bf16(((flat_index*17+layer*23+62)%257)/128-1)",
        "routing_pattern": "(token*17+slot*31+layer)%256; eight unique IDs per token",
        "checkpoint_expert_tensors": weights_before, "accuracy_status": "unresolved",
        "performance_accepted": false, "scope": "diagnostic; per-repeat sync/D2H; no performance score",
    }));

    for tokens in cases {
        let input_elements = tokens.checked_mul(edge.hidden).expect("probe input size overflow");
        let input_values = (0..input_elements).map(|index| {
            bf16::from_f32(((index % 257 * 17 + layer % 257 * 23 + 62) % 257) as f32 / 128.0 - 1.0)
        }).collect::<Vec<_>>();
        let ids = (0..tokens).flat_map(|token| {
            (0..8).map(move |slot| ((token % 256 * 17 + slot * 31 + layer % 256) % 256) as i32)
        }).collect::<Vec<_>>();
        for row in ids.chunks_exact(8) {
            assert_eq!(row.iter().collect::<std::collections::BTreeSet<_>>().len(), 8);
        }
        let routing_weights = vec![0.125f32; tokens.checked_mul(8).expect("probe route size overflow")];
        let input = transfers::to_cuda(&Tensor::from_bf16(vec![tokens, edge.hidden], &input_values)?, device)?;
        let expert_ids = transfers::to_cuda(&Tensor::from_i32(vec![tokens, 8], &ids)?, device)?;
        let expert_weights = transfers::to_cuda(&Tensor::from_f32(vec![tokens, 8], &routing_weights)?, device)?;
        let workspace = CudaBuffer::alloc_zeros_async(
            ops::edge_nvfp4_moe_workspace_size(route_config, tokens, edge.hidden, edge.intermediate)?, device, ctx.stream(),
        )?;
        let mut output = CudaBuffer::alloc_zeros_async(
            input_elements.checked_mul(DType::BF16.size_in_bytes()).expect("probe output size overflow"), device, ctx.stream(),
        )?.as_tensor(Shape::new(vec![tokens, edge.hidden]), DType::BF16)?;
        ctx.synchronize()?;
        let input_before = read_gpu_bytes(&input)?;
        let ids_before = read_gpu_bytes(&expert_ids)?;
        let routing_before = read_gpu_bytes(&expert_weights)?;
        let expected_input = input_values.iter().flat_map(|value| value.to_bits().to_le_bytes()).collect::<Vec<_>>();
        let expected_ids = ids.iter().flat_map(|value| value.to_le_bytes()).collect::<Vec<_>>();
        let expected_routing = routing_weights.iter().flat_map(|value| value.to_le_bytes()).collect::<Vec<_>>();
        assert!(input_before == expected_input, "BF16 upload changed probe input");
        assert!(ids_before == expected_ids, "I32 upload changed fixed expert IDs");
        assert!(routing_before == expected_routing, "F32 upload changed fixed routing weights");
        println!("{}", serde_json::json!({
            "schema": "apxinf.qwen35.fixed_routing_moe_probe.case.v1", "tokens": tokens,
            "input_raw_fnv1a64": raw_fnv1a64(&input_before), "expert_ids_raw_fnv1a64": raw_fnv1a64(&ids_before),
            "routing_weights_raw_fnv1a64": raw_fnv1a64(&routing_before), "workspace_bytes": workspace.len(),
            "workspace_initial_byte": 0, "workspace_reinitialized_between_repeats": false,
            "same_workspace_all_repeats": true, "output_dtype": "bf16", "fixed_routing": true,
        }));
        let mut baseline: Option<Vec<u8>> = None;
        let mut different_repeats = 0usize;
        for iteration in 0..PROBE_REPEATS {
            ops::edge_nvfp4_moe(&ctx, EdgeNvfp4MoeArgs {
                config: route_config, intermediate: edge.intermediate, input: &input,
                expert_ids: &expert_ids, expert_weights: &expert_weights,
                weights: EdgeNvfp4MoeWeights {
                    fc1_weight: &edge.fc1_weight, fc1_scales: &edge.fc1_scales, fc1_alpha: &edge.fc1_alpha,
                    input_global_scale: &edge.input_global_scale, fc2_weight: &edge.fc2_weight,
                    fc2_scales: &edge.fc2_scales, fc2_alpha: &edge.fc2_alpha, down_input_scale: &edge.down_input_scale,
                }, workspace: &workspace, output: &mut output,
            })?;
            ctx.synchronize()?;
            assert_eq!(output.shape().dims(), [tokens, edge.hidden]);
            assert_eq!(output.dtype(), DType::BF16);
            let bytes = read_gpu_bytes(&output)?;
            assert_eq!(bytes.len(), input_elements * 2);
            let values = bytes.chunks_exact(2).map(|raw| bf16::from_bits(u16::from_le_bytes([raw[0], raw[1]])).to_f32()).collect::<Vec<_>>();
            assert!(values.iter().all(|value| value.is_finite()), "nonfinite output at tokens={tokens} iteration={iteration}");
            assert!(read_gpu_bytes(&input)? == input_before, "input mutated at tokens={tokens} iteration={iteration}");
            assert!(read_gpu_bytes(&expert_ids)? == ids_before, "expert IDs mutated at tokens={tokens} iteration={iteration}");
            assert!(read_gpu_bytes(&expert_weights)? == routing_before, "routing weights mutated at tokens={tokens} iteration={iteration}");
            let first = baseline.get_or_insert_with(|| bytes.clone());
            let byte_mismatches = first.iter().zip(&bytes).filter(|(left, right)| left != right).count();
            let first_byte_mismatch = first.iter().zip(&bytes).position(|(left, right)| left != right);
            let mut max_abs = 0f64;
            let mut difference_squared = 0f64;
            let mut baseline_squared = 0f64;
            for (raw, value) in first.chunks_exact(2).zip(&values) {
                let expected = f64::from(bf16::from_bits(u16::from_le_bytes([raw[0], raw[1]])).to_f32());
                let difference = f64::from(*value) - expected;
                max_abs = max_abs.max(difference.abs());
                difference_squared += difference * difference;
                baseline_squared += expected * expected;
            }
            let relative_l2 = if baseline_squared > 0.0 { Some((difference_squared / baseline_squared).sqrt()) }
                else if difference_squared == 0.0 { Some(0.0) } else { None };
            different_repeats += usize::from(byte_mismatches != 0);
            println!("{}", serde_json::json!({
                "schema": "apxinf.qwen35.fixed_routing_moe_probe.iteration.v1", "tokens": tokens, "iteration": iteration,
                "output_raw_fnv1a64": raw_fnv1a64(&bytes), "baseline_output_raw_fnv1a64": raw_fnv1a64(first),
                "output_size_bytes": bytes.len(), "byte_mismatches_vs_first": byte_mismatches,
                "first_byte_mismatch": first_byte_mismatch, "max_absolute_difference": max_abs,
                "relative_l2": relative_l2, "relative_l2_defined": relative_l2.is_some(),
                "baseline_l2_squared": baseline_squared, "input_ids_routing_weights_bytewise_unchanged": true,
                "finite": true, "accuracy_status": "unresolved", "performance_accepted": false,
            }));
        }
        println!("{}", serde_json::json!({
            "schema": "apxinf.qwen35.fixed_routing_moe_probe.summary.v1", "tokens": tokens,
            "completed_iterations": PROBE_REPEATS, "non_bitwise_iterations": different_repeats,
            "bitwise_stable": different_repeats == 0, "fixed_routing": true,
            "diagnostic_completed": true, "accuracy_status": "unresolved", "performance_accepted": false,
        }));
    }
    ctx.synchronize()?;
    let weights_after = expert_fingerprints(edge)?;
    let weights_unchanged = weights_before == weights_after;
    println!("{}", serde_json::json!({
        "schema": "apxinf.qwen35.fixed_routing_moe_probe.weight_audit.v1",
        "checkpoint_expert_tensors_after": weights_after, "before_after_fingerprints_equal": weights_unchanged,
        "scope": "checkpoint tensor fingerprints before/after all cases; routing weights checked bytewise every call",
        "accuracy_status": "unresolved", "performance_accepted": false,
    }));
    assert!(weights_unchanged, "checkpoint expert tensor fingerprints changed during probe");
    Ok(())
}

// Experimental arithmetic-policy gate; safe production operator only.
mod edge_ordered_reduce_tests {
    use super::*;
    const POLICY_ENV: &str = "APXINF_EDGE_NVFP4_MOE_ORDERED_REDUCE";
    #[derive(Debug)]
    struct DiagnosticWorkspaceRegion {
        name: &'static str,
        offset: usize,
        bytes: usize,
    }

    struct DiagnosticWorkspaceLayout {
        old_total: usize,
        routes: usize,
        permuted_m: usize,
        tile_entries: usize,
        padded_inter_sf: usize,
        total: usize,
        regions: Vec<DiagnosticWorkspaceRegion>,
    }

    fn diagnostic_align_up(value: usize, alignment: usize) -> usize {
        value.checked_add(alignment - 1).expect("diagnostic alignment overflow")
            .checked_div(alignment).unwrap()
            .checked_mul(alignment).expect("diagnostic aligned size overflow")
    }

    // Diagnostic partition only: mirrors runner.cu WorkspaceLayout, not an
    // independent mathematical reference or a new native/public workspace ABI.
    fn diagnostic_workspace_layout(
        tokens: usize, top_k: usize, hidden: usize, intermediate: usize, experts: usize, enabled: bool,
    ) -> DiagnosticWorkspaceLayout {
        assert!(tokens > 0 && top_k > 0 && experts > 0);
        assert_eq!(hidden % 128, 0);
        assert_eq!(intermediate % 64, 0);
        let routes = tokens.checked_mul(top_k).expect("diagnostic route overflow");
        let permuted_m = diagnostic_align_up(
            routes.checked_add(experts.checked_mul(127).unwrap()).unwrap(), 128,
        );
        let tile_entries = (permuted_m / 128).max(1);
        let padded_inter_sf = diagnostic_align_up(intermediate / 16, 4);
        let mut cursor = 0usize;
        let mut regions = Vec::new();
        let sizes = [
            ("input_fp4", routes.checked_mul(hidden).unwrap() / 2),
            ("input_sf", routes.checked_mul(hidden).unwrap() / 16),
            ("tile_group", tile_entries.checked_mul(4).unwrap()),
            ("tile_limit", tile_entries.checked_mul(4).unwrap()),
            ("permuted_to_expanded", permuted_m.checked_mul(4).unwrap()),
            ("num_tiles", 4),
            ("fc1_fp4", permuted_m.checked_mul(intermediate).unwrap() / 2),
            ("fc1_sf", permuted_m.checked_mul(padded_inter_sf).unwrap()),
        ];
        for (name, bytes) in sizes {
            regions.push(DiagnosticWorkspaceRegion { name, offset: cursor, bytes });
            cursor = diagnostic_align_up(cursor.checked_add(bytes).unwrap(), 256);
        }
        let old_total = cursor.checked_add(256).unwrap();
        if enabled {
            let bytes = routes.checked_mul(hidden).unwrap().checked_mul(2).unwrap();
            regions.push(DiagnosticWorkspaceRegion { name: "expanded_bf16", offset: cursor, bytes });
            cursor = diagnostic_align_up(cursor.checked_add(bytes).unwrap(), 256);
        }
        let total = cursor.checked_add(256).unwrap();
        DiagnosticWorkspaceLayout { old_total, routes, permuted_m, tile_entries, padded_inter_sf, total, regions }
    }

    fn diagnostic_byte_difference(actual: &[u8], expected: &[u8], offset: usize) -> serde_json::Value {
        assert_eq!(actual.len(), expected.len());
        let first = actual.iter().zip(expected).position(|(a, b)| a != b);
        serde_json::json!({
            "byte_mismatches": actual.iter().zip(expected).filter(|(a, b)| a != b).count(),
            "first_byte_difference": first.map(|index| serde_json::json!({
                "region_byte": index, "workspace_byte": offset + index,
                "baseline_byte": expected[index], "actual_byte": actual[index],
            })),
        })
    }

    const GUARD_BYTES: usize = 256;
    const GUARD_VALUE: u8 = 0xa5;

    fn guarded_output(
        ctx: &CudaContext, rows: usize, hidden: usize,
    ) -> Result<(CudaBuffer, Tensor), Box<dyn std::error::Error>> {
        let payload = rows.checked_mul(hidden).unwrap().checked_mul(2).unwrap();
        let buffer = CudaBuffer::alloc_zeros_async(payload.checked_add(2 * GUARD_BYTES).unwrap(), ctx.device_id(), ctx.stream())?;
        buffer.memset_async(GUARD_VALUE, ctx.stream())?;
        let tensor = buffer.view(GUARD_BYTES, payload)?.as_tensor(Shape::new(vec![rows, hidden]), DType::BF16)?;
        CudaBuffer::from_tensor(&tensor)?.memset_async(0, ctx.stream())?;
        Ok((buffer, tensor))
    }

    fn read_guarded(
        buffer: &CudaBuffer,
    ) -> Result<(Vec<u8>, bool, serde_json::Value), Box<dyn std::error::Error>> {
        let mut bytes = vec![0u8; buffer.len()];
        buffer.copy_to_host(&mut bytes)?;
        let prefix = &bytes[..GUARD_BYTES];
        let suffix = &bytes[bytes.len() - GUARD_BYTES..];
        let expected = [GUARD_VALUE; GUARD_BYTES];
        let intact = prefix == expected && suffix == expected;
        let record = serde_json::json!({
            "bytes_each": GUARD_BYTES, "expected_byte": GUARD_VALUE, "intact": intact,
            "prefix_difference": diagnostic_byte_difference(prefix, &expected, 0),
            "suffix_difference": diagnostic_byte_difference(suffix, &expected, bytes.len() - GUARD_BYTES),
        });
        Ok((bytes[GUARD_BYTES..bytes.len() - GUARD_BYTES].to_vec(), intact, record))
    }

    // Independent host reduction only: input is the actual already-weighted
    // BF16 route storage; +0 then eight strict f32 adds, one final RNE BF16.
    // Rust is compiled without fast-math. No second route-weight multiply.
    fn ordered_cpu_reference(expanded: &[u8], tokens: usize, hidden: usize) -> Vec<u8> {
        assert_eq!(expanded.len(), tokens.checked_mul(8).unwrap().checked_mul(hidden).unwrap().checked_mul(2).unwrap());
        let mut output = Vec::with_capacity(tokens.checked_mul(hidden).unwrap().checked_mul(2).unwrap());
        for token in 0..tokens {
            for column in 0..hidden {
                let mut sum = 0.0f32;
                for slot in 0..8 {
                    let index = ((token * 8 + slot) * hidden + column) * 2;
                    let value = bf16::from_bits(u16::from_le_bytes([expanded[index], expanded[index + 1]])).to_f32();
                    assert!(value.is_finite(), "expanded route contribution is nonfinite");
                    sum += value;
                }
                assert!(sum.is_finite(), "ordered host reference overflowed");
                output.extend_from_slice(&bf16::from_f32(sum).to_bits().to_le_bytes());
            }
        }
        output
    }

    #[test]
    fn ordered_reference_literal_rounding_contract() {
        // Two BF16 half-ULP contributions survive FP32 accumulation, whereas
        // rounding back to BF16 after each add would lose both.
        let cases = [
            ([0x3f80u16, 0x3b80, 0x3b80, 0, 0, 0, 0, 0], 0x3f81u16),
            ([0x8000u16; 8], 0u16), // Explicit +0 start.
            ([1u16; 8], 8u16), // BF16 subnormals are not flushed.
        ];
        for (slots, expected) in cases {
            let input = slots.into_iter().flat_map(u16::to_le_bytes).collect::<Vec<_>>();
            assert_eq!(ordered_cpu_reference(&input, 1, 1), expected.to_le_bytes());
        }
    }

    /// Process-isolated workspace query gate. No GPU allocation or context.
    /// Run separately with ENV unset/0/1/invalid; never mutate it in-process.
    #[test]
    #[ignore = "experimental native workspace-policy query; separate processes for unset/0/1/invalid"]
    fn edge_ordered_reduce_workspace_policy_gate() -> Result<(), Box<dyn std::error::Error>> {
        assert!(ops::edge_nvfp4_moe_available());
        let requested = std::env::var(POLICY_ENV).ok();
        let enabled = requested.as_deref() == Some("1");
        let valid = matches!(requested.as_deref(), None | Some("0") | Some("1"));
        for tokens in [1usize, 4, 2048, 4096, 8192, 16384, 49152] {
            let layout = diagnostic_workspace_layout(tokens, 8, 2048, 512, 256, enabled);
            let result = ops::edge_nvfp4_moe_workspace_size(RoutedMoeConfig::new(256, 8), tokens, 2048, 512);
            if valid {
                let actual = result?;
                println!("{}", serde_json::json!({
                    "schema": "apxinf.qwen35.ordered_reduce.workspace_policy.v1", "tokens": tokens,
                    "requested_policy": requested, "enabled": enabled, "old_workspace_bytes": layout.old_total,
                    "expected_workspace_bytes": layout.total, "actual_workspace_bytes": actual,
                    "regions": layout.regions.iter().map(|r| serde_json::json!({"name": r.name, "offset": r.offset, "bytes": r.bytes})).collect::<Vec<_>>(),
                    "mirror_is_diagnostic_not_math_reference": true,
                }));
                assert_eq!(actual, layout.total);
            } else {
                assert!(result.is_err(), "invalid native policy value was silently accepted");
                println!("{}", serde_json::json!({
                    "schema": "apxinf.qwen35.ordered_reduce.workspace_policy.v1", "tokens": tokens,
                    "requested_policy": requested, "invalid_value_rejected": true, "error": result.err().map(|error| error.to_string()),
                }));
            }
        }
        Ok(())
    }

    /// Full safe-op integration gate for the new accumulation policy only.
    /// Entire output is independently reduced from the actual expanded storage.
    #[test]
    #[ignore = "experimental ordered-route accumulation; real checkpoint, full-output CPU reference and repeated guards/readonly"]
    fn real_checkpoint_edge_ordered_reduce_integration_gate() -> Result<(), Box<dyn std::error::Error>> {
        assert_eq!(std::env::var(POLICY_ENV).ok().as_deref(), Some("1"), "integration gate requires explicit native policy=1");
        assert!(ops::edge_nvfp4_moe_available());
        let checkpoint = checkpoint();
        let config_path = checkpoint.join("config.json");
        let config = Qwen35MoeConfig::from_json_file(&config_path)?;
        let layer = std::env::var(PROBE_LAYER_ENV).ok().map(|v| v.parse::<usize>().expect("invalid layer")).unwrap_or(27);
        let device = std::env::var("APXINF_CUDA_DEVICE").ok().map(|v| v.parse::<usize>().expect("invalid device")).unwrap_or(0);
        let cases = if std::env::var_os(PROBE_TOKENS_ENV).is_some() { probe_token_cases() } else { vec![1, 4] };
        let ctx = CudaContext::new_with_autotune(device, false)?;
        let moe = load_qwen35_moe_layer_from_path(&ctx, &config, &checkpoint, layer)?;
        let DeviceNvfp4RoutedExperts::Edge(edge) = &moe.routed_experts else { return Err("ordered gate requires Edge NVFP4".into()); };
        assert_eq!((edge.hidden, edge.intermediate, edge.experts), (2048, 512, 256));
        assert_eq!(config.text_config.num_experts_per_tok, 8);
        ctx.synchronize()?;
        let uploaded_weights = expert_fingerprints(edge)?;
        println!("{}", serde_json::json!({
            "schema": "apxinf.qwen35.ordered_reduce.integration.config.v1", "layer": layer, "device": device,
            "checkpoint": checkpoint, "requested_token_cases": cases, "native_policy": 1,
            "repeats_per_case": PROBE_REPEATS, "autotune": false,
            "probe_source_fnv1a64": raw_fnv1a64(include_bytes!("qwen35_real_moe_bench.rs")),
            "native_runner_fnv1a64": raw_fnv1a64(include_bytes!("../../apxinf-cuda-new/native/kernels/edge_nvfp4_moe/runner.cu")),
            "ordered_kernel_fnv1a64": raw_fnv1a64(include_bytes!("../../apxinf-cuda-new/native/kernels/edge_nvfp4_moe/ordered_route_reduce.cuh")),
            "uploaded_weight_fingerprints": uploaded_weights,
            "reference": "all T*H outputs; +0 then slot0..7 strict f32 add of weighted BF16 route values, final BF16 RNE",
            "second_weight_multiply": false, "original_sum_bytewise_equivalence_claimed": false,
            "scope": "new ordered reduction policy only; not independent FC1/FC2 math/full-model acceptance",
            "performance_accepted": false,
        }));
        for tokens in cases {
            let route_config = RoutedMoeConfig::new(256, 8);
            let routes = tokens.checked_mul(8).unwrap();
            let elements = tokens.checked_mul(edge.hidden).unwrap();
            let values = (0..elements).map(|i| bf16::from_f32(((i % 257 * 17 + layer % 257 * 23 + 62) % 257) as f32 / 128.0 - 1.0)).collect::<Vec<_>>();
            let ids = (0..tokens).flat_map(|t| (0..8).map(move |s| ((t % 256 * 17 + s * 31 + layer % 256) % 256) as i32)).collect::<Vec<_>>();
            let weights = vec![0.125f32; routes];
            let input = transfers::to_cuda(&Tensor::from_bf16(vec![tokens, edge.hidden], &values)?, device)?;
            let expert_ids = transfers::to_cuda(&Tensor::from_i32(vec![tokens, 8], &ids)?, device)?;
            let expert_weights = transfers::to_cuda(&Tensor::from_f32(vec![tokens, 8], &weights)?, device)?;
            let layout = diagnostic_workspace_layout(tokens, 8, edge.hidden, edge.intermediate, 256, true);
            let capi_size = ops::edge_nvfp4_moe_workspace_size(route_config, tokens, edge.hidden, edge.intermediate)?;
            assert_eq!(layout.total, capi_size);
            assert_eq!(layout.regions.len(), 9);
            let expanded = &layout.regions[8];
            assert_eq!(expanded.offset, layout.old_total - 256);
            assert_eq!(expanded.bytes, routes.checked_mul(edge.hidden).unwrap().checked_mul(2).unwrap());
            let whole_workspace = CudaBuffer::alloc_zeros_async(capi_size.checked_add(2 * GUARD_BYTES).unwrap(), device, ctx.stream())?;
            whole_workspace.memset_async(GUARD_VALUE, ctx.stream())?;
            let workspace = whole_workspace.view(GUARD_BYTES, capi_size)?;
            workspace.memset_async(0, ctx.stream())?;
            let expanded_view = workspace.view(expanded.offset, expanded.bytes)?;
            let expanded_tensor = expanded_view.as_tensor(Shape::new(vec![expanded.bytes]), DType::I8)?;
            let (whole_output, mut output) = guarded_output(&ctx, tokens, edge.hidden)?;
            ctx.synchronize()?;
            assert_eq!(read_gpu_bytes(&input)?, values.iter().flat_map(|v| v.to_bits().to_le_bytes()).collect::<Vec<_>>());
            assert_eq!(read_gpu_bytes(&expert_ids)?, ids.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>());
            assert_eq!(read_gpu_bytes(&expert_weights)?, weights.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>());
            let frozen_tensors = [
                ("input", &input), ("expert_ids", &expert_ids), ("routing_weights", &expert_weights),
                ("fc1_weight", &edge.fc1_weight), ("fc1_scales", &edge.fc1_scales), ("fc1_alpha", &edge.fc1_alpha),
                ("input_global_scale", &edge.input_global_scale), ("fc2_weight", &edge.fc2_weight),
                ("fc2_scales", &edge.fc2_scales), ("fc2_alpha", &edge.fc2_alpha), ("down_input_scale", &edge.down_input_scale),
            ].into_iter().map(|(name, tensor)| Ok((name, tensor, read_gpu_bytes(tensor)?)))
                .collect::<Result<Vec<_>, Box<dyn std::error::Error>>>()?;
            println!("{}", serde_json::json!({
                "schema": "apxinf.qwen35.ordered_reduce.integration.case.v1", "tokens": tokens,
                "old_workspace_bytes": layout.old_total, "ordered_workspace_bytes": capi_size,
                "routes": layout.routes, "permuted_m": layout.permuted_m,
                "tile_entries": layout.tile_entries, "padded_inter_sf": layout.padded_inter_sf,
                "expanded_offset": expanded.offset, "expanded_bytes": expanded.bytes,
                "old_eight_region_offsets_preserved": true, "guard_bytes_each": GUARD_BYTES,
                "regions": layout.regions.iter().map(|r| serde_json::json!({"name": r.name, "offset": r.offset, "bytes": r.bytes})).collect::<Vec<_>>(),
                "reference_elements_per_repeat": elements, "performance_accepted": false,
            }));
            let mut first_output: Option<Vec<u8>> = None;
            let mut first_workspace: Option<Vec<u8>> = None;
            for iteration in 0..PROBE_REPEATS {
                ctx.synchronize()?;
                CudaBuffer::from_tensor(&output)?.memset_async(0xff, ctx.stream())?;
                expanded_view.memset_async(0xff, ctx.stream())?;
                ctx.synchronize()?;
                assert!(read_gpu_bytes(&output)?.iter().all(|&b| b == 0xff));
                assert!(read_gpu_bytes(&expanded_tensor)?.iter().all(|&b| b == 0xff));
                ops::edge_nvfp4_moe(&ctx, EdgeNvfp4MoeArgs {
                    config: route_config, intermediate: edge.intermediate, input: &input,
                    expert_ids: &expert_ids, expert_weights: &expert_weights,
                    weights: EdgeNvfp4MoeWeights {
                        fc1_weight: &edge.fc1_weight, fc1_scales: &edge.fc1_scales, fc1_alpha: &edge.fc1_alpha,
                        input_global_scale: &edge.input_global_scale, fc2_weight: &edge.fc2_weight,
                        fc2_scales: &edge.fc2_scales, fc2_alpha: &edge.fc2_alpha, down_input_scale: &edge.down_input_scale,
                    }, workspace: &workspace, output: &mut output,
                })?;
                ctx.synchronize()?;
                let (bytes, c_guards_ok, c_guards) = read_guarded(&whole_output)?;
                let (workspace_bytes, ws_guards_ok, ws_guards) = read_guarded(&whole_workspace)?;
                let expanded_bytes = &workspace_bytes[expanded.offset..expanded.offset + expanded.bytes];
                let finite = bytes.chunks_exact(2).all(|v| bf16::from_bits(u16::from_le_bytes([v[0], v[1]])).to_f32().is_finite());
                let expanded_finite = expanded_bytes.chunks_exact(2).all(|v| bf16::from_bits(u16::from_le_bytes([v[0], v[1]])).to_f32().is_finite());
                let output_nonzero = bytes.chunks_exact(2).any(|v| u16::from_le_bytes([v[0], v[1]]) & 0x7fff != 0);
                let expanded_nonzero = expanded_bytes.chunks_exact(2).any(|v| u16::from_le_bytes([v[0], v[1]]) & 0x7fff != 0);
                let reference = if expanded_finite { Some(ordered_cpu_reference(expanded_bytes, tokens, edge.hidden)) } else { None };
                let reference_equal = reference.as_ref().map(|r| r == &bytes).unwrap_or(false);
                let first_c = first_output.get_or_insert_with(|| bytes.clone());
                let first_w = first_workspace.get_or_insert_with(|| workspace_bytes.clone());
                let output_stable = bytes == *first_c;
                let workspace_stable = workspace_bytes == *first_w;
                let mut readonly_ok = true;
                let mut readonly = Vec::with_capacity(frozen_tensors.len());
                for (name, tensor, frozen) in &frozen_tensors {
                    let now = read_gpu_bytes(tensor)?;
                    let same = now == *frozen;
                    readonly_ok &= same;
                    readonly.push(serde_json::json!({"name": name, "full_size_bytes": now.len(),
                        "bytewise_unchanged": same, "raw_fnv1a64": raw_fnv1a64(&now),
                        "difference": diagnostic_byte_difference(&now, frozen, 0)}));
                }
                println!("{}", serde_json::json!({
                    "schema": "apxinf.qwen35.ordered_reduce.integration.iteration.v1", "tokens": tokens, "iteration": iteration,
                    "output_nan_poison_verified": true, "expanded_nan_poison_verified": true,
                    "output_finite": finite, "expanded_finite": expanded_finite,
                    "output_has_nonzero": output_nonzero, "expanded_has_nonzero": expanded_nonzero,
                    "full_output_cpu_reference_bytewise_equal": reference_equal,
                    "reference_elements": elements, "output_raw_fnv1a64": raw_fnv1a64(&bytes),
                    "reference_raw_fnv1a64": reference.as_ref().map(|r| raw_fnv1a64(r)),
                    "reference_difference": reference.as_ref().map(|r| diagnostic_byte_difference(&bytes, r, 0)),
                    "output_bitstable_vs_first": output_stable,
                    "output_difference_vs_first": diagnostic_byte_difference(&bytes, first_c, 0),
                    "whole_workspace_bitstable_vs_first": workspace_stable,
                    "regions": layout.regions.iter().map(|r| {
                        let now = &workspace_bytes[r.offset..r.offset + r.bytes];
                        let first = &first_w[r.offset..r.offset + r.bytes];
                        serde_json::json!({"name": r.name, "full_size_bytes": r.bytes, "raw_fnv1a64": raw_fnv1a64(now),
                            "bytewise_equal_vs_first": now == first, "difference": diagnostic_byte_difference(now, first, r.offset)})
                    }).collect::<Vec<_>>(),
                    "output_guards": c_guards, "workspace_guards": ws_guards,
                    "input_weight_tensors": readonly, "all_readonly_tensors_bytewise_unchanged": readonly_ok,
                    "original_sum_bytewise_equivalence_claimed": false, "performance_accepted": false,
                }));
                assert!(c_guards_ok && ws_guards_ok, "operator overwrote workspace/output guards");
                assert!(readonly_ok, "operator changed readonly input/routing/weights");
                assert!(finite && expanded_finite, "ordered operation left nonfinite output or expanded poison");
                assert!(output_nonzero && expanded_nonzero, "real fixture unexpectedly produced all-zero output");
                assert!(reference_equal, "ordered GPU full output differs from strict CPU f32->BF16 reduction");
                assert!(output_stable && workspace_stable, "ordered operator is not bitstable for frozen inputs");
            }
            println!("{}", serde_json::json!({
                "schema": "apxinf.qwen35.ordered_reduce.integration.summary.v1", "tokens": tokens,
                "completed_full_safe_op_repeats": PROBE_REPEATS, "full_output_reference_elements_per_repeat": elements,
                "ordered_policy_reference_gate_completed": true, "original_sum_bytewise_equivalence_claimed": false,
                "scope": "new reduction policy only; FC1/FC2 math and full model accuracy not independently accepted",
                "performance_accepted": false,
            }));
        }
        Ok(())
    }
}

// Experimental arithmetic-policy gate; safe production operator only.
mod edge_ordered_reduce_capture_tests {
    use super::*;
    const POLICY_ENV: &str = "APXINF_EDGE_NVFP4_MOE_ORDERED_REDUCE";
    #[derive(Debug)]
    struct DiagnosticWorkspaceRegion {
        name: &'static str,
        offset: usize,
        bytes: usize,
    }

    struct DiagnosticWorkspaceLayout {
        old_total: usize,
        routes: usize,
        permuted_m: usize,
        tile_entries: usize,
        padded_inter_sf: usize,
        total: usize,
        regions: Vec<DiagnosticWorkspaceRegion>,
    }

    fn diagnostic_align_up(value: usize, alignment: usize) -> usize {
        value.checked_add(alignment - 1).expect("diagnostic alignment overflow")
            .checked_div(alignment).unwrap()
            .checked_mul(alignment).expect("diagnostic aligned size overflow")
    }

    // Diagnostic partition only: mirrors runner.cu WorkspaceLayout, not an
    // independent mathematical reference or a new native/public workspace ABI.
    fn diagnostic_workspace_layout(
        tokens: usize, top_k: usize, hidden: usize, intermediate: usize, experts: usize, enabled: bool,
    ) -> DiagnosticWorkspaceLayout {
        assert!(tokens > 0 && top_k > 0 && experts > 0);
        assert_eq!(hidden % 128, 0);
        assert_eq!(intermediate % 64, 0);
        let routes = tokens.checked_mul(top_k).expect("diagnostic route overflow");
        let permuted_m = diagnostic_align_up(
            routes.checked_add(experts.checked_mul(127).unwrap()).unwrap(), 128,
        );
        let tile_entries = (permuted_m / 128).max(1);
        let padded_inter_sf = diagnostic_align_up(intermediate / 16, 4);
        let mut cursor = 0usize;
        let mut regions = Vec::new();
        let sizes = [
            ("input_fp4", routes.checked_mul(hidden).unwrap() / 2),
            ("input_sf", routes.checked_mul(hidden).unwrap() / 16),
            ("tile_group", tile_entries.checked_mul(4).unwrap()),
            ("tile_limit", tile_entries.checked_mul(4).unwrap()),
            ("permuted_to_expanded", permuted_m.checked_mul(4).unwrap()),
            ("num_tiles", 4),
            ("fc1_fp4", permuted_m.checked_mul(intermediate).unwrap() / 2),
            ("fc1_sf", permuted_m.checked_mul(padded_inter_sf).unwrap()),
        ];
        for (name, bytes) in sizes {
            regions.push(DiagnosticWorkspaceRegion { name, offset: cursor, bytes });
            cursor = diagnostic_align_up(cursor.checked_add(bytes).unwrap(), 256);
        }
        let old_total = cursor.checked_add(256).unwrap();
        if enabled {
            let bytes = routes.checked_mul(hidden).unwrap().checked_mul(2).unwrap();
            regions.push(DiagnosticWorkspaceRegion { name: "expanded_bf16", offset: cursor, bytes });
            cursor = diagnostic_align_up(cursor.checked_add(bytes).unwrap(), 256);
        }
        let total = cursor.checked_add(256).unwrap();
        DiagnosticWorkspaceLayout { old_total, routes, permuted_m, tile_entries, padded_inter_sf, total, regions }
    }

    fn diagnostic_byte_difference(actual: &[u8], expected: &[u8], offset: usize) -> serde_json::Value {
        assert_eq!(actual.len(), expected.len());
        let first = actual.iter().zip(expected).position(|(a, b)| a != b);
        serde_json::json!({
            "byte_mismatches": actual.iter().zip(expected).filter(|(a, b)| a != b).count(),
            "first_byte_difference": first.map(|index| serde_json::json!({
                "region_byte": index, "workspace_byte": offset + index,
                "baseline_byte": expected[index], "actual_byte": actual[index],
            })),
        })
    }

    const GUARD_BYTES: usize = 256;
    const GUARD_VALUE: u8 = 0xa5;

    fn guarded_output(
        ctx: &CudaContext, rows: usize, hidden: usize,
    ) -> Result<(CudaBuffer, Tensor), Box<dyn std::error::Error>> {
        let payload = rows.checked_mul(hidden).unwrap().checked_mul(2).unwrap();
        let buffer = CudaBuffer::alloc_zeros_async(payload.checked_add(2 * GUARD_BYTES).unwrap(), ctx.device_id(), ctx.stream())?;
        buffer.memset_async(GUARD_VALUE, ctx.stream())?;
        let tensor = buffer.view(GUARD_BYTES, payload)?.as_tensor(Shape::new(vec![rows, hidden]), DType::BF16)?;
        CudaBuffer::from_tensor(&tensor)?.memset_async(0, ctx.stream())?;
        Ok((buffer, tensor))
    }

    fn read_guarded(
        buffer: &CudaBuffer,
    ) -> Result<(Vec<u8>, bool, serde_json::Value), Box<dyn std::error::Error>> {
        let mut bytes = vec![0u8; buffer.len()];
        buffer.copy_to_host(&mut bytes)?;
        let prefix = &bytes[..GUARD_BYTES];
        let suffix = &bytes[bytes.len() - GUARD_BYTES..];
        let expected = [GUARD_VALUE; GUARD_BYTES];
        let intact = prefix == expected && suffix == expected;
        let record = serde_json::json!({
            "bytes_each": GUARD_BYTES, "expected_byte": GUARD_VALUE, "intact": intact,
            "prefix_difference": diagnostic_byte_difference(prefix, &expected, 0),
            "suffix_difference": diagnostic_byte_difference(suffix, &expected, bytes.len() - GUARD_BYTES),
        });
        Ok((bytes[GUARD_BYTES..bytes.len() - GUARD_BYTES].to_vec(), intact, record))
    }

    // Independent host reduction only: input is the actual already-weighted
    // BF16 route storage; +0 then eight strict f32 adds, one final RNE BF16.
    // Rust is compiled without fast-math. No second route-weight multiply.
    fn ordered_cpu_reference(expanded: &[u8], tokens: usize, hidden: usize) -> Vec<u8> {
        assert_eq!(expanded.len(), tokens.checked_mul(8).unwrap().checked_mul(hidden).unwrap().checked_mul(2).unwrap());
        let mut output = Vec::with_capacity(tokens.checked_mul(hidden).unwrap().checked_mul(2).unwrap());
        for token in 0..tokens {
            for column in 0..hidden {
                let mut sum = 0.0f32;
                for slot in 0..8 {
                    let index = ((token * 8 + slot) * hidden + column) * 2;
                    let value = bf16::from_bits(u16::from_le_bytes([expanded[index], expanded[index + 1]])).to_f32();
                    assert!(value.is_finite(), "expanded route contribution is nonfinite");
                    sum += value;
                }
                assert!(sum.is_finite(), "ordered host reference overflowed");
                output.extend_from_slice(&bf16::from_f32(sum).to_bits().to_le_bytes());
            }
        }
        output
    }

    fn run_ordered_safe(
        ctx: &CudaContext, edge: &DeviceEdgeNvfp4Experts,
        input: &Tensor, ids: &Tensor, weights: &Tensor,
        workspace: &CudaBuffer, output: &mut Tensor,
    ) -> apxinf_core::Result<()> {
        ops::edge_nvfp4_moe(ctx, EdgeNvfp4MoeArgs {
            config: RoutedMoeConfig::new(256, 8), intermediate: edge.intermediate,
            input, expert_ids: ids, expert_weights: weights,
            weights: EdgeNvfp4MoeWeights {
                fc1_weight: &edge.fc1_weight, fc1_scales: &edge.fc1_scales, fc1_alpha: &edge.fc1_alpha,
                input_global_scale: &edge.input_global_scale, fc2_weight: &edge.fc2_weight,
                fc2_scales: &edge.fc2_scales, fc2_alpha: &edge.fc2_alpha, down_input_scale: &edge.down_input_scale,
            }, workspace, output,
        })
    }

    /// Full safe-op capture/eager equivalence gate for the new accumulation policy only.
    /// Entire output is independently reduced from the actual expanded storage.
    #[test]
    #[ignore = "experimental ordered-route graph/eager equivalence with full reference/internal canary"]
    fn real_checkpoint_edge_ordered_reduce_capture_eager_gate() -> Result<(), Box<dyn std::error::Error>> {
        assert_eq!(std::env::var(POLICY_ENV).ok().as_deref(), Some("1"), "integration gate requires explicit native policy=1");
        assert!(ops::edge_nvfp4_moe_available());
        let checkpoint = checkpoint();
        let config_path = checkpoint.join("config.json");
        let config = Qwen35MoeConfig::from_json_file(&config_path)?;
        let layer = std::env::var(PROBE_LAYER_ENV).ok().map(|v| v.parse::<usize>().expect("invalid layer")).unwrap_or(27);
        let device = std::env::var("APXINF_CUDA_DEVICE").ok().map(|v| v.parse::<usize>().expect("invalid device")).unwrap_or(0);
        let cases = if std::env::var_os(PROBE_TOKENS_ENV).is_some() { probe_token_cases() } else { vec![1, 4] };
        let ctx = CudaContext::new_with_autotune(device, false)?;
        let moe = load_qwen35_moe_layer_from_path(&ctx, &config, &checkpoint, layer)?;
        let DeviceNvfp4RoutedExperts::Edge(edge) = &moe.routed_experts else { return Err("ordered gate requires Edge NVFP4".into()); };
        assert_eq!((edge.hidden, edge.intermediate, edge.experts), (2048, 512, 256));
        assert_eq!(config.text_config.num_experts_per_tok, 8);
        ctx.synchronize()?;
        let uploaded_weights = expert_fingerprints(edge)?;
        println!("{}", serde_json::json!({
            "schema": "apxinf.qwen35.ordered_reduce.capture_eager.config.v1", "layer": layer, "device": device,
            "checkpoint": checkpoint, "requested_token_cases": cases, "native_policy": 1,
            "pairs_per_case": PROBE_REPEATS, "calls_per_pair": 2, "autotune": false,
            "probe_source_fnv1a64": raw_fnv1a64(include_bytes!("qwen35_real_moe_bench.rs")),
            "native_runner_fnv1a64": raw_fnv1a64(include_bytes!("../../apxinf-cuda-new/native/kernels/edge_nvfp4_moe/runner.cu")),
            "ordered_kernel_fnv1a64": raw_fnv1a64(include_bytes!("../../apxinf-cuda-new/native/kernels/edge_nvfp4_moe/ordered_route_reduce.cuh")),
            "uploaded_weight_fingerprints": uploaded_weights,
            "reference": "all T*H outputs; +0 then slot0..7 strict f32 add of weighted BF16 route values, final BF16 RNE",
            "second_weight_multiply": false, "original_sum_bytewise_equivalence_claimed": false,
            "scope": "new ordered reduction policy only; not independent FC1/FC2 math/full-model acceptance",
            "performance_accepted": false,
        }));
        for tokens in cases {
            let route_config = RoutedMoeConfig::new(256, 8);
            let routes = tokens.checked_mul(8).unwrap();
            let elements = tokens.checked_mul(edge.hidden).unwrap();
            let values = (0..elements).map(|i| bf16::from_f32(((i % 257 * 17 + layer % 257 * 23 + 62) % 257) as f32 / 128.0 - 1.0)).collect::<Vec<_>>();
            let ids = (0..tokens).flat_map(|t| (0..8).map(move |s| ((t % 256 * 17 + s * 31 + layer % 256) % 256) as i32)).collect::<Vec<_>>();
            let weights = vec![0.125f32; routes];
            let input = transfers::to_cuda(&Tensor::from_bf16(vec![tokens, edge.hidden], &values)?, device)?;
            let expert_ids = transfers::to_cuda(&Tensor::from_i32(vec![tokens, 8], &ids)?, device)?;
            let expert_weights = transfers::to_cuda(&Tensor::from_f32(vec![tokens, 8], &weights)?, device)?;
            let layout = diagnostic_workspace_layout(tokens, 8, edge.hidden, edge.intermediate, 256, true);
            let capi_size = ops::edge_nvfp4_moe_workspace_size(route_config, tokens, edge.hidden, edge.intermediate)?;
            assert_eq!(layout.total, capi_size);
            assert_eq!(layout.regions.len(), 9);
            let expanded = &layout.regions[8];
            assert_eq!(expanded.offset, layout.old_total - 256);
            assert_eq!(expanded.bytes, routes.checked_mul(edge.hidden).unwrap().checked_mul(2).unwrap());
            assert_eq!(expanded.offset.checked_add(expanded.bytes).unwrap(), capi_size - 256);
            let whole_workspace = CudaBuffer::alloc_zeros_async(capi_size.checked_add(2 * GUARD_BYTES).unwrap(), device, ctx.stream())?;
            whole_workspace.memset_async(GUARD_VALUE, ctx.stream())?;
            let workspace = whole_workspace.view(GUARD_BYTES, capi_size)?;
            workspace.memset_async(0, ctx.stream())?;
            let expanded_view = workspace.view(expanded.offset, expanded.bytes)?;
            let expanded_tensor = expanded_view.as_tensor(Shape::new(vec![expanded.bytes]), DType::I8)?;
            let (whole_output, mut output) = guarded_output(&ctx, tokens, edge.hidden)?;
            ctx.synchronize()?;
            assert_eq!(read_gpu_bytes(&input)?, values.iter().flat_map(|v| v.to_bits().to_le_bytes()).collect::<Vec<_>>());
            assert_eq!(read_gpu_bytes(&expert_ids)?, ids.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>());
            assert_eq!(read_gpu_bytes(&expert_weights)?, weights.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>());
            let frozen_tensors = [
                ("input", &input), ("expert_ids", &expert_ids), ("routing_weights", &expert_weights),
                ("fc1_weight", &edge.fc1_weight), ("fc1_scales", &edge.fc1_scales), ("fc1_alpha", &edge.fc1_alpha),
                ("input_global_scale", &edge.input_global_scale), ("fc2_weight", &edge.fc2_weight),
                ("fc2_scales", &edge.fc2_scales), ("fc2_alpha", &edge.fc2_alpha), ("down_input_scale", &edge.down_input_scale),
            ].into_iter().map(|(name, tensor)| Ok((name, tensor, read_gpu_bytes(tensor)?)))
                .collect::<Result<Vec<_>, Box<dyn std::error::Error>>>()?;
            println!("{}", serde_json::json!({
                "schema": "apxinf.qwen35.ordered_reduce.capture_eager.case.v1", "tokens": tokens,
                "old_workspace_bytes": layout.old_total, "ordered_workspace_bytes": capi_size,
                "routes": layout.routes, "permuted_m": layout.permuted_m,
                "tile_entries": layout.tile_entries, "padded_inter_sf": layout.padded_inter_sf,
                "expanded_offset": expanded.offset, "expanded_bytes": expanded.bytes,
                "old_eight_region_offsets_preserved": true, "guard_bytes_each": GUARD_BYTES,
                "regions": layout.regions.iter().map(|r| serde_json::json!({"name": r.name, "offset": r.offset, "bytes": r.bytes})).collect::<Vec<_>>(),
                "reference_elements_per_repeat": elements, "performance_accepted": false,
            }));
            let mut first_output: Option<Vec<u8>> = None;
            let mut first_workspace: Option<Vec<u8>> = None;
            // Warm all existing module/load paths outside capture. Keep all
            // inputs, weights, workspace and output allocations alive until
            // graph destruction after the final synchronized replay.
            for _ in 0..2 {
                run_ordered_safe(&ctx, edge, &input, &expert_ids, &expert_weights, &workspace, &mut output)?;
            }
            ctx.synchronize()?;
            let graph = apxinf_cuda_new::capture(&ctx, || {
                run_ordered_safe(&ctx, edge, &input, &expert_ids, &expert_weights, &workspace, &mut output)
            })?;
            let internal_tail = workspace.view(capi_size - 256, 256)?;
            let internal_tail_tensor = internal_tail.as_tensor(Shape::new(vec![256]), DType::I8)?;
            for execution in 0..PROBE_REPEATS * 2 {
                let iteration = execution / 2;
                let graph_mode = execution % 2 == 1;
                ctx.synchronize()?;
                CudaBuffer::from_tensor(&output)?.memset_async(0xff, ctx.stream())?;
                expanded_view.memset_async(0xff, ctx.stream())?;
                internal_tail.memset_async(GUARD_VALUE, ctx.stream())?;
                ctx.synchronize()?;
                assert!(read_gpu_bytes(&output)?.iter().all(|&b| b == 0xff));
                assert!(read_gpu_bytes(&expanded_tensor)?.iter().all(|&b| b == 0xff));
                assert!(read_gpu_bytes(&internal_tail_tensor)?.iter().all(|&b| b == GUARD_VALUE));
                if graph_mode {
                    graph.replay()?;
                } else {
                    run_ordered_safe(&ctx, edge, &input, &expert_ids, &expert_weights, &workspace, &mut output)?;
                }
                ctx.synchronize()?;
                let (bytes, c_guards_ok, c_guards) = read_guarded(&whole_output)?;
                let (workspace_bytes, ws_guards_ok, ws_guards) = read_guarded(&whole_workspace)?;
                let internal_tail_ok = workspace_bytes[capi_size - 256..].iter().all(|&b| b == GUARD_VALUE);
                let expanded_bytes = &workspace_bytes[expanded.offset..expanded.offset + expanded.bytes];
                let finite = bytes.chunks_exact(2).all(|v| bf16::from_bits(u16::from_le_bytes([v[0], v[1]])).to_f32().is_finite());
                let expanded_finite = expanded_bytes.chunks_exact(2).all(|v| bf16::from_bits(u16::from_le_bytes([v[0], v[1]])).to_f32().is_finite());
                let output_nonzero = bytes.chunks_exact(2).any(|v| u16::from_le_bytes([v[0], v[1]]) & 0x7fff != 0);
                let expanded_nonzero = expanded_bytes.chunks_exact(2).any(|v| u16::from_le_bytes([v[0], v[1]]) & 0x7fff != 0);
                let reference = if expanded_finite { Some(ordered_cpu_reference(expanded_bytes, tokens, edge.hidden)) } else { None };
                let reference_equal = reference.as_ref().map(|r| r == &bytes).unwrap_or(false);
                let first_c = first_output.get_or_insert_with(|| bytes.clone());
                let first_w = first_workspace.get_or_insert_with(|| workspace_bytes.clone());
                let output_stable = bytes == *first_c;
                let workspace_stable = workspace_bytes == *first_w;
                let mut readonly_ok = true;
                let mut readonly = Vec::with_capacity(frozen_tensors.len());
                for (name, tensor, frozen) in &frozen_tensors {
                    let now = read_gpu_bytes(tensor)?;
                    let same = now == *frozen;
                    readonly_ok &= same;
                    readonly.push(serde_json::json!({"name": name, "full_size_bytes": now.len(),
                        "bytewise_unchanged": same, "raw_fnv1a64": raw_fnv1a64(&now),
                        "difference": diagnostic_byte_difference(&now, frozen, 0)}));
                }
                println!("{}", serde_json::json!({
                    "schema": "apxinf.qwen35.ordered_reduce.capture_eager.iteration.v1", "tokens": tokens, "iteration": iteration, "execution": execution,
                    "mode": if graph_mode { "graph" } else { "eager" },
                    "output_nan_poison_verified": true, "expanded_nan_poison_verified": true,
                    "output_finite": finite, "expanded_finite": expanded_finite,
                    "output_has_nonzero": output_nonzero, "expanded_has_nonzero": expanded_nonzero,
                    "full_output_cpu_reference_bytewise_equal": reference_equal,
                    "reference_elements": elements, "output_raw_fnv1a64": raw_fnv1a64(&bytes),
                    "reference_raw_fnv1a64": reference.as_ref().map(|r| raw_fnv1a64(r)),
                    "reference_difference": reference.as_ref().map(|r| diagnostic_byte_difference(&bytes, r, 0)),
                    "output_bitstable_vs_first": output_stable,
                    "output_difference_vs_first": diagnostic_byte_difference(&bytes, first_c, 0),
                    "whole_workspace_bitstable_vs_first": workspace_stable,
                    "regions": layout.regions.iter().map(|r| {
                        let now = &workspace_bytes[r.offset..r.offset + r.bytes];
                        let first = &first_w[r.offset..r.offset + r.bytes];
                        serde_json::json!({"name": r.name, "full_size_bytes": r.bytes, "raw_fnv1a64": raw_fnv1a64(now),
                            "bytewise_equal_vs_first": now == first, "difference": diagnostic_byte_difference(now, first, r.offset)})
                    }).collect::<Vec<_>>(),
                    "output_guards": c_guards, "workspace_guards": ws_guards,
                    "internal_trailing_256_canary_intact": internal_tail_ok,
                    "internal_trailing_difference": diagnostic_byte_difference(&workspace_bytes[capi_size - 256..], &[GUARD_VALUE; 256], capi_size - 256),
                    "input_weight_tensors": readonly, "all_readonly_tensors_bytewise_unchanged": readonly_ok,
                    "original_sum_bytewise_equivalence_claimed": false, "performance_accepted": false,
                }));
                assert!(c_guards_ok && ws_guards_ok && internal_tail_ok, "operator overwrote outer guards or internal trailing canary");
                assert!(readonly_ok, "operator changed readonly input/routing/weights");
                assert!(finite && expanded_finite, "ordered operation left nonfinite output or expanded poison");
                assert!(output_nonzero && expanded_nonzero, "real fixture unexpectedly produced all-zero output");
                assert!(reference_equal, "ordered GPU full output differs from strict CPU f32->BF16 reduction");
                assert!(output_stable && workspace_stable, "ordered operator is not bitstable for frozen inputs");
            }
            println!("{}", serde_json::json!({
                "schema": "apxinf.qwen35.ordered_reduce.capture_eager.summary.v1", "tokens": tokens,
                "completed_eager_calls": PROBE_REPEATS, "completed_graph_replays": PROBE_REPEATS,
                "warmup_eager_calls": 2, "same_allocation_graph_captures": 1, "full_output_reference_elements_per_repeat": elements,
                "ordered_policy_reference_gate_completed": true, "original_sum_bytewise_equivalence_claimed": false,
                "scope": "new reduction policy only; FC1/FC2 math and full model accuracy not independently accepted",
                "performance_accepted": false,
            }));
        }
        Ok(())
    }
}
