//! Staged image attention: exact dispatch parity and native workspace ownership.
use crate::{ops, transfers, CudaBuffer, CudaContext, ExecutionSession, PreparedPhase};
use apxinf_core::{DType, Shape, Tensor};

fn tensor(ctx: &CudaContext, shape: Vec<usize>, values: &[f32]) -> Tensor {
    transfers::to_cuda(
        &Tensor::from_bf16(
            shape,
            &values
                .iter()
                .map(|&v| half::bf16::from_f32(v))
                .collect::<Vec<_>>(),
        )
        .unwrap(),
        ctx.device_id(),
    )
    .unwrap()
}
fn read(t: &Tensor) -> Vec<half::bf16> {
    transfers::to_cpu(t)
        .unwrap()
        .to_f32_vec()
        .unwrap()
        .into_iter()
        .map(half::bf16::from_f32)
        .collect()
}

#[test]
fn gpu_staged_tiled_exact_dispatch_graph_and_workspace_release() {
    let ctx = CudaContext::new_with_recipe_options(0, false, Some(String::new())).unwrap();
    // Cross a query chunk, key tile, odd reduction, and unequal image boundary.
    let (tokens, heads, dim) = (260, 2, 7);
    let n = tokens * heads * dim;
    let q: Vec<_> = (0..n)
        .map(|i| {
            if i % dim == 0 {
                (1 + i % 17) as f32 * f32::from_bits(0x00010000)
            } else {
                ((i * 17 % 97) as f32 - 48.) / 512.
            }
        })
        .collect();
    let k: Vec<_> = (0..n)
        .map(|i| {
            if i % dim == 0 {
                (1 + i % 7) as f32 * 1e35
            } else {
                ((i * 13 % 83) as f32 - 41.) / 512.
            }
        })
        .collect();
    let v: Vec<_> = (0..n)
        .map(|i| ((i * 23 % 127) as f32 - 63.) / 32.)
        .collect();
    let query = tensor(&ctx, vec![tokens, heads, dim], &q);
    let key = tensor(&ctx, vec![tokens, heads, dim], &k);
    let value = tensor(&ctx, vec![tokens, heads, dim], &v);
    let original = (read(&query), read(&key), read(&value));
    let mut legacy_out = tensor(&ctx, vec![tokens, heads, dim], &vec![0.; n]);
    let guarded = tensor(&ctx, vec![n + 32], &vec![42.; n + 32]);
    let mut tiled_out = CudaBuffer::from_tensor(&guarded)
        .unwrap()
        .view(16 * 2, n * 2)
        .unwrap()
        .as_tensor(Shape::new(vec![tokens, heads, dim]), DType::BF16)
        .unwrap();
    let offsets = [0u32, 129, 260];
    let boundaries = CudaBuffer::alloc(12, 0).unwrap();
    boundaries
        .copy_from_host(
            &offsets
                .iter()
                .flat_map(|x| x.to_le_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap();
    let prepare = |out: &mut Tensor, workspace| {
        let mut args =
            ops::SegmentedAttentionArgs::new(&query, &key, &value, out, &boundaries, &offsets);
        args.numerics = ops::AttentionNumerics::StagedBf16;
        args.scale = 1000.;
        args.policy.online_tune = false;
        args.policy.cache_dir = Some(String::new());
        args.policy.workspace_limit = workspace;
        let normalized = crate::ops::attention::normalize_segmented_attention(&ctx, args).unwrap();
        crate::ops::attention::execution::prepare(&ctx, normalized).unwrap()
    };
    let bytes = 2 * heads * 128 * 131 * 4;
    let legacy = prepare(&mut legacy_out, bytes - 1);
    assert!(
        legacy.summary().contains("config=1 workspace=0"),
        "{}",
        legacy.summary()
    );
    legacy.enqueue_for_test().unwrap();
    ctx.synchronize().unwrap();
    let expected = read(&legacy_out);
    assert!(expected.iter().all(|x| x.to_f32().is_finite()));
    drop(legacy);
    // Warm module loading before sampling free memory. No native workspace is
    // retained by an inactive execution session.
    let warm = prepare(&mut tiled_out, bytes);
    warm.enqueue_for_test().unwrap();
    ctx.synchronize().unwrap();
    drop(warm);
    ctx.synchronize().unwrap();
    let before = ctx.memory_info().unwrap().0;
    for _ in 0..4 {
        let tiled = prepare(&mut tiled_out, bytes);
        assert!(
            tiled
                .summary()
                .contains(&format!("config=0 workspace={bytes}")),
            "{}",
            tiled.summary()
        );
        let pointer = tiled.workspace_address_for_test();
        assert_eq!(
            unsafe {
                crate::ffi::abi::attention::apxinf_attention_test_is_live_device_pointer(pointer)
            },
            1
        );
        tiled.enqueue_for_test().unwrap();
        ctx.synchronize().unwrap();
        assert_eq!(read(&tiled_out), expected);
        drop(tiled);
        assert_eq!(
            unsafe {
                crate::ffi::abi::attention::apxinf_attention_test_is_live_device_pointer(pointer)
            },
            0,
            "native workspace remains allocated after execution drop"
        );
    }
    ctx.synchronize().unwrap();
    let after = ctx.memory_info().unwrap().0;
    eprintln!("EAGER_RELEASE system_free_before={before} system_free_after={after}");
    for cycle in 0..4 {
        let mut weak = None;
        let mut pointer = std::ptr::null();
        let phase = PreparedPhase::prepare_and_capture(
            &ctx,
            ExecutionSession::with_capacity(1024 * 1024, 0).unwrap(),
            || {
                let mut out = tiled_out.clone();
                let tiled = prepare(&mut out, bytes);
                assert!(tiled.summary().contains("config=0 "));
                weak = Some(std::rc::Rc::downgrade(&tiled));
                pointer = tiled.workspace_address_for_test();
                tiled.enqueue_for_test()
            },
        )
        .unwrap();
        for _ in 0..3 {
            phase.replay().unwrap();
        }
        ctx.synchronize().unwrap();
        assert_eq!(read(&tiled_out), expected);
        assert_eq!((read(&query), read(&key), read(&value)), original);
        let guards = read(&guarded);
        assert_eq!(&guards[..16], &[half::bf16::from_f32(42.); 16]);
        assert_eq!(&guards[n + 16..], &[half::bf16::from_f32(42.); 16]);
        drop(phase);
        ctx.synchronize().unwrap();
        assert!(
            weak.unwrap().upgrade().is_none(),
            "graph retained native execution after drop"
        );
        let graph_after = ctx.memory_info().unwrap().0;
        eprintln!("GRAPH_RELEASE cycle={cycle} before={before} after={graph_after}");
        assert_eq!(
            unsafe {
                crate::ffi::abi::attention::apxinf_attention_test_is_live_device_pointer(pointer)
            },
            0,
            "graph workspace remains allocated after phase drop"
        );
    }
}

#[test]
fn gpu_staged_tiled_dispatch_boundaries_and_rejected_offsets() {
    let ctx = CudaContext::new_with_recipe_options(0, false, Some(String::new())).unwrap();
    for (tokens, dim, config) in [
        (127, 7, 1),
        (128, 7, 0),
        (128, 256, 0),
        (128, 257, 1),
        (4096, 7, 0),
        (4097, 7, 1),
    ] {
        let data = tensor(&ctx, vec![tokens, 1, dim], &vec![0.; tokens * dim]);
        let mut out = tensor(&ctx, vec![tokens, 1, dim], &vec![0.; tokens * dim]);
        let offsets = [0u32, tokens as u32];
        let boundaries = CudaBuffer::alloc(8, 0).unwrap();
        boundaries
            .copy_from_host(
                &offsets
                    .iter()
                    .flat_map(|x| x.to_le_bytes())
                    .collect::<Vec<_>>(),
            )
            .unwrap();
        let mut args =
            ops::SegmentedAttentionArgs::new(&data, &data, &data, &mut out, &boundaries, &offsets);
        args.numerics = ops::AttentionNumerics::StagedBf16;
        args.policy.online_tune = false;
        args.policy.cache_dir = Some(String::new());
        let normalized = crate::ops::attention::normalize_segmented_attention(&ctx, args).unwrap();
        let execution = crate::ops::attention::execution::prepare(&ctx, normalized).unwrap();
        assert!(
            execution.summary().contains(&format!("config={config} ")),
            "tokens={tokens} dim={dim}: {}",
            execution.summary()
        );
        // Host metadata outside the declared tensor extent is rejected before
        // any candidate is prepared or any device workspace is allocated.
        let bad = [0u32, tokens as u32 + 1];
        let mut args =
            ops::SegmentedAttentionArgs::new(&data, &data, &data, &mut out, &boundaries, &bad);
        args.numerics = ops::AttentionNumerics::StagedBf16;
        assert!(crate::ops::attention::normalize_segmented_attention(&ctx, args).is_err());
    }
}
