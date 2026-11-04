//! Diagnostic-only default-H2D/non-blocking-stream ordering probes.
//! These tests do not certify model quality or performance, whether or not a
//! race reproduces. Run ignored tests serially under /tmp/apxinf-gpu.lock.

use std::ffi::c_void;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use apxinf_core::{DType, Shape, Tensor};
use apxinf_cuda_new::{transfers, CudaBuffer, CudaContext, CudaStream};

// Test-local declaration only; no production FFI/public API changes. Linux's
// CUDART_CB is the C ABI. The crate already links this same CUDA runtime.
#[link(name = "cudart")]
extern "C" {
    fn cudaLaunchHostFunc(
        stream: *mut c_void,
        callback: unsafe extern "C" fn(*mut c_void),
        user_data: *mut c_void,
    ) -> i32;
}

const CALLBACK_MS: u64 = 50;

struct CallbackState {
    origin: Instant,
    started_us: AtomicU64,
    finished_us: AtomicU64,
    completed: AtomicBool,
}

fn micros(duration: Duration) -> u64 {
    duration.as_micros().min(u128::from(u64::MAX)) as u64
}

// No CUDA APIs, locks, allocations, main-thread handshakes, or panics here.
// Sleep has a fixed requested duration; OS scheduling can extend elapsed time.
unsafe extern "C" fn delay_callback(pointer: *mut c_void) {
    let state = unsafe { &*(pointer as *const CallbackState) };
    state.started_us.store(micros(state.origin.elapsed()), Ordering::Release);
    std::thread::sleep(Duration::from_millis(CALLBACK_MS));
    state.finished_us.store(micros(state.origin.elapsed()), Ordering::Release);
    state.completed.store(true, Ordering::Release);
}

// The callback borrows this heap allocation. Even on an early Result/panic,
// drain the owning stream before releasing it. If drain fails, leak the tiny
// state rather than free userdata that a still-running callback might access.
struct PendingCallback<'a> {
    ctx: &'a CudaContext,
    state: Option<Box<CallbackState>>,
    queued: bool,
}

impl<'a> PendingCallback<'a> {
    fn enqueue(ctx: &'a CudaContext) -> Result<Self, String> {
        let mut pending = Self {
            ctx,
            state: Some(Box::new(CallbackState {
                origin: Instant::now(),
                started_us: AtomicU64::new(0),
                finished_us: AtomicU64::new(0),
                completed: AtomicBool::new(false),
            })),
            queued: false,
        };
        let pointer = pending.state.as_deref_mut().unwrap() as *mut CallbackState;
        let status = unsafe {
            cudaLaunchHostFunc(ctx.stream().handle(), delay_callback, pointer.cast())
        };
        if status != 0 {
            return Err(format!("cudaLaunchHostFunc returned {status}"));
        }
        pending.queued = true;
        Ok(pending)
    }

    fn drain(&mut self) -> Result<(), String> {
        self.ctx.synchronize()?;
        self.queued = false;
        if !self.state.as_ref().unwrap().completed.load(Ordering::Acquire) {
            return Err("callback incomplete after successful ctx synchronize".into());
        }
        Ok(())
    }
}

impl Drop for PendingCallback<'_> {
    fn drop(&mut self) {
        if self.queued && self.ctx.synchronize().is_err() {
            if let Some(state) = self.state.take() {
                let _ = Box::leak(state);
            }
        }
    }
}

fn tensor(buffer: &CudaBuffer, elements: usize) -> Result<Tensor, String> {
    buffer.as_tensor(Shape::new(vec![elements]), DType::I32)
}

fn digest(values: &[i32]) -> String {
    let mut hash = 0xcbf29ce484222325u64;
    for byte in values.iter().flat_map(|value| value.to_le_bytes()) {
        hash = (hash ^ u64::from(byte)).wrapping_mul(0x100000001b3);
    }
    format!("{hash:016x}")
}

fn pending_zero_case(ctx: &CudaContext, fenced: bool) -> Result<(), String> {
    // Allocate/initialize everything before callback submission. No device
    // malloc/free/readback is inserted between pending zero, H2D and consumer.
    let destination = CudaBuffer::alloc(4, ctx.device_id())?;
    let observed = CudaBuffer::alloc(4, ctx.device_id())?;
    let destination_tensor = tensor(&destination, 1)?;
    let observed_tensor = tensor(&observed, 1)?;
    let host = Tensor::from_i32(vec![1], &[123]).map_err(|error| error.to_string())?;
    destination.zero_async(ctx.stream())?;
    observed.zero_async(ctx.stream())?;
    ctx.synchronize()?;

    let total_started = Instant::now();
    let mut callback = PendingCallback::enqueue(ctx)?;
    // This is the old write, queued behind the bounded callback on ctx.
    destination.zero_async(ctx.stream())?;
    let before_wait = Instant::now();
    if fenced {
        callback.drain()?; // prior ctx zero -> H2D
    }
    let prior_ctx_wait_us = micros(before_wait.elapsed());
    let upload_started = Instant::now();
    transfers::copy_cpu_to_cuda(&host, &destination_tensor)
        .map_err(|error| error.to_string())?;
    let upload_us = micros(upload_started.elapsed());
    let upload_return_since_callback_us = micros(callback.state.as_ref().unwrap().origin.elapsed());
    let callback_done_at_upload_return = callback.state.as_ref().unwrap().completed.load(Ordering::Acquire);
    let after_wait = Instant::now();
    if fenced {
        CudaStream::default_stream().synchronize()?; // H2D -> ctx consumer
    }
    let default_wait_us = micros(after_wait.elapsed());
    // The same-stream D2D is the consumer: it freezes what ctx sees before a
    // later readback's full-device fence can conceal a late producer.
    transfers::copy_cuda_to_cuda(ctx, &destination_tensor, &observed_tensor)
        .map_err(|error| error.to_string())?;
    let final_wait = Instant::now();
    callback.drain()?;
    let final_ctx_wait_us = micros(final_wait.elapsed());
    let values = transfers::to_cpu(&observed_tensor)
        .map_err(|error| error.to_string())?
        .as_i32().map_err(|error| error.to_string())?.to_vec();
    assert_eq!(values.len(), 1);
    assert_eq!(host.as_i32().map_err(|error| error.to_string())?, &[123]);
    let status = if fenced && values == [123] {
        "ordered_control_exact"
    } else if values == [123] {
        "not_reproduced_on_this_device_not_a_safety_proof"
    } else if values == [0] {
        "observed_zero_ordering_mismatch"
    } else {
        "observed_other_ordering_mismatch"
    };
    let state = callback.state.as_ref().unwrap();
    println!(
        "{{\"probe\":\"pending_zero\",\"diagnostic_only\":true,\"quality_pass\":false,\"performance_accepted\":false,\"fenced_control\":{fenced},\"status\":\"{status}\",\"value\":{},\"requested_callback_ms\":{CALLBACK_MS},\"callback_started_us\":{},\"callback_finished_us\":{},\"callback_done_at_upload_return\":{callback_done_at_upload_return},\"upload_return_since_callback_us\":{upload_return_since_callback_us},\"upload_us\":{upload_us},\"prior_ctx_wait_us\":{prior_ctx_wait_us},\"default_wait_us\":{default_wait_us},\"final_ctx_wait_us\":{final_ctx_wait_us},\"total_us\":{}}}",
        values[0], state.started_us.load(Ordering::Acquire),
        state.finished_us.load(Ordering::Acquire), micros(total_started.elapsed()),
    );
    if fenced {
        assert_eq!(values, [123], "the conservative ordered control must be exact");
    }
    Ok(())
}

#[test]
#[ignore = "GPU diagnostic; exclusive GPU lock, --nocapture, and external timeout required"]
fn qwen35_pageable_h2d_pending_zero_probe() {
    let ctx = CudaContext::new_with_autotune(0, false).unwrap();
    pending_zero_case(&ctx, false).unwrap();
    pending_zero_case(&ctx, true).unwrap();
}

#[test]
#[ignore = "GPU diagnostic; no pending zero, observed mismatches are reported without quality acceptance"]
fn qwen35_pageable_h2d_reused_upload_probe() {
    let ctx = CudaContext::new_with_autotune(0, false).unwrap();
    let mut mismatched_rounds = 0usize;
    for elements in [1usize, 16, 4096] {
        let destination = CudaBuffer::alloc(elements * 4, ctx.device_id()).unwrap();
        let observed = CudaBuffer::alloc(elements * 4, ctx.device_id()).unwrap();
        let destination_tensor = tensor(&destination, elements).unwrap();
        let observed_tensor = tensor(&observed, elements).unwrap();
        destination.zero_async(ctx.stream()).unwrap();
        observed.zero_async(ctx.stream()).unwrap();
        ctx.synchronize().unwrap();
        for round in 0..20usize {
            let expected: Vec<i32> = (0..elements)
                .map(|index| 123 + (round as i32 * 65537) + index as i32)
                .collect();
            let host = Tensor::from_i32(vec![elements], &expected).unwrap();
            let started = Instant::now();
            transfers::copy_cpu_to_cuda(&host, &destination_tensor).unwrap();
            let upload_us = micros(started.elapsed());
            transfers::copy_cuda_to_cuda(&ctx, &destination_tensor, &observed_tensor).unwrap();
            ctx.synchronize().unwrap();
            let actual = transfers::to_cpu(&observed_tensor).unwrap().as_i32().unwrap().to_vec();
            assert_eq!(actual.len(), elements);
            assert_eq!(host.as_i32().unwrap(), expected.as_slice(), "CPU input changed");
            let mismatch_count = actual.iter().zip(&expected).filter(|(a, b)| a != b).count();
            mismatched_rounds += usize::from(mismatch_count != 0);
            println!(
                "{{\"probe\":\"reused_upload\",\"diagnostic_only\":true,\"quality_pass\":false,\"performance_accepted\":false,\"elements\":{elements},\"round\":{round},\"input_digest_fnv1a64\":\"{}\",\"observed_digest_fnv1a64\":\"{}\",\"mismatch_count\":{mismatch_count},\"upload_us\":{upload_us},\"total_us\":{}}}",
                digest(&expected), digest(&actual), micros(started.elapsed()),
            );
        }
    }
    println!("{{\"probe\":\"reused_upload_summary\",\"rounds\":60,\"mismatched_rounds\":{mismatched_rounds},\"diagnostic_only\":true,\"quality_pass\":false,\"performance_accepted\":false}}");
}

#[test]
#[ignore = "GPU correctness regression; run serially under the shared GPU lock"]
fn qwen35_ctx_h2d_pending_zero_is_ordered() {
    let ctx = CudaContext::new_with_autotune(0, false).unwrap();
    // No destination initialization before callback: this models initial
    // ctx-stream alloc_zeros_async followed immediately by a CPU upload.
    let destination = CudaBuffer::alloc(4, ctx.device_id()).unwrap();
    let observed = CudaBuffer::alloc(4, ctx.device_id()).unwrap();
    let destination_tensor = tensor(&destination, 1).unwrap();
    let observed_tensor = tensor(&observed, 1).unwrap();
    let mut callback = PendingCallback::enqueue(&ctx).unwrap();
    destination.zero_async(ctx.stream()).unwrap();
    let started = Instant::now();
    // A temporary CPU Tensor is moved into the helper; the helper owns its
    // allocation until the copy completes, even though no caller retains it.
    transfers::copy_cpu_to_cuda_on_stream(
        &ctx, Tensor::from_i32(vec![1], &[123]).unwrap(), &destination_tensor,
    ).unwrap();
    let helper_us = micros(started.elapsed());
    assert!(callback.state.as_ref().unwrap().completed.load(Ordering::Acquire));
    transfers::copy_cuda_to_cuda(&ctx, &destination_tensor, &observed_tensor).unwrap();
    callback.drain().unwrap();
    let actual = transfers::to_cpu(&observed_tensor).unwrap().as_i32().unwrap().to_vec();
    assert_eq!(actual, [123], "ctx upload must follow the delayed initial zero");
    let state = callback.state.as_ref().unwrap();
    println!(
        "{{\"probe\":\"fixed_ctx_h2d_pending_zero\",\"value\":123,\"helper_us\":{helper_us},\"callback_started_us\":{},\"callback_finished_us\":{},\"performance_accepted\":false}}",
        state.started_us.load(Ordering::Acquire), state.finished_us.load(Ordering::Acquire),
    );
}

#[test]
#[ignore = "GPU correctness regression for temporary source completion and repeated scratch writes"]
fn qwen35_ctx_h2d_temporary_source_is_complete_and_reusable() {
    let ctx = CudaContext::new_with_autotune(0, false).unwrap();
    for elements in [1usize, 16, 4096] {
        let destination = CudaBuffer::alloc(elements * 4, ctx.device_id()).unwrap();
        let observed = CudaBuffer::alloc(elements * 4, ctx.device_id()).unwrap();
        let destination_tensor = tensor(&destination, elements).unwrap();
        let observed_tensor = tensor(&observed, elements).unwrap();
        for round in 0..20usize {
            destination.zero_async(ctx.stream()).unwrap();
            let expected: Vec<i32> = (0..elements)
                .map(|index| 123 + (round as i32 * 65537) + index as i32).collect();
            let started = Instant::now();
            transfers::copy_cpu_to_cuda_on_stream(
                &ctx, Tensor::from_i32(vec![elements], &expected).unwrap(), &destination_tensor,
            ).unwrap();
            // Host staging has already been dropped inside the synchronous
            // helper. Consume only after that lifetime boundary.
            transfers::copy_cuda_to_cuda(&ctx, &destination_tensor, &observed_tensor).unwrap();
            ctx.synchronize().unwrap();
            let actual = transfers::to_cpu(&observed_tensor).unwrap().as_i32().unwrap().to_vec();
            assert_eq!(actual, expected, "elements={elements} round={round}");
            println!(
                "{{\"probe\":\"fixed_ctx_h2d_reuse\",\"elements\":{elements},\"round\":{round},\"expected_digest\":\"{}\",\"actual_digest\":\"{}\",\"total_us\":{},\"performance_accepted\":false}}",
                digest(&expected), digest(&actual), micros(started.elapsed()),
            );
        }
    }
    // The allocating convenience upload also consumes temporary CPU staging.
    let uploaded = transfers::to_cuda_on_stream(
        &ctx, Tensor::from_i32(vec![2], &[321, 654]).unwrap(),
    ).unwrap();
    assert_eq!(transfers::to_cpu(&uploaded).unwrap().as_i32().unwrap(), &[321, 654]);
}

#[test]
#[ignore = "GPU capture-contract regression; upload is explicitly rejected, never captured with temporary host pointers"]
fn qwen35_ctx_h2d_rejects_capture_and_recovers() {
    let ctx = CudaContext::new_with_autotune(0, false).unwrap();
    let destination = transfers::to_cuda_on_stream(
        &ctx, Tensor::from_i32(vec![1], &[123]).unwrap(),
    ).unwrap();
    for allocating in [false, true] {
        let rejected = apxinf_cuda_new::capture(&ctx, || {
            let host = Tensor::from_i32(vec![1], &[456])?;
            if allocating {
                transfers::to_cuda_on_stream(&ctx, host).map(|_| ())
            } else {
                transfers::copy_cpu_to_cuda_on_stream(&ctx, host, &destination)
            }
        });
        let message = rejected.err().expect("capture must reject temporary CPU upload").to_string();
        assert!(message.contains("not supported during CUDA Graph capture"), "{message}");
        ctx.synchronize().unwrap();
        assert_eq!(transfers::to_cpu(&destination).unwrap().as_i32().unwrap(), &[123]);
    }
    transfers::copy_cpu_to_cuda_on_stream(
        &ctx, Tensor::from_i32(vec![1], &[789]).unwrap(), &destination,
    ).unwrap();
    assert_eq!(transfers::to_cpu(&destination).unwrap().as_i32().unwrap(), &[789]);
}
