//! Safe CUDA tensor transfers and bounds-checked device copy primitives.

use apxinf_core::{Device, Error, Result, Storage, Tensor};

use crate::ffi;
use crate::CudaBuffer;

/// Transfer a CPU tensor to a CUDA device.
pub fn to_cuda(tensor: &Tensor, device_id: usize) -> Result<Tensor> {
    if tensor.device() != Device::Cpu {
        return Err(Error::Other("tensor is already on GPU".into()));
    }
    let bytes = tensor
        .storage()
        .as_cpu()
        .ok_or_else(|| Error::Other("expected CPU storage".into()))?;
    let buffer = CudaBuffer::alloc(bytes.len(), device_id).map_err(Error::Cuda)?;
    buffer.copy_from_host(bytes).map_err(Error::Cuda)?;
    Ok(buffer.into_tensor(tensor.shape().clone(), tensor.dtype()))
}

/// Copy a CPU tensor into shape-identical, stable-address CUDA storage.
pub fn copy_cpu_to_cuda(source: &Tensor, destination: &Tensor) -> Result<()> {
    if source.device() != Device::Cpu {
        return Err(Error::Other("copy source must be a CPU tensor".into()));
    }
    let device_id = match destination.device() {
        Device::Cuda(device_id) => device_id,
        device => return Err(Error::UnsupportedDevice(device)),
    };
    if source.shape() != destination.shape() || source.dtype() != destination.dtype() {
        return Err(Error::Other(format!(
            "fixed CUDA input mismatch: source {:?} {}, destination {:?} {}",
            source.shape().dims(),
            source.dtype(),
            destination.shape().dims(),
            destination.dtype()
        )));
    }
    let source = source
        .storage()
        .as_cpu()
        .ok_or_else(|| Error::Other("expected CPU storage".into()))?;
    let destination = destination
        .storage()
        .as_gpu()
        .ok_or_else(|| Error::Other("expected CUDA storage".into()))?;
    copy_host_to_device(device_id, source, destination.ptr()).map_err(Error::Cuda)
}

/// Upload an owned CPU tensor on the context stream and wait for completion.
///
/// Ownership keeps temporary host bytes alive until DMA finishes. This is a
/// conservative, synchronous host boundary, not a capture-safe async staging
/// API. It rejects CUDA Graph capture before allocating or submitting work.
/// Existing default-stream transfer functions keep their original behavior.
pub fn to_cuda_on_stream(ctx: &crate::CudaContext, source: Tensor) -> Result<Tensor> {
    if source.device() != Device::Cpu {
        return Err(Error::Other("copy source must be a CPU tensor".into()));
    }
    ensure_upload_outside_capture(ctx)?;
    let bytes = source.storage().as_cpu()
        .ok_or_else(|| Error::Other("expected CPU storage".into()))?.len();
    let buffer = CudaBuffer::alloc(bytes, ctx.device_id()).map_err(Error::Cuda)?;
    let destination = buffer.into_tensor(source.shape().clone(), source.dtype());
    copy_cpu_to_cuda_on_stream(ctx, source, &destination)?;
    Ok(destination)
}

/// Copy owned CPU staging into fixed CUDA storage on the context stream.
///
/// Prior writes/reads, this H2D, and subsequent kernels share stream order.
/// This function waits for that stream before dropping the CPU source or
/// returning success. On a submission/completion error it conservatively
/// retains both source and destination owners: an error cannot prove pending
/// DMA has stopped. Capture is explicitly unsupported; callers must upload
/// changing inputs outside capture, then capture only device-side consumers.
pub fn copy_cpu_to_cuda_on_stream(
    ctx: &crate::CudaContext,
    source: Tensor,
    destination: &Tensor,
) -> Result<()> {
    validate_cpu_upload(&source, destination, ctx.device_id())?;
    ensure_upload_outside_capture(ctx)?;
    let destination = CudaBuffer::from_tensor(destination).map_err(Error::Cuda)?;
    let bytes = source.storage().as_cpu()
        .ok_or_else(|| Error::Other("expected CPU storage".into()))?;
    if bytes.len() != destination.len() {
        return Err(Error::Other("fixed CUDA input storage length mismatch".into()));
    }
    let submitted = ctx.stream().with_current_device(|| unsafe {
        ffi::check_cuda(ffi::cudaMemcpyAsync(
            destination.ptr(), bytes.as_ptr().cast(), bytes.len(),
            ffi::cudaMemcpyKind::cudaMemcpyHostToDevice, ctx.stream().handle(),
        ))
    });
    if let Err(error) = submitted.and_then(|()| ctx.synchronize()) {
        // A CUDA error can describe earlier asynchronous work. Do not infer
        // from it that this copy did not queue or may no longer access memory.
        std::mem::forget((source, destination));
        return Err(Error::Cuda(error));
    }
    Ok(())
}

fn validate_cpu_upload(source: &Tensor, destination: &Tensor, device_id: usize) -> Result<()> {
    if source.device() != Device::Cpu {
        return Err(Error::Other("copy source must be a CPU tensor".into()));
    }
    if destination.device() != Device::Cuda(device_id) {
        return Err(Error::Other("fixed CUDA copy destination does not match context device".into()));
    }
    if source.shape() != destination.shape() || source.dtype() != destination.dtype() {
        return Err(Error::Other("fixed CUDA input shape/dtype mismatch".into()));
    }
    Ok(())
}

fn ensure_upload_outside_capture(ctx: &crate::CudaContext) -> Result<()> {
    if crate::workspace::is_capturing() {
        return Err(Error::Other("owned CPU upload is not supported during CUDA Graph capture; upload before capture".into()));
    }
    // Also recognize captures started through a caller's raw CUDA interface,
    // outside the Rust capture-retention boundary. CUDA enum ABI is int32.
    let mut capture_status = 0i32;
    ctx.stream().with_current_device(|| unsafe {
        ffi::check_cuda(ffi::cudaStreamIsCapturing(ctx.stream().handle(), &mut capture_status))
    }).map_err(Error::Cuda)?;
    if capture_status != 0 {
        return Err(Error::Other("owned CPU upload is not supported during CUDA Graph capture; upload before capture".into()));
    }
    Ok(())
}

/// Copy a shape-identical CUDA tensor into stable-address CUDA storage on the
/// context stream. Source and destination must be on the same device.
pub fn copy_cuda_to_cuda(
    ctx: &crate::CudaContext,
    source: &Tensor,
    destination: &Tensor,
) -> Result<()> {
    if source.device() != destination.device()
        || source.device() != Device::Cuda(ctx.device_id())
        || source.shape() != destination.shape()
        || source.dtype() != destination.dtype()
    {
        return Err(Error::Other(
            "fixed CUDA copy requires matching device, shape and dtype".into(),
        ));
    }
    let source = CudaBuffer::from_tensor(source).map_err(Error::Cuda)?;
    let destination = CudaBuffer::from_tensor(destination).map_err(Error::Cuda)?;
    unsafe {
        ffi::check_cuda(ffi::cudaMemcpyAsync(
            destination.ptr(),
            source.ptr(),
            source.len(),
            ffi::cudaMemcpyKind::cudaMemcpyDeviceToDevice,
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)
    }
}

/// Copy a contiguous CUDA matrix into a column range of another row-major
/// CUDA matrix. Source and destination must share row count, dtype and device.
///
/// This is used by native MTP fusion to form
/// `[normalized_embedding, normalized_target_hidden]` without staging either
/// half on the host.
pub fn copy_cuda_columns(
    ctx: &crate::CudaContext,
    source: &Tensor,
    destination: &Tensor,
    destination_column: usize,
) -> Result<()> {
    let source_dims = source.shape().dims();
    let destination_dims = destination.shape().dims();
    if source_dims.len() != 2
        || destination_dims.len() != 2
        || source_dims[0] != destination_dims[0]
        || source.dtype() != destination.dtype()
        || source.device() != destination.device()
        || source.device() != Device::Cuda(ctx.device_id())
        || destination_column
            .checked_add(source_dims[1])
            .is_none_or(|end| end > destination_dims[1])
    {
        return Err(Error::Other(
            "CUDA column copy requires compatible row-major matrices".into(),
        ));
    }
    let element = source.dtype().size_in_bytes();
    let width = source_dims[1]
        .checked_mul(element)
        .ok_or_else(|| Error::Other("CUDA column-copy width overflow".into()))?;
    let source_pitch = width;
    let destination_pitch = destination_dims[1]
        .checked_mul(element)
        .ok_or_else(|| Error::Other("CUDA column-copy pitch overflow".into()))?;
    let source = CudaBuffer::from_tensor(source).map_err(Error::Cuda)?;
    let destination = CudaBuffer::from_tensor(destination).map_err(Error::Cuda)?;
    unsafe {
        ffi::check_cuda(ffi::cudaMemcpy2DAsync(
            destination
                .address()
                .ptr()
                .cast::<u8>()
                .add(destination_column * element)
                .cast(),
            destination_pitch,
            source.address().ptr(),
            source_pitch,
            width,
            source_dims[0],
            ffi::cudaMemcpyKind::cudaMemcpyDeviceToDevice,
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)
    }
}

/// Transfer a CUDA tensor back to CPU.
pub fn to_cpu(tensor: &Tensor) -> Result<Tensor> {
    let handle = match tensor.storage() {
        Storage::Gpu { handle, .. } => handle,
        _ => return Err(Error::Other("tensor is not on GPU".into())),
    };
    let device_id = match tensor.device() {
        Device::Cuda(device_id) => device_id,
        device => return Err(Error::UnsupportedDevice(device)),
    };
    let bytes = copy_device_to_host(device_id, handle.ptr(), handle.len()).map_err(Error::Cuda)?;
    Tensor::from_raw(tensor.shape().clone(), tensor.dtype(), Device::Cpu, bytes)
}

/// Copy `rows` strided runs of `width` device bytes from a CUDA tensor into a
/// device buffer at `destination_offset`, using `source_pitch` and
/// `destination_pitch` as the run strides.
pub(crate) fn copy_tensor_2d_to_buffer(
    ctx: &crate::CudaContext,
    source: &Tensor,
    destination: &CudaBuffer,
    destination_offset: usize,
    destination_pitch: usize,
    source_pitch: usize,
    width: usize,
    rows: usize,
) -> Result<()> {
    if rows == 0 || width == 0 {
        return Ok(());
    }
    if source.device() != Device::Cuda(ctx.device_id()) {
        return Err(Error::DeviceMismatch {
            expected: Device::Cuda(ctx.device_id()),
            got: source.device(),
        });
    }
    if destination.device() != ctx.device_id() {
        return Err(Error::Other(format!(
            "2D CUDA destination is on device {}, expected {}",
            destination.device(),
            ctx.device_id()
        )));
    }
    if width > source_pitch || width > destination_pitch {
        return Err(Error::Other(format!(
            "2D CUDA copy width {width} exceeds source/destination pitch {source_pitch}/{destination_pitch}"
        )));
    }
    let source = source
        .storage()
        .as_gpu()
        .ok_or_else(|| Error::Other("expected CUDA source tensor".into()))?;
    let source_required = rows
        .checked_mul(source_pitch)
        .ok_or_else(|| Error::Other("2D CUDA source size overflow".into()))?;
    if source_required > source.len() {
        return Err(Error::Other(format!(
            "2D CUDA source requires {source_required} bytes, has {}",
            source.len()
        )));
    }
    let destination_required = destination_offset
        .checked_add(
            rows.saturating_sub(1)
                .checked_mul(destination_pitch)
                .and_then(|offset| offset.checked_add(width))
                .ok_or_else(|| Error::Other("2D CUDA destination size overflow".into()))?,
        )
        .ok_or_else(|| Error::Other("2D CUDA destination offset overflow".into()))?;
    if destination_required > destination.len() {
        return Err(Error::Other(format!(
            "2D CUDA destination requires {destination_required} bytes, has {}",
            destination.len()
        )));
    }
    unsafe {
        ffi::check_cuda(ffi::cudaMemcpy2DAsync(
            destination
                .ptr()
                .cast::<u8>()
                .add(destination_offset)
                .cast(),
            destination_pitch,
            source.ptr() as *const std::ffi::c_void,
            source_pitch,
            width,
            rows,
            ffi::cudaMemcpyKind::cudaMemcpyDeviceToDevice,
            ctx.stream().handle(),
        ))
        .map_err(Error::Cuda)
    }
}

pub(crate) fn copy_host_to_device(
    device_id: usize,
    source: &[u8],
    destination: usize,
) -> std::result::Result<(), String> {
    let device = i32::try_from(device_id)
        .map_err(|_| format!("CUDA device id {device_id} does not fit in i32"))?;
    unsafe {
        ffi::check_cuda(ffi::cudaSetDevice(device))?;
        ffi::check_cuda(ffi::cudaMemcpy(
            destination as *mut std::ffi::c_void,
            source.as_ptr().cast(),
            source.len(),
            ffi::cudaMemcpyKind::cudaMemcpyHostToDevice,
        ))
    }
}

pub(crate) fn copy_device_to_host(
    device_id: usize,
    source: usize,
    len: usize,
) -> std::result::Result<Vec<u8>, String> {
    let device = i32::try_from(device_id)
        .map_err(|_| format!("CUDA device id {device_id} does not fit in i32"))?;
    let mut destination = vec![0u8; len];
    unsafe {
        ffi::check_cuda(ffi::cudaSetDevice(device))?;
        ffi::check_cuda(ffi::cudaDeviceSynchronize())?;
        ffi::check_cuda(ffi::cudaMemcpy(
            destination.as_mut_ptr().cast(),
            source as *const std::ffi::c_void,
            len,
            ffi::cudaMemcpyKind::cudaMemcpyDeviceToHost,
        ))?;
    }
    Ok(destination)
}

#[cfg(test)]
mod tests {
    use super::validate_cpu_upload;
    use apxinf_core::storage::GpuStorageHandle;
    use apxinf_core::{DType, Device, Shape, Storage, Tensor};

    // Empty CUDA metadata has no allocation to dereference or release. These
    // validation tests call no CUDA APIs and require no GPU context.
    fn empty_cuda(device: usize, shape: Vec<usize>, dtype: DType) -> Tensor {
        Tensor::from_raw_parts(
            Shape::new(shape), dtype, Device::Cuda(device),
            Storage::Gpu {
                device: Device::Cuda(device),
                handle: GpuStorageHandle { ptr: 0, len: 0, _prevent_leak: None },
            },
        )
    }

    #[test]
    fn ctx_upload_validates_cpu_shape_dtype_and_context_device_without_cuda() {
        let source = Tensor::from_i32(vec![0], &[]).unwrap();
        let destination = empty_cuda(0, vec![0], DType::I32);
        validate_cpu_upload(&source, &destination, 0).unwrap();
        assert!(validate_cpu_upload(&source, &destination, 1).unwrap_err()
            .to_string().contains("context device"));
        assert!(validate_cpu_upload(&source, &empty_cuda(0, vec![0, 1], DType::I32), 0)
            .unwrap_err().to_string().contains("shape/dtype"));
        assert!(validate_cpu_upload(&source, &empty_cuda(0, vec![0], DType::F32), 0)
            .unwrap_err().to_string().contains("shape/dtype"));
        assert!(validate_cpu_upload(&destination, &destination, 0).unwrap_err()
            .to_string().contains("CPU tensor"));
        assert!(validate_cpu_upload(&source, &source, 0).unwrap_err()
            .to_string().contains("context device"));
    }
}
