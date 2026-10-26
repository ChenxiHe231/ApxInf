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
