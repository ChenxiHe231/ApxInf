//! CUDA-owned storage for the portable [`apxinf_core::KvCache`] interface.
//!
//! This is deliberately a storage/lifetime adapter.  The canonical physical
//! layout is the one consumed by cuda-new L3 KV-cache Attention:
//! `[batch=1, capacity, kv_heads, head_dim]`.  Candidate selection remains in
//! the L3 Attention implementation.

use apxinf_core::{DType, Device, Error, KvCache, Result, Shape, Tensor};
use std::sync::atomic::{AtomicU8, Ordering};

use crate::{ffi, CudaBuffer, CudaContext};

const MAX_ELEMENT_BYTES: usize = std::mem::size_of::<f32>();

pub struct CudaKVCache {
    k_buffers: Vec<CudaBuffer>,
    v_buffers: Vec<CudaBuffer>,
    n_kv_heads: usize,
    head_dim: usize,
    max_seq_len: usize,
    seq_len: usize,
    device_id: usize,
    dtype: AtomicU8,
}

impl CudaKVCache {
    pub fn new(
        device_id: usize,
        n_layers: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
    ) -> Result<Self> {
        if n_layers == 0 || n_kv_heads == 0 || head_dim == 0 || max_seq_len == 0 {
            return Err(Error::Other(
                "CUDA KV cache dimensions must all be non-zero".into(),
            ));
        }
        let layer_elements = n_kv_heads
            .checked_mul(head_dim)
            .and_then(|count| count.checked_mul(max_seq_len))
            .ok_or_else(|| Error::Other("CUDA KV cache size overflow".into()))?;
        // The portable Backend factory does not carry a dtype.  Reserve the
        // largest supported element width once, then bind the exact logical
        // prefix after the first append establishes F16 or BF16.
        let layer_bytes = layer_elements
            .checked_mul(MAX_ELEMENT_BYTES)
            .ok_or_else(|| Error::Other("CUDA KV cache byte size overflow".into()))?;
        let allocate = || CudaBuffer::alloc_zeros(layer_bytes, device_id).map_err(Error::Cuda);
        let k_buffers = (0..n_layers)
            .map(|_| allocate())
            .collect::<Result<Vec<_>>>()?;
        let v_buffers = (0..n_layers)
            .map(|_| allocate())
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            k_buffers,
            v_buffers,
            n_kv_heads,
            head_dim,
            max_seq_len,
            seq_len: 0,
            device_id,
            dtype: AtomicU8::new(0),
        })
    }

    pub fn device_id(&self) -> usize {
        self.device_id
    }

    pub fn n_kv_heads(&self) -> usize {
        self.n_kv_heads
    }

    pub fn head_dim(&self) -> usize {
        self.head_dim
    }

    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }

    pub fn dtype(&self) -> Option<DType> {
        match self.dtype.load(Ordering::Acquire) {
            1 => Some(DType::F16),
            2 => Some(DType::BF16),
            _ => None,
        }
    }

    pub fn k_buffer(&self, layer_idx: usize) -> &CudaBuffer {
        &self.k_buffers[layer_idx]
    }

    pub fn v_buffer(&self, layer_idx: usize) -> &CudaBuffer {
        &self.v_buffers[layer_idx]
    }

    fn logical_bytes(&self, dtype: DType) -> Result<usize> {
        self.max_seq_len
            .checked_mul(self.n_kv_heads)
            .and_then(|count| count.checked_mul(self.head_dim))
            .and_then(|count| count.checked_mul(dtype.size_in_bytes()))
            .ok_or_else(|| Error::Other("CUDA KV cache logical size overflow".into()))
    }

    fn layer_tensor(&self, layer_idx: usize, key: bool, dtype: DType) -> Result<Tensor> {
        let buffers = if key {
            &self.k_buffers
        } else {
            &self.v_buffers
        };
        let buffer = buffers.get(layer_idx).ok_or_else(|| {
            Error::Other(format!(
                "CUDA KV cache layer {layer_idx} is out of range for {} layers",
                buffers.len()
            ))
        })?;
        let bytes = self.logical_bytes(dtype)?;
        let view = buffer.view(0, bytes).map_err(Error::Cuda)?;
        view.as_tensor(
            Shape::new(vec![1, self.max_seq_len, self.n_kv_heads, self.head_dim]),
            dtype,
        )
        .map_err(Error::Cuda)
    }

    pub(crate) fn key_tensor(&self, layer_idx: usize, dtype: DType) -> Result<Tensor> {
        self.layer_tensor(layer_idx, true, dtype)
    }

    pub(crate) fn value_tensor(&self, layer_idx: usize, dtype: DType) -> Result<Tensor> {
        self.layer_tensor(layer_idx, false, dtype)
    }

    /// Append token-major K/V rows without introducing an operator-selection
    /// path.  This is a stream-ordered storage copy; Attention itself is
    /// dispatched exclusively through cuda-new L3.
    pub fn append(
        &self,
        ctx: &CudaContext,
        layer_idx: usize,
        k: &Tensor,
        v: &Tensor,
        append_len: usize,
    ) -> Result<()> {
        if ctx.device_id() != self.device_id {
            return Err(Error::DeviceMismatch {
                expected: Device::Cuda(self.device_id),
                got: Device::Cuda(ctx.device_id()),
            });
        }
        if layer_idx >= self.k_buffers.len() {
            return Err(Error::Other(format!(
                "CUDA KV cache layer {layer_idx} is out of range for {} layers",
                self.k_buffers.len()
            )));
        }
        if append_len == 0 {
            return Err(Error::Other(
                "CUDA KV append length must be non-zero".into(),
            ));
        }
        let end = self
            .seq_len
            .checked_add(append_len)
            .ok_or_else(|| Error::Other("CUDA KV cache position overflow".into()))?;
        if end > self.max_seq_len {
            return Err(Error::Other(format!(
                "CUDA KV append ends at {end}, beyond capacity {}",
                self.max_seq_len
            )));
        }
        let dtype = k.dtype();
        if !matches!(dtype, DType::F16 | DType::BF16) || v.dtype() != dtype {
            return Err(Error::Other(
                "cuda-new KV cache supports matching F16 or BF16 K/V".into(),
            ));
        }
        if let Some(cache_dtype) = self.dtype() {
            if cache_dtype != dtype {
                return Err(Error::DTypeMismatch {
                    expected: cache_dtype,
                    got: dtype,
                });
            }
        }
        let expected_shape = [append_len, self.n_kv_heads, self.head_dim];
        for tensor in [k, v] {
            if tensor.device() != Device::Cuda(self.device_id) {
                return Err(Error::DeviceMismatch {
                    expected: Device::Cuda(self.device_id),
                    got: tensor.device(),
                });
            }
            if tensor.shape().dims() != expected_shape {
                return Err(Error::ShapeMismatch {
                    expected: format!("[{append_len}, {}, {}]", self.n_kv_heads, self.head_dim),
                    got: tensor.shape().to_string(),
                });
            }
        }
        let row_elements = self
            .n_kv_heads
            .checked_mul(self.head_dim)
            .ok_or_else(|| Error::Other("CUDA KV row size overflow".into()))?;
        let byte_offset = self
            .seq_len
            .checked_mul(row_elements)
            .and_then(|count| count.checked_mul(dtype.size_in_bytes()))
            .ok_or_else(|| Error::Other("CUDA KV append offset overflow".into()))?;
        let copy_bytes = append_len
            .checked_mul(row_elements)
            .and_then(|count| count.checked_mul(dtype.size_in_bytes()))
            .ok_or_else(|| Error::Other("CUDA KV append size overflow".into()))?;
        let k_source = CudaBuffer::from_tensor(k).map_err(Error::Cuda)?;
        let v_source = CudaBuffer::from_tensor(v).map_err(Error::Cuda)?;
        let dtype_code = if dtype == DType::F16 { 1 } else { 2 };
        match self
            .dtype
            .compare_exchange(0, dtype_code, Ordering::AcqRel, Ordering::Acquire)
        {
            Ok(_) => {}
            Err(current) if current == dtype_code => {}
            Err(_) => {
                return Err(Error::Other(
                    "CUDA KV cache dtype changed during append".into(),
                ))
            }
        }
        for (destination, source) in [
            (&self.k_buffers[layer_idx], &k_source),
            (&self.v_buffers[layer_idx], &v_source),
        ] {
            let destination = destination
                .view(byte_offset, copy_bytes)
                .map_err(Error::Cuda)?;
            unsafe {
                ffi::check_cuda(ffi::cudaMemcpyAsync(
                    destination.ptr(),
                    source.ptr(),
                    copy_bytes,
                    ffi::cudaMemcpyKind::cudaMemcpyDeviceToDevice,
                    ctx.stream().handle(),
                ))
                .map_err(Error::Cuda)?;
            }
        }
        Ok(())
    }
}

impl KvCache for CudaKVCache {
    fn append(
        &mut self,
        _layer_idx: usize,
        _k: &Tensor,
        _v: &Tensor,
        _append_len: usize,
    ) -> Result<()> {
        Err(Error::Other(
            "CUDA KV append requires the backend stream; call Backend::kv_append".into(),
        ))
    }

    fn advance(&mut self, n: usize) {
        self.seq_len = self.seq_len.saturating_add(n);
    }

    fn seq_len(&self) -> usize {
        self.seq_len
    }

    fn clear(&mut self) -> Result<()> {
        unsafe {
            ffi::check_cuda(ffi::cudaSetDevice(
                i32::try_from(self.device_id)
                    .map_err(|_| Error::Other("CUDA device id exceeds i32".into()))?,
            ))
            .map_err(Error::Cuda)?;
            for buffer in self.k_buffers.iter().chain(&self.v_buffers) {
                ffi::check_cuda(ffi::cudaMemset(buffer.ptr(), 0, buffer.len()))
                    .map_err(Error::Cuda)?;
            }
        }
        self.seq_len = 0;
        self.dtype.store(0, Ordering::Release);
        Ok(())
    }

    fn n_layers(&self) -> usize {
        self.k_buffers.len()
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn std::any::Any {
        self
    }
}
