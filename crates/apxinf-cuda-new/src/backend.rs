//! Portable [`apxinf_core::Backend`] over the cuda-new runtime.
//!
//! Models written against `dyn Backend` — llama and qwen3-vl — compose the
//! model-neutral trait methods into an architecture. This adapter lets those
//! models run on cuda-new without a legacy backend, which is what lets the LLM
//! and VLM families finish their migration off `apxinf-cuda`.
//!
//! The trait is object-safe; [`CudaNewBackend`] is the concrete type the model
//! registry hands back through `Arc<dyn Backend>`. Numeric parity with the
//! legacy backend is a contract: every operator delegates to a cuda-new op
//! whose arithmetic matches the kernel it replaces.

use std::any::Any;

use apxinf_core::{
    Backend, DType, Device, Error, Graph, KvCache, NormalGenerator, Result, Shape, Tensor,
    TokenSampler, TokenSamplingSpec,
};

use crate::ops;
use crate::{CudaContext, CudaKvCache};

/// The cuda-new backend returned to `dyn Backend` consumers.
pub struct CudaNewBackend {
    ctx: std::sync::Arc<CudaContext>,
}

// The context owns a native runtime pointer and a CUDA stream; both are valid
// across threads for the enqueue-only operations this trait exposes, matching
// the legacy backend's `unsafe impl Send/Sync` on its stream and cuBLAS handle.
unsafe impl Send for CudaNewBackend {}
unsafe impl Sync for CudaNewBackend {}

impl CudaNewBackend {
    pub fn new(ctx: std::sync::Arc<CudaContext>) -> Self {
        Self { ctx }
    }

    pub fn context(&self) -> &CudaContext {
        &self.ctx
    }

    /// Shared context handle, for models that own a `CudaContext` themselves
    /// and sample through this backend (see qwen38).
    pub fn shared_context(&self) -> &std::sync::Arc<CudaContext> {
        &self.ctx
    }

    /// Device ordinal this backend targets.
    pub fn device_id(&self) -> usize {
        self.ctx.device_id()
    }

    /// Copy a host tensor onto this backend's device.
    pub fn to_device(&self, tensor: &Tensor) -> Result<Tensor> {
        crate::transfers::to_cuda(tensor, self.ctx.device_id())
    }

    /// Copy a device tensor back to the host.
    pub fn to_cpu(&self, tensor: &Tensor) -> Result<Tensor> {
        crate::transfers::to_cpu(tensor)
    }

    /// Block until all queued work completes.
    pub fn synchronize(&self) -> Result<()> {
        self.ctx.synchronize().map_err(Error::Cuda)
    }

    /// Capture `operation` into a CUDA graph and return both. A failing
    /// operation aborts the capture before propagating the error.
    pub fn capture_graph<T>(
        &self,
        operation: impl FnOnce() -> Result<T>,
    ) -> Result<(Box<dyn Graph>, T)> {
        struct CaptureScope<'a> {
            backend: &'a CudaNewBackend,
            active: bool,
        }
        impl Drop for CaptureScope<'_> {
            fn drop(&mut self) {
                if self.active {
                    crate::graph::abort(self.backend.context());
                }
            }
        }
        use apxinf_core::Backend as _;
        self.begin_capture()?;
        let mut scope = CaptureScope {
            backend: self,
            active: true,
        };
        let output = operation()?;
        scope.active = false;
        Ok((self.end_capture()?, output))
    }
}

impl Backend for CudaNewBackend {
    fn rms_norm(&self, input: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
        crate::kernels::norm::rms_bf16(&self.ctx, input, weight, eps)
    }

    fn silu(&self, input: &Tensor) -> Result<Tensor> {
        crate::kernels::activation::silu(&self.ctx, input)
    }

    fn add(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        // Dtype-dispatching like the legacy backend: smolvla's Xavier FP16
        // path adds residuals through `Backend::add`. BF16 keeps the exact
        // cuda-new operator route it had before.
        crate::kernels::elementwise::add(&self.ctx, a, b)
    }

    fn mul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        let dims = a.shape().clone();
        let output = self.ctx.allocate_output(dims, DType::BF16)?;
        ops::elementwise_mul(&self.ctx, a, b, &output)?;
        Ok(output)
    }

    fn scale(&self, input: &Tensor, factor: f32) -> Result<Tensor> {
        crate::kernels::elementwise::scale(&self.ctx, input, factor)
    }

    fn matmul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        crate::kernels::gemm::matmul(&self.ctx, a, b)
    }

    fn rope(
        &self,
        input: &Tensor,
        n_heads: usize,
        head_dim: usize,
        theta: f32,
        pos_offset: u32,
    ) -> Result<Tensor> {
        ops::rope_apply_batched(&self.ctx, input, n_heads, head_dim, theta, pos_offset)
    }

    fn rope_mrope(
        &self,
        input: &Tensor,
        n_heads: usize,
        head_dim: usize,
        theta: f32,
        sections: [usize; 3],
        pos_ids: &[u32],
    ) -> Result<Tensor> {
        ops::rope_apply_mrope(
            &self.ctx, input, n_heads, head_dim, theta, sections, pos_ids,
        )
    }

    fn layer_norm(
        &self,
        input: &Tensor,
        weight: &Tensor,
        bias: &Tensor,
        eps: f32,
    ) -> Result<Tensor> {
        let dims = input.shape().clone();
        let mut output = self.ctx.allocate_output(dims, DType::BF16)?;
        let args = ops::LayerNormArgs::new(input, weight, bias, &mut output, eps);
        ops::layer_norm(&self.ctx, args)?;
        Ok(output)
    }

    fn gelu_tanh(&self, input: &Tensor) -> Result<Tensor> {
        crate::kernels::activation::gelu_tanh(&self.ctx, input)
    }

    fn add_bias(&self, input: &Tensor, bias: &Tensor) -> Result<Tensor> {
        let dims = input.shape().clone();
        let output = self.ctx.allocate_output(dims, DType::BF16)?;
        ops::elementwise_add_bias(&self.ctx, input, bias, &output)?;
        Ok(output)
    }

    fn rope_vision_2d(
        &self,
        input: &Tensor,
        n_heads: usize,
        head_dim: usize,
        theta: f32,
        pos_ids: &[u32],
    ) -> Result<Tensor> {
        ops::rope_apply_vision_2d(&self.ctx, input, n_heads, head_dim, theta, pos_ids)
    }

    fn concat_2d(&self, tensors: &[&Tensor]) -> Result<Tensor> {
        use apxinf_core::Shape;
        if tensors.is_empty() {
            return Err(Error::Other("concat_2d: empty input".into()));
        }
        let dtype = tensors[0].dtype();
        let dims = tensors[0].shape().dims();
        if dims.len() != 2 {
            return Err(Error::Other(format!(
                "concat_2d: expected 2D, got {}D",
                dims.len()
            )));
        }
        let rows = dims[0];
        let total_cols = tensors.iter().try_fold(0usize, |total, tensor| {
            let dims = tensor.shape().dims();
            if dims.len() != 2 || dims[0] != rows || tensor.dtype() != dtype {
                return Err(Error::Other("concat_2d: shape/dtype mismatch".into()));
            }
            total
                .checked_add(dims[1])
                .ok_or_else(|| Error::Other("concat_2d: column count overflow".into()))
        })?;
        let elem = dtype.size_in_bytes();
        let dst_pitch = total_cols
            .checked_mul(elem)
            .ok_or_else(|| Error::Other("concat_2d: destination pitch overflow".into()))?;
        let bytes = rows
            .checked_mul(dst_pitch)
            .ok_or_else(|| Error::Other("concat_2d: output size overflow".into()))?;
        let output = crate::workspace::output_buffer(&self.ctx, bytes)?;
        let mut col_offset = 0usize;
        for tensor in tensors {
            let cols = tensor.shape().dims()[1];
            let width = cols
                .checked_mul(elem)
                .ok_or_else(|| Error::Other("concat_2d: source pitch overflow".into()))?;
            crate::transfers::copy_tensor_2d_to_buffer(
                &self.ctx,
                tensor,
                &output,
                col_offset * elem,
                dst_pitch,
                width,
                width,
                rows,
            )?;
            col_offset += cols;
        }
        Ok(output.into_tensor(Shape::new(vec![rows, total_cols]), dtype))
    }

    fn vision_sdpa(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        _seq_len: usize,
        _n_heads: usize,
        _head_dim: usize,
    ) -> Result<Tensor> {
        // [seq, heads, dim] -> [1, seq, heads, dim] dense non-causal attention.
        let q4 = q.reshape(vec![
            1,
            q.shape().dims()[0],
            q.shape().dims()[1],
            q.shape().dims()[2],
        ])?;
        let k4 = k.reshape(vec![
            1,
            k.shape().dims()[0],
            k.shape().dims()[1],
            k.shape().dims()[2],
        ])?;
        let v4 = v.reshape(vec![
            1,
            v.shape().dims()[0],
            v.shape().dims()[1],
            v.shape().dims()[2],
        ])?;
        let mut out = self.ctx.allocate_output(q4.shape().clone(), DType::BF16)?;
        ops::attention(&self.ctx, ops::AttentionArgs::new(&q4, &k4, &v4, &mut out))?;
        let dims = out.shape().dims().to_vec();
        Ok(out.reshape(vec![dims[1], dims[2] * dims[3]])?)
    }

    fn embedding(&self, table: &Tensor, ids: &[u32]) -> Result<Tensor> {
        let width = table.shape().dims()[1];
        let output = self
            .ctx
            .allocate_output(Shape::new(vec![ids.len(), width]), DType::BF16)?;
        let ids_i32: Vec<u8> = ids
            .iter()
            .flat_map(|&id| (id as i32).to_ne_bytes())
            .collect();
        let ids_buffer =
            crate::CudaBuffer::alloc(ids_i32.len(), self.ctx.device_id()).map_err(Error::Cuda)?;
        ids_buffer.copy_from_host(&ids_i32).map_err(Error::Cuda)?;
        let ids_tensor = ids_buffer
            .as_tensor(Shape::new(vec![ids.len()]), DType::I32)
            .map_err(Error::Cuda)?;
        ops::embedding_gather(&self.ctx, table, &ids_tensor, &output)?;
        Ok(output)
    }

    fn sdpa_decode(
        &self,
        q: &Tensor,
        kv: &mut dyn KvCache,
        layer_idx: usize,
        n_heads: usize,
        _n_kv_heads: usize,
        head_dim: usize,
        kv_len: usize,
        _max_seq_len: usize,
    ) -> Result<Tensor> {
        self.sdpa(q, kv, layer_idx, n_heads, head_dim, kv_len)
    }

    fn sdpa_prefill(
        &self,
        q: &Tensor,
        kv: &mut dyn KvCache,
        layer_idx: usize,
        n_heads: usize,
        _n_kv_heads: usize,
        head_dim: usize,
        kv_len: usize,
        _max_seq_len: usize,
    ) -> Result<Tensor> {
        self.sdpa(q, kv, layer_idx, n_heads, head_dim, kv_len)
    }

    fn create_kv_cache(
        &self,
        n_layers: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
    ) -> Box<dyn KvCache> {
        Box::new(
            CudaKvCache::new(
                self.ctx.clone(),
                n_layers,
                n_kv_heads,
                head_dim,
                max_seq_len,
            )
            .expect("KV cache allocation"),
        )
    }

    fn kv_append(
        &self,
        kv: &mut dyn KvCache,
        layer_idx: usize,
        k: &Tensor,
        v: &Tensor,
        append_len: usize,
    ) -> Result<()> {
        kv.append(layer_idx, k, v, append_len)
    }

    fn synchronize(&self) -> Result<()> {
        self.ctx.synchronize().map_err(Error::Cuda)
    }

    fn begin_capture(&self) -> Result<()> {
        crate::graph::begin(&self.ctx).map_err(Error::Cuda)
    }

    fn end_capture(&self) -> Result<Box<dyn Graph>> {
        let graph = crate::graph::end(&self.ctx).map_err(Error::Cuda)?;
        Ok(Box::new(graph))
    }

    fn device(&self) -> Device {
        Device::Cuda(self.ctx.device_id())
    }

    fn to_device(&self, tensor: &Tensor) -> Result<Tensor> {
        crate::transfers::to_cuda(tensor, self.ctx.device_id())
    }

    fn to_cpu(&self, tensor: &Tensor) -> Result<Tensor> {
        crate::transfers::to_cpu(tensor)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

impl CudaNewBackend {
    #[allow(clippy::too_many_arguments)]
    fn sdpa(
        &self,
        q: &Tensor,
        kv: &mut dyn KvCache,
        layer_idx: usize,
        n_heads: usize,
        head_dim: usize,
        kv_len: usize,
    ) -> Result<Tensor> {
        let cache = kv
            .as_any()
            .downcast_ref::<CudaKvCache>()
            .ok_or_else(|| Error::Other("expected a cuda-new KV cache".into()))?;
        let q_dims = q.shape().dims().to_vec();
        if q_dims.len() != 3 || q_dims[1] != n_heads || q_dims[2] != head_dim {
            return Err(Error::Other(format!(
                "attention query must be [seq, {n_heads}, {head_dim}], got {q_dims:?}"
            )));
        }
        let seq_len = q_dims[0];
        if seq_len == 0 || kv_len < seq_len {
            return Err(Error::Other(
                "attention kv_len is smaller than the query".into(),
            ));
        }
        let q4 = q.reshape(vec![1, seq_len, n_heads, head_dim])?;
        let key = cache.key(layer_idx);
        let value = cache.value(layer_idx);
        let mut out = self.ctx.allocate_output(q4.shape().clone(), DType::BF16)?;
        // Causal mask: query token i occupies cache position query_start + i,
        // so both fresh prefill (query_start 0) and decode (query_start
        // kv_len-1) use the same causal contract.
        let mut args = ops::KvCacheAttentionArgs::new(&q4, key, value, &mut out);
        args.valid_key_tokens = kv_len;
        args.query_start = kv_len - seq_len;
        args.mask = ops::AttentionMask::Causal;
        args.scale = 1.0 / (head_dim as f32).sqrt();
        ops::kv_cache_attention(&self.ctx, args)?;
        Ok(out.reshape(vec![seq_len, n_heads * head_dim])?)
    }
}

impl apxinf_core::SamplingBackend for CudaNewBackend {
    fn create_token_sampler(&self, spec: TokenSamplingSpec) -> Result<Box<dyn TokenSampler>> {
        crate::sampling::create_token_sampler(&self.ctx, spec)
    }

    fn create_normal_generator(&self, output: Tensor) -> Result<Box<dyn NormalGenerator>> {
        crate::sampling::create_normal_generator(&self.ctx, output)
    }
}

#[cfg(test)]
mod tests {
    use std::panic::{catch_unwind, AssertUnwindSafe};
    use std::sync::Arc;

    use apxinf_core::{Backend, Shape, Tensor};
    use half::bf16;

    use super::*;

    #[test]
    fn concat_2d_concatenates_columns() {
        let _guard = crate::CUDA_TEST_LOCK.lock().unwrap();
        let backend = CudaNewBackend::new(Arc::new(CudaContext::new(0).unwrap()));
        let left = Tensor::from_bf16(
            Shape::new(vec![2, 2]),
            &[
                bf16::from_f32(1.0),
                bf16::from_f32(2.0),
                bf16::from_f32(3.0),
                bf16::from_f32(4.0),
            ],
        )
        .unwrap();
        let right = Tensor::from_bf16(
            Shape::new(vec![2, 1]),
            &[bf16::from_f32(5.0), bf16::from_f32(6.0)],
        )
        .unwrap();
        let left = backend.to_device(&left).unwrap();
        let right = backend.to_device(&right).unwrap();

        let output = backend.concat_2d(&[&left, &right]).unwrap();
        backend.synchronize().unwrap();
        let output = backend.to_cpu(&output).unwrap();

        assert_eq!(output.shape().dims(), [2, 3]);
        let actual: Vec<f32> = output
            .as_bf16()
            .unwrap()
            .iter()
            .map(|value| value.to_f32())
            .collect();
        assert_eq!(actual, [1.0, 2.0, 5.0, 3.0, 4.0, 6.0]);
    }

    #[test]
    fn capture_graph_aborts_after_panic() {
        let _guard = crate::CUDA_TEST_LOCK.lock().unwrap();
        let backend = CudaNewBackend::new(Arc::new(CudaContext::new(0).unwrap()));
        let panicked = catch_unwind(AssertUnwindSafe(|| {
            let _ = backend.capture_graph::<()>(|| panic!("capture failure"));
        }));
        assert!(panicked.is_err());

        let buffer = crate::CudaBuffer::alloc(16, backend.device_id()).unwrap();
        let (graph, ()) = backend
            .capture_graph(|| {
                buffer
                    .zero_async(backend.context().stream())
                    .map_err(Error::Cuda)
            })
            .unwrap();
        graph.replay().unwrap();
        backend.synchronize().unwrap();
    }
}
