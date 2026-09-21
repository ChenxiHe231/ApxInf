//! Thin implementation of the portable `apxinf-core` backend contract.
//!
//! This module owns no tuning or candidate registry.  Where cuda-new has an
//! L3 semantic, the adapter allocates/binds tensors and enters that semantic;
//! the native L3 runtime remains the sole operator-selection authority.

use apxinf_core::{
    Backend, DType, Device, Error, Graph, KvCache, NormalGenerator, Result, SamplingBackend, Shape,
    Tensor, TokenSampler, TokenSamplingSpec,
};

use crate::ops::{
    self, AttentionArgs, AttentionMask, GemmArgs, KvCacheAttentionArgs, LayerNormArgs,
    PointwiseActivation, PointwiseArgs, PointwiseSemantic, RmsNormArgs,
};
use crate::{kernels, transfers, CudaBuffer, CudaContext, CudaKVCache};

struct CudaGraph {
    graph: crate::CapturedGraph,
}

impl Graph for CudaGraph {
    fn replay(&self) -> Result<()> {
        self.graph.replay()
    }
}

pub struct CudaBackend {
    ctx: CudaContext,
}

impl CudaBackend {
    pub fn new(device_id: usize) -> Result<Self> {
        Ok(Self {
            ctx: CudaContext::new(device_id).map_err(Error::Cuda)?,
        })
    }

    pub fn context(&self) -> &CudaContext {
        &self.ctx
    }

    pub fn device_id(&self) -> usize {
        self.ctx.device_id()
    }

    /// Compatibility spelling for callers which previously requested relaxed
    /// capture.  cuda-new uses one thread-local capture policy so unrelated
    /// work cannot invalidate the graph.
    pub fn begin_capture_relaxed(&self) -> Result<()> {
        self.begin_capture()
    }

    fn output(&self, shape: Shape, dtype: DType) -> Result<Tensor> {
        self.ctx.allocate_output(shape, dtype)
    }

    fn matrix_view(tensor: &Tensor, operation: &str) -> Result<(Tensor, bool)> {
        match tensor.shape().dims() {
            [_cols] => Ok((tensor.reshape(Shape::new(vec![1, tensor.numel()]))?, true)),
            [_rows, _cols] => Ok((tensor.clone(), false)),
            dimensions => Err(Error::Other(format!(
                "{operation} requires a vector or matrix, got rank {}",
                dimensions.len()
            ))),
        }
    }

    fn rms_norm_l3(&self, input: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
        let (matrix, was_vector) = Self::matrix_view(input, "normalization")?;
        let mut output = self.output(matrix.shape().clone(), input.dtype())?;
        ops::rms_norm(
            &self.ctx,
            RmsNormArgs::new(&matrix, weight, &mut output, eps),
        )?;
        if was_vector {
            output.reshape(input.shape().clone())
        } else {
            Ok(output)
        }
    }

    fn layer_norm_l3(
        &self,
        input: &Tensor,
        weight: &Tensor,
        bias: &Tensor,
        eps: f32,
    ) -> Result<Tensor> {
        let (matrix, was_vector) = Self::matrix_view(input, "normalization")?;
        let mut output = self.output(matrix.shape().clone(), input.dtype())?;
        ops::layer_norm(
            &self.ctx,
            LayerNormArgs::new(&matrix, weight, bias, &mut output, eps),
        )?;
        if was_vector {
            output.reshape(input.shape().clone())
        } else {
            Ok(output)
        }
    }

    fn pointwise_activation(
        &self,
        input: &Tensor,
        bias: Option<&Tensor>,
        activation: PointwiseActivation,
    ) -> Result<Tensor> {
        let (matrix, was_vector) = Self::matrix_view(input, "pointwise activation")?;
        let mut output = self.output(matrix.shape().clone(), input.dtype())?;
        let mut args = PointwiseArgs::new(PointwiseSemantic::BiasActivation, &matrix, &mut output);
        args.bias = bias;
        args.activation = activation;
        ops::pointwise(&self.ctx, args)?;
        if was_vector {
            output.reshape(input.shape().clone())
        } else {
            Ok(output)
        }
    }

    fn sdpa_with_cache(
        &self,
        q: &Tensor,
        kv: &mut dyn KvCache,
        layer_idx: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        kv_len: usize,
        max_seq_len: usize,
    ) -> Result<Tensor> {
        let cache = kv
            .as_any()
            .downcast_ref::<CudaKVCache>()
            .ok_or_else(|| Error::Other("expected cuda-new CudaKVCache".into()))?;
        if cache.device_id() != self.device_id()
            || cache.n_kv_heads() != n_kv_heads
            || cache.head_dim() != head_dim
            || cache.max_seq_len() != max_seq_len
        {
            return Err(Error::Other(
                "KV cache device or geometry does not match Attention".into(),
            ));
        }
        let dimensions = q.shape().dims();
        if dimensions.len() != 3
            || dimensions[1] != n_heads
            || dimensions[2] != head_dim
            || kv_len == 0
            || kv_len > max_seq_len
        {
            return Err(Error::Other("invalid portable KV Attention shape".into()));
        }
        let query_tokens = dimensions[0];
        let query_start = kv_len
            .checked_sub(query_tokens)
            .ok_or_else(|| Error::Other("KV Attention has fewer keys than query tokens".into()))?;
        let dtype = cache
            .dtype()
            .ok_or_else(|| Error::Other("KV Attention cache has not been initialized".into()))?;
        if q.dtype() != dtype {
            return Err(Error::DTypeMismatch {
                expected: dtype,
                got: q.dtype(),
            });
        }
        let query = q.reshape(Shape::new(vec![1, query_tokens, n_heads, head_dim]))?;
        let key = cache.key_tensor(layer_idx, dtype)?;
        let value = cache.value_tensor(layer_idx, dtype)?;
        let mut output = self.output(query.shape().clone(), dtype)?;
        let mut args = KvCacheAttentionArgs::new(&query, &key, &value, &mut output);
        args.valid_key_tokens = kv_len;
        args.query_start = query_start;
        args.mask = AttentionMask::Causal;
        ops::kv_cache_attention(&self.ctx, args)?;
        output.reshape(Shape::new(vec![query_tokens, n_heads * head_dim]))
    }
}

impl SamplingBackend for CudaBackend {
    fn create_token_sampler(&self, spec: TokenSamplingSpec) -> Result<Box<dyn TokenSampler>> {
        crate::sampling::create_token_sampler(&self.ctx, spec)
    }

    fn create_normal_generator(&self, output: Tensor) -> Result<Box<dyn NormalGenerator>> {
        crate::sampling::create_normal_generator(&self.ctx, output)
    }
}

impl Backend for CudaBackend {
    fn rms_norm(&self, input: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
        match input.dtype() {
            DType::F16 | DType::BF16 => self.rms_norm_l3(input, weight, eps),
            // Standalone F32 normalization is not yet an L3 semantic.  Keep
            // the already-safe migrated primitive available to the portable
            // backend until that candidate exists.
            DType::F32 => kernels::norm::rms(&self.ctx, input, weight, eps),
            dtype => Err(Error::Other(format!(
                "portable CUDA RMSNorm does not support {dtype}"
            ))),
        }
    }

    fn silu(&self, input: &Tensor) -> Result<Tensor> {
        match input.dtype() {
            DType::F16 | DType::BF16 => {
                self.pointwise_activation(input, None, PointwiseActivation::Silu)
            }
            DType::F32 => kernels::activation::silu(&self.ctx, input),
            dtype => Err(Error::Other(format!(
                "portable CUDA SiLU does not support {dtype}"
            ))),
        }
    }

    fn add(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        kernels::elementwise::add(&self.ctx, a, b)
    }

    fn mul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        kernels::elementwise::mul(&self.ctx, a, b)
    }

    fn scale(&self, input: &Tensor, factor: f32) -> Result<Tensor> {
        kernels::elementwise::scale(&self.ctx, input, factor)
    }

    fn matmul(&self, a: &Tensor, b: &Tensor) -> Result<Tensor> {
        let a_dims = a.shape().dims();
        let b_dims = b.shape().dims();
        if a_dims.len() != 2 || b_dims.len() != 2 || a_dims[1] != b_dims[0] {
            return Err(Error::Other(format!(
                "portable CUDA matmul requires [M,K] @ [K,N], got {:?} and {:?}",
                a_dims, b_dims
            )));
        }
        let mut output = self.output(Shape::new(vec![a_dims[0], b_dims[1]]), a.dtype())?;
        ops::gemm(&self.ctx, GemmArgs::new(a, b, &mut output))?;
        Ok(output)
    }

    fn rope(
        &self,
        input: &Tensor,
        n_heads: usize,
        head_dim: usize,
        theta: f32,
        pos_offset: u32,
    ) -> Result<Tensor> {
        kernels::rope::apply_batched(&self.ctx, input, n_heads, head_dim, theta, pos_offset)
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
        let seq_len = input.shape().dims().first().copied().unwrap_or(0);
        if pos_ids.len() != seq_len.saturating_mul(3) {
            return Err(Error::Other("mRoPE position count mismatch".into()));
        }
        let bytes: Vec<u8> = pos_ids
            .iter()
            .flat_map(|value| value.to_ne_bytes())
            .collect();
        let positions = CudaBuffer::alloc(bytes.len(), self.device_id()).map_err(Error::Cuda)?;
        positions.copy_from_host(&bytes).map_err(Error::Cuda)?;
        kernels::rope::apply_mrope(
            &self.ctx, input, n_heads, head_dim, theta, sections, &positions,
        )
    }

    fn layer_norm(
        &self,
        input: &Tensor,
        weight: &Tensor,
        bias: &Tensor,
        eps: f32,
    ) -> Result<Tensor> {
        match input.dtype() {
            DType::F16 | DType::BF16 => self.layer_norm_l3(input, weight, bias, eps),
            dtype => Err(Error::Other(format!(
                "portable CUDA LayerNorm does not support {dtype}"
            ))),
        }
    }

    fn gelu_tanh(&self, input: &Tensor) -> Result<Tensor> {
        match input.dtype() {
            DType::F16 | DType::BF16 => {
                self.pointwise_activation(input, None, PointwiseActivation::Gelu)
            }
            dtype => Err(Error::Other(format!(
                "portable CUDA GELU does not support {dtype}"
            ))),
        }
    }

    fn add_bias(&self, input: &Tensor, bias: &Tensor) -> Result<Tensor> {
        match input.dtype() {
            DType::F16 | DType::BF16 => {
                self.pointwise_activation(input, Some(bias), PointwiseActivation::None)
            }
            dtype => Err(Error::Other(format!(
                "portable CUDA bias add does not support {dtype}"
            ))),
        }
    }

    fn rope_vision_2d(
        &self,
        input: &Tensor,
        n_heads: usize,
        head_dim: usize,
        theta: f32,
        pos_ids: &[u32],
    ) -> Result<Tensor> {
        let seq_len = input.shape().dims().first().copied().unwrap_or(0);
        if pos_ids.len() != seq_len.saturating_mul(2) {
            return Err(Error::Other("vision RoPE position count mismatch".into()));
        }
        let bytes: Vec<u8> = pos_ids
            .iter()
            .flat_map(|value| value.to_ne_bytes())
            .collect();
        let positions = CudaBuffer::alloc(bytes.len(), self.device_id()).map_err(Error::Cuda)?;
        positions.copy_from_host(&bytes).map_err(Error::Cuda)?;
        kernels::rope::apply_vision_2d(&self.ctx, input, n_heads, head_dim, theta, &positions)
    }

    fn concat_2d(&self, tensors: &[&Tensor]) -> Result<Tensor> {
        let first = tensors
            .first()
            .ok_or_else(|| Error::Other("concat_2d requires inputs".into()))?;
        let first_dims = first.shape().dims();
        if first_dims.len() != 2 {
            return Err(Error::Other("concat_2d requires matrices".into()));
        }
        let rows = first_dims[0];
        let dtype = first.dtype();
        let mut total_cols = 0usize;
        for tensor in tensors {
            let dims = tensor.shape().dims();
            if dims.len() != 2
                || dims[0] != rows
                || tensor.dtype() != dtype
                || tensor.device() != self.device()
            {
                return Err(Error::Other(
                    "concat_2d inputs must have matching rows, dtype, and device".into(),
                ));
            }
            total_cols = total_cols
                .checked_add(dims[1])
                .ok_or_else(|| Error::Other("concat_2d width overflow".into()))?;
        }
        let output = self.output(Shape::new(vec![rows, total_cols]), dtype)?;
        let output_buffer = CudaBuffer::from_tensor(&output).map_err(Error::Cuda)?;
        let destination_pitch = total_cols * dtype.size_in_bytes();
        let mut column = 0usize;
        for tensor in tensors {
            let width = tensor.shape().dims()[1] * dtype.size_in_bytes();
            transfers::copy_tensor_2d_to_buffer(
                &self.ctx,
                tensor,
                &output_buffer,
                column * dtype.size_in_bytes(),
                destination_pitch,
                width,
                width,
                rows,
            )?;
            column += tensor.shape().dims()[1];
        }
        Ok(output)
    }

    fn vision_sdpa(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        seq_len: usize,
        n_heads: usize,
        head_dim: usize,
    ) -> Result<Tensor> {
        let expected = [seq_len, n_heads, head_dim];
        if q.shape().dims() != expected
            || k.shape().dims() != expected
            || v.shape().dims() != expected
        {
            return Err(Error::Other("vision Attention shape mismatch".into()));
        }
        let shape = Shape::new(vec![1, seq_len, n_heads, head_dim]);
        let query = q.reshape(shape.clone())?;
        let key = k.reshape(shape.clone())?;
        let value = v.reshape(shape.clone())?;
        let mut output = self.output(shape, q.dtype())?;
        let mut args = AttentionArgs::new(&query, &key, &value, &mut output);
        args.mask = AttentionMask::None;
        ops::attention(&self.ctx, args)?;
        output.reshape(Shape::new(vec![seq_len, n_heads * head_dim]))
    }

    fn embedding(&self, table: &Tensor, ids: &[u32]) -> Result<Tensor> {
        if ids.is_empty() {
            return Err(Error::Other("embedding requires at least one token".into()));
        }
        let bytes: Vec<u8> = ids.iter().flat_map(|value| value.to_ne_bytes()).collect();
        let ids = CudaBuffer::alloc(bytes.len(), self.device_id()).map_err(Error::Cuda)?;
        ids.copy_from_host(&bytes).map_err(Error::Cuda)?;
        kernels::embedding::lookup(&self.ctx, table, &ids, bytes.len() / 4)
    }

    fn sdpa_decode(
        &self,
        q: &Tensor,
        kv: &mut dyn KvCache,
        layer_idx: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        kv_len: usize,
        max_seq_len: usize,
    ) -> Result<Tensor> {
        self.sdpa_with_cache(
            q,
            kv,
            layer_idx,
            n_heads,
            n_kv_heads,
            head_dim,
            kv_len,
            max_seq_len,
        )
    }

    fn sdpa_prefill(
        &self,
        q: &Tensor,
        kv: &mut dyn KvCache,
        layer_idx: usize,
        n_heads: usize,
        n_kv_heads: usize,
        head_dim: usize,
        kv_len: usize,
        max_seq_len: usize,
    ) -> Result<Tensor> {
        self.sdpa_with_cache(
            q,
            kv,
            layer_idx,
            n_heads,
            n_kv_heads,
            head_dim,
            kv_len,
            max_seq_len,
        )
    }

    fn create_kv_cache(
        &self,
        n_layers: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
    ) -> Box<dyn KvCache> {
        Box::new(
            CudaKVCache::new(
                self.device_id(),
                n_layers,
                n_kv_heads,
                head_dim,
                max_seq_len,
            )
            .expect("portable Backend::create_kv_cache cannot report CUDA allocation failure"),
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
        let cache = kv
            .as_any_mut()
            .downcast_mut::<CudaKVCache>()
            .ok_or_else(|| Error::Other("expected cuda-new CudaKVCache".into()))?;
        CudaKVCache::append(cache, &self.ctx, layer_idx, k, v, append_len)
    }

    fn synchronize(&self) -> Result<()> {
        self.ctx.synchronize().map_err(Error::Cuda)
    }

    fn begin_capture(&self) -> Result<()> {
        crate::graph::begin(&self.ctx).map_err(Error::Cuda)
    }

    fn end_capture(&self) -> Result<Box<dyn Graph>> {
        Ok(Box::new(CudaGraph {
            graph: crate::graph::end(&self.ctx).map_err(Error::Cuda)?,
        }))
    }

    fn device(&self) -> Device {
        Device::Cuda(self.device_id())
    }

    fn to_device(&self, tensor: &Tensor) -> Result<Tensor> {
        transfers::to_cuda(tensor, self.device_id())
    }

    fn to_cpu(&self, tensor: &Tensor) -> Result<Tensor> {
        transfers::to_cpu(tensor)
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}
