//! CUDA KV cache used by the portable backend adapter.
//!
//! The cache layout is isolated here because cuda-new's attention operator
//! consumes sequence-major `[1, capacity, heads, dim]` tensors. The portable
//! backend and model code only see the `KvCache` contract.

use std::any::Any;
use std::sync::Arc;

use apxinf_core::{DType, Error, KvCache, Result, Shape, Tensor};

use crate::{ops, CudaContext};

pub struct CudaKvCache {
    ctx: Arc<CudaContext>,
    keys: Vec<Tensor>,
    values: Vec<Tensor>,
    n_kv_heads: usize,
    head_dim: usize,
    max_seq_len: usize,
    seq_len: usize,
}

impl CudaKvCache {
    pub(crate) fn new(
        ctx: Arc<CudaContext>,
        n_layers: usize,
        n_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
    ) -> Result<Self> {
        let shape = Shape::new(vec![1, max_seq_len, n_kv_heads, head_dim]);
        let mut keys = Vec::with_capacity(n_layers);
        let mut values = Vec::with_capacity(n_layers);
        for _ in 0..n_layers {
            keys.push(ctx.allocate_output(shape.clone(), DType::BF16)?);
            values.push(ctx.allocate_output(shape.clone(), DType::BF16)?);
        }
        Ok(Self {
            ctx,
            keys,
            values,
            n_kv_heads,
            head_dim,
            max_seq_len,
            seq_len: 0,
        })
    }

    pub fn key(&self, layer_idx: usize) -> &Tensor {
        &self.keys[layer_idx]
    }

    pub fn value(&self, layer_idx: usize) -> &Tensor {
        &self.values[layer_idx]
    }
}

impl KvCache for CudaKvCache {
    fn append(
        &mut self,
        layer_idx: usize,
        k: &Tensor,
        v: &Tensor,
        append_len: usize,
    ) -> Result<()> {
        ops::cache_append(
            &self.ctx,
            &self.keys[layer_idx],
            k,
            self.n_kv_heads,
            self.head_dim,
            self.max_seq_len,
            self.seq_len,
            append_len,
        )?;
        ops::cache_append(
            &self.ctx,
            &self.values[layer_idx],
            v,
            self.n_kv_heads,
            self.head_dim,
            self.max_seq_len,
            self.seq_len,
            append_len,
        )
    }

    fn advance(&mut self, n: usize) {
        self.seq_len += n;
    }

    fn seq_len(&self) -> usize {
        self.seq_len
    }

    fn clear(&mut self) -> Result<()> {
        for tensor in self.keys.iter().chain(self.values.iter()) {
            let buffer = crate::CudaBuffer::from_tensor(tensor).map_err(Error::Cuda)?;
            buffer.zero().map_err(Error::Cuda)?;
        }
        self.seq_len = 0;
        Ok(())
    }

    fn n_layers(&self) -> usize {
        self.keys.len()
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}
