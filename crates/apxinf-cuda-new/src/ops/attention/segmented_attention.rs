use apxinf_core::{Error, Result, Shape, Tensor};

use super::{attention, AttentionArgs, AttentionMask, AttentionPolicy};
use crate::{CudaBuffer, CudaContext};

/// Packed variable-length self-attention. `offsets` partitions the leading
/// token dimension of matching `[tokens, heads, head_dim]` Q/K/V tensors.
pub struct SegmentedAttentionArgs<'a> {
    pub query: &'a Tensor,
    pub key: &'a Tensor,
    pub value: &'a Tensor,
    pub out: &'a mut Tensor,
    pub offsets: &'a [u32],
    pub scale: f32,
    pub policy: AttentionPolicy,
}

impl<'a> SegmentedAttentionArgs<'a> {
    pub fn new(
        query: &'a Tensor,
        key: &'a Tensor,
        value: &'a Tensor,
        out: &'a mut Tensor,
        offsets: &'a [u32],
    ) -> Self {
        let head_dim = query.shape().dims().get(2).copied().unwrap_or(1);
        Self {
            query,
            key,
            value,
            out,
            offsets,
            scale: 1.0 / (head_dim as f32).sqrt(),
            policy: AttentionPolicy::default(),
        }
    }
}

fn invalid(message: impl Into<String>) -> Error {
    Error::Other(message.into())
}

pub fn segmented_attention(ctx: &CudaContext, args: SegmentedAttentionArgs<'_>) -> Result<()> {
    let shape = args.query.shape().dims();
    if shape.len() != 3
        || shape.contains(&0)
        || args.key.shape().dims() != shape
        || args.value.shape().dims() != shape
        || args.out.shape().dims() != shape
        || args.key.dtype() != args.query.dtype()
        || args.value.dtype() != args.query.dtype()
        || args.out.dtype() != args.query.dtype()
    {
        return Err(invalid(
            "segmented Attention requires matching non-empty [tokens, heads, head_dim] tensors",
        ));
    }
    if args.offsets.len() < 2
        || args.offsets[0] != 0
        || args.offsets.last().copied() != Some(shape[0] as u32)
        || args.offsets.windows(2).any(|pair| pair[0] > pair[1])
    {
        return Err(invalid(
            "segmented Attention offsets must be monotonic and span all tokens",
        ));
    }
    if !args.scale.is_finite() || args.scale <= 0.0 {
        return Err(invalid(
            "segmented Attention scale must be finite and positive",
        ));
    }

    let row_bytes = shape[1]
        .checked_mul(shape[2])
        .and_then(|elements| elements.checked_mul(args.query.dtype().size_in_bytes()))
        .ok_or_else(|| invalid("segmented Attention row size overflow"))?;
    let query = CudaBuffer::from_tensor(args.query).map_err(Error::Cuda)?;
    let key = CudaBuffer::from_tensor(args.key).map_err(Error::Cuda)?;
    let value = CudaBuffer::from_tensor(args.value).map_err(Error::Cuda)?;
    let output = CudaBuffer::from_tensor(args.out).map_err(Error::Cuda)?;

    for bounds in args.offsets.windows(2) {
        let start = bounds[0] as usize;
        let tokens = (bounds[1] - bounds[0]) as usize;
        if tokens == 0 {
            continue;
        }
        let offset = start
            .checked_mul(row_bytes)
            .ok_or_else(|| invalid("segmented Attention byte offset overflow"))?;
        let bytes = tokens
            .checked_mul(row_bytes)
            .ok_or_else(|| invalid("segmented Attention byte size overflow"))?;
        let segment_shape = Shape::new(vec![1, tokens, shape[1], shape[2]]);
        let q = query
            .view(offset, bytes)
            .map_err(Error::Cuda)?
            .as_tensor(segment_shape.clone(), args.query.dtype())
            .map_err(Error::Cuda)?;
        let k = key
            .view(offset, bytes)
            .map_err(Error::Cuda)?
            .as_tensor(segment_shape.clone(), args.query.dtype())
            .map_err(Error::Cuda)?;
        let v = value
            .view(offset, bytes)
            .map_err(Error::Cuda)?
            .as_tensor(segment_shape.clone(), args.query.dtype())
            .map_err(Error::Cuda)?;
        let mut out = output
            .view(offset, bytes)
            .map_err(Error::Cuda)?
            .as_tensor(segment_shape, args.query.dtype())
            .map_err(Error::Cuda)?;
        attention(
            ctx,
            AttentionArgs {
                query: &q,
                key: &k,
                value: &v,
                out: &mut out,
                mask: AttentionMask::None,
                scale: args.scale,
                policy: args.policy.clone(),
            },
        )?;
    }
    Ok(())
}
