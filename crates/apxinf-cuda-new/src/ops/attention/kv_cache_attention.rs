use apxinf_core::{Result, Tensor};

use super::{attention, AttentionArgs, AttentionMask, AttentionPolicy};
use crate::CudaContext;

/// Contiguous KV-cache attention. The query tokens are interpreted as the
/// trailing positions of `key_cache`/`value_cache`.
pub struct KvCacheAttentionArgs<'a> {
    pub query: &'a Tensor,
    pub key_cache: &'a Tensor,
    pub value_cache: &'a Tensor,
    pub out: &'a mut Tensor,
    pub scale: f32,
    pub policy: AttentionPolicy,
}

impl<'a> KvCacheAttentionArgs<'a> {
    pub fn new(
        query: &'a Tensor,
        key_cache: &'a Tensor,
        value_cache: &'a Tensor,
        out: &'a mut Tensor,
    ) -> Self {
        let head_dim = query.shape().dims().get(3).copied().unwrap_or(1);
        Self {
            query,
            key_cache,
            value_cache,
            out,
            scale: 1.0 / (head_dim as f32).sqrt(),
            policy: AttentionPolicy::default(),
        }
    }
}

pub fn kv_cache_attention(ctx: &CudaContext, args: KvCacheAttentionArgs<'_>) -> Result<()> {
    attention(
        ctx,
        AttentionArgs {
            query: args.query,
            key: args.key_cache,
            value: args.value_cache,
            out: args.out,
            mask: AttentionMask::Causal,
            scale: args.scale,
            policy: args.policy,
        },
    )
}
