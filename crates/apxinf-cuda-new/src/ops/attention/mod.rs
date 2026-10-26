mod attention;
pub(crate) mod contracts;
pub(crate) mod execution;
mod fixed;
mod gates;
mod kv_cache_attention;
mod planning;
mod segmented_attention;

pub use attention::{attention, packed_qkv_attention};
pub use contracts::{AttentionArgs, AttentionMask, AttentionPolicy, PackedQkvAttentionArgs};
pub use fixed::*;
pub use gates::{apply_output_gate, split_query_and_gate};
pub use kv_cache_attention::{
    decode_attention, decode_attention_workspace_bytes, kv_cache_attention, KvCacheAttentionArgs,
    KvCacheDecodeMeta,
};
pub use planning::{
    expert_qkv_prepare, expert_sigmoid_gate_mul, full_attn_prepare, gqa, sigmoid_gate_mul,
};
pub use segmented_attention::{segmented_attention, SegmentedAttentionArgs};

#[cfg(test)]
pub(crate) use kv_cache_attention::normalize as normalize_kv_cache_attention;
#[cfg(test)]
pub(crate) use segmented_attention::normalize as normalize_segmented_attention;
