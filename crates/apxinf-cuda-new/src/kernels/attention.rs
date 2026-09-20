use apxinf_core::Tensor;

/// Compatibility result for packed-QKV split operators.
pub struct QkvTensors {
    pub q: Tensor,
    pub k: Tensor,
    pub v: Tensor,
}
