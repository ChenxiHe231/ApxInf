//! CUDA-facing seam for the GR00T runtime.
//!
//! GR00T uses cuda-new's public `ops` surface and runtime backend.

#[cfg(feature = "cuda")]
pub(crate) use crate::accelerator::downcast_cuda_new_arc as downcast_arc;
#[cfg(feature = "cuda")]
pub(crate) use apxinf_cuda_new::CudaNewBackend as CudaBackend;
#[cfg(feature = "cuda")]
pub(crate) use apxinf_cuda_new::{
    ops, transfers, CudaBuffer as DeviceBuffer, CudaNewBackend as RuntimeBackend,
};

/// The model opts into packed8 only for its measured BF16 epilogues.
const fn use_packed8_bias_activation(
    sm: u32,
    rows: usize,
    cols: usize,
    activation: i32,
    legacy: bool,
) -> bool {
    sm == 110
        && !legacy
        && matches!(
            (rows, cols, activation),
            (41, 1536, 0) | (41, 4608, 0) | (41, 6144, 1) | (256, 4096, 1) | (512, 4096, 1)
        )
}

#[cfg(feature = "cuda")]
pub(super) fn try_packed8_bias_activation(
    backend: &RuntimeBackend,
    input: &apxinf_core::Tensor,
    bias: &apxinf_core::Tensor,
    activation: i32,
) -> apxinf_core::Result<Option<apxinf_core::Tensor>> {
    let [rows, cols] = input.shape().dims() else {
        return Err(apxinf_core::Error::Other(
            "GR00T packed8 bias activation expects a matrix".into(),
        ));
    };
    if !use_packed8_bias_activation(
        backend.context().caps().sm,
        *rows,
        *cols,
        activation,
        std::env::var_os("APXINF_GR00T_BF16_LEGACY_PACKED8_BIAS_ACTIVATION").is_some(),
    ) {
        return Ok(None);
    }
    ops::activation::try_bias_activation_bf16_packed8(backend.context(), input, bias, activation)
}

#[cfg(test)]
mod tests {
    use super::use_packed8_bias_activation;

    #[test]
    fn packed8_policy_keeps_measured_shapes_and_legacy_opt_out() {
        for (rows, cols, activation) in [
            (41, 1536, 0),
            (41, 4608, 0),
            (41, 6144, 1),
            (256, 4096, 1),
            (512, 4096, 1),
        ] {
            assert!(use_packed8_bias_activation(
                110, rows, cols, activation, false
            ));
            assert!(!use_packed8_bias_activation(
                87, rows, cols, activation, false
            ));
            assert!(!use_packed8_bias_activation(
                110, rows, cols, activation, true
            ));
        }
        for (rows, cols, activation) in [(40, 1536, 0), (41, 1536, 1), (128, 4096, 1)] {
            assert!(!use_packed8_bias_activation(
                110, rows, cols, activation, false
            ));
        }
    }
}
