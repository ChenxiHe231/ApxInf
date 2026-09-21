//! Source-level guardrails for the direct/prepared lifecycle boundary.

const RUST_DIRECT_EXECUTION: &[(&str, &str)] = &[
    ("gather", include_str!("../gather/execution.rs")),
    ("norm", include_str!("../norm/execution.rs")),
    ("pointwise", include_str!("../pointwise/execution.rs")),
    ("quantization", include_str!("../quantization/execution.rs")),
    ("rope", include_str!("../rope/execution.rs")),
];

const NATIVE_DIRECT_HEADERS: &[(&str, &str)] = &[
    ("gather", include_str!("../../../native/include/apxinf_cuda/gather.h")),
    ("norm", include_str!("../../../native/include/apxinf_cuda/norm.h")),
    ("pointwise", include_str!("../../../native/include/apxinf_cuda/pointwise.h")),
    (
        "quantization",
        include_str!("../../../native/include/apxinf_cuda/quantization.h"),
    ),
    ("rope", include_str!("../../../native/include/apxinf_cuda/rope.h")),
];

#[test]
fn stateless_families_have_no_execution_cache_or_synchronization() {
    for (family, source) in RUST_DIRECT_EXECUTION {
        for forbidden in [
            "ExecutionKey",
            "lookup_execution",
            "store_execution",
            "use_execution",
            ".synchronize()",
        ] {
            assert!(
                !source.contains(forbidden),
                "{family} direct execution contains forbidden {forbidden}"
            );
        }
        assert!(source.contains("validate_capture_target"));
        assert!(source.contains(&format!("apxinf_{family}_launch")));
    }
}

#[test]
fn stateless_native_abi_exposes_only_direct_launch() {
    for (family, header) in NATIVE_DIRECT_HEADERS {
        assert!(header.contains(&format!("apxinf_{family}_launch")));
        for forbidden in ["_prepare", "_enqueue", "_destroy", "_summary"] {
            assert!(
                !header.contains(&format!("apxinf_{family}{forbidden}")),
                "{family} header exposes prepared lifecycle symbol {forbidden}"
            );
        }
    }
}

#[test]
fn norm_public_surface_is_typed_by_semantic() {
    let contracts = include_str!("../norm/contracts.rs");
    let public = include_str!("../norm/norm.rs");
    assert!(!contracts.contains("pub struct NormArgs"));
    assert!(!contracts.contains("pub enum NormSemantic"));
    for args in [
        "RmsNormArgs",
        "LayerNormArgs",
        "AdaptiveRmsNormArgs",
        "BiasResidualArgs",
        "BiasResidualRmsNormArgs",
        "BiasResidualLayerNormArgs",
        "AdaGateResidualArgs",
        "AdaGateResidualRmsNormArgs",
        "BiasThenResidualArgs",
    ] {
        assert!(public.contains(&format!("pub struct {args}")), "missing {args}");
    }
}
