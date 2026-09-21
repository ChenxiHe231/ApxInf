//! Source-level guardrails for the direct/prepared lifecycle boundary.

const RUST_DIRECT_LAUNCH: &[(&str, &str)] = &[
    ("gather", include_str!("../gather/launch.rs")),
    ("norm", include_str!("../norm/launch.rs")),
    ("pointwise", include_str!("../pointwise/launch.rs")),
    ("quantization", include_str!("../quantization/launch.rs")),
    ("rope", include_str!("../rope/launch.rs")),
];

const NATIVE_DIRECT_LAUNCH: &[(&str, &str)] = &[
    ("gather", include_str!("../../../native/adapters/gather.cu")),
    ("norm", include_str!("../../../native/adapters/norm.cu")),
    ("pointwise", include_str!("../../../native/adapters/pointwise.cu")),
    (
        "quantization",
        include_str!("../../../native/adapters/quantization.cu"),
    ),
    ("rope", include_str!("../../../native/adapters/rope.cu")),
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
    for (family, source) in RUST_DIRECT_LAUNCH {
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
        assert!(!header.contains("policy"), "{family} direct ABI exposes policy");
    }
    for (family, source) in NATIVE_DIRECT_LAUNCH {
        for forbidden in ["Implementation", "Execution", "registry(", "candidate"] {
            assert!(
                !source.contains(forbidden),
                "{family} direct adapter contains forbidden {forbidden}"
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

#[test]
fn compatibility_layers_are_absent() {
    let build = include_str!("../../../build.rs");
    let raw_ffi = include_str!("../../ffi/raw/mod.rs");
    let manifest = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));

    for obsolete in ["ported/", "ported.rs", "mod ported"] {
        assert!(!build.contains(obsolete), "build still names {obsolete}");
        assert!(!raw_ffi.contains(obsolete), "FFI still names {obsolete}");
    }
    for removed in [
        "src/kernels",
        "src/backend.rs",
        "src/kv_cache.rs",
        "src/ffi/raw/operators.rs",
        "native/adapters/tensor_ops.cu",
        "native/adapters/bf16_ops.cu",
        "native/adapters/mixed_precision_ops.cu",
        "native/adapters/int8_ops.cu",
    ] {
        assert!(!manifest.join(removed).exists(), "compatibility path remains: {removed}");
    }
    assert!(raw_ffi.contains("mod sampling"));
    assert!(!raw_ffi.contains("mod operators"));
}
