use std::env;

#[path = "aot/link.rs"]
mod aot;
#[path = "build_support/cuda_arch.rs"]
mod cuda_arch;
#[path = "build_support/nvcc_build.rs"]
mod nvcc_build;

use cuda_arch::{gencode_args, is_cutlass_sm100_family, select_cuda_arch, ArchSource};

const FNV1A_128_OFFSET: u128 = 0x6c62272e07bb014262b821756295c58d;
const FNV1A_128_PRIME: u128 = 0x0000000001000000000000000000013b;

fn hash_bytes(hash: &mut u128, bytes: &[u8]) {
    for byte in bytes {
        *hash ^= u128::from(*byte);
        *hash = hash.wrapping_mul(FNV1A_128_PRIME);
    }
}

fn collect_kernel_inputs(root: &std::path::Path, files: &mut Vec<std::path::PathBuf>) {
    let Ok(entries) = std::fs::read_dir(root) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_kernel_inputs(&path, files);
            continue;
        }
        if path
            .file_name()
            .is_some_and(|name| name.to_string_lossy().starts_with("._"))
        {
            continue;
        }
        if path.extension().is_some_and(|extension| {
            matches!(
                extension.to_string_lossy().as_ref(),
                "cu" | "cuh" | "h" | "hh" | "hpp"
            )
        }) {
            files.push(path);
        }
    }
}

fn computed_kernel_build_id(
    cuda_sources_root: &std::path::Path,
    target_arch: &str,
    nvcc_arch: Option<&str>,
    cutlass_arch: Option<&str>,
) -> String {
    let mut hash = FNV1A_128_OFFSET;
    for value in [
        "apxinf-cuda-kernel-build-v1",
        env!("CARGO_PKG_VERSION"),
        target_arch,
        nvcc_arch.unwrap_or("native"),
        cutlass_arch.unwrap_or("native"),
    ] {
        hash_bytes(&mut hash, value.as_bytes());
        hash_bytes(&mut hash, &[0]);
    }

    let mut files = Vec::new();
    collect_kernel_inputs(cuda_sources_root, &mut files);
    files.sort_unstable();
    for path in files {
        let relative = path.strip_prefix(cuda_sources_root).unwrap_or(&path);
        hash_bytes(&mut hash, relative.to_string_lossy().as_bytes());
        hash_bytes(&mut hash, &[0]);
        let bytes = std::fs::read(&path)
            .unwrap_or_else(|error| panic!("read kernel build input {}: {error}", path.display()));
        hash_bytes(&mut hash, &bytes);
        hash_bytes(&mut hash, &[0xff]);
    }

    format!("kb1-{hash:032x}")
}

fn emit_kernel_build_id(build_id: &str) {
    assert!(
        !build_id.is_empty() && !build_id.contains('\n') && !build_id.contains('\r'),
        "APXINF_KERNEL_BUILD_ID must be non-empty and single-line"
    );
    println!("cargo:rustc-env=APXINF_KERNEL_BUILD_ID={build_id}");
}

fn emit_rerun_if_changed_tree(root: &std::path::Path) {
    let Ok(entries) = std::fs::read_dir(root) else {
        return;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_dir() {
            emit_rerun_if_changed_tree(&path);
        } else {
            println!("cargo:rerun-if-changed={}", path.display());
        }
    }
}

/// The full argv of an nvcc invocation is compared by the shared object cache
/// in `build_support/nvcc_build.rs`, alongside the bounded worker pool that
/// compiles independent translation units concurrently.
use nvcc_build::{describe_command, object_is_current, run_parallel, CompileJob};

fn is_fa2_sm80_family(arch: &str) -> bool {
    matches!(arch, "sm_80" | "sm_86" | "sm_87" | "sm_89")
}

fn is_pre_sm80(arch: &str) -> bool {
    matches!(arch, "sm_70" | "sm_72" | "sm_75")
}

// Architectures that compile the vendored FlashAttention-2 BF16 forward
// kernels. The sm80 family and Blackwell (sm_120/121, GB10/DGX Spark) all run
// the same v2.7.4 instantiations; the -arch flag selects the real target.
fn is_fa2_bf16_arch(arch: &str) -> bool {
    is_fa2_sm80_family(arch) || matches!(arch, "sm_120" | "sm_120a" | "sm_121" | "sm_121a")
}

fn is_cutlass_sm89_family(arch: &str) -> bool {
    matches!(arch, "sm_89")
}

fn main() {
    println!("cargo:rustc-check-cfg=cfg(apxinf_aot_gemm_swiglu_bf16)");
    println!("cargo:rustc-check-cfg=cfg(apxinf_aot_gdn_prefill_bf16)");
    println!("cargo:rustc-check-cfg=cfg(apxinf_aot_vision_attention)");
    println!("cargo:rustc-check-cfg=cfg(apxinf_aot_action_attention)");
    println!("cargo:rustc-check-cfg=cfg(apxinf_aot_language_attention)");
    println!("cargo:rustc-check-cfg=cfg(apxinf_fa4_attention_cuda)");
    println!("cargo:rustc-check-cfg=cfg(apxinf_gdn_prefill_bf16_cuda)");
    println!("cargo:rustc-check-cfg=cfg(nvtx_header)");
    println!("cargo:rustc-check-cfg=cfg(nvtx_v2)");
    println!("cargo:rustc-check-cfg=cfg(nvtx_v3)");
    println!("cargo:rustc-check-cfg=cfg(apxinf_cutlass_fmha)");
    println!("cargo:rustc-check-cfg=cfg(apxinf_cutlass_gemm)");
    println!("cargo:rustc-check-cfg=cfg(apxinf_cutlass_bf16_sm89)");
    println!("cargo:rustc-check-cfg=cfg(apxinf_cutlass_int8_sm80)");
    println!("cargo:rustc-check-cfg=cfg(apxinf_fa2_sm80)");
    println!("cargo:rustc-check-cfg=cfg(apxinf_fa2_head_special)");
    println!("cargo:rustc-check-cfg=cfg(apxinf_fa2_f16_sm100)");
    println!("cargo:rustc-check-cfg=cfg(apxinf_fa2_direct_e4m3_sm100)");
    println!("cargo:rerun-if-env-changed=APXINF_CUDA_ARCH");
    println!("cargo:rerun-if-env-changed=APXINF_CUDA_ARCH_CUTLASS");
    println!("cargo:rerun-if-env-changed=APXINF_KERNEL_BUILD_ID");
    // A build input location, not an inference-time optimization switch.
    println!("cargo:rerun-if-env-changed=APXINF_CUDA_AOT_MANIFEST");
    println!("cargo:rerun-if-changed=aot/link.rs");
    println!("cargo:rerun-if-changed=aot/bundle.rs");
    println!("cargo:rerun-if-changed=aot/manifest.json");
    println!("cargo:rerun-if-env-changed=APXINF_FA2_HEAD64_MAXREG");
    println!("cargo:rerun-if-env-changed=CUDA_VISIBLE_DEVICES");
    println!("cargo:rerun-if-env-changed=NVIDIA_VISIBLE_DEVICES");
    println!("cargo:rerun-if-changed=build_support/cuda_arch.rs");
    // Only try to link CUDA if the toolkit is available.
    let cuda_path = env::var("CUDA_PATH")
        .or_else(|_| env::var("CUDA_HOME"))
        .unwrap_or_else(|_| "/usr/local/cuda".to_string());

    // Candidate lib directories. Desktop CUDA uses lib64/; embedded / cross
    // layouts split libs across several subdirs. Drive OS 7 puts libcublas
    // under `thor/targets/aarch64-linux/lib/` while libnvToolsExt lives in
    // the top-level lib64/. Add every candidate that exists so -lcudart,
    // -lcublas, and -lnvtx all resolve regardless of where each landed.
    let arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let candidate_lib_dirs = [
        format!("{cuda_path}/lib64"),
        format!("{cuda_path}/lib"),
        format!("{cuda_path}/targets/{arch}/lib"),
        format!("{cuda_path}/targets/aarch64-linux/lib"),
        format!("{cuda_path}/targets/x86_64-linux/lib"),
        format!("{cuda_path}/thor/targets/aarch64-linux/lib"),
    ];
    let lib_dirs: Vec<String> = candidate_lib_dirs
        .iter()
        .filter(|d| std::path::Path::new(d).exists())
        .cloned()
        .collect();

    // Check if CUDA is available
    let has_cuda = !lib_dirs.is_empty();

    if has_cuda {
        for d in &lib_dirs {
            println!("cargo:rustc-link-search=native={d}");
        }
        println!("cargo:rustc-link-lib=cudart");
        println!("cargo:rustc-link-lib=cuda");
        println!("cargo:rustc-link-lib=cublas");
        println!("cargo:rustc-link-lib=cublasLt");

        // NVTX naming split: desktop CUDA 12 ships `libnvtx3interop.so`, but
        // Drive OS 7 (and older embedded CUDA runtimes) ship the classic
        // `libnvToolsExt.so` (NVTX v2). Both export `nvtxRangePushA` /
        // `nvtxRangePop` — we just have to pick the right `-l` name. Emit a
        // cfg the `nvtx` module reads. Search across all lib dirs since the
        // NVTX lib may sit in a different subdir than cublas (Drive OS does
        // exactly this).
        if std::env::var("CARGO_FEATURE_NVTX").is_ok() {
            let has_v3 = lib_dirs.iter().any(|d| {
                std::path::Path::new(&format!("{d}/libnvtx3interop.so")).exists()
                    || std::path::Path::new(&format!("{d}/libnvtx3interop.so.1")).exists()
            });
            let has_v2 = lib_dirs.iter().any(|d| {
                std::path::Path::new(&format!("{d}/libnvToolsExt.so")).exists()
                    || std::path::Path::new(&format!("{d}/libnvToolsExt.so.1")).exists()
            });
            let header = format!("{cuda_path}/include/nvtx3/nvToolsExt.h");
            if std::path::Path::new(&header).exists() {
                println!("cargo:rerun-if-changed=adapters/nvtx.c");
                println!("cargo:rerun-if-changed={header}");
                cc::Build::new()
                    .file("adapters/nvtx.c")
                    .include(format!("{cuda_path}/include"))
                    .compile("apxinf_nvtx");
                println!("cargo:rustc-link-lib=dl");
                println!("cargo:rustc-cfg=nvtx_header");
            } else if has_v3 {
                // Desktop CUDA 12.x. `nvtx.rs` will use `#[link(name = "nvtx3interop")]`.
                println!("cargo:rustc-cfg=nvtx_v3");
            } else if has_v2 {
                // Drive OS / embedded CUDA. `nvtx.rs` falls back to `-lnvToolsExt`.
                println!("cargo:rustc-cfg=nvtx_v2");
            } else {
                println!("cargo:warning=NVTX unavailable: no NVTX3 headers or legacy libraries; markers disabled");
            }
            // Suppress the "unexpected cfg" warning under Rust 1.80+ check-cfg.
        }

        // Compile CUDA kernels if nvcc is available
        let manifest_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
        let kernels_dir = format!("{manifest_dir}/kernels");
        let adapters_dir = format!("{manifest_dir}/adapters");
        if std::path::Path::new(&kernels_dir).exists() {
            // Track the directory as well as its current members so adding a
            // new top-level .cu file invalidates an existing Cargo build.
            println!("cargo:rerun-if-changed={kernels_dir}");
            println!("cargo:rerun-if-changed={adapters_dir}");
            // The build ID hashes every CUDA/CUTLASS source below kernels/.
            // Track the same complete tree so an in-place header edit cannot
            // leave Cargo using a stale build ID or stale device object.
            emit_rerun_if_changed_tree(std::path::Path::new(&kernels_dir));
            emit_rerun_if_changed_tree(std::path::Path::new(&adapters_dir));
            // Pick a target arch: explicit override > native CUDA device
            // detection. Never infer a GPU architecture from the CPU target:
            // Orin, Thor-U, and Thor are all aarch64 but require different
            // device code. Cross-compilation therefore requires an override.
            let arch_for_target = env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
            let out_dir = env::var("OUT_DIR").unwrap();
            let nvcc = {
                let bundled = format!("{cuda_path}/bin/nvcc");
                if std::path::Path::new(&bundled).exists() {
                    bundled
                } else {
                    "nvcc".to_string()
                }
            };
            let host = env::var("HOST").unwrap_or_default();
            let target = env::var("TARGET").unwrap_or_default();
            let selection = select_cuda_arch(
                env::var("APXINF_CUDA_ARCH").ok(),
                env::var("APXINF_CUDA_ARCH_CUTLASS").ok(),
                &host,
                &target,
                std::path::Path::new(&nvcc),
                std::path::Path::new(&out_dir),
            )
            .unwrap_or_else(|error| {
                panic!(
                    "CUDA architecture selection failed: {error}\n\
                     Set APXINF_CUDA_ARCH explicitly when auto-detection is unavailable \
                     (Orin: sm_87, Thor-U: sm_101, Thor: sm_110)."
                )
            });
            match &selection.source {
                ArchSource::Explicit => println!(
                    "cargo:warning=ApxInf CUDA architecture: {} (explicit), CUTLASS: {}",
                    selection.nvcc_arch, selection.cutlass_arch
                ),
                ArchSource::Detected { device_count } => println!(
                    "cargo:warning=ApxInf CUDA architecture: {} (auto-detected from {device_count} visible GPU(s)), CUTLASS: {}",
                    selection.nvcc_arch, selection.cutlass_arch
                ),
            }
            let nvcc_arch = Some(selection.nvcc_arch);
            let cutlass_arch = Some(selection.cutlass_arch);
            let aot_manifest =
                env::var_os("APXINF_CUDA_AOT_MANIFEST").map(std::path::PathBuf::from);
            if matches!(nvcc_arch.as_deref(), Some("sm_110" | "sm_110a")) && aot_manifest.is_none()
            {
                println!("cargo:warning=SM110 AOT bundle is absent; fixed BF16 operators are unavailable. See crates/apxinf-cuda/aot/README.md");
            }
            // A bundle may be exported for another crate's recipes: one
            // `APXINF_CUDA_AOT_MANIFEST` feeds both CUDA crates. Carrying none
            // of this crate's kernels means "no bundle", not a build error.
            let aot_bundle = aot_manifest.and_then(|path| {
                let bundle = aot::Bundle::load(
                    std::path::Path::new(&manifest_dir),
                    std::path::Path::new(&path),
                    &target,
                    nvcc_arch.as_deref().unwrap(),
                    std::path::Path::new(&nvcc),
                );
                if bundle.is_none() {
                    println!(
                        "cargo:warning=AOT bundle {} carries none of this crate's reviewed kernels; fixed BF16 operators are unavailable. See crates/apxinf-cuda/aot/README.md",
                        path.display()
                    );
                }
                bundle
            });
            let kernel_build_id = env::var("APXINF_KERNEL_BUILD_ID").unwrap_or_else(|_| {
                computed_kernel_build_id(
                    std::path::Path::new(&manifest_dir),
                    &arch_for_target,
                    nvcc_arch.as_deref(),
                    cutlass_arch.as_deref(),
                )
            });
            let kernel_build_id = match &aot_bundle {
                Some(bundle) => {
                    let mut hash = 0x6c62272e07bb014262b821756295c58du128;
                    hash_bytes(&mut hash, bundle.fingerprint.as_bytes());
                    format!("{kernel_build_id}-aot-{hash:032x}")
                }
                None => kernel_build_id,
            };
            emit_kernel_build_id(&kernel_build_id);

            // Compile host adapters from an explicit list. Template-heavy
            // CUTLASS operators are added below as their own translation
            // units; adapters include declaration-only headers and own only
            // the stable C ABI.
            let mut kernel_files = vec![
                std::path::Path::new(&adapters_dir).join("core_kernels_adapter.cu"),
                std::path::Path::new(&adapters_dir).join("sampling_adapter.cu"),
                std::path::Path::new(&adapters_dir).join("static_bf16_adapter.cu"),
                std::path::Path::new(&adapters_dir).join("w8a8_adapter.cu"),
                std::path::Path::new(&adapters_dir).join("custom_kernels.cu"),
                std::path::Path::new(&adapters_dir).join("cublas_adapter.cu"),
                std::path::Path::new(&adapters_dir).join("cublaslt_adapter.cu"),
            ];
            assert!(
                kernel_files.iter().all(|path| path.is_file()),
                "one or more required CUDA adapters are missing under {adapters_dir}"
            );

            // Per-operator AOT attribution arms. A feature without a usable
            // bundle is a hard error: a typo that quietly measures the in-tree
            // arm is the failure mode these exist to prevent.
            let want_swiglu_aot =
                env::var_os("CARGO_FEATURE_AOT_GEMM_SWIGLU_BF16").is_some();
            let want_gdn_aot = env::var_os("CARGO_FEATURE_AOT_GDN_PREFILL_BF16").is_some();
            let want_vision_aot = env::var_os("CARGO_FEATURE_AOT_VISION_ATTENTION").is_some();
            let want_action_aot = env::var_os("CARGO_FEATURE_AOT_ACTION_ATTENTION").is_some();
            let want_language_aot =
                env::var_os("CARGO_FEATURE_AOT_LANGUAGE_ATTENTION").is_some();
            for (want, feature) in [
                (want_swiglu_aot, "aot-gemm-swiglu-bf16"),
                (want_gdn_aot, "aot-gdn-prefill-bf16"),
                (want_vision_aot, "aot-vision-attention"),
                (want_action_aot, "aot-action-attention"),
                (want_language_aot, "aot-language-attention"),
            ] {
                assert!(
                    !want || aot_bundle.is_some(),
                    "feature {feature} needs a usable SM110 AOT bundle; set \
                     APXINF_CUDA_AOT_MANIFEST. Refusing to silently build the \
                     in-tree arm instead."
                );
            }

            if let Some(bundle) = &aot_bundle {
                bundle.kernel("gemm-swiglu-bf16");
                bundle.kernel("gdn-prefill-bf16");
                if want_gdn_aot {
                    kernel_files
                        .push(std::path::Path::new(&adapters_dir).join("gdn_bf16_aot_adapter.cu"));
                    println!("cargo:rustc-cfg=apxinf_aot_gdn_prefill_bf16");
                }
                if want_swiglu_aot {
                    kernel_files.push(
                        std::path::Path::new(&adapters_dir).join("quack_m256n256_adapter.cu"),
                    );
                    println!("cargo:rustc-cfg=apxinf_aot_gemm_swiglu_bf16");
                }
                // The bundle-completeness assertion is unconditional: a bundle
                // that is missing one of the four attention objects is a broken
                // bundle whichever arm is being built.
                for id in [
                    "language-attention",
                    "action-attention",
                    "vision-attention-624",
                    "vision-attention-2200",
                ] {
                    bundle.kernel(id);
                }
                if want_vision_aot {
                    kernel_files.push(
                        std::path::Path::new(&adapters_dir).join("fa4_fixed_groups_adapter.cu"),
                    );
                    println!("cargo:rustc-cfg=apxinf_aot_vision_attention");
                }
                if want_action_aot {
                    kernel_files.push(
                        std::path::Path::new(&adapters_dir)
                            .join("fa4_d256_split_batch_adapter.cu"),
                    );
                    println!("cargo:rustc-cfg=apxinf_aot_action_attention");
                }
                if want_language_aot {
                    kernel_files.push(
                        std::path::Path::new(&adapters_dir).join("fa4_d256_native_adapter.cu"),
                    );
                    println!("cargo:rustc-cfg=apxinf_aot_language_attention");
                }
            }

            // Vendored FlashInfer Cake GDN prefill (BF16 I/O, full-DV). The
            // kernel translation unit must stay separate from the descriptor
            // one: the generated header declares its own `CUtensorMap` and
            // <cuda.h> declares another.
            let gdn_cuda_launch =
                std::path::Path::new(&adapters_dir).join("gdn_bf16_cuda_launch.cu");
            let gdn_cuda_adapter =
                std::path::Path::new(&adapters_dir).join("gdn_bf16_cuda_adapter.cu");
            // The conv/pack/gate prologue and the state transpose, in one
            // translation unit shared by both arms. It cannot be a header: the
            // kernels are `__global__` and would be emitted with external
            // linkage in every includer.
            let gdn_prologue =
                std::path::Path::new(&adapters_dir).join("gdn_bf16_prefill_prologue.cu");

            let cutlass_root = std::path::Path::new(&kernels_dir).join("cutlass");
            let cutlass_fmha_operator = cutlass_root.join("fmha_sm100.cu");
            let cutlass_gemm_operator = cutlass_root.join("fp8_gemm_sm100.cu");
            let cutlass_fp8_dual_operator = cutlass_root.join("fp8_dual_geglu_sm100.cu");
            let cutlass_fp8_header = cutlass_root.join("fp8_operators_sm100.h");
            let cutlass_bf16_operator = cutlass_root.join("bf16_gemm_sm100.cu");
            let cutlass_bf16_dual_operator = cutlass_root.join("bf16_dual_geglu_sm100.cu");
            let cutlass_bf16_header = cutlass_root.join("bf16_operators_sm100.h");
            let cutlass_bf16_sm89_operator = cutlass_root.join("bf16_gemm_sm89.cu");
            let cutlass_bf16_sm89_header = cutlass_root.join("bf16_operators_sm89.h");
            let cutlass_int8_operator = cutlass_root.join("w8a8_gemm_sm80.cu");
            let cutlass_fmha = std::path::Path::new(&adapters_dir).join("cutlass_fmha_adapter.cu");
            // In-tree CUDA arms of the three FA4 attention operators. One
            // translation unit each because the operator's numeric contract IS
            // its `-DAPXINF_*` flag line (see `fmha_d256_sm100.cuh`), and the
            // flags configure the vendored FMHA headers without taking part in
            // name mangling.
            let fa4_vision_operator = cutlass_root.join("fmha_vision_sm100.cu");
            let fa4_language_operator = cutlass_root.join("fmha_d256_language_sm100.cu");
            let fa4_action_operator = cutlass_root.join("fmha_d256_action_sm100.cu");
            let fa4_vision_cuda =
                std::path::Path::new(&adapters_dir).join("fa4_fixed_groups_cuda_adapter.cu");
            let fa4_language_cuda =
                std::path::Path::new(&adapters_dir).join("fa4_d256_cuda_adapter.cu");
            let fa4_action_cuda =
                std::path::Path::new(&adapters_dir).join("fa4_d256_split_batch_cuda_adapter.cu");
            // Q duplication and the LSE merge are plain CUDA kernels shared by
            // BOTH arms, so a difference can only come from the attention
            // launch -- the same discipline `gdn_bf16_prefill_prologue.cu`
            // follows.
            let fa4_split_batch_merge =
                std::path::Path::new(&adapters_dir).join("fa4_d256_split_batch_merge.cu");
            let cutlass_gemm = std::path::Path::new(&adapters_dir).join("cutlass_fp8_adapter.cu");
            let cutlass_bf16 = std::path::Path::new(&adapters_dir).join("cutlass_bf16_adapter.cu");
            let cutlass_bf16_sm89 =
                std::path::Path::new(&adapters_dir).join("cutlass_bf16_sm89_adapter.cu");
            let cutlass_int8 = std::path::Path::new(&adapters_dir).join("cutlass_w8a8_adapter.cu");
            let mut cutlass_includes = Vec::new();
            if cutlass_arch.as_deref().is_some_and(is_cutlass_sm100_family) {
                let fmha = cutlass_root.join("fmha");
                let cutlass = cutlass_root.join("include");
                let cutlass_utils = cutlass_root.join("tools/util/include");
                assert!(
                    fmha.is_dir() && cutlass.is_dir() && cutlass_utils.is_dir(),
                    "vendored CUTLASS/FMHA headers are incomplete under {}",
                    cutlass_root.display()
                );
                assert!(
                    cutlass_gemm_operator.is_file()
                        && cutlass_fp8_dual_operator.is_file()
                        && cutlass_fp8_header.is_file()
                        && cutlass_bf16_operator.is_file()
                        && cutlass_bf16_dual_operator.is_file()
                        && cutlass_bf16_header.is_file()
                        && cutlass_fmha_operator.is_file()
                        && cutlass_gemm.is_file()
                        && cutlass_bf16.is_file()
                        && cutlass_fmha.is_file(),
                    "CUTLASS operators or native C ABI adapters are missing"
                );
                cutlass_includes.extend([fmha, cutlass, cutlass_utils]);
                kernel_files.extend([
                    cutlass_gemm_operator.clone(),
                    cutlass_fp8_dual_operator.clone(),
                    cutlass_bf16_operator.clone(),
                    cutlass_bf16_dual_operator.clone(),
                    cutlass_gemm.clone(),
                    cutlass_bf16.clone(),
                ]);
                kernel_files.push(cutlass_fmha.clone());
                println!("cargo:rustc-cfg=apxinf_cutlass_gemm");
                println!("cargo:rustc-cfg=apxinf_cutlass_fmha");
                // The three FA4 attention operators are SM110-only (single-Q
                // stage `setmaxnreg`, `UTCHMMA`) and need no bundle, so they
                // are compiled on every SM110 target INCLUDING the AOT arms --
                // linking both implementations is what makes the in-process
                // equivalence tests possible.
                if matches!(nvcc_arch.as_deref(), Some("sm_110" | "sm_110a")) {
                    assert!(
                        fa4_vision_operator.is_file()
                            && fa4_language_operator.is_file()
                            && fa4_action_operator.is_file()
                            && cutlass_root.join("fmha_d256_sm100.cuh").is_file()
                            && fa4_vision_cuda.is_file()
                            && fa4_language_cuda.is_file()
                            && fa4_action_cuda.is_file()
                            && fa4_split_batch_merge.is_file(),
                        "in-tree FA4 attention operators or adapters are missing"
                    );
                    kernel_files.extend([
                        fa4_vision_operator.clone(),
                        fa4_language_operator.clone(),
                        fa4_action_operator.clone(),
                        fa4_vision_cuda.clone(),
                        fa4_language_cuda.clone(),
                        fa4_action_cuda.clone(),
                        fa4_split_batch_merge.clone(),
                    ]);
                    println!("cargo:rustc-cfg=apxinf_fa4_attention_cuda");
                }
                emit_rerun_if_changed_tree(&cutlass_root);
            }

            // The vendored GDN kernel is SM110-only (it issues UTCHMMA) and
            // needs no bundle, so it is compiled on every SM110 target
            // including the AOT arm -- linking both is what makes the
            // in-process equivalence test possible.
            if matches!(nvcc_arch.as_deref(), Some("sm_110" | "sm_110a")) {
                let gdn_root = std::path::Path::new(&kernels_dir).join("flashinfer_gdn");
                assert!(
                    gdn_cuda_launch.is_file()
                        && gdn_cuda_adapter.is_file()
                        && gdn_prologue.is_file()
                        && gdn_root.join("cake_gdn_prefill_b384fd71263b.cu").is_file()
                        && gdn_root.join("gdn_prefill_generated.cuh").is_file(),
                    "vendored FlashInfer GDN sources or adapters are missing under {}",
                    gdn_root.display()
                );
                kernel_files.push(gdn_prologue.clone());
                kernel_files.push(gdn_cuda_launch.clone());
                kernel_files.push(gdn_cuda_adapter.clone());
                emit_rerun_if_changed_tree(&gdn_root);
                println!("cargo:rustc-cfg=apxinf_gdn_prefill_bf16_cuda");
            }

            if cutlass_arch.as_deref().is_some_and(is_cutlass_sm89_family) {
                let cutlass = cutlass_root.join("include");
                let cutlass_utils = cutlass_root.join("tools/util/include");
                assert!(
                    cutlass_bf16_sm89_operator.is_file()
                        && cutlass_bf16_sm89_header.is_file()
                        && cutlass_bf16_sm89.is_file()
                        && cutlass.is_dir()
                        && cutlass_utils.is_dir(),
                    "CUTLASS BF16 SM89 adapter or headers are missing under {}",
                    cutlass_root.display()
                );
                cutlass_includes.extend([cutlass_root.clone(), cutlass, cutlass_utils]);
                kernel_files.extend([
                    cutlass_bf16_sm89_operator.clone(),
                    cutlass_bf16_sm89.clone(),
                ]);
                println!("cargo:rustc-cfg=apxinf_cutlass_bf16_sm89");
                emit_rerun_if_changed_tree(&cutlass_root);
                emit_rerun_if_changed_tree(std::path::Path::new(&adapters_dir));
            }

            let mut cutlass_int8_includes = Vec::new();
            if nvcc_arch.as_deref().is_some_and(is_fa2_sm80_family) {
                let cutlass = cutlass_root.join("include");
                let cutlass_utils = cutlass_root.join("tools/util/include");
                let extensions = cutlass_root.join("extensions");
                assert!(
                    cutlass_int8_operator.is_file()
                        && cutlass_int8.is_file()
                        && cutlass.is_dir()
                        && cutlass_utils.is_dir()
                        && extensions
                            .join("epilogue/epilogue_per_row_per_col_scale.h")
                            .is_file()
                        && extensions
                            .join("gemm/gemm_universal_base_compat.h")
                            .is_file()
                        && extensions
                            .join("gemm/gemm_with_epilogue_visitor.h")
                            .is_file(),
                    "vendored SM80 INT8 CUTLASS sources are incomplete under {}",
                    cutlass_root.display()
                );
                cutlass_int8_includes.extend([cutlass_root.clone(), cutlass, cutlass_utils]);
                kernel_files.push(cutlass_int8.clone());
                println!("cargo:rustc-cfg=apxinf_cutlass_int8_sm80");
                emit_rerun_if_changed_tree(&extensions);
            }

            let fa2_root = cutlass_root.join("fa2");
            let fa2_compat = cutlass_root.join("fa2_compat");
            let fa2_operator = cutlass_root.join("fa2_bf16_sm80.cu");
            let fa2_hdim64_apx = cutlass_root.join("fa2_hdim64_bf16.cu");
            let fa2_hdim96_apx = cutlass_root.join("fa2_hdim96_bf16.cu");
            let fa2_wrapper = std::path::Path::new(&adapters_dir).join("fa2_adapter.cu");
            let mut fa2_sources = Vec::new();
            let mut fa2_direct_e4m3_sources = Vec::new();
            let mut fa2_includes = Vec::new();
            let fa2_sm80 = nvcc_arch.as_deref().is_some_and(is_fa2_bf16_arch);
            let fa2_f16_sm100 = nvcc_arch.as_deref().is_some_and(is_cutlass_sm100_family);
            let fa2_head_special = fa2_sm80 || fa2_f16_sm100;
            if fa2_sm80 || fa2_f16_sm100 {
                let fa2_hdim96 = fa2_root.join("flash_attn/flash_fwd_hdim96_bf16_sm80.cu");
                let fa2_hdim128 = fa2_root.join("flash_attn/flash_fwd_hdim128_bf16_sm80.cu");
                let fa2_hdim256 = fa2_root.join("flash_attn/flash_fwd_hdim256_bf16_sm80.cu");
                let fa2_f16_hdim96 = fa2_root.join("flash_attn/flash_fwd_hdim96_fp16.cu");
                let fa2_f16_hdim256 = fa2_root.join("flash_attn/flash_fwd_hdim256_fp16.cu");
                let fa2_cutlass = fa2_root.join("cutlass/include");
                assert!(
                    fa2_operator.is_file()
                        && fa2_hdim64_apx.is_file()
                        && fa2_hdim96_apx.is_file()
                        && fa2_wrapper.is_file()
                        && fa2_hdim96.is_file()
                        && fa2_hdim128.is_file()
                        && fa2_hdim256.is_file()
                        && fa2_f16_hdim96.is_file()
                        && fa2_f16_hdim256.is_file()
                        && fa2_cutlass.is_dir(),
                    "vendored FlashAttention-2 sources are incomplete under {}",
                    fa2_root.display()
                );
                fa2_sources.extend([
                    fa2_wrapper.clone(),
                    fa2_hdim64_apx,
                    fa2_hdim96_apx,
                    fa2_hdim96,
                    fa2_hdim128,
                    fa2_hdim256,
                    fa2_f16_hdim96,
                    fa2_f16_hdim256,
                ]);
                let fa2_split_hdim256 =
                    fa2_root.join("flash_attn/flash_fwd_split_hdim256_bf16_sm80.cu");
                assert!(
                    fa2_split_hdim256.is_file(),
                    "vendored FlashAttention-2 split-KV source is incomplete under {}",
                    fa2_root.display()
                );
                fa2_sources.push(fa2_split_hdim256);
                // The head-64 and head-256 specialisations are ordinary
                // Ampere-MMA FA2 kernels; Blackwell runs them. They were tied
                // to the SM80 family only because that was the first device
                // that needed them. `fa2_head_special` is the axis that
                // decides whether the specialised dispatch exists, and it is
                // now on wherever the underlying hdim kernels are compiled.
                if fa2_head_special {
                    fa2_sources
                        .push(std::path::Path::new(&adapters_dir).join("fa2_head64_adapter.cu"));
                    fa2_sources
                        .push(std::path::Path::new(&adapters_dir).join("fa2_head256_adapter.cu"));
                }
                if fa2_f16_sm100 {
                    println!("cargo:rustc-cfg=apxinf_fa2_f16_sm100");
                    let direct_operator = cutlass_root.join("fa2_f16_e4m3_sm100.cu");
                    let direct_wrapper =
                        std::path::Path::new(&adapters_dir).join("fa2_direct_e4m3_adapter.cu");
                    let direct_hdim256 = fa2_root.join("flash_attn/flash_fwd_hdim256_fp16_e4m3.cu");
                    assert!(
                        direct_operator.is_file()
                            && direct_wrapper.is_file()
                            && direct_hdim256.is_file(),
                        "FA2 direct-E4M3 FA2 sources are incomplete"
                    );
                    fa2_direct_e4m3_sources.extend([direct_wrapper, direct_hdim256]);
                    kernel_files.extend(fa2_direct_e4m3_sources.iter().cloned());
                    println!("cargo:rustc-cfg=apxinf_fa2_direct_e4m3_sm100");
                }
                fa2_includes.extend([fa2_compat.clone(), fa2_root.clone(), fa2_cutlass]);
                kernel_files.extend(fa2_sources.iter().cloned());
                if fa2_sm80 {
                    println!("cargo:rustc-cfg=apxinf_fa2_sm80");
                }
                if fa2_head_special {
                    println!("cargo:rustc-cfg=apxinf_fa2_head_special");
                }
                emit_rerun_if_changed_tree(&fa2_root);
                emit_rerun_if_changed_tree(&fa2_compat);
            }

            if !kernel_files.is_empty() {
                let target_include_dirs = [
                    format!("{cuda_path}/include"),
                    format!("{cuda_path}/targets/{arch}/include"),
                    format!("{cuda_path}/targets/aarch64-linux/include"),
                    format!("{cuda_path}/thor/targets/aarch64-linux/include"),
                ];
                let workers = nvcc_build::parallelism();
                // `--generate-code arch=compute_NNNa,code=sm_NNNa`, never
                // `-arch=sm_NNNa`: see `cuda_arch::gencode_args`. Without this
                // no arch-specific SM110 instruction can be assembled, so the
                // CUTLASS `reg_reconfig.h` SM110 gate below would turn
                // `setmaxnreg` into a hard ptxas error instead of a speedup.
                let generic_codegen = gencode_args(nvcc_arch.clone());
                let cutlass_codegen = gencode_args(cutlass_arch.clone());
                // Translation units that include vendored CUTLASS headers.
                // They get the arch-specific `sm_110a` codegen, the CUTLASS
                // include path, the relaxed-constexpr/extended-lambda pair and
                // `-DNDEBUG`. The list used to be spelled out twice as an
                // `entry == &..` chain; it is now one set because the SM110
                // operator translation units are appended to it.
                let cutlass_tus: Vec<std::path::PathBuf> = vec![
                    cutlass_fmha.clone(),
                    cutlass_gemm_operator.clone(),
                    cutlass_fp8_dual_operator.clone(),
                    cutlass_bf16_operator.clone(),
                    cutlass_bf16_dual_operator.clone(),
                    cutlass_bf16_sm89_operator.clone(),
                    cutlass_bf16_sm89.clone(),
                    cutlass_gemm.clone(),
                    cutlass_bf16.clone(),
                    fa4_vision_operator.clone(),
                    fa4_language_operator.clone(),
                    fa4_action_operator.clone(),
                ];
                // The numeric contract of each FA4 attention operator, as a
                // flag line. This is the SAME vocabulary as
                // `devlocal/aot2cuda/harness/MERGED_FLAGS.sh`, which is
                // authoritative and carries the per-flag provenance; changing
                // a value here without re-running that harness silently drops
                // bit-exactness to roughly 50%, not to an error. Every macro
                // defaults to its UPSTREAM value in the headers, so the stock
                // `cutlass_fmha_adapter.cu` translation unit -- which takes NO
                // `-DAPXINF_*` -- stays byte-identical to pristine `main`.
                //
                // Shared by the two head_dim-256 operators. `EX2_APPROX`,
                // `DEBUG_SHOW_SO`, `FUSE_ROWSUM`, `P_FLUSH_HALF`,
                // `CORR_RING2` and `LOAD_SHAPE_PER_MODE` were unconditional in
                // the fork and their upstream defaults are the other way
                // round, so they have to be spelled out.
                //
                // `ROWSUM_RESCALE_RN` is intentionally NOT shared. The two
                // d256 AOT objects encode different arithmetic here: language
                // keeps the multiply and packed add separately rounded, while
                // action contracts them into FFMA. Each operator pins its own
                // value below.
                let fa4_d256_common: &[&str] = &[
                    "-DAPXINF_EX2_APPROX=1",
                    "-DAPXINF_EX2_FTZ=1",
                    "-DAPXINF_DEBUG_SHOW_SO=0",
                    "-DAPXINF_FUSE_ROWSUM=1",
                    "-DAPXINF_P_FLUSH_HALF=1",
                    "-DAPXINF_CORR_RING2=1",
                    "-DAPXINF_LOAD_SHAPE_PER_MODE=1",
                ];
                let mut jobs: Vec<CompileJob> = Vec::new();
                for entry in &kernel_files {
                    println!("cargo:rerun-if-changed={}", entry.display());
                    let stem = entry.file_stem().unwrap().to_string_lossy().to_string();
                    let obj = format!("{out_dir}/{stem}.o");
                    let mut cmd = std::process::Command::new(&nvcc);
                    cmd.args([
                        "-c",
                        &entry.to_string_lossy(),
                        "-o",
                        &obj,
                        "--compiler-options",
                        "-fPIC",
                        "-O3",
                        "-std=c++17",
                    ]);
                    if let Some(bundle) = &aot_bundle {
                        for kernel in bundle.kernels.values() {
                            cmd.arg(format!("-I{}", kernel.header.parent().unwrap().display()));
                        }
                    }
                    let is_cutlass_tu = cutlass_tus.contains(entry);
                    // The vendored GDN kernel is not a CUTLASS translation
                    // unit but does need the arch-specific `sm_110a` target:
                    // it issues UTCHMMA, which the non-`a` target rejects.
                    let is_arch_specific = is_cutlass_tu || entry == &gdn_cuda_launch;
                    cmd.args(if is_arch_specific {
                        &cutlass_codegen
                    } else {
                        &generic_codegen
                    });
                    if entry == &gdn_cuda_launch {
                        // 16 qk heads, 32 v heads, 2^1 v per qk, 2^5 o heads.
                        // Not defined in the vendored source; see
                        // kernels/flashinfer_gdn/README.md. Deliberately no
                        // `--use_fast_math`.
                        cmd.args([
                            "-DHEAD_GROUP_LOG2=1",
                            "-DNUM_O_HEADS_LOG2=5",
                            "-DIS_GQA=0",
                            "-DNDEBUG",
                        ]);
                    }
                    if nvcc_arch.as_deref().is_some_and(is_pre_sm80) {
                        cmd.arg("-DAPXINF_GDN_WMMA_UNSUPPORTED=1");
                    }
                    if entry.ends_with("cublaslt_adapter.cu")
                        && nvcc_arch.as_deref().is_some_and(is_cutlass_sm100_family)
                    {
                        // Keep the measured Blackwell FP8 NN layout. Other
                        // targets stage KN weights into cuBLASLt's TN layout.
                        cmd.arg("-DAPXINF_FP8_NN_LAYOUT=1");
                    }
                    for include in &target_include_dirs {
                        if std::path::Path::new(include).exists() {
                            cmd.arg(format!("-I{include}"));
                        }
                    }
                    if is_cutlass_tu {
                        cmd.arg("--expt-relaxed-constexpr");
                        cmd.arg("--expt-extended-lambda");
                        // `nvcc -O3` does not imply `-DNDEBUG`. Without it
                        // every `CUTLASS_ASSERT` survives into device code:
                        // the FMHA adapter shipped 97 `BPT.TRAP` instructions
                        // on `main`. Scoped to the CUTLASS translation units so
                        // the other adapters keep their asserts.
                        cmd.arg("-DNDEBUG");
                        for include in &cutlass_includes {
                            cmd.arg(format!("-I{}", include.display()));
                        }
                    }
                    if entry == &cutlass_fp8_dual_operator {
                        cmd.arg("-DAPXINF_FP8_DUAL_GEGLU_PRODUCTION=1");
                    }
                    if entry == &cutlass_bf16_dual_operator {
                        cmd.arg("-DAPXINF_BF16_DUAL_GEGLU_PRODUCTION=1");
                    }
                    if entry == &fa4_vision_operator {
                        // vision hd64: `flash_fwd_sm100.py` contract, two Q
                        // stages, PersistentTileScheduler, ex2 emulation ON,
                        // `rcp.approx` normalisation (FINAL_NORM=1), R2P mask.
                        // `ACC_SCALE_HALF`'s UPSTREAM default is 1, so the 0
                        // here is load-bearing. The last two are bit-neutral
                        // scheduling knobs (report 21; re-verified element for
                        // element in report 32 §1).
                        //
                        // 🔴 DO NOT re-add `-DAPXINF_NO_ORDER_S=1`. Report 21
                        // §①.4 adopted it as that round's largest single win
                        // (1.0648 -> 1.0424 at L=624, 2.3%) and recorded it as
                        // bit-neutral; that was measured on the `+-0.05`
                        // uniform stimulus, which report 31 §2.3 showed is the
                        // degenerate `mean(V)` regime. It is NOT bit-neutral.
                        // Dropping CUTLASS' `OrderBarrierSoftmax` between the
                        // two softmax warpgroups makes the operator differ from
                        // the AOT object by exactly one BF16 ULP on ~3 of every
                        // 1e8 elements of a model-shaped (heavy-tailed) input,
                        // in BOTH AOT objects, and that single bit moves the
                        // Qwen-Drive trajectory by 0.322 m (= 2 BF16 ULP of one
                        // waypoint x `trajectory_scale[0]`). Report 32 §1.1:
                        // the model-shaped gate goes 3/100073472 -> 0, and the
                        // captured `vis01.qkv` tensor that produced the
                        // in-model divergence goes 12509183 -> 12509184.
                        //
                        // The four scheduling numbers below (`NUM_REGS_*` and
                        // `FMHA_KV_STAGES`) are a MEASURED local optimum of
                        // the repo path and every neighbour swept is worse --
                        // see report 30 and the tables beside each knob's
                        // `#define`. The vision fork's standalone L=624 driver
                        // prefers 192/96/32 and KV=6; both lose here, as do
                        // `CORR_TILE=8` and `CORR_RING2=1`, so those two stay
                        // at their upstream defaults for this operator.
                        cmd.args([
                            "-DAPXINF_MASK_R2P=1",
                            "-DAPXINF_KV_DESCENDING=1",
                            "-DAPXINF_RESCALE_THRESHOLD=8.0f",
                            "-DAPXINF_EX2_APPROX=1",
                            "-DAPXINF_EX2_FTZ=1",
                            "-DAPXINF_DEBUG_SHOW_SO=0",
                            "-DAPXINF_FUSE_ROWSUM=1",
                            "-DAPXINF_EX2_EMU_FREQ=16",
                            "-DAPXINF_EX2_EMU_RES=4",
                            "-DAPXINF_EX2_EMU_START_FRG=1",
                            "-DAPXINF_ACC_SCALE_HALF=0",
                            "-DAPXINF_FINAL_NORM=1",
                            "-DAPXINF_NO_LSE=1",
                            "-DAPXINF_PERSISTENT=1",
                            "-DAPXINF_NUM_REGS_SOFTMAX=200",
                            "-DAPXINF_NUM_REGS_CORRECTION=64",
                            "-DAPXINF_NUM_REGS_OTHER=48",
                            "-DAPXINF_FMHA_KV_STAGES=4",
                            "-DAPXINF_SKIP_PAD_QSTAGE=1",
                            "-DAPXINF_P_CHUNKS16=1",
                            // 🔴 report 35. Bit-neutral HERE (measured: the
                            // model-shaped gate stays 0/100073472 over 8 seeds
                            // and all five scales stay 12509184/12509184),
                            // because `RESCALE_THRESHOLD=8.0f` keeps
                            // `acc_scale` at exactly 1.0f on essentially every
                            // tile and `row_sum * 1.0f` is exact either way.
                            // It is set anyway: the defect it removes is
                            // LATENT, not absent, and it would surface on any
                            // input whose row max climbs by more than 2^8
                            // between KV tiles. See the macro's block in
                            // sm100_fmha_fwd_mainloop_tma_warpspecialized.hpp.
                            "-DAPXINF_ROWSUM_RESCALE_RN=1",
                        ]);
                    }
                    if entry == &fa4_action_operator {
                        // action hd256 non-causal: DESCENDING KV, mask first,
                        // no ex2 emulation, `rcp.approx` + 0/NaN->1 guard.
                        // `LSE_FA4` is the ONE operator that needs it: this is
                        // the only arm that emits an LSE, and its split-batch
                        // merge weight is `sigmoid(lse0 - lse1)`, which at
                        // `key_tokens = 3436` has no leading term left to
                        // absorb a 1-fp32-ULP log difference. Report 27.
                        cmd.args(fa4_d256_common);
                        cmd.args([
                            "-DAPXINF_EX2_EMU_RES=0",
                            "-DAPXINF_RESCALE_THRESHOLD=8.0f",
                            "-DAPXINF_KV_DESCENDING=1",
                            "-DAPXINF_ROWSUM_QUARTERS=0",
                            "-DAPXINF_ACC_SCALE_HALF=0",
                            // The action AOT contracts carried row_sum scaling
                            // with the first packed accumulator add. Forcing
                            // add.rn.f32x2 caused the one real Qwen-Drive LSE
                            // mismatch; FFMA restores 1600/1600 bit equality.
                            "-DAPXINF_ROWSUM_RESCALE_RN=0",
                            "-DAPXINF_FINAL_NORM=2",
                            "-DAPXINF_LSE_FA4=1",
                        ]);
                    }
                    if entry == &fa4_language_operator {
                        // language hd256 causal: ASCENDING KV, mask last, no
                        // rescale threshold, ex2 emulation 14/6/0, `div.rn`
                        // with 0/NaN->0.0, row_sum over contiguous quarters
                        // seeded (s,s) -- QUARTERS and ACC_SCALE_HALF are a
                        // PAIR, 1/1 gives the 33 known elements and the other
                        // three combinations give 418/530/571 -- and the
                        // two-slot S pipeline that report 24's barrier fixed.
                        cmd.args(fa4_d256_common);
                        cmd.args([
                            "-DAPXINF_EX2_EMU_FREQ=14",
                            "-DAPXINF_EX2_EMU_RES=6",
                            "-DAPXINF_EX2_EMU_START_FRG=0",
                            "-DAPXINF_RESCALE_THRESHOLD=0.0f",
                            "-DAPXINF_KV_DESCENDING=0",
                            "-DAPXINF_ROWSUM_QUARTERS=1",
                            "-DAPXINF_ACC_SCALE_HALF=1",
                            // The language AOT uses separately rounded
                            // mul.f32 + add.rn.f32x2 (report 35).
                            "-DAPXINF_ROWSUM_RESCALE_RN=1",
                            "-DAPXINF_FINAL_NORM=3",
                            "-DAPXINF_S_PIPE_STAGES=2",
                        ]);
                    }
                    if entry == &cutlass_int8 {
                        cmd.arg("--expt-relaxed-constexpr");
                        cmd.arg("--expt-extended-lambda");
                        for include in &cutlass_int8_includes {
                            cmd.arg(format!("-I{}", include.display()));
                        }
                    }
                    if fa2_sources.contains(entry) {
                        cmd.args(["--expt-relaxed-constexpr", "--expt-extended-lambda"]);
                        // Reference SDPA specializations use libdevice exp/log in the
                        // split combiner; fast math changes BF16 output rounding.
                        if !entry.ends_with("fa2_head256_adapter.cu")
                            && !entry.ends_with("fa2_head64_adapter.cu")
                        {
                            cmd.arg("--use_fast_math");
                        }
                        cmd.args([
                            "-U__CUDA_NO_HALF_OPERATORS__",
                            "-U__CUDA_NO_HALF_CONVERSIONS__",
                            "-U__CUDA_NO_HALF2_OPERATORS__",
                            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                            "-DFLASH_NAMESPACE=apxinf_fa2",
                        ]);
                        if fa2_sm80 {
                            cmd.arg("-DAPXINF_FA2_SM80=1");
                        }
                        // Only the dispatch adapter and the two specialisations
                        // read this; the hdim kernels do not, and putting it on
                        // their command line would rebuild them for nothing.
                        if fa2_head_special
                            && (entry == &fa2_wrapper
                                || entry.ends_with("fa2_head64_adapter.cu")
                                || entry.ends_with("fa2_head256_adapter.cu"))
                        {
                            cmd.arg("-DAPXINF_FA2_HEAD_SPECIAL=1");
                        }
                        // The head-64 vision kernel is latency-bound at 16.67%
                        // theoretical occupancy: ptxas gives it 244 registers,
                        // which together with its 73.7 KB of shared memory caps
                        // residency at two blocks per SM on a 20-SM part.
                        //
                        // Capping the count buys a third block and loses more
                        // to spills than it gains: on Thor the scene goes from
                        // 889.9 ms to 911.9 at 168 registers, bit-identically,
                        // so this stays off. Kept as a knob
                        // because the trade is a per-board measurement.
                        if entry.ends_with("fa2_head64_adapter.cu") {
                            if let Ok(cap) = std::env::var("APXINF_FA2_HEAD64_MAXREG") {
                                if cap.trim().parse::<u32>().is_ok() {
                                    cmd.arg(format!("-maxrregcount={}", cap.trim()));
                                }
                            }
                        }
                        cmd.arg("-DAPXINF_FA2_SPLITKV=1");
                        // Drop the FlashAttention-2 feature axes the adapter
                        // never reaches. `fill_params` value-initializes the
                        // parameter block and then leaves `alibi_slopes_ptr`
                        // null and `softcap` zero; dropout is pinned to the
                        // keep-everything encoding (`p_dropout == 1.0`,
                        // `rp_dropout == 1.0`); and the window is either fully
                        // open (-1/-1) or causal (right == 0), so `Is_local` is
                        // never taken. Each macro removes one bool template
                        // axis and the instantiations are their product, so the
                        // four together cut ptxas work by up to 16x.
                        //
                        // That matters most on Orin, where ptxas needs multiple
                        // hours per FA2 translation unit at sm_87 -- measured at
                        // over 2.5h on flash_fwd_hdim128_bf16_sm80.cu alone,
                        // 99.8% CPU and 5.4 GB RSS.
                        //
                        // UNEVEN_K is deliberately not disabled: seqlen_k is a
                        // runtime value and is not block-aligned in general.
                        cmd.args([
                            "-DFLASHATTENTION_DISABLE_DROPOUT",
                            "-DFLASHATTENTION_DISABLE_ALIBI",
                            "-DFLASHATTENTION_DISABLE_SOFTCAP",
                            "-DFLASHATTENTION_DISABLE_LOCAL",
                        ]);
                        for include in &fa2_includes {
                            cmd.arg(format!("-I{}", include.display()));
                        }
                    }
                    if fa2_direct_e4m3_sources.contains(entry) {
                        cmd.args([
                            "--expt-relaxed-constexpr",
                            "--expt-extended-lambda",
                            "--use_fast_math",
                            "-U__CUDA_NO_HALF_OPERATORS__",
                            "-U__CUDA_NO_HALF_CONVERSIONS__",
                            "-U__CUDA_NO_HALF2_OPERATORS__",
                            "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
                            "-DFLASH_NAMESPACE=apxinf_fa2_direct_e4m3",
                            "-DAPXINF_FA2_DIRECT_E4M3=1",
                        ]);
                        // The direct-E4M3 path uses the same fixed feature set.
                        cmd.args([
                            "-DFLASHATTENTION_DISABLE_DROPOUT",
                            "-DFLASHATTENTION_DISABLE_ALIBI",
                            "-DFLASHATTENTION_DISABLE_SOFTCAP",
                            "-DFLASHATTENTION_DISABLE_LOCAL",
                        ]);
                        for include in &fa2_includes {
                            cmd.arg(format!("-I{}", include.display()));
                        }
                    }
                    // Every rerun of this script used to re-run nvcc for all
                    // ~25 translation units, which is about 40 minutes on this
                    // board and is paid for editing one line of Rust. Make the
                    // step incremental the way make is: nvcc writes a
                    // dependency list next to the object, and the object is
                    // reused when it is newer than every file in that list and
                    // the command line that produced it is unchanged. The
                    // cache-hit check runs cheaply on the build thread; the
                    // misses go to the worker pool below.
                    let deps = format!("{out_dir}/{stem}.d");
                    let stamp = format!("{out_dir}/{stem}.cmdline");
                    let cmdline = describe_command(&cmd);
                    if object_is_current(&obj, &deps, &stamp, &cmdline) {
                        continue;
                    }
                    cmd.arg("-MD").arg("-MF").arg(&deps);
                    jobs.push(CompileJob {
                        command: cmd,
                        action: format!("compile {}", entry.display()),
                        stamp: stamp.into(),
                        cmdline,
                    });
                }
                run_parallel(jobs, workers);

                // Create a static library from all kernel objects
                let mut objs: Vec<String> = kernel_files
                    .iter()
                    .map(|e| {
                        let stem = e.file_stem().unwrap().to_string_lossy().to_string();
                        format!("{out_dir}/{stem}.o")
                    })
                    .collect();
                if let Some(bundle) = &aot_bundle {
                    objs.extend(
                        bundle
                            .kernels
                            .values()
                            .map(|kernel| kernel.object.to_string_lossy().into_owned()),
                    );
                }

                let lib_path = format!("{out_dir}/libapxinf_kernels.a");
                // `ar rcs` updates/replaces named members but does not remove
                // objects that disappeared from the source list. Recreate the
                // archive so a renamed adapter (for example static_fp8.cu)
                // cannot leave duplicate, stale C ABI symbols behind.
                if let Err(error) = std::fs::remove_file(&lib_path) {
                    assert_eq!(
                        error.kind(),
                        std::io::ErrorKind::NotFound,
                        "remove stale CUDA archive {lib_path}: {error}"
                    );
                }
                let status = std::process::Command::new("ar")
                    .arg("rcs")
                    .arg(&lib_path)
                    .args(&objs)
                    .status()
                    .expect("failed to run ar");

                assert!(status.success(), "ar failed");

                println!("cargo:rustc-link-search=native={out_dir}");
                println!("cargo:rustc-link-lib=static=apxinf_kernels");
                if let Some(bundle) = &aot_bundle {
                    bundle.link_runtime(std::path::Path::new(&out_dir));
                }
                // Keep CUDA math DSOs after the static archive. GNU ld's
                // --as-needed otherwise discards cublasLt before it sees the
                // static inference archive's references.
                println!("cargo:rustc-link-lib=cublasLt");
                println!("cargo:rustc-link-lib=cublas");
                println!("cargo:rustc-link-lib=cudart");
                println!("cargo:rustc-link-lib=stdc++");
            }
        }

        println!("cargo:rustc-cfg=feature=\"cuda\"");
    } else {
        emit_kernel_build_id(
            &env::var("APXINF_KERNEL_BUILD_ID")
                .unwrap_or_else(|_| format!("no-cuda-{}", env!("CARGO_PKG_VERSION"))),
        );
        println!("cargo:warning=CUDA not found — building without GPU support");
        println!("cargo:rustc-cfg=feature=\"no_cuda\"");
    }
}
