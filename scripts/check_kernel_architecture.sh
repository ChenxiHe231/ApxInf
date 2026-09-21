#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

cuda_root=crates/apxinf-cuda-new

if [[ -e crates/apxinf-cuda ]]; then
    echo 'kernel architecture violation: legacy crates/apxinf-cuda must be removed' >&2
    exit 1
fi

if find "$cuda_root" -type f -name '._*' -print -quit | grep -q .; then
    echo 'kernel architecture violation: AppleDouble ._* files must not exist under src/' >&2
    exit 1
fi

fail_if_present() {
    local pattern="$1"
    local label="$2"
    shift 2
    if search_pattern "$pattern" "$@"; then
        echo "kernel architecture violation: $label" >&2
        exit 1
    fi
}

search_pattern() {
    local pattern="$1"
    shift
    if command -v rg >/dev/null 2>&1; then
        rg -n -g '*.rs' "$pattern" "$@"
        return
    fi

    local paths=()
    local grep_globs=(--include='*.rs')
    while (($#)); do
        if [[ "$1" == '-g' ]]; then
            local glob="$2"
            case "$glob" in
                '*.rs') ;;
                '!**/tests/**') grep_globs+=(--exclude-dir=tests) ;;
                *)
                    echo "kernel architecture check: unsupported grep fallback glob: $glob" >&2
                    exit 2
                    ;;
            esac
            shift 2
        else
            paths+=("$1")
            shift
        fi
    done
    grep -R -n -E "${grep_globs[@]}" "$pattern" "${paths[@]}"
}

search_cuda_pattern() {
    local pattern="$1"
    shift
    if command -v rg >/dev/null 2>&1; then
        rg -n -g '*.cu' -g '*.cuh' "$pattern" "$@"
        return
    fi
    grep -R -n -E --include='*.cu' --include='*.cuh' "$pattern" "$@"
}

fail_if_cuda_present() {
    local pattern="$1"
    local label="$2"
    shift 2
    if search_cuda_pattern "$pattern" "$@"; then
        echo "kernel architecture violation: $label" >&2
        exit 1
    fi
}

if [[ -d "$cuda_root/src/launch" ]] &&
    find "$cuda_root/src/launch" -type f -name '*.rs' -print -quit | grep -q .; then
    echo "kernel architecture violation: src/launch must not contain Rust modules" >&2
    exit 1
fi

for legacy_module in bf16 fp8 w8a8 decode; do
    if [[ -e "$cuda_root/src/ops/${legacy_module}.rs" ]]; then
        echo "kernel architecture violation: public kernels/${legacy_module}.rs uses a precision/stage classification" >&2
        exit 1
    fi
done

if [[ -d "$cuda_root/src/native" ]]; then
    echo 'kernel architecture violation: Rust src/native forwarding layer must not exist' >&2
    exit 1
fi
fail_if_present 'crate::native|native_contracts!' \
    'removed Rust native forwarding layer is still referenced' \
    "$cuda_root/src" -g '*.rs'
fail_if_present '^[[:space:]]*pub[[:space:]]+(unsafe[[:space:]]+fn|fn[^;]*(\*const|\*mut)[[:space:]])' \
    'safe public kernel contracts must not expose unsafe functions or raw pointers' \
    "$cuda_root/src/ops" -g '*.rs'
fail_if_present 'std::env::|env::var(_os)?' \
    'kernel execution paths must not read environment variables' \
    "$cuda_root/src/ops" -g '*.rs' -g '!**/tests/**'
fail_if_present 'crate::launch|launch::|mod launch' \
    'removed launch layer is still referenced' \
    "$cuda_root/src" crates/apxinf-model/src/pi05 -g '*.rs'
fail_if_present 'kernels::(bf16|fp8|w8a8|decode)(::|\b)' \
    'callers must use physical operator modules, not precision/stage modules' \
    "$cuda_root/src" crates/apxinf-model/src -g '*.rs'

if find "$cuda_root/native/kernels" -maxdepth 1 -type f -name '*.cu' -print -quit |
    grep -q .; then
    echo 'kernel architecture violation: kernels root must not contain host adapter .cu files' >&2
    exit 1
fi
fail_if_cuda_present 'extern[[:space:]]+"C"' \
    'pure CUDA/CUTLASS operator sources must not export C ABI symbols' \
    "$cuda_root/native/kernels"
fail_if_cuda_present 'cublasLt' \
    'cuBLASLt vendor planning belongs under adapters/, not kernels/' \
    "$cuda_root/native/kernels"

echo 'kernel architecture checks passed'
