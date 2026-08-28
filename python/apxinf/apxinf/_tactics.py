"""PI0.5 tactic routing for source-checkout Python deployments."""

from __future__ import annotations

import ctypes
from pathlib import Path


_DEFAULT_TACTICS = {
    (87, "bf16"): "orin_sm87_bf16_v2_v3_h10_tactics.json",
    (89, "bf16"): "rtx4090_sm89_bf16_v2_v3_h10_tactics.json",
    (101, "fp8"): "thor_u_cutlass_tactics.json",
    (110, "bf16"): "thor_sm110_bf16_v2_v3_h10_tactics.json",
    (110, "fp8"): "thor_sm110_fp8_native_v2_v3_h10_tactics.json",
}
_SOURCE_ROOT = Path(__file__).resolve().parents[3]


def cuda_sm(device: str) -> int | None:
    """Return CUDA's integer compute capability for ``cuda:N`` (e.g. 110)."""
    if device == "cuda":
        device_index = 0
    elif device.startswith("cuda:"):
        try:
            device_index = int(device.removeprefix("cuda:"))
        except ValueError as error:
            raise ValueError(f"invalid CUDA device {device!r}; expected cuda:N") from error
    else:
        return None

    try:
        cudart = ctypes.CDLL("libcudart.so")
    except OSError as error:
        raise RuntimeError(f"cannot query {device}: failed to load CUDA runtime: {error}") from error

    # cudaDevAttrComputeCapabilityMajor/Minor from cuda_runtime_api.h.
    def attribute(code: int) -> int:
        value = ctypes.c_int()
        status = cudart.cudaDeviceGetAttribute(ctypes.byref(value), code, device_index)
        if status != 0:
            raise RuntimeError(
                f"cannot query {device} compute capability: cudaDeviceGetAttribute "
                f"returned {status}"
            )
        return value.value

    return attribute(75) * 10 + attribute(76)


def resolve_pi05_tactics(
    device: str,
    precision: str,
    *,
    model_dir: Path | None = None,
    override: Path | None = None,
) -> Path | None:
    """Resolve explicit, checkpoint-local, then source-tree PI0.5 tactics."""
    if override is not None:
        return Path(override)
    if model_dir is not None:
        checkpoint_tactics = Path(model_dir) / "tactics.json"
        if checkpoint_tactics.is_file():
            return checkpoint_tactics
    sm = cuda_sm(device)
    filename = _DEFAULT_TACTICS.get((sm, precision))
    if filename is None:
        return None
    path = _SOURCE_ROOT / "configs" / "pi05" / filename
    if not path.is_file():
        raise FileNotFoundError(f"default tactics for SM{sm} {precision} are missing: {path}")
    return path
