"""WallOSS policy: raw robot observations to deployable action chunks.

WallOSS uses Qwen2.5-VL preprocessing, which is intentionally owned by this
model module.  The Rust runtime receives only the canonical patch/token/mask
contract; no PI0.5 processor or model implementation is imported here.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import numpy as np
from PIL import Image

from ..registry import register_policy

__all__ = ["WallossPolicy"]

_ROLE_START = "<|im_start|>"
_ROLE_END = "<|im_end|>"
_VISION_START = "<|vision_start|>"
_VISION_END = "<|vision_end|>"
_IMAGE_PAD = "<|image_pad|>"
_PROPRI = "<|propri|>"
_ACTION = "<|action|>"
_CAMERA_LABELS = {"face_view": "front view", "right_wrist_view": "right wrist view"}
_DEFAULT_IMAGE_KEYS = ("observation/image", "observation/wrist_image")


def _lookup(data: Mapping[str, Any], path: str) -> Any:
    if path in data:
        return data[path]
    value: Any = data
    for part in path.split("/"):
        if not isinstance(value, Mapping) or part not in value:
            raise KeyError(path)
        value = value[part]
    return value


def _round_by_factor(value: int, factor: int) -> int:
    return max(factor, round(value / factor) * factor)


def _smart_resize(height: int, width: int, factor: int, min_pixels: int, max_pixels: int):
    new_h = _round_by_factor(height, factor)
    new_w = _round_by_factor(width, factor)
    if new_h * new_w > max_pixels:
        beta = (height * width / max_pixels) ** 0.5
        new_h = max(factor, int(height / beta // factor) * factor)
        new_w = max(factor, int(width / beta // factor) * factor)
    elif new_h * new_w < min_pixels:
        beta = (min_pixels / (height * width)) ** 0.5
        new_h = (int(height * beta) + factor - 1) // factor * factor
        new_w = (int(width * beta) + factor - 1) // factor * factor
    return new_h, new_w


def _load_normalizer(path: Path, norm_key: str) -> tuple[np.ndarray, np.ndarray]:
    try:
        import torch
    except ImportError as error:  # pragma: no cover - dependency error is user-facing
        raise ImportError("WallossPolicy requires the 'walloss' extra (torch + transformers)") from error
    state = torch.load(path, map_location="cpu", weights_only=True)
    available = sorted(key[4:] for key in state if key.startswith("min."))
    resolved = norm_key
    if resolved not in available:
        matches = [key for key in available if key.startswith(f"{norm_key}_")]
        if len(matches) == 1:
            resolved = matches[0]
        else:
            raise KeyError(f"norm_key={norm_key!r} not in {path.name}; available={available}")
    minimum = state[f"min.{resolved}"].detach().cpu().float().numpy()
    delta = state[f"delta.{resolved}"].detach().cpu().float().numpy()
    return np.asarray(minimum, np.float32), np.asarray(delta, np.float32)


def _checkpoint_state_bins(model_dir: Path) -> int:
    """Resolve state-token bins from checkpoint metadata, with legacy fallback."""
    documents = []
    config_json = model_dir / "config.json"
    if config_json.is_file():
        documents.append((config_json, json.loads(config_json.read_text())))
    config_yaml = None
    for name in ("config.yml", "config.yaml"):
        candidate = model_dir / name
        if candidate.is_file():
            config_yaml = candidate
            break
    if config_yaml is not None:
        try:
            import yaml
        except ImportError as error:  # pragma: no cover - declared by the walloss extra
            raise ImportError("WallossPolicy requires PyYAML to read config.yml") from error
        documents.append((config_yaml, yaml.safe_load(config_yaml.read_text()) or {}))

    for path, document in documents:
        if not isinstance(document, Mapping):
            raise ValueError(f"{path.name} must contain a mapping")
        value = document.get("state_bins")
        data = document.get("data")
        if value is None and isinstance(data, Mapping):
            value = data.get("state_bins")
        if value is not None:
            bins = int(value)
            if bins < 2:
                raise ValueError(f"{path.name} state_bins must be >= 2, got {bins}")
            return bins
    return 256


class _WallossProcessor:
    def __init__(self, model_dir: Path, *, image_keys: Sequence[str], camera_names: Sequence[str],
                 state_key: str, prompt_key: str, action_horizon: int, action_dim: int,
                 norm_key: str, state_bins: int):
        try:
            from transformers import AutoImageProcessor, AutoTokenizer
        except ImportError as error:  # pragma: no cover
            raise ImportError("WallossPolicy requires the 'walloss' extra (torch + transformers)") from error

        self.image_keys = tuple(image_keys)
        self.camera_names = tuple(camera_names)
        self.state_key = str(state_key)
        self.prompt_key = str(prompt_key)
        self.action_horizon = int(action_horizon)
        self.action_dim = int(action_dim)
        self.state_bins = int(state_bins)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.image_processor = AutoImageProcessor.from_pretrained(model_dir, use_fast=False)
        self.tokenizer.add_tokens([_PROPRI, _ACTION])
        self.action_token_id = int(self.tokenizer.convert_tokens_to_ids(_ACTION))
        self.image_pad_token_id = int(self.tokenizer.convert_tokens_to_ids(_IMAGE_PAD))
        self.propri_min, self.propri_delta = _load_normalizer(
            model_dir / "normalizer_propri.pth", norm_key
        )
        self.factor = int(self.image_processor.patch_size) * int(self.image_processor.merge_size)
        self.min_pixels = int(getattr(self.image_processor, "min_pixels", 56 * 56))
        self.max_pixels = int(getattr(self.image_processor, "max_pixels", 14 * 14 * 4 * 1280))

    def _prompt(self, instruction: str, state: np.ndarray, active: np.ndarray) -> str:
        normalized = np.clip((state - self.propri_min) / self.propri_delta * 2 - 1, -1, 1)
        # Keep numpy's float64 linspace: this mirrors the training/reference
        # processor exactly at bin boundaries.
        edges = np.linspace(-1.0, 1.0, self.state_bins + 1)[:-1]
        bins = np.clip(np.digitize(normalized, edges) - 1, 0, self.state_bins - 1)
        state_text = " ".join(str(int(value)) for value in bins[active])
        cameras = "".join(
            f" {_CAMERA_LABELS.get(name, name.replace('_', ' '))}: "
            f"{_VISION_START}{_IMAGE_PAD}{_VISION_END}"
            for name in self.camera_names
        )
        return (
            f"{_ROLE_START}system\nYou are a helpful assistant.{_ROLE_END}\n"
            f"{_ROLE_START}user\nObservation:{cameras}\nInstruction: {instruction}"
            f"\nPredict the next action in robot action.\nProprioception: {state_text}\n"
            f"{_ROLE_END}\n{_ROLE_START}assistant\n" + _ACTION * self.action_horizon
        )

    def __call__(self, observation: Mapping[str, Any]):
        raw_state = np.asarray(_lookup(observation, self.state_key), dtype=np.float32).reshape(-1)
        if raw_state.size > self.action_dim:
            raise ValueError(f"state has {raw_state.size} values, maximum is {self.action_dim}")
        state = np.zeros(self.action_dim, dtype=np.float32)
        state[: raw_state.size] = raw_state
        agent_pos_mask = observation.get("agent_pos_mask")
        if agent_pos_mask is None:
            active = np.zeros(self.action_dim, dtype=bool)
            active[: raw_state.size] = True
        else:
            active = np.asarray(agent_pos_mask, dtype=np.float32).reshape(-1).astype(bool)
            if active.size != self.action_dim:
                raise ValueError(
                    f"agent_pos_mask has {active.size} values, expected {self.action_dim}"
                )
        prompt = _lookup(observation, self.prompt_key)
        if not isinstance(prompt, str):
            raise TypeError("prompt must be a string")

        images = []
        for key in self.image_keys:
            image = np.asarray(_lookup(observation, key))
            if image.ndim != 3 or image.shape[-1] != 3 or image.dtype != np.uint8:
                raise ValueError(f"{key} must be HxWx3 uint8 RGB, got {image.shape} {image.dtype}")
            new_h, new_w = _smart_resize(
                image.shape[0], image.shape[1], self.factor, self.min_pixels, self.max_pixels
            )
            images.append(np.asarray(Image.fromarray(image).resize((new_w, new_h))))

        processed = self.image_processor(images=images, return_tensors="pt")
        grids = processed["image_grid_thw"].detach().cpu().numpy()
        if grids.shape != (2, 3) or not np.array_equal(grids, np.array([[1, 18, 18]] * 2)):
            raise ValueError(
                f"WallOSS runtime currently requires two 18x18 image grids; got {grids.tolist()} "
                "(256x256 input images smart-resize to the required 252x252)"
            )
        patches = np.ascontiguousarray(
            processed["pixel_values"].detach().cpu().float().numpy(), dtype=np.float32
        )

        ids = self.tokenizer(
            self._prompt(prompt, state, active), add_special_tokens=False, return_attention_mask=False
        )["input_ids"]
        expanded = []
        image_index = 0
        merge_sq = int(self.image_processor.merge_size) ** 2
        for token in ids:
            if token == self.image_pad_token_id:
                count = int(np.prod(grids[image_index]) // merge_sq)
                expanded.extend([token] * count)
                image_index += 1
            else:
                expanded.append(token)
        if image_index != len(images):
            raise ValueError(f"prompt contained {image_index} image placeholders for {len(images)} images")

        dof = observation.get("dof_mask")
        if dof is None:
            dof = active.astype(np.float32)
        dof = np.asarray(dof, dtype=np.float32)
        if dof.shape == (self.action_dim,):
            dof = np.broadcast_to(dof, (self.action_horizon, self.action_dim))
        if dof.shape != (self.action_horizon, self.action_dim):
            raise ValueError(
                f"dof_mask shape {dof.shape}, expected [{self.action_dim}] or "
                f"[{self.action_horizon}, {self.action_dim}]"
            )
        action_mask = np.ascontiguousarray(dof, dtype=np.float32)
        return patches, np.ascontiguousarray(expanded, dtype=np.uint32), action_mask


@register_policy("walloss")
@register_policy("wall_oss_05")
class WallossPolicy:
    """Public WallOSS L2 policy. The websocket server consumes it unchanged."""

    def __init__(self, model, processor: _WallossProcessor, *, action_min, action_delta,
                 action_dim: int, metadata: Optional[Mapping[str, Any]] = None):
        self.model = model
        self.processor = processor
        self._action_min = np.asarray(action_min, np.float32)[:action_dim]
        self._action_delta = np.asarray(action_delta, np.float32)[:action_dim]
        self._action_dim = int(action_dim)
        self.metadata = {
            "model_type": "walloss", "action_horizon": model.action_horizon,
            "action_dim": self._action_dim, "model_action_dim": model.action_dim,
            "num_views": model.num_views, "image_keys": list(processor.image_keys),
            "state_key": processor.state_key, "prompt_key": processor.prompt_key,
            "dof_mask_key": "dof_mask",
            "discrete_state": True,
            "state_bins": processor.state_bins,
            **(dict(metadata) if metadata else {}),
        }

    @classmethod
    def from_pretrained(cls, model_dir, *, model=None, checkpoint=None, device="cuda:0",
                        precision="auto", tactics=None, norm_key="x2_normal",
                        action_dim=None, image_keys=_DEFAULT_IMAGE_KEYS,
                        camera_names=("face_view", "right_wrist_view"),
                        state_key="observation/state", prompt_key="prompt",
                        discrete_state=None, state_bins=None, seed=0,
                        metadata=None, **kwargs):
        if kwargs:
            raise TypeError(f"unsupported WallossPolicy options: {sorted(kwargs)}")
        if discrete_state is False:
            raise ValueError(
                "WallOSS checkpoints require discretized state in the prompt; "
                "discrete_state=False is unsupported"
            )
        model_dir = Path(model_dir)
        if len(image_keys) != 2 or len(camera_names) != 2:
            raise ValueError("WallOSS runtime currently requires exactly two camera views")
        resolved_state_bins = (
            int(state_bins) if state_bins is not None else _checkpoint_state_bins(model_dir)
        )
        if resolved_state_bins < 2:
            raise ValueError(f"state_bins must be >= 2, got {resolved_state_bins}")
        if model is None:
            import apxinf_py
            ckpt = str(checkpoint) if checkpoint is not None else str(model_dir / "model.safetensors")
            model = apxinf_py.Model.load(
                "walloss", ckpt, device=device, precision=precision,
                **({"tactics": str(tactics)} if tactics else {}), sampling_seed=int(seed),
            )
        width = int(action_dim) if action_dim is not None else int(model.action_dim)
        if width < 1 or width > int(model.action_dim):
            raise ValueError(f"action_dim must be in 1..={model.action_dim}, got {width}")
        action_min, action_delta = _load_normalizer(model_dir / "normalizer_action.pth", norm_key)
        if action_min.size < width or action_delta.size < width:
            raise ValueError(
                f"normalizer {norm_key!r} has width {action_min.size}, requested {width}"
            )
        processor = _WallossProcessor(
            model_dir, image_keys=image_keys, camera_names=camera_names,
            state_key=state_key, prompt_key=prompt_key,
            action_horizon=model.action_horizon, action_dim=model.action_dim, norm_key=norm_key,
            state_bins=resolved_state_bins,
        )
        reset = getattr(model, "reset_sampling", None)
        if callable(reset):
            reset(int(seed))
        return cls(model, processor, action_min=action_min, action_delta=action_delta,
                   action_dim=width, metadata=metadata)

    def infer(self, observation: Mapping[str, Any], *, noise=None) -> dict:
        started = time.perf_counter()
        patches, token_ids, action_mask = self.processor(observation)
        model_started = time.perf_counter()
        normalized = np.asarray(
            self.model._infer_patches(
                patches, token_ids,
                **({"noise": np.ascontiguousarray(noise, dtype=np.float32)} if noise is not None else {}),
                action_mask=action_mask,
            ), dtype=np.float32,
        )
        model_ms = (time.perf_counter() - model_started) * 1000
        actions = (normalized[:, : self._action_dim] + 1) * 0.5
        actions = np.ascontiguousarray(actions * self._action_delta + self._action_min, np.float32)
        return {"actions": actions, "normalized_actions": normalized, "token_ids": token_ids,
                "noise": noise, "timing": {"model_ms": model_ms,
                "total_ms": (time.perf_counter() - started) * 1000}}

    @property
    def action_dim(self):
        return self._action_dim

    @property
    def action_horizon(self):
        return int(self.model.action_horizon)

    def close(self):
        close = getattr(self.model, "close", None)
        if callable(close):
            close()
