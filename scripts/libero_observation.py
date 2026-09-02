"""Translate native LIBERO simulator observations into ApxInf Observations."""

from __future__ import annotations

import math
import pathlib

import numpy as np
from PIL import Image


IMAGE_SIZE = 224


def quat_to_axis_angle(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64).copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    denominator = math.sqrt(max(0.0, 1.0 - quat[3] * quat[3]))
    if math.isclose(denominator, 0.0):
        return np.zeros(3, dtype=np.float32)
    return (quat[:3] * 2.0 * math.acos(quat[3]) / denominator).astype(np.float32)


def resize_images(base: np.ndarray, wrist: np.ndarray) -> np.ndarray:
    """Apply the OpenPI LIBERO camera orientation and resize convention."""

    def resize_with_pad(image: np.ndarray) -> np.ndarray:
        image = np.ascontiguousarray(image[::-1, ::-1])
        height, width = image.shape[:2]
        ratio = max(width / IMAGE_SIZE, height / IMAGE_SIZE)
        resized_height = int(height / ratio)
        resized_width = int(width / ratio)
        resized = np.asarray(
            Image.fromarray(image).resize(
                (resized_width, resized_height), resample=Image.Resampling.BILINEAR
            )
        )
        canvas = np.zeros((IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
        offset_y = (IMAGE_SIZE - resized_height) // 2
        offset_x = (IMAGE_SIZE - resized_width) // 2
        canvas[
            offset_y : offset_y + resized_height,
            offset_x : offset_x + resized_width,
        ] = resized
        return canvas

    return np.stack([resize_with_pad(base), resize_with_pad(wrist)], axis=0)


def make_env(task, seed: int):
    """Build the same off-screen LIBERO environment used by evaluation."""
    try:
        from libero.libero import get_libero_path
        from libero.libero.envs import OffScreenRenderEnv
    except ImportError as error:
        raise ImportError(
            "native LIBERO observations require the LIBERO and MuJoCo evaluation "
            "dependencies; install LIBERO as described in README.md"
        ) from error

    bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=str(bddl),
        camera_heights=256,
        camera_widths=256,
    )
    env.seed(seed)
    return env


def to_apxinf_observation(
    observation,
    *,
    prompt: str,
    image_keys: tuple[str, str],
    prompt_key: str,
    state_key: str,
) -> dict:
    """Convert one raw simulator frame using the evaluation-time convention."""
    images = resize_images(
        observation["agentview_image"],
        observation["robot0_eye_in_hand_image"],
    )
    state = np.concatenate(
        (
            observation["robot0_eef_pos"],
            quat_to_axis_angle(observation["robot0_eef_quat"]),
            observation["robot0_gripper_qpos"],
        )
    ).astype(np.float32, copy=False)
    return {
        image_keys[0]: images[0],
        image_keys[1]: images[1],
        state_key: state,
        prompt_key: prompt,
    }
