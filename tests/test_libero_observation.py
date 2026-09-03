import numpy as np

from scripts.libero_observation import libero_images, libero_state


def test_libero_images_only_reorients_raw_frames():
    base = np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3)
    wrist = base + 1

    images = libero_images(base, wrist)

    assert images.shape == (2, 4, 5, 3)
    np.testing.assert_array_equal(images[0], base[::-1, ::-1])
    np.testing.assert_array_equal(images[1], wrist[::-1, ::-1])


def test_libero_state_collapses_mirrored_gripper_joints():
    observation = {
        "robot0_eef_pos": np.array([0.1, 0.2, 0.3]),
        "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0]),
        "robot0_gripper_qpos": np.array([0.04, -0.04]),
    }

    state = libero_state(observation)

    np.testing.assert_array_equal(
        state,
        np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.04], dtype=np.float32),
    )
