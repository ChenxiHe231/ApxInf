from __future__ import annotations


class _CapturedPolicy:
    metadata = {}


def test_franka_libero_leaves_checkpoint_state_semantics_unchanged(monkeypatch):
    from apxinf import build_robot_policy
    from apxinf.robots import presets

    captured = {}

    def fake_load(model_dir, **kwargs):
        captured["model_dir"] = model_dir
        captured.update(kwargs)
        return _CapturedPolicy()

    monkeypatch.setattr(presets.AutoPolicy, "from_pretrained", fake_load)

    build_robot_policy("franka_libero", "/checkpoint")

    assert captured["model_dir"] == "/checkpoint"
    assert captured["image_keys"] == (
        "observation/image",
        "observation/wrist_image",
    )
    assert captured["state_key"] == "observation/state"
    assert captured["prompt_key"] == "prompt"
    assert captured["action_dim"] == 7
    assert "discrete_state" not in captured


def test_robot_policy_user_overrides_take_precedence(monkeypatch):
    from apxinf import build_robot_policy
    from apxinf.robots import presets

    captured = {}

    def fake_load(model_dir, **kwargs):
        captured.update(kwargs)
        return _CapturedPolicy()

    monkeypatch.setattr(presets.AutoPolicy, "from_pretrained", fake_load)

    build_robot_policy(
        "franka_libero",
        "/checkpoint",
        image_keys=("front", "hand"),
        state_key="state",
        prompt_key="instruction",
        action_dim=9,
        discrete_state=True,
    )

    assert captured["image_keys"] == ("front", "hand")
    assert captured["state_key"] == "state"
    assert captured["prompt_key"] == "instruction"
    assert captured["action_dim"] == 9
    assert captured["discrete_state"] is True
