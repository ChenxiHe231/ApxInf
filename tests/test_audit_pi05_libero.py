import json
import pathlib
import subprocess
import sys


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]


def _row(task_id: int, precision: str) -> dict:
    return {
        "status": "completed",
        "suite": "libero_10",
        "task_id": task_id,
        "trial_id": 0,
        "precision": precision,
        "attempt": 1,
        "success": True,
        "action_steps": 5,
        "replans": 1,
        "preprocess_seconds": 0.01,
        "inference_seconds": 0.02,
        "elapsed_seconds": 0.03,
        "first_action_abs_checksum": 1.0,
        "image_input": "openpi_uint8_hwc",
        "seed": 7,
        "max_steps": 520,
        "replan_steps": 5,
    }


def _write_campaign(tmp_path: pathlib.Path, task_count: int, precision: str):
    rows = [_row(task_id, precision) for task_id in range(task_count)]
    results = tmp_path / "results.jsonl"
    results.write_text("".join(json.dumps(row) + "\n" for row in rows))
    per_task = {
        str(task_id): {"completed": 1, "successes": 1, "success_rate": 1.0}
        for task_id in range(task_count)
    }
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "schema": "apxinf.libero-eval.v2",
                "suites": ["libero_10"],
                "precision": precision,
                "rollout_protocol": {
                    "max_steps": 520,
                    "replan_steps": 5,
                    "wait_steps": 10,
                },
                "expected_runs": task_count,
                "completed_runs": task_count,
                "missing_runs": [],
                "successes": task_count,
                "success_rate": 1.0,
                "per_suite": {
                    "libero_10": {
                        "completed": task_count,
                        "successes": task_count,
                        "success_rate": 1.0,
                        "per_task": per_task,
                    }
                },
            }
        )
    )
    return results, summary


def test_campaign_auditor_accepts_current_eval_schema(tmp_path):
    results, summary = _write_campaign(tmp_path, 1, "bf16")
    output = tmp_path / "audit.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/audit_pi05_libero_campaign.py",
            "--results-jsonl",
            str(results),
            "--summary-json",
            str(summary),
            "--output",
            str(output),
            "--precision",
            "bf16",
            "--task-ids",
            "0",
            "--trials-per-task",
            "1",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(output.read_text())["passed"] is True


def test_integrity_auditor_accepts_current_eval_schema(tmp_path):
    results, summary = _write_campaign(tmp_path, 10, "fp8")
    calibration = tmp_path / "calibration.json"
    calibration.write_text(json.dumps({"bootstrap_margin": 2.35}))
    parity = tmp_path / "parity.json"
    parity.write_text(
        json.dumps(
            {
                "schema": "apxinf.pi05.libero-calibration-sweep.v1",
                "min_cosine": 0.997,
                "max_relative_l2": 0.10,
                "results": [
                    {
                        "passed": True,
                        "fixtures": [
                            {"passed": True, "cosine": 0.999, "relative_l2": 0.01}
                            for _ in range(10)
                        ],
                    }
                ],
            }
        )
    )
    output = tmp_path / "integrity-audit.json"
    subprocess.run(
        [
            sys.executable,
            "scripts/audit_pi05_libero_integrity.py",
            "--results-jsonl",
            str(results),
            "--summary-json",
            str(summary),
            "--calibration",
            str(calibration),
            "--parity-calibration",
            str(calibration),
            "--parity-json",
            str(parity),
            "--output",
            str(output),
            "--trials-per-task",
            "1",
            "--require-zero-technical-errors",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(output.read_text())["passed"] is True
