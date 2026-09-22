import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from examples.g1_dex3_training import train_full_queue as queue


def _config(root: Path, policy: str = "act", space: str = "joint28", width: int = 28):
    path = root / "configs" / f"{policy}_{space}_full.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "steps": 40000,
                "output_dir": str(root / "runs" / f"{policy}_{space}_full"),
                "dataset": {
                    "episodes": None,
                    "exclude_episodes": None,
                    "eval_split": 0.0,
                    "root": str(root / "datasets" / space),
                },
                "policy": {
                    "type": policy,
                    "input_features": {"observation.state": {"shape": [28]}},
                    "output_features": {"action": {"shape": [width]}},
                },
            }
        )
    )


def test_config_guard_rejects_episodes(tmp_path):
    _config(tmp_path)
    path = tmp_path / "configs/act_joint28_full.json"
    data = json.loads(path.read_text())
    data["dataset"]["episodes"] = [0]
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="episodes"):
        queue._config_guard(tmp_path, "act", "joint28", 28)


@pytest.mark.parametrize("matching", [False, True])
def test_completed_requires_matching_weights_digest(tmp_path, matching):
    run = tmp_path / "runs/act_joint28_full"
    model = run / "checkpoints/040000/pretrained_model"
    model.mkdir(parents=True)
    weights = model / "model.safetensors"
    weights.write_bytes(b"weights")
    report = {
        "verified": True,
        "final_step": 40000,
        "checkpoint_path": str(model.parent),
        "weights_sha256": hashlib.sha256(b"weights" if matching else b"wrong").hexdigest(),
    }
    (run / "final_model_validation.json").write_text(json.dumps(report))
    if matching:
        assert queue._completed(tmp_path, "act_joint28_full")
    else:
        with pytest.raises(ValueError, match="incomplete"):
            queue._completed(tmp_path, "act_joint28_full")


def test_run_stops_after_training_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(queue, "_smoke_guard", lambda *args: None)
    monkeypatch.setattr(queue, "_config_guard", lambda root, policy, space, width: root / "config.json")
    monkeypatch.setattr(queue, "_completed", lambda *args: False)
    monkeypatch.setattr(queue.shutil, "disk_usage", lambda _: SimpleNamespace(free=queue.MIN_FREE))
    calls = []

    def fake_run(command, log):
        calls.append(command)
        log.parent.mkdir(parents=True, exist_ok=True)
        return 7

    monkeypatch.setattr(queue, "_run", fake_run)
    assert queue.run_queue(tmp_path) == 7
    assert len(calls) == 1
    assert (tmp_path / "logs/act_joint28_full.exit").read_text() == "7\n"


def test_all_guards_run_before_any_subprocess(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(queue, "_smoke_guard", lambda root, name, width: seen.append(("smoke", name)))
    monkeypatch.setattr(queue, "_config_guard", lambda root, policy, space, width: root / "config.json")
    monkeypatch.setattr(queue, "_completed", lambda *args: False)
    monkeypatch.setattr(queue.shutil, "disk_usage", lambda _: SimpleNamespace(free=queue.MIN_FREE))
    monkeypatch.setattr(queue, "_run", lambda *args: pytest.fail("subprocess started"))
    with pytest.raises(pytest.fail.Exception):
        queue.run_queue(tmp_path)
    assert len(seen) == 14


def test_successfully_completed_run_is_skipped(tmp_path, monkeypatch):
    monkeypatch.setattr(queue, "prune_superseded_smoke", lambda *args: None)
    monkeypatch.setattr(queue, "_smoke_guard", lambda *args: None)
    monkeypatch.setattr(queue, "_config_guard", lambda root, policy, space, width: root / "config.json")
    monkeypatch.setattr(queue, "_completed", lambda *args: True)
    monkeypatch.setattr(queue, "_run", lambda *args: pytest.fail("completed run was started"))
    assert queue.run_queue(tmp_path) == 0


def test_verification_failure_stops_before_next_training(tmp_path, monkeypatch):
    monkeypatch.setattr(queue, "_smoke_guard", lambda *args: None)
    monkeypatch.setattr(queue, "_config_guard", lambda root, policy, space, width: root / "config.json")
    monkeypatch.setattr(queue, "_completed", lambda *args: False)
    monkeypatch.setattr(queue.shutil, "disk_usage", lambda _: SimpleNamespace(free=queue.MIN_FREE))
    commands = []

    def fake_run(command, log):
        log.parent.mkdir(parents=True, exist_ok=True)
        commands.append(command)
        return 0 if len(commands) == 1 else 9

    monkeypatch.setattr(queue, "_run", fake_run)
    assert queue.run_queue(tmp_path) == 9
    assert len(commands) == 2
    assert "lerobot.scripts.lerobot_train" in commands[0]
    assert "examples.g1_dex3_training.finalize_baseline" in commands[1]
    assert (tmp_path / "logs/act_joint28_full.exit").read_text() == "0\n"
    assert (tmp_path / "logs/act_joint28_full-verify.exit").read_text() == "9\n"
