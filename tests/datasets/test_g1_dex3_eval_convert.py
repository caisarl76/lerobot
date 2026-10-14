"""eval_session_to_lerobot.py: a --trials session becomes a LeRobot v3.0 dataset with meta/eval_session.json."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
sys.path.insert(0, str(Path(__file__).parents[2] / "examples" / "g1_dex3_training"))
import eval_session  # noqa: E402
from eval_session_to_lerobot import convert, frame_rows  # noqa: E402

from lerobot.datasets.lerobot_dataset import LeRobotDataset  # noqa: E402

ARGS = {"task": "pick the bottle", "action_space": "sonic78", "policy_server": "tcp://ws:5560",
        "failure_reasons": None, "trials": 3}  # fmt: skip


def trial(session, i, t0, n_episode, verdict, reason="", args=ARGS, cam=lambda t: True, model=""):
    """10 handoff ticks, n_episode episode ticks, 10 blend-out ticks at 50 Hz; camera at 30 Hz from t0 - 0.5 s.
    state[:, 0] and token[:, 0] hold the tick's wall time; each JPEG is a flat grey of value 10 * (index % 20)."""
    n = 20 + n_episode
    wall = t0 - 0.2 + 0.02 * np.arange(n)
    phase = ["handoff blend"] * 10 + ["episode"] * n_episode + ["blend out"] * 10
    log = {
        "wall": list(wall),
        "phase": phase,
        "state": [np.full(28, w, np.float32) for w in wall],
        "token": [np.full(64, w, np.float32) for w in wall],
        "hands": [np.zeros(14, np.float32) for _ in wall],
        "joint_ref": [np.full(31, np.nan, np.float32) for _ in wall],
    }
    t_cam = t0 - 0.5 + np.arange(int((n * 0.02 + 0.5) * 30)) / 30
    frames = [
        (t, "egocentric", cv2.imencode(".jpg", np.full((48, 64, 3), 10 * (k % 20), np.uint8))[1].tobytes())
        for k, t in enumerate(t_cam)
        if cam(t - t0)
    ]
    extra = {
        "episode_end": "operator stop",
        "args": json.dumps(args),
        "verdict": verdict,
        "reason": reason,
        "note": "",
        "model": model,
    }
    eval_session.write_trial(session, i, log, extra, frames)
    return wall, t_cam


def test_frame_rows():
    times = np.array([1.0, 2.0, 3.0])
    assert list(frame_rows(times, np.array([0.5, 1.0, 2.5, 9.0]))) == [0, 0, 1, 2]


def test_convert_two_episodes_and_a_skipped_trial(tmp_path):
    session = tmp_path / "raw"
    wall0, cam0 = trial(session, 0, 1000.0, 60, "success")
    trial(session, 1, 2000.0, 0, "fail", "stalled / froze")  # never reached the episode phase
    wall2, _ = trial(session, 2, 3000.0, 45, "fail", "grasp failed / missed")
    meta = convert(session, tmp_path / "dataset", "local/test_eval", ["reach", "grasp"], fps=30)
    assert [(s["trial"], s["verdict"], s["reason"], s["why"]) for s in meta["skipped"]] == [
        (1, "fail", "stalled / froze", "no episode ticks")
    ]
    assert [(e["trial"], e["verdict"]) for e in meta["episodes"].values()] == [(0, "success"), (2, "fail")]
    assert meta["subtasks"] == ["reach", "grasp"] and meta["model"] == "tcp://ws:5560"
    assert json.loads((tmp_path / "dataset/meta/eval_session.json").read_text()) == meta

    ds = LeRobotDataset("local/test_eval", root=tmp_path / "dataset")
    assert ds.meta.total_episodes == 2 and ds.fps == 30
    span0 = wall0[10 + 59] - wall0[10]
    assert ds.meta.episodes[0]["length"] == int(np.floor(span0 * 30)) + 1
    item = ds[5]  # episode 0, frame 5
    t = wall0[10] + 5 / 30
    expected_tick = wall0[frame_rows(wall0, np.array([t]))[0]]
    assert item["observation.state"][0].item() == pytest.approx(expected_tick, abs=1e-3)
    assert item["action"].shape == (78,) and item["action"][0].item() == pytest.approx(
        expected_tick, abs=1e-3
    )
    k = frame_rows(cam0, np.array([t]))[0]
    grey = item["observation.images.egocentric"].float().mean().item() * 255
    assert grey == pytest.approx(10 * (k % 20), abs=8)  # flat grey through the video codec
    assert item["task"] == "pick the bottle"


def test_skipped_no_camera_trial_keeps_its_verdict(tmp_path, capsys):
    session = tmp_path / "raw"
    trial(session, 0, 1000.0, 30, "success")
    trial(session, 1, 2000.0, 30, "success")
    trial(session, 2, 3000.0, 30, "fail", "dropped the object", cam=lambda t: False)
    meta = convert(session, tmp_path / "dataset", "local/x", ["reach"])
    line = json.loads((session / "results.jsonl").read_text().splitlines()[2])
    assert meta["skipped"] == [{**line, "why": "no camera frames"}]
    assert "success 2/3" in capsys.readouterr().out


def test_mixed_session_settings_are_rejected(tmp_path):
    session = tmp_path / "raw"
    trial(session, 0, 1000.0, 30, "success")
    laptop = {**ARGS, "task": "close the laptop", "action_space": "joint31"}
    trial(session, 1, 2000.0, 30, "success", args=laptop)
    with pytest.raises(ValueError, match="trial 1: task, action_space"):
        convert(session, tmp_path / "dataset", "local/x", ["reach"])


def test_each_episode_records_its_model_and_a_mixed_model_session_is_rejected(tmp_path):
    session = tmp_path / "raw"
    trial(session, 0, 1000.0, 30, "success", model="/m/he_full/pretrained_model")
    trial(session, 1, 2000.0, 30, "fail", "dropped the object", model="/m/he_full/pretrained_model")
    meta = convert(session, tmp_path / "dataset", "local/x", ["reach"])
    assert meta["model"] == "/m/he_full/pretrained_model"
    assert [e["model"] for e in meta["episodes"].values()] == ["/m/he_full/pretrained_model"] * 2
    trial(session, 2, 3000.0, 30, "success", model="/m/base_full/pretrained_model")
    with pytest.raises(ValueError, match="different models"):
        convert(session, tmp_path / "dataset2", "local/x", ["reach"])


def test_camera_gaps_at_episode_boundaries_warn(tmp_path, capsys):
    session = tmp_path / "raw"
    trial(session, 0, 1000.0, 30, "success", cam=lambda t: t < -0.1)  # frozen before wall_start
    trial(session, 1, 2000.0, 30, "success", cam=lambda t: t > 0.4)  # starts late
    trial(session, 2, 3000.0, 30, "success", cam=lambda t: t < 0.3)  # stops early
    convert(session, tmp_path / "dataset", "local/x", ["reach"])
    out = capsys.readouterr().out
    assert "trial 0 egocentric: camera gap" in out and "at -0.1" in out  # the shown JPEG's age counts
    assert "trial 1 egocentric: camera gap" in out and "at 0.00 s" in out
    assert "trial 2 egocentric: camera gap" in out


def test_refuses_existing_output_and_missing_frames(tmp_path):
    session = tmp_path / "raw"
    trial(session, 0, 1000.0, 30, "success")
    (tmp_path / "dataset").mkdir()
    with pytest.raises(FileExistsError):
        convert(session, tmp_path / "dataset", "local/x", ["reach"])
    (session / "trial_000_frames.npz").unlink()
    with pytest.raises(FileNotFoundError, match="trial_000_frames.npz"):
        convert(session, tmp_path / "other", "local/x", ["reach"])


def test_overwrite_never_deletes_the_raw_session(tmp_path):
    session = tmp_path / "raw" / "s01"
    trial(session, 0, 1000.0, 30, "success")
    script = Path(__file__).parents[2] / "examples" / "g1_dex3_training" / "eval_session_to_lerobot.py"
    r = subprocess.run(
        [sys.executable, str(script), "--session-dir", str(session), "--out", str(tmp_path / "raw"),
         "--subtasks", "reach", "--overwrite"],
        capture_output=True, text=True,
    )  # fmt: skip
    assert r.returncode != 0 and "refusing" in r.stderr
    assert (session / "results.jsonl").exists()
