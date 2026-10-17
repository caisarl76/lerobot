"""eval_session.py: the operator verdict per trial and the session files of sonic_policy_streamer.py --trials."""

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / "examples" / "g1_dex3_training"))
import eval_session as es


def answers(*replies):
    it = iter(replies)
    return lambda prompt="": next(it)


def test_parse_reasons():
    assert es.parse_reasons(None) == list(es.DEFAULT_FAILURE_REASONS)
    assert es.parse_reasons(" a , b,,c ") == ["a", "b", "c"]
    with pytest.raises(ValueError):
        es.parse_reasons(" , ")


def test_ask_verdict_success_redo_and_fail():
    reasons = ["x", "y", es.SAFETY_STOP]
    assert es.ask_verdict(reasons, "completed", answers("S")) == {
        "verdict": "success",
        "reason": "",
        "note": "",
    }
    assert es.ask_verdict(reasons, "completed", answers("r")) is None
    # invalid inputs are asked again: "q" verdict, "0" / "9" / "" reason (no default without a watchdog stop)
    got = es.ask_verdict(reasons, "operator stop", answers("q", "f", "0", "9", "", "2", " slipped "))
    assert got == {"verdict": "fail", "reason": "y", "note": "slipped"}


def test_ask_verdict_watchdog_preselects_safety_stop():
    reasons = ["x", "y", es.SAFETY_STOP]
    got = es.ask_verdict(reasons, "watchdog: arm joint 3 measured at 7.0 rad/s > 6", answers("f", "", ""))
    assert got == {"verdict": "fail", "reason": es.SAFETY_STOP, "note": ""}


def test_gate_verdict_consumes_the_file(tmp_path):
    (tmp_path / "verdict.1").write_text(json.dumps({"verdict": "fail", "reason": 2, "note": "n"}))
    assert es.gate_verdict(tmp_path, 1, ["a", "b"]) == {"verdict": "fail", "reason": "b", "note": "n"}
    assert not (tmp_path / "verdict.1").exists()
    (tmp_path / "verdict.2").write_text(json.dumps({"verdict": "redo"}))
    assert es.gate_verdict(tmp_path, 2, ["a"]) is None


def test_stale_stop_file_is_cleared_and_stop_is_consumed(tmp_path):
    (tmp_path / "stop.0").touch()
    es.start_trial_input(tmp_path, 0)  # left over from before the trial: must not end it
    assert not es.stop_requested(tmp_path, 0)
    (tmp_path / "stop.0").touch()
    assert es.stop_requested(tmp_path, 0)
    assert not (tmp_path / "stop.0").exists()


def fake_log(n=20, episode=slice(5, 15), t0=1000.0):
    phase = ["planner token"] * n
    phase[episode] = ["episode"] * (episode.stop - episode.start)
    return {
        "wall": list(t0 + 0.02 * np.arange(n)),
        "phase": phase,
        "state": [np.full(28, i, np.float32) for i in range(n)],
    }


def test_write_trial_files_resume_and_summary(tmp_path):
    assert es.next_trial(tmp_path / "s") == 0
    frames = [(1000.1, "egocentric", b"\xff\xd8a"), (1000.2, "egocentric", b"\xff\xd8bb")]
    extra = {"episode_end": "operator stop", "args": "{}", "verdict": "fail", "reason": "y", "note": ""}
    line = es.write_trial(tmp_path / "s", 0, fake_log(), extra, frames)
    assert line["trial"] == 0 and line["verdict"] == "fail" and line["frames"] == 2
    assert line["wall_start"] == pytest.approx(1000.1) and line["wall_stop"] == pytest.approx(1000.28)
    npz = np.load(tmp_path / "s/trial_000.npz")
    assert str(npz["verdict"]) == "fail" and npz["state"].shape == (20, 28)
    t, names, jpegs = es.load_frames(tmp_path / "s/trial_000_frames.npz")
    assert (
        list(t) == [1000.1, 1000.2]
        and list(names) == ["egocentric"] * 2
        and jpegs == [b"\xff\xd8a", b"\xff\xd8bb"]
    )
    no_episode = fake_log(episode=slice(0, 0))
    line = es.write_trial(tmp_path / "s", 1, no_episode, {**extra, "verdict": "success", "reason": ""}, [])
    assert math.isnan(line["wall_start"])
    assert es.next_trial(tmp_path / "s") == 2  # a restarted session continues at trial 2
    assert not list((tmp_path / "s").glob("*.tmp*"))
    assert es.summarize(tmp_path / "s") == "success 1/2; first attempt 1/2; y: 1"


def right_hand_log(closures):
    """An episode whose right Dex3 closure (sum |q| over its 7 joints) steps through `closures`."""
    log = fake_log(n=len(closures), episode=slice(0, len(closures)))
    log["state"] = [np.r_[np.zeros(21), np.full(7, c / 7)].astype(np.float32) for c in closures]
    return log


def test_first_attempt_score_counts_a_regrasp_as_a_miss(tmp_path):
    extra = {"episode_end": "operator stop", "args": "{}", "verdict": "success", "reason": "", "note": ""}
    held = [0.1, 3.8, 4.2, 4.4, 0.1]  # closed on the bottle, handed over
    regrasp = [0.1, 6.4, 2.7, 4.2, 0.1]  # closed on air first (s03 trial 4)
    for i, closures in enumerate((held, regrasp)):
        es.write_trial(tmp_path / "s", i, right_hand_log(closures), extra, [])
    assert not es.closed_on_air(tmp_path / "s/trial_000.npz")
    assert es.closed_on_air(tmp_path / "s/trial_001.npz")
    assert es.summarize(tmp_path / "s") == "success 2/2; first attempt 1/2"


def _gate_after(tmp_path, first, second, trial=3, delay=0.2):
    """Run gate_verdict while a thread writes `first` now and `second` after `delay` seconds."""
    import threading
    import time

    path = tmp_path / f"verdict.{trial}"
    path.write_text(first)
    threading.Timer(delay, lambda: path.write_text(second)).start()
    t0 = time.time()
    got = es.gate_verdict(tmp_path, trial, ["a", "b"])
    return got, time.time() - t0


@pytest.mark.parametrize(
    "bad",
    [
        {"verdict": "fail", "reason": 0},
        {"verdict": "fail", "reason": 3},
        {"verdict": "fail"},
        {"verdict": "s"},
    ],
)
def test_gate_verdict_rejects_invalid_files_and_keeps_waiting(tmp_path, bad):
    good = json.dumps({"verdict": "fail", "reason": 1})
    got, waited = _gate_after(tmp_path, json.dumps(bad), good)
    assert got == {"verdict": "fail", "reason": "a", "note": ""}
    assert waited >= 0.15


def test_gate_verdict_does_not_consume_a_half_written_file(tmp_path):
    good = json.dumps({"verdict": "success"})
    got, waited = _gate_after(tmp_path, "", good)
    assert got == {"verdict": "success", "reason": "", "note": ""}
    assert waited >= 0.15
