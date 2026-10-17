"""sim_trials.run_trials: the host's side of the judged-trial gate protocol, against a fake scene and a fake streamer
thread that follows sonic_policy_streamer.py --trials --gate-dir (settled.i -> ended.i -> verdict.i, redo repeats i)."""

import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[2] / "examples" / "g1_dex3_training"))
import eval_session as es
import sim_trials

TASK = {
    "table_xy_world": [0.78, 0.0], "bottle_xy_world": [0.5, -0.15], "jitter_m": 0.0,
    "judge": {"table_top_z": 0.8, "bottle_radius": 0.033, "bottle_height": 0.2, "zone_center_pelvis": [0.35, -0.2, 0.25],
              "zone_radius": 0.08, "grasp_hand": 1},
}  # fmt: skip


class FakeScene:
    """Standing robot, bottle on the table, hands far from it. lag_ms per step: `slow_resets` lists the reset counts
    whose trial runs behind the wall clock (500 ms)."""

    def __init__(self, slow_resets=()):
        self.t, self.lag_ms, self.resets, self.slow = 0.0, [], 0, set(slow_resets)
        self.events = []
        self.fall = type("F", (), {"armed": False})()

    def reset_trial(self, seed, bottle_xy, jitter_m):
        self.resets += 1
        self.events.append(("reset", self.t))

    def place_bottle(self, seed, bottle_xy, jitter_m):
        self.events.append(("bottle", self.t))

    def step(self):
        self.t += 0.005

    def record(self):
        pass

    def pace(self, wall0):
        self.lag_ms.append(500.0 if self.resets in self.slow else 0.0)

    def pelvis_state(self):
        return np.array([0.0, 0.0, 0.79]), np.array([1.0, 0.0, 0.0, 0.0])

    def bottle_state(self):
        return np.array([0.5, -0.15, 0.9]), np.array([1.0, 0.0, 0.0, 0.0])

    def palms(self):
        return np.array([[0.1, 0.2, 0.6], [0.1, -0.2, 0.6]])

    def check_fall(self):
        return None

    def cut_trial(self, i):
        pass

    def mark(self, event):
        pass


def fake_streamer(gate: Path, n_trials: int, verdicts: list, stop: threading.Event):
    reasons = es.parse_reasons(None)
    trial = 0
    while trial < n_trials and not stop.is_set():
        settled = gate / f"settled.{trial}"
        while not settled.exists():
            if stop.is_set():
                return
            time.sleep(0.01)
        settled.unlink()
        es.start_trial_input(gate, trial)
        time.sleep(0.1)  # the episode
        es.write_ended(gate, trial, "completed")
        v = es.gate_verdict(gate, trial, reasons)
        verdicts.append("redo" if v is None else v["verdict"])
        if v is not None:
            trial += 1


def run(tmp_path, scene, n_trials=2, **kw):
    gate, verdicts, stop, out = tmp_path / "gate", [], threading.Event(), {}
    gate.mkdir(exist_ok=True)
    streamer = threading.Thread(target=fake_streamer, args=(gate, n_trials, verdicts, stop), daemon=True)
    streamer.start()
    host = threading.Thread(
        target=lambda: out.update(
            r=sim_trials.run_trials(scene, gate, tmp_path / "session", n_trials, 0, TASK, 0.0, **kw)
        ),
        daemon=True,
    )
    host.start()
    host.join(timeout=20)
    streamer.join(timeout=5)  # it reads the last verdict after the host has returned
    stop.set()
    return out.get("r"), verdicts, gate


def test_a_lagging_trial_is_redone_without_deadlock(tmp_path):
    # reset 1 is trial 0's first attempt: lag p99 500 ms > 100 -> redo; the redo must not delete verdict.0 before the
    # streamer has read it (test 4 run 2: both sides waited forever)
    result, verdicts, _ = run(tmp_path, FakeScene(slow_resets={1}))
    assert result == ("completed", "")
    assert verdicts == ["redo", "fail", "fail"]


def test_stale_gate_files_of_a_crashed_run_are_cleared_first(tmp_path):
    gate = tmp_path / "gate"
    gate.mkdir()
    for name in ("ended.0", "stop.0", "verdict.0"):
        (gate / name).write_text("stale")
    result, verdicts, _ = run(tmp_path, FakeScene())
    assert result == ("completed", "")
    assert verdicts == ["fail", "fail"]


def test_one_attempt_per_stack_ends_the_stack_on_a_redo(tmp_path):
    # one stack per trial (spec fallback): a redo inside the same stack would start from the planner's turned heading
    result, verdicts, _ = run(tmp_path, FakeScene(slow_resets={1}), one_attempt=True)
    assert result[0] == "redo" and "lag" in result[1]
    assert verdicts == ["redo"]


def test_bottle_is_placed_after_the_settle(tmp_path):
    # smoke trial 0: the bottle placed at the reset met the hand still raised in the default pose and was bumped
    scene = FakeScene()
    run(tmp_path, scene, n_trials=1)
    (_, t_reset), (kind, t_bottle) = scene.events[0], scene.events[1]
    assert kind == "bottle" and t_bottle >= t_reset + 5.0 - 1e-6
