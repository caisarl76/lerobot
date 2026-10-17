"""Judged trials of the Isaac sim host (sonic_official_sim_host.py TRIALS=n): the host's side of the gate protocol of
sonic_policy_streamer.py --trials --gate-dir. Per trial: reset robot + bottle, settle 5 s -> settled.i, judge every
physics step (early verdict -> stop.i), streamer's ended.i -> verdict.i; sim_state_{i:03d}.npz per trial."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

try:
    from .eval_session import clear_trial_files, next_trial, read_ended
    from .handover_success import Judge
    from .wbc_common import lag_stats, tilt_deg
except ImportError:
    from eval_session import clear_trial_files, next_trial, read_ended
    from handover_success import Judge
    from wbc_common import lag_stats, tilt_deg

LAG_REDO_P99_MS = 100.0  # relaxed test 0b gate (user, 2026-10-15): a trial lagging more is redone


def yaw_deg(quat_wxyz) -> float:
    w, x, y, z = quat_wxyz
    return float(np.degrees(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))))


def run_trials(scene, gate: Path, session_dir: Path, n_trials: int, seed: int, task: dict, wall0: float,
               deploy_alive=lambda: True, one_attempt: bool = False) -> tuple[str, str]:  # fmt: skip
    """Returns (reason, detail) when all trials are judged or the deploy dies. one_attempt (one stack per trial, spec
    fallback): return ("redo", note) after a redone attempt, so the next stack retries trial i from a fresh deploy (the
    planner keeps its world heading across a reset)."""
    trial, cleared = next_trial(session_dir), None
    while trial < n_trials:
        if not deploy_alive():
            return "error", f"deploy exited before trial {trial}"
        # stale files of a crashed run; not on a redo: the streamer may not have read verdict.i yet
        if trial != cleared:
            clear_trial_files(gate, trial)
            cleared = trial
        for attempt in (1, 2):
            scene.reset_trial(seed + trial, task["bottle_xy_world"], task["jitter_m"])
            t0 = scene.t
            while scene.t < t0 + 5:
                if not deploy_alive():
                    return "error", f"deploy exited before trial {trial}"
                scene.step()
                scene.record()  # the settle is on record: test 4 checks the robot does not step or turn
                scene.pace(wall0)
            pos, quat = scene.pelvis_state()
            if pos[2] > 0.6 and tilt_deg(quat) < 20:
                break
            if attempt == 2:
                return "reset_failed", f"robot down after two resets before trial {trial}"
        # the bottle comes after the settle: placed at the reset it meets the hand still raised in the default pose
        scene.place_bottle(seed + trial, task["bottle_xy_world"], task["jitter_m"])
        scene.fall.armed = True
        judge = Judge(task["judge"])
        judge.table_xy = np.asarray(task["table_xy_world"], float)
        # re-anchor the pacing while the robot stands: a stall before (load on the shared machine) does not carry
        # into the trial; a loaded machine cannot run the sim faster than real time to catch up
        stall_ms = scene.lag_ms[-1]
        wall0 = time.monotonic() - scene.t
        (gate / f"settled.{trial}").touch()
        pos, quat = scene.pelvis_state()
        scene.mark(f"trial {trial}: settled.{trial}, lag before re-anchoring {stall_ms:.0f} ms, xy {pos[:2].round(3)}, "
                   f"yaw {yaw_deg(quat):.1f}")  # fmt: skip
        lag0 = len(scene.lag_ms)
        early_written = False
        while (ended := read_ended(gate, trial)) is None:
            if not deploy_alive():
                return "error", f"deploy exited during trial {trial}"
            scene.step()
            scene.record()
            bottle_pos, bottle_quat = scene.bottle_state()
            pelvis_pos, pelvis_quat = scene.pelvis_state()
            judge.feed(
                scene.t, bottle_pos, bottle_quat, scene.palms(), pelvis_pos, pelvis_quat, scene.check_fall()
            )
            if not early_written and judge.early() is not None:
                (gate / f"stop.{trial}").touch()
                early_written = True
                scene.mark(f"trial {trial}: early verdict {judge.early()}")
            scene.pace(wall0)
        verdict = Judge.to_gate(judge.final(ended))
        lag = lag_stats(scene.lag_ms[lag0:])
        if lag["p99_ms"] > LAG_REDO_P99_MS:
            verdict = {"verdict": "redo", "reason": "", "note": f"sim lag p99 {lag['p99_ms']:.0f} ms"}
        tmp = gate / f".verdict.{trial}.tmp"
        tmp.write_text(json.dumps(verdict))
        tmp.replace(gate / f"verdict.{trial}")
        scene.cut_trial(trial)
        scene.mark(f"trial {trial}: ended '{ended}', verdict {verdict}, lag {lag}")
        if verdict["verdict"] != "redo":
            trial += 1
        elif one_attempt:
            return "redo", verdict["note"]
    return "completed", ""
