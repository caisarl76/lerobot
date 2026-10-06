"""Gate, termination and fall helpers shared by the sim hosts (sim_scene.py, sonic_official_sim_host.py) and
sonic_policy_streamer.py: GATE/done carries how the streamer's episode ended, sim/termination.json records why the
sim run stopped, and FallDetector decides when the robot fell. numpy only, so it runs on the workstation and in the
sim container.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

FALL_TILT_DEG, FALL_Z_M, FALL_HOLD_S = 20.0, 0.4, 0.2  # gear_sonic itself resets the robot below z 0.2 m


def write_done(gate_dir: Path, episode_end: str) -> None:
    """GATE/done with how the streamer's episode ended ("completed" or the reason it stopped early)."""
    tmp = Path(gate_dir) / "done.tmp"
    tmp.write_text(json.dumps({"episode_end": episode_end}))
    os.replace(tmp, Path(gate_dir) / "done")  # atomic: a reader never sees a half-written file


def read_done(gate_dir: Path) -> str | None:
    p = Path(gate_dir) / "done"
    if not p.exists():
        return None
    text = p.read_text().strip()
    return json.loads(text)["episode_end"] if text else "unknown"


def write_termination(out_dir: Path, reason: str, phase: str, detail: str = "", **fields) -> None:
    """sim/termination.json: reason is completed | fell | aborted | timeout | error."""
    record = {"reason": reason, "t_wall": time.time(), "phase": phase, "detail": detail, **fields}
    (Path(out_dir) / "termination.json").write_text(json.dumps(record, indent=1, default=str))


def tilt_deg(quat_wxyz) -> float:
    _, x, y, _ = quat_wxyz
    return float(np.degrees(np.arccos(np.clip(1 - 2 * (x * x + y * y), -1, 1))))


class FallDetector:
    """Armed after GATE/settled (the robot hangs on the band before). Fires when the pelvis tilts more than 20 deg,
    drops below 0.4 m or no foot touches the floor, sustained for 0.2 s, or once the simulator itself reported a fall
    (latch(): gear_sonic resets the robot upright below 0.2 m inside sim_step, which could hide a fast collapse)."""

    def __init__(self):
        self.armed, self._since, self.latched = False, None, None

    def latch(self, reason: str) -> None:
        if self.armed and self.latched is None:
            self.latched = reason

    def update(self, t: float, quat_wxyz, z: float, floor_contacts: int) -> str | None:
        if not self.armed:
            return None
        if self.latched:
            return self.latched
        tilt = tilt_deg(quat_wxyz)
        if tilt <= FALL_TILT_DEG and z >= FALL_Z_M and floor_contacts > 0:
            self._since = None
            return None
        self._since = t if self._since is None else self._since
        if t - self._since >= FALL_HOLD_S:
            return f"tilt {tilt:.1f} deg, pelvis z {z:.3f} m, floor contacts {floor_contacts}"
        return None
