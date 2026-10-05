"""Pure helpers of the WBC backend comparison (docs/superpowers/specs/2026-10-01-g1-wbc-backends-design.md), shared by
sonic_policy_streamer.py, the sim hosts (sim_scene.py, decoupled_wbc_sim_host.py), sonic_stream_eval.py and
wbc_compare.py. numpy only (msgpack imported where used), so it runs on the workstation, in the sim container and in
the scoring container.

joint_ref (31): arms 14 (motor 15..28) | Dex3 hands 14 (dataset order) | waist 3 (motor 12..14: yaw, roll, pitch).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np

try:
    from .sonic_targets import ACTION_NAMES, NOMINAL_BODY
except ImportError:
    from sonic_targets import ACTION_NAMES, NOMINAL_BODY

WAIST_NAMES = ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint")
REF_NAMES = ACTION_NAMES + WAIST_NAMES
WAIST = slice(28, 31)
SYNTHETIC_WAIST_S = 16.0  # yaw 0-5 s, roll 5-10 s, pitch 10-16 s
JOINT_TOPIC = b"joints"
FALL_TILT_DEG, FALL_Z_M, FALL_HOLD_S = 20.0, 0.4, 0.2  # gear_sonic itself resets the robot below z 0.2 m
MIN_COVERAGE, MAX_HELD = 0.98, 0.02


def to_joint_ref(chunk) -> np.ndarray:
    """[N,28] or [N,31] joints -> [N,31] joint_ref (28D gets the nominal waist)."""
    chunk = np.asarray(chunk, np.float32)
    if chunk.ndim != 2 or chunk.shape[1] not in (28, 31):
        raise ValueError(f"expected a [N,28] or [N,31] joint chunk, got {chunk.shape}")
    if chunk.shape[1] == 31:
        return chunk
    return np.hstack([chunk, np.tile(NOMINAL_BODY[12:15].astype(np.float32), (len(chunk), 1))])


def replay_chunk(actions, k: int, horizon: int) -> np.ndarray:
    """Recorded actions[k : k + horizon], the last row held past the episode's end."""
    actions = np.asarray(actions, np.float32)
    if not 0 <= k < len(actions) or horizon < 1:
        raise ValueError(f"replay frame {k} outside the episode's {len(actions)} frames (horizon {horizon})")
    out = actions[k : k + horizon]
    return np.vstack([out, np.repeat(out[-1:], horizon - len(out), axis=0)])


def synthetic_waist(t) -> np.ndarray:
    """Waist test track [N,3] (yaw, roll, pitch rad) at episode times t (s), one axis at a time: yaw +-0.4 sine at
    0.2 Hz (0-5 s), roll +-0.15 sine at 0.2 Hz (5-10 s), pitch 0 -> 0.3 -> 0 (2 s ramp, 2 s hold, 2 s ramp; 10-16 s),
    then zero."""
    t = np.asarray(t, np.float64)
    w = np.zeros((len(t), 3))
    s = np.sin(2 * np.pi * 0.2 * t)
    yaw, roll = (t >= 0) & (t < 5), (t >= 5) & (t < 10)
    w[yaw, 0] = 0.4 * s[yaw]
    w[roll, 1] = 0.15 * s[roll]
    tp = t - 10
    w[:, 2] = 0.3 * np.clip(np.minimum(tp / 2, (6 - tp) / 2), 0, 1)
    return w


def check_replay_width(width: int, action_space: str, synthetic: bool) -> None:
    expected = {"sonic78": 78, "joint28": 28, "joint31": 28 if synthetic else 31}[action_space]
    if width != expected:
        raise ValueError(
            f"--replay dataset has {width}D actions; --action-space {action_space}"
            f"{' --synthetic-waist' if synthetic else ''} needs {expected}D"
        )


def check_synthetic_duration(duration_s: float) -> None:
    if duration_s < SYNTHETIC_WAIST_S:
        raise ValueError(f"--synthetic-waist needs >= {SYNTHETIC_WAIST_S:g} s, the run is {duration_s:.1f} s")


def pack_joint_message(seq: int, phase: str, frame: int, joint_ref, t_wall: float | None = None) -> bytes:
    """Streamer -> decoupled_wbc_sim_host.py, one per 50 Hz tick. q_body: motor order 0..28 (legs nominal and
    ignored by backend C, waist used for 31D); q_hand: dataset order (left thumb0-2, middle0-1, index0-1; right
    thumb0-2, index0-1, middle0-1)."""
    import msgpack

    ref = np.asarray(joint_ref, np.float32)
    if ref.shape != (31,) or not np.isfinite(ref).all():
        raise ValueError(f"joint_ref must be 31 finite values, got {ref}")
    body = NOMINAL_BODY.astype(np.float32).copy()
    body[15:29], body[12:15] = ref[:14], ref[WAIST]
    payload = {
        "t_wall": time.time() if t_wall is None else float(t_wall),
        "seq": int(seq),
        "phase": phase,
        "frame": int(frame),
        "q_body": body.tolist(),
        "q_hand": ref[14:28].tolist(),
    }
    return JOINT_TOPIC + msgpack.packb(payload)


def unpack_joint_message(raw: bytes) -> dict:
    """Inverse of pack_joint_message; values are not checked for finiteness (the host decides to hold)."""
    import msgpack

    if not raw.startswith(JOINT_TOPIC):
        raise ValueError("not a joint message")
    msg = msgpack.unpackb(raw[len(JOINT_TOPIC) :], raw=False)
    msg["q_body"], msg["q_hand"] = (
        np.asarray(msg["q_body"], np.float64),
        np.asarray(msg["q_hand"], np.float64),
    )
    if msg["q_body"].shape != (29,) or msg["q_hand"].shape != (14,):
        raise ValueError("joint message must carry q_body[29] and q_hand[14]")
    return msg


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
    drops below 0.4 m or no foot touches the floor, sustained for 0.2 s."""

    def __init__(self):
        self.armed, self._since = False, None

    def update(self, t: float, quat_wxyz, z: float, floor_contacts: int) -> str | None:
        if not self.armed:
            return None
        tilt = tilt_deg(quat_wxyz)
        if tilt <= FALL_TILT_DEG and z >= FALL_Z_M and floor_contacts > 0:
            self._since = None
            return None
        self._since = t if self._since is None else self._since
        if t - self._since >= FALL_HOLD_S:
            return f"tilt {tilt:.1f} deg, pelvis z {z:.3f} m, floor contacts {floor_contacts}"
        return None


def rpy_from_matrix(r) -> np.ndarray:
    """Roll, pitch, yaw of R = Rz(yaw) Ry(pitch) Rx(roll) (pinocchio rpy.matrixToRpy convention)."""
    r = np.asarray(r, np.float64)
    return np.array(
        [np.arctan2(r[2, 1], r[2, 2]), np.arcsin(np.clip(-r[2, 0], -1, 1)), np.arctan2(r[1, 0], r[0, 0])]
    )


def log_settings(args_json: str) -> tuple[str, str]:
    """(backend, action_space) of a streamer log; logs from before the backends existed are SONIC token runs."""
    args = json.loads(args_json)
    return args.get("backend", "sonic"), args.get("action_space", "sonic78")


def run_validity(termination, phase, frame, wall, sim_wall, n_frames: int, held_fraction: float = 0.0):
    """(valid, reason): only runs that completed, cover >= 98% of the episode's frames, stay inside the sim
    recording and (backend C) held their targets on <= 2% of episode records are scored. Same rule for A and C."""
    if termination is None:
        return False, "no termination.json"
    if termination.get("reason") != "completed":
        return False, f"termination: {termination.get('reason')} ({termination.get('detail', '')})"
    ep = np.asarray(phase) == "episode"
    if not ep.any():
        return False, "no episode ticks"
    coverage = len(np.unique(np.asarray(frame)[ep])) / n_frames
    if coverage < MIN_COVERAGE:
        return False, f"scored frames cover {coverage:.1%} of the episode (< {MIN_COVERAGE:.0%})"
    w = np.asarray(wall)[ep]
    if w.min() < sim_wall[0] or w.max() > sim_wall[-1]:
        return False, "streamer episode ticks fall outside the sim recording"
    if held_fraction > MAX_HELD:
        return (
            False,
            f"targets held (stale or non-finite messages) on {held_fraction:.1%} of records (> {MAX_HELD:.0%})",
        )
    return True, "ok"
