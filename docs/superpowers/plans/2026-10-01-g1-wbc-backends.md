# G1 WBC Backends (SONIC vs GR00T decoupled WBC) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Drive 28D / 31D joint-target policies (and recorded actions) on the simulated G1 through two whole-body controllers — SONIC (A) and NVIDIA's GR00T decoupled WBC (C) — in the same scene, with one validity-gated scorer and numeric comparison gates.

**Architecture:** `sonic_policy_streamer.py` gains `--replay`, `--action-space joint31`, `--synthetic-waist` and `--backend decoupled`; A keeps its 30 Hz joint chunk → encoder → 50 Hz token path and logs a 50 Hz `joint_ref` computed alongside. The scene/recording code of the existing sim host moves into `sim_scene.py`, shared by the A host and a new C host that runs upstream `decoupled_wbc` at a pinned commit. Pure logic (joint message, replay, synthetic waist, gates, fall detection, validity) lives in `wbc_common.py` and is unit-tested locally; everything MuJoCo/upstream-specific is checked with host self-checks in the sim container.

**Tech Stack:** Python 3.12 (workstation, `uv run`), Python 3.10 sim container `jihun/sonic-vla-sim:startupfix-20260907` (gear_sonic, mujoco 3.12, pinocchio 2.7, torch CPU, zmq, msgpack), onnxruntime from a mounted folder, NVlabs/GR00T-WholeBodyControl `b042411fae`, ZMQ + msgpack, LeRobot datasets.

**Spec:** `docs/superpowers/specs/2026-10-01-g1-wbc-backends-design.md` (revision 3, approved).

## Global Constraints

- No installs on the H100 host or the G1 PC2 system Python; extra Python packages only in mounted folders (`PYTHONPATH`).
- Upstream pinned at `b042411fae` (`b042411fae38ee4d1af9aac82a37a1f8d14d6dd0`); mount only its `decoupled_wbc/` at `/upstream/decoupled_wbc` (never shadow the container's `gear_sonic`).
- Preserve A's path exactly: 30 Hz joint chunk → `joint_chunk_to_sonic` → 50 Hz token interpolation (`ChunkResampler`). `joint_ref` is a second resampler over the same joint chunk, for logging only.
- 31D layout: `[arms 14 | hands 14 | waist 3]`, waist motor order yaw, roll, pitch (indices 12–14); first 28 columns equal `joint28`.
- Backend C contract: WBC yaml `decoupled_wbc/control/main/teleop/configs/g1_29dof_gear_wbc.yaml`; lower-body config = its `GEAR_WBC_CONFIG` (516 obs, 15 actions); `model_path` = `policy/GR00T-WholeBodyControl-Balance.onnx,policy/GR00T-WholeBodyControl-Walk.onnx`; body gains `MOTOR_KP/MOTOR_KD`; hand gains kp 1.5 / kd 0.1; `IdentityPolicy` upper body; every wrapper goal complete (`target_upper_body_pose`, `base_height_command=[0.74]`, `navigate_cmd=[0,0,0]`); RL output activated on the lower-body policy directly, never via a toggle-only wrapper goal.
- Physics 200 Hz (A's scene `SIMULATE_DT` 0.005 and upstream `sim_frequency` 200 — this corrects the spec's "500 Hz", which came from the standalone sim2mujoco yaml), control 50 Hz.
- Dex3 right-hand order: A in sim uses `--dex3-right-order swap` (NVIDIA bridge); C addresses MuJoCo joints by name and uses `--dex3-right-order dataset`.
- HE episodes: `--start planner`, `TABLE_GAP_CM=30`.
- Repeats 3 per configuration × episode; gates G0/G1/G2 and the 31D check exactly as in the spec.
- Ask the user before: choosing GPUs for sim batches, and queueing any training on the H100.
- Local tests: `uv run python -m unittest <file>` (pytest is not installed in the base env).

## Review Focus

1. Synthetic waist on an episode shorter than the 16 s track — the streamer must refuse at startup, not stream a truncated track (Task 2 `check_synthetic_duration`, called in Task 3).
2. Stale or non-finite joint messages during a C run — a run with held targets on > 2% of episode records must be invalid, not scored as normal (Task 2 `run_validity` test).
3. The streamer ends the episode early (watchdog, stale chunk/state) — the host must record `aborted`, and the scorer must reject it (Task 2 `read_done` / `run_validity` tests).
4. `--replay` with a dataset whose action width does not match `--action-space` (e.g. a 78D token dataset with `joint28`) — fail at startup with a clear message (Task 2 `check_replay_width` test).
5. Old streamer logs without `joint_ref` / `backend` (all pre-change runs) — the scorer must still score them as backend `sonic` with recorded-action metrics (Task 2 `log_settings` test).

## Paths used throughout

```
H100 host (ssh alias h100), all work in containers:
A    = /mnt/data01/jhkim/model_weight/sonic_roundtrip_20260923        (audit dir, mounted as /audit or /a)
RO   = /mnt/data01/jhkim/model_weight/g1_dex3_20260922                (mounted as /run-output)
HE   = /run-output/humanoid_everyday_g1_20260923                      (HE datasets/runs/configs)
CODE = $A/code_wbc   (this directory's scripts, CODE_DIR for the launcher)
SIM  = jihun/sonic-vla-sim:startupfix-20260907 (python /opt/sonic-sim/bin/python)
LR   = image 4cbe2a3f7fc6 (venv /run-output/environment/venv/bin/python)
```

Code sync to the H100 (used by several tasks; dirs there are written through a container):

```bash
cd /home/jihun/work/lerobot/examples/g1_dex3_training
tar c --exclude=__pycache__ . | ssh h100 "docker run --rm -i -v /mnt/data01/jhkim/model_weight/sonic_roundtrip_20260923:/a --entrypoint sh 4cbe2a3f7fc6 -c 'rm -rf /a/code_wbc && mkdir -p /a/code_wbc && tar x -C /a/code_wbc'"
```

---

### Task 1: 31D encoder input (waist into SONIC's body target)

**Files:**
- Modify: `examples/g1_dex3_training/sonic_targets.py` (`build_encoder_inputs`, `joint_chunk_to_sonic` docstring)
- Test: `tests/datasets/test_g1_dex3_sonic_targets.py`

**Interfaces:**
- Produces: `build_encoder_inputs(actions [N,28] | [N,31], limits [28,2], ...)` — columns 28–30 (waist yaw, roll, pitch) replace `NOMINAL_BODY[12:15]`; `joint_chunk_to_sonic(chunk [N,28] | [N,31], limits, encoder, fps)` → `[N,78]`. 28D behaviour bit-identical.

- [ ] **Step 1: Write the failing tests** (append to `SonicTargetTests`)

```python
    def test_31d_with_nominal_waist_equals_28d(self):
        rng = np.random.default_rng(0)
        a28 = rng.uniform(-0.5, 0.5, (45, 28))
        a31 = np.hstack([a28, np.tile(NOMINAL_BODY[12:15], (45, 1))])
        i28, h28, _ = build_encoder_inputs(a28, self.limits, arm_speed_limit=None, hand_speed_limit=None)
        i31, h31, _ = build_encoder_inputs(a31, self.limits, arm_speed_limit=None, hand_speed_limit=None)
        np.testing.assert_array_equal(i31, i28)
        np.testing.assert_array_equal(h31, h28)

    def test_31d_waist_enters_body_target(self):
        a31 = np.zeros((30, 31))
        a31[:, 28:31] = [0.3, -0.1, 0.2]
        inputs, _, report = build_encoder_inputs(a31, self.limits)
        body = inputs[0, 4:294].reshape(10, 29)
        for motor, val in ((12, 0.3), (13, -0.1), (14, 0.2)):
            slot = int(np.flatnonzero(ISAAC_FROM_MOTOR == motor)[0])
            np.testing.assert_allclose(body[:, slot], val, atol=1e-6)
        self.assertEqual(report["lower_body_assumption"], "fixed_nominal_standing_legs_waist_from_action")

    def test_rejects_other_widths(self):
        with self.assertRaises(ValueError):
            build_encoder_inputs(np.zeros((5, 30)), self.limits)
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run python -m unittest tests/datasets/test_g1_dex3_sonic_targets.py`
Expected: FAIL/ERROR in the two 31D tests (`expected nonempty finite actions [N,28]`).

- [ ] **Step 3: Implement.** In `build_encoder_inputs` replace the shape check and split off the waist; add it to the body target after the arms:

```python
    actions = np.asarray(actions, dtype=np.float64)
    limits = np.asarray(limits, dtype=np.float64)
    if actions.ndim != 2 or actions.shape[1] not in (28, 31) or not len(actions) or not np.isfinite(actions).all():
        raise ValueError("expected nonempty finite actions [N,28] or [N,31]")
    waist, actions = actions[:, 28:31], actions[:, :28]  # waist is [N,0] for 28D
```

and, right after `body[:, 15:] = filtered[:, :14]`:

```python
    if waist.shape[1]:  # 31D: commanded waist (yaw, roll, pitch) instead of the nominal one, no slew limit
        body[:, 12:15] = np.column_stack([np.interp(dense_times, source_times, waist[:, i]) for i in range(3)])
```

and in `report`:

```python
        "lower_body_assumption": "fixed_nominal_standing_legs_waist_from_action"
        if waist.shape[1]
        else "fixed_nominal_standing_legs_and_waist",
```

Update `joint_chunk_to_sonic`'s docstring first line to `[N,28] or [N,31] joints (31D: + waist yaw/roll/pitch) -> [N,78] tokens + hands`. Update the module docstring sentence "The input has no measured legs, waist, or root pose" to "The input has no measured legs or root pose; the waist is nominal unless a 31D action supplies it."

- [ ] **Step 4: Run to verify they pass**

Run: `uv run python -m unittest tests/datasets/test_g1_dex3_sonic_targets.py tests/datasets/test_g1_dex3_sonic_token_stream.py`
Expected: all OK (existing 15 + 3 new).

- [ ] **Step 5: Commit**

```bash
git add examples/g1_dex3_training/sonic_targets.py tests/datasets/test_g1_dex3_sonic_targets.py
git commit -m "feat(g1-dex3): 31D joint actions put the commanded waist into SONIC's encoder target"
```

---

### Task 2: `wbc_common.py` — pure helpers shared by streamer, hosts, scorer and gates

**Files:**
- Create: `examples/g1_dex3_training/wbc_common.py`
- Test: `tests/datasets/test_g1_dex3_wbc_common.py`

**Interfaces:**
- Consumes: `sonic_targets.NOMINAL_BODY`, `sonic_targets.ACTION_NAMES` (arms 14 + hands 14, dataset order).
- Produces (exact names used by later tasks):
  - `WAIST_NAMES: tuple[str, str, str]`, `REF_NAMES: tuple[str, ...]` (31 names: `ACTION_NAMES + WAIST_NAMES`), `WAIST = slice(28, 31)`, `SYNTHETIC_WAIST_S = 16.0`
  - `to_joint_ref(chunk) -> np.ndarray[N,31] float32`
  - `replay_chunk(actions, k, horizon) -> np.ndarray[horizon, D]`
  - `synthetic_waist(t) -> np.ndarray[N,3]`
  - `check_replay_width(width, action_space, synthetic) -> None` (raises `ValueError`)
  - `check_synthetic_duration(duration_s) -> None` (raises `ValueError`)
  - `JOINT_TOPIC = b"joints"`, `pack_joint_message(seq, phase, frame, joint_ref, t_wall=None) -> bytes`, `unpack_joint_message(raw) -> dict` (keys `t_wall, seq, phase, frame, q_body[29], q_hand[14]`)
  - `write_done(gate_dir, episode_end)`, `read_done(gate_dir) -> str | None`
  - `write_termination(out_dir, reason, phase, detail="", **fields)`
  - `FallDetector` (`.armed`, `.update(t, quat_wxyz, z, floor_contacts) -> str | None`), `tilt_deg(quat_wxyz) -> float`
  - `rpy_from_matrix(R) -> np.ndarray[3]` (roll, pitch, yaw; R = Rz·Ry·Rx, as pinocchio `matrixToRpy`)
  - `log_settings(args_json) -> tuple[str, str]` (backend, action_space)
  - `run_validity(termination, phase, frame, wall, sim_wall, n_frames, held_fraction=0.0) -> tuple[bool, str]`

- [ ] **Step 1: Write the failing tests** — `tests/datasets/test_g1_dex3_wbc_common.py`:

```python
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[2] / "examples" / "g1_dex3_training"))
from sonic_targets import NOMINAL_BODY
from wbc_common import (
    REF_NAMES,
    FallDetector,
    check_replay_width,
    check_synthetic_duration,
    log_settings,
    pack_joint_message,
    read_done,
    replay_chunk,
    rpy_from_matrix,
    run_validity,
    synthetic_waist,
    to_joint_ref,
    unpack_joint_message,
    write_done,
)


def rot(roll, pitch, yaw):
    cr, sr, cp, sp, cy, sy = np.cos(roll), np.sin(roll), np.cos(pitch), np.sin(pitch), np.cos(yaw), np.sin(yaw)
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


class WbcCommonTests(unittest.TestCase):
    def test_joint_ref_pads_nominal_waist_and_passes_31d(self):
        c28 = np.arange(56, dtype=np.float32).reshape(2, 28)
        ref = to_joint_ref(c28)
        self.assertEqual(ref.shape, (2, 31))
        np.testing.assert_array_equal(ref[:, :28], c28)
        np.testing.assert_allclose(ref[:, 28:], np.tile(NOMINAL_BODY[12:15], (2, 1)))
        c31 = np.ones((3, 31), np.float32)
        np.testing.assert_array_equal(to_joint_ref(c31), c31)
        with self.assertRaises(ValueError):
            to_joint_ref(np.zeros((2, 30)))
        self.assertEqual(len(REF_NAMES), 31)

    def test_replay_chunk_pads_with_last_row(self):
        acts = np.arange(10, dtype=np.float32)[:, None] * np.ones((1, 28), np.float32)
        out = replay_chunk(acts, 7, 5)
        self.assertEqual(out.shape, (5, 28))
        np.testing.assert_array_equal(out[:, 0], [7, 8, 9, 9, 9])
        with self.assertRaises(ValueError):
            replay_chunk(acts, 10, 5)

    def test_synthetic_waist_track(self):
        t = np.arange(0, 20, 0.02)
        w = synthetic_waist(t)
        self.assertEqual(w.shape, (len(t), 3))
        for tt in (0.0, 5.0, 10.0, 16.0, 19.0):
            np.testing.assert_allclose(synthetic_waist([tt])[0], 0, atol=1e-9)
        self.assertAlmostEqual(synthetic_waist([1.25])[0, 0], 0.4, places=6)
        self.assertAlmostEqual(synthetic_waist([6.25])[0, 1], 0.15, places=6)
        self.assertAlmostEqual(synthetic_waist([11.0])[0, 2], 0.15, places=6)
        self.assertAlmostEqual(synthetic_waist([13.0])[0, 2], 0.3, places=6)
        self.assertLess(np.abs(np.diff(w, axis=0)).max(), 0.02)  # smooth at 50 Hz (no jumps)
        self.assertTrue(np.all((np.abs(w) > 1e-9).sum(1) <= 1))  # one axis at a time

    def test_startup_checks(self):
        check_replay_width(28, "joint28", False)
        check_replay_width(28, "joint31", True)
        check_replay_width(78, "sonic78", False)
        with self.assertRaises(ValueError):
            check_replay_width(78, "joint28", False)
        with self.assertRaises(ValueError):
            check_replay_width(28, "joint31", False)
        check_synthetic_duration(16.0)
        with self.assertRaises(ValueError):
            check_synthetic_duration(14.6)

    def test_joint_message_round_trip(self):
        ref = np.linspace(-1, 1, 31).astype(np.float32)
        msg = unpack_joint_message(pack_joint_message(7, "episode", 12, ref, t_wall=123.5))
        self.assertEqual((msg["seq"], msg["phase"], msg["frame"], msg["t_wall"]), (7, "episode", 12, 123.5))
        np.testing.assert_allclose(msg["q_body"][15:29], ref[:14], atol=1e-7)
        np.testing.assert_allclose(msg["q_body"][12:15], ref[28:31], atol=1e-7)
        np.testing.assert_allclose(msg["q_body"][:12], NOMINAL_BODY[:12], atol=1e-7)
        np.testing.assert_allclose(msg["q_hand"], ref[14:28], atol=1e-7)
        with self.assertRaises(ValueError):
            pack_joint_message(0, "x", -1, np.full(31, np.nan))
        with self.assertRaises(ValueError):
            unpack_joint_message(b"pose" + b"\x00")

    def test_done_gate_round_trip(self):
        with tempfile.TemporaryDirectory() as d:
            gate = Path(d)
            self.assertIsNone(read_done(gate))
            write_done(gate, "watchdog: arm joint 3 measured at 7.0 rad/s > 6")
            self.assertTrue(read_done(gate).startswith("watchdog"))
            (gate / "done").write_text("")
            self.assertEqual(read_done(gate), "unknown")

    def test_fall_detector(self):
        f = FallDetector()
        tilted = [np.cos(np.radians(15)), np.sin(np.radians(15)), 0, 0]  # 30 deg tilt about x
        self.assertIsNone(f.update(0.0, tilted, 0.7, 8))  # not armed
        f.armed = True
        self.assertIsNone(f.update(1.0, tilted, 0.7, 8))
        self.assertIsNone(f.update(1.1, [1, 0, 0, 0], 0.7, 8))  # transient, reset
        self.assertIsNone(f.update(1.2, tilted, 0.7, 8))
        self.assertIsNotNone(f.update(1.41, tilted, 0.7, 8))
        g = FallDetector()
        g.armed = True
        g.update(0.0, [1, 0, 0, 0], 0.7, 0)
        self.assertIsNotNone(g.update(0.25, [1, 0, 0, 0], 0.7, 0))  # both feet off the floor

    def test_rpy_from_matrix(self):
        np.testing.assert_allclose(rpy_from_matrix(rot(0.1, -0.2, 0.3)), [0.1, -0.2, 0.3], atol=1e-9)

    def test_log_settings_defaults_for_old_logs(self):
        self.assertEqual(log_settings(json.dumps({"action_space": "sonic78"})), ("sonic", "sonic78"))
        self.assertEqual(log_settings(json.dumps({})), ("sonic", "sonic78"))
        self.assertEqual(
            log_settings(json.dumps({"backend": "decoupled", "action_space": "joint31"})), ("decoupled", "joint31")
        )

    def test_run_validity(self):
        phase = np.array(["blend in"] + ["episode"] * 100 + ["return"])
        frame = np.r_[-1, np.arange(100), -1]
        wall = np.arange(102) * 0.02 + 10
        sim_wall = np.arange(0, 200) * 0.02 + 9
        ok = {"reason": "completed"}
        self.assertEqual(run_validity(ok, phase, frame, wall, sim_wall, 100), (True, "ok"))
        self.assertFalse(run_validity(None, phase, frame, wall, sim_wall, 100)[0])
        self.assertFalse(run_validity({"reason": "fell", "detail": "tilt"}, phase, frame, wall, sim_wall, 100)[0])
        self.assertFalse(run_validity({"reason": "aborted"}, phase, frame, wall, sim_wall, 100)[0])
        self.assertFalse(run_validity(ok, phase, frame, wall, sim_wall, 110)[0])  # 100/110 < 98 %
        self.assertFalse(run_validity(ok, phase, frame, wall, sim_wall[:50], 100)[0])  # tail past the sim end
        self.assertFalse(run_validity(ok, phase, frame, wall, sim_wall, 100, held_fraction=0.03)[0])
        self.assertTrue(run_validity(ok, phase, frame, wall, sim_wall, 100, held_fraction=0.01)[0])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run python -m unittest tests/datasets/test_g1_dex3_wbc_common.py`
Expected: ERROR `ModuleNotFoundError: No module named 'wbc_common'`.

- [ ] **Step 3: Implement** `examples/g1_dex3_training/wbc_common.py`:

```python
"""Pure helpers of the WBC backend comparison (docs/superpowers/specs/2026-10-01-g1-wbc-backends-design.md), shared by
sonic_policy_streamer.py, the sim hosts (sim_scene.py, decoupled_wbc_sim_host.py), sonic_stream_eval.py and
wbc_compare.py. numpy only (msgpack imported where used), so it runs on the workstation, in the sim container and in
the scoring container.

joint_ref (31): arms 14 (motor 15..28) | Dex3 hands 14 (dataset order) | waist 3 (motor 12..14: yaw, roll, pitch).
"""

from __future__ import annotations

import json
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
    msg["q_body"], msg["q_hand"] = np.asarray(msg["q_body"], np.float64), np.asarray(msg["q_hand"], np.float64)
    if msg["q_body"].shape != (29,) or msg["q_hand"].shape != (14,):
        raise ValueError("joint message must carry q_body[29] and q_hand[14]")
    return msg


def write_done(gate_dir: Path, episode_end: str) -> None:
    """GATE/done with how the streamer's episode ended ("completed" or the reason it stopped early)."""
    (Path(gate_dir) / "done").write_text(json.dumps({"episode_end": episode_end}))


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
        return False, f"targets held (stale or non-finite messages) on {held_fraction:.1%} of records (> {MAX_HELD:.0%})"
    return True, "ok"
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run python -m unittest tests/datasets/test_g1_dex3_wbc_common.py`
Expected: OK (10 tests).

- [ ] **Step 5: Commit**

```bash
git add examples/g1_dex3_training/wbc_common.py tests/datasets/test_g1_dex3_wbc_common.py
git commit -m "feat(g1-dex3): shared WBC helpers: joint message, replay, synthetic waist, gates, fall detection, run validity"
```

---

### Task 3: Streamer — replay, joint31, synthetic waist, joint_ref log, backend decoupled

**Files:**
- Modify: `examples/g1_dex3_training/sonic_policy_streamer.py`
- Test: `tests/datasets/test_g1_dex3_streamer_cli.py` (new)

**Interfaces:**
- Consumes: Task 2 `WAIST, check_replay_width, check_synthetic_duration, pack_joint_message, replay_chunk, synthetic_waist, to_joint_ref, write_done`; Task 1 `joint_chunk_to_sonic` (28D/31D).
- Produces:
  - CLI: `--action-space {sonic78,joint28,joint31}`, `--backend {sonic,decoupled}`, `--replay`, `--replay-horizon N` (default 40), `--synthetic-waist`.
  - Log `.npz` keys: existing + `joint_ref [T,31]` (NaN outside joint runs and outside `episode` for A), `episode_end` (str); `args` JSON contains `backend`, `action_space`.
  - `GATE/done` content via `write_done(gate_dir, episode_end)`.
  - Module function `measured_ref(msg, action_space) -> np.ndarray[31]`.
  - Decoupled phases sent: `wait first chunk`, `leadin` (2 s), `hold` (1 s), `episode`, `return` (2 s).

- [ ] **Step 1: Write the failing tests** — `tests/datasets/test_g1_dex3_streamer_cli.py`:

```python
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

HERE = Path(__file__).parents[2] / "examples" / "g1_dex3_training"
sys.path.insert(0, str(HERE))
from sonic_policy_streamer import measured_ref
from sonic_targets import NOMINAL_BODY


def cli(*args):
    return subprocess.run(
        [sys.executable, str(HERE / "sonic_policy_streamer.py"), *args], capture_output=True, text=True, timeout=120
    )


class StreamerCliTests(unittest.TestCase):
    def test_measured_ref_layout(self):
        body = np.arange(29, dtype=np.float32) / 100
        msg = {"body_q": body, "left_hand_q": np.full(7, 0.5), "right_hand_q": np.full(7, 0.7)}
        ref = measured_ref(msg, "joint31")
        np.testing.assert_allclose(ref[:14], body[15:29])
        np.testing.assert_allclose(ref[14:21], 0.5)
        np.testing.assert_allclose(ref[28:31], body[12:15])
        np.testing.assert_allclose(measured_ref(msg, "joint28")[28:31], NOMINAL_BODY[12:15])

    def test_argument_errors(self):
        cases = [
            (["--replay", "--policy-server", "tcp://x:1", "--dataset-root", "d", "--episode", "0"], "--replay excludes"),
            (["--backend", "decoupled", "--replay", "--dataset-root", "d", "--episode", "0"], "--backend decoupled needs"),
            (["--synthetic-waist", "--action-space", "joint28", "--replay", "--dataset-root", "d", "--episode", "0",
              "--backend", "decoupled"], "--synthetic-waist needs"),
            (["--dataset-root", "d", "--episode", "0"], "exactly one of"),
        ]  # fmt: skip
        for args, text in cases:
            r = cli(*args)
            self.assertEqual(r.returncode, 2, args)
            self.assertIn(text, r.stderr, args)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run python -m unittest tests/datasets/test_g1_dex3_streamer_cli.py`
Expected: ERROR `cannot import name 'measured_ref'`.

- [ ] **Step 3: Implement — imports.** Add `from types import SimpleNamespace` to the stdlib imports. In both branches of the local-import `try/except`, add:

```python
    from .sonic_targets import NOMINAL_BODY
    from .wbc_common import (
        WAIST,
        check_replay_width,
        check_synthetic_duration,
        pack_joint_message,
        replay_chunk,
        synthetic_waist,
        to_joint_ref,
        write_done,
    )
```

(without the leading dots in the `except ImportError` branch).

- [ ] **Step 4: Implement — `measured_ref` and `DatasetImages.actions`.** After `observation_state`:

```python
def measured_ref(msg: dict, action_space: str) -> np.ndarray:
    """31D joint_ref of the measured pose (arms, hands, waist); 28D runs get the nominal waist (backend C's
    lower_body waist location ignores it)."""
    ref = np.r_[observation_state(msg), np.asarray(msg["body_q"], np.float32)[12:15]].astype(np.float32)
    if action_space != "joint31":
        ref[WAIST] = NOMINAL_BODY[12:15]
    return ref
```

In `DatasetImages`:

```python
    def actions(self) -> np.ndarray:
        """The episode's recorded action column [n, D] (for --replay)."""
        out = np.stack([np.asarray(v, np.float32) for v in self.ds.hf_dataset["action"]])
        if len(out) != self.n:
            raise ValueError(f"episode has {self.n} frames but {len(out)} actions")
        return out
```

- [ ] **Step 5: Implement — CLI and validation.** Replace the `--action-space` argument and add the new ones:

```python
    p.add_argument(
        "--action-space",
        choices=["sonic78", "joint28", "joint31"],
        default="sonic78",
        help="joint28: the policy outputs arm + Dex3 hand joints (28D); joint31: + waist yaw/roll/pitch (motor 12-14). "
        "With --backend sonic each chunk is encoded to SONIC tokens here (needs --encoder-model, "
        "--observation-config, --robot-xml)",
    )
    p.add_argument(
        "--backend",
        choices=["sonic", "decoupled"],
        default="sonic",
        help="sonic: SONIC tokens to NVIDIA's deploy; decoupled: 50 Hz joint targets to decoupled_wbc_sim_host.py "
        "(GR00T decoupled WBC; joint action spaces only)",
    )
    p.add_argument(
        "--replay",
        action="store_true",
        help="stream the episode's recorded actions (--dataset-root) instead of a policy: chunks of --replay-horizon "
        "frames starting at the frame of each --replan-s time",
    )
    p.add_argument("--replay-horizon", type=int, default=40, help="--replay chunk length in frames (GR00T: 40)")
    p.add_argument(
        "--synthetic-waist",
        action="store_true",
        help="--action-space joint31: append wbc_common.synthetic_waist (16 s test track) to 28D chunks",
    )
```

Replace `if bool(a.policy_path) == bool(a.policy_server): p.error(...)` and the `joint28 = None ... if a.action_space == "joint28": ...` block with:

```python
    joint_space = a.action_space in ("joint28", "joint31")
    if a.replay:
        if a.policy_path or a.policy_server:
            p.error("--replay excludes --policy-path and --policy-server")
        if a.images != "dataset":
            p.error("--replay needs --images dataset")
    elif bool(a.policy_path) == bool(a.policy_server):
        p.error("give exactly one of --policy-path or --policy-server (or --replay)")
    if a.backend == "decoupled" and not joint_space:
        p.error("--backend decoupled needs --action-space joint28 or joint31")
    if a.backend == "decoupled" and (a.startup_tokens or a.max_token_step > 0):
        p.error("--backend decoupled excludes --startup-tokens and --max-token-step (token-only options)")
    if a.synthetic_waist and a.action_space != "joint31":
        p.error("--synthetic-waist needs --action-space joint31")
    joint28 = None  # (robot joint limits, SONIC encoder) for encoding joint chunks
    if joint_space and a.backend == "sonic":
        if not (a.encoder_model and a.observation_config and a.robot_xml):
            p.error(f"--action-space {a.action_space} needs --encoder-model, --observation-config and --robot-xml")
        joint28 = (load_joint_limits(a.robot_xml), SonicEncoder(a.encoder_model, a.observation_config))
```

Note the argument-error test order: `--backend decoupled --replay` without `--action-space` must hit "--backend decoupled needs" — it does, because `--replay` passes its own checks first.

- [ ] **Step 6: Implement — policy for replay and replay actions.** Replace the `policy = (...)` assignment with:

```python
    if a.replay:  # no policy: images are not needed, actions come from the dataset
        policy = SimpleNamespace(image_keys=[], shapes={}, n_obs=1, chunk=lambda *_: None)
    else:
        policy = (
            RemotePolicy(a.policy_server, a.policy_timeout_s)
            if a.policy_server
            else ChunkPolicy(a.policy_path, a.device, a.backbone_dtype, a.noise_seed)
        )
```

After `images = DatasetImages(...)` / `LiveImages(...)`:

```python
    replay = None
    if a.replay:
        replay = images.actions()
        check_replay_width(replay.shape[1], a.action_space, a.synthetic_waist)
    if a.synthetic_waist:
        check_synthetic_duration(a.duration_s or images.n / images.fps)
```

- [ ] **Step 7: Implement — logging helpers.** Replace the `log = {...}` line and the whole `send` function with:

```python
    log = {k: [] for k in ("wall", "t_ep", "phase", "frame", "token", "hands", "chunk_t0", "state", "joint_ref")}
    no_token, no_ref = np.full(64, np.nan, np.float32), np.full(31, np.nan, np.float32)
    last_sent, slewed = [None], [0]

    def record(token, hands, phase, t_ep, k, chunk_t0, joint_ref):
        msg = state.latest()  # the ZMQ socket is only touched from this (main) thread
        ref = no_ref if joint_ref is None else np.asarray(joint_ref, np.float32)
        for key, val in (
            ("wall", time.time()),
            ("t_ep", t_ep),
            ("phase", phase),
            ("frame", k),
            ("token", token),
            ("hands", hands),
            ("chunk_t0", chunk_t0),
            ("state", observation_state(msg) if msg else np.full(28, np.nan)),
            ("joint_ref", ref),
        ):
            log[key].append(val)  # fmt: skip
        if msg and t_ep >= 0:
            history.append((t_ep, log["state"][-1]))
            del history[:-200]

    def send(token, hands, phase, t_ep=-1.0, k=-1, chunk_t0=np.nan, joint_ref=None):
        token = np.asarray(token, np.float32)
        if a.max_token_step > 0 and last_sent[0] is not None:  # slew limit on every phase
            step = token - last_sent[0]
            if np.abs(step).max() > a.max_token_step:
                token = last_sent[0] + np.clip(step, -a.max_token_step, a.max_token_step)
                slewed[0] += 1
        last_sent[0] = token
        pub.send(pose_message(token, hands, frame[0]))
        frame[0] += 1
        record(token, hands, phase, t_ep, k, chunk_t0, joint_ref)

    def send_joint(ref, phase, t_ep=-1.0, k=-1, chunk_t0=np.nan):
        """--backend decoupled: one joint message (wbc_common.pack_joint_message) per tick."""
        ref = np.asarray(ref, np.float32)
        pub.send(pack_joint_message(frame[0], phase, k, ref))
        frame[0] += 1
        record(no_token, ref[14:28], phase, t_ep, k, chunk_t0, ref)
```

(Keep `zero_h`, `frame`, `history` definitions that precede `log`.)

- [ ] **Step 8: Implement — `infer`.** Replace the body of `infer` with:

```python
    def infer(t_ep, robot_hist):
        obs = [images.frame(max(t_ep - i / images.fps, 0.0)) for i in reversed(range(policy.n_obs))]
        k, task = obs[-1][0], a.task or obs[-1][3]
        if replay is not None:
            joints = replay_chunk(replay, k, a.replay_horizon)
        else:
            # --state-source dataset: recorded state (isolates execution from state feedback); robot: closed loop
            states = [o[2] for o in obs] if a.state_source == "dataset" else robot_hist
            joints = policy.chunk(states, [o[1] for o in obs], task)
        if a.synthetic_waist:
            times = (k + np.arange(len(joints))) / images.fps
            joints = np.hstack([joints[:, :28], synthetic_waist(times).astype(np.float32)])
        if not np.isfinite(joints).all():
            raise ValueError(f"rejected chunk at t={t_ep:.2f}s (nonfinite)")
        if a.action_space == "sonic78":
            chunk, joints = joints, None
        else:
            width = 31 if a.action_space == "joint31" else 28
            if joints.shape[1] != width:
                raise ValueError(f"policy chunk is {joints.shape[1]}D, --action-space {a.action_space} needs {width}D")
            # backend sonic: the 30 Hz joint chunk -> tokens + hands, exactly as before (A's baseline path)
            chunk = None if a.backend == "decoupled" else joint_chunk_to_sonic(joints, *joint28, fps=images.fps)
            joints = to_joint_ref(joints)
        if chunk is not None and np.abs(chunk[:, :64]).max() > TOKEN_BOUND:
            raise ValueError(f"rejected chunk at t={t_ep:.2f}s (|token| > {TOKEN_BOUND})")
        return k, chunk, joints
```

`joint_chunk_to_sonic` receives the raw 28D chunk for joint28 (unchanged baseline) and the 31D chunk for joint31.

- [ ] **Step 9: Implement — startup.** Today `main()` runs, after the `print(f"[streamer] policy {...}")` line: (a) A's
first steps (`gate("deploy_ready")`, `time.sleep(0.5)`, `pub.send(command_message(start=True, planner=True))` + print,
`gate("settled")`, the 10 s wait for `g1_debug`), then (b) the helper definitions (`zero_h`, `frame`, `history`, `log`,
`record`/`send`/`send_joint`, `arm_watchdog`, `robot_states`, `ticks`, `infer`), then (c) A's planner handoff / table
startup / first chunk / 1 s "blend in". Reorder to: new definitions → (b) unchanged → one `if dec: ... else: (a) + (c)`.

New definitions, directly after the print line:

```python
    dec = a.backend == "decoupled"
    jres = ChunkResampler(images.fps) if a.action_space != "sonic78" else None  # 50 Hz joint_ref

    def wait_state():
        deadline = time.monotonic() + 10  # the deploy / C host publishes g1_debug once control runs
        while state.latest() is None:
            if time.monotonic() > deadline:
                raise RuntimeError("no g1_debug robot state within 10 s")
            time.sleep(0.05)

    def chunk_t0_of(k):
        """Replay chunk rows are dataset frames k, k+1, ...: their time base is frame k's time."""
        return k / images.fps
```

Then (b) unchanged, then:

```python
    if dec:
        gate("deploy_ready", a.gate_dir)
        gate("settled", a.gate_dir)
        wait_state()
        rest_ref = measured_ref(state.latest(), a.action_space)
        slot = worker.submit(infer, 0.0, robot_states(0.0))
        for _ in ticks(10**9):
            if slot["done"].is_set():
                break
            send_joint(rest_ref, "wait first chunk")
        if "error" in slot:
            raise RuntimeError(slot["error"])
        _, _, first_j = slot["result"]
        jres.set_chunk(first_j, t0=0.0)
        for i in ticks(100):  # 2 s lead-in in joint space from the measured pose
            w = (i + 1) / 100
            send_joint((1 - w) * rest_ref + w * first_j[0], "leadin")
        for _ in ticks(50):  # 1 s hold
            send_joint(first_j[0], "hold")
    else:
        # (a) A's first steps, unchanged, except the 10 s wait loop becomes wait_state()
        # (c) A's planner handoff / table startup / first chunk / blend in, unchanged except these lines:
        #       _, first, first_j = slot["result"]          (was: _, first = slot["result"])
        #       resampler.set_chunk(first, t0=0.0)
        #       if jres is not None:
        #           jres.set_chunk(first_j, t0=0.0)
```

`resampler = ChunkResampler(images.fps)` stays inside (c); C never uses it.

- [ ] **Step 10: Implement — episode loop.** Before the loop add `episode_end = "completed"`. Replace the chunk-arrival block and the three `break` sites and the final send:

```python
        if slot is not None and slot["done"].is_set():
            if "error" in slot:  # keep streaming the last good chunk; stop if it goes stale
                print(f"[streamer] {slot['error']}", flush=True)
            else:
                k_c, chunk, joints = slot["result"]
                t0 = chunk_t0_of(k_c) if replay is not None else slot_t
                if chunk is not None:
                    resampler.set_chunk(chunk, t0=t0, blend_s=a.chunk_blend_s, now=t_ep)
                if joints is not None:
                    jres.set_chunk(joints, t0=t0, blend_s=a.chunk_blend_s, now=t_ep)
                last_chunk_t, chunk_t0 = t_ep, t0
                latencies.append(slot["latency"])
            slot = None
        if t_ep - last_chunk_t > a.max_chunk_age_s:
            episode_end = f"no valid chunk for {a.max_chunk_age_s:g} s"
            print(f"[streamer] {episode_end}: ending episode", flush=True)
            break
        state.latest()
        if state.age() > a.max_state_age_s:  # never plan or stream on frozen joints
            episode_end = f"robot state {state.age():.2f} s old"
            print(f"[streamer] {episode_end}: ending episode", flush=True)
            break
        reason = arm_watchdog()
        if reason:
            episode_end = f"watchdog: {reason}"
            print(f"[streamer] {episode_end}: ending episode", flush=True)
            break
        if t_ep >= next_replan and not worker.busy:
            slot, slot_t = worker.submit(infer, t_ep, robot_states(t_ep)), t_ep
            next_replan = t_ep + a.replan_s
        k = min(int(t_ep * images.fps), images.n - 1)
        if dec:
            send_joint(jres.token_at(t_ep), "episode", t_ep, k, chunk_t0)
        else:
            out = resampler.token_at(t_ep)
            send(out[:64], out[64:], "episode", t_ep, k, chunk_t0, joint_ref=None if jres is None else jres.token_at(t_ep))
```

For policies `t0 = slot_t` exactly as before, so A's token timing is unchanged.

- [ ] **Step 11: Implement — shutdown, log, done.** After the latency print, wrap the existing A shutdown (from `print(f"[streamer] token slew limit ...")` through the final `pub.send(command_message(...))` + its print) as the `else:` of:

```python
    if dec:
        last = log["joint_ref"][-1]
        for i in ticks(100):  # 2 s back to the measured start pose
            w = (i + 1) / 100
            send_joint((1 - w) * last + w * rest_ref, "return")
    else:
        ...  # existing A shutdown, unchanged
```

Change the log save and the done gate:

```python
    if a.log:
        np.savez_compressed(
            a.log,
            **{k: np.asarray(v) for k, v in log.items()},
            episode=-1 if a.episode is None else a.episode,
            episode_end=episode_end,
            args=json.dumps(vars(a), default=str),  # the exact settings of this run
        )
    if a.gate_dir:
        write_done(a.gate_dir, episode_end)
```

Update the module docstring: add a paragraph "--backend decoupled sends 50 Hz joint targets (wbc_common.pack_joint_message) to decoupled_wbc_sim_host.py instead of tokens; --replay streams recorded actions; every joint run logs a 50 Hz joint_ref (the target after resampling, blending and lead-in), computed alongside A's unchanged token path."

- [ ] **Step 12: Run tests**

Run: `uv run python -m unittest tests/datasets/test_g1_dex3_streamer_cli.py tests/datasets/test_g1_dex3_wbc_common.py tests/datasets/test_g1_dex3_sonic_targets.py tests/datasets/test_g1_dex3_sonic_token_stream.py`
Expected: all OK. Also `uv run python examples/g1_dex3_training/sonic_policy_streamer.py --help` prints the new options.

- [ ] **Step 13: Commit**

```bash
git add examples/g1_dex3_training/sonic_policy_streamer.py tests/datasets/test_g1_dex3_streamer_cli.py
git commit -m "feat(g1-dex3): streamer --replay, joint31, synthetic waist, 50 Hz joint_ref log and --backend decoupled"
```

---

### Task 4: Pre-change baseline run (host + scorer before extraction)

Purpose: the regression reference for Task 5 (spec test 7). Uses the new streamer with the **old** host and scorer.

**Files:** none changed. Output: `$A/WBC_regress_before_r1`.

**Interfaces:**
- Consumes: Task 3 streamer (policy path unchanged), current `sonic_official_sim_host.py`, `sonic_stream_eval.py`, `sonic_official_sim_eval.sh` at `HEAD`.
- Produces: `$A/WBC_regress_before_r1/stream_eval.json` with `palm_err_vs_original_cm.p50/p95`.

- [ ] **Step 1: Ask the user which GPU to use for sim runs** (the sim host renders with EGL; GPUs 0/6 run training). Record the answer as `GPU`.

- [ ] **Step 2: Sync code** to `code_wbc_before`: the code-sync command from "Paths used throughout" with `code_wbc` replaced by `code_wbc_before`. At this point the host and scorer in the working tree are still the pre-change versions; only the streamer has changed.

- [ ] **Step 3: Confirm the HE policy and dataset paths exist**

```bash
ssh h100 'ls -d /mnt/data01/jhkim/model_weight/g1_dex3_20260922/humanoid_everyday_g1_20260923/runs/groot_sonic78sonicstate_ho5_official_full/checkpoints/last/pretrained_model /mnt/data01/jhkim/model_weight/g1_dex3_20260922/humanoid_everyday_g1_20260923/datasets/{sonic78_nolimit_g2,joint28}'
```

Expected: three paths listed.

- [ ] **Step 4: Run once with seed 0 (deterministic GR00T sampling)**

```bash
cd /home/jihun/work/lerobot/examples/g1_dex3_training
CODE_DIR=/mnt/data01/jhkim/model_weight/sonic_roundtrip_20260923/code_wbc_before \
JOINT28_DIR=/run-output/humanoid_everyday_g1_20260923/datasets/joint28 SERVER_EXTRA="--noise-seed 0" \
./sonic_official_sim_eval.sh WBC_regress_before_r1 ../humanoid_everyday_g1_20260923/runs/groot_sonic78sonicstate_ho5_official_full \
  ../humanoid_everyday_g1_20260923/datasets/sonic78_nolimit_g2 1293 $GPU 30 --start planner
```

Expected: ends with `WBC_regress_before_r1 scored` and prints p50/p95. Note both numbers.

---

### Task 5: `sim_scene.py` extraction, A host termination + fall detection + hand-gain log

**Files:**
- Create: `examples/g1_dex3_training/sim_scene.py`
- Modify (rewrite): `examples/g1_dex3_training/sonic_official_sim_host.py`

**Interfaces:**
- Consumes: Task 2 `FallDetector, read_done, write_termination`; `sonic_targets.ACTION_NAMES`.
- Produces: `sim_scene.BODY` (29 motor-order names), `sim_scene.HANDS` (14, dataset order = `ACTION_NAMES[14:]`), `class Scene(out: Path, table_gap_cm: str | None)` with attributes `sim, env, m, d, dt, pelvis, torso, qadr, hadr, lock, t, view, rec, fall` and methods `step()`, `mark(event)`, `floor_contacts() -> int`, `table_clearance() -> (float, int)`, `place_table()`, `record(**extra)`, `check_fall() -> str | None`, `pace(wall0)`, `close(reason, detail="", **fields)`. `sim_state.npz` gains `torso_R [T,3,3]`; `sim/termination.json` written by both hosts.

- [ ] **Step 1: Create `sim_scene.py`** (code moved from the host; behaviour identical, plus `torso_R`, fall detection, termination):

```python
"""Shared MuJoCo scene for the sim hosts: gear_sonic's G1 + Dex3 scene with the optional tabletop, the 50 Hz state
recording (sim_state.npz), the video, the event log, fall detection and the termination record (termination.json).

Used by sonic_official_sim_host.py (backend A: NVIDIA deploy) and decoupled_wbc_sim_host.py (backend C). Importing it
has no side effects; Scene(...) builds the simulator. Runs in jihun/sonic-vla-sim (gear_sonic under /workspace).
"""

from __future__ import annotations

import os
import sys
import threading
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import cv2  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

try:
    from .sonic_targets import ACTION_NAMES
    from .wbc_common import FallDetector, write_termination
except ImportError:
    from sonic_targets import ACTION_NAMES
    from wbc_common import FallDetector, write_termination

GEAR_ROOT = Path("/workspace/GR00T-WholeBodyControl")
BODY = (
    [f"{s}_{j}_joint" for s in ("left", "right") for j in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")]
    + ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]
    + [f"{s}_{j}_joint" for s in ("left", "right") for j in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")]
)  # fmt: skip
HANDS = list(ACTION_NAMES[14:])  # dataset order: left thumb0-2, middle0-1, index0-1; right thumb0-2, index0-1, middle0-1
FPS, W, H = 30, 640, 480
REC_KEYS = ("wall", "t", "body_q", "hand_q", "pelvis", "palm", "wrist_R", "torso_R", "floor_contacts", "table_clear",
            "table_hits")  # fmt: skip


class Scene:
    def __init__(self, out: Path, table_gap_cm: str | None):
        sys.path.insert(0, str(GEAR_ROOT))
        from gear_sonic.utils.mujoco_sim.base_sim import BaseSimulator
        from gear_sonic.utils.mujoco_sim.configs import SimLoopConfig

        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        config = SimLoopConfig(interface="sim", enable_onscreen=False, hand_profile="dex3").load_wbc_yaml()
        # Optional tabletop (TABLE_GAP_CM): 80 cm table, 3 cm top. It starts parked 10 m away (the elastic-band hang
        # and the Backspace reset would otherwise drop the hands onto it) and moves in once the robot has settled
        # standing: near edge TABLE_GAP_CM in front of the torso front (pelvis x + 0.08 m). Written next to the
        # original scene inside this container so its mesh paths resolve.
        self.table_gap = float(table_gap_cm) if table_gap_cm else None
        if self.table_gap is not None:
            scene = GEAR_ROOT / config["ROBOT_SCENE"]
            xml = scene.read_text().replace(
                "</mujoco>",
                '<worldbody><body name="table" pos="10 0 0.785"><geom name="table_top" type="box" '
                'size="0.4 0.8 0.015" rgba="0.62 0.46 0.3 1"/></body></worldbody></mujoco>',
            )
            (scene.parent / "scene_43dof_table.xml").write_text(xml)
            config["ROBOT_SCENE"] = str((scene.parent / "scene_43dof_table.xml").relative_to(GEAR_ROOT))
        self.config = config
        self.sim = BaseSimulator(config=config, onscreen=False, offscreen=False, env_name="default")
        self.env = self.sim.sim_env
        m, d = self.m, self.d = self.env.mj_model, self.env.mj_data
        self.dt = config["SIMULATE_DT"]
        self.pelvis, self.torso = m.body("pelvis").id, m.body("torso_link").id
        self.floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self.qadr = np.array([m.jnt_qposadr[m.joint(n).id] for n in BODY])
        self.hadr = np.array([m.jnt_qposadr[m.joint(n).id] for n in HANDS])
        self.palm_ids = [[m.body(f"{s}_hand_{f}_0_link").id for f in ("index", "middle")] for s in ("left", "right")]
        self.wrist_ids = [m.body(f"{s}_wrist_yaw_link").id for s in ("left", "right")]
        self.table = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "table_top")
        self.arm_geoms = [
            g
            for g in range(m.ngeom)
            if (m.geom_contype[g] or m.geom_conaffinity[g])
            and any(k in m.body(m.geom_bodyid[g]).name for k in ("elbow", "wrist", "hand", "shoulder_yaw"))
        ]
        self.lock = threading.Lock()
        self.t = 0.0
        self.view = {"phase": "hanging on elastic band", "stop": False}
        self.rec = {k: [] for k in REC_KEYS}
        self.next_rec = 0.0
        self.fall = FallDetector()
        self.writer = cv2.VideoWriter(str(self.out / "sim_raw.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
        threading.Thread(target=self._render_loop, daemon=True).start()

    def step(self) -> None:
        with self.lock:
            self.env.sim_step()
            self.t += self.dt

    def mark(self, event: str) -> None:
        print(f"[{self.t:8.3f}] {event}", flush=True)
        with open(self.out / "events.txt", "a") as f:
            f.write(f"{time.time():.3f} {self.t:.3f} {event}\n")

    def floor_contacts(self) -> int:
        d = self.d
        return sum(1 for i in range(d.ncon) if self.floor in (d.contact[i].geom1, d.contact[i].geom2))

    def table_clearance(self):
        """Closest arm collision geom to the table (m) and arm/hand-table contact count; nan/0 without a table."""
        if self.table < 0:
            return np.nan, 0
        ft = np.zeros(6)
        clear = min(mujoco.mj_geomDistance(self.m, self.d, g, self.table, 0.5, ft) for g in self.arm_geoms)
        hits = sum(1 for i in range(self.d.ncon) if self.table in (self.d.contact[i].geom1, self.d.contact[i].geom2))
        return clear, hits

    def place_table(self) -> None:
        if self.table < 0:
            return
        with self.lock:
            edge = self.d.qpos[0] + 0.08 + self.table_gap / 100
            self.m.body_pos[self.m.geom_bodyid[self.table]] = [edge + 0.4, self.d.qpos[1], 0.785]
            mujoco.mj_forward(self.m, self.d)
        self.mark(f"table placed: near edge {self.table_gap:g} cm from the torso (x = {edge:.3f} m)")

    def record(self, **extra) -> None:
        """One row every 20 ms of sim time (call after settling); extra adds host-specific keys."""
        if self.t + 1e-9 < self.next_rec:
            return
        self.next_rec = self.t + 0.02
        d = self.d
        with self.lock:
            clear, hits = self.table_clearance()
            row = {
                "wall": time.time(),
                "t": self.t,
                "body_q": d.qpos[self.qadr].copy(),
                "hand_q": d.qpos[self.hadr].copy(),
                "pelvis": d.qpos[:7].copy(),
                "palm": np.stack([d.xpos[ids].mean(0) for ids in self.palm_ids]),
                "wrist_R": np.stack([d.xmat[i].reshape(3, 3) for i in self.wrist_ids]),
                "torso_R": d.xmat[self.torso].reshape(3, 3).copy(),
                "floor_contacts": self.floor_contacts(),
                "table_clear": clear,
                "table_hits": hits,
            }
        row.update(extra)
        for k, v in row.items():
            self.rec.setdefault(k, []).append(v)

    def check_fall(self) -> str | None:
        return self.fall.update(self.t, self.d.qpos[3:7], float(self.d.qpos[2]), self.floor_contacts())

    def pace(self, wall0: float) -> None:
        lag = self.t - (time.monotonic() - wall0)
        if lag > 0:
            time.sleep(lag)

    def close(self, reason: str, detail: str = "", **fields) -> None:
        self.view["stop"] = True
        time.sleep(0.3)
        self.writer.release()
        if self.rec["t"]:
            np.savez_compressed(self.out / "sim_state.npz", **{k: np.array(v) for k, v in self.rec.items()})
        write_termination(self.out, reason, self.view["phase"], detail, **fields)
        self.mark(f"termination: {reason} {detail}")

    def _render_loop(self) -> None:
        m, d = self.m, self.d
        snap = mujoco.MjData(m)
        cam = mujoco.MjvCamera()
        cam.type, cam.trackbodyid, cam.distance, cam.azimuth, cam.elevation = (
            mujoco.mjtCamera.mjCAMERA_TRACKING, self.pelvis, 2.4, 150, -15,
        )  # fmt: skip
        renderer = mujoco.Renderer(m, H, W)
        next_t = 0.0
        while not self.view["stop"]:
            with self.lock:
                t, ready = self.t, self.t >= next_t
                if ready:
                    snap.qpos[:], snap.qvel[:] = d.qpos, d.qvel
            if not ready:
                time.sleep(0.002)
                continue
            next_t += 1 / FPS
            mujoco.mj_forward(m, snap)
            renderer.update_scene(snap, camera=cam)
            img = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
            for txt, y in ((f"t = {t:6.2f} s   {self.view['phase']}", 24), (f"pelvis z {snap.qpos[2]:.3f} m", 46)):
                cv2.putText(img, txt, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
                cv2.putText(img, txt, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            self.writer.write(img)
```

Before writing, confirm `HANDS` equals the old host's list (`left thumb_0..2, middle_0..1, index_0..1; right thumb_0..2, index_0..1, middle_0..1`) — it does, since `ACTION_NAMES[14:]` lists exactly those.

- [ ] **Step 2: Rewrite `sonic_official_sim_host.py`** to use `Scene` (deploy-specific code stays):

```python
"""Sim host for sonic_policy_streamer.py --backend sonic: gear_sonic MuJoCo + NVIDIA g1_deploy_onnx_ref, recorded.

The streamer owns every deploy command (ZMQ "command"/"pose" on 5556). This process runs the physics (sim_scene.Scene),
performs the confirmed operator steps that belong to the simulator, and exchanges gate files with the streamer:
  deploy prints "Init Done"               -> write GATE/deploy_ready (streamer sends start + planner)
  deploy enters CONTROL (+4 s, still hung) -> sim key "9" (release band), +1 s sim key "Backspace" (reset)
  robot settled for 5 s                   -> write GATE/settled (streamer hands off to POSE and streams); fall
                                             detection armed
  streamer writes GATE/done               -> stop 5 s later; termination.json says completed only if the streamer's
                                             episode completed
Logs robot state every 20 ms with wall-clock time, the Dex3 gains the deploy sends (events.txt), and renders a video.
Usage (in jihun/sonic-vla-sim, cwd gear_sonic_deploy): python sonic_official_sim_host.py OUT_DIR GATE_DIR
"""

import contextlib
import os
import pty
import subprocess
import sys
import threading
import time
from pathlib import Path

from sim_scene import Scene
from wbc_common import read_done

OUT, GATE = Path(sys.argv[1]), Path(sys.argv[2])
GATE.mkdir(parents=True, exist_ok=True)
DEPLOY = [
    "bash",
    "-lc",
    "source scripts/setup_env.sh >/dev/null 2>&1; exec target/release/g1_deploy_onnx_ref lo "
    "policy/sonic_v1_1/model_decoder.onnx reference/example/ "
    "--obs-config policy/sonic_v1_1/observation_config.yaml --encoder-file policy/sonic_v1_1/model_encoder.onnx "
    "--planner-file planner/target_vel/V2/planner_sonic.onnx --input-type zmq_manager --output-type zmq "
    "--zmq-host localhost --disable-crc-check",
]
scene = Scene(OUT, os.environ.get("TABLE_GAP_CM"))
master, slave = pty.openpty()
proc = subprocess.Popen(DEPLOY, stdin=slave, stdout=slave, stderr=slave, close_fds=True)
os.close(slave)
lines = []
log_f = open(OUT / "deploy.log", "w")  # noqa: SIM115 - written by the reader thread for the whole run


def reader():
    buf = b""
    while True:
        try:
            chunk = os.read(master, 4096)
        except OSError:
            return
        if not chunk:
            return
        buf += chunk
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            text = line.decode(errors="replace").rstrip("\r")
            lines.append(text)
            log_f.write(f"[{scene.t:8.3f}] {text}\n")
            log_f.flush()


threading.Thread(target=reader, daemon=True).start()


def seen(pattern):
    return any(pattern in text for text in lines)


def log_hand_gains(when):
    """The Dex3 kp/kd the deploy's last hand commands carry (spec test 8: expected 1.5 / 0.1 on all 14 motors)."""
    br = scene.env.unitree_bridge
    cmds = [c for cmd in (br.left_hand_cmd, br.right_hand_cmd) for c in cmd.motor_cmd]
    kp, kd = sorted({round(float(c.kp), 3) for c in cmds}), sorted({round(float(c.kd), 3) for c in cmds})
    scene.mark(f"hand gains {when}: kp {kp} kd {kd}")


steps, t_control, t_settled, t_done, episode_end = {}, None, None, None, None
reason, detail = "error", "deploy exited before the run finished"
scene.mark("sim + deploy started")
wall0 = time.monotonic()
try:
    while proc.poll() is None:
        scene.step()
        t = scene.t
        if "ready" not in steps and seen("Init Done"):
            steps["ready"] = t
            (GATE / "deploy_ready").touch()
            scene.mark("Init Done -> deploy_ready")
            scene.view["phase"] = "Init Done: waiting for the streamer's start command"
            log_hand_gains("after Init Done")
        if t_control is None and seen("transitioning to CONTROL state"):
            t_control = t
            scene.mark("deploy in CONTROL (planner mode)")
            scene.view["phase"] = "planner mode, hung"
        if t_control is not None and "band" not in steps and t >= t_control + 4:
            steps["band"] = t
            scene.sim.handle_keyboard_button("9")
            scene.mark("sim key '9': band released")
            scene.view["phase"] = "band released"
        if "band" in steps and "reset" not in steps and t >= steps["band"] + 1:
            steps["reset"] = t
            scene.sim.handle_keyboard_button("backspace")
            scene.mark("sim key 'Backspace': reset onto ground")
            scene.view["phase"] = "settling on ground"
        if "reset" in steps and t_settled is None and t >= steps["reset"] + 5:
            t_settled = t
            scene.place_table()  # bring the table in now that the robot stands in planner mode
            scene.fall.armed = True
            (GATE / "settled").touch()
            scene.mark("settled -> streamer may hand off to POSE")
            scene.view["phase"] = "policy streaming (POSE mode)"
        if t_settled is not None and "gains_streaming" not in steps and t >= t_settled + 8:
            steps["gains_streaming"] = t
            log_hand_gains("while streaming")
        if t_done is None and (GATE / "done").exists():
            t_done, episode_end = t, read_done(GATE)
            scene.mark(f"streamer done: {episode_end}")
            scene.view["phase"] = "streamer done"
        if t_done is not None and t >= t_done + 5:  # keep 5 s of the planner hand-back on record
            reason, detail = ("completed", "") if episode_end == "completed" else ("aborted", f"streamer: {episode_end}")
            break
        if t_settled is not None:
            scene.record()
            fell = scene.check_fall()
            if fell:
                reason, detail = "fell", fell
                break
        if t > 1800:
            reason, detail = "timeout", "1800 s of sim time"
            break
        scene.pace(wall0)
except Exception as e:
    reason, detail = "error", repr(e)
    raise
finally:
    scene.mark("stop" if proc.poll() is None else f"deploy exited with code {proc.returncode}")
    with contextlib.suppress(OSError):
        os.write(master, b"o")
    time.sleep(0.5)
    proc.terminate()
    scene.close(reason, detail)
```

(The `+ 8 s` gain check lands after the streamer's 1 s planner token + 2 s handoff + episode start; the deploy only sends hand commands once it runs.)

- [ ] **Step 3: Run local unit tests** (nothing local imports MuJoCo, they must still pass)

Run: `uv run python -m unittest tests/datasets/test_g1_dex3_wbc_common.py tests/datasets/test_g1_dex3_streamer_cli.py`
Expected: OK.

- [ ] **Step 4: Regression run (spec test 7) and hand gains (spec test 8).** Sync code to `code_wbc` (command in "Paths used throughout"), then run the Task 4 command with `CODE_DIR=.../code_wbc` and `OUT=WBC_regress_after_r1`, and again as `WBC_regress_after_r2`. The *old* scorer is still in `code_wbc` (Task 6 replaces it), so the numbers are comparable.

Expected:
- both runs `scored`; `sim/termination.json` has `"reason": "completed"`;
- |p50(after_r1) − p50(before_r1)| ≤ max(0.5, |p50(after_r1) − p50(after_r2)|) and the same for p95 with 1.0 cm;
- `grep "hand gains while streaming" $A/WBC_regress_after_r1/sim/events.txt` shows `kp [1.5] kd [0.1]`.

If the gains differ, stop: update `HAND_KP/HAND_KD` in Task 7 to the logged values and tell the user. If the regression fails, diff the two hosts' `events.txt` timelines before anything else.

Also on the H100: `sonic_joint28_stream_check.py` still passes (spec test 7, second half) — run it exactly as its docstring says, in the H100 prep container with `/run-output` and `/sonic-model` mounted, using the synced `code_wbc` copy; expected: whole episodes 100% identical, 40+ frame chunks ≥ 99.9%.

- [ ] **Step 5: Commit**

```bash
git add examples/g1_dex3_training/sim_scene.py examples/g1_dex3_training/sonic_official_sim_host.py
git commit -m "refactor(g1-dex3): shared sim_scene for the sim hosts; A host writes termination.json, detects falls, logs Dex3 gains"
```

---

### Task 6: Backend-aware, validity-gated scorer

**Files:**
- Modify (rewrite): `examples/g1_dex3_training/sonic_stream_eval.py`

**Interfaces:**
- Consumes: Task 2 `run_validity, log_settings, rpy_from_matrix`; Task 3 log keys (`joint_ref`, `args`); Task 5 `sim_state.npz` keys (`torso_R`) + `sim/termination.json`; Task 7 C keys (`applied_ref`, `held`, `rpy_cmd`, `clipped`) and termination `counts`.
- Produces `stream_eval.json` keys (used by Task 8 `wbc_compare.py`): `valid`, `reason` (if invalid), `backend`, `action_space`, `episode`, `task`, `frames_scored`, `episode_frames`, `stability{tilt_max_deg, floor_contacts_min, leg_dev_max_rad, pelvis_z_min}`, `palm_err_vs_original_cm{p50,p95,max}` (original = recorded action), `palm_err_p95_cm_left_right`, `palm_err_vs_joint_ref_cm` (or null), `palm_err_vs_applied_cm` (or null), `wrist_orientation_err_deg`, `table{hits_max, hit_records, clear_min_cm}`, `smoothness{arm_speed_p95, arm_jerk_p95, palm_jerk_p95}`, `torso_err_rad{roll,pitch,yaw}` (or null), `rpy_cmd_err_rad` (or null), `c_counts` (or null), `policy_token_vs_stored_abs` / `policy_hands_vs_stored_abs_rad` (sonic only), `gate_5cm_p95`. Exit code 2 when invalid.

- [ ] **Step 1: Rewrite the scorer**:

```python
"""Score a sonic_policy_streamer.py run recorded by a sim host (sonic_official_sim_host.py or
decoupled_wbc_sim_host.py).

Validity first (wbc_common.run_validity, same rule for both backends): runs that did not complete, cover < 98% of the
episode or (backend C) held their targets on > 2% of records are reported as {"valid": false} and exit 2.
Then, during the "episode" phase: stability; palm error in the pelvis frame against FK of the recorded action
("original"), of the streamer's 50 Hz joint_ref (controller error; joint runs only) and, for backend C, of the applied
target after clipping/hold; wrist orientation; table contacts; arm speed / jerk and palm jerk; for 31D runs the torso
orientation reached vs commanded and (C) the lower-body policy's rpy command vs FK of the commanded waist; for token
runs the tokens/hands vs the dataset's stored ones. A-native runs (token policies, stored tokens) have no joint_ref.
Usage: python sonic_stream_eval.py RUN_DIR JOINT28_ROOT SONIC_ROOT EPISODE
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/code")
from robot_run_smoothness import windowed_diff
from sonic_targets import NOMINAL_BODY
from wbc_common import log_settings, rpy_from_matrix, run_validity


def quat_to_rot(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])  # fmt: skip


def stats(x):
    x = np.asarray(x, float).ravel()
    return {"p50": round(float(np.median(x)), 3), "p95": round(float(np.percentile(x, 95)), 3), "max": round(float(x.max()), 3)}  # fmt: skip


def cm(x):
    return {k: round(v * 100, 2) for k, v in stats(x).items()}


def body_from(arms, waist):
    b = NOMINAL_BODY.copy()
    b[15:29], b[12:15] = arms, waist
    return b


def score(run: Path, src: str, conv: str, ep: int) -> dict:
    from sonic_roundtrip_audit import Robot, v3_episode, v3_meta

    s = np.load(run / "streamer.npz", allow_pickle=True)
    sim = np.load(run / "sim" / "sim_state.npz")
    term_path = run / "sim" / "termination.json"
    term = json.loads(term_path.read_text()) if term_path.exists() else None
    backend, action_space = log_settings(str(s["args"]))
    rows, info = v3_meta(src)
    row = next(r for r in rows if r["episode_index"] == ep)
    orig = v3_episode(src, row, info, ["frame_index", "action"])["action"].astype(np.float64)
    ep_all = np.flatnonzero(s["phase"] == "episode")
    win = np.zeros(len(sim["wall"]), bool)
    if len(ep_all):
        win = (sim["wall"] >= s["wall"][ep_all[0]]) & (sim["wall"] <= s["wall"][ep_all[-1]])
    held = float(np.mean(sim["held"][win])) if backend == "decoupled" and "held" in sim.files and win.any() else 0.0
    valid, reason = run_validity(term, s["phase"], s["frame"], s["wall"], sim["wall"], len(orig), held)
    head = {"episode": ep, "backend": backend, "action_space": action_space}
    if not valid:
        return {**head, "valid": False, "reason": reason}

    robot = Robot()
    torso_id = robot.m.body("torso_link").id
    wrist_ids = [robot.m.body(n).id for n in ("left_wrist_yaw_link", "right_wrist_yaw_link")]
    sim_idx = np.clip(np.searchsorted(sim["wall"], s["wall"][ep_all]), 0, len(sim["wall"]) - 1)
    frames = s["frame"][ep_all]
    first = np.r_[True, np.diff(frames) != 0]  # first streamer tick of each source frame
    ep_idx, sim_idx, frames = ep_all[first], sim_idx[first], frames[first]
    jref = s["joint_ref"][ep_idx] if "joint_ref" in s.files else None
    has_ref = jref is not None and np.isfinite(jref).all()
    waist31 = action_space == "joint31" and has_ref
    applied = sim["applied_ref"] if "applied_ref" in sim.files else None
    nominal_waist = NOMINAL_BODY[12:15]

    palm_orig, palm_ref, palm_app, orient_err, torso_err, rpy_err, waist_cmd = [], [], [], [], [], [], []
    for i, (k, j) in enumerate(zip(frames, sim_idx, strict=True)):
        rot = quat_to_rot(sim["pelvis"][j, 3:7])
        reached = (sim["palm"][j] - sim["pelvis"][j, :3]) @ rot  # pelvis frame
        waist = jref[i, 28:31] if waist31 else nominal_waist
        palm_orig.append(np.linalg.norm(reached - robot.fk_pelvis_frame(body_from(orig[k, :14], waist)), axis=1))
        for h, wid in enumerate(wrist_ids):  # robot.kin is at the recorded-action pose now
            r_reached = rot.T @ sim["wrist_R"][j, h]
            r_int = robot.kin.xmat[wid].reshape(3, 3)
            orient_err.append(np.degrees(np.arccos(np.clip((np.trace(r_reached.T @ r_int) - 1) / 2, -1, 1))))
        if has_ref:
            palm_ref.append(np.linalg.norm(reached - robot.fk_pelvis_frame(body_from(jref[i, :14], waist)), axis=1))
            if waist31:  # torso orientation relative to the yaw-only pelvis frame, as upstream defines the command
                cmd_rpy = rpy_from_matrix(robot.kin.xmat[torso_id].reshape(3, 3))
                yaw = np.arctan2(rot[1, 0], rot[0, 0])
                rz = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
                torso_err.append(np.abs(rpy_from_matrix(rz.T @ sim["torso_R"][j]) - cmd_rpy))
                waist_cmd.append(np.abs(jref[i, 28:31]).max())
                if "rpy_cmd" in sim.files:
                    rpy_err.append(np.abs(sim["rpy_cmd"][j] - cmd_rpy))
        if applied is not None:
            ap = applied[j]
            ap_waist = ap[28:31] if waist31 else nominal_waist
            palm_app.append(np.linalg.norm(reached - robot.fk_pelvis_frame(body_from(ap[:14], ap_waist)), axis=1))
    palm_orig = np.array(palm_orig)
    if rpy_err and max(waist_cmd) > 0.05 and np.abs(sim["rpy_cmd"][win]).max() < 1e-3:
        return {**head, "valid": False, "reason": "waist commanded but the lower-body policy's rpy command stayed 0"}

    q = sim["pelvis"][win, 3:7]
    tilt = np.degrees(np.arccos(np.clip(1 - 2 * (q[:, 1] ** 2 + q[:, 2] ** 2), -1, 1)))
    arm, palm = sim["body_q"][win, 15:29], sim["palm"][win]
    jerk = np.abs(windowed_diff(windowed_diff(windowed_diff(arm)))).max(1)
    palm_jerk = np.linalg.norm(windowed_diff(windowed_diff(windowed_diff(palm))), axis=-1).max(1)
    out = {
        **head, "valid": True, "task": row["tasks"][0], "frames_scored": int(len(frames)), "episode_frames": int(len(orig)),
        "stability": {"tilt_max_deg": round(float(tilt.max()), 2), "floor_contacts_min": int(sim["floor_contacts"][win].min()),
                      "leg_dev_max_rad": round(float(np.abs(sim["body_q"][win, :12] - NOMINAL_BODY[:12]).max()), 3),
                      "pelvis_z_min": round(float(sim["pelvis"][win, 2].min()), 3)},
        "palm_err_vs_original_cm": cm(palm_orig),
        "palm_err_p95_cm_left_right": [round(float(np.percentile(palm_orig[:, h], 95)) * 100, 2) for h in (0, 1)],
        "palm_err_vs_joint_ref_cm": cm(palm_ref) if palm_ref else None,
        "palm_err_vs_applied_cm": cm(palm_app) if palm_app else None,
        "wrist_orientation_err_deg": stats(orient_err),
        "table": {"hits_max": int(sim["table_hits"][win].max()), "hit_records": int((sim["table_hits"][win] > 0).sum()),
                  "clear_min_cm": round(float(np.nanmin(sim["table_clear"][win])) * 100, 2) if np.isfinite(sim["table_clear"][win]).any() else None},
        "smoothness": {"arm_speed_p95": round(float(np.percentile(np.abs(np.diff(arm, axis=0)).max(1) / 0.02, 95)), 3),
                       "arm_jerk_p95": round(float(np.percentile(jerk, 95)), 1),
                       "palm_jerk_p95": round(float(np.percentile(palm_jerk, 95)), 1)},
        "torso_err_rad": {a: stats(np.array(torso_err)[:, n]) for n, a in enumerate(("roll", "pitch", "yaw"))} if torso_err else None,
        "rpy_cmd_err_rad": stats(rpy_err) if rpy_err else None,
        "c_counts": term.get("counts") if backend == "decoupled" else None,
        "gate_5cm_p95": "PASS" if np.percentile(palm_orig, 95) <= 0.05 else "FAIL",
    }  # fmt: skip
    if backend == "sonic":
        stored = v3_episode(conv, row, info, ["frame_index", "action"])["action"].astype(np.float64)
        if stored.shape[1] == 78:
            out["policy_token_vs_stored_abs"] = stats(np.abs(s["token"][ep_idx] - stored[frames, :64]))
            out["policy_hands_vs_stored_abs_rad"] = stats(np.abs(s["hands"][ep_idx] - stored[frames, 64:]))
    return out


def main():
    run, src, conv, ep = Path(sys.argv[1]), sys.argv[2], sys.argv[3], int(sys.argv[4])
    out = score(run, src, conv, ep)
    (run / "stream_eval.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
    sys.exit(0 if out["valid"] else 2)


if __name__ == "__main__":
    main()
```

Notes for the implementer: `leg_dev_max_rad` now covers legs only (`:12`); the old code included the waist (`:15`), which a 31D run moves on purpose. `robot.kin` after the joint_ref FK is at the commanded pose, so the torso rpy read right after it is the commanded one — keep that order.

- [ ] **Step 2: Re-score the Task 5 runs with the new scorer** (they have termination.json): sync code to `code_wbc`, then

```bash
ssh h100 "docker run --rm -e PYTHONPATH=/audit/pylib -e MUJOCO_GL=disable -v /mnt/data01/jhkim/model_weight/sonic_roundtrip_20260923:/audit -v /mnt/data01/jhkim/model_weight/sonic_roundtrip_20260923/code_wbc:/code:ro -v /mnt/data01/jhkim/model_weight/g1_dex3_20260922:/run-output:ro -v /home/kube/sonic_vla_integration/20260906/source/gear_sonic:/gear_sonic:ro -v /home/kube/sonic_vla_integration/20260906/source/gear_sonic_deploy:/gear_sonic_deploy:ro -v /home/kube/sonic_vla_integration/20260906/models/sonic/sonic_v1_1:/sonic-model:ro --entrypoint /run-output/environment/venv/bin/python 4cbe2a3f7fc6 /code/sonic_stream_eval.py /audit/WBC_regress_after_r1 /run-output/humanoid_everyday_g1_20260923/datasets/joint28 /run-output/humanoid_everyday_g1_20260923/datasets/sonic78_nolimit_g2 1293"
```

Expected: `"valid": true`, `palm_err_vs_original_cm` equal (±0.01) to the old scorer's numbers for the same run, `palm_err_vs_joint_ref_cm: null` (token policy), `torso_err_rad: null`, token metrics present.

- [ ] **Step 3: Invalid-run check (spec test 6).** Copy the run to `WBC_invalid_fell` and `WBC_invalid_tail` (through a container), then: in `WBC_invalid_fell/sim/termination.json` set `"reason": "fell"`; in `WBC_invalid_tail` truncate `sim_state.npz` to its first half (`python -c` in the LR container: load, slice every array `[:len//2]`, save). Score both with the Step 2 command.

Expected: both print `"valid": false` with reasons `termination: fell (...)` and `streamer episode ticks fall outside the sim recording`; exit code 2.

- [ ] **Step 4: Commit**

```bash
git add examples/g1_dex3_training/sonic_stream_eval.py
git commit -m "feat(g1-dex3): scorer rejects invalid runs, scores joint_ref/applied/recorded palm error, torso, table, jerk"
```

---

### Task 7: Backend C host, launcher `BACKEND`/`REPLAY`, self-checks

**Files:**
- Create: `examples/g1_dex3_training/decoupled_wbc_sim_host.py`
- Modify: `examples/g1_dex3_training/sonic_official_sim_eval.sh`

**Interfaces:**
- Consumes: Task 5 `Scene, BODY, HANDS`; Task 2 `REF_NAMES, WAIST_NAMES, unpack_joint_message, pack_joint_message, read_done, rpy_from_matrix`; Task 3 streamer `--backend decoupled`.
- Produces: host CLI `decoupled_wbc_sim_host.py OUT GATE [--waist-location lower_body|lower_and_upper_body] [--action-port 5556] [--state-port 5557] [--selfcheck contract|state|command|waist|stand]`; `sim_state.npz` extra keys `applied_ref[T,31]`, `clipped[T,31]`, `held[T]`, `seq[T]`, `rpy_cmd[T,3]`; termination `counts{messages, stale, nonfinite, seq_gap, clipped_values}`. Launcher env `BACKEND`, `HOST_ARGS`, `REPLAY`, `SONIC_DIR`.

- [ ] **Step 1: Prepare upstream and onnxruntime on the H100** (no host installs; everything in `$A`, written through containers)

```bash
ssh h100 'docker run --rm -v /mnt/data01/jhkim/model_weight/sonic_roundtrip_20260923:/a --entrypoint sh 4cbe2a3f7fc6 -c "
set -e; cd /a; C=b042411fae38ee4d1af9aac82a37a1f8d14d6dd0
rm -rf upstream_b042411fae && mkdir upstream_b042411fae && cd upstream_b042411fae
curl -sSL https://codeload.github.com/NVlabs/GR00T-WholeBodyControl/tar.gz/\$C | tar xz --strip-components=1 GR00T-WholeBodyControl-\$C/decoupled_wbc
for f in Balance Walk; do p=decoupled_wbc/sim2mujoco/resources/robots/g1/policy/GR00T-WholeBodyControl-\$f.onnx
  curl -sSL -o \$p https://media.githubusercontent.com/media/NVlabs/GR00T-WholeBodyControl/\$C/\$p; done
sha256sum decoupled_wbc/sim2mujoco/resources/robots/g1/policy/*.onnx | tee ONNX_SHA256.txt
head -c 200 decoupled_wbc/sim2mujoco/resources/robots/g1/policy/GR00T-WholeBodyControl-Balance.onnx | od -c | head -2"'
```

Expected: two SHA-256 lines; the `od` output is binary protobuf, **not** `version https://git-lfs` text (if it is an LFS pointer, the media URL failed — stop and report).

```bash
ssh h100 'docker run --rm -v /mnt/data01/jhkim/model_weight/sonic_roundtrip_20260923:/a --entrypoint bash jihun/sonic-vla-sim:startupfix-20260907 -c "
set -e; /opt/sonic-sim/bin/python -m pip install --no-cache-dir --target /a/pylib_ort310 onnxruntime==1.20.1
rm -rf /a/pylib_ort310/numpy /a/pylib_ort310/numpy-* /a/pylib_ort310/bin
PYTHONPATH=/a/pylib_ort310 /opt/sonic-sim/bin/python -c \"import onnxruntime, numpy; print(onnxruntime.__version__, numpy.__version__)\""'
```

Expected: `1.20.1 <container numpy version>`. (`pip --target` writes only into `$A/pylib_ort310`; the container's own numpy is kept.)

- [ ] **Step 2: Write `decoupled_wbc_sim_host.py`**:

```python
"""Sim host for sonic_policy_streamer.py --backend decoupled: NVIDIA's GR00T decoupled WBC (lower-body RL for legs +
waist, arm and Dex3 hand joint targets straight to PD) in the same scene as sonic_official_sim_host.py (sim_scene).

Contract (docs/superpowers/specs/2026-10-01-g1-wbc-backends-design.md; upstream NVlabs/GR00T-WholeBodyControl
b042411fae with only decoupled_wbc/ mounted at /upstream/decoupled_wbc; onnxruntime on PYTHONPATH):
  lower body  G1GearWbcPolicy with GEAR_WBC_CONFIG of control/main/teleop/configs/g1_29dof_gear_wbc.yaml (516 obs,
              15 actions) and the Balance/Walk ONNX; walking command 0, height 0.74 (Balance runs throughout)
  upper body  IdentityPolicy: the streamer's 50 Hz targets as they are; every wrapper goal is complete
  waist       robot model waist_location lower_body (28D: torso command 0) or lower_and_upper_body (31D: torso
              roll/pitch/yaw from FK of the commanded waist, RL moves waist + legs to follow)
  activation  RL output switched on through the lower-body policy (key "]"), never a toggle-only wrapper goal
  gains       body MOTOR_KP / MOTOR_KD of g1_29dof_gear_wbc.yaml; Dex3 kp 1.5 / kd 0.1 (A's deploy defaults,
              dex3_hands.hpp), written into gear_sonic's bridge command slots by joint name (no DDS, no slot swap)
  rates       physics 200 Hz (A's scene; upstream sim_frequency 200), control 50 Hz
  messages    ZMQ SUB "joints" (wbc_common.pack_joint_message) on --action-port; drops messages older than 0.1 s,
              holds the previous target on stale or non-finite ones, clips to the MuJoCo joint ranges (counted);
              ZMQ PUB "g1_debug" (body_q motor order, left/right_hand_q dataset order) on --state-port
Timeline: band on, RL off -> 1 s RL on -> 3 s band released -> 4 s reset onto the ground -> 9 s policy flags checked,
table placed, GATE/deploy_ready + GATE/settled, fall detection armed -> GATE/done + 3 s: stop. Streamer silent for
5 s without GATE/done: aborted.
Usage (in jihun/sonic-vla-sim, cwd gear_sonic_deploy):
  python decoupled_wbc_sim_host.py OUT_DIR GATE_DIR [--waist-location ...] [--selfcheck contract|state|command|waist|stand]
"""

from __future__ import annotations

import argparse
import hashlib
import os
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import msgpack
import mujoco
import numpy as np
import yaml
import zmq

from sim_scene import BODY, HANDS, Scene
from wbc_common import REF_NAMES, WAIST_NAMES, pack_joint_message, read_done, rpy_from_matrix, tilt_deg, unpack_joint_message

UPSTREAM = Path("/upstream/decoupled_wbc")
WBC_YAML = UPSTREAM / "control/main/teleop/configs/g1_29dof_gear_wbc.yaml"
POLICY_DIR = UPSTREAM / "sim2mujoco/resources/robots/g1"
MODEL_PATH = "policy/GR00T-WholeBodyControl-Balance.onnx,policy/GR00T-WholeBodyControl-Walk.onnx"
HAND_KP, HAND_KD = 1.5, 0.1  # Dex3Hands defaults in A's deploy (dex3_hands.hpp:396-397), kept by setAllJointsCommand
DECIMATION = 4  # 200 Hz physics -> 50 Hz control
MAX_MSG_AGE_S, SILENCE_S = 0.1, 5.0
GOAL_CONST = {"base_height_command": np.array([0.74]), "navigate_cmd": np.zeros(3)}
OBS_DIM = 86  # one frame of the lower-body observation (516 = 86 x 6)
T_ACTIVATE, T_BAND, T_RESET, T_SETTLED = 1.0, 3.0, 4.0, 9.0


def make_wbc(waist_location: str):
    from decoupled_wbc.control.policy.g1_decoupled_whole_body_policy import G1DecoupledWholeBodyPolicy
    from decoupled_wbc.control.policy.g1_gear_wbc_policy import G1GearWbcPolicy
    from decoupled_wbc.control.policy.identity_policy import IdentityPolicy
    from decoupled_wbc.control.robot_model.instantiation.g1 import instantiate_g1_robot_model

    cfg = yaml.safe_load(WBC_YAML.read_text())
    robot_model = instantiate_g1_robot_model(waist_location=waist_location)
    lower = G1GearWbcPolicy(
        robot_model=robot_model, config=str(UPSTREAM.parent / cfg["GEAR_WBC_CONFIG"]), model_path=MODEL_PATH
    )
    if (lower.config["num_obs"], lower.config["num_actions"]) != (516, 15):
        raise RuntimeError(f"unexpected lower-body config {lower.config['num_obs']} obs / {lower.config['num_actions']}")
    wbc = G1DecoupledWholeBodyPolicy(robot_model=robot_model, upper_body_policy=IdentityPolicy(), lower_body_policy=lower)
    return wbc, robot_model, np.asarray(cfg["MOTOR_KP"], float), np.asarray(cfg["MOTOR_KD"], float)


class Plant:
    """PD targets by joint name, written into gear_sonic's bridge command slots (its sim_step computes the torques)."""

    def __init__(self, scene: Scene, kp: np.ndarray, kd: np.ndarray):
        env, m = scene.env, scene.m
        names = lambda idx: [m.joint(int(j)).name for j in idx]  # noqa: E731 - bridge slots hold 1-based joint ids
        self.body_slots = names(env.body_joint_index)
        self.left_slots, self.right_slots = names(env.left_hand_index), names(env.right_hand_index)
        if sorted(self.body_slots) != sorted(BODY) or sorted(self.left_slots + self.right_slots) != sorted(HANDS):
            raise RuntimeError(f"bridge slots {self.body_slots} / {self.left_slots + self.right_slots} do not match the scene")

        def cmd(n):
            return SimpleNamespace(motor_cmd=[SimpleNamespace(q=0.0, dq=0.0, kp=0.0, kd=0.0, tau=0.0) for _ in range(n)])

        self.br = env.unitree_bridge
        self.br.low_cmd, self.br.left_hand_cmd, self.br.right_hand_cmd = cmd(len(self.body_slots)), cmd(7), cmd(7)
        for i, n in enumerate(self.body_slots):
            c = self.br.low_cmd.motor_cmd[i]
            c.kp, c.kd = kp[BODY.index(n)], kd[BODY.index(n)]
        for c in self.br.left_hand_cmd.motor_cmd + self.br.right_hand_cmd.motor_cmd:
            c.kp, c.kd = HAND_KP, HAND_KD

    def set(self, targets: dict) -> None:
        for i, n in enumerate(self.body_slots):
            self.br.low_cmd.motor_cmd[i].q = float(targets[n])
        for slots, cmd in ((self.left_slots, self.br.left_hand_cmd), (self.right_slots, self.br.right_hand_cmd)):
            for i, n in enumerate(slots):
                cmd.motor_cmd[i].q = float(targets[n])


class Controller:
    """50 Hz decoupled-WBC step: MuJoCo state -> upstream observation, complete goal, PD targets by joint name."""

    def __init__(self, scene: Scene, waist_location: str):
        self.scene = scene
        self.wbc, self.rm, kp, kd = make_wbc(waist_location)
        self.lower = self.wbc.lower_body_policy
        m = scene.m
        self.rm_names = list(self.rm.joint_names)
        missing = [n for n in self.rm_names if mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) < 0]
        if missing:
            raise RuntimeError(f"robot-model joints missing in the MuJoCo scene: {missing}")
        self.rm_qadr = np.array([m.jnt_qposadr[m.joint(n).id] for n in self.rm_names])
        self.rm_vadr = np.array([m.jnt_dofadr[m.joint(n).id] for n in self.rm_names])
        self.upper_names = [self.rm_names[i] for i in self.rm.get_joint_group_indices("upper_body")]
        self.plant = Plant(scene, kp, kd)
        self.ref = dict(zip(self.upper_names, np.asarray(self.rm.get_initial_upper_body_pose(), float)))
        for n in REF_NAMES:
            self.ref.setdefault(n, 0.0)
        self.ranges = {n: m.jnt_range[m.joint(n).id].copy() for n in REF_NAMES}
        self.clipped, self.held, self.seq = np.zeros(len(REF_NAMES), bool), False, -1
        self.counts = {"messages": 0, "stale": 0, "nonfinite": 0, "seq_gap": 0, "clipped_values": 0}
        self.last_msg_wall = None

    def apply_message(self, msg: dict) -> None:
        self.held = False
        if time.time() - msg["t_wall"] > MAX_MSG_AGE_S:
            self.counts["stale"] += 1
            self.held = True
            return
        vals = np.r_[msg["q_body"][15:29], msg["q_hand"], msg["q_body"][12:15]]
        if not np.isfinite(vals).all():
            self.counts["nonfinite"] += 1
            self.held = True
            return
        if self.seq >= 0:
            self.counts["seq_gap"] += max(0, int(msg["seq"]) - self.seq - 1)
        self.seq, self.last_msg_wall = int(msg["seq"]), time.time()
        self.counts["messages"] += 1
        for i, n in enumerate(REF_NAMES):
            lo, hi = self.ranges[n]
            v = float(np.clip(vals[i], lo, hi))
            self.clipped[i] = v != vals[i]
            self.ref[n] = v
        self.counts["clipped_values"] += int(self.clipped.sum())

    def applied_ref(self) -> np.ndarray:
        return np.array([self.ref[n] for n in REF_NAMES], np.float32)

    def observation(self) -> dict:
        d = self.scene.d
        return {"q": d.qpos[self.rm_qadr].copy(), "dq": d.qvel[self.rm_vadr].copy(),
                "floating_base_pose": d.qpos[:7].copy(), "floating_base_vel": d.qvel[:6].copy()}  # fmt: skip

    def goal(self) -> dict:
        return {"target_upper_body_pose": np.array([self.ref[n] for n in self.upper_names]), **GOAL_CONST}

    def step(self, t: float) -> None:
        self.wbc.set_observation(self.observation())
        self.wbc.set_goal(self.goal())  # always complete: IdentityPolicy returns exactly this goal
        q = self.wbc.get_action(time=t)["q"]
        targets = dict(zip(self.rm_names, q))
        targets.update({n: self.ref[n] for n in HANDS})  # hands straight from the joint message
        with self.scene.lock:
            self.plant.set(targets)

    def activate(self) -> None:
        self.lower.handle_keyboard_button("]")  # use_policy_action = True, on the lower-body policy only

    def flags(self) -> tuple[bool, bool]:
        return bool(self.lower.use_policy_action), bool(self.lower.use_teleop_policy_cmd)

    def rpy_cmd(self) -> np.ndarray:
        return np.asarray(self.lower.obs_buffer[-OBS_DIM:][4:7], np.float64).copy()


def state_message(scene: Scene) -> bytes:
    d = scene.d
    return b"g1_debug" + msgpack.packb(
        {"body_q": d.qpos[scene.qadr].tolist(), "left_hand_q": d.qpos[scene.hadr[:7]].tolist(),
         "right_hand_q": d.qpos[scene.hadr[7:]].tolist()}  # fmt: skip
    )


def startup(scene: Scene, ctl: Controller, steps: dict, t: float) -> None:
    """Band hang -> RL on -> band released -> reset onto the ground (shared by the run and the stand self-check)."""
    if "active" not in steps and t >= T_ACTIVATE:
        steps["active"] = t
        ctl.activate()
        scene.mark("lower-body RL output on")
        scene.view["phase"] = "RL on, hung"
    if "band" not in steps and t >= T_BAND:
        steps["band"] = t
        scene.sim.handle_keyboard_button("9")
        scene.mark("sim key '9': band released")
        scene.view["phase"] = "band released"
    if "reset" not in steps and t >= T_RESET:
        steps["reset"] = t
        scene.sim.handle_keyboard_button("backspace")
        scene.mark("sim key 'Backspace': reset onto ground")
        scene.view["phase"] = "settling on ground"


def run(a) -> None:
    scene = Scene(a.out, os.environ.get("TABLE_GAP_CM"))
    ctl = Controller(scene, a.waist_location)
    ctx = zmq.Context()
    sub = ctx.socket(zmq.SUB)
    sub.setsockopt(zmq.SUBSCRIBE, b"joints")
    sub.setsockopt(zmq.CONFLATE, 1)
    sub.connect(f"tcp://localhost:{a.action_port}")
    pub = ctx.socket(zmq.PUB)
    pub.bind(f"tcp://*:{a.state_port}")
    steps, n, t_done, episode_end = {}, 0, None, None
    reason, detail = "error", ""
    scene.mark(f"decoupled WBC host started (waist_location {a.waist_location})")
    wall0 = time.monotonic()
    try:
        while True:
            scene.step()
            n += 1
            t = scene.t
            if n % DECIMATION == 0:
                try:
                    ctl.apply_message(unpack_joint_message(sub.recv(zmq.NOBLOCK)))
                except zmq.Again:
                    pass
                ctl.step(t)
                pub.send(state_message(scene))
                startup(scene, ctl, steps, t)
                if "settled" not in steps and t >= T_SETTLED:
                    if ctl.flags() != (True, True):
                        raise RuntimeError(f"policy flags (use_policy_action, use_teleop_policy_cmd) = {ctl.flags()}")
                    steps["settled"] = t
                    scene.place_table()
                    scene.fall.armed = True
                    (a.gate / "deploy_ready").touch()
                    (a.gate / "settled").touch()
                    scene.mark("settled -> deploy_ready + settled")
                    scene.view["phase"] = "streaming joint targets"
                if "settled" in steps:
                    scene.record(applied_ref=ctl.applied_ref(), clipped=ctl.clipped.copy(), held=ctl.held,
                                 seq=ctl.seq, rpy_cmd=ctl.rpy_cmd())  # fmt: skip
                    fell = scene.check_fall()
                    if fell:
                        reason, detail = "fell", fell
                        break
                if t_done is None and (a.gate / "done").exists():
                    t_done, episode_end = t, read_done(a.gate)
                    scene.mark(f"streamer done: {episode_end}")
                if t_done is not None and t >= t_done + 3:
                    reason, detail = ("completed", "") if episode_end == "completed" else ("aborted", f"streamer: {episode_end}")
                    break
                if t_done is None and ctl.last_msg_wall is not None and time.time() - ctl.last_msg_wall > SILENCE_S:
                    reason, detail = "aborted", f"no joint message for {SILENCE_S:g} s and no GATE/done"
                    break
                if t > 1800:
                    reason, detail = "timeout", "1800 s of sim time"
                    break
            scene.pace(wall0)
    except Exception:
        detail = traceback.format_exc(limit=4)
        raise
    finally:
        scene.close(reason, detail, counts=ctl.counts, waist_location=a.waist_location)


def selfcheck(a) -> None:
    scene = Scene(a.out, None)
    m, d = scene.m, scene.d
    if a.selfcheck == "contract":
        from decoupled_wbc.control.utils.gear_wbc_utils import get_gravity_orientation

        ctl = Controller(scene, a.waist_location)
        for f in MODEL_PATH.split(","):
            print(f, hashlib.sha256((POLICY_DIR / f).read_bytes()).hexdigest())
        print("upper_body", ctl.upper_names)
        print("lower_body", [ctl.rm_names[i] for i in ctl.rm.get_joint_group_indices("lower_body")])
        assert set(REF_NAMES[:14]) <= set(ctl.upper_names), "arms must be upper-body joints"
        assert (set(WAIST_NAMES) <= set(ctl.upper_names)) == (a.waist_location == "lower_and_upper_body")
        d.qpos[3:7] = [np.cos(0.1), 0, np.sin(0.1), 0]  # 11.5 deg pitch: quaternion order must be MuJoCo's (w,x,y,z)
        mujoco.mj_forward(m, d)
        expect = d.xmat[scene.pelvis].reshape(3, 3).T @ np.array([0.0, 0.0, -1.0])
        got = np.asarray(get_gravity_orientation(d.qpos[3:7].copy()), float)
        assert np.allclose(got, expect, atol=1e-6), (got, expect)
    elif a.selfcheck == "state":
        names = list(BODY) + list(HANDS)
        vals = {}
        for i, n in enumerate(names):
            lo, hi = m.jnt_range[m.joint(n).id]
            vals[n] = lo + (hi - lo) * (i + 1) / (len(names) + 1)
            d.qpos[m.jnt_qposadr[m.joint(n).id]] = vals[n]
        msg = msgpack.unpackb(state_message(scene)[len(b"g1_debug") :], raw=False)
        assert np.allclose(msg["body_q"], [vals[n] for n in BODY])
        assert np.allclose(msg["left_hand_q"], [vals[n] for n in HANDS[:7]])
        assert np.allclose(msg["right_hand_q"], [vals[n] for n in HANDS[7:]])
    elif a.selfcheck == "command":
        ctl = Controller(scene, "lower_body")
        ref = np.array([lo + (hi - lo) * (i + 1) / 32 for i, (lo, hi) in enumerate(ctl.ranges[n] for n in REF_NAMES)], np.float32)
        assert len(set(np.round(ref[:28], 6))) == 28, "test values must be distinct"
        ctl.apply_message(unpack_joint_message(pack_joint_message(0, "selfcheck", -1, ref)))
        ctl.step(0.0)  # also proves a complete goal: get_action runs without KeyError (spec test 9)
        br = scene.env.unitree_bridge
        for i, n in enumerate(ctl.plant.body_slots):
            if n in REF_NAMES[:14]:
                assert abs(br.low_cmd.motor_cmd[i].q - ref[REF_NAMES.index(n)]) < 1e-6, n
        for slots, cmd in ((ctl.plant.left_slots, br.left_hand_cmd), (ctl.plant.right_slots, br.right_hand_cmd)):
            for i, n in enumerate(slots):
                assert abs(cmd.motor_cmd[i].q - ref[REF_NAMES.index(n)]) < 1e-6, n
        print("right-hand bridge slots:", ctl.plant.right_slots)
    elif a.selfcheck == "waist":
        ctl = Controller(scene, "lower_and_upper_body")
        ctl.activate()
        ref = ctl.applied_ref()
        ref[28:31] = [0.2, 0.1, 0.15]
        for k in range(3):  # the torso command enters the observation one tick after get_action sets it
            ctl.apply_message(unpack_joint_message(pack_joint_message(k, "selfcheck", -1, ref)))
            ctl.step(0.02 * k)
        assert ctl.flags() == (True, True), ctl.flags()
        kin = mujoco.MjData(m)
        kin.qpos[3] = 1.0
        for n, v in zip(WAIST_NAMES, ref[28:31], strict=True):
            kin.qpos[m.jnt_qposadr[m.joint(n).id]] = v
        mujoco.mj_kinematics(m, kin)
        expect = rpy_from_matrix(kin.xmat[scene.torso].reshape(3, 3))
        got = ctl.rpy_cmd()
        assert np.abs(got).max() > 0 and np.allclose(got, expect, atol=0.01), (got, expect)
    elif a.selfcheck == "stand":
        ctl = Controller(scene, a.waist_location)
        steps, n, tilts = {}, 0, []
        scene.fall.armed = False
        while scene.t < T_SETTLED + 30:
            scene.step()
            n += 1
            if n % DECIMATION == 0:
                ctl.step(scene.t)
                startup(scene, ctl, steps, scene.t)
                if scene.t >= T_SETTLED:
                    scene.fall.armed = True
                    tilts.append(tilt_deg(d.qpos[3:7]))
                    fell = scene.check_fall()
                    assert fell is None, fell
        assert max(tilts) < 3.0, f"max tilt {max(tilts):.2f} deg"
        print(f"stand: max tilt {max(tilts):.2f} deg over 30 s")
    scene.view["stop"] = True
    print(f"selfcheck {a.selfcheck}: ok")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("out", type=Path)
    p.add_argument("gate", type=Path)
    p.add_argument("--waist-location", choices=["lower_body", "lower_and_upper_body"], default="lower_body")
    p.add_argument("--action-port", type=int, default=5556)
    p.add_argument("--state-port", type=int, default=5557)
    p.add_argument("--selfcheck", choices=["contract", "state", "command", "waist", "stand"])
    a = p.parse_args()
    a.gate.mkdir(parents=True, exist_ok=True)
    selfcheck(a) if a.selfcheck else run(a)


if __name__ == "__main__":
    main()
```

If the "stand" self-check fails because the Backspace reset (fresh pose, stale observation history) destabilizes the RL policy, drop the reset step from `startup()` (release only) and re-run; record the choice in the module docstring and in the results doc.

- [ ] **Step 3: Run the self-checks in the sim container** (sync code to `code_wbc` first)

```bash
for c in contract state command waist stand; do
ssh h100 "docker run --rm --gpus device=$GPU -e NVIDIA_DRIVER_CAPABILITIES=all -e MUJOCO_GL=egl -e PYTHONPATH=/pylib_ort:/upstream \
 -w /workspace/GR00T-WholeBodyControl/gear_sonic_deploy \
 -v /mnt/data01/jhkim/model_weight/sonic_roundtrip_20260923/code_wbc:/code:ro \
 -v /mnt/data01/jhkim/model_weight/sonic_roundtrip_20260923/upstream_b042411fae/decoupled_wbc:/upstream/decoupled_wbc:ro \
 -v /mnt/data01/jhkim/model_weight/sonic_roundtrip_20260923/pylib_ort310:/pylib_ort:ro \
 --entrypoint /opt/sonic-sim/bin/python jihun/sonic-vla-sim:startupfix-20260907 -u /code/decoupled_wbc_sim_host.py /tmp/o /tmp/g --selfcheck $c" 2>&1 | tail -6
done
```

Expected: each ends with `selfcheck <name>: ok`; `contract` prints two ONNX SHA-256 values equal to `ONNX_SHA256.txt` and the upper/lower joint lists; `command` prints the right-hand slot names; `stand` prints a max tilt < 3°.

Also run `--selfcheck contract --waist-location lower_and_upper_body` (waist must then be in `upper_body`).

- [ ] **Step 4: Parameterize the launcher.** In `sonic_official_sim_eval.sh`:

a) Header comment — append to the `# Env:` line: `BACKEND (sonic | decoupled; default sonic), HOST_ARGS (extra host args, e.g. --waist-location lower_and_upper_body), REPLAY (1: stream the episode's recorded actions; no policy server, RUN_NAME is ignored), SONIC_DIR (stored-token dataset for scoring; default the episode dataset).`

b) After the `A=...; C=...; ... D=/run-output/datasets` line add:

```bash
BACKEND=${BACKEND:-sonic}; REPLAY=${REPLAY:-0}; SONIC=${SONIC_DIR:-$D/$DS}
case $BACKEND in
  sonic) HOST_PY=sonic_official_sim_host.py; HOST_MNT="" ;;
  decoupled) HOST_PY=decoupled_wbc_sim_host.py
    HOST_MNT="-v $A/upstream_b042411fae/decoupled_wbc:/upstream/decoupled_wbc:ro -v $A/pylib_ort310:/pylib_ort:ro -e PYTHONPATH=/pylib_ort:/upstream" ;;
  *) echo "BACKEND must be sonic or decoupled" >&2; exit 2 ;;
esac
if [ "$REPLAY" = 1 ]; then SRC="--replay"; else SRC="--policy-server tcp://localhost:5560"; fi
```

c) In the host `docker run`: insert `$HOST_MNT` right after `-v \$SD/gate:/gate`, and replace `-u /code/sonic_official_sim_host.py /out /gate` with `-u /code/$HOST_PY /out /gate $HOST_ARGS`.

d) Wrap the policy-server start so replay skips it — replace

`docker run -d --name P-$OUT ... sonic_policy_server.py --policy-path $R/$RUN/checkpoints/last/pretrained_model $SERVER_EXTRA >/dev/null &&
until docker logs P-$OUT 2>&1 | grep -q 'ready on port'; do sleep 5; done &&`

with

`{ [ $REPLAY = 1 ] || { docker run -d --name P-$OUT ... (unchanged) ... $SERVER_EXTRA >/dev/null &&
until docker logs P-$OUT 2>&1 | grep -q 'ready on port'; do sleep 5; done; }; } &&`

e) In the streamer `docker run`: add `-v \$M:/sonic-model:ro` after `-v \$SD/gate:/gate`, and replace `--policy-server tcp://localhost:5560` with `$SRC`.

f) In the scoring command replace `$D/$DS $EP` with `$SONIC $EP`.

Check: `bash -n sonic_official_sim_eval.sh` → no output.

- [ ] **Step 5: First end-to-end C run (28D replay)**

```bash
BACKEND=decoupled REPLAY=1 HOST_ARGS="--waist-location lower_body" CODE_DIR=/mnt/data01/jhkim/model_weight/sonic_roundtrip_20260923/code_wbc \
JOINT28_DIR=/run-output/humanoid_everyday_g1_20260923/datasets/joint28 SONIC_DIR=/run-output/humanoid_everyday_g1_20260923/datasets/sonic78_nolimit_g2 \
./sonic_official_sim_eval.sh WBC_C28_smoke - ../humanoid_everyday_g1_20260923/datasets/joint28_g2 1293 $GPU 30 \
  --action-space joint28 --backend decoupled --dex3-right-order dataset
```

Note: the launcher always passes `--dex3-right-order swap`; the later `--dex3-right-order dataset` in EXTRA overrides it (argparse keeps the last value). Expected: `scored`, `"valid": true`, `palm_err_vs_joint_ref_cm` and `palm_err_vs_applied_cm` present, `c_counts.stale` and `nonfinite` 0, tilt and table numbers printed. If invalid, read `sim/termination.json` and `sim/events.txt` first.

- [ ] **Step 6: Commit**

```bash
git add examples/g1_dex3_training/decoupled_wbc_sim_host.py examples/g1_dex3_training/sonic_official_sim_eval.sh
git commit -m "feat(g1-dex3): backend C sim host (upstream decoupled WBC b042411fae) and BACKEND/REPLAY launcher options"
```

---

### Task 8: `wbc_compare.py` — gates G0/G1/G2 and the 31D check

**Files:**
- Create: `examples/g1_dex3_training/wbc_compare.py`
- Test: `tests/datasets/test_g1_dex3_wbc_compare.py`

**Interfaces:**
- Consumes: Task 6 `stream_eval.json` keys; run dirs named `WBC_<cfg>_ep<E>_r<R>` with cfg in `Astored, A28, C28, A31syn, C31syn` (closed loop: `A28cl, C28cl, Anative`).
- Produces: `load(root) -> dict[cfg][ep] -> list[dict]`, `g0(runs)`, `g1(runs, test, ref)`, `g2(runs)`, `syn(runs, cfg, base)`, each `-> {"pass": bool, "rows": [...]}`; CLI `python wbc_compare.py AUDIT_DIR [--out FILE]`.

- [ ] **Step 1: Write the failing test** — `tests/datasets/test_g1_dex3_wbc_compare.py`:

```python
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2] / "examples" / "g1_dex3_training"))
from wbc_compare import g0, g1, g2, syn


def r(orig50=3.0, orig95=8.0, ref95=8.0, tilt=3.0, hits=0, valid=True):
    return {"valid": valid, "palm_err_vs_original_cm": {"p50": orig50, "p95": orig95},
            "palm_err_vs_joint_ref_cm": {"p95": ref95}, "stability": {"tilt_max_deg": tilt}, "table": {"hits_max": hits}}  # fmt: skip


class CompareTests(unittest.TestCase):
    def test_g0_uses_spread_or_floor(self):
        runs = {"Astored": {1: [r(3.0), r(3.2), r(3.1)]}, "A28": {1: [r(3.4)] * 3}}
        self.assertTrue(g0(runs)["pass"])  # |3.4 - 3.1| = 0.3 <= 0.5
        runs["A28"] = {1: [r(3.8)] * 3}
        self.assertFalse(g0(runs)["pass"])  # 0.7 > max(0.5, 0.2)

    def test_g1_balance(self):
        runs = {"A28": {1: [r(tilt=3.0, hits=1)] * 3}, "C28": {1: [r(tilt=4.5, hits=1)] * 3}}
        self.assertTrue(g1(runs, "C28", "A28")["pass"])
        runs["C28"] = {1: [r(tilt=4.5, hits=1)] * 2 + [r(valid=False)]}
        self.assertFalse(g1(runs, "C28", "A28")["pass"])  # an invalid repeat fails G1
        runs["C28"] = {1: [r(tilt=5.5, hits=1)] * 3}
        self.assertFalse(g1(runs, "C28", "A28")["pass"])

    def test_g2_needs_mean_gain_and_four_of_six(self):
        a = {e: [r(ref95=10.0)] * 3 for e in range(6)}
        c = {e: [r(ref95=8.5 if e < 4 else 10.5)] * 3 for e in range(6)}
        self.assertFalse(g2({"A28": a, "C28": c})["pass"])  # mean gain 0.83 < 1.0
        c = {e: [r(ref95=8.0 if e < 4 else 10.2)] * 3 for e in range(6)}
        self.assertTrue(g2({"A28": a, "C28": c})["pass"])  # gain 1.27, 4/6 better

    def test_syn_within_two_cm_of_28d(self):
        runs = {"C28": {1: [r(ref95=8.0)] * 3}, "C31syn": {1: [r(ref95=9.5)] * 3}}
        self.assertTrue(syn(runs, "C31syn", "C28")["pass"])
        runs["C31syn"] = {1: [r(ref95=10.5)] * 3}
        self.assertFalse(syn(runs, "C31syn", "C28")["pass"])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run to verify it fails** — `uv run python -m unittest tests/datasets/test_g1_dex3_wbc_compare.py` → ERROR no module.

- [ ] **Step 3: Implement** `examples/g1_dex3_training/wbc_compare.py`:

```python
"""Comparison gates of the WBC backend comparison (spec: docs/superpowers/specs/2026-10-01-g1-wbc-backends-design.md)
from sonic_stream_eval.py results. Run dirs: WBC_<config>_ep<episode>_r<repeat> (Astored, A28, C28, A31syn, C31syn;
closed loop A28cl, C28cl, Anative). Statistics are per-episode means over valid repeats; spread = max - min.
  G0  online A faithful: |A28 - Astored| <= max(0.5 cm, spread of Astored) on recorded-action palm p50, max(1.0, .) p95
  G1  equal balance: every repeat valid; mean max tilt <= reference + 2 deg; table hits <= reference's
  G2  C better: mean joint_ref palm p95 over the episodes >= 1.0 cm lower than A28, lower on >= 4 of 6 episodes
  syn 31D synthetic waist: valid, joint_ref palm p95 <= the same backend's 28D replay + 2 cm
Usage: python wbc_compare.py AUDIT_DIR [--out compare.json]
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

NAME = re.compile(r"WBC_(?P<cfg>[A-Za-z0-9]+)_ep(?P<ep>\d+)_r(?P<r>\d+)$")


def load(root) -> dict:
    runs = defaultdict(lambda: defaultdict(list))
    for p in sorted(Path(root).glob("WBC_*/stream_eval.json")):
        m = NAME.match(p.parent.name)
        if m:
            runs[m["cfg"]][int(m["ep"])].append(json.loads(p.read_text()))
    return runs


def metric(results, *path) -> list:
    out = []
    for res in results:
        if not res.get("valid"):
            continue
        v = res
        for k in path:
            v = None if v is None else v.get(k)
        if v is not None:
            out.append(float(v))
    return out


def g0(runs) -> dict:
    rows = []
    for ep in sorted(set(runs.get("Astored", {})) | set(runs.get("A28", {}))):
        for q, floor in (("p50", 0.5), ("p95", 1.0)):
            s = metric(runs.get("Astored", {}).get(ep, []), "palm_err_vs_original_cm", q)
            a = metric(runs.get("A28", {}).get(ep, []), "palm_err_vs_original_cm", q)
            if not s or not a:
                rows.append({"episode": ep, "q": q, "pass": False, "why": "no valid runs"})
                continue
            tol, diff = max(floor, max(s) - min(s)), abs(np.mean(a) - np.mean(s))
            rows.append({"episode": ep, "q": q, "diff_cm": round(diff, 2), "tol_cm": round(tol, 2), "pass": diff <= tol})
    return {"pass": bool(rows) and all(x["pass"] for x in rows), "rows": rows}


def g1(runs, test="C28", ref="A28") -> dict:
    rows = []
    for ep in sorted(runs.get(test, {})):
        c, a = runs[test][ep], runs.get(ref, {}).get(ep, [])
        ct, at = metric(c, "stability", "tilt_max_deg"), metric(a, "stability", "tilt_max_deg")
        ch, ah = metric(c, "table", "hits_max"), metric(a, "table", "hits_max")
        all_valid = bool(c) and all(x.get("valid") for x in c)
        ok = all_valid and bool(ct) and bool(at) and np.mean(ct) <= np.mean(at) + 2 and max(ch) <= max(ah)
        rows.append({"episode": ep, "all_valid": all_valid, "tilt": round(float(np.mean(ct)), 2) if ct else None,
                     "tilt_ref": round(float(np.mean(at)), 2) if at else None, "hits": max(ch) if ch else None,
                     "hits_ref": max(ah) if ah else None, "pass": bool(ok)})  # fmt: skip
    return {"pass": bool(rows) and all(x["pass"] for x in rows), "rows": rows}


def g2(runs) -> dict:
    eps = sorted(set(runs.get("C28", {})) & set(runs.get("A28", {})))
    c = [metric(runs["C28"][e], "palm_err_vs_joint_ref_cm", "p95") for e in eps]
    a = [metric(runs["A28"][e], "palm_err_vs_joint_ref_cm", "p95") for e in eps]
    pairs = [(e, np.mean(ci), np.mean(ai)) for e, ci, ai in zip(eps, c, a, strict=True) if ci and ai]
    if len(pairs) < 6:
        return {"pass": False, "why": f"{len(pairs)} episodes with valid runs on both backends (need 6)", "rows": []}
    gain = float(np.mean([ai - ci for _, ci, ai in pairs]))
    better = sum(ci < ai for _, ci, ai in pairs)
    rows = [{"episode": e, "c_p95": round(ci, 2), "a_p95": round(ai, 2)} for e, ci, ai in pairs]
    return {"pass": gain >= 1.0 and better >= 4, "gain_cm": round(gain, 2), "better": better, "rows": rows}


def syn(runs, cfg="C31syn", base="C28") -> dict:
    rows = []
    for ep in sorted(runs.get(cfg, {})):
        s, b = runs[cfg][ep], runs.get(base, {}).get(ep, [])
        sv, bv = metric(s, "palm_err_vs_joint_ref_cm", "p95"), metric(b, "palm_err_vs_joint_ref_cm", "p95")
        all_valid = bool(s) and all(x.get("valid") for x in s)
        ok = all_valid and bool(sv) and bool(bv) and np.mean(sv) <= np.mean(bv) + 2
        rows.append({"episode": ep, "all_valid": all_valid, "p95": round(float(np.mean(sv)), 2) if sv else None,
                     "base_p95": round(float(np.mean(bv)), 2) if bv else None, "pass": bool(ok)})  # fmt: skip
    return {"pass": bool(rows) and all(x["pass"] for x in rows), "rows": rows}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("root")
    p.add_argument("--out")
    a = p.parse_args()
    runs = load(a.root)
    out = {"counts": {cfg: {ep: len(v) for ep, v in eps.items()} for cfg, eps in runs.items()},
           "G0": g0(runs), "G1": g1(runs, "C28", "A28"), "G2": g2(runs),
           "syn_A": syn(runs, "A31syn", "A28"), "syn_C": syn(runs, "C31syn", "C28")}  # fmt: skip
    if "C28cl" in runs:
        out["G1_closed"] = g1(runs, "C28cl", "A28cl")
    text = json.dumps(out, indent=1, default=float)
    print(text)
    if a.out:
        Path(a.out).write_text(text)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run to verify it passes** — `uv run python -m unittest tests/datasets/test_g1_dex3_wbc_compare.py` → OK (4 tests).

- [ ] **Step 5: Commit**

```bash
git add examples/g1_dex3_training/wbc_compare.py tests/datasets/test_g1_dex3_wbc_compare.py
git commit -m "feat(g1-dex3): wbc_compare gates G0-G2 and the synthetic-waist check"
```

---

### Task 9: Step 1 — 28D replay comparison (6 episodes × 3 configs × 3 repeats)

**Files:** none (results under `$A/WBC_*`, summary `$A/wbc_compare_step1.json`).

**Interfaces:**
- Consumes: Tasks 3–8. Produces: the G0/G1/G2 verdicts for the results doc.

- [ ] **Step 1: Pick the episodes.** Find the HE ho5 held-out split and the task of each episode:

```bash
ssh h100 'ls /mnt/data01/jhkim/model_weight/g1_dex3_20260922/humanoid_everyday_g1_20260923/configs/ | grep -i -E "held|split"'
```

Then, in the LR container (`/run-output` mounted), with `SPLIT` = that file:

```python
import json, random, pyarrow.parquet as pq, glob
j = json.load(open(SPLIT)); held = j if isinstance(j, list) else next(v for v in j.values() if isinstance(v, list))
rows = [r for f in sorted(glob.glob("/run-output/humanoid_everyday_g1_20260923/datasets/joint28/meta/episodes/*/*.parquet"))
        for r in pq.read_table(f, columns=["episode_index", "tasks", "length"]).to_pylist()]
info = {r["episode_index"]: (r["tasks"][0], r["length"]) for r in rows if r["episode_index"] in set(held)}
pick = [1293, 1300]; used = {info[e][0] for e in pick}
rng = random.Random(0); cand = sorted(info); rng.shuffle(cand)
for e in cand:
    if len(pick) == 6: break
    if info[e][0] not in used: pick.append(e); used.add(info[e][0])
syn = [e for e in pick if info[e][1] >= 480][:2]  # >= 16 s at 30 Hz for the synthetic waist
print(pick, [info[e] for e in pick], "synthetic:", syn)
```

Expected: 6 episodes (1293, 1300 + 4 from other tasks). If fewer than 2 of them are ≥ 480 frames, extend `syn` with the next held-out episodes of ≥ 480 frames (any task) and note it. Write the list into the results doc draft.

- [ ] **Step 2: Ask the user for the GPU(s) for this batch** (54 runs × ~6 min ≈ 5.5 h on one GPU; two GPUs halve it). Do not start before the answer.

- [ ] **Step 3: Run the batch** (one GPU shown; split the episode list across GPUs if two):

```bash
cd /home/jihun/work/lerobot/examples/g1_dex3_training
export CODE_DIR=/mnt/data01/jhkim/model_weight/sonic_roundtrip_20260923/code_wbc
export JOINT28_DIR=/run-output/humanoid_everyday_g1_20260923/datasets/joint28
export SONIC_DIR=/run-output/humanoid_everyday_g1_20260923/datasets/sonic78_nolimit_g2
HE=../humanoid_everyday_g1_20260923/datasets
ENC="--encoder-model /sonic-model/model_encoder.onnx --observation-config /sonic-model/observation_config.yaml --robot-xml /run-output/environment/g1_29dof_with_hand.xml"
for E in $EPISODES; do for R in 1 2 3; do
  REPLAY=1 ./sonic_official_sim_eval.sh WBC_Astored_ep${E}_r$R - $HE/sonic78_nolimit_g2 $E $GPU 30 --start planner --action-space sonic78
  REPLAY=1 ./sonic_official_sim_eval.sh WBC_A28_ep${E}_r$R - $HE/joint28_g2 $E $GPU 30 --start planner --action-space joint28 $ENC
  BACKEND=decoupled REPLAY=1 HOST_ARGS="--waist-location lower_body" \
    ./sonic_official_sim_eval.sh WBC_C28_ep${E}_r$R - $HE/joint28_g2 $E $GPU 30 --action-space joint28 --backend decoupled --dex3-right-order dataset
done; done
```

`joint28_g2` is `joint28` with re-encoded video (AV1, `g=2`; data and meta shared), which `DatasetImages` decodes ~15x faster; scoring still reads `joint28` (`JOINT28_DIR`). Expected per run: `... scored`. Invalid runs are expected to happen occasionally; they stay in the record and count against G1 for C.

- [ ] **Step 4: Gates**

```bash
ssh h100 "docker run --rm -v /mnt/data01/jhkim/model_weight/sonic_roundtrip_20260923:/audit -v /mnt/data01/jhkim/model_weight/sonic_roundtrip_20260923/code_wbc:/code:ro -v /mnt/data01/jhkim/model_weight/g1_dex3_20260922:/run-output:ro --entrypoint /run-output/environment/venv/bin/python 4cbe2a3f7fc6 /code/wbc_compare.py /audit --out /audit/wbc_compare_step1.json"
```

Expected: JSON with `G0`, `G1`, `G2`. If G0 fails, stop: the online A path differs from the stored tokens and A-vs-C conclusions are not allowed (investigate `policy_token_vs_stored_abs` of A28 runs first). Report G0/G1/G2 to the user before Step 2 of the evaluation.

---

### Task 10: Step 2 — synthetic 31D waist (A and C)

**Files:** none (results `$A/WBC_{A,C}31syn_*`).

**Interfaces:** Consumes Tasks 1, 3, 7, 8 and the `syn` episodes from Task 9 Step 1.

- [ ] **Step 1: Run** (2 episodes × 2 backends × 3 repeats ≈ 1.2 h):

```bash
for E in $SYN_EPISODES; do for R in 1 2 3; do
  REPLAY=1 ./sonic_official_sim_eval.sh WBC_A31syn_ep${E}_r$R - $HE/joint28_g2 $E $GPU 30 --start planner --action-space joint31 --synthetic-waist $ENC
  BACKEND=decoupled REPLAY=1 HOST_ARGS="--waist-location lower_and_upper_body" \
    ./sonic_official_sim_eval.sh WBC_C31syn_ep${E}_r$R - $HE/joint28_g2 $E $GPU 30 --action-space joint31 --synthetic-waist --backend decoupled --dex3-right-order dataset
done; done
```

- [ ] **Step 2: Gates** — rerun the Task 9 Step 4 command with `--out /audit/wbc_compare_step2.json`. Expected: `syn_A`, `syn_C` verdicts; C31syn results carry `torso_err_rad` and `rpy_cmd_err_rad`; a C run whose waist never reached the policy is already invalid (scorer). Report to the user.

---

### Task 11: Step 3 — 28D GR00T on HE and closed loop

**Files:**
- Create: `examples/g1_dex3_training/augment_joint_quantiles.py` (copied from the H100 copy used on 2026-09-25)

**Interfaces:** Consumes Tasks 3, 7, 8. Produces closed-loop results `WBC_{A28cl,C28cl,Anative}_*`.

- [ ] **Step 1: Bring the quantile script into the repo**

```bash
ssh h100 'find /mnt/data01/jhkim/model_weight -maxdepth 4 -name augment_joint_quantiles.py 2>/dev/null'
```

Copy the most recent one (by mtime) to `examples/g1_dex3_training/augment_joint_quantiles.py` via `ssh h100 "docker run --rm -v <dir>:/s:ro --entrypoint cat 4cbe2a3f7fc6 /s/augment_joint_quantiles.py" > examples/g1_dex3_training/augment_joint_quantiles.py`. Read it fully; it must take `--root` and write exact q01/q99 into `meta/stats.json`. Commit:

```bash
git add examples/g1_dex3_training/augment_joint_quantiles.py
git commit -m "chore(g1-dex3): add augment_joint_quantiles.py (exact q01/q99), used since 2026-09-25"
```

- [ ] **Step 2: Recompute HE joint28 quantiles.** Check whether `joint28_g2` shares `meta/` with `joint28`:

```bash
ssh h100 'ls -la /mnt/data01/jhkim/model_weight/g1_dex3_20260922/humanoid_everyday_g1_20260923/datasets/joint28_g2/'
```

Back up and recompute on the real `meta/` (a container with `/run-output` writable, LR image):

```bash
cp meta/stats.json meta/stats.json.bak-quantile-20261001   # inside the container, in the real dataset dir
/run-output/environment/venv/bin/python /code/augment_joint_quantiles.py --root <real dataset dir>
```

Expected: the script reports updated `q01/q99` for `action` and `observation.state`.

- [ ] **Step 3: Ask the user for the H100 queue slot** (GPUs 0/6 are shared with the official retraining). The GPU queue is run by another session (`queue_official`); give the user the config path and let them or that session queue it. Do not edit queue files from this session.

- [ ] **Step 4: Write the training config** from the HE GR00T official config:

```bash
ssh h100 'ls /mnt/data01/jhkim/model_weight/g1_dex3_20260922/humanoid_everyday_g1_20260923/configs/ | grep groot'
```

Copy `groot_sonic78sonicstate_ho5_official_full.json` (or the exact HE GR00T official name listed) to `groot_joint28_ho5_official_full.json` and set, with `jq`: `.dataset.root` → the `joint28_g2` dataset dir, `.job_name` → `groot_joint28_ho5_official_full`, `.output_dir` → `/run-output/humanoid_everyday_g1_20260923/runs/groot_joint28_ho5_official_full`. Leave every recipe field unchanged. Show the diff (`diff <(jq -S . old) <(jq -S . new)`) to the user with the queue request.

- [ ] **Step 5: After training — verify the saved normalizer.** In the LR container:

```python
import glob, json, numpy as np
from safetensors.numpy import load_file
run = "/run-output/humanoid_everyday_g1_20260923/runs/groot_joint28_ho5_official_full/checkpoints/last/pretrained_model"
stats = json.load(open("<real joint28 dataset dir>/meta/stats.json"))
for f in glob.glob(f"{run}/*.safetensors"):
    t = load_file(f)
    for k, v in t.items():
        for feat in ("action", "observation.state"):
            for q in ("q01", "q99"):
                if k.startswith(feat) and k.endswith(q):
                    print(f, k, float(np.abs(v - np.asarray(stats[feat][q])).max()))
```

Expected: every printed difference ≤ 1e-6. If no key matches, list the keys of the processor files and compare the q01/q99 entries by name; stop if they differ from the recomputed stats.

- [ ] **Step 6: Closed-loop runs** (same 6 episodes × 3 repeats; GPU from the user):

```bash
RUNJ=../humanoid_everyday_g1_20260923/runs/groot_joint28_ho5_official_full
RUNN=../humanoid_everyday_g1_20260923/runs/groot_sonic78sonicstate_ho5_official_full
for E in $EPISODES; do for R in 1 2 3; do
  ./sonic_official_sim_eval.sh WBC_A28cl_ep${E}_r$R $RUNJ $HE/joint28_g2 $E $GPU 30 --start planner --action-space joint28 $ENC
  BACKEND=decoupled HOST_ARGS="--waist-location lower_body" \
    ./sonic_official_sim_eval.sh WBC_C28cl_ep${E}_r$R $RUNJ $HE/joint28_g2 $E $GPU 30 --action-space joint28 --backend decoupled --dex3-right-order dataset
  ./sonic_official_sim_eval.sh WBC_Anative_ep${E}_r$R $RUNN $HE/sonic78_nolimit_g2 $E $GPU 30 --start planner
done; done
```

Then the Task 9 Step 4 command with `--out /audit/wbc_compare_closed.json`. Expected: `G1_closed` verdict; palm error reported both ways for A28cl/C28cl (`palm_err_vs_joint_ref_cm`, `palm_err_vs_original_cm`), recorded-action only for Anative.

---

### Task 12: Results doc and pointers

**Files:**
- Create: `docs/research/<run-date>-g1-wbc-backends.md`
- Modify: `CLAUDE.md` (G1 bullet: one sentence + link), memory file `groot-jerky-predictions-open-issue.md` only if the current-work pointer changes

- [ ] **Step 1: Write the results doc**: what was built (files, CLI), the backend C contract (incl. the 200 Hz correction, activation via `"]"`, any startup change from Task 7), episode list, per-config tables (palm p50/p95 three ways, wrist orientation, tilt, table hits, arm speed/jerk, palm jerk, invalid-run counts and reasons), G0/G1/G2/syn verdicts with numbers, closed-loop results when available, open issues. No host names, user names or credentials (public repo).

- [ ] **Step 2: Add the CLAUDE.md pointer** next to the existing G1 links: "WBC backend comparison (SONIC vs GR00T decoupled WBC, 28D/31D): [`docs/research/<run-date>-g1-wbc-backends.md`](...)."

- [ ] **Step 3: Commit**

```bash
git add docs/research/<run-date>-g1-wbc-backends.md CLAUDE.md
git commit -m "docs(g1-dex3): WBC backend comparison results (SONIC vs GR00T decoupled WBC)"
```
