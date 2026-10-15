# G1 Isaac Lab closed-loop handover evaluation (SONIC deploy) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run sonic78 checkpoints closed loop on handover_bottle in Isaac Lab 2.3.2 behind the unchanged SONIC C++ deploy, with an automatic verdict per trial written into the existing `eval_session` session format.

**Architecture:** Isaac Lab replaces only the MuJoCo physics behind the deploy's Unitree DDS interface. `isaac_scene.Scene` mirrors `sim_scene.Scene`; `sonic_official_sim_host.py` picks the scene with `SIM=isaac`; a pure-numpy `handover_success.Judge` acts as the operator through the existing `--gate-dir` files, plus one new file `ended.{i}` from the streamer.

**Tech Stack:** Isaac Sim 5.1 + Isaac Lab 2.3.2 (conda env `sonic`, Python 3.11), unitree_sdk2py 1.0.1 + cyclonedds 0.10.2, ZMQ, numpy, OpenCV, pytest (`uv run` in the lerobot repo).

**Spec:** `docs/superpowers/specs/2026-10-14-g1-isaac-sonic-eval-design.md` (approved at 15dbfbc4). Read it before any task.

## Global Constraints

- Isaac Lab 2.3.2 / Isaac Sim 5.1, conda env `/home/jihun/anaconda3/envs/sonic` (no new Isaac install; `/` has 14 GB free).
- Body controller: unchanged `g1_deploy_onnx_ref` with `sonic_v1_1` from `~/work/GR00T-WholeBodyControl/gear_sonic_deploy`.
- Robot model: `gear_sonic/data/robot_model/model_data/g1/g1_29dof_with_hand.urdf` (43 revolute joints, the twin of the MuJoCo `g1_29dof_with_hand.xml`).
- Physics dt 0.005 s; head camera rendered at 30 Hz, 640 x 480, published as `egocentric` with the `g1_head_camera_server.py` JSON protocol.
- Torque law `tau = tau_ff + kp (q* - q) + kd (dq* - dq)`, clipped to the gear_sonic `motor_effort_limit_list`.
- Isaac on GPU 0, policy server on GPU 1. GPU 1 had 7.1 GB in use on 2026-10-15: check `nvidia-smi` before every run; a policy server needs up to 12 GB (Pi0.5 he bf16).
- Judge reasons use `eval_session.DEFAULT_FAILURE_REASONS` indices (1-based).
- Public repo: no host names, user names, addresses or credentials in committed files.
- **Stop rule (user):** Task 1 is test 0a. If it fails, stop and report; do not start Task 2.
- Commits end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. Never commit `examples/g1_dex3_training/prepare_wbt_sonic78.py` (the user's uncommitted edit) or untracked research files.

## Review Focus

1. **Isaac joint order differs from the deploy's motor order.** Isaac's articulation orders joints breadth-first; the bridge indexes motors in `sim_scene.BODY` / hand order. A wrong map makes the robot fall at once. Pinned by the Task 3 startup check and the `test_isaac_scene_pure.py` map test.
2. **Quaternion order.** Isaac `root_state_w` is (w, x, y, z), the bridge expects (w, x, y, z) for `floating_base_pose[3:7]`: same order, but `torso_R` / `wrist_R` come from quaternions and need the right conversion. Pinned by the pure `quat_wxyz_to_matrix` test in Task 3.
3. **A stale `ended.{i}` or `stop.{i}` from a crash finalizing trial i at once.** Pinned by the Task 4 host test `test_clear_trial_files`.
4. **The policy server is not running or serves another model.** The streamer already stops the session on a model swap; the launcher must fail before starting Isaac if the server port does not answer. Pinned by the Task 7 preflight step.
5. **Real-time drift during a long session** (shader cache, Python GC): lag measured in every trial, not only in test 0. Pinned by the Task 3 lag log and its check in Task 7.

---

## File structure

All new code in `examples/g1_dex3_training/` (the G1 pipeline lives there); tests in `tests/datasets/` next to the other `test_g1_dex3_*` files.

| File | Responsibility |
| --- | --- |
| `isaac_scene.py` (new) | Isaac Lab scene with the `sim_scene.Scene` interface; pure helpers (`pd_torques`, `quat_wxyz_to_matrix`, `load_camera`) at the top, Isaac imports only inside `Scene.__init__`. |
| `isaac_rtf_probe.py` (new) | Test 0a: unpaced real-time factor of the scene with the camera on. |
| `cameras/d435i_head.json` (new) | Head camera intrinsics and mount (calibration knob). |
| `handover_success.py` (new) | Pure-numpy judge. |
| `tasks/handover_bottle.json` (new, Task 6) | Bottle, zone, jitter, judge constants and their source runs. |
| `derive_handover_task.py` (new, Task 6) | Builds `tasks/handover_bottle.json` from MuJoCo replays. |
| `eval_session.py` (edit) | `write_ended` / `read_ended` / `clear_trial_files`. |
| `sonic_policy_streamer.py` (edit) | Calls `write_ended` in `--trials` + `--gate-dir` mode. |
| `sim_scene.py` (edit) | Adds `key(name)` and `bridge` so the host no longer reaches into `scene.sim` / `scene.env`. |
| `sonic_official_sim_host.py` (edit) | `SIM` switch, local deploy path, trials loop, judge, lag log. |
| `wbc_compare.py` (edit) | Gate `GI`. |
| `sonic_official_sim_eval.sh` (edit) | `SIM=isaac` local mode. |
| `tests/datasets/test_g1_dex3_isaac_scene_pure.py` (new) | Pure helpers of `isaac_scene.py`. |
| `tests/datasets/test_g1_dex3_handover_success.py` (new) | Judge. |
| `tests/datasets/test_g1_dex3_eval_session.py` (edit) | ended / clear helpers. |
| `tests/datasets/test_g1_dex3_wbc_common.py` or a new `test_g1_dex3_wbc_compare_gi.py` | Gate GI. |

Shorthands used below: `L=~/work/lerobot`, `D=$L/examples/g1_dex3_training`, `G=~/work/GR00T-WholeBodyControl`, `PY=/home/jihun/anaconda3/envs/sonic/bin/python`.

---

### Task 1: Isaac scene skeleton and test 0a (unpaced real-time factor) — STOP GATE

**Files:**
- Create: `examples/g1_dex3_training/cameras/d435i_head.json`
- Create: `examples/g1_dex3_training/isaac_scene.py`
- Create: `examples/g1_dex3_training/isaac_rtf_probe.py`
- Test: `tests/datasets/test_g1_dex3_isaac_scene_pure.py`

**Interfaces:**
- Produces: `isaac_scene.load_camera(path) -> dict`; `isaac_scene.pd_torques(cmd: dict, q, dq, limit) -> np.ndarray` where `cmd` has arrays `q, dq, kp, kd, tau` of equal length; `isaac_scene.Scene(out: Path, table_gap_cm: str | None, camera_json: Path = CAMERA_JSON, camera_port: int = 5555)` with attributes `t, dt, cmd, pinned_root, rec, view, fall` and methods `step()`, `place_table()`, `pace(wall0)`, `mark(event)`, `close(reason, detail="", **fields)`; module constants `BODY`, `HANDS` (re-exported from `sim_scene`), `URDF`, `CAMERA_JSON`, `FPS = 30`, `W, H = 640, 480`.

- [ ] **Step 1: Write the failing test for the pure helpers**

`tests/datasets/test_g1_dex3_isaac_scene_pure.py`:

```python
"""isaac_scene.py pure helpers (no Isaac import): camera config and the PD torque law of gear_sonic base_sim."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / "examples" / "g1_dex3_training"))
import isaac_scene as isc


def test_pd_torques_matches_base_sim_and_clips():
    cmd = {"q": np.array([1.0, 0.0]), "dq": np.zeros(2), "kp": np.array([100.0, 10.0]),
           "kd": np.array([2.0, 1.0]), "tau": np.array([0.5, 0.0])}  # fmt: skip
    q, dq, limit = np.array([0.9, 0.0]), np.array([0.0, 1.0]), np.array([50.0, 50.0])
    np.testing.assert_allclose(isc.pd_torques(cmd, q, dq, limit), [0.5 + 100 * 0.1, -1.0])
    cmd["kp"][0] = 1000.0  # 100 Nm unclipped
    assert isc.pd_torques(cmd, q, dq, limit)[0] == 50.0


def test_load_camera_requires_all_keys(tmp_path):
    cam = isc.load_camera(isc.CAMERA_JSON)
    assert (cam["width"], cam["height"]) == (640, 480)
    assert cam["parent"] == "torso_link" and len(cam["pos"]) == 3 and len(cam["rot_wxyz"]) == 4
    bad = tmp_path / "c.json"
    bad.write_text(json.dumps({k: v for k, v in cam.items() if k != "fx"}))
    with pytest.raises(ValueError, match="fx"):
        isc.load_camera(bad)
```

- [ ] **Step 2: Run it to verify it fails**

Run: `cd $L && uv run pytest tests/datasets/test_g1_dex3_isaac_scene_pure.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'isaac_scene'`.

- [ ] **Step 3: Write the camera config**

`examples/g1_dex3_training/cameras/d435i_head.json` (pos / rot are the URDF `d435_joint` origin on `torso_link`: xyz 0.0576235 0.01753 0.41987, rpy 0 0.8307767 0; rot_wxyz = (cos(p/2), 0, sin(p/2), 0); intrinsics are the common D435 colour values at 640 x 480):

```json
{
  "parent": "torso_link",
  "pos": [0.0576235, 0.01753, 0.41987],
  "rot_wxyz": [0.91506, 0.0, 0.40334, 0.0],
  "width": 640,
  "height": 480,
  "fx": 615.0,
  "fy": 615.0,
  "cx": 320.0,
  "cy": 240.0,
  "source": "nominal D435 colour 640x480 and the URDF d435_joint mount; replace fx fy cx cy with `rs-enumerate-devices -c` from the robot's head camera at 640x480"
}
```

Note on the spec: the spec names the OpenCV-intrinsics path with distortion. The D435 colour stream reports zero distortion, so a pinhole camera from the intrinsic matrix is equivalent; the camera is attached to `torso_link` with the `d435_joint` offset, which works whether or not the URDF importer merges the fixed `d435_link`. If the robot's `rs-enumerate-devices -c` shows non-zero coefficients, switch to the OpenCV path then.

- [ ] **Step 4: Write `isaac_scene.py` (skeleton: pure helpers, scene, PD, camera, pinned root, RTF-relevant parts)**

```python
"""Isaac Lab scene for the sim hosts, with the interface of sim_scene.Scene (MuJoCo): G1 29 + Dex3 from
g1_29dof_with_hand.urdf (the twin of gear_sonic's MuJoCo model), the optional tabletop, a bottle, and the head
camera (D435 mount on torso_link, 640 x 480 at 30 Hz, published like g1_head_camera_server.py as "egocentric").

Torques: tau = tau_ff + kp (q* - q) + kd (dq* - dq) from self.cmd, clipped to gear_sonic's motor effort limits,
applied in effort mode at dt 0.005 s, as gear_sonic base_sim does. Importing this module has no side effects:
Isaac starts in Scene(...). Runs in the `sonic` conda env (Isaac Sim 5.1, Isaac Lab 2.3.2).
"""

from __future__ import annotations

import base64
import contextlib
import json
import os
import sys
import threading
import time
from pathlib import Path

import numpy as np

try:
    from .sim_scene import BODY, HANDS, REC_KEYS
    from .wbc_common import FallDetector, write_termination
except ImportError:
    from sim_scene import BODY, HANDS, REC_KEYS
    from wbc_common import FallDetector, write_termination

GEAR_ROOT = Path(os.environ.get("GEAR_ROOT", Path.home() / "work" / "GR00T-WholeBodyControl"))
URDF = GEAR_ROOT / "gear_sonic/data/robot_model/model_data/g1/g1_29dof_with_hand.urdf"
CAMERA_JSON = Path(__file__).parent / "cameras" / "d435i_head.json"
FPS, W, H = 30, 640, 480
DT = 0.005
STAND_Z = 0.79  # pelvis height standing on the default pose; Task 3 replaces it with the measured value
CAMERA_KEYS = ("parent", "pos", "rot_wxyz", "width", "height", "fx", "fy", "cx", "cy")


def load_camera(path) -> dict:
    cam = json.loads(Path(path).read_text())
    missing = [k for k in CAMERA_KEYS if k not in cam]
    if missing:
        raise ValueError(f"{path} lacks {missing}")
    return cam


def pd_torques(cmd: dict, q, dq, limit) -> np.ndarray:
    """gear_sonic base_sim compute_body_torques / compute_hand_torques, then the np.clip of sim_step."""
    tau = cmd["tau"] + cmd["kp"] * (cmd["q"] - q) + cmd["kd"] * (cmd["dq"] - dq)
    return np.clip(tau, -limit, limit)


def zero_cmd(n: int) -> dict:
    return {k: np.zeros(n) for k in ("q", "dq", "kp", "kd", "tau")}


class Scene:
    def __init__(self, out: Path, table_gap_cm: str | None, camera_json: Path = CAMERA_JSON, camera_port: int = 5555):
        from isaaclab.app import AppLauncher

        self.app = AppLauncher(headless=True, enable_cameras=True, device="cuda:0").app
        import isaaclab.sim as sim_utils
        import zmq
        from isaaclab.actuators import ImplicitActuatorCfg
        from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
        from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
        from isaaclab.sensors import Camera, CameraCfg, ContactSensorCfg
        from isaaclab.utils import configclass

        self.out = Path(out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.table_gap = float(table_gap_cm) if table_gap_cm else None
        cam = load_camera(camera_json)

        def act(expr, friction):  # effort mode: no drive stiffness; MJCF passive damping 0.05, armature 0.01
            return ImplicitActuatorCfg(joint_names_expr=expr, stiffness=0.0, damping=0.05, armature=0.01,
                                       friction=friction, effort_limit_sim=1000.0, velocity_limit_sim=100.0)  # fmt: skip

        @configclass
        class CfgScene(InteractiveSceneCfg):
            ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
            light = AssetBaseCfg(prim_path="/World/light", spawn=sim_utils.DomeLightCfg(intensity=2500.0))
            robot = ArticulationCfg(
                prim_path="{ENV_REGEX_NS}/Robot",
                spawn=sim_utils.UrdfFileCfg(
                    asset_path=str(URDF), fix_base=False, merge_fixed_joints=True, activate_contact_sensors=True,
                    joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                        gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.0, damping=0.0)
                    ),
                    articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                        enabled_self_collisions=False, solver_position_iteration_count=8,
                        solver_velocity_iteration_count=4,
                    ),
                ),  # fmt: skip
                init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, STAND_Z)),
                actuators={
                    "main": act([".*_hip_.*", ".*_knee_.*", ".*_ankle_.*", "waist_.*", ".*_shoulder_.*",
                                 ".*_elbow_.*"], 0.2),  # fmt: skip
                    "fine": act([".*_wrist_.*", ".*_hand_.*"], 0.1),
                },
            )
            table = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Table",
                spawn=sim_utils.CuboidCfg(
                    size=(0.8, 1.6, 0.03),
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.62, 0.46, 0.3)),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(10.0, 0.0, 0.785)),  # parked, as sim_scene
            )
            bottle = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Bottle",
                spawn=sim_utils.CylinderCfg(
                    radius=0.033, height=0.20,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
                    collision_props=sim_utils.CollisionPropertiesCfg(),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.5, 0.9)),
                ),  # fmt: skip
                init_state=RigidObjectCfg.InitialStateCfg(pos=(10.0, 0.0, 0.90)),
            )
            feet = ContactSensorCfg(
                prim_path="{ENV_REGEX_NS}/Robot/.*_ankle_roll_link", filter_prim_paths_expr=["/World/ground"]
            )

        self.sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=DT, render_interval=1, device="cuda:0"))
        self.scene = InteractiveScene(CfgScene(num_envs=1, env_spacing=4.0))
        k = [cam["fx"], 0.0, cam["cx"], 0.0, cam["fy"], cam["cy"], 0.0, 0.0, 1.0]
        self.camera = Camera(
            CameraCfg(
                prim_path=f"/World/envs/env_0/Robot/{cam['parent']}/head_cam",
                height=cam["height"], width=cam["width"], data_types=["rgb"], update_period=0.0,
                spawn=sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
                    intrinsic_matrix=k, width=cam["width"], height=cam["height"]
                ),
                offset=CameraCfg.OffsetCfg(pos=tuple(cam["pos"]), rot=tuple(cam["rot_wxyz"]), convention="world"),
            )
        )  # fmt: skip
        self.sim.reset()
        self.robot, self.table_obj = self.scene["robot"], self.scene["table"]
        self.bottle, self.feet = self.scene["bottle"], self.scene["feet"]
        self.body_ids, _ = self.robot.find_joints(BODY, preserve_order=True)
        self.hand_ids, _ = self.robot.find_joints(HANDS, preserve_order=True)
        self.limit = self._effort_limits()
        self.cmd = zero_cmd(len(BODY) + len(HANDS))
        self.pinned_root: np.ndarray | None = None  # (7,) pos + quat wxyz; held each step while set (band)
        self.dt, self.t, self.next_frame = DT, 0.0, 0.0
        self.lock = threading.Lock()
        self.view = {"phase": "hanging on elastic band", "stop": False}
        self.rec = {k: [] for k in REC_KEYS}
        self.next_rec = 0.0
        self.fall = FallDetector()
        self.frame_ms: list[float] = []
        self.pub = zmq.Context().socket(zmq.PUB)
        self.pub.bind(f"tcp://*:{camera_port}")

    def _effort_limits(self) -> np.ndarray:
        """gear_sonic's motor_effort_limit_list (body 29 then hands 14), the same SimLoopConfig call as sim_scene."""
        sys.path.insert(0, str(GEAR_ROOT))
        from gear_sonic.utils.mujoco_sim.configs import SimLoopConfig

        cfg = SimLoopConfig(interface="sim", enable_onscreen=False, hand_profile="dex3").load_wbc_yaml()
        lim = np.asarray(cfg["motor_effort_limit_list"], dtype=float)
        if lim.size != len(BODY) + len(HANDS):
            raise SystemExit(f"motor_effort_limit_list has {lim.size} entries, expected {len(BODY) + len(HANDS)}")
        return lim

    def joint_state(self) -> tuple[np.ndarray, np.ndarray]:
        ids = self.body_ids + self.hand_ids
        d = self.robot.data
        return d.joint_pos[0, ids].cpu().numpy(), d.joint_vel[0, ids].cpu().numpy()

    def step(self) -> None:
        import torch

        with self.lock:
            q, dq = self.joint_state()
            tau = pd_torques(self.cmd, q, dq, self.limit)
            self.tau = tau
            self.robot.set_joint_effort_target(
                torch.tensor(tau, dtype=torch.float32, device=self.sim.device).unsqueeze(0),
                joint_ids=self.body_ids + self.hand_ids,
            )
            if self.pinned_root is not None:
                pose = torch.tensor(self.pinned_root, dtype=torch.float32, device=self.sim.device).unsqueeze(0)
                self.robot.write_root_pose_to_sim(pose)
                self.robot.write_root_velocity_to_sim(torch.zeros(1, 6, device=self.sim.device))
            render = self.t + 1e-9 >= self.next_frame
            self.scene.write_data_to_sim()
            self.sim.step(render=render)
            self.scene.update(DT)
            self.t += DT
        if render:
            self.next_frame += 1 / FPS
            self._publish_frame()

    def _publish_frame(self) -> None:
        import cv2

        t0 = time.perf_counter()
        self.camera.update(0.0, force_recompute=True)
        rgb = self.camera.data.output["rgb"][0, ..., :3].cpu().numpy()
        _, jpg = cv2.imencode(".jpg", rgb, [int(cv2.IMWRITE_JPEG_QUALITY), 90])  # RGB as is, like the camera server
        msg = {"timestamps": {"egocentric": time.time()}, "images": {"egocentric": base64.b64encode(jpg).decode()}}
        with contextlib.suppress(Exception):
            import zmq

            self.pub.send_string(json.dumps(msg), zmq.NOBLOCK)
        self.last_rgb = rgb
        self.frame_ms.append((time.perf_counter() - t0) * 1e3)

    def place_table(self) -> None:
        if self.table_gap is None:
            return
        import torch

        with self.lock:
            root = self.robot.data.root_state_w[0].cpu().numpy()
            edge = root[0] + 0.08 + self.table_gap / 100
            pose = torch.tensor([[edge + 0.4, root[1], 0.785, 1.0, 0.0, 0.0, 0.0]], device=self.sim.device)
            self.table_obj.write_root_pose_to_sim(pose)
        self.mark(f"table placed: near edge {self.table_gap:g} cm from the torso (x = {edge:.3f} m)")

    def mark(self, event: str) -> None:
        print(f"[{self.t:8.3f}] {event}", flush=True)
        with open(self.out / "events.txt", "a") as f:
            f.write(f"{time.time():.3f} {self.t:.3f} {event}\n")

    def pace(self, wall0: float) -> None:
        lag = self.t - (time.monotonic() - wall0)
        if lag > 0:
            time.sleep(lag)

    def close(self, reason: str, detail: str = "", **fields) -> None:
        self.view["stop"] = True
        if self.rec["t"]:
            np.savez_compressed(self.out / "sim_state.npz", **{k: np.array(v) for k, v in self.rec.items()})
        write_termination(self.out, reason, self.view["phase"], detail, **fields)
        self.mark(f"termination: {reason} {detail}")
        self.app.close()
```

- [ ] **Step 5: Run the pure test to verify it passes**

Run: `cd $L && uv run pytest tests/datasets/test_g1_dex3_isaac_scene_pure.py -v`
Expected: 2 passed (the module imports without Isaac).

- [ ] **Step 6: Check the Isaac API names used above exist in this env**

Run:
```bash
$PY -c "
import inspect, isaaclab.sim as s
print(inspect.signature(s.PinholeCameraCfg.from_intrinsic_matrix))
print('merge_fixed_joints' in s.UrdfFileCfg.__dataclass_fields__ or hasattr(s.UrdfFileCfg, 'merge_fixed_joints'))
from isaaclab.sensors import ContactSensorCfg; print('filter_prim_paths_expr' in ContactSensorCfg.__dataclass_fields__)
"
```
Expected: a signature containing `intrinsic_matrix, width, height`; `True`; `True`. If a name differs, fix `isaac_scene.py` to the env's API before Step 7 (the env is Isaac Lab 2.3.2; `isaaclab` 0.54.2).

- [ ] **Step 7: Write the probe `isaac_rtf_probe.py`**

```python
"""Spec test 0a: unpaced real-time factor of the Isaac scene with the head camera rendering at 30 Hz.

The robot is held on the band at standing height (root pinned, feet on the floor), joints PD-held at gear_sonic's
DEFAULT_DOF_ANGLES with its MOTOR_KP / MOTOR_KD (hands 1.5 / 0.1 at 0), the table placed: the per-step work of a
real run without the deploy. Pass: sim time / wall time >= 1.2 in every 1 s window after a 2 s warm-up.
Usage (sonic env): CUDA_VISIBLE_DEVICES=0 python isaac_rtf_probe.py OUT_DIR [--seconds 60]
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

import isaac_scene as isc

p = argparse.ArgumentParser()
p.add_argument("out", type=Path)
p.add_argument("--seconds", type=float, default=60.0)
a = p.parse_args()

scene = isc.Scene(a.out, "5")
sys.path.insert(0, str(isc.GEAR_ROOT))
from gear_sonic.utils.mujoco_sim.configs import SimLoopConfig  # noqa: E402

cfg = SimLoopConfig(interface="sim", enable_onscreen=False, hand_profile="dex3").load_wbc_yaml()
nb, nh = len(isc.BODY), len(isc.HANDS)
scene.cmd["q"][:nb] = cfg["DEFAULT_DOF_ANGLES"]
scene.cmd["kp"][:nb], scene.cmd["kd"][:nb] = cfg["MOTOR_KP"], cfg["MOTOR_KD"]
scene.cmd["kp"][nb:], scene.cmd["kd"][nb:] = 1.5, 0.1
scene.pinned_root = np.array([0.0, 0.0, isc.STAND_Z, 1.0, 0.0, 0.0, 0.0])
scene.place_table()

windows, wall0, t_win, w_win = [], time.monotonic(), 0.0, time.monotonic()
while scene.t < a.seconds:
    scene.step()
    if scene.t - t_win >= 1.0 - 1e-9:
        now = time.monotonic()
        windows.append({"t": round(scene.t, 3), "rtf": round((scene.t - t_win) / (now - w_win), 3)})
        t_win, w_win = scene.t, now
steady = [w["rtf"] for w in windows if w["t"] > 2.0]
feet = int((scene.feet.data.net_forces_w[0].norm(dim=-1) > 1.0).sum())
result = {
    "pass": bool(steady) and min(steady) >= 1.2,
    "rtf_min": min(steady), "rtf_median": float(np.median(steady)),
    "overall_rtf": round(scene.t / (time.monotonic() - wall0), 3),
    "frame_ms_p95": float(np.percentile(scene.frame_ms, 95)), "frames": len(scene.frame_ms),
    "feet_in_contact": feet, "windows": windows,
}  # fmt: skip
(a.out / "rtf.json").write_text(json.dumps(result, indent=1))
import cv2  # noqa: E402

cv2.imwrite(str(a.out / "head_cam_last.png"), cv2.cvtColor(scene.last_rgb, cv2.COLOR_RGB2BGR))
print(json.dumps({k: v for k, v in result.items() if k != "windows"}), flush=True)
scene.close("completed")
```

- [ ] **Step 8: Run test 0a**

Check GPU 0 is free enough first (`nvidia-smi`; Isaac needs about 6-8 GB). Then:
```bash
cd $D && CUDA_VISIBLE_DEVICES=0 $PY isaac_rtf_probe.py /mnt/data/jihun/g1_isaac_eval/rtf_probe_01 --seconds 60
```
Expected: one JSON line with `"pass": true`, `rtf_min >= 1.2`, `feet_in_contact` 1 or 2, `frames` about 1800, and `head_cam_last.png` showing the table from the head view.

Checks before trusting the number:
- `feet_in_contact` 0 means `STAND_Z` floats the robot: lower it by 1 cm steps until 2, rerun.
- `head_cam_last.png` must look like the robot's head view (table edge in the lower part of the image). Black image: the camera prim path is wrong.

- [ ] **Step 9: STOP GATE**

If `"pass": false`: stop. Report `rtf_min`, `rtf_median`, `frame_ms_p95` and GPU memory to the user, with the spec's consequence (closed loop invalid on the 3060; move to the H100). Commit the probe and its result anyway (Step 10) so the measurement is kept. Do not start Task 2.

If `"pass": true`: continue.

- [ ] **Step 10: Commit**

```bash
cd $L && git add examples/g1_dex3_training/isaac_scene.py examples/g1_dex3_training/isaac_rtf_probe.py \
  examples/g1_dex3_training/cameras/d435i_head.json tests/datasets/test_g1_dex3_isaac_scene_pure.py
git commit -m "feat(g1-dex3): Isaac Lab G1 scene skeleton and real-time-factor probe (spec test 0a)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: handover judge

**Files:**
- Create: `examples/g1_dex3_training/handover_success.py`
- Test: `tests/datasets/test_g1_dex3_handover_success.py`

**Interfaces:**
- Produces: `handover_success.Judge(cfg: dict)`; `Judge.feed(t: float, bottle_pos, bottle_quat_wxyz, palms: np.ndarray (2, 3), pelvis_pos, pelvis_quat_wxyz, fallen: str | None) -> None`; `Judge.early() -> dict | None`; `Judge.final(episode_end: str) -> dict | None` (None = redo); `DEFAULTS: dict` (judge constants); verdict dicts are `{"verdict": "success"}` or `{"verdict": "fail", "reason": int, "note": str}` (reason = 1-based index into `eval_session.DEFAULT_FAILURE_REASONS`). `Judge.to_gate(v) -> dict` gives the `verdict.i` JSON (`reason` as a number, `note` string).
- Consumes: task config keys (Task 6 writes them): `table_top_z`, `bottle_radius`, `bottle_height`, `zone_center_pelvis` (3,), `zone_radius`, `grasp_hand` (0 left, 1 right).

- [ ] **Step 1: Write the failing tests**

`tests/datasets/test_g1_dex3_handover_success.py`:

```python
"""handover_success.Judge on synthetic trajectories (spec test 1)."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[2] / "examples" / "g1_dex3_training"))
import eval_session as es
import handover_success as hs

CFG = {**hs.DEFAULTS, "table_top_z": 0.80, "bottle_radius": 0.033, "bottle_height": 0.20,
       "zone_center_pelvis": [0.35, -0.20, 0.25], "zone_radius": 0.08, "grasp_hand": 1}  # fmt: skip
UP = np.array([1.0, 0.0, 0.0, 0.0])
PELVIS = np.array([0.0, 0.0, 0.79])
BOTTLE0 = np.array([0.45, -0.15, 0.80 + 0.10])  # centre; bottom on the table
FAR = np.array([[0.2, 0.3, 0.9], [0.2, -0.3, 0.9]])


def reason(name):
    return es.DEFAULT_FAILURE_REASONS.index(name) + 1


def run(frames, episode_end="completed"):
    j = hs.Judge(CFG)
    for t, (bottle, palms, quat, fallen) in enumerate(frames):
        j.feed(t * 0.02, bottle, quat, palms, PELVIS, UP, fallen)
        if (v := j.early()) is not None:
            return v
    return j.final(episode_end)


def palms_at(right):
    return np.array([[0.2, 0.3, 0.9], right])


def zone_world():
    return PELVIS + np.array(CFG["zone_center_pelvis"])


def lift_and_carry(n_hold):
    """reach (palm 4 cm from the axis), lift 15 cm, carry the bottle + palm into the zone, hold n_hold frames."""
    out = []
    grip = BOTTLE0 + [0.0, -0.04, 0.0]
    out += [(BOTTLE0, palms_at(grip), UP, None)] * 5
    lifted = BOTTLE0 + [0, 0, 0.15]
    out += [(lifted, palms_at(lifted + [0, -0.04, 0]), UP, None)] * 5
    zp = zone_world()
    out += [(zp + [0, 0.04, 0], palms_at(zp), UP, None)] * n_hold
    return out


def test_success_after_one_second_in_zone():
    assert run(lift_and_carry(51)) == {"verdict": "success"}


def test_success_is_latched_even_if_dropped_after():
    frames = lift_and_carry(51) + [(np.array([0.5, 0.0, 0.03]), FAR, UP, None)] * 5
    assert run(frames) == {"verdict": "success"}


def test_palm_in_zone_but_bottle_left_on_table_is_not_success():
    zp = zone_world()
    frames = [(BOTTLE0, palms_at(zp), UP, None)] * 80
    assert run(frames)["reason"] == reason("did not reach the object")


def test_never_near_is_reason_1():
    assert run([(BOTTLE0, FAR, UP, None)] * 20)["reason"] == reason("did not reach the object")


def test_reached_never_lifted_is_reason_2():
    frames = [(BOTTLE0, palms_at(BOTTLE0 + [0, -0.04, 0]), UP, None)] * 20
    assert run(frames)["reason"] == reason("grasp failed / missed")


def test_lift_then_floor_is_early_reason_3():
    frames = lift_and_carry(5) + [(np.array([0.5, 0.0, 0.03]), FAR, UP, None)]
    v = run(frames, episode_end="should not be used")
    assert v["reason"] == reason("dropped the object")


def test_lift_then_back_on_table_is_final_reason_3():
    frames = lift_and_carry(5) + [(BOTTLE0 + [0, 0, 0.01], FAR, UP, None)] * 3
    assert run(frames)["reason"] == reason("dropped the object")


def test_tipped_before_lift_is_reason_5():
    tipped = np.array([np.cos(np.pi / 4), np.sin(np.pi / 4), 0.0, 0.0])  # 90 deg about x
    assert run([(BOTTLE0, FAR, tipped, None)])["reason"] == reason("collision (table, knocked over)")


def test_off_the_table_before_lift_is_reason_5():
    assert run([(np.array([0.5, 0.0, 0.03]), FAR, UP, None)])["reason"] == reason("collision (table, knocked over)")


def test_lifted_held_outside_zone_is_reason_4():
    lifted = BOTTLE0 + [0, 0, 0.15]
    frames = [(lifted, palms_at(lifted + [0, -0.04, 0]), UP, None)] * 80
    v = run(frames)
    assert v["reason"] == reason("wrong object / target") and "zone" in v["note"]


def test_lifted_then_dipped_is_reason_6():
    lifted, dipped = BOTTLE0 + [0, 0, 0.15], BOTTLE0 + [0, 0, 0.05]
    frames = [(lifted, palms_at(lifted), UP, None)] * 3 + [(dipped, palms_at(dipped), UP, None)] * 3
    assert run(frames)["reason"] == reason("stalled / froze")


def test_watchdog_and_errors():
    frames = [(BOTTLE0, FAR, UP, None)] * 3
    assert run(frames, "watchdog: arm speed")["reason"] == reason("safety stop (watchdog)")
    assert run(frames, "no valid chunk for 2 s") is None
    assert run(frames, "robot state 0.50 s old") is None


def test_fall_is_other_with_note():
    v = run([(BOTTLE0, FAR, UP, "tilt 40 deg")])
    assert v["reason"] == reason("other") and v["note"] == "fell"


def test_to_gate():
    assert hs.Judge.to_gate({"verdict": "success"}) == {"verdict": "success", "reason": "", "note": ""}
    assert hs.Judge.to_gate(None) == {"verdict": "redo", "reason": "", "note": ""}
    assert hs.Judge.to_gate({"verdict": "fail", "reason": 3, "note": "x"}) == {"verdict": "fail", "reason": "3", "note": "x"}
```

- [ ] **Step 2: Run them to verify they fail**

Run: `cd $L && uv run pytest tests/datasets/test_g1_dex3_handover_success.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'handover_success'`.

- [ ] **Step 3: Write `handover_success.py`**

```python
"""Automatic verdict for handover_bottle trials in simulation (spec: Judge rules). Pure numpy.

Fed once per physics step. early() ends a trial at once (success, knocked over before a lift, dropped on the floor,
fall); the first early verdict is latched. final(episode_end) classifies the trial when the streamer reports its
end (ended.i). Reasons are 1-based indices into eval_session.DEFAULT_FAILURE_REASONS; final() returns None for redo.
"""

from __future__ import annotations

import numpy as np

try:
    from .eval_session import DEFAULT_FAILURE_REASONS
except ImportError:
    from eval_session import DEFAULT_FAILURE_REASONS

DEFAULTS = {
    "lift_m": 0.10,  # bottle bottom above the table top that counts as lifted
    "hold_s": 1.0,  # time in the zone for success
    "reach_margin_m": 0.04,  # palm-to-axis distance <= radius + margin counts as reached / carrying
    "down_m": 0.02,  # bottom within this of the table top after a lift: back on the table
    "floor_z_m": 0.10,  # bottle centre below this: on the floor
    "tip_deg": 60.0,  # bottle axis tilt that counts as knocked over (before a lift)
    "table_half_xy": [0.4, 0.8],  # table top half sizes (sim_scene table)
}
REDO_PREFIXES = ("no valid chunk", "robot state")


def reason(name: str) -> int:
    return DEFAULT_FAILURE_REASONS.index(name) + 1


def _rot(q) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])  # fmt: skip


class Judge:
    def __init__(self, cfg: dict):
        self.c = {**DEFAULTS, **cfg}
        self.reached = self.lifted = False
        self.zone_since: float | None = None
        self.latched: dict | None = None
        self.table_xy: np.ndarray | None = None  # table centre xy, set by the host when the table is placed
        self.last: dict = {}

    def _axis_distance(self, palm, bottle, quat) -> float:
        axis = _rot(quat)[:, 2]
        h = self.c["bottle_height"] / 2
        s = np.clip(np.dot(palm - bottle, axis), -h, h)
        return float(np.linalg.norm(palm - (bottle + s * axis)))

    def feed(self, t, bottle_pos, bottle_quat_wxyz, palms, pelvis_pos, pelvis_quat_wxyz, fallen) -> None:
        if self.latched is not None:
            return
        c = self.c
        bottle, quat, palms = np.asarray(bottle_pos, float), np.asarray(bottle_quat_wxyz, float), np.asarray(palms, float)
        bottom = bottle[2] - c["bottle_height"] / 2
        reach = c["bottle_radius"] + c["reach_margin_m"]
        near = [self._axis_distance(p, bottle, quat) <= reach for p in palms]
        self.reached |= any(near)
        lifted_now = bottom >= c["table_top_z"] + c["lift_m"]
        self.lifted |= lifted_now
        tilt = np.degrees(np.arccos(np.clip(_rot(quat)[2, 2], -1, 1)))
        self.last = {"bottom": bottom, "lifted_now": lifted_now}
        if fallen:
            self.latched = {"verdict": "fail", "reason": reason("other"), "note": "fell"}
            return
        if bottle[2] < c["floor_z_m"]:
            name = "dropped the object" if self.lifted else "collision (table, knocked over)"
            self.latched = {"verdict": "fail", "reason": reason(name), "note": "on the floor"}
            return
        off_table = self.table_xy is not None and np.any(
            np.abs(bottle[:2] - self.table_xy) > np.asarray(c["table_half_xy"])
        )
        if not self.lifted and (tilt > c["tip_deg"] or off_table):
            self.latched = {"verdict": "fail", "reason": reason("collision (table, knocked over)"), "note": ""}
            return
        palm = palms[c["grasp_hand"]]
        zone = np.asarray(pelvis_pos, float) + _rot(pelvis_quat_wxyz) @ np.asarray(c["zone_center_pelvis"], float)
        in_zone = lifted_now and near[c["grasp_hand"]] and np.linalg.norm(palm - zone) <= c["zone_radius"]
        if not in_zone:
            self.zone_since = None
        elif self.zone_since is None:
            self.zone_since = t
        elif t - self.zone_since >= c["hold_s"] - 1e-9:
            self.latched = {"verdict": "success"}

    def early(self) -> dict | None:
        return self.latched

    def final(self, episode_end: str) -> dict | None:
        if self.latched is not None:
            return self.latched
        if episode_end.startswith("watchdog"):
            return {"verdict": "fail", "reason": reason("safety stop (watchdog)"), "note": episode_end}
        if episode_end.startswith(REDO_PREFIXES):
            return None
        if not self.reached:
            return {"verdict": "fail", "reason": reason("did not reach the object"), "note": ""}
        if not self.lifted:
            return {"verdict": "fail", "reason": reason("grasp failed / missed"), "note": ""}
        if self.last["bottom"] <= self.c["table_top_z"] + self.c["down_m"]:
            return {"verdict": "fail", "reason": reason("dropped the object"), "note": "back on the table"}
        if self.last["lifted_now"]:
            return {"verdict": "fail", "reason": reason("wrong object / target"), "note": "not in handover zone"}
        return {"verdict": "fail", "reason": reason("stalled / froze"), "note": "lifted, never held"}

    @staticmethod
    def to_gate(v: dict | None) -> dict:
        """verdict.i JSON for eval_session.gate_verdict (None = redo)."""
        if v is None:
            return {"verdict": "redo", "reason": "", "note": ""}
        if v["verdict"] == "success":
            return {"verdict": "success", "reason": "", "note": ""}
        return {"verdict": "fail", "reason": str(v["reason"]), "note": v.get("note", "")}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `cd $L && uv run pytest tests/datasets/test_g1_dex3_handover_success.py -v`
Expected: 14 passed. If `test_palm_in_zone_but_bottle_left_on_table_is_not_success` fails with reason 2 instead of 1: the zone palm must be far from the bottle (it is 0.35/-0.20/0.25 from the pelvis vs bottle at 0.45/-0.15/0.90, about 13 cm), so check `_axis_distance`.

- [ ] **Step 5: Check `gate_verdict` accepts `to_gate` output**

Run:
```bash
cd $L && uv run python -c "
import sys, json, tempfile, pathlib; sys.path.insert(0, 'examples/g1_dex3_training')
import eval_session as es, handover_success as hs
d = pathlib.Path(tempfile.mkdtemp()); (d / 'verdict.0').write_text(json.dumps(hs.Judge.to_gate({'verdict': 'fail', 'reason': 3, 'note': 'x'})))
print(es.gate_verdict(d, 0, list(es.DEFAULT_FAILURE_REASONS)))"
```
Expected: `{'verdict': 'fail', 'reason': 'dropped the object', 'note': 'x'}`.

- [ ] **Step 6: Commit**

```bash
cd $L && git add examples/g1_dex3_training/handover_success.py tests/datasets/test_g1_dex3_handover_success.py
git commit -m "feat(g1-dex3): automatic handover_bottle judge for sim trials

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: full Isaac scene behind the DDS bridge, host SIM switch, test 0b (paced lag)

**Files:**
- Modify: `examples/g1_dex3_training/isaac_scene.py`
- Modify: `examples/g1_dex3_training/sim_scene.py` (add `key`, `bridge`)
- Modify: `examples/g1_dex3_training/sonic_official_sim_host.py:24-40,80-148`
- Test: `tests/datasets/test_g1_dex3_isaac_scene_pure.py` (add quaternion and obs tests)

**Interfaces:**
- Consumes: Task 1 `Scene`, `pd_torques`.
- Produces: `isaac_scene.quat_wxyz_to_matrix(q) -> (3, 3)`; `isaac_scene.make_obs(state: dict) -> dict` (bridge `PublishLowState` dict from plain arrays); `Scene.bridge` (gear_sonic `UnitreeSdk2Bridge`), `Scene.key(name)` (`"9"`, `"backspace"`), `Scene.record(**extra)`, `Scene.check_fall()`, `Scene.floor_contacts()`, `Scene.table_clearance()`, `Scene.palms() -> (2, 3)`, `Scene.bottle_state() -> (pos (3,), quat (4,))`, `Scene.pelvis_state() -> (pos, quat)`, `Scene.reset_trial(seed, bottle_xy, jitter_m)`, `Scene.cut_trial(i)`, `Scene.lag_ms: list[float]`; `sim_scene.Scene.key(name)`, `sim_scene.Scene.bridge`.

- [ ] **Step 1: Install the DDS stack into the `sonic` env**

```bash
CYCLONEDDS_HOME=$HOME/work/cyclonedds_v0.10.2 /home/jihun/anaconda3/envs/sonic/bin/pip install cyclonedds==0.10.2
/home/jihun/anaconda3/envs/sonic/bin/pip install -e $G/external_dependencies/unitree_sdk2_python --no-deps
$PY -c "import unitree_sdk2py, cyclonedds; from unitree_sdk2py.core.channel import ChannelFactoryInitialize; print('ok')"
$PY -c "import sys; sys.path.insert(0, '$G'); from gear_sonic.utils.mujoco_sim.unitree_sdk2py_bridge import UnitreeSdk2Bridge; from gear_sonic.utils.mujoco_sim.configs import SimLoopConfig; print('ok')"
```
Expected: `ok` twice. If the second import fails on a missing package, `pip install` exactly that package (the same version as in `$G/.venv_sim`: `$G/.venv_sim/bin/pip show <pkg>`), nothing else.

- [ ] **Step 2: Write the failing pure tests**

Append to `tests/datasets/test_g1_dex3_isaac_scene_pure.py`:

```python
def test_quat_wxyz_to_matrix():
    np.testing.assert_allclose(isc.quat_wxyz_to_matrix([1, 0, 0, 0]), np.eye(3), atol=1e-12)
    r = isc.quat_wxyz_to_matrix([np.cos(np.pi / 4), 0, 0, np.sin(np.pi / 4)])  # +90 deg about z
    np.testing.assert_allclose(r @ [1, 0, 0], [0, 1, 0], atol=1e-12)


def test_make_obs_has_every_bridge_key_and_splits_hands():
    nb, nh = len(isc.BODY), len(isc.HANDS)
    state = {"q": np.arange(nb + nh, dtype=float), "dq": np.zeros(nb + nh), "ddq": np.zeros(nb + nh),
             "tau": np.zeros(nb + nh), "root_pose": np.r_[0, 0, 0.79, 1, 0, 0, 0], "root_vel": np.zeros(6),
             "root_acc": np.zeros(6), "torso_quat": np.r_[1, 0, 0, 0], "torso_vel": np.zeros(6), "time": 1.5}  # fmt: skip
    obs = isc.make_obs(state)
    for k in ("body_q", "body_dq", "body_ddq", "body_tau_est", "floating_base_pose", "floating_base_vel",
              "floating_base_acc", "secondary_imu_quat", "secondary_imu_vel", "left_hand_q", "left_hand_dq",
              "right_hand_q", "right_hand_dq", "time"):  # fmt: skip
        assert k in obs, k
    assert obs["body_q"].shape == (nb,) and obs["left_hand_q"][0] == nb and obs["right_hand_q"][0] == nb + 7
```

Run: `cd $L && uv run pytest tests/datasets/test_g1_dex3_isaac_scene_pure.py -v`
Expected: the 2 new tests FAIL with `AttributeError: module 'isaac_scene' has no attribute 'quat_wxyz_to_matrix'`.

- [ ] **Step 3: Add the pure helpers to `isaac_scene.py`** (below `zero_cmd`)

```python
def quat_wxyz_to_matrix(q) -> np.ndarray:
    w, x, y, z = (float(v) for v in q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])  # fmt: skip


def make_obs(s: dict) -> dict:
    """UnitreeSdk2Bridge.PublishLowState input from plain arrays (joints in BODY + HANDS order, hands left 7 then
    right 7 in the bridge's motor order; quaternions w, x, y, z; velocities [lin 3, ang 3])."""
    nb, nh = len(BODY), len(HANDS) // 2
    obs = {
        "body_q": s["q"][:nb], "body_dq": s["dq"][:nb], "body_ddq": s["ddq"][:nb], "body_tau_est": s["tau"][:nb],
        "floating_base_pose": s["root_pose"], "floating_base_vel": s["root_vel"], "floating_base_acc": s["root_acc"],
        "secondary_imu_quat": s["torso_quat"], "secondary_imu_vel": s["torso_vel"], "time": s["time"],
    }  # fmt: skip
    for side, sl in (("left", slice(nb, nb + nh)), ("right", slice(nb + nh, nb + 2 * nh))):
        obs[f"{side}_hand_q"], obs[f"{side}_hand_dq"] = s["q"][sl], s["dq"][sl]
    return obs
```

Hand order: `sim_scene.HANDS` is the *dataset* order (comment in `sim_scene.py`: right hand thumb0-2, index0-1, middle0-1), the bridge's Dex3 motor order is thumb0-2, middle0-1, index0-1 per hand (`--dex3-right-order swap` exists for this). Use the bridge order for the articulation map: define in `isaac_scene.py`

```python
DEX3_MOTORS = [f"{s}_hand_{j}_joint" for s in ("left", "right")
               for j in ("thumb_0", "thumb_1", "thumb_2", "middle_0", "middle_1", "index_0", "index_1")]  # fmt: skip
```

and replace `HANDS` by `DEX3_MOTORS` in `Scene.__init__` (`find_joints`) and in `make_obs` (length only). Before committing, confirm this order against `gear_sonic/utils/mujoco_sim/base_sim.py` (`left_hand_index` construction) with
`grep -n "hand_index\|HAND_JOINT\|hand_joint" $G/gear_sonic/utils/mujoco_sim/base_sim.py $G/gear_sonic/utils/mujoco_sim/robot.py`
and make `DEX3_MOTORS` match the MuJoCo actuator order exactly. The startup check in Step 5 compares the two.

Run the pure tests again. Expected: all pass.

- [ ] **Step 4: Wire the bridge, keys, recording and fall check into `Scene`**

Add to `Scene.__init__` after `self.sim.reset()` (DDS on `lo`, domain 0, as `g1_29dof_sonic_model12.yaml`):

```python
        sys.path.insert(0, str(GEAR_ROOT))
        from gear_sonic.utils.mujoco_sim.configs import SimLoopConfig
        from gear_sonic.utils.mujoco_sim.unitree_sdk2py_bridge import UnitreeSdk2Bridge
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize

        self.config = SimLoopConfig(interface="sim", enable_onscreen=False, hand_profile="dex3").load_wbc_yaml()
        ChannelFactoryInitialize(self.config["DOMAIN_ID"], self.config["INTERFACE"])
        self.bridge = UnitreeSdk2Bridge(self.config)
        self.torso_id = self.robot.find_bodies("torso_link")[0][0]
        self.pelvis_id = self.robot.find_bodies("pelvis")[0][0]
        self.palm_ids = [self.robot.find_bodies([f"{s}_hand_{f}_0_link" for f in ("index", "middle")],
                                                preserve_order=True)[0] for s in ("left", "right")]  # fmt: skip
        self.wrist_ids = [self.robot.find_bodies(f"{s}_wrist_yaw_link")[0][0] for s in ("left", "right")]
        self.prev_root_vel = np.zeros(6)
        self.lag_ms, self.wall0 = [], None
        self.pinned_root = np.array([0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0])  # band: hang until "backspace"
```

Add these methods:

```python
    def _bridge_cmd(self) -> None:
        """self.cmd from the latest rt/lowcmd and Dex3 commands (zero until the deploy has sent any)."""
        br = self.bridge
        if not br.cmd_received():
            return
        nb, nh = len(BODY), len(DEX3_MOTORS) // 2
        motors = [br.low_cmd.motor_cmd[i] for i in range(nb)]
        motors += [br.left_hand_cmd.motor_cmd[i] for i in range(nh)] + [br.right_hand_cmd.motor_cmd[i] for i in range(nh)]
        for k, attr in (("q", "q"), ("dq", "dq"), ("kp", "kp"), ("kd", "kd"), ("tau", "tau")):
            self.cmd[k] = np.array([getattr(m, attr) for m in motors], dtype=float)

    def _state(self) -> dict:
        d, ids = self.robot.data, self.body_ids + self.hand_ids
        root = d.root_state_w[0].cpu().numpy()
        root_vel = root[7:13]
        acc = (root_vel - self.prev_root_vel) / DT
        self.prev_root_vel = root_vel.copy()
        torso_lin = d.body_lin_vel_w[0, self.torso_id].cpu().numpy()
        torso_ang = d.body_ang_vel_w[0, self.torso_id].cpu().numpy()
        return {
            "q": d.joint_pos[0, ids].cpu().numpy(), "dq": d.joint_vel[0, ids].cpu().numpy(),
            "ddq": d.joint_acc[0, ids].cpu().numpy(), "tau": getattr(self, "tau", np.zeros(len(ids))),
            "root_pose": root[:7], "root_vel": root_vel, "root_acc": acc,
            "torso_quat": d.body_quat_w[0, self.torso_id].cpu().numpy(), "torso_vel": np.r_[torso_lin, torso_ang],
            "time": self.t,
        }  # fmt: skip

    def key(self, name: str) -> None:
        """The two simulator keys the host uses: '9' releases the band (gear_sonic: the robot drops), 'backspace'
        resets the robot standing on the floor. Band = root pinned at 1.0 m; release frees it; reset writes the
        default standing state (root at STAND_Z, joints at DEFAULT_DOF_ANGLES, hands 0, velocities 0)."""
        import torch

        with self.lock:
            if name == "9":
                self.pinned_root = None
            elif name == "backspace":
                self.pinned_root = None
                q = np.zeros(len(BODY) + len(DEX3_MOTORS))
                q[: len(BODY)] = self.config["DEFAULT_DOF_ANGLES"]
                dev = self.sim.device
                self.robot.write_root_pose_to_sim(torch.tensor([[0.0, 0.0, STAND_Z, 1.0, 0.0, 0.0, 0.0]], device=dev))
                self.robot.write_root_velocity_to_sim(torch.zeros(1, 6, device=dev))
                pos = self.robot.data.default_joint_pos.clone()
                pos[0, self.body_ids + self.hand_ids] = torch.tensor(q, dtype=torch.float32, device=dev)
                self.robot.write_joint_state_to_sim(pos, torch.zeros_like(pos))
            else:
                raise ValueError(f"unsupported key {name!r}")

    def palms(self) -> np.ndarray:
        p = self.robot.data.body_pos_w[0].cpu().numpy()
        return np.stack([p[ids].mean(0) for ids in self.palm_ids])

    def bottle_state(self) -> tuple[np.ndarray, np.ndarray]:
        s = self.bottle.data.root_state_w[0].cpu().numpy()
        return s[:3], s[3:7]

    def pelvis_state(self) -> tuple[np.ndarray, np.ndarray]:
        s = self.robot.data.root_state_w[0].cpu().numpy()
        return s[:3], s[3:7]

    def floor_contacts(self) -> int:
        return int((self.feet.data.net_forces_w[0].norm(dim=-1) > 1.0).sum())

    def table_clearance(self):
        """Lowest wrist/hand body origin above the table top (m) and 0 hits; nan/0 without a table.
        ponytail: body origins, not collision geometry like the MuJoCo scene; add a contact sensor on the hands if
        the clearance diagnostics are needed for Isaac runs."""
        if self.table_gap is None:
            return np.nan, 0
        z = self.robot.data.body_pos_w[0, self.wrist_ids + sum(self.palm_ids, []), 2].cpu().numpy()
        return float(z.min() - 0.80), 0

    def record(self, **extra) -> None:
        if self.t + 1e-9 < self.next_rec:
            return
        self.next_rec = self.t + 0.02
        with self.lock:
            s, quats = self._state(), self.robot.data.body_quat_w[0].cpu().numpy()
            clear, hits = self.table_clearance()
            bottle_pos, bottle_quat = self.bottle_state()
            row = {
                "wall": time.time(), "t": self.t, "body_q": s["q"][: len(BODY)].copy(),
                "hand_q": s["q"][len(BODY):].copy(), "pelvis": s["root_pose"].copy(), "palm": self.palms(),
                "wrist_R": np.stack([quat_wxyz_to_matrix(quats[i]) for i in self.wrist_ids]),
                "torso_R": quat_wxyz_to_matrix(quats[self.torso_id]), "floor_contacts": self.floor_contacts(),
                "table_clear": clear, "table_hits": hits, "bottle_pos": bottle_pos, "bottle_quat": bottle_quat,
            }  # fmt: skip
        row.update(extra)
        for k, v in row.items():
            self.rec.setdefault(k, []).append(v)

    def check_fall(self) -> str | None:
        pos, quat = self.pelvis_state()
        return self.fall.update(self.t, quat, float(pos[2]), self.floor_contacts())
```

In `step()`, before `pd_torques`: call `self._bridge_cmd()`; after `self.t += DT`: `self.bridge.PublishLowState(make_obs(self._state()))`. In `pace()`, append the lag: replace the body with

```python
        if self.wall0 is None:
            self.wall0 = wall0
        ahead = self.t - (time.monotonic() - wall0)
        self.lag_ms.append(max(0.0, -ahead) * 1e3)
        if ahead > 0:
            time.sleep(ahead)
```

Note `record()`: the `"hand_q"` column is now in bridge motor order (`DEX3_MOTORS`), while the MuJoCo scene records dataset order (`HANDS`). Reorder before storing so the two scenes write the same columns:
`hand_perm = [DEX3_MOTORS.index(n) for n in HANDS]` once in `__init__`, and store `s["q"][len(BODY):][hand_perm]`.

- [ ] **Step 5: Startup checks**

At the end of `Scene.__init__`:

```python
        names = self.robot.joint_names
        missing = [n for n in BODY + DEX3_MOTORS if n not in names]
        if missing:
            raise SystemExit(f"Isaac articulation lacks {missing}")
        if len(names) != len(BODY) + len(DEX3_MOTORS):
            raise SystemExit(f"Isaac articulation has {len(names)} joints, expected {len(BODY) + len(DEX3_MOTORS)}")
        if self.config["NUM_MOTORS"] != len(BODY) or self.config["NUM_HAND_MOTORS"] != len(DEX3_MOTORS) // 2:
            raise SystemExit("gear_sonic config motor counts differ from BODY / DEX3_MOTORS")
```

and in `_publish_frame`, before encoding: `if rgb.shape[:2] != (H, W): raise SystemExit(f"head camera gives {rgb.shape[1]}x{rgb.shape[0]}, not {W}x{H}")`.

- [ ] **Step 6: Give `sim_scene.Scene` the same two hooks** (so the host never touches `scene.sim` / `scene.env`)

In `sim_scene.py`, class `Scene`, add:

```python
    @property
    def bridge(self):
        return self.env.unitree_bridge

    def key(self, name: str) -> None:
        self.sim.handle_keyboard_button(name)
```

- [ ] **Step 7: Host `SIM` switch, local deploy, lag log**

In `sonic_official_sim_host.py`:

1. Replace `from sim_scene import Scene` with:

```python
SIM = os.environ.get("SIM", "mujoco")
if SIM == "isaac":
    from isaac_scene import GEAR_ROOT, Scene

    os.chdir(GEAR_ROOT / "gear_sonic_deploy")  # the deploy paths below are relative to gear_sonic_deploy
elif SIM == "mujoco":
    from sim_scene import Scene
else:
    raise SystemExit(f"SIM must be mujoco or isaac, not {SIM!r}")
```

2. Replace `scene.sim.handle_keyboard_button("9")` with `scene.key("9")`, `scene.sim.handle_keyboard_button("backspace")` with `scene.key("backspace")`, and in `log_hand_gains` `br = scene.env.unitree_bridge` with `br = scene.bridge`.
3. In the `finally:` block, before `scene.close(...)`, add (Isaac only has `lag_ms`):

```python
    lag = getattr(scene, "lag_ms", None)
    if lag:
        lag_a = np.asarray(lag)
        scene.mark(f"lag vs wall clock: max {lag_a.max():.1f} ms, p99 {np.percentile(lag_a, 99):.1f} ms, "
                   f"steps over 20 ms {int((lag_a > 20).sum())} of {lag_a.size}")  # fmt: skip
        (OUT / "lag.json").write_text(json.dumps({"max_ms": float(lag_a.max()), "p99_ms": float(np.percentile(lag_a, 99)),
                                                  "over_20ms": int((lag_a > 20).sum()), "steps": int(lag_a.size)}))  # fmt: skip
```

with `import json` and `import numpy as np` at the top.

- [ ] **Step 8: Run the pure tests and the existing host-related tests**

Run: `cd $L && uv run pytest tests/datasets/test_g1_dex3_isaac_scene_pure.py tests/datasets/test_g1_dex3_wbc_common.py tests/datasets/test_g1_dex3_stream_backends.py -v`
Expected: all pass (the MuJoCo path is unchanged apart from `key` / `bridge`).

- [ ] **Step 9: Test 0b — one replay with the deploy, paced lag**

Terminal 1 (host, GPU 0):
```bash
cd $D && mkdir -p /mnt/data/jihun/g1_isaac_eval/t0b/{sim,gate}
SIM=isaac TABLE_GAP_CM=5 CUDA_VISIBLE_DEVICES=0 $PY -u sonic_official_sim_host.py \
  /mnt/data/jihun/g1_isaac_eval/t0b/sim /mnt/data/jihun/g1_isaac_eval/t0b/gate
```
Datasets (already on the workstation): `DS=/mnt/data/jihun/datasets/G1_WBT_GR00T/handover_bottle_sonic78_50hz_261002`, joint28 for scoring `J28=/mnt/data/jihun/datasets/G1_WBT_GR00T/handover_bottle_joint28ref_261002`, held-out list `$DS.heldout.json` (must contain 27 and 35; check before running).

Terminal 2 (streamer, replay of held-out episode 27; the flags of the MuJoCo replays in `sonic_official_sim_eval.sh` with `REPLAY=1`, backend A, plus the batch flags of `g1_sim_eval_20261002/sim_batch.sh`):
```bash
cd $L && uv run python examples/g1_dex3_training/sonic_policy_streamer.py --replay --dataset-root $DS --episode 27 \
  --state-source robot --max-chunk-age-s 2.0 --gate-dir /mnt/data/jihun/g1_isaac_eval/t0b/gate \
  --log /mnt/data/jihun/g1_isaac_eval/t0b/streamer.npz --dex3-right-order swap --start planner --max-token-step 0.05
```
If the streamer rejects a flag in replay mode, drop that flag (replay has no policy server) and note it in the commit message.
Expected: host log shows `Init Done -> deploy_ready`, `deploy in CONTROL`, `band released`, `reset onto ground`, `settled`, the streamer replays to `completed`, and `lag.json` has `max_ms <= 20` (spec test 0b).

Checks:
- The robot falls right after `reset onto ground`: joint order or `STAND_Z`. Print `self.robot.data.root_state_w[0, 2]` 1 s after the reset; set `STAND_Z` to that measured standing height and rerun.
- `max_ms > 20`: report `lag.json` and the probe's `frame_ms_p95`; this is the spec's stop condition as well (move to the H100). Stop and report.

- [ ] **Step 10: Commit**

```bash
cd $L && git add examples/g1_dex3_training/isaac_scene.py examples/g1_dex3_training/sim_scene.py \
  examples/g1_dex3_training/sonic_official_sim_host.py tests/datasets/test_g1_dex3_isaac_scene_pure.py
git commit -m "feat(g1-dex3): Isaac scene behind the SONIC deploy's DDS interface; host SIM switch and lag log (spec test 0b)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: `ended.{i}`, host trials loop with the judge, reset stability (test 4)

**Files:**
- Modify: `examples/g1_dex3_training/eval_session.py` (add three helpers)
- Modify: `examples/g1_dex3_training/sonic_policy_streamer.py:1036-1040`
- Modify: `examples/g1_dex3_training/isaac_scene.py` (add `reset_trial`, `cut_trial`)
- Modify: `examples/g1_dex3_training/sonic_official_sim_host.py` (trials loop)
- Test: `tests/datasets/test_g1_dex3_eval_session.py`

**Interfaces:**
- Produces: `eval_session.write_ended(gate_dir: Path, trial: int, episode_end: str) -> None`; `eval_session.read_ended(gate_dir: Path, trial: int) -> str | None` (deletes the file once read); `eval_session.clear_trial_files(gate_dir: Path, trial: int) -> None` (removes `ended.i`, `stop.i`, `settled.i`, `verdict.i`); `Scene.reset_trial(seed: int, bottle_xy: tuple[float, float], jitter_m: float) -> None`; `Scene.cut_trial(i: int) -> None` (writes `sim_state_{i:03d}.npz`, clears `rec`).
- Consumes: Task 2 `Judge`, Task 3 `Scene`.

- [ ] **Step 1: Write the failing tests** (append to `tests/datasets/test_g1_dex3_eval_session.py`)

```python
def test_ended_roundtrip_and_clear(tmp_path):
    es.write_ended(tmp_path, 3, "watchdog: arm speed")
    assert es.read_ended(tmp_path, 2) is None
    assert es.read_ended(tmp_path, 3) == "watchdog: arm speed"
    assert es.read_ended(tmp_path, 3) is None  # consumed


def test_clear_trial_files(tmp_path):
    for name in ("ended.4", "stop.4", "settled.4", "verdict.4", "ended.5"):
        (tmp_path / name).write_text("x")
    es.clear_trial_files(tmp_path, 4)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["ended.5"]
```

Run: `cd $L && uv run pytest tests/datasets/test_g1_dex3_eval_session.py -v -k "ended or clear"`
Expected: FAIL with `AttributeError: module 'eval_session' has no attribute 'write_ended'`.

- [ ] **Step 2: Implement the helpers in `eval_session.py`** (after `start_trial_input`)

```python
def write_ended(gate_dir: Path, trial: int, episode_end: str) -> None:
    """--trials with --gate-dir: tell the sim host how trial `trial` ended, before waiting for verdict.{trial}."""
    tmp = Path(gate_dir) / f".ended.{trial}.tmp"
    tmp.write_text(episode_end)
    tmp.replace(Path(gate_dir) / f"ended.{trial}")  # atomic: the host never reads a half-written file


def read_ended(gate_dir: Path, trial: int) -> str | None:
    path = Path(gate_dir) / f"ended.{trial}"
    if not path.exists():
        return None
    text = path.read_text()
    path.unlink()
    return text


def clear_trial_files(gate_dir: Path, trial: int) -> None:
    """Sim host, before a trial: drop gate files of trial `trial` left by a crash or an earlier run."""
    for name in ("ended", "stop", "settled", "verdict"):
        (Path(gate_dir) / f"{name}.{trial}").unlink(missing_ok=True)
```

Run the two tests. Expected: pass.

- [ ] **Step 3: Streamer change** (`sonic_policy_streamer.py`, in the trials loop right after `episode_end, frames = run_episode(trial)` and its print)

```python
            if a.gate_dir:
                write_ended(a.gate_dir, trial, episode_end)
```

and add `write_ended` to both `from eval_session import (...)` / `from .eval_session import (...)` lists (the file imports `write_done` from `wbc_common` the same way around L99 and L133; put `write_ended` in the `eval_session` import group).

Run: `cd $L && uv run pytest tests/datasets/test_g1_dex3_streamer_cli.py tests/datasets/test_g1_dex3_eval_trials.py -v`
Expected: pass. If `test_g1_dex3_eval_trials.py` fakes the gate dir and now sees an extra `ended.i` file, update that test's expected file list (and only that).

- [ ] **Step 4: `reset_trial` and `cut_trial` in `isaac_scene.py`**

```python
    def reset_trial(self, seed: int, bottle_xy: tuple[float, float], jitter_m: float) -> None:
        """Robot back to the deploy's standing pose (as key 'backspace'); bottle on the table at bottle_xy (world,
        table placed) + uniform +-jitter_m in x and y, upright; trial records cleared."""
        import torch

        self.key("backspace")
        rng = np.random.default_rng(seed)
        x, y = np.asarray(bottle_xy) + rng.uniform(-jitter_m, jitter_m, 2)
        z = 0.80 + 0.10  # table top 0.80 (centre 0.785 + 0.015) + half the bottle height
        with self.lock:
            dev = self.sim.device
            self.bottle.write_root_pose_to_sim(torch.tensor([[x, y, z, 1.0, 0.0, 0.0, 0.0]], device=dev))
            self.bottle.write_root_velocity_to_sim(torch.zeros(1, 6, device=dev))
        self.rec = {k: [] for k in REC_KEYS}
        self.next_rec = self.t
        self.fall = FallDetector()
        self.mark(f"trial reset: seed {seed}, bottle at ({x:.3f}, {y:.3f})")

    def cut_trial(self, i: int) -> None:
        if self.rec["t"]:
            np.savez_compressed(self.out / f"sim_state_{i:03d}.npz", **{k: np.array(v) for k, v in self.rec.items()})
        self.rec = {k: [] for k in REC_KEYS}
```

(`video_{i:03d}.mp4`: not in this task; the session already stores the head-camera JPEGs per trial (`trial_{i:03d}_frames.npz`). ponytail: add a third-person video writer if reviewing failures from the head view alone is not enough.)

- [ ] **Step 5: Host trials loop** (`sonic_official_sim_host.py`)

After the deploy has reached `settled` for the first time (the existing `t_settled` branch), when `TRIALS` is set, switch to a trial loop instead of waiting for `GATE/done`. Add at the top:

```python
TRIALS = int(os.environ.get("TRIALS", "0"))
SESSION_DIR = Path(os.environ["SESSION_DIR"]) if TRIALS else None
SEED = int(os.environ.get("SEED", "0"))
```

and replace the body of the main `while` loop from `if t_done is None and (GATE / "done").exists():` down to the `scene.record()` block with a branch:

```python
        if TRIALS and t_settled is not None:
            reason, detail = run_trials(scene, proc)  # returns when all trials are judged or the deploy dies
            break
```

where `run_trials` is a new function in the same file:

```python
def run_trials(scene, proc):
    """Spec data flow per trial. The streamer runs --trials TRIALS --gate-dir GATE --session-dir SESSION_DIR."""
    from eval_session import clear_trial_files, next_trial, read_ended
    from handover_success import Judge

    task = json.loads((Path(__file__).parent / "tasks" / "handover_bottle.json").read_text())
    trial = next_trial(SESSION_DIR)
    while trial < TRIALS:
        if proc.poll() is not None:
            return "error", f"deploy exited during trial {trial}"
        clear_trial_files(GATE, trial)
        for attempt in (1, 2):
            scene.reset_trial(SEED + trial, task["bottle_xy_world"], task["jitter_m"])
            t0 = scene.t
            while scene.t < t0 + 5:
                scene.step()
                scene.pace(wall0)
            if scene.check_fall() is None and scene.pelvis_state()[0][2] > 0.6:
                break
            if attempt == 2:
                return "reset_failed", f"robot down after two resets before trial {trial}"
        scene.fall.armed = True
        judge = Judge(task["judge"])
        judge.table_xy = np.asarray(task["table_xy_world"])
        (GATE / f"settled.{trial}").touch()
        scene.mark(f"trial {trial}: settled.{trial}")
        verdict, early_written = None, False
        while True:
            scene.step()
            scene.record()
            bottle_pos, bottle_quat = scene.bottle_state()
            pelvis_pos, pelvis_quat = scene.pelvis_state()
            judge.feed(scene.t, bottle_pos, bottle_quat, scene.palms(), pelvis_pos, pelvis_quat, scene.check_fall())
            if not early_written and judge.early() is not None:
                (GATE / f"stop.{trial}").touch()
                early_written = True
                scene.mark(f"trial {trial}: early verdict {judge.early()}")
            ended = read_ended(GATE, trial)
            if ended is not None:
                verdict = judge.final(ended)
                break
            if proc.poll() is not None:
                return "error", f"deploy exited during trial {trial}"
            scene.pace(wall0)
        gate = Judge.to_gate(verdict)
        tmp = GATE / f".verdict.{trial}.tmp"
        tmp.write_text(json.dumps(gate))
        tmp.replace(GATE / f"verdict.{trial}")
        scene.cut_trial(trial)
        scene.mark(f"trial {trial}: ended '{ended}', verdict {gate}")
        if gate["verdict"] != "redo":
            trial += 1
    return "completed", ""
```

Task-config keys used here (`bottle_xy_world`, `table_xy_world`, `jitter_m`, `judge`) are written in Task 6. Until then, Task 4's run uses a hand-written `tasks/handover_bottle.json` with the robot at the origin and `TABLE_GAP_CM=5`: table centre x = 0.08 + 0.05 + 0.4 = 0.53 m, `table_xy_world` [0.53, 0.0], `bottle_xy_world` [0.25, -0.15] (0.12 m behind the near edge at 0.13 m, right of centre), `jitter_m` 0.03, `judge` = `{"table_top_z": 0.80, "bottle_radius": 0.033, "bottle_height": 0.20, "zone_center_pelvis": [0.35, -0.20, 0.25], "zone_radius": 0.08, "grasp_hand": 1}`, `"source": "placeholder until derive_handover_task.py (Task 6)"`.

The streamer side for trials: `sonic_policy_streamer.py ... --trials TRIALS --gate-dir GATE --session-dir SESSION_DIR --images zmq --camera-host localhost --camera-port 5555 --task "pick drink bottle and hand over" --duration-s 40 --state-source robot` plus the policy-server flags of the robot runs (`g1_groot_real_run.sh`); Task 7 puts this in the launcher.

- [ ] **Step 6: Run the unit tests**

Run: `cd $L && uv run pytest tests/datasets/test_g1_dex3_eval_session.py tests/datasets/test_g1_dex3_eval_trials.py tests/datasets/test_g1_dex3_handover_success.py -v`
Expected: all pass.

- [ ] **Step 7: Test 4 — reset stability with the policy idle**

Run the host with `TRIALS=10`, `SESSION_DIR=/mnt/data/jihun/g1_isaac_eval/t4_session`, and a streamer in `--trials 10 --gate-dir ... --session-dir ... --replay --duration-s 5` mode (replay of a still segment is enough: the point is ten resets under a running deploy).
Expected: 10 trials judged (verdicts will be fails, reason 1); host `events.txt` has no `reset_failed`; for each trial, the pelvis xy in `sim_state_{i}.npz` moves less than 5 cm and the yaw less than 5 deg during the 5 s settle (check with a 5-line numpy script over the ten files; the spec's odometry concern: the planner must not step or turn after the teleport).

If the robot steps or turns after a reset: the planner holds an odometry target from before the teleport. Use the spec's fallback (one full stack per trial, `--trials i+1`): run `TRIALS=1` per process from a shell loop over i with the same `SESSION_DIR`; record that decision in the result doc (Task 7).

- [ ] **Step 8: Commit**

```bash
cd $L && git add examples/g1_dex3_training/eval_session.py examples/g1_dex3_training/sonic_policy_streamer.py \
  examples/g1_dex3_training/isaac_scene.py examples/g1_dex3_training/sonic_official_sim_host.py \
  examples/g1_dex3_training/tasks/handover_bottle.json tests/datasets/test_g1_dex3_eval_session.py
git commit -m "feat(g1-dex3): sim trials: streamer reports ended.i, host judges each trial; reset stability checked

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: replay parity, gate GI (test 2)

**Files:**
- Modify: `examples/g1_dex3_training/wbc_compare.py` (add `gi`, wire into `main`)
- Test: `tests/datasets/test_g1_dex3_wbc_compare_gi.py`

**Interfaces:**
- Consumes: `wbc_compare.load`, `metric`, `uncovered`, `episodes_of`, `mean`; run dirs `WBC_I28_ep<E>_r<R>` (Isaac) and `WBC_A28_ep<E>_r<R>` (MuJoCo A), each with `stream_eval.json` from `sonic_stream_eval.py`.
- Produces: `wbc_compare.gi(runs, test="I28", ref="A28", episodes=None) -> dict` with `{"pass": bool, "rows": [...]}`.

- [ ] **Step 1: Write the failing test**

```python
"""wbc_compare.gi: Isaac vs MuJoCo backend A replay parity (spec test 2)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2] / "examples" / "g1_dex3_training"))
import wbc_compare as wc


def res(p50, p95, tilt, valid=True):
    return {"valid": valid, "palm_err_vs_original_cm": {"p50": p50, "p95": p95}, "stability": {"tilt_max_deg": tilt}}


def runs(i, a):
    return {"I28": {27: i}, "A28": {27: a}}


def test_gi_pass_and_each_failure():
    a = [res(2.0, 4.0, 5.0)] * 3
    assert wc.gi(runs([res(2.5, 5.5, 6.0)] * 3, a))["pass"]
    assert not wc.gi(runs([res(3.2, 5.5, 6.0)] * 3, a))["pass"]  # p50 diff 1.2 cm
    assert not wc.gi(runs([res(2.5, 6.5, 6.0)] * 3, a))["pass"]  # p95 diff 2.5 cm
    assert not wc.gi(runs([res(2.5, 5.5, 7.5)] * 3, a))["pass"]  # tilt +2.5 deg
    assert not wc.gi(runs([res(2.5, 5.5, 6.0)] * 2 + [res(0, 0, 0, valid=False)], a))["pass"]
    assert not wc.gi(runs([res(2.5, 5.5, 6.0)] * 2, a))["pass"]  # fewer than REPEATS
```

Run: `cd $L && uv run pytest tests/datasets/test_g1_dex3_wbc_compare_gi.py -v`
Expected: FAIL with `AttributeError: module 'wbc_compare' has no attribute 'gi'`.

- [ ] **Step 2: Implement `gi`** (after `g2` in `wbc_compare.py`; also add a `GI` line to the module docstring's gate list: `GI  Isaac parity: |I28 - A28| <= 1.0 cm palm p50, <= 2.0 cm p95 per episode; every repeat valid; mean max tilt <= A28 + 2 deg`)

```python
def gi(runs, test="I28", ref="A28", episodes=None) -> dict:
    rows = []
    for ep in episodes_of(runs, episodes, test, ref):
        i_runs, a_runs = runs.get(test, {}).get(ep, []), runs.get(ref, {}).get(ep, [])
        row = {"episode": ep}
        if why := uncovered(**{test: i_runs, ref: a_runs}):
            rows.append({**row, "pass": False, "why": why})
            continue
        ok = True
        for q, tol in (("p50", 1.0), ("p95", 2.0)):
            i, a = metric(i_runs, "palm_err_vs_original_cm", q), metric(a_runs, "palm_err_vs_original_cm", q)
            diff = abs(np.mean(i) - np.mean(a))
            row[f"{q}_diff_cm"] = round(float(diff), 2)
            ok = ok and diff <= tol
        it, at = metric(i_runs, "stability", "tilt_max_deg"), metric(a_runs, "stability", "tilt_max_deg")
        row.update(tilt=mean(it), tilt_ref=mean(at))
        ok = ok and bool(it) and bool(at) and np.mean(it) <= np.mean(at) + 2
        rows.append({**row, "pass": bool(ok)})
    return {"pass": bool(rows) and all(x["pass"] for x in rows), "rows": rows}
```

In `main()`, add `"GI": gi(runs, episodes=a.episodes)` to the `out` dict next to G0/G1/G2.

Run the test. Expected: pass.

- [ ] **Step 3: Produce the runs**

- MuJoCo A: `REPLAY=1` of episodes 27 and 35, 3 repeats each, with the existing launcher on the H100 (`sonic_official_sim_eval.sh WBC_A28_ep27_r0 - <handover dataset> 27 <gpu> 5` etc.; `REPLAY=1` ignores `RUN_NAME`), then copy the six run dirs (with `stream_eval.json`) to `/mnt/data/jihun/g1_isaac_eval/parity/`.
- Isaac: the same six runs locally with `SIM=isaac` (Task 3 Step 9 commands, run dir `R=/mnt/data/jihun/g1_isaac_eval/parity/WBC_I28_ep<E>_r<R>` with `$R/sim`, `$R/gate`, `--log $R/streamer.npz`), then score each exactly as the launcher does for MuJoCo (`sonic_official_sim_eval.sh` line 39: `sonic_stream_eval.py OUT JOINT28_DIR SONIC_DIR EPISODE > OUT/stream_eval.json`):
  ```bash
  cd $D && $PY sonic_stream_eval.py $R $J28 $DS <E> > $R/stream_eval.json
  ```
  `sonic_stream_eval.py` reads `sim/sim_state.npz`, `sim/termination.json` and `streamer.npz` from the run dir; the Isaac scene writes the same files and `REC_KEYS` columns (plus `bottle_pos` / `bottle_quat`, which the scorer ignores).

- [ ] **Step 4: Gate**

Run: `cd $D && $PY wbc_compare.py /mnt/data/jihun/g1_isaac_eval/parity --episodes 27,35 --out /mnt/data/jihun/g1_isaac_eval/parity/compare.json && $PY -c "import json; print(json.load(open('/mnt/data/jihun/g1_isaac_eval/parity/compare.json'))['GI'])"`
Expected: `'pass': True`.

If it fails: compare `palm_err` per phase (`sonic_stream_phases.py`) to find where Isaac departs; usual causes in order: joint friction (PhysX `friction` is not MuJoCo `frictionloss`; try 0 and compare), armature, `STAND_Z`, hand order. Fix in `isaac_scene.py`, rerun the Isaac side only. Do not run policies (Task 7) before GI passes.

- [ ] **Step 5: Commit**

```bash
cd $L && git add examples/g1_dex3_training/wbc_compare.py tests/datasets/test_g1_dex3_wbc_compare_gi.py
git commit -m "feat(g1-dex3): Isaac replay parity gate GI in wbc_compare (spec test 2)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: task config from replays and the reach check (test 3)

**Files:**
- Create: `examples/g1_dex3_training/derive_handover_task.py`
- Modify: `examples/g1_dex3_training/tasks/handover_bottle.json` (replaces the Task 4 placeholder)
- Test: `tests/datasets/test_g1_dex3_derive_handover_task.py`

**Interfaces:**
- Consumes: MuJoCo A replay run dirs from Task 5 (`sim/sim_state.npz` with `REC_KEYS`, `t`), the streamer replay log's hand targets (`joint_ref` hand columns; read the column names from `sonic_policy_streamer.py`'s 50 Hz log keys before writing Step 3), `TABLE_GAP_CM` of those runs.
- Produces: `derive_handover_task.events(hand_targets: np.ndarray (T, 14), lo: np.ndarray, hi: np.ndarray) -> (grasp_hand: int, close_idx: int, release_idx: int)`; `derive_handover_task.bottle_from_palm(palm (3,), wrist_R (3, 3), offset_m) -> (3,)`; JSON keys `table_xy_world`, `bottle_xy_world`, `jitter_m`, `judge{...}`, `per_episode`, `source_runs`.

- [ ] **Step 1: Write the failing tests**

```python
"""derive_handover_task.py: grasp / release frames and the bottle position in front of the palm."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[2] / "examples" / "g1_dex3_training"))
import derive_handover_task as dt


def test_events_right_hand_close_then_reopen():
    lo, hi = np.zeros(14), np.ones(14)
    h = np.zeros((100, 14))
    h[30:70, 7:] = 1.0  # right hand closes at 30, reopens at 70
    h[50:60, :7] = 1.0  # left closes later
    assert dt.events(h, lo, hi) == (1, 30, 70)


def test_bottle_from_palm_offsets_along_palm_normal():
    r = np.eye(3)  # palm normal = wrist y axis (Dex3 palm faces -y on the right hand: sign set in the module)
    p = dt.bottle_from_palm(np.array([0.4, -0.2, 0.9]), r, 0.05, hand=1)
    assert np.isclose(np.linalg.norm(p - [0.4, -0.2, 0.9]), 0.05)
```

Run: `cd $L && uv run pytest tests/datasets/test_g1_dex3_derive_handover_task.py -v`
Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 2: Find the palm normal axis of each Dex3 hand**

Run (MuJoCo, local `.venv_sim`):
```bash
$G/.venv_sim/bin/python -c "
import mujoco, numpy as np
m = mujoco.MjModel.from_xml_path('$G/gear_sonic/data/robot_model/model_data/g1/scene_43dof.xml'); d = mujoco.MjData(m); mujoco.mj_forward(m, d)
for s in ('left', 'right'):
    w = d.xmat[m.body(f'{s}_wrist_yaw_link').id].reshape(3, 3); palm = d.xpos[[m.body(f'{s}_hand_{f}_0_link').id for f in ('index', 'middle')]].mean(0)
    th = d.xpos[m.body(f'{s}_hand_thumb_0_link').id]
    print(s, 'thumb - palm in wrist frame:', np.round(w.T @ (th - palm), 3))"
```
The palm faces the thumb side: the wrist-frame axis with the largest |component| of `thumb - palm`, with its sign, is the palm normal for that hand. Write the two (axis, sign) pairs into `PALM_NORMAL = {0: (axis, sign), 1: (axis, sign)}` in Step 3, and update the test's comment and the `hand=` argument accordingly.

- [ ] **Step 3: Write `derive_handover_task.py`**

```python
"""Bottle nominal position and handover zone for tasks/handover_bottle.json from MuJoCo backend A replays of the
held-out handover episodes (spec: Bottle position and handover zone). Usage:
python derive_handover_task.py RUN_DIR [RUN_DIR ...] --table-gap-cm 5 --out tasks/handover_bottle.json
"""

import argparse
import json
from pathlib import Path

import numpy as np

PALM_NORMAL = {0: (1, 1.0), 1: (1, -1.0)}  # hand -> (wrist-frame axis, sign); set from Task 6 Step 2
PALM_HALF_THICKNESS = 0.015
RADIUS, HEIGHT, TABLE_TOP = 0.033, 0.20, 0.80


def events(hand_targets, lo, hi):
    """(grasp hand, close index, release index): the hand whose mean normalised target first crosses 50 %, the
    crossing, and the first later index where the same hand reopens below 50 %."""
    x = (np.asarray(hand_targets) - lo) / np.where(hi > lo, hi - lo, 1.0)
    closed = np.stack([x[:, :7].mean(1) > 0.5, x[:, 7:].mean(1) > 0.5], 1)
    first = [int(np.argmax(c)) if c.any() else len(c) for c in closed.T]
    hand = int(np.argmin(first))
    close = first[hand]
    reopen = np.flatnonzero(~closed[close:, hand])
    release = close + int(reopen[0]) if reopen.size else len(closed) - 1
    return hand, close, release


def bottle_from_palm(palm, wrist_r, offset_m, hand):
    axis, sign = PALM_NORMAL[hand]
    return np.asarray(palm) + sign * offset_m * np.asarray(wrist_r)[:, axis]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+", type=Path)
    p.add_argument("--table-gap-cm", type=float, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    per = []
    for run in a.runs:
        s = np.load(run / "sim" / "sim_state.npz")
        hands = s["hand_q"]  # dataset order, left 7 then right 7 (sim_scene.HANDS)
        hand, close, release = events(hands, hands.min(0), hands.max(0))
        pelvis0 = s["pelvis"][0]
        edge = pelvis0[0] + 0.08 + a.table_gap_cm / 100
        b = bottle_from_palm(s["palm"][close, hand], s["wrist_R"][close, hand], RADIUS + PALM_HALF_THICKNESS, hand)
        pel_r = s["pelvis"][release]
        w, x, y, z = pel_r[3:7]
        rot = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])  # fmt: skip
        zone = rot.T @ (s["palm"][release, hand] - pel_r[:3])
        per.append({"run": run.name, "grasp_hand": hand, "close_t": float(s["t"][close]),
                    "release_t": float(s["t"][release]), "bottle_from_edge_x": float(b[0] - edge),
                    "bottle_y_from_pelvis": float(b[1] - pelvis0[1]), "zone_center_pelvis": zone.tolist()})  # fmt: skip
    hands = {r["grasp_hand"] for r in per}
    if len(hands) != 1:
        raise SystemExit(f"episodes grasp with different hands: {per}")
    bx = float(np.mean([r["bottle_from_edge_x"] for r in per]))
    by = float(np.mean([r["bottle_y_from_pelvis"] for r in per]))
    edge0 = 0.08 + a.table_gap_cm / 100  # robot at the origin after the trial reset
    cfg = {
        "table_xy_world": [edge0 + 0.4, 0.0],
        "bottle_xy_world": [edge0 + bx, by],
        "jitter_m": 0.03,
        "judge": {"table_top_z": TABLE_TOP, "bottle_radius": RADIUS, "bottle_height": HEIGHT,
                  "zone_center_pelvis": np.mean([r["zone_center_pelvis"] for r in per], 0).round(4).tolist(),
                  "zone_radius": 0.08, "grasp_hand": hands.pop()},  # fmt: skip
        "table_gap_cm": a.table_gap_cm,
        "per_episode": per,
        "source_runs": [r["run"] for r in per],
    }
    a.out.write_text(json.dumps(cfg, indent=1))
    print(json.dumps(cfg["judge"]), cfg["bottle_xy_world"])


if __name__ == "__main__":
    main()
```

Run the tests. Expected: pass.

- [ ] **Step 4: Build the config from the MuJoCo A replays (Task 5 runs, repeat 0 of each episode)**

Run: `cd $D && $PY derive_handover_task.py /mnt/data/jihun/g1_isaac_eval/parity/WBC_A28_ep27_r0 /mnt/data/jihun/g1_isaac_eval/parity/WBC_A28_ep35_r0 --table-gap-cm 5 --out tasks/handover_bottle.json`
Expected: one grasp hand for both episodes; bottle 5-30 cm behind the table edge; zone centre in front of the robot (x > 0.2 m from the pelvis). The two per-episode bottle positions within about 10 cm of each other; if not, report both and use the mean.

- [ ] **Step 5: Test 3 — reach and zone on Isaac replays with the bottle placed**

Rerun the Isaac replays of episodes 27 and 35 (one repeat each) with the bottle at `bottle_xy_world` (add `BOTTLE=1` handling in the host's non-trials path: after `place_table()`, call `scene.reset_trial(SEED, task["bottle_xy_world"], 0.0)` without the robot reset — split `reset_trial` into `place_bottle(...)` used by both if needed). From each `sim_state.npz`, check: grasping palm within `r + 4 cm` of the bottle axis at the close frame (`bottle_pos` / `bottle_quat` are recorded), and palm inside the zone at the release frame.
Expected: both true for both episodes. If not, check the bottle state first (an open-loop replay can knock it over: tilt from `bottle_quat`); only then adjust `PALM_HALF_THICKNESS` or `reach_margin_m`.

- [ ] **Step 6: Commit**

```bash
cd $L && git add examples/g1_dex3_training/derive_handover_task.py examples/g1_dex3_training/tasks/handover_bottle.json \
  examples/g1_dex3_training/sonic_official_sim_host.py tests/datasets/test_g1_dex3_derive_handover_task.py
git commit -m "feat(g1-dex3): handover task config from replays; reach and zone checked on Isaac replays (spec test 3)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: launcher local mode, closed-loop sessions (test 5), result doc

**Files:**
- Modify: `examples/g1_dex3_training/sonic_official_sim_eval.sh`
- Create: `docs/research/2026-10-XX-g1-isaac-handover-eval.md` (date of the run)
- Modify: `CLAUDE.md` (one sentence pointing to the result doc, in the G1 bullet)

**Interfaces:**
- Consumes: everything above; checkpoints `/mnt/data/jihun/g1_models/wbt_groot_he_full` and `/mnt/data/jihun/g1_models/wbt_dex_pi05_dex100_he_full` are *not* the 41-episode handover models; use the handover 41-episode he checkpoints listed in `docs/research/g1-model-registry.md` (GR00T he 41-ep, Pi0.5 he 41-ep) and copy them to `/mnt/data/jihun/g1_models/` if they are not there.

- [ ] **Step 1: Local mode in the launcher**

At the top of `sonic_official_sim_eval.sh`, before the H100 path, add a `SIM=isaac` branch that runs everything locally and exits:

```bash
if [ "${SIM:-mujoco}" = isaac ]; then
  # Local Isaac mode (spec): host on ISAAC_GPU, policy server on POLICY_GPU, streamer here. Usage:
  #   SIM=isaac TRIALS=10 SESSION_DIR=... POLICY_PATH=... sonic_official_sim_eval.sh OUT - - - - TABLE_GAP_CM
  : "${TRIALS:?}" "${SESSION_DIR:?}" "${POLICY_PATH:?}"
  PY=${PY:-/home/jihun/anaconda3/envs/sonic/bin/python}; ISAAC_GPU=${ISAAC_GPU:-0}; POLICY_GPU=${POLICY_GPU:-1}
  OUTD=$LOG_DIR/$OUT; mkdir -p "$OUTD/sim" "$OUTD/gate" "$SESSION_DIR"
  CUDA_VISIBLE_DEVICES=$POLICY_GPU uv run python sonic_policy_server.py --policy-path "$POLICY_PATH" $SERVER_EXTRA > "$OUTD/server.log" 2>&1 &
  SERVER=$!
  for _ in $(seq 120); do (exec 3<>/dev/tcp/localhost/5560) 2>/dev/null && break; sleep 2; done
  (exec 3<>/dev/tcp/localhost/5560) 2>/dev/null || { echo "policy server did not open 5560" >&2; kill $SERVER; exit 1; }
  SIM=isaac TRIALS=$TRIALS SESSION_DIR=$SESSION_DIR SEED=${SEED:-0} TABLE_GAP_CM=$GAP CUDA_VISIBLE_DEVICES=$ISAAC_GPU \
    $PY -u sonic_official_sim_host.py "$OUTD/sim" "$OUTD/gate" > "$OUTD/host.log" 2>&1 &
  HOST=$!
  uv run python sonic_policy_streamer.py --policy-server tcp://localhost:5560 --gate-dir "$OUTD/gate" \
    --trials "$TRIALS" --session-dir "$SESSION_DIR" --images zmq --camera-host localhost --camera-port 5555 \
    --task "pick drink bottle and hand over" --duration-s 40 --state-source robot $EXTRA 2>&1 | tee "$OUTD/streamer.log"
  wait $HOST; kill $SERVER 2>/dev/null
  exit 0
fi
```

Then compare the streamer flags with the real-robot launcher `g1_groot_real_run.sh` (start mode `--start planner`, `--max-token-step 0.05`, `--policy-timeout-s 5`, noise / blend settings of the 41-episode robot sessions) and pass the same ones through `EXTRA`, so sim and robot sessions differ only in the physics. Write the exact flag list into the result doc.

Note: the launcher argument order is `OUT RUN DS EP GPU GAP`; in Isaac mode `RUN DS EP GPU` are unused (`-`).

- [ ] **Step 2: Smoke session**

`SIM=isaac TRIALS=2 SESSION_DIR=/mnt/data/jihun/g1_isaac_eval/smoke POLICY_PATH=<GR00T he 41-ep> bash sonic_official_sim_eval.sh smoke - - - - 5 <EXTRA flags>`
Expected: two trials saved; `results.jsonl` has 2 lines; `host.log` lag line max <= 20 ms for both trials (Review Focus 5).

- [ ] **Step 3: Test 5 — two sessions of 10 trials**

Run GR00T he and Pi0.5 he (41-episode handover checkpoints), `TRIALS=10`, `SEED=0`, one `SESSION_DIR` each. Then:
`uv run python examples/g1_dex3_training/eval_session_to_lerobot.py <session_dir> ...` (same arguments as for robot sessions, see `docs/research/2026-10-13-g1-eval-logger.md`) and open both in `~/work/lerobot-dataset-visualizer`.
Expected (pass criterion): both sessions complete and open in the visualizer; at least one success over the two sessions; not all 20 trials reason 1.

- [ ] **Step 4: Result doc**

`docs/research/2026-10-XX-g1-isaac-handover-eval.md`, in the style of `docs/research/2026-10-02-g1-wbc-backends.md` (no host names or user names): status paragraph; test 0a/0b numbers (rtf_min, rtf_median, lag max / p99); GI table; task config values and their source runs; reset-stability result and whether the per-trial fallback was used; the two session results (success rate with a Wilson 95 % interval, failure reasons count) next to the robot results (GR00T he 5/5, Pi0.5 he 3/5); the exact streamer flags; known limits (table clearance from body origins; no third-person video; nominal camera intrinsics until calibrated).

- [ ] **Step 5: Point CLAUDE.md to it** (one sentence at the end of the G1 bullet: "Isaac Lab closed-loop handover evaluation (SONIC deploy, automatic verdicts): `docs/research/2026-10-XX-g1-isaac-handover-eval.md`.")

- [ ] **Step 6: Commit**

```bash
cd $L && git add examples/g1_dex3_training/sonic_official_sim_eval.sh docs/research/2026-10-*-g1-isaac-handover-eval.md CLAUDE.md
git commit -m "feat(g1-dex3): local Isaac eval mode; first GR00T / Pi0.5 handover sessions in Isaac (spec test 5)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```
