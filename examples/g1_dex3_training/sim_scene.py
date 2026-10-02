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
