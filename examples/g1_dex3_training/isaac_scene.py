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
    from .sim_scene import BODY, REC_KEYS
    from .wbc_common import FallDetector, write_termination
except ImportError:
    from sim_scene import BODY, REC_KEYS
    from wbc_common import FallDetector, write_termination

GEAR_ROOT = Path(os.environ.get("GEAR_ROOT", Path.home() / "work" / "GR00T-WholeBodyControl"))
URDF = GEAR_ROOT / "gear_sonic/data/robot_model/model_data/g1/g1_29dof_with_hand.urdf"
CAMERA_JSON = Path(__file__).parent / "cameras" / "d435i_head.json"
FPS, W, H = 30, 640, 480
DT = 0.005
STAND_Z = 0.79  # pelvis height standing on the default pose; Task 3 replaces it with the measured value
DEX3 = ("thumb_0", "thumb_1", "thumb_2", "middle_0", "middle_1", "index_0", "index_1")  # bridge / actuator order
DEX3_MOTORS = [f"{s}_hand_{j}_joint" for s in ("left", "right") for j in DEX3]
# gear_sonic g1_29dof_with_hand.xml actuator order, which motor_effort_limit_list follows
MUJOCO_ACTUATORS = BODY[:15] + BODY[15:22] + DEX3_MOTORS[:7] + BODY[22:] + DEX3_MOTORS[7:]
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


def effort_limits(actuator_limits) -> np.ndarray:
    """motor_effort_limit_list (MuJoCo actuator order) as BODY + DEX3_MOTORS, the joint order of Scene.cmd."""
    lim = dict(zip(MUJOCO_ACTUATORS, np.asarray(actuator_limits, dtype=float), strict=True))
    return np.array([lim[n] for n in BODY + DEX3_MOTORS])


def zero_cmd(n: int) -> dict:
    return {k: np.zeros(n) for k in ("q", "dq", "kp", "kd", "tau")}


class Scene:
    def __init__(self, out: Path, table_gap_cm: str | None, camera_json: Path = CAMERA_JSON, camera_port: int = 5555,
                 device: str = "cuda:0", render: bool = True):
        from isaaclab.app import AppLauncher

        self.app = AppLauncher(headless=True, enable_cameras=True, device=device).app
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

        self.sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=DT, render_interval=1, device=device))
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
        self.hand_ids, _ = self.robot.find_joints(DEX3_MOTORS, preserve_order=True)
        self.limit = self._effort_limits()
        self.cmd = zero_cmd(len(BODY) + len(DEX3_MOTORS))
        self.pinned_root: np.ndarray | None = None  # (7,) pos + quat wxyz; held each step while set (band)
        self.dt, self.t, self.next_frame = DT, 0.0, 0.0 if render else float("inf")
        self.lock = threading.Lock()
        self.view = {"phase": "hanging on elastic band", "stop": False}
        self.rec = {k: [] for k in REC_KEYS}
        self.next_rec = 0.0
        self.fall = FallDetector()
        self.frame_ms: list[float] = []
        self.pub = zmq.Context().socket(zmq.PUB)
        self.pub.bind(f"tcp://*:{camera_port}")

    def _effort_limits(self) -> np.ndarray:
        """gear_sonic's motor_effort_limit_list from the same wbc yaml as its MuJoCo sim, in Scene.cmd order."""
        sys.path.insert(0, str(GEAR_ROOT))
        from gear_sonic.utils.mujoco_sim.configs import SimLoopConfig

        self.config = SimLoopConfig(interface="sim", enable_onscreen=False).load_wbc_yaml()
        return effort_limits(self.config["motor_effort_limit_list"])

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
        # no app.close(): Kit can hang in it; callers end the process with os._exit after close()
