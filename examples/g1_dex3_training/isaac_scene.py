"""Isaac Lab scene for the sim hosts, with the interface of sim_scene.Scene (MuJoCo): G1 29 + Dex3 from
g1_29dof_with_hand.urdf (the twin of gear_sonic's MuJoCo model), the optional tabletop, a bottle, and the head
camera (D435 mount on torso_link, 640 x 480 at 30 Hz, published like g1_head_camera_server.py as "egocentric").

The SONIC deploy talks to this scene exactly as to gear_sonic's MuJoCo sim: Unitree DDS on lo through gear_sonic's
UnitreeSdk2Bridge (rt/lowcmd, rt/lowstate, rt/dex3/*, rt/secondary_imu, rt/odostate). Torques: tau = tau_ff +
kp (q* - q) + kd (dq* - dq) from the latest commands, clipped to gear_sonic's motor effort limits, applied in effort
mode at dt 0.005 s, as gear_sonic base_sim does. Physics runs on the CPU (PhysX): on the workstation RTX 3060 GPU
physics for this single robot ran at 0.49x real time, CPU physics at 1.37x (spec test 0a).

Importing this module has no side effects: Isaac starts in Scene(...). Runs in the `sonic` conda env (Isaac Sim 5.1,
Isaac Lab 2.3.2). Kit can hang in app.close(), so callers end the process with os._exit after close().
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
# cyclonedds 0.10.2 for Python 3.11 is built from source against this install; the system libddsc is older
CYCLONEDDS_LIB = Path(os.environ.get("CYCLONEDDS_LIB", Path.home() / "work" / "cyclonedds_v0.10.2" / "lib"))
URDF = GEAR_ROOT / "gear_sonic/data/robot_model/model_data/g1/g1_29dof_with_hand.urdf"
CAMERA_JSON = Path(__file__).parent / "cameras" / "d435i_head.json"
FPS, W, H = 30, 640, 480
DT = 0.005
STAND_Z = 0.775  # pelvis height standing on the default pose: both feet on the floor in test 0a
BAND_Z = 1.0  # root height while hanging on the band (gear_sonic: the deploy starts with the robot hung)
TABLE_TOP_Z = 0.80  # table centre 0.785 + half its 3 cm top, as sim_scene
DEX3 = ("thumb_0", "thumb_1", "thumb_2", "middle_0", "middle_1", "index_0", "index_1")  # bridge / actuator order
DEX3_MOTORS = [f"{s}_hand_{j}_joint" for s in ("left", "right") for j in DEX3]
# gear_sonic g1_29dof_with_hand.xml actuator order, which motor_effort_limit_list follows
MUJOCO_ACTUATORS = BODY[:15] + BODY[15:22] + DEX3_MOTORS[:7] + BODY[22:] + DEX3_MOTORS[7:]
HAND_PERM = [DEX3_MOTORS.index(n) for n in HANDS]  # bridge order -> dataset order (sim_scene records HANDS)
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


def quat_wxyz_to_matrix(q) -> np.ndarray:
    w, x, y, z = (float(v) for v in q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])  # fmt: skip


def make_obs(s: dict) -> dict:
    """UnitreeSdk2Bridge.PublishLowState input from plain arrays (joints in BODY + DEX3_MOTORS order; quaternions
    w, x, y, z; velocities [lin 3, ang 3])."""
    nb, nh = len(BODY), len(DEX3_MOTORS) // 2
    obs = {
        "body_q": s["q"][:nb], "body_dq": s["dq"][:nb], "body_ddq": s["ddq"][:nb], "body_tau_est": s["tau"][:nb],
        "floating_base_pose": s["root_pose"], "floating_base_vel": s["root_vel"], "floating_base_acc": s["root_acc"],
        "secondary_imu_quat": s["torso_quat"], "secondary_imu_vel": s["torso_vel"], "time": s["time"],
    }  # fmt: skip
    for side, sl in (("left", slice(nb, nb + nh)), ("right", slice(nb + nh, nb + 2 * nh))):
        obs[f"{side}_hand_q"], obs[f"{side}_hand_dq"] = s["q"][sl], s["dq"][sl]
    return obs


class Scene:
    def __init__(self, out: Path, table_gap_cm: str | None, camera_json: Path = CAMERA_JSON, camera_port: int = 5555,
                 device: str = "cpu", render: bool = True, dds: bool = True):  # fmt: skip
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
                init_state=ArticulationCfg.InitialStateCfg(pos=(0.0, 0.0, BAND_Z)),
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
                init_state=RigidObjectCfg.InitialStateCfg(pos=(10.0, 2.0, 0.90)),
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
        self._check_joints()
        self.body_ids, _ = self.robot.find_joints(BODY, preserve_order=True)
        self.hand_ids, _ = self.robot.find_joints(DEX3_MOTORS, preserve_order=True)
        self.torso_id = self.robot.find_bodies("torso_link")[0][0]
        self.palm_ids = [self.robot.find_bodies([f"{s}_hand_{f}_0_link" for f in ("index", "middle")],
                                                preserve_order=True)[0] for s in ("left", "right")]  # fmt: skip
        self.wrist_ids = [self.robot.find_bodies(f"{s}_wrist_yaw_link")[0][0] for s in ("left", "right")]
        self.limit = self._effort_limits()
        self.cmd = zero_cmd(len(BODY) + len(DEX3_MOTORS))
        self.tau = np.zeros(len(BODY) + len(DEX3_MOTORS))
        self.pinned_root: np.ndarray | None = np.array([0.0, 0.0, BAND_Z, 1.0, 0.0, 0.0, 0.0])  # band
        self.prev_root_vel = np.zeros(6)
        self.dt, self.t, self.next_frame = DT, 0.0, 0.0 if render else float("inf")
        self.lock = threading.Lock()
        self.view = {"phase": "hanging on elastic band", "stop": False}
        self.rec = {k: [] for k in REC_KEYS}
        self.next_rec = 0.0
        self.fall = FallDetector()
        self.frame_ms: list[float] = []
        self.lag_ms: list[float] = []
        self.pub = zmq.Context().socket(zmq.PUB)
        self.pub.bind(f"tcp://*:{camera_port}")
        self.bridge = self._start_dds() if dds else None

    def _check_joints(self) -> None:
        names = self.robot.joint_names
        missing = [n for n in BODY + DEX3_MOTORS if n not in names]
        if missing:
            raise SystemExit(f"Isaac articulation lacks {missing}")
        if len(names) != len(BODY) + len(DEX3_MOTORS):
            raise SystemExit(f"Isaac articulation has {len(names)} joints, expected {len(BODY) + len(DEX3_MOTORS)}")

    def _effort_limits(self) -> np.ndarray:
        """gear_sonic's motor_effort_limit_list from the same wbc yaml as its MuJoCo sim, in Scene.cmd order."""
        sys.path.insert(0, str(GEAR_ROOT))
        from gear_sonic.utils.mujoco_sim.configs import SimLoopConfig

        self.config = SimLoopConfig(interface="sim", enable_onscreen=False).load_wbc_yaml()
        if self.config["NUM_MOTORS"] != len(BODY) or self.config["NUM_HAND_MOTORS"] != len(DEX3_MOTORS) // 2:
            raise SystemExit("gear_sonic config motor counts differ from BODY / DEX3_MOTORS")
        return effort_limits(self.config["motor_effort_limit_list"])

    def _start_dds(self):
        """The bridge on DDS domain / interface of the wbc yaml (0, lo), as gear_sonic's simulator_factory."""
        import ctypes

        ctypes.CDLL(str(CYCLONEDDS_LIB / "libddsc.so.0"), mode=ctypes.RTLD_GLOBAL)  # before the system libddsc
        from gear_sonic.utils.mujoco_sim.unitree_sdk2py_bridge import UnitreeSdk2Bridge
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize

        ChannelFactoryInitialize(self.config["DOMAIN_ID"], self.config["INTERFACE"])
        return UnitreeSdk2Bridge(self.config)

    def _bridge_cmd(self) -> None:
        """self.cmd from the latest rt/lowcmd and Dex3 commands (zero until the deploy has sent any)."""
        br = self.bridge
        if br is None or not br.cmd_received():
            return
        nb, nh = len(BODY), len(DEX3_MOTORS) // 2
        motors = [br.low_cmd.motor_cmd[i] for i in range(nb)]
        motors += [br.left_hand_cmd.motor_cmd[i] for i in range(nh)] + [br.right_hand_cmd.motor_cmd[i] for i in range(nh)]
        for k in ("q", "dq", "kp", "kd", "tau"):
            self.cmd[k] = np.array([getattr(m, k) for m in motors], dtype=float)

    def joint_state(self) -> tuple[np.ndarray, np.ndarray]:
        ids = self.body_ids + self.hand_ids
        d = self.robot.data
        return d.joint_pos[0, ids].cpu().numpy(), d.joint_vel[0, ids].cpu().numpy()

    def _state(self) -> dict:
        d, ids = self.robot.data, self.body_ids + self.hand_ids
        root = d.root_state_w[0].cpu().numpy()
        root_vel = root[7:13]
        acc = (root_vel - self.prev_root_vel) / DT  # finite difference, no gravity term (MuJoCo qacc[:6])
        self.prev_root_vel = root_vel.copy()
        torso_lin = d.body_lin_vel_w[0, self.torso_id].cpu().numpy()
        torso_ang = d.body_ang_vel_w[0, self.torso_id].cpu().numpy()
        return {
            "q": d.joint_pos[0, ids].cpu().numpy(), "dq": d.joint_vel[0, ids].cpu().numpy(),
            "ddq": d.joint_acc[0, ids].cpu().numpy(), "tau": self.tau, "root_pose": root[:7], "root_vel": root_vel,
            "root_acc": acc, "torso_quat": d.body_quat_w[0, self.torso_id].cpu().numpy(),
            "torso_vel": np.r_[torso_lin, torso_ang], "time": self.t,
        }  # fmt: skip

    def step(self) -> None:
        import torch

        with self.lock:
            self._bridge_cmd()
            q, dq = self.joint_state()
            self.tau = pd_torques(self.cmd, q, dq, self.limit)
            dev = self.sim.device
            self.robot.set_joint_effort_target(
                torch.tensor(self.tau, dtype=torch.float32, device=dev).unsqueeze(0),
                joint_ids=self.body_ids + self.hand_ids,
            )
            if self.pinned_root is not None:
                self.robot.write_root_pose_to_sim(torch.tensor(self.pinned_root, dtype=torch.float32, device=dev)[None])
                self.robot.write_root_velocity_to_sim(torch.zeros(1, 6, device=dev))
            render = self.t + 1e-9 >= self.next_frame
            self.scene.write_data_to_sim()
            self.sim.step(render=render)
            self.scene.update(DT)
            self.t += DT
            if self.bridge is not None:
                self.bridge.PublishLowState(make_obs(self._state()))
        if render:
            self.next_frame += 1 / FPS
            self._publish_frame()

    def _publish_frame(self) -> None:
        import cv2
        import zmq

        t0 = time.perf_counter()
        self.camera.update(0.0, force_recompute=True)
        rgb = self.camera.data.output["rgb"][0, ..., :3].cpu().numpy()
        if rgb.shape[:2] != (H, W):
            raise SystemExit(f"head camera gives {rgb.shape[1]}x{rgb.shape[0]}, not {W}x{H}")
        _, jpg = cv2.imencode(".jpg", rgb, [int(cv2.IMWRITE_JPEG_QUALITY), 90])  # RGB as is, like the camera server
        msg = {"timestamps": {"egocentric": time.time()}, "images": {"egocentric": base64.b64encode(jpg).decode()}}
        with contextlib.suppress(zmq.Again):  # no subscriber / buffer full: drop, as the camera server does
            self.pub.send_string(json.dumps(msg), zmq.NOBLOCK)
        self.last_rgb = rgb
        self.frame_ms.append((time.perf_counter() - t0) * 1e3)

    def key(self, name: str) -> None:
        """The two simulator keys the host uses: '9' releases the band (gear_sonic: the robot drops), 'backspace'
        resets the robot standing on the floor (root at STAND_Z, joints at DEFAULT_DOF_ANGLES, hands 0, at rest)."""
        import torch

        with self.lock:
            if name == "9":
                self.pinned_root = None
            elif name == "backspace":
                self.pinned_root = None
                dev = self.sim.device
                self.robot.write_root_pose_to_sim(torch.tensor([[0.0, 0.0, STAND_Z, 1.0, 0.0, 0.0, 0.0]], device=dev))
                self.robot.write_root_velocity_to_sim(torch.zeros(1, 6, device=dev))
                q = np.zeros(len(BODY) + len(DEX3_MOTORS))
                q[: len(BODY)] = self.config["DEFAULT_DOF_ANGLES"]
                pos = self.robot.data.default_joint_pos.clone()
                pos[0, self.body_ids + self.hand_ids] = torch.tensor(q, dtype=pos.dtype, device=pos.device)
                self.robot.write_joint_state_to_sim(pos, torch.zeros_like(pos))
                self.prev_root_vel = np.zeros(6)
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
        """Lowest wrist / palm-finger body origin above the table top (m) and 0 hits; nan/0 without a table.
        ponytail: body origins, not collision geometry like the MuJoCo scene; add a hand contact sensor if the
        clearance diagnostics are needed for Isaac runs."""
        if self.table_gap is None:
            return np.nan, 0
        ids = self.wrist_ids + [i for pair in self.palm_ids for i in pair]
        z = self.robot.data.body_pos_w[0, ids, 2].cpu().numpy()
        return float(z.min() - TABLE_TOP_Z), 0

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

    def record(self, **extra) -> None:
        """One row every 20 ms of sim time, the columns of sim_scene.Scene.record plus the bottle pose."""
        if self.t + 1e-9 < self.next_rec:
            return
        self.next_rec = self.t + 0.02
        with self.lock:
            s, quats = self._state(), self.robot.data.body_quat_w[0].cpu().numpy()
            clear, hits = self.table_clearance()
            bottle_pos, bottle_quat = self.bottle_state()
            row = {
                "wall": time.time(), "t": self.t, "body_q": s["q"][: len(BODY)].copy(),
                "hand_q": s["q"][len(BODY):][HAND_PERM].copy(), "pelvis": s["root_pose"].copy(), "palm": self.palms(),
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

    def mark(self, event: str) -> None:
        print(f"[{self.t:8.3f}] {event}", flush=True)
        with open(self.out / "events.txt", "a") as f:
            f.write(f"{time.time():.3f} {self.t:.3f} {event}\n")

    def pace(self, wall0: float) -> None:
        """Wait while the sim is ahead of the wall clock; log how far it is behind otherwise (spec test 0b)."""
        ahead = self.t - (time.monotonic() - wall0)
        self.lag_ms.append(max(0.0, -ahead) * 1e3)
        if ahead > 0:
            time.sleep(ahead)

    def close(self, reason: str, detail: str = "", **fields) -> None:
        self.view["stop"] = True
        if self.rec["t"]:
            np.savez_compressed(self.out / "sim_state.npz", **{k: np.array(v) for k, v in self.rec.items()})
        write_termination(self.out, reason, self.view["phase"], detail, **fields)
        self.mark(f"termination: {reason} {detail}")
