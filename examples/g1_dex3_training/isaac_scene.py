"""Isaac Lab scene for the sim hosts, with the interface of sim_scene.Scene (MuJoCo): G1 29 + Dex3 from
Unitree's g1_29dof_with_hand_rev_1_0.urdf (unitree_ros, as in gear_sonic/data/robots/g1), the optional tabletop, a bottle, and the head
camera (D435 mount on torso_link, 640 x 480 at 30 Hz, published like g1_head_camera_server.py as "egocentric").

The SONIC deploy talks to this scene exactly as to gear_sonic's MuJoCo sim: Unitree DDS on lo through gear_sonic's
UnitreeSdk2Bridge (rt/lowcmd, rt/lowstate, rt/dex3/*, rt/secondary_imu, rt/odostate), which runs in a separate
process (dds_publisher.py) fed through shared memory. Torques: tau = tau_ff +
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
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

try:
    from .dds_publisher import NB, NH, Exchange, make_obs  # noqa: F401 - make_obs re-exported
    from .sim_scene import BODY, HANDS, REC_KEYS
    from .wbc_common import FallDetector, write_termination
except ImportError:
    from dds_publisher import NB, NH, Exchange, make_obs  # noqa: F401 - make_obs re-exported
    from sim_scene import BODY, HANDS, REC_KEYS
    from wbc_common import FallDetector, write_termination

GEAR_ROOT = Path(os.environ.get("GEAR_ROOT", Path.home() / "work" / "GR00T-WholeBodyControl"))
# cyclonedds 0.10.2 for Python 3.11 is built from source against this install; the system libddsc is older
CYCLONEDDS_LIB = Path(os.environ.get("CYCLONEDDS_LIB", Path.home() / "work" / "cyclonedds_v0.10.2" / "lib"))
# Unitree's official G1 + Dex3 model (unitree_ros, identical up to the material declarations). Runs to 2026-10-27 used
# gear_sonic robot_model's older g1_29dof_with_hand.urdf: same arm and head-camera kinematics in the world, but a
# 1.8 kg heavier torso, torso_link 1 cm higher and the thumb_1 limit -0.72 instead of -0.61 rad
URDF = GEAR_ROOT / "gear_sonic/data/robots/g1/g1_29dof_with_hand_rev_1_0.urdf"
CAMERA_JSON = Path(__file__).parent / "cameras" / "d435i_head.json"
FPS, W, H = 30, 640, 480
CAMERA_FPS = float(os.environ.get("HEAD_CAMERA_FPS", FPS))  # e.g. 15 for COMPOSITE_BG: its extra render pass lags at 30
DT = 0.005
STAND_Z = 0.775  # pelvis height standing on the default pose: both feet on the floor in test 0a
BAND_Z = 1.0  # root height while hanging on the band (gear_sonic: the deploy starts with the robot hung)
# the robot handover table is a black flight case: depth (x), width (y), top height (m), measured on the real setup
# (2026-10-19; the head-frame fit gave 0.42 x 0.32 x 0.846); replays and the parity runs used sim_scene's 0.80 m slab
CASE = (0.475, 0.35, float(os.environ.get("CASE_TOP_Z", "0.845")))  # CASE_TOP_Z: e.g. the +5 cm runs
TABLE_TOP_Z = CASE[2]
# URDF_COLLIDER=convex_decomposition: the robot's collision meshes keep their concavity (convex_hull, the default, makes
# the wrist_yaw mesh 10.5x and the Dex3 palm 1.5x their volume: the hand meets the bottle where the real one passes)
COLLIDER = os.environ.get("URDF_COLLIDER", "convex_hull")
# clear PET bottle: 17 cm with the cap (measured), radius from the head frames. BOTTLE_H is the body without the cap;
# the root sits at its middle. Profile (base up): body r 3.3 cm to 11 cm, a shoulder cone, a 1.35 cm neck from 12.5 cm,
# a solid 1.65 cm cap on top: the hand passes the narrow neck, a full-radius cylinder catches it (every knock-over hit
# the top 4 cm)
BOTTLE_R, BOTTLE_H = 0.033, 0.155
TOUCH = [f"{s}_{n}_link" for s in ("left", "right") for n in ("wrist_yaw", *(f"hand_{f}" for f in (
    "thumb_0", "thumb_1", "thumb_2", "index_0", "index_1", "middle_0", "middle_1")))]  # touch_force columns
BOTTLE_PARTS = (  # (name, shape, radius, height, centre height above the base)
    ("body", "cylinder", BOTTLE_R, 0.110, 0.055),
    ("shoulder", "cone", BOTTLE_R, 0.050, 0.135),
    ("neck", "cylinder", 0.0135, 0.030, 0.140),
    ("cap", "cylinder", 0.0165, 0.017, BOTTLE_H + 0.0085),
)
DEX3 = ("thumb_0", "thumb_1", "thumb_2", "middle_0", "middle_1", "index_0", "index_1")  # bridge / actuator order
DEX3_MOTORS = [f"{s}_hand_{j}_joint" for s in ("left", "right") for j in DEX3]
# gear_sonic g1_29dof_with_hand.xml actuator order, which motor_effort_limit_list follows
MUJOCO_ACTUATORS = BODY[:15] + BODY[15:22] + DEX3_MOTORS[:7] + BODY[22:] + DEX3_MOTORS[7:]
HAND_PERM = [DEX3_MOTORS.index(n) for n in HANDS]
assert (len(BODY), len(DEX3_MOTORS) // 2) == (NB, NH), "dds_publisher layout differs from BODY / DEX3_MOTORS"  # bridge order -> dataset order (sim_scene records HANDS)
CAMERA_KEYS = ("parent", "pos", "rot_wxyz", "width", "height", "fx", "fy", "cx", "cy")


def robot_look(link: str) -> str:
    """The material a robot link renders with: the URDF's dark parts, the black Dex3 (from the wrist_yaw motor on), metal."""
    if "_hand_" in link or link.endswith("wrist_yaw_link"):
        return "dex3_black"
    return "g1_dark" if link in ("pelvis", "left_hip_pitch_link", "right_hip_pitch_link", "left_ankle_roll_link",
                                 "right_ankle_roll_link") else "g1_metal"


def sample_bottle_xy(seed: int, center, jitter_m) -> np.ndarray:
    """The trial's bottle position: center + uniform +-jitter_m, a scalar or per axis (x, y); seeded by the trial."""
    rng = np.random.default_rng(seed)
    j = np.broadcast_to(np.asarray(jitter_m, float), (2,))
    return np.asarray(center, float) + rng.uniform(-j, j)


def parse_friction(text: str | None) -> tuple[float, float] | None:
    """BOTTLE_FRICTION / CASE_FRICTION: "mu" or "static,dynamic"; unset = the PhysX default (0.5, no material)."""
    if not text:
        return None
    v = [float(x) for x in text.split(",")]
    static, dynamic = (v[0], v[0]) if len(v) == 1 else v
    if dynamic > static:
        raise ValueError(f"dynamic friction {dynamic} above static {static}")
    return static, dynamic


def composite(rgb: np.ndarray, seg: np.ndarray, id_to_labels: dict, bg: np.ndarray) -> np.ndarray:
    """COMPOSITE_BG test: the rendered robot and bottle pixels over a real head frame (everything else from bg)."""
    keep = [int(i) for i, path in id_to_labels.items() if "/Robot/" in str(path) or "/Bottle/" in str(path)]
    return np.where(np.isin(seg, keep)[..., None], rgb, bg)


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


def look_at_wxyz(pos, target) -> np.ndarray:
    """Quaternion (w, x, y, z) of a camera at pos looking at target, Isaac "world" convention (x forward, z up), no roll."""
    f = np.asarray(target, float) - np.asarray(pos, float)
    f /= np.linalg.norm(f)
    left = np.cross([0.0, 0.0, 1.0], f)
    left /= np.linalg.norm(left)
    m = np.stack([f, left, np.cross(f, left)], 1)
    w = np.sqrt(max(1.0 + np.trace(m), 1e-12)) / 2  # cameras here never look straight back (trace > -1)
    return np.array([w, (m[2, 1] - m[1, 2]) / (4 * w), (m[0, 2] - m[2, 0]) / (4 * w), (m[1, 0] - m[0, 1]) / (4 * w)])


def plane_homography(cam: dict, ref_pos, ref_rot, cur_pos, cur_rot, plane_z: float) -> np.ndarray:
    """COMPOSITE_FOLLOW: the 3x3 map from the reference camera's pixels to the current camera's, for points on the plane
    z = plane_z (the case top). Cameras in the Isaac "world" convention (x forward, y left, z up)."""
    def ray(rot, u, v):
        return rot @ np.array([1.0, (cam["cx"] - u) / cam["fx"], (cam["cy"] - v) / cam["fy"]])
    def pix(pos, rot, p):
        q = np.asarray(rot).T @ (p - np.asarray(pos))
        return [cam["cx"] - cam["fx"] * q[1] / q[0], cam["cy"] - cam["fy"] * q[2] / q[0]]
    src, dst = [], []
    for u, v in ((100, 150), (540, 150), (540, 450), (100, 450)):  # well inside the image, below the horizon
        d = ray(np.asarray(cur_rot), u, v)
        p = np.asarray(cur_pos, float) + d * (plane_z - cur_pos[2]) / d[2]
        src.append(pix(ref_pos, ref_rot, p))
        dst.append([u, v])
    rows = []
    for (x, y), (u, v) in zip(src, dst, strict=True):  # DLT with h33 = 1
        rows += [[x, y, 1, 0, 0, 0, -u * x, -u * y, u], [0, 0, 0, x, y, 1, -v * x, -v * y, v]]
    a = np.asarray(rows, float)
    h = np.linalg.solve(a[:, :8], a[:, 8])
    return np.append(h, 1.0).reshape(3, 3)


def follow_background(bg: np.ndarray, m: np.ndarray, fill: np.ndarray) -> np.ndarray:
    """COMPOSITE_FOLLOW: bg warped by the homography m; pixels the real frame doesn't cover (the head turned past its edge)
    come from fill, the render. BORDER_REPLICATE smeared the edge column into stripes over up to a third of the frame."""
    import cv2

    h, w = fill.shape[:2]
    out = cv2.warpPerspective(bg, m, (w, h))
    covered = cv2.warpPerspective(np.ones(bg.shape[:2], np.uint8), m, (w, h), flags=cv2.INTER_NEAREST)
    return np.where(covered[..., None] > 0, out, fill)


def quat_wxyz_to_matrix(q) -> np.ndarray:
    w, x, y, z = (float(v) for v in q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])  # fmt: skip


def imu_velocities(root_quat, root_lin_w, root_ang_w, torso_quat, torso_lin_w, torso_ang_w):
    """Isaac world-frame link velocities in gear_sonic MuJoCo's conventions: base = [linear world, angular body frame]
    (free-joint qvel[:6]), torso IMU = [linear, angular] both in the torso frame (mj_objectVelocity, flg_local=1)."""
    rb, rt = quat_wxyz_to_matrix(root_quat), quat_wxyz_to_matrix(torso_quat)
    base = np.r_[np.asarray(root_lin_w, float), rb.T @ np.asarray(root_ang_w, float)]
    torso = np.r_[rt.T @ np.asarray(torso_lin_w, float), rt.T @ np.asarray(torso_ang_w, float)]
    return base, torso


class Scene:
    def __init__(self, out: Path, table_gap_cm: str | None, camera_json: Path = CAMERA_JSON, camera_port: int = 5555,
                 device: str = "cpu", render: bool = True, dds: bool = True):  # fmt: skip
        from isaaclab.app import AppLauncher

        # pinned (taskset, SIM_CPUS): size Kit's task pool, PhysX and USD threads to the allowed cores; Isaac otherwise
        # starts a thread per machine core and oversubscribes the pin (a 6-8 core pin then ran at ~0.8x real time)
        n = len(os.sched_getaffinity(0))
        kit = ""
        if n < (os.cpu_count() or n):
            os.environ.update(PXR_WORK_THREAD_LIMIT=str(n), OPENBLAS_NUM_THREADS=str(n))
            kit = f"--/plugins/carb.tasking.plugin/threadCount={n} --/physics/numThreads={max(1, n - 2)}"
        self.app = AppLauncher(headless=True, enable_cameras=True, device=device, kit_args=kit).app
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
        self.table_top_z = TABLE_TOP_Z  # the judge reads it (sim_trials)
        cam = load_camera(camera_json)

        def act(expr, friction):  # effort mode: no drive stiffness; MJCF passive damping 0.05, armature 0.01
            return ImplicitActuatorCfg(joint_names_expr=expr, stiffness=0.0, damping=0.05, armature=0.01,
                                       friction=friction, effort_limit_sim=1000.0, velocity_limit_sim=100.0)  # fmt: skip

        # GROUND_USD: a local copy of Isaac's Grid/default_environment.usd (+ Materials/Textures) for runs without
        # internet (private network namespace); the default fetches it from NVIDIA's asset server
        ground_usd = os.environ.get("GROUND_USD")
        ground_cfg = sim_utils.GroundPlaneCfg(usd_path=ground_usd) if ground_usd else sim_utils.GroundPlaneCfg()
        # FLOOR_FRICTION=mu (or static,dynamic): the feet-floor friction ("max" combine: the floor's value wins over the
        # feet's default 0.5). Default PhysX 0.5: the robot pivots ~4x more than on the real rubber tiles during a reach
        if (fl := parse_friction(os.environ.get("FLOOR_FRICTION"))) is not None:
            ground_cfg.physics_material = sim_utils.RigidBodyMaterialCfg(
                static_friction=fl[0], dynamic_friction=fl[1], friction_combine_mode="max")

        # contact experiments (the sim bottle tips under touches the real one survives): friction materials for the bottle
        # and the case top (average combine, as PhysX), and BOTTLE_COM_Z = centre of mass above the bottle's base (m)
        mats = {k: parse_friction(os.environ.get(f"{k}_FRICTION")) for k in ("BOTTLE", "CASE")}
        mats = {k: v and sim_utils.RigidBodyMaterialCfg(static_friction=v[0], dynamic_friction=v[1]) for k, v in mats.items()}

        @configclass
        class CfgScene(InteractiveSceneCfg):
            ground = AssetBaseCfg(prim_path="/World/ground", spawn=ground_cfg)
            # grey concrete look without the grid: a visual-only slab 1 mm above the physics ground
            floor = AssetBaseCfg(
                prim_path="/World/floor_look",
                spawn=sim_utils.CuboidCfg(size=(20.0, 20.0, 0.002), visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.25, 0.25, 0.24), roughness=0.9)),
                init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
            )  # fmt: skip
            light = AssetBaseCfg(prim_path="/World/light", spawn=sim_utils.DomeLightCfg(intensity=float(os.environ.get("DOME_LIGHT", "1200"))))
            robot = ArticulationCfg(
                prim_path="{ENV_REGEX_NS}/Robot",
                spawn=sim_utils.UrdfFileCfg(
                    asset_path=str(URDF), fix_base=False, merge_fixed_joints=True, activate_contact_sensors=True,
                    # one conversion reused by every run (default: a new timestamped dir per run, ~35 MB each)
                    usd_dir=os.environ.get("ISAAC_USD_DIR", f"/tmp/IsaacLab/{URDF.stem}") + (
                        "_decomp" if COLLIDER == "convex_decomposition" else ""),
                    collider_type=COLLIDER,
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
                    size=CASE,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
                    collision_props=sim_utils.CollisionPropertiesCfg(), physics_material=mats["CASE"],
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.03, 0.03, 0.03), roughness=0.35),
                ),
                init_state=RigidObjectCfg.InitialStateCfg(pos=(10.0, 0.0, CASE[2] / 2)),  # parked, as sim_scene
            )
            bottle = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/Bottle",
                spawn=sim_utils.CylinderCfg(  # the rigid root; its own cylinder is hidden, BOTTLE_PARTS collide
                    radius=BOTTLE_R, height=BOTTLE_H,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(),
                    mass_props=sim_utils.MassPropertiesCfg(mass=0.5), activate_contact_sensors=True,
                    # clear PET: a light glossy surface (OmniGlass renders black in the real-time RTX path)
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.72, 0.76, 0.78), roughness=0.05),
                ),  # fmt: skip
                init_state=RigidObjectCfg.InitialStateCfg(pos=(10.0, 2.0, 0.90)),
            )
            feet = ContactSensorCfg(
                prim_path="{ENV_REGEX_NS}/Robot/.*_ankle_roll_link", filter_prim_paths_expr=["/World/ground"]
            )
            touch = ContactSensorCfg(  # the hands' push on the bottle (contact experiments)
                prim_path="{ENV_REGEX_NS}/Bottle", filter_prim_paths_expr=[f"/World/envs/env_0/Robot/{n}" for n in TOUCH]
            )

        self.sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=DT, render_interval=1, device=device))
        self.scene = InteractiveScene(CfgScene(num_envs=1, env_spacing=4.0))
        from pxr import UsdGeom

        stage = sim_utils.get_current_stage()
        UsdGeom.Imageable(stage.GetPrimAtPath("/World/envs/env_0/Bottle/geometry/mesh")).MakeInvisible()
        # clear PET with water: a thin glass shell around a light grey core matches the real bottle's look in the head
        # frames (body brightness 134 vs 144 real; the opaque grey was 190, glass alone 71: it shows the black case)
        import carb

        carb.settings.get_settings().set("/rtx/translucency/enabled", True)  # glass renders black without it
        pet = sim_utils.GlassMdlCfg(glass_color=(0.95, 0.97, 1.0), glass_ior=1.45, thin_walled=True)
        water = sim_utils.CylinderCfg(radius=0.027, height=0.100, visual_material=sim_utils.PreviewSurfaceCfg(
            diffuse_color=(0.88, 0.915, 0.94), roughness=0.3))  # visual only
        water.func("/World/envs/env_0/Bottle/water", water, translation=(0.0, 0.0, 0.055 - BOTTLE_H / 2))
        # colliders under the rigid root: one body, its mass stays 0.5 kg
        for name, shape, r, h, zc in BOTTLE_PARTS:
            look = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.78, 0.86, 0.12)) if name == "cap" else pet
            part = (sim_utils.ConeCfg if shape == "cone" else sim_utils.CylinderCfg)(
                radius=r, height=h, collision_props=sim_utils.CollisionPropertiesCfg(), visual_material=look,
                physics_material=mats["BOTTLE"])
            part.func(f"/World/envs/env_0/Bottle/{name}", part, translation=(0.0, 0.0, zc - BOTTLE_H / 2))
        if os.environ.get("BOTTLE_COM_Z"):  # default: uniform density over the colliders (~6.5 cm above the base)
            from pxr import Gf, UsdPhysics

            UsdPhysics.MassAPI(stage.GetPrimAtPath("/World/envs/env_0/Bottle")).CreateCenterOfMassAttr(
                Gf.Vec3f(0.0, 0.0, float(os.environ["BOTTLE_COM_Z"]) - BOTTLE_H / 2))
        # Isaac's URDF importer drops the URDF colours (every link renders white): satin aluminium for the G1's shell (the
        # head-camera forearm 163 vs the real 192-210 grey levels; gear_sonic g1_29dof_textured.usd's 0.575 / 0.9 / 0.25
        # renders it at 140, 0.9 / 0.2 looks like white plastic), its textured dark parts, and the black Dex3 hands
        looks = {"g1_metal": sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85,) * 3, metallic=0.4, roughness=0.4),
                 "g1_dark": sim_utils.PreviewSurfaceCfg(diffuse_color=(0.105,) * 3, metallic=0.85, roughness=0.3),
                 "dex3_black": sim_utils.PreviewSurfaceCfg(diffuse_color=(0.03, 0.03, 0.03), roughness=0.4)}  # fmt: skip
        for name, look in looks.items():
            look.func(f"/World/Looks/{name}", look)
        for path in sim_utils.find_matching_prim_paths("/World/envs/env_0/Robot/.*"):
            sim_utils.bind_visual_material(path, f"/World/Looks/{robot_look(path.rsplit('/', 1)[1])}")
        k = [cam["fx"], 0.0, cam["cx"], 0.0, cam["fy"], cam["cy"], 0.0, 0.0, 1.0]
        # COMPOSITE_BG=<real head frame .npy, the camera server's array convention>: publish the rendered robot and bottle
        # over it (a test of the rendered scene's look; the case and room come from the real frame)
        self.bg = np.load(os.environ["COMPOSITE_BG"]) if os.environ.get("COMPOSITE_BG") else None
        # COMPOSITE_FOLLOW=1: warp that background with the head's motion since the trial's start (the case-top plane's
        # homography from the camera pose at place_bottle to the current one), as the real view moves during a reach
        self.bg_follow, self.bg_ref, self.cam = os.environ.get("COMPOSITE_FOLLOW") == "1", None, cam
        self.camera = Camera(
            CameraCfg(
                prim_path=f"/World/envs/env_0/Robot/{cam['parent']}/head_cam",
                height=cam["height"], width=cam["width"], update_period=0.0,
                data_types=["rgb"] + (["instance_id_segmentation_fast"] if self.bg is not None else []),
                colorize_instance_id_segmentation=False,
                spawn=sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
                    intrinsic_matrix=k, width=cam["width"], height=cam["height"]
                ),
                offset=CameraCfg.OffsetCfg(pos=tuple(cam["pos"]), rot=tuple(cam["rot_wxyz"]), convention="world"),
            )
        )  # fmt: skip
        # THIRD_PERSON_CAM=1 (or "px,py,pz,tx,ty,tz"): a fixed world camera for videos, rendered on request only
        self.view_cam = None
        if os.environ.get("THIRD_PERSON_CAM"):
            v = [float(x) for x in os.environ["THIRD_PERSON_CAM"].split(",")] if "," in os.environ["THIRD_PERSON_CAM"] \
                else [1.25, -0.95, 1.45, 0.38, -0.05, 0.92]
            self.view_cam = Camera(CameraCfg(
                prim_path="/World/third_cam", height=480, width=640, data_types=["rgb"], update_period=0.0,
                spawn=sim_utils.PinholeCameraCfg(focal_length=18.0, horizontal_aperture=20.955),
                offset=CameraCfg.OffsetCfg(pos=tuple(v[:3]), rot=tuple(look_at_wxyz(v[:3], v[3:])), convention="world"),
            ))  # fmt: skip
        self.sim.reset()
        self.robot, self.table_obj = self.scene["robot"], self.scene["table"]
        self.bottle, self.feet, self.touch = self.scene["bottle"], self.scene["feet"], self.scene["touch"]
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
        self.dds, self.dds_proc = self._start_dds() if dds else (None, None)

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
        """dds_publisher.py on the wbc yaml's DDS domain / interface (0, lo), as gear_sonic's simulator_factory."""
        cfg = {**self.config, "GEAR_ROOT": str(GEAR_ROOT), "CYCLONEDDS_LIB": str(CYCLONEDDS_LIB)}
        cfg_path = self.out / "dds_config.json"
        cfg_path.write_text(json.dumps(cfg, default=str))
        ex = Exchange()
        log = open(self.out / "dds_publisher.log", "w")  # noqa: SIM115 - the child writes it for the whole run
        proc = subprocess.Popen(
            [sys.executable, "-u", str(Path(__file__).with_name("dds_publisher.py")), *ex.names, str(cfg_path),
             str(os.getpid())], stdout=log, stderr=subprocess.STDOUT,
        )  # fmt: skip
        deadline = time.monotonic() + 30
        while "up:" not in (self.out / "dds_publisher.log").read_text():
            if proc.poll() is not None or time.monotonic() > deadline:
                proc.kill()
                ex.close(unlink=True)
                raise SystemExit(f"dds_publisher did not start; see {self.out / 'dds_publisher.log'}")
            time.sleep(0.1)
        return ex, proc

    def _bridge_cmd(self) -> None:
        """self.cmd from the deploy's latest commands (zero until it has sent any)."""
        if self.dds is not None and (c := self.dds.read_cmd()) is not None:
            self.cmd = c

    @property
    def bridge(self):
        """Read-only view of the latest Dex3 commands for the host's gain log (the bridge itself is in the
        publisher process)."""
        c = self.cmd
        side = [[SimpleNamespace(kp=c["kp"][i], kd=c["kd"][i]) for i in range(NB + k * NH, NB + (k + 1) * NH)]
                for k in (0, 1)]  # fmt: skip
        return SimpleNamespace(left_hand_cmd=SimpleNamespace(motor_cmd=side[0]),
                               right_hand_cmd=SimpleNamespace(motor_cmd=side[1]))  # fmt: skip

    def joint_state(self) -> tuple[np.ndarray, np.ndarray]:
        ids = self.body_ids + self.hand_ids
        d = self.robot.data
        return d.joint_pos[0, ids].cpu().numpy(), d.joint_vel[0, ids].cpu().numpy()

    def _state(self) -> dict:
        d, ids = self.robot.data, self.body_ids + self.hand_ids
        root = d.root_link_state_w[0].cpu().numpy()  # link origin, as MuJoCo's free joint (not the centre of mass)
        torso_quat = d.body_quat_w[0, self.torso_id].cpu().numpy()
        root_vel, torso_vel = imu_velocities(
            root[3:7], root[7:10], root[10:13], torso_quat,
            d.body_link_lin_vel_w[0, self.torso_id].cpu().numpy(), d.body_link_ang_vel_w[0, self.torso_id].cpu().numpy(),
        )  # fmt: skip
        acc = (root_vel - self.prev_root_vel) / DT  # finite difference, no gravity term (MuJoCo qacc[:6])
        self.prev_root_vel = root_vel.copy()
        return {
            "q": d.joint_pos[0, ids].cpu().numpy(), "dq": d.joint_vel[0, ids].cpu().numpy(),
            "ddq": d.joint_acc[0, ids].cpu().numpy(), "tau": self.tau, "root_pose": root[:7], "root_vel": root_vel,
            "root_acc": acc, "torso_quat": torso_quat, "torso_vel": torso_vel, "time": self.t,
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
            if self.dds is not None:
                self.dds.write_state(self._state())
        if render:
            self.next_frame += 1 / CAMERA_FPS
            self._publish_frame()

    def camera_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Head camera position and rotation in the world, Isaac "world" convention (x forward, z up): the torso link's
        physics pose and the mount (the camera prim's own pose is not refreshed with CPU physics)."""
        d = self.robot.data
        rt = quat_wxyz_to_matrix(d.body_link_quat_w[0, self.torso_id].cpu().numpy())
        return d.body_link_pos_w[0, self.torso_id].cpu().numpy() + rt @ np.asarray(self.cam["pos"]), rt @ quat_wxyz_to_matrix(
            self.cam["rot_wxyz"])

    def third_person_rgb(self) -> np.ndarray:
        """THIRD_PERSON_CAM view now (RGB uint8)."""
        self.view_cam.update(0.0, force_recompute=True)
        return self.view_cam.data.output["rgb"][0, ..., :3].cpu().numpy()

    def _publish_frame(self) -> None:
        import cv2
        import zmq

        t0 = time.perf_counter()
        self.camera.update(0.0, force_recompute=True)
        rgb = self.camera.data.output["rgb"][0, ..., :3].cpu().numpy()
        if rgb.shape[:2] != (H, W):
            raise SystemExit(f"head camera gives {rgb.shape[1]}x{rgb.shape[0]}, not {W}x{H}")
        if self.bg is not None:
            seg = self.camera.data.output["instance_id_segmentation_fast"][0, ..., 0].cpu().numpy()
            bg = self.bg
            if self.bg_follow and self.bg_ref is not None:
                m = plane_homography(self.cam, *self.bg_ref, *self.camera_pose(), TABLE_TOP_Z)
                bg = follow_background(self.bg, m, rgb)
            rgb = composite(rgb, seg, self.camera.data.info[0]["instance_id_segmentation_fast"]["idToLabels"], bg)
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
            pose = torch.tensor([[edge + CASE[0] / 2, root[1], CASE[2] / 2, 1.0, 0.0, 0.0, 0.0]], device=self.sim.device)
            self.table_obj.write_root_pose_to_sim(pose)
        self.mark(f"table placed: near edge {self.table_gap:g} cm from the torso (x = {edge:.3f} m)")

    def reset_trial(self, seed: int, bottle_xy, jitter_m: float) -> None:
        """Robot back to the deploy's standing pose (as key 'backspace'); bottle parked off the table (the caller places
        it once the robot has settled); trial records cleared."""
        import torch

        self.key("backspace")
        with self.lock:
            dev = self.sim.device
            park = torch.tensor([[10.0, 2.0, BOTTLE_H / 2, 1.0, 0.0, 0.0, 0.0]], dtype=torch.float32, device=dev)
            self.bottle.write_root_pose_to_sim(park)
            self.bottle.write_root_velocity_to_sim(torch.zeros(1, 6, device=dev))
        self.rec = {k: [] for k in REC_KEYS}
        self.next_rec = self.t
        self.fall = FallDetector()

    def place_bottle(self, seed: int, bottle_xy, jitter_m: float) -> None:
        """Bottle upright on the table at bottle_xy (world, table placed) + uniform +-jitter_m (scalar or x, y), at rest."""
        import torch

        x, y = sample_bottle_xy(seed, bottle_xy, jitter_m)
        with self.lock:
            dev = self.sim.device
            z = TABLE_TOP_Z + BOTTLE_H / 2
            pose = torch.tensor([[x, y, z, 1.0, 0.0, 0.0, 0.0]], dtype=torch.float32, device=dev)
            self.bottle.write_root_pose_to_sim(pose)
            self.bottle.write_root_velocity_to_sim(torch.zeros(1, 6, device=dev))
        self.mark(f"bottle placed: seed {seed}, at ({x:.3f}, {y:.3f})")
        if self.bg_follow:
            self.bg_ref = self.camera_pose()  # the background frame shows the scene from the standing pose at the start

    def cut_trial(self, i: int) -> None:
        if self.rec["t"]:
            np.savez_compressed(
                self.out / f"sim_state_{i:03d}.npz", **{k: np.array(v) for k, v in self.rec.items()}
            )
        self.rec = {k: [] for k in REC_KEYS}

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
                "cmd_q": self.cmd["q"][: len(BODY)].copy(),
                "touch_force": self.touch.data.force_matrix_w[0, 0].norm(dim=-1).cpu().numpy(),  # N, per TOUCH link
                "cmd_kp": self.cmd["kp"][: len(BODY)].copy(), "cmd_kd": self.cmd["kd"][: len(BODY)].copy(),  # the deploy's body setpoints (tracking vs the measured body_q)
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
        if self.dds is not None:
            self.dds.set_stop()
            with contextlib.suppress(subprocess.TimeoutExpired):
                self.dds_proc.wait(timeout=2)
            self.dds_proc.kill()
            self.dds.close(unlink=True)
