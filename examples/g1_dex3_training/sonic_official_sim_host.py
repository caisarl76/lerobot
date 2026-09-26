"""Sim host for sonic_policy_streamer.py: gear_sonic MuJoCo + NVIDIA g1_deploy_onnx_ref, recorded.

The streamer owns every deploy command (ZMQ "command"/"pose" on 5556). This process runs the physics, performs
the confirmed operator steps that belong to the simulator, and exchanges gate files with the streamer:
  deploy prints "Init Done"               -> write GATE/deploy_ready (streamer sends start + planner)
  deploy enters CONTROL (+4 s, still hung) -> sim key "9" (release band), +1 s sim key "Backspace" (reset)
  robot settled for 5 s                   -> write GATE/settled (streamer hands off to POSE and streams)
  streamer writes GATE/done               -> stop 3 s later
Logs robot state every 20 ms with wall-clock time (to align with the streamer's log) and renders a video.
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

os.environ.setdefault("MUJOCO_GL", "egl")
import cv2  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, "/workspace/GR00T-WholeBodyControl")
from gear_sonic.utils.mujoco_sim.base_sim import BaseSimulator  # noqa: E402
from gear_sonic.utils.mujoco_sim.configs import SimLoopConfig  # noqa: E402

OUT, GATE = Path(sys.argv[1]), Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
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
BODY = (
    [f"{s}_{j}_joint" for s in ("left", "right") for j in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")]
    + ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]
    + [f"{s}_{j}_joint" for s in ("left", "right") for j in ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")]
)  # fmt: skip
HANDS = [
    f"left_hand_{j}_joint"
    for j in ("thumb_0", "thumb_1", "thumb_2", "middle_0", "middle_1", "index_0", "index_1")
] + [
    f"right_hand_{j}_joint"
    for j in ("thumb_0", "thumb_1", "thumb_2", "index_0", "index_1", "middle_0", "middle_1")
]
FPS, W, H = 30, 640, 480

config = SimLoopConfig(interface="sim", enable_onscreen=False, hand_profile="dex3").load_wbc_yaml()
# Optional tabletop (TABLE_GAP_CM): 80 cm table, 3 cm top. It starts parked 10 m away (the elastic-band hang and
# the Backspace reset would otherwise drop the hands onto it) and moves in once the robot has settled standing in
# planner mode: near edge TABLE_GAP_CM in front of the torso front (pelvis x + 0.08 m). Written next to the
# original scene inside this container so its mesh paths resolve.
TABLE_GAP = os.environ.get("TABLE_GAP_CM")
if TABLE_GAP:
    root = Path("/workspace/GR00T-WholeBodyControl")
    scene = root / config["ROBOT_SCENE"]
    xml = scene.read_text().replace(
        "</mujoco>",
        '<worldbody><body name="table" pos="10 0 0.785"><geom name="table_top" type="box" '
        'size="0.4 0.8 0.015" rgba="0.62 0.46 0.3 1"/></body></worldbody></mujoco>',
    )
    (scene.parent / "scene_43dof_table.xml").write_text(xml)
    config["ROBOT_SCENE"] = str((scene.parent / "scene_43dof_table.xml").relative_to(root))
sim = BaseSimulator(config=config, onscreen=False, offscreen=False, env_name="default")
env = sim.sim_env
m, d = env.mj_model, env.mj_data
SIM_DT = config["SIMULATE_DT"]
pelvis = m.body("pelvis").id
floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
qadr = np.array([m.jnt_qposadr[m.joint(n).id] for n in BODY])
hadr = np.array([m.jnt_qposadr[m.joint(n).id] for n in HANDS])
palm_ids = [[m.body(f"{s}_hand_{f}_0_link").id for f in ("index", "middle")] for s in ("left", "right")]
wrist_ids = [m.body(f"{s}_wrist_yaw_link").id for s in ("left", "right")]
table = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "table_top")
arm_geoms = [
    g
    for g in range(m.ngeom)
    if (m.geom_contype[g] or m.geom_conaffinity[g])
    and any(k in m.body(m.geom_bodyid[g]).name for k in ("elbow", "wrist", "hand", "shoulder_yaw"))
]


def table_clearance():
    """Closest arm collision geom to the table (m) and arm/hand-table contact count; nan/0 without a table."""
    if table < 0:
        return np.nan, 0
    ft = np.zeros(6)
    clear = min(mujoco.mj_geomDistance(m, d, g, table, 0.5, ft) for g in arm_geoms)
    hits = sum(1 for i in range(d.ncon) if table in (d.contact[i].geom1, d.contact[i].geom2))
    return clear, hits


master, slave = pty.openpty()
proc = subprocess.Popen(DEPLOY, stdin=slave, stdout=slave, stderr=slave, close_fds=True)
os.close(slave)
lines, sim_t, lock = [], [0.0], threading.Lock()
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
            log_f.write(f"[{sim_t[0]:8.3f}] {text}\n")
            log_f.flush()


threading.Thread(target=reader, daemon=True).start()


def mark(event):
    print(f"[{sim_t[0]:8.3f}] {event}", flush=True)
    with open(OUT / "events.txt", "a") as f:
        f.write(f"{time.time():.3f} {sim_t[0]:.3f} {event}\n")


def seen(pattern):
    return any(pattern in text for text in lines)


view = {"phase": "hanging on elastic band", "stop": False}
snap = mujoco.MjData(m)
cam = mujoco.MjvCamera()
cam.type, cam.trackbodyid, cam.distance, cam.azimuth, cam.elevation = (
    mujoco.mjtCamera.mjCAMERA_TRACKING,
    pelvis,
    2.4,
    150,
    -15,
)
writer = cv2.VideoWriter(str(OUT / "sim_raw.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))


def render_loop():
    renderer = mujoco.Renderer(m, H, W)
    next_t = 0.0
    while not view["stop"]:
        with lock:
            t, ready = sim_t[0], sim_t[0] >= next_t
            if ready:
                snap.qpos[:], snap.qvel[:] = d.qpos, d.qvel
        if not ready:
            time.sleep(0.002)
            continue
        next_t += 1 / FPS
        mujoco.mj_forward(m, snap)
        renderer.update_scene(snap, camera=cam)
        img = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
        for txt, y in ((f"t = {t:6.2f} s   {view['phase']}", 24), (f"pelvis z {snap.qpos[2]:.3f} m", 46)):
            cv2.putText(img, txt, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, txt, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        writer.write(img)


threading.Thread(target=render_loop, daemon=True).start()

rec = {
    k: []
    for k in (
        "wall",
        "t",
        "body_q",
        "hand_q",
        "pelvis",
        "palm",
        "wrist_R",
        "floor_contacts",
        "table_clear",
        "table_hits",
    )
}
steps, t_control, t_settled, t_done, next_rec = {}, None, None, None, 0.0
mark("sim + deploy started")
wall0 = time.monotonic()
try:
    while proc.poll() is None:
        with lock:
            env.sim_step()
            sim_t[0] += SIM_DT
        t = sim_t[0]
        if "ready" not in steps and seen("Init Done"):
            steps["ready"] = t
            (GATE / "deploy_ready").touch()
            mark("Init Done -> deploy_ready")
            view["phase"] = "Init Done: waiting for the streamer's start command"
        if t_control is None and seen("transitioning to CONTROL state"):
            t_control = t
            mark("deploy in CONTROL (planner mode)")
            view["phase"] = "planner mode, hung"
        if t_control is not None and "band" not in steps and t >= t_control + 4:
            steps["band"] = t
            sim.handle_keyboard_button("9")
            mark("sim key '9': band released")
            view["phase"] = "band released"
        if "band" in steps and "reset" not in steps and t >= steps["band"] + 1:
            steps["reset"] = t
            sim.handle_keyboard_button("backspace")
            mark("sim key 'Backspace': reset onto ground")
            view["phase"] = "settling on ground"
        if "reset" in steps and t_settled is None and t >= steps["reset"] + 5:
            t_settled = t
            if table >= 0:  # bring the table in now that the robot stands in planner mode
                with lock:
                    edge = d.qpos[0] + 0.08 + float(TABLE_GAP) / 100
                    m.body_pos[m.geom_bodyid[table]] = [edge + 0.4, d.qpos[1], 0.785]
                    mujoco.mj_forward(m, d)
                mark(f"table placed: near edge {float(TABLE_GAP):g} cm from the torso (x = {edge:.3f} m)")
            (GATE / "settled").touch()
            mark("settled -> streamer may hand off to POSE")
            view["phase"] = "policy streaming (POSE mode)"
        if t_done is None and (GATE / "done").exists():
            t_done = t
            mark("streamer done")
            view["phase"] = "streamer done"
        if t_done is not None and t >= t_done + 3:
            break
        if t_settled is not None and t + 1e-9 >= next_rec:
            next_rec = t + 0.02
            with lock:
                ncon = sum(1 for i in range(d.ncon) if floor in (d.contact[i].geom1, d.contact[i].geom2))
                clear, hits = table_clearance()
                for key, val in (
                    ("wall", time.time()),
                    ("t", t),
                    ("body_q", d.qpos[qadr].copy()),
                    ("hand_q", d.qpos[hadr].copy()),
                    ("pelvis", d.qpos[:7].copy()),
                    ("palm", np.stack([d.xpos[ids].mean(0) for ids in palm_ids])),
                    ("wrist_R", np.stack([d.xmat[i].reshape(3, 3) for i in wrist_ids])),
                    ("floor_contacts", ncon),
                    ("table_clear", clear),
                    ("table_hits", hits),
                ):
                    rec[key].append(val)
        if t > 1800:
            mark("timeout")
            break
        lag = sim_t[0] - (time.monotonic() - wall0)
        if lag > 0:
            time.sleep(lag)
finally:
    mark("stop" if proc.poll() is None else f"deploy exited with code {proc.returncode}")
    with contextlib.suppress(OSError):
        os.write(master, b"o")
    time.sleep(0.5)
    proc.terminate()
    view["stop"] = True
    time.sleep(0.3)
    writer.release()
    if rec["t"]:
        np.savez_compressed(OUT / "sim_state.npz", **{k: np.array(v) for k, v in rec.items()})
