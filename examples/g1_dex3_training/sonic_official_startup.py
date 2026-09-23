"""Official SONIC v1.1 sim2sim startup, recorded: gear_sonic MuJoCo loop + C++ g1_deploy_onnx_ref.

Sequence (operator procedure): robot hangs on the elastic band -> deploy prints "Init Done" -> "]" starts
control -> Enter enters planner mode -> sim key "9" releases the band -> sim key "Backspace" resets the
robot onto the ground -> settle. Keys to the deploy go through a pty (its keyboard handler reads stdin);
sim keys call BaseSimulator.handle_keyboard_button, the same path as a viewer keypress.
Runs inside jihun/sonic-vla-sim with /opt/sonic-sim/bin/python from gear_sonic_deploy/.
"""

import csv
import json
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

OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "/out")
OUT.mkdir(parents=True, exist_ok=True)
DEPLOY = [
    "bash",
    "-lc",
    "source scripts/setup_env.sh >/dev/null 2>&1; exec target/release/g1_deploy_onnx_ref lo "
    "policy/sonic_v1_1/model_decoder.onnx reference/example/ "
    "--obs-config policy/sonic_v1_1/observation_config.yaml --encoder-file policy/sonic_v1_1/model_encoder.onnx "
    "--planner-file planner/target_vel/V2/planner_sonic.onnx --input-type keyboard --output-type zmq "
    "--zmq-host localhost --disable-crc-check",
]
FPS, W, H = 30, 640, 480
MAX_WAIT_INIT_S, SETTLE_S = 600, 12

config = SimLoopConfig(interface="sim", enable_onscreen=False, hand_profile="dex3").load_wbc_yaml()
sim = BaseSimulator(config=config, onscreen=False, offscreen=False, env_name="default")
env = sim.sim_env
m, d = env.mj_model, env.mj_data
pelvis = m.body("pelvis").id
floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
(OUT / "sim_config.json").write_text(
    json.dumps({k: v for k, v in config.items() if isinstance(v, (str, int, float, bool))}, indent=1)
)

# ---- deploy process on a pty, log every line with sim time
master, slave = pty.openpty()
proc = subprocess.Popen(DEPLOY, stdin=slave, stdout=slave, stderr=slave, close_fds=True)
os.close(slave)
log_f = open(OUT / "deploy.log", "w")  # noqa: SIM115 - written by the reader thread for the whole run
events, lines = [], []
sim_t = [0.0]


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
            lines.append((sim_t[0], text))
            log_f.write(f"[{sim_t[0]:8.3f}] {text}\n")
            log_f.flush()


threading.Thread(target=reader, daemon=True).start()


def mark(name):
    events.append({"t": round(sim_t[0], 3), "event": name})
    print(f"[{sim_t[0]:8.3f}] {name}", flush=True)


def send(ch, name):
    os.write(master, ch.encode())
    mark(name)


def seen(pattern, since=0.0):
    return any(pattern in text for t, text in lines if t >= since)


# ---- renderer on a snapshot copy so rendering never stalls the physics loop
snap = mujoco.MjData(m)
lock = threading.Lock()
cam = mujoco.MjvCamera()
cam.type, cam.trackbodyid, cam.distance, cam.azimuth, cam.elevation = (
    mujoco.mjtCamera.mjCAMERA_TRACKING,
    pelvis,
    3.0,
    135,
    -12,
)
writer = cv2.VideoWriter(str(OUT / "startup_raw.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
telemetry = []
state = {"phase": "sim start: hanging on elastic band", "stop": False}


def tilt_deg(q):
    w, x, y, z = q
    return float(np.degrees(np.arccos(np.clip(1 - 2 * (x * x + y * y), -1, 1))))


def render_loop():
    renderer = mujoco.Renderer(m, H, W)
    next_t = 0.0
    while not state["stop"]:
        with lock:
            t = sim_t[0]
            if t < next_t:
                ready = False
            else:
                ready = True
                snap.qpos[:], snap.qvel[:], snap.time = d.qpos, d.qvel, d.time
                ncon = sum(1 for i in range(d.ncon) if floor in (d.contact[i].geom1, d.contact[i].geom2))
                band = bool(env.elastic_band and env.elastic_band.enable)
        if not ready:
            time.sleep(0.002)
            continue
        next_t += 1 / FPS
        mujoco.mj_forward(m, snap)
        renderer.update_scene(snap, camera=cam)
        img = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
        z, tl = float(snap.qpos[2]), tilt_deg(snap.qpos[3:7])
        row = {
            "t": round(t, 3),
            "phase": state["phase"],
            "pelvis_z": round(z, 4),
            "tilt_deg": round(tl, 2),
            "floor_contacts": ncon,
            "band": band,
        }
        telemetry.append(row)
        for i, txt in enumerate(
            [
                f"t = {t:6.2f} s   {state['phase']}",
                f"pelvis z {z:.3f} m   tilt {tl:4.1f} deg   floor contacts {ncon}   band {'ON' if band else 'off'}",
            ]
        ):
            cv2.putText(
                img, txt, (12, 24 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 3, cv2.LINE_AA
            )
            cv2.putText(
                img, txt, (12, 24 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA
            )
        writer.write(img)


threading.Thread(target=render_loop, daemon=True).start()

# ---- scripted operator, driven by sim time (sim runs in real time like run_sim_loop)
plan = []  # (condition, action)
t_init = [None]


def step_plan():
    t = sim_t[0]
    if t_init[0] is None:
        if seen("Init Done"):
            t_init[0] = t
            mark("deploy printed 'Init Done'")
            state["phase"] = "Init Done: deploy holding default pose"
        return t > MAX_WAIT_INIT_S
    dt = t - t_init[0]
    for key, at, action in (
        ("start", 2.0, lambda: send("]", "']' start control")),
        ("planner", 5.0, lambda: send("\r", "Enter: planner mode")),
        ("band", 9.0, lambda: (sim.handle_keyboard_button("9"), mark("sim key '9': elastic band released"))),
        (
            "reset",
            10.0,
            lambda: (
                sim.handle_keyboard_button("backspace"),
                mark("sim key 'Backspace': robot reset onto ground"),
            ),
        ),
    ):
        if dt >= at and key not in state:
            state[key] = True
            action()
            state["phase"] = {
                "start": "control started (reference-motion mode, paused)",
                "planner": "planner mode (idle stand)",
                "band": "band released: falling",
                "reset": "reset onto ground: settling",
            }[key]
    return dt > 10.0 + SETTLE_S


mark("sim + deploy started")
try:
    wall0 = time.monotonic()
    while proc.poll() is None:
        with lock:
            env.sim_step()
            sim_t[0] += env.sim_dt if hasattr(env, "sim_dt") else config["SIMULATE_DT"]
        if step_plan():
            break
        lag = sim_t[0] - (time.monotonic() - wall0)
        if lag > 0:
            time.sleep(lag)
finally:
    mark("stop" if proc.poll() is None else f"deploy exited with code {proc.returncode}")
    try:
        send("o", "'o' emergency stop")
        time.sleep(0.5)
    except OSError:
        pass
    proc.terminate()
    state["stop"] = True
    time.sleep(0.3)
    writer.release()
    with open(OUT / "telemetry.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(telemetry[0]))
        w.writeheader()
        w.writerows(telemetry)
    (OUT / "events.json").write_text(json.dumps(events, indent=1))
    print(
        "frames", len(telemetry), "realtime lag s", round(time.monotonic() - wall0 - sim_t[0], 2), flush=True
    )
