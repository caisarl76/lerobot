"""Replay stored sonic78 tokens through the official SONIC v1.1 deploy (decoder only) in gear_sonic MuJoCo.

Startup = the confirmed operator sequence (Init Done -> start in planner mode -> sim "9" -> sim "Backspace"
-> settle), driven over ZMQ like gear_sonic/scripts/run_vla_inference.py (initial_pose="standing"):
latent initial token -> POSE mode -> per episode: 1 s linear token blend initial->first token, 1 s hold,
stream the episode's stored tokens + Dex3 hands at 50 Hz (30 Hz tokens held), 1 s blend back, 2 s hold.
Logs robot state every 50 Hz tick; renders video with ghost spheres at the palm positions the original
actions intend (FK with nominal legs/waist, placed on the robot's actual pelvis).
Usage (in jihun/sonic-vla-sim, cwd gear_sonic_deploy): python sonic_official_replay.py OUT EP.npz [...]
"""

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
import zmq  # noqa: E402

sys.path.insert(0, "/workspace/GR00T-WholeBodyControl")
sys.path.insert(0, "/code")
from gear_sonic.utils.inference.initial_poses import LATENT_INITIAL_MOTION_TOKEN  # noqa: E402
from gear_sonic.utils.mujoco_sim.base_sim import BaseSimulator  # noqa: E402
from gear_sonic.utils.mujoco_sim.configs import SimLoopConfig  # noqa: E402
from gear_sonic.utils.teleop.zmq.zmq_planner_sender import (  # noqa: E402
    build_command_message,
    pack_pose_message,
)
from sonic_targets import NOMINAL_BODY  # noqa: E402
from sonic_token_stream import ChunkResampler  # noqa: E402

OUT = Path(sys.argv[1])
EPISODES = [Path(p) for p in sys.argv[2:]]
OUT.mkdir(parents=True, exist_ok=True)
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
    [
        f"{s}_{j}_joint"
        for s in ("left", "right")
        for j in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")
    ]
    + ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]
    + [
        f"{s}_{j}_joint"
        for s in ("left", "right")
        for j in (
            "shoulder_pitch",
            "shoulder_roll",
            "shoulder_yaw",
            "elbow",
            "wrist_roll",
            "wrist_pitch",
            "wrist_yaw",
        )
    ]
)
HANDS = [
    f"left_hand_{j}_joint"
    for j in ("thumb_0", "thumb_1", "thumb_2", "middle_0", "middle_1", "index_0", "index_1")
] + [
    f"right_hand_{j}_joint"
    for j in ("thumb_0", "thumb_1", "thumb_2", "index_0", "index_1", "middle_0", "middle_1")
]
FPS, W, H, TICK = 30, 640, 480, 0.02

config = SimLoopConfig(interface="sim", enable_onscreen=False, hand_profile="dex3").load_wbc_yaml()
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
fk = mujoco.MjData(m)


def palms(data):
    return np.stack([data.xpos[ids].mean(0) for ids in palm_ids]), np.stack(
        [data.xmat[i].reshape(3, 3) for i in wrist_ids]
    )


def intended(arms14):
    """Palm points + wrist frames the original action intends, on the robot's current pelvis pose."""
    fk.qpos[:7] = d.qpos[:7]
    fk.qpos[qadr] = np.r_[NOMINAL_BODY[:15], arms14]
    mujoco.mj_kinematics(m, fk)
    return palms(fk)


# ---- deploy on a pty (stdin keys only needed for 'o'); ZMQ PUB like run_vla_inference
master, slave = pty.openpty()
proc = subprocess.Popen(DEPLOY, stdin=slave, stdout=slave, stderr=slave, close_fds=True)
os.close(slave)
ctx = zmq.Context()
pub = ctx.socket(zmq.PUB)
pub.bind("tcp://*:5556")
# Gradual handoff (user decision 2026-09-24): start POSE mode from the planner's current token, then blend to the
# standing token over HANDOFF_BLEND_S seconds. 0 keeps the original jump straight to the standing token.
HANDOFF_S = float(os.environ.get("HANDOFF_BLEND_S", "0"))
state_sub = ctx.socket(zmq.SUB)
state_sub.setsockopt_string(zmq.SUBSCRIBE, "g1_debug")
state_sub.setsockopt(zmq.CONFLATE, 1)
state_sub.connect("tcp://localhost:5557")


def deploy_token():
    """The token the deploy is decoding right now (g1_debug token_state), or None."""
    import msgpack

    try:
        raw = state_sub.recv(zmq.NOBLOCK)
    except zmq.Again:
        return None
    tok = np.asarray(msgpack.unpackb(raw[len("g1_debug") :], raw=False).get("token_state", []), np.float32)
    return tok if tok.shape == (64,) else None


lines, events, sim_t = [], [], [0.0]
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
            lines.append((sim_t[0], text))
            log_f.write(f"[{sim_t[0]:8.3f}] {text}\n")
            log_f.flush()


threading.Thread(target=reader, daemon=True).start()


def mark(name):
    events.append({"t": round(sim_t[0], 3), "event": name})
    print(f"[{sim_t[0]:8.3f}] {name}", flush=True)


frame_counter = [0]


def send_token(tok, hands14, frame=None):
    idx = frame_counter[0] if frame is None else frame
    pub.send(
        pack_pose_message(
            {
                "token_state": np.asarray(tok, np.float32).reshape(1, 64),
                "frame_index": np.array([idx], np.int64),
                "left_hand_joints": np.asarray(hands14[:7], np.float32).reshape(1, 7),
                "right_hand_joints": np.asarray(hands14[7:], np.float32).reshape(1, 7),
            },
            topic="pose",
            version=4,
        )
    )
    if frame is None:
        frame_counter[0] += 1


def command(start, planner, name):
    pub.send(build_command_message(start=start, stop=not start, planner=planner))
    mark(name)


# ---- token schedule after POSE handoff: list of (phase, episode, source_frame, token, hands, original_arms)
eps = [dict(np.load(p, allow_pickle=True), name=p.stem) for p in EPISODES]
init_tok, zero_h = LATENT_INITIAL_MOTION_TOKEN.astype(np.float32), np.zeros(14, np.float32)
schedule = [("hold initial", -1, -1, init_tok, zero_h, None)] * 100
for e_i, e in enumerate(eps):
    tok, hands, orig, fps = e["tokens"], e["hands"], e["original"], float(e["fps"])
    for a in np.linspace(0, 1, 51)[1:]:
        schedule.append(("blend in", e_i, 0, (1 - a) * init_tok + a * tok[0], a * hands[0], orig[0, :14]))
    schedule += [("hold first frame", e_i, 0, tok[0], hands[0], orig[0, :14])] * 50
    token_fps, interp = float(e.get("token_fps", fps)), bool(e.get("interp", False))
    chunked = bool(e.get("chunked", False))
    resampler, next_chunk_t = ChunkResampler(fps), 0.0
    n_src = len(orig)
    n_ticks = int(np.floor((n_src - 1) / fps * 50)) + 1
    for j in range(n_ticks):
        k = min(int(np.floor(j * fps / 50 + 1e-9)), n_src - 1)  # source frame: original + hands held
        if chunked:  # VLA-like stream: 40-token chunk every 0.4 s from a 0.1 s old observation
            t = j / 50
            if t + 1e-9 >= next_chunk_t:
                s0 = min(int(np.floor(max(next_chunk_t - 0.1, 0.0) * fps + 1e-9)), len(tok) - 1)
                resampler.set_chunk(tok[s0 : s0 + 40], t0=s0 / fps)
                next_chunk_t += 0.4
            tok_j = resampler.token_at(t)
        elif token_fps == 50:  # tokens encoded at 50 Hz: one per tick
            tok_j = tok[min(j, len(tok) - 1)]
        elif interp:  # 30 Hz tokens linearly interpolated at the 50 Hz tick time
            xk = j * fps / 50
            k0 = min(int(np.floor(xk + 1e-9)), len(tok) - 1)
            k1, w = min(k0 + 1, len(tok) - 1), xk - k0
            tok_j = (1 - w) * tok[k0] + w * tok[k1]
        else:
            tok_j = tok[k]
        schedule.append(("episode", e_i, k, tok_j, hands[k], orig[k, :14]))
    last = schedule[-1]
    for a in np.linspace(0, 1, 51)[1:]:
        schedule.append(("blend out", e_i, -1, (1 - a) * last[3] + a * init_tok, (1 - a) * last[4], None))
    schedule += [("hold initial", -1, -1, init_tok, zero_h, None)] * 100
(OUT / "episodes.json").write_text(
    json.dumps(
        [
            {
                "name": e["name"],
                "task": str(e["task"]),
                "fps": float(e["fps"]),
                "frames": int(len(e["tokens"])),
            }
            for e in eps
        ],
        indent=1,
    )
)

# ---- renderer thread on a snapshot, with ghost spheres at intended palm points
snap = mujoco.MjData(m)
lock = threading.Lock()
cam = mujoco.MjvCamera()
cam.type, cam.trackbodyid, cam.distance, cam.azimuth, cam.elevation = (
    mujoco.mjtCamera.mjCAMERA_TRACKING,
    pelvis,
    2.4,
    150,
    -15,
)
writer = cv2.VideoWriter(str(OUT / "replay_raw.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
view = {"phase": "sim start: hanging on elastic band", "ghost": None, "err": None, "label": "", "stop": False}


def render_loop():
    renderer = mujoco.Renderer(m, H, W)
    next_t = 0.0
    while not view["stop"]:
        with lock:
            t = sim_t[0]
            ready = t >= next_t
            if ready:
                snap.qpos[:], snap.qvel[:] = d.qpos, d.qvel
                ghost, err, phase, label = view["ghost"], view["err"], view["phase"], view["label"]
        if not ready:
            time.sleep(0.002)
            continue
        next_t += 1 / FPS
        mujoco.mj_forward(m, snap)
        renderer.update_scene(snap, camera=cam)
        if ghost is not None:
            for p, rgba in zip(ghost, ([0.16, 0.47, 0.84, 0.55], [0.92, 0.41, 0.2, 0.55]), strict=False):
                sc = renderer.scene
                if sc.ngeom < sc.maxgeom:
                    mujoco.mjv_initGeom(
                        sc.geoms[sc.ngeom],
                        mujoco.mjtGeom.mjGEOM_SPHERE,
                        np.array([0.035, 0, 0]),
                        p,
                        np.eye(3).ravel(),
                        np.array(rgba, np.float32),
                    )
                    sc.ngeom += 1
        img = cv2.cvtColor(renderer.render(), cv2.COLOR_RGB2BGR)
        txt = [f"t = {t:6.2f} s   {phase}", label]
        if err is not None:
            txt.append(
                f"palm error vs original: left {err[0] * 100:4.1f} cm   right {err[1] * 100:4.1f} cm   (spheres = original target)"
            )
        for i, s in enumerate(t_ for t_ in txt if t_):
            cv2.putText(img, s, (12, 24 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(
                img, s, (12, 24 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA
            )
        writer.write(img)


threading.Thread(target=render_loop, daemon=True).start()

# ---- main loop
rec = {
    k: []
    for k in (
        "t",
        "phase",
        "episode",
        "frame",
        "body_q",
        "hand_q",
        "hand_cmd",
        "pelvis",
        "palm",
        "wrist_R",
        "palm_int",
        "wrist_R_int",
        "orig_arms",
        "floor_contacts",
    )
}
t_init, sched_i, next_tick, done, planner_tok = None, None, 0.0, False, None
mark("sim + deploy started")
wall0 = time.monotonic()
try:
    while proc.poll() is None and not done:
        with lock:
            env.sim_step()
            sim_t[0] += SIM_DT
        t = sim_t[0]
        if t_init is None:
            if any("Init Done" in s for _, s in lines):
                t_init = t
                mark("deploy printed 'Init Done'")
                view["phase"] = "Init Done"
            elif t > 600:
                mark("timeout waiting for Init Done")
                break
        else:
            dt = t - t_init
            if dt >= 2 and "start" not in view:
                view["start"] = 1
                command(True, True, "command: start control, PLANNER mode")
                view["phase"] = "planner mode (idle stand), hung"
            if dt >= 6 and "band" not in view:
                view["band"] = 1
                sim.handle_keyboard_button("9")
                mark("sim key '9': band released")
                view["phase"] = "band released"
            if dt >= 7 and "reset" not in view:
                view["reset"] = 1
                sim.handle_keyboard_button("backspace")
                mark("sim key 'Backspace': reset onto ground")
                view["phase"] = "settling on ground (planner idle)"
            if HANDOFF_S > 0 and dt >= 11 and "latent" not in view:
                latest = deploy_token()  # keep the newest token until the switch
                planner_tok = latest if latest is not None else planner_tok
            if dt >= 12 and "latent" not in view:
                view["latent"] = 1
                if HANDOFF_S > 0:
                    if planner_tok is None:
                        raise RuntimeError("no token_state from the deploy for the gradual handoff")
                    send_token(planner_tok, zero_h, frame=0)
                    n = int(HANDOFF_S * 50)
                    schedule[:0] = [("handoff blend", -1, -1, (1 - a) * planner_tok + a * init_tok, zero_h, None)
                                    for a in np.linspace(0, 1, n + 1)[1:]]  # fmt: skip
                    mark(f"planner token sent; {HANDOFF_S:g} s blend to the standing token "
                         f"(token distance {float(np.abs(planner_tok - init_tok).max()):.3f})")  # fmt: skip
                else:
                    send_token(init_tok, zero_h, frame=0)
                    mark("latent initial token sent")
            if dt >= 13 and "pose" not in view:
                view["pose"] = 1
                command(True, False, "command: POSE mode (streamed tokens)")
                sched_i = 0
                next_tick = t
            if sched_i is not None and t + 1e-9 >= next_tick:
                next_tick += TICK
                if sched_i >= len(schedule):
                    done = True
                    continue
                phase, e_i, k, tok, hands, orig = schedule[sched_i]
                sched_i += 1
                send_token(tok, hands)
                with lock:
                    palm, wrot = palms(d)
                    if orig is not None:
                        palm_i, wrot_i = intended(orig)
                        err = np.linalg.norm(palm - palm_i, axis=1)
                    else:
                        palm_i, wrot_i, err = np.full((2, 3), np.nan), np.full((2, 3, 3), np.nan), None
                    view["ghost"] = None if orig is None else palm_i
                    view["err"] = err
                    view["phase"] = phase
                    view["label"] = "" if e_i < 0 else f"{eps[e_i]['name']} - {eps[e_i]['task']} - frame {k}"
                    ncon = sum(1 for i in range(d.ncon) if floor in (d.contact[i].geom1, d.contact[i].geom2))
                    for key, val in (
                        ("t", t),
                        ("phase", phase),
                        ("episode", e_i),
                        ("frame", k),
                        ("body_q", d.qpos[qadr].copy()),
                        ("hand_q", d.qpos[hadr].copy()),
                        ("hand_cmd", np.asarray(hands, np.float32)),
                        ("pelvis", d.qpos[:7].copy()),
                        ("palm", palm),
                        ("wrist_R", wrot),
                        ("palm_int", palm_i),
                        ("wrist_R_int", wrot_i),
                        ("orig_arms", np.full(14, np.nan) if orig is None else orig),
                        ("floor_contacts", ncon),
                    ):
                        rec[key].append(val)
        lag = sim_t[0] - (time.monotonic() - wall0)
        if lag > 0:
            time.sleep(lag)
finally:
    mark("stop" if proc.poll() is None else f"deploy exited with code {proc.returncode}")
    try:
        command(False, False, "command: stop")
        os.write(master, b"o")
    except OSError:
        pass
    time.sleep(0.5)
    proc.terminate()
    view["stop"] = True
    time.sleep(0.3)
    writer.release()
    if rec["t"]:
        np.savez_compressed(OUT / "replay.npz", **{k: np.array(v) for k, v in rec.items()})
    (OUT / "events.json").write_text(json.dumps(events, indent=1))
    print("ticks", len(rec["t"]), "realtime lag s", round(time.monotonic() - wall0 - sim_t[0], 2), flush=True)
