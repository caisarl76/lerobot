"""Sim host for sonic_policy_streamer.py --backend sonic: gear_sonic MuJoCo + NVIDIA g1_deploy_onnx_ref, recorded.

The streamer owns every deploy command (ZMQ "command"/"pose" on 5556). This process runs the physics (sim_scene.Scene),
performs the confirmed operator steps that belong to the simulator, and exchanges gate files with the streamer:
  deploy prints "Init Done"               -> write GATE/deploy_ready (streamer sends start + planner)
  deploy enters CONTROL (+4 s, still hung) -> sim key "9" (release band), +1 s sim key "Backspace" (reset)
  robot settled for 5 s                   -> write GATE/settled (streamer hands off to POSE and streams); fall
                                             detection armed
  streamer writes GATE/done               -> stop 5 s later; termination.json says completed only if the streamer's
                                             episode completed
With TRIALS=n (SIM=isaac, streamer in --trials n --gate-dir mode) the first settle starts judged trials instead
(sim_trials.run_trials, tasks/handover_bottle.json): from trial next_trial(SESSION_DIR) up to n, and the stack ends on
a redo. One stack per trial (TRIALS=i+1, a fresh GATE_DIR each) is the evaluation mode: the deploy's planner keeps its
heading across a reset (test 4).
Logs robot state every 20 ms with wall-clock time, the Dex3 gains the deploy sends (events.txt), and renders a video.
Usage (in jihun/sonic-vla-sim, cwd gear_sonic_deploy): python sonic_official_sim_host.py OUT_DIR GATE_DIR
"""

import contextlib
import json
import os
import pty
import subprocess
import sys
import threading
import time
from pathlib import Path

SIM = os.environ.get("SIM", "mujoco")
if SIM == "isaac":
    from isaac_scene import GEAR_ROOT, Scene

    sys.argv[1:3] = [str(Path(a).resolve()) for a in sys.argv[1:3]]  # OUT, GATE before the chdir
    os.chdir(GEAR_ROOT / "gear_sonic_deploy")  # the deploy paths below are relative to gear_sonic_deploy
elif SIM == "mujoco":
    from sim_scene import Scene
else:
    raise SystemExit(f"SIM must be mujoco or isaac, not {SIM!r}")
from wbc_common import lag_stats, read_done  # noqa: E402 - after the SIM switch (chdir)

OUT, GATE = Path(sys.argv[1]), Path(sys.argv[2])
# TRIALS > 0: judged trials (Isaac scene), streamer in --trials --gate-dir mode
TRIALS = int(os.environ.get("TRIALS", "0"))
SESSION_DIR = Path(os.environ["SESSION_DIR"]) if TRIALS else None
SEED = int(os.environ.get("SEED", "0"))
if TRIALS and SIM != "isaac":
    raise SystemExit("TRIALS needs SIM=isaac (sim_scene has no reset_trial)")
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
    try:
        br = scene.bridge
        cmds = [c for cmd in (br.left_hand_cmd, br.right_hand_cmd) for c in cmd.motor_cmd]
        kp, kd = sorted({round(float(c.kp), 3) for c in cmds}), sorted({round(float(c.kd), 3) for c in cmds})
        scene.mark(f"hand gains {when}: kp {kp} kd {kd}")
    except Exception as e:  # a diagnostic must never stop the run
        scene.mark(f"hand gains {when}: unavailable ({e!r})")


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
            scene.key("9")
            scene.mark("sim key '9': band released")
            scene.view["phase"] = "band released"
        if "band" in steps and "reset" not in steps and t >= steps["band"] + 1:
            steps["reset"] = t
            scene.key("backspace")
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
        if TRIALS and t_settled is not None:
            from sim_trials import run_trials

            task = json.loads((Path(__file__).parent / "tasks" / "handover_bottle.json").read_text())
            reason, detail = run_trials(
                scene,
                GATE,
                SESSION_DIR,
                TRIALS,
                SEED,
                task,
                wall0,
                lambda: proc.poll() is None,
                one_attempt=True,
            )
            break
        if t_done is None and (GATE / "done").exists():
            t_done, episode_end = t, read_done(GATE)
            scene.mark(f"streamer done: {episode_end}")
            scene.view["phase"] = "streamer done"
        if t_done is not None and t >= t_done + 5:  # keep 5 s of the planner hand-back on record
            reason, detail = (
                ("completed", "") if episode_end == "completed" else ("aborted", f"streamer: {episode_end}")
            )
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
    lag = getattr(scene, "lag_ms", None)
    if lag:
        stats = lag_stats(lag)
        scene.mark(f"lag vs wall clock: {stats}")
        (OUT / "lag.json").write_text(json.dumps(stats))
    try:
        scene.close(reason, detail)
    finally:
        if SIM == "isaac":  # Kit can hang at interpreter exit, also after an error in close
            os._exit(1 if reason in ("error", "timeout") else 0)
