"""sonic_policy_streamer.py --trials: camera recording and a two-trial session against a fake rig."""

import base64
import json
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import msgpack
import numpy as np
import pytest

zmq = pytest.importorskip("zmq")
cv2 = pytest.importorskip("cv2")
pytest.importorskip("msgpack")

HERE = Path(__file__).parents[2] / "examples" / "g1_dex3_training"
sys.path.insert(0, str(HERE))
import eval_session  # noqa: E402
from sonic_policy_streamer import LiveImages  # noqa: E402


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def jpeg(value: int) -> bytes:
    return cv2.imencode(".jpg", np.full((48, 64, 3), value, np.uint8))[1].tobytes()


def camera_message(value: int) -> str:
    return json.dumps({"images": {"egocentric": base64.b64encode(jpeg(value)).decode("ascii")}})


def test_live_images_records_received_jpegs():
    port, ctx = free_port(), zmq.Context()
    pub = ctx.socket(zmq.PUB)
    pub.bind(f"tcp://127.0.0.1:{port}")
    key = "observation.images.egocentric"
    images = LiveImages(ctx, "127.0.0.1", port, [key], {key: (3, 48, 64)}, "head_camera", "t")
    try:
        deadline = time.monotonic() + 10
        while not images.latest and time.monotonic() < deadline:  # PUB/SUB slow joiner
            pub.send_string(camera_message(10))
            time.sleep(0.05)
        assert images.stop_recording() == []  # nothing recorded before start_recording()
        images.start_recording()
        for v in (20, 30, 40):
            pub.send_string(camera_message(v))
            time.sleep(0.1)
        rec = images.stop_recording()
        pub.send_string(camera_message(50))
        time.sleep(0.1)
        assert images.stop_recording() == []
    finally:
        images.close()
        pub.close(0)
        ctx.term()
    assert [name for _, name, _ in rec] == ["egocentric"] * 3
    assert [b for _, _, b in rec] == [jpeg(20), jpeg(30), jpeg(40)]  # the bytes as sent, not re-encoded
    assert all(abs(t - time.time()) < 5 for t, _, _ in rec)


def serve_policy(ctx, port, stop):
    rep = ctx.socket(zmq.REP)
    rep.bind(f"tcp://127.0.0.1:{port}")
    key = "observation.images.egocentric"
    while not stop.is_set():
        if not rep.poll(50):
            continue
        req = msgpack.unpackb(rep.recv(), raw=False)
        if req["op"] == "info":
            out = {
                "image_keys": [key],
                "n_obs": 1,
                "shapes": {key: [3, 48, 64]},
                "model": "/m/fake/pretrained_model",
            }
        else:
            out = {"chunk": np.zeros((40, 78), np.float32).tobytes(), "shape": [40, 78]}
        rep.send(msgpack.packb(out, use_bin_type=True))
    rep.close(0)


def publish(ctx, port, stop, period, make):
    pub = ctx.socket(zmq.PUB)
    pub.bind(f"tcp://127.0.0.1:{port}")
    i = 0
    while not stop.is_set():
        make(pub, i)
        i += 1
        time.sleep(period)
    pub.close(0)


def deploy_state(pub, i):
    msg = {
        "body_q": [0.0] * 29,
        "left_hand_q": [0.0] * 7,
        "right_hand_q": [0.0] * 7,
        "token_state": [0.0] * 64,
    }
    pub.send(b"g1_debug" + msgpack.packb(msg, use_bin_type=True))


def camera(pub, i):
    pub.send_string(camera_message(i % 250))


def wait_gone(path: Path, timeout=120):
    deadline = time.monotonic() + timeout
    while path.exists():
        assert time.monotonic() < deadline, f"{path} was never consumed"
        time.sleep(0.05)


def drive(gate: Path):
    """The operator: trial 0 redone once, then success at the duration cap; trial 1 stopped early and failed."""
    for name in ("deploy_ready", "settled.0"):
        (gate / name).touch()
    (gate / "verdict.0").write_text(json.dumps({"verdict": "redo"}))
    wait_gone(gate / "verdict.0")
    (gate / "settled.0").touch()
    (gate / "verdict.0").write_text(json.dumps({"verdict": "success"}))
    wait_gone(gate / "verdict.0")
    (gate / "settled.1").touch()
    wait_gone(gate / "settled.1")
    time.sleep(3.5)  # past the takeover and the 1 s blend-in: inside the episode
    (gate / "stop.1").touch()
    (gate / "verdict.1").write_text(json.dumps({"verdict": "fail", "reason": 2, "note": "x"}))


def test_two_trial_session_against_a_fake_rig(tmp_path):
    ports = {k: free_port() for k in ("policy", "state", "camera", "action")}
    ctx, stop = zmq.Context(), threading.Event()
    threads = [
        threading.Thread(target=serve_policy, args=(ctx, ports["policy"], stop)),
        threading.Thread(target=publish, args=(ctx, ports["state"], stop, 0.02, deploy_state)),
        threading.Thread(target=publish, args=(ctx, ports["camera"], stop, 1 / 30, camera)),
    ]
    gate, session = tmp_path / "gate", tmp_path / "session"
    gate.mkdir()
    threads.append(threading.Thread(target=drive, args=(gate,), daemon=True))
    for t in threads:
        t.start()
    try:
        r = subprocess.run(
            [sys.executable, str(HERE / "sonic_policy_streamer.py"),
             "--policy-server", f"tcp://127.0.0.1:{ports['policy']}", "--images", "zmq",
             "--camera-host", "127.0.0.1", "--camera-port", str(ports["camera"]),
             "--deploy-host", "127.0.0.1", "--state-port", str(ports["state"]), "--action-port", str(ports["action"]),
             "--task", "pick", "--duration-s", "6", "--start", "planner", "--end", "planner",
             "--handoff-blend-s", "0.2", "--trials", "2", "--session-dir", str(session), "--gate-dir", str(gate)],
            capture_output=True, text=True, timeout=240, stdin=subprocess.DEVNULL,
        )  # fmt: skip
    finally:
        stop.set()
        for t in threads[:3]:
            t.join()
        ctx.term()
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    lines = [json.loads(s) for s in (session / "results.jsonl").read_text().splitlines()]
    assert [(x["trial"], x["verdict"], x["episode_end"]) for x in lines] == [
        (0, "success", "completed"),
        (1, "fail", "operator stop"),
    ]
    assert [x["model"] for x in lines] == ["/m/fake/pretrained_model"] * 2  # asked from the server each trial
    assert lines[1]["reason"] == eval_session.DEFAULT_FAILURE_REASONS[1] and lines[1]["note"] == "x"
    assert 5.5 < lines[0]["wall_stop"] - lines[0]["wall_start"] < 6.1  # the duration cap
    assert 0.2 < lines[1]["wall_stop"] - lines[1]["wall_start"] < 5.0  # stopped by the operator
    t, names, jpegs = eval_session.load_frames(session / "trial_001_frames.npz")
    assert len(jpegs) > 30 and set(names) == {"egocentric"} and np.all(np.diff(t) > 0)
    assert cv2.imdecode(np.frombuffer(jpegs[0], np.uint8), cv2.IMREAD_COLOR).shape == (48, 64, 3)
    log = np.load(session / "trial_000.npz")
    assert str(log["verdict"]) == "success" and json.loads(str(log["args"]))["trials"] == 2
    assert "success 1/2" in r.stdout


# ---------------------------------------------------------------- faults during a trial: the robot is handed back
# The deploy holds the last streamed token in POSE mode (no stream timeout in g1_deploy_onnx_ref's zmq_manager), so the
# streamer must blend out and send PLANNER on every fault, including errors that end the process.
FAULT_AT_S = 3.5  # after settled.0 is consumed: past the takeover and the 1 s blend-in, inside the episode


def fault_policy(ctx, port, stop, fault):
    """serve_policy, but chunk requests go unanswered once fault["policy"] is set (or after n warm-up chunks)."""
    rep = ctx.socket(zmq.REP)
    rep.bind(f"tcp://127.0.0.1:{port}")
    key, chunks = "observation.images.egocentric", 0
    while not stop.is_set():
        if not rep.poll(50):
            continue
        req = msgpack.unpackb(rep.recv(), raw=False)
        if req["op"] != "info":
            chunks += 1
            if fault.get("policy") or chunks > fault.get("policy_after_chunks", 10**9):
                rep.close(0)  # a dead server: the REQ times out; rebind for its reconnect
                rep = ctx.socket(zmq.REP)
                rep.bind(f"tcp://127.0.0.1:{port}")
                continue
        out = (
            {
                "image_keys": [key],
                "n_obs": 1,
                "shapes": {key: [3, 48, 64]},
                "model": "/m/fake/pretrained_model",
            }
            if req["op"] == "info"
            else {"chunk": np.full((40, 78), 0.1, np.float32).tobytes(), "shape": [40, 78]}
        )
        rep.send(msgpack.packb(out, use_bin_type=True))
    rep.close(0)


def listen_commands(ctx, port, stop, out, echo):
    """The deploy's side of the action port: (start, stop, planner) of every command message, in order, and the
    latest pose token in echo["token"] (the deploy copies it into g1_debug token_state)."""
    sub = ctx.socket(zmq.SUB)
    sub.setsockopt(zmq.SUBSCRIBE, b"command")
    sub.setsockopt(zmq.SUBSCRIBE, b"pose")
    sub.connect(f"tcp://127.0.0.1:{port}")
    while not stop.is_set():
        if sub.poll(50):
            msg = sub.recv()
            if msg.startswith(b"pose"):
                start = len(b"pose") + 1280
                echo["token"] = np.frombuffer(msg[start : start + 256], "<f4").tolist()
            else:
                out.append(tuple(msg[-3:]))
    sub.close(0)


def run_faulted_trial(
    tmp_path, fault_name, sigint=False, policy_after_chunks=None, duration_s=20, close_stdout=False
):
    ports = {k: free_port() for k in ("policy", "state", "camera", "action")}
    ctx, stop, fault, commands, echo = zmq.Context(), threading.Event(), {}, [], {"token": [0.0] * 64}
    if policy_after_chunks is not None:
        fault["policy_after_chunks"] = policy_after_chunks

    def state(pub, i):
        if fault.get("state_undecodable"):
            pub.send(b"g1_debug\xc1")  # 0xc1: never valid msgpack
        elif fault.get("state_malformed"):
            pub.send(b"g1_debug" + msgpack.packb({"body_q": [0.0] * 28, "left_hand_q": [0.0] * 7}))
        elif fault.get("state_short_hand"):  # a 27D observation if accepted
            msg = {
                "body_q": [0.0] * 29,
                "left_hand_q": [0.0] * 6,
                "right_hand_q": [0.0] * 7,
                "token_state": echo["token"],
            }
            pub.send(b"g1_debug" + msgpack.packb(msg, use_bin_type=True))
        elif not fault.get("state_stopped"):
            # a deploy listening to another command source reports that source's tokens, not ours
            token = [0.7] * 64 if fault.get("deploy_elsewhere") else echo["token"]
            msg = {
                "body_q": [0.0] * 29,
                "left_hand_q": [0.0] * 7,
                "right_hand_q": [0.0] * 7,
                "token_state": token,
            }
            pub.send(b"g1_debug" + msgpack.packb(msg, use_bin_type=True))

    def cam(pub, i):
        if fault.get("camera_empty"):  # the image server's message while capture is stalled
            pub.send_string(json.dumps({"images": {}}))
        elif fault.get("camera_undecodable"):
            pub.send_string(json.dumps({"images": {"egocentric": base64.b64encode(b"not a jpeg").decode()}}))
        elif not fault.get("camera_frozen"):
            camera(pub, i)

    threads = [
        threading.Thread(target=fault_policy, args=(ctx, ports["policy"], stop, fault)),
        threading.Thread(target=publish, args=(ctx, ports["state"], stop, 0.02, state)),
        threading.Thread(target=publish, args=(ctx, ports["camera"], stop, 1 / 30, cam)),
        threading.Thread(target=listen_commands, args=(ctx, ports["action"], stop, commands, echo)),
    ]
    gate, session = tmp_path / "gate", tmp_path / "session"
    gate.mkdir()
    for t in threads:
        t.start()
    try:
        proc = subprocess.Popen(
            [sys.executable, str(HERE / "sonic_policy_streamer.py"),
             "--policy-server", f"tcp://127.0.0.1:{ports['policy']}", "--images", "zmq",
             "--camera-host", "127.0.0.1", "--camera-port", str(ports["camera"]),
             "--deploy-host", "127.0.0.1", "--state-port", str(ports["state"]), "--action-port", str(ports["action"]),
             "--task", "pick", "--duration-s", str(duration_s), "--start", "planner", "--end", "planner",
             "--handoff-blend-s", "0.2", "--trials", "1", "--session-dir", str(session), "--gate-dir", str(gate)],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, stdin=subprocess.DEVNULL,
        )  # fmt: skip
        lines = []
        reader = threading.Thread(target=lambda: lines.extend(proc.stdout), daemon=True)
        if not close_stdout:  # else nothing reads the pipe, so closing it races no reader
            reader.start()
        for name in ("deploy_ready", "settled.0"):
            (gate / name).touch()
        wait_gone(gate / "settled.0")
        time.sleep(FAULT_AT_S)
        if sigint:
            proc.send_signal(signal.SIGINT)
        elif close_stdout:  # the log consumer (tee) goes away: every later print raises BrokenPipeError
            proc.stdout.close()
        elif fault_name:
            fault[fault_name] = True
        deadline = time.monotonic() + 60
        while proc.poll() is None and not any("waiting for" in s for s in lines):
            assert time.monotonic() < deadline, "the trial never ended"
            time.sleep(0.05)
        if proc.poll() is None:  # the trial ended normally: the operator scores it
            (gate / "verdict.0").write_text(json.dumps({"verdict": "fail", "reason": 1}))
        proc.wait(timeout=60)
        if reader.is_alive() or not close_stdout:
            reader.join(timeout=10)
        time.sleep(0.2)  # the last command in flight
    finally:
        if proc.poll() is None:
            proc.kill()
        stop.set()
        for t in threads:
            t.join()
        ctx.term()
    out = "".join(lines)
    ended = next((s.split("ended: ", 1)[1].strip() for s in lines if "[eval] trial 1/1 ended: " in s), None)
    return proc.returncode, out, commands, ended


START_PLANNER, POSE = (1, 0, 1), (1, 0, 0)


@pytest.mark.parametrize(
    "fault_name, episode_end",
    [
        ("policy", "no valid chunk for 1 s"),
        ("state_stopped", "robot state"),
        ("state_undecodable", "robot state"),
        ("state_malformed", "robot state"),
        ("camera_frozen", "camera frame"),
        ("camera_empty", "camera frame"),
        ("camera_undecodable", "camera frame"),
        ("state_short_hand", "robot state"),
        ("deploy_elsewhere", "deploy token_state"),
    ],
)
def test_a_fault_inside_the_episode_ends_the_trial_and_hands_back(tmp_path, fault_name, episode_end):
    code, out, commands, ended = run_faulted_trial(tmp_path, fault_name)
    assert code == 0, out[-3000:]
    assert ended is not None and episode_end in ended, (ended, out[-3000:])
    assert commands == [START_PLANNER, POSE, START_PLANNER], out[-3000:]


def test_a_policy_dead_at_the_first_chunk_still_hands_back(tmp_path):
    # the 3 warm-up chunks are answered, the first episode chunk is not
    code, out, commands, _ = run_faulted_trial(tmp_path, None, policy_after_chunks=3)
    assert code != 0 and "did not answer" in out, out[-3000:]
    assert commands == [START_PLANNER, POSE, START_PLANNER], out[-3000:]


def test_ctrl_c_inside_the_episode_hands_back_before_exiting(tmp_path):
    code, out, commands, _ = run_faulted_trial(tmp_path, None, sigint=True)
    assert code != 0 and "KeyboardInterrupt" in out, out[-3000:]
    assert commands == [START_PLANNER, POSE, START_PLANNER], out[-3000:]


def test_a_deploy_that_echoes_the_streamed_tokens_runs_to_the_cap(tmp_path):
    code, out, commands, ended = run_faulted_trial(tmp_path, None, duration_s=8)
    assert code == 0 and ended == "completed", (ended, out[-3000:])
    assert commands == [START_PLANNER, POSE, START_PLANNER], out[-3000:]


def test_a_closed_log_pipe_does_not_prevent_the_handback(tmp_path):
    code, _, commands, _ = run_faulted_trial(tmp_path, None, duration_s=8, close_stdout=True)
    assert code != 0
    assert commands == [START_PLANNER, POSE, START_PLANNER]
