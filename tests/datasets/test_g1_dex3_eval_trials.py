"""sonic_policy_streamer.py --trials: camera recording and a two-trial session against a fake rig."""

import base64
import json
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
