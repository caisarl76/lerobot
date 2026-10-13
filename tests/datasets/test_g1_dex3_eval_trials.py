"""sonic_policy_streamer.py --trials: camera recording and a two-trial session against a fake rig."""

import base64
import json
import socket
import sys
import time
from pathlib import Path

import numpy as np
import pytest

zmq = pytest.importorskip("zmq")
cv2 = pytest.importorskip("cv2")
pytest.importorskip("msgpack")

HERE = Path(__file__).parents[2] / "examples" / "g1_dex3_training"
sys.path.insert(0, str(HERE))
from sonic_policy_streamer import LiveImages


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
