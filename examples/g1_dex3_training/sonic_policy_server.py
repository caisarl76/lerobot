"""Policy server for sonic_policy_streamer.py --policy-server: runs a LeRobot SONIC-token policy on a GPU host (H100)
and answers one request per replan (every ~0.4 s) with a 30 Hz chunk. The robot-side streamer keeps all real-time
work (50 Hz tokens, table startup/shutdown, planner hand-back, safety stops) and opens the connection, which works
across a VPN that only allows robot -> H100 (decided 2026-09-27).

Protocol (ZMQ REQ/REP, msgpack):
  {"op": "info"}  -> {"image_keys": [...], "n_obs": int, "shapes": {key: [C, H, W]}}
  {"op": "chunk", "states": [[28 floats]] * n_obs, "images": [{key: jpeg bytes}] * n_obs, "task": str}
                  -> {"chunk": float32 bytes, "shape": [T, 78], "latency_s": float}
JPEGs carry RGB arrays encoded and decoded without colour conversion (RGB in, RGB out).

Usage (H100, lerobot container with the GPU and the port published, e.g. docker run -p 5560:5560 ...):
  python sonic_policy_server.py --policy-path /run-output/runs/groot_sonic78nolimit_ho5_full/checkpoints/last/pretrained_model
"""

from __future__ import annotations

import argparse
import time

import cv2
import msgpack
import numpy as np
import zmq
from sonic_policy_streamer import ChunkPolicy


def decode(jpeg: bytes, shape: tuple) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    _, h, w = shape
    return img if img.shape[:2] == (h, w) else cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--policy-path", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--port", type=int, default=5560)
    a = p.parse_args()

    policy = ChunkPolicy(a.policy_path, a.device)
    blank = [{k: np.zeros(policy.shapes[k][1:] + (3,), np.uint8) for k in policy.image_keys}] * policy.n_obs
    for _ in range(3):  # warm up on this (the serving) thread: VLA first calls take 2-5 s
        policy.chunk([np.zeros(28, np.float32)] * policy.n_obs, blank, "warm up")
    sock = zmq.Context().socket(zmq.REP)
    sock.bind(f"tcp://*:{a.port}")
    print(
        f"[server] {a.policy_path} ready on port {a.port}: {policy.image_keys}, n_obs {policy.n_obs}",
        flush=True,
    )
    served = 0
    while True:
        req = msgpack.unpackb(sock.recv(), raw=False)
        try:
            if req["op"] == "info":
                rep = {"image_keys": policy.image_keys, "n_obs": policy.n_obs, "shapes": policy.shapes}
            else:
                started = time.monotonic()
                images = [{k: decode(v, policy.shapes[k]) for k, v in im.items()} for im in req["images"]]
                states = [np.asarray(s, np.float32) for s in req["states"]]
                chunk = np.ascontiguousarray(policy.chunk(states, images, req["task"]), np.float32)
                rep = {
                    "chunk": chunk.tobytes(),
                    "shape": list(chunk.shape),
                    "latency_s": time.monotonic() - started,
                }
                served += 1
                if served % 50 == 0:
                    print(f"[server] {served} chunks, last {rep['latency_s']:.2f} s", flush=True)
        except Exception as exc:  # answer every request, or the client's REQ socket stalls
            rep = {"error": repr(exc)}
        sock.send(msgpack.packb(rep, use_bin_type=True))


if __name__ == "__main__":
    main()
