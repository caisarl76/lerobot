"""Policy server for sonic_policy_streamer.py --policy-server that runs an official Isaac-GR00T checkpoint (GR00T N1.7,
embodiment UNITREE_G1_SONIC: 46D state, 40 x 78 chunks at 50 Hz) behind our protocol, so the robot side (planner
start and hand-back, chunk blend, token slew limit, watchdogs, eval sessions) stays as for our LeRobot models.

Protocol: as sonic_policy_server.py, plus "state_dim": 46 in the info reply. The streamer then sends
stream_backends.sonic46_state (left/right leg, waist, left/right arm, left/right hand, projected gravity: the order of
the unitree_g1_sonic modality config) instead of the 28D state. One observation step, no RTC.
The model resizes and crops the 640 x 480 head-camera frame itself (shortest edge 256, crop 0.95).

Usage (Isaac-GR00T 2a08eec environment, e.g. /mnt/data/jihun/venvs/isaac-gr00t-2a08eec, with the repo on PYTHONPATH):
  python gr00t_official_server.py --model-path /mnt/data/jihun/g1_models/wbt_groot_official46_handover261020_base_20k
"""

from __future__ import annotations

import argparse
import time

import cv2
import msgpack
import numpy as np
import torch
import zmq
from gr00t.model.gr00t_n1d7 import processing_gr00t_n1d7
from gr00t.policy.gr00t_policy import Gr00tPolicy
from huggingface_hub import snapshot_download

_build_processor = processing_gr00t_n1d7.build_processor


def build_local_processor(model_name, loading_kwargs):
    """transformers 4.57.3 asks the Hub about a tokenizer named by repo id (is_base_mistral) even offline; a cached
    snapshot path does not (the same fix as the official training runs' launcher)."""
    try:
        model_name = snapshot_download(model_name, local_files_only=True)
    except Exception:  # not cached: let the original call download it
        pass
    return _build_processor(model_name, loading_kwargs)


processing_gr00t_n1d7.build_processor = build_local_processor

IMAGE_KEY = "observation.images.egocentric"  # the PC2 head camera stream (g1_head_camera_server.py --name egocentric)
IMAGE_SHAPE = (3, 480, 640)  # (C, H, W) as recorded in the training videos
STATE_DIMS = {
    "left_leg": 6,
    "right_leg": 6,
    "waist": 3,
    "left_arm": 7,
    "right_arm": 7,
    "left_hand": 7,
    "right_hand": 7,
    "projected_gravity": 3,
}
ACTION_KEYS = ("motion_token", "left_hand_joints", "right_hand_joints")  # 64 + 7 + 7 = our 78D chunk columns
TASK_KEY = "annotation.human.task_description"


def observation(state: np.ndarray, image: np.ndarray, task: str) -> dict:
    """Gr00tPolicy observation (batch 1, one time step) from a 46D sonic46_state and an RGB H x W x 3 frame."""
    state = np.asarray(state, np.float32)
    if state.shape != (sum(STATE_DIMS.values()),):
        raise ValueError(f"expected a {sum(STATE_DIMS.values())}D state, got {state.shape}")
    parts = np.split(state, np.cumsum(list(STATE_DIMS.values()))[:-1])
    return {
        "video": {"ego_view": image[None, None]},
        "state": {k: v[None, None] for k, v in zip(STATE_DIMS, parts, strict=True)},
        "language": {TASK_KEY: [[task]]},
    }


class OfficialPolicy:
    def __init__(self, model_path: str, device: str, noise_scale: float | None):
        self.policy = Gr00tPolicy("UNITREE_G1_SONIC", model_path, device=device, strict=True)
        # noise_scale scales the flow sampler's initial noise (gr00t_n1d7 draws it with torch.randn); 0 gives the
        # deterministic mean-path chunk, our reference GR00T setting (sonic_policy_server.py --noise-scale)
        self.noise_scale = noise_scale

    def chunk(self, state: np.ndarray, image: np.ndarray, task: str) -> np.ndarray:
        randn = torch.randn
        if self.noise_scale is not None:

            def scaled_randn(*args, **kwargs):
                return randn(*args, **kwargs) * self.noise_scale

            torch.randn = scaled_randn
        try:
            action, _ = self.policy.get_action(observation(state, image, task))
        finally:
            torch.randn = randn
        return np.concatenate([np.asarray(action[k][0], np.float32) for k in ACTION_KEYS], axis=-1)


def decode(jpeg: bytes) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)  # RGB in, RGB out (no conversion)
    _, h, w = IMAGE_SHAPE
    return img if img.shape[:2] == (h, w) else cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-path", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--port", type=int, default=5562)
    p.add_argument(
        "--noise-scale",
        type=float,
        help="scale the sampler's initial noise: 0 = deterministic chunks (our GR00T reference). Default: unscaled",
    )
    a = p.parse_args()

    policy = OfficialPolicy(a.model_path, a.device, a.noise_scale)
    warm = np.zeros(46, np.float32)
    warm[-1] = -1.0  # standing upright: gravity along -z of the pelvis
    for _ in range(3):  # warm up on this (the serving) thread
        policy.chunk(warm, np.zeros(IMAGE_SHAPE[1:] + (3,), np.uint8), "warm up")
    sock = zmq.Context().socket(zmq.REP)
    sock.bind(f"tcp://*:{a.port}")
    print(f"[server] {a.model_path} ready on port {a.port}: UNITREE_G1_SONIC, 46D state", flush=True)
    served = 0
    while True:
        req = msgpack.unpackb(sock.recv(), raw=False)
        try:
            if req["op"] == "info":
                rep = {
                    "image_keys": [IMAGE_KEY],
                    "n_obs": 1,
                    "shapes": {IMAGE_KEY: list(IMAGE_SHAPE)},
                    "model": a.model_path,
                    "state_dim": 46,
                }
            else:
                if req.get("rtc"):
                    raise ValueError("the official GR00T server has no RTC (run the streamer without --rtc)")
                started = time.monotonic()
                (state,), (images,) = req["states"], req["images"]
                chunk = policy.chunk(np.asarray(state, np.float32), decode(images[IMAGE_KEY]), req["task"])
                rep = {
                    "chunk": np.ascontiguousarray(chunk).tobytes(),
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
