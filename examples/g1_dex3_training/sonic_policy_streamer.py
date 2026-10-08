"""Stream a LeRobot SONIC-token policy (78D = token 64 + Dex3 hands 14, 30 Hz) to NVIDIA's g1_deploy_onnx_ref.
--action-space joint28 instead takes a 28D arm + Dex3 hand joint policy and encodes each chunk to tokens here
(sonic_targets.joint_chunk_to_sonic, the official encoder on CPU, speed limits off as in sonic78_nolimit).

Runtime (decided 2026-09-24): the C++ deploy decodes at 50 Hz; this process is its token source, like
gear_sonic/scripts/run_vla_inference.py with initial_pose="standing":
  robot state  <- ZMQ SUB "g1_debug" (port 5557, msgpack; body_q in motor order, Dex3 hand q)
  commands     -> ZMQ PUB "command" (start / stop / planner) on port 5556
  tokens+hands -> ZMQ PUB "pose" protocol v4 on port 5556, every 20 ms
The policy predicts a 30 Hz chunk every --replan-s from the latest observation; each 50 Hz tick sends
ChunkResampler.token_at(t) (look-ahead interpolation, verified in the deploy). Episode start: POSE mode with
the planner's current token -> --handoff-blend-s blend to the latent initial (standing) token -> 1 s blend to the
first predicted token. End: 1 s blend back, 2 s hold, stop. At a table (--startup-tokens, sonic_table_startup.py):
the handoff blends into a table-safe arm path instead of the standing token (whose hands rise into an 80 cm table);
episodes start from the path's final pose and end by playing the path in reverse back to the arms-down stance.
Right Dex3 hand in dataset order (thumb, index, middle), which the real robot follows; --dex3-right-order swap
only for NVIDIA's MuJoCo bridge.

Images: --images dataset feeds the recorded episode's camera frames at the elapsed time (evaluation without a
sim/real camera gap); --images zmq reads the live head camera from LeRobot's ImageServer (real robot).
Policy: --policy-path runs it here; --policy-server asks sonic_policy_server.py on a GPU host (H100 over the VPN),
while this process keeps all real-time work, so a VPN hiccup only delays a chunk. Operator gates: --gate-dir waits for flag files (sim host), otherwise press Enter.

--backend decoupled sends 50 Hz joint targets (wbc_common.pack_joint_message) to decoupled_wbc_sim_host.py instead of
tokens; --replay streams recorded actions; every joint run logs a 50 Hz joint_ref (the target after resampling,
blending and lead-in), computed alongside A's unchanged token path. Each backend's startup, per-tick send and
shutdown live in stream_backends.py; this file keeps the CLI, the policy / replay input, inference and the watchdogs.
"""

from __future__ import annotations

import argparse
import json
import math
import queue
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING

import numpy as np

try:
    from lerobot.utils.import_utils import _zmq_available, require_package
except ImportError:  # robot PC without lerobot (and torch): --policy-server with --images zmq needs neither
    import importlib.util

    _zmq_available = importlib.util.find_spec("zmq") is not None

    def require_package(name: str, extra: str, import_name: str) -> None:
        if importlib.util.find_spec(import_name) is None:
            raise ImportError(f"{name} is required: pip install {name}")


if TYPE_CHECKING or _zmq_available:
    import zmq

try:
    from .sonic_targets import SonicEncoder, joint_chunk_to_sonic, load_joint_limits
    from .stream_backends import (
        RIGHT_ORDER,
        RIGHT_SWAP,
        TICK,
        DecoupledBackend,
        SonicBackend,
        load_startup,
        measured_ref as measured_ref,  # re-exported for the tests
        observation_state,
        ticks,
    )
    from .wbc_common import (
        check_replay_width,
        check_synthetic_duration,
        replay_chunk,
        synthetic_waist,
        to_joint_ref,
        write_done,
    )
except ImportError:
    from sonic_targets import SonicEncoder, joint_chunk_to_sonic, load_joint_limits
    from stream_backends import (
        RIGHT_ORDER,
        RIGHT_SWAP,
        TICK,
        DecoupledBackend,
        SonicBackend,
        load_startup,
        measured_ref as measured_ref,  # re-exported for the tests
        observation_state,
        ticks,
    )
    from wbc_common import (
        check_replay_width,
        check_synthetic_duration,
        replay_chunk,
        synthetic_waist,
        to_joint_ref,
        write_done,
    )

TOKEN_BOUND = 1.25  # run_vla_inference rejects chunks whose token magnitude exceeds this


# Arm joint range (motor order 15..28, rad) of observation.state over all Unitree + Humanoid Everyday joint28
# episodes; the watchdog ends an episode that leaves it by more than --arm-range-margin.
ARM_LO = np.array(
    [
        -2.596,
        -0.33,
        -1.53,
        -1.046,
        -1.861,
        -1.612,
        -1.61,
        -2.609,
        -1.629,
        -1.627,
        -1.043,
        -1.949,
        -1.618,
        -1.611,
    ]
)
ARM_HI = np.array(
    [1.473, 1.637, 1.676, 1.526, 1.97, 1.618, 1.625, 1.586, 0.284, 1.579, 1.419, 1.599, 1.634, 1.616]
)
ARM_SPEED_TICKS = 3  # measured arm speed over 3 ticks (60 ms): robust to one late or repeated state message


class StateSubscriber:
    def __init__(self, ctx: zmq.Context, host: str, port: int):
        self.sock = ctx.socket(zmq.SUB)
        self.sock.setsockopt_string(zmq.SUBSCRIBE, "g1_debug")
        self.sock.setsockopt(zmq.CONFLATE, 1)
        self.sock.connect(f"tcp://{host}:{port}")
        self.msg, self.t = None, None

    def latest(self) -> dict | None:
        import msgpack  # shipped with the deploy images, not a lerobot dependency

        try:
            raw = self.sock.recv(zmq.NOBLOCK)
            self.msg = msgpack.unpackb(raw[len("g1_debug") :], raw=False)
            self.t = time.monotonic()
        except zmq.Again:
            pass
        return self.msg

    def age(self) -> float:
        """Seconds since the last g1_debug message (inf before the first)."""
        return float("inf") if self.t is None else time.monotonic() - self.t


class DatasetImages:
    """Camera frames of one recorded episode, looked up by elapsed time (30 Hz, held at the end)."""

    def __init__(self, root: str, episode: int, keys: list[str]):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        self.ds = LeRobotDataset("local/sonic78", root=root, episodes=[episode], video_backend="torchcodec")
        self.keys, self.fps, self.n = keys, self.ds.fps, len(self.ds)

    def frame(self, t: float) -> tuple[int, dict, np.ndarray, str]:
        """Frame index, camera images (HWC uint8), recorded observation.state and task text at time t."""
        k = min(max(int(t * self.fps), 0), self.n - 1)
        item = self.ds[k]
        imgs = {key: (item[key].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8) for key in self.keys}
        return k, imgs, item["observation.state"].numpy().astype(np.float32), str(item.get("task", ""))

    def actions(self) -> np.ndarray:
        """The episode's recorded action column [n, D] (for --replay)."""
        out = np.stack([np.asarray(v, np.float32) for v in self.ds.hf_dataset["action"]])
        if len(out) != self.n:
            raise ValueError(f"episode has {self.n} frames but {len(out)} actions")
        return out


def rtc_prefix(last_raw, offset: int | None):
    """The unplayed tail [1, T - offset, A] of the last normalized chunk, or None (no chunk yet, or all played)."""
    if last_raw is None or offset is None or offset >= last_raw.shape[1]:
        return None
    return last_raw[:, max(0, offset) :]


def rtc_args(
    enabled: bool, last_accepted: bool, t_ep: float, chunk_t0: float, latency_s: float, fps: float
) -> tuple:
    """(offset, delay) for the next --rtc request, or () for a plain one. The policy keeps its last returned chunk as
    the prefix source, so RTC is only valid when that chunk was the one accepted and now playing: after a rejected,
    failed or timed-out request the next request is plain."""
    if not (enabled and last_accepted):
        return ()
    return round((t_ep - chunk_t0) * fps), math.ceil(latency_s * fps)


def dataset_obs(ds, k: int, n: int, image_keys: list[str]) -> tuple[list[np.ndarray], list[dict], str]:
    """Last n dataset frames up to k (oldest first): states, images (HWC uint8), task. For offline diagnostics."""
    items = [ds[max(k - i, 0)] for i in reversed(range(n))]
    imgs = [
        {key: (it[key].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8) for key in image_keys}
        for it in items
    ]
    states = [it["observation.state"].numpy().astype(np.float32) for it in items]
    return states, imgs, str(items[-1].get("task", ""))


def cast_groot_backbone(policy, dtype) -> None:
    """Cast the frozen GR00T VLM backbone's parameters (not buffers) to dtype; the action head stays float32.

    Fits GR00T N1.7 on a 12 GB GPU (float32 checkpoint 12.6 GB -> 9.5 GB). Our fine-tunes freeze the backbone
    (tune_llm / tune_visual off), so its float32 values are the released bf16 weights and bf16 loses nothing
    (checked for groot_sonic78nolimit_ho5_full: all 1.52B backbone values bf16-exact). Activations in the
    backbone then run in dtype, so outputs differ slightly from the float32 model.
    """
    if policy.config.type != "groot":
        raise ValueError(f"--backbone-dtype only applies to GR00T, not {policy.config.type}")
    for param in policy._groot_model.backbone.parameters():
        if param.is_floating_point():
            param.data = param.data.to(dtype)


class ChunkPolicy:
    def __init__(
        self,
        path: str,
        device: str,
        backbone_dtype: str | None = None,
        noise_seed: int | None = None,
        noise_scale: float | None = None,
    ):
        import torch

        self.noise_seed = noise_seed
        # noise_scale ("temperature") scales the flow/diffusion sampler's initial noise; 0 gives the deterministic
        # mean-path sample (smoother and, open loop, more accurate chunks; less motion inside a chunk)
        self.noise_scale = noise_scale

        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import get_policy_class, make_pre_post_processors

        cfg = PreTrainedConfig.from_pretrained(path)
        # Load on the CPU first: a float32 checkpoint may not fit a 12 GB GPU before the backbone cast or
        # before a bf16 config (Pi0.5 `dtype`, VLA-JEPA `torch_dtype`) converts it.
        cfg.device = "cpu"
        policy = get_policy_class(cfg.type).from_pretrained(path, config=cfg)
        if backbone_dtype:
            cast_groot_backbone(policy, getattr(torch, backbone_dtype))
        cfg.device = device
        self.policy = policy.to(device).eval()
        if noise_scale is not None and cfg.type == "pi05":
            sample_noise = policy.model.sample_noise

            def scaled_noise(shape, device):
                return sample_noise(shape, device) * noise_scale

            policy.model.sample_noise = scaled_noise
        self.pre, self.post = make_pre_post_processors(
            policy_cfg=cfg,
            pretrained_path=path,
            preprocessor_overrides={"device_processor": {"device": device}},
        )

        self.device = torch.device(device)
        self.image_keys = [k for k in cfg.input_features if "image" in k]
        self.shapes = {k: tuple(cfg.input_features[k].shape) for k in self.image_keys}  # (C, H, W)
        self.n_obs = int(getattr(cfg, "n_obs_steps", 1) or 1)  # diffusion conditions on 2 frames

    def chunk(self, states: list[np.ndarray], images: list[dict], task: str, **rtc) -> np.ndarray:
        """states/images: the last n_obs observations, oldest first (1/fps apart); task: instruction text.

        rtc: optional predict_action_chunk kwargs (prev_chunk_left_over in the model's normalized space,
        inference_delay, execution_horizon). The normalized chunk is kept in self.last_raw for the next call.
        """
        import torch

        if self.noise_seed is not None:
            # same sampling noise for every chunk: a flow-matching policy (GR00T) then gives consistent chunks
            # for similar observations instead of a fresh random sample at every replan
            torch.manual_seed(self.noise_seed)
        # Pi0.5's guided RTC differentiates through the sampler, which inference_mode forbids; GR00T's RTC
        # inpaints without gradients and needs inference_mode's lower memory to fit a 12 GB GPU
        guided = rtc and self.policy.config.type == "pi05"
        randn = torch.randn
        if self.noise_scale is not None and self.policy.config.type != "pi05":
            # GR00T and Diffusion draw their initial noise with torch.randn; scale it for this call only
            def scaled_randn(*args, **kwargs):
                return randn(*args, **kwargs) * self.noise_scale

            torch.randn = scaled_randn
        try:
            with torch.no_grad() if guided else torch.inference_mode():
                return self._chunk(states, images, task, **rtc)
        finally:
            torch.randn = randn

    def _chunk(self, states, images, task, **rtc):
        import torch

        from lerobot.policies.utils import prepare_observation_for_inference

        frames = [
            prepare_observation_for_inference(
                {"observation.state": st, **im}, self.device, task=task, robot_type="unitree_g1"
            )
            for st, im in zip(states, images, strict=True)
        ]
        batch = frames[-1]
        if len(frames) > 1:  # stack the time axis: [1, n_obs, ...]
            for key, val in batch.items():
                if isinstance(val, torch.Tensor):
                    batch[key] = torch.cat([f[key].unsqueeze(1) for f in frames], dim=1)
        batch["task"] = [task]
        actions = self.policy.predict_action_chunk(self.pre(batch), **rtc)
        self.last_raw = actions  # normalized [1, T, 78], the RTC prefix source for the next chunk
        return self.post(actions).squeeze(0).float().cpu().numpy()  # [T, 78]

    def chunk_rtc(self, states, images, task: str, offset: int | None, delay: int) -> np.ndarray:
        """Closed-loop RTC: condition on the unplayed tail of the last chunk. offset = rows of that chunk already
        played at this observation, delay = rows that will play while this chunk is computed (kept frozen)."""
        prev = rtc_prefix(getattr(self, "last_raw", None), offset)
        if prev is None:
            return self.chunk(states, images, task)
        if self.policy.config.type == "pi05" and self.policy.config.rtc_config is None:
            # Pi0.5's guided RTC only runs with an RTC processor (as openloop_smooth.py enables it)
            from lerobot.policies.rtc.configuration_rtc import RTCConfig

            self.policy.config.rtc_config = RTCConfig()
            self.policy.init_rtc_processor()
        return self.chunk(
            states,
            images,
            task,
            prev_chunk_left_over=prev,
            inference_delay=delay,
            execution_horizon=prev.shape[1],
        )


class LiveImages:
    """Latest head-camera frames from LeRobot's ImageServer (robots/unitree_g1/run_g1_server.py --camera: ZMQ JSON,
    one base64 JPEG per camera), returned as the policy's image keys at their training size. A camera named like a
    key is used as is; otherwise --stereo-camera is split into left/right halves (the Unitree datasets'
    cam_left_high / cam_right_high). The server JPEG-encodes RGB arrays without conversion, so decoding without
    conversion gives RGB back. A receiver thread owns the socket; frame() only reads the latest images."""

    fps, n = 30, 10**9

    def __init__(self, ctx, host: str, port: int, keys: list[str], shapes: dict, stereo: str, task: str):
        self.keys, self.shapes, self.stereo, self.task = keys, shapes, stereo, task
        self.latest, self.lock = {}, threading.Lock()
        sock = ctx.socket(zmq.SUB)
        sock.setsockopt(zmq.CONFLATE, 1)
        sock.setsockopt_string(zmq.SUBSCRIBE, "")
        sock.connect(f"tcp://{host}:{port}")
        threading.Thread(target=self._recv, args=(sock,), daemon=True).start()

    def _recv(self, sock):
        import base64

        import cv2

        while True:
            msg = json.loads(sock.recv_string())
            decoded = {
                name: cv2.imdecode(np.frombuffer(base64.b64decode(b64), np.uint8), cv2.IMREAD_COLOR)
                for name, b64 in msg.get("images", {}).items()
            }
            with self.lock:
                self.latest.update(decoded)

    def frame(self, t: float, wait_s: float = 10.0) -> tuple[int, dict, None, str]:
        import cv2

        deadline = time.monotonic() + wait_s
        while True:
            with self.lock:
                cams = dict(self.latest)
            if all(k.split(".")[-1] in cams for k in self.keys) or self.stereo in cams:
                break
            if time.monotonic() > deadline:
                raise RuntimeError(
                    f"no camera frames for {self.keys} (or stereo '{self.stereo}') from the image server"
                )
            time.sleep(0.02)
        imgs = {}
        for i, key in enumerate(self.keys):
            name = key.split(".")[-1]  # observation.images.cam_left_high -> cam_left_high
            if name in cams:
                img = cams[name]
            else:  # side-by-side stereo: keys in order left, right
                full = cams[self.stereo]
                half = full.shape[1] // 2
                img = full[:, :half] if "left" in name or (i == 0 and "right" not in name) else full[:, half:]
            _, h, w = self.shapes[key]
            imgs[key] = (
                img if img.shape[:2] == (h, w) else cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
            )
        return int(t * self.fps), imgs, None, self.task


class RemotePolicy:
    """ChunkPolicy stand-in that asks sonic_policy_server.py (on H100, over the VPN: the robot side opens the
    connection). Lazy-pirate REQ: a request that times out drops the socket and raises, so the streamer keeps its
    last chunk and ends the episode after --max-chunk-age-s. Frames travel as JPEG (RGB in, RGB out)."""

    def __init__(self, addr: str, timeout_s: float):
        self.addr, self.timeout_ms, self.sock = addr, int(timeout_s * 1000), None
        info = self._ask({"op": "info"})
        self._close()  # the inference thread opens its own socket (ZMQ sockets are not thread-safe)
        self.image_keys, self.n_obs = info["image_keys"], int(info["n_obs"])
        self.shapes = {k: tuple(v) for k, v in info["shapes"].items()}

    def _ask(self, req: dict) -> dict:
        import msgpack

        if self.sock is None:
            self.sock = zmq.Context.instance().socket(zmq.REQ)
            self.sock.setsockopt(zmq.LINGER, 0)
            self.sock.connect(self.addr)
        self.sock.send(msgpack.packb(req, use_bin_type=True))
        if not self.sock.poll(self.timeout_ms):
            self._close()
            raise TimeoutError(f"policy server {self.addr} did not answer within {self.timeout_ms} ms")
        rep = msgpack.unpackb(self.sock.recv(), raw=False)
        if "error" in rep:
            raise RuntimeError(f"policy server: {rep['error']}")
        return rep

    def _close(self):
        if self.sock is not None:
            self.sock.close()
            self.sock = None

    def chunk(
        self, states: list[np.ndarray], images: list[dict], task: str, rtc: dict | None = None
    ) -> np.ndarray:
        import cv2

        jpegs = [
            {k: cv2.imencode(".jpg", im[k], [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes() for k in im}
            for im in images
        ]
        rep = self._ask(
            {
                "op": "chunk",
                "states": [np.asarray(x, np.float32).tolist() for x in states],
                "images": jpegs,
                "task": task,
                **({"rtc": rtc} if rtc else {}),
            }
        )
        return np.frombuffer(rep["chunk"], np.float32).reshape(rep["shape"])

    def chunk_rtc(self, states, images, task: str, offset: int | None, delay: int) -> np.ndarray:
        rtc = None if offset is None else {"offset": int(offset), "delay": int(delay)}
        return self.chunk(states, images, task, rtc)


class InferenceWorker:
    """One long-lived inference thread. A fresh thread per call re-pays per-thread setup on every call (GR00T and
    MolmoAct2: 2-3 s instead of 0.1-0.2 s), longer than the chunk-age limit."""

    def __init__(self):
        self.jobs, self.slot = queue.Queue(), None
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while True:
            fn, args, slot = self.jobs.get()
            started = time.monotonic()
            try:
                slot["result"] = fn(*args)
            except Exception as exc:  # reported by the caller; the streamer keeps the last good chunk
                slot["error"] = repr(exc)
            slot["latency"] = time.monotonic() - started
            slot["done"].set()

    def submit(self, fn, *args) -> dict:
        self.slot = {"done": threading.Event()}
        self.jobs.put((fn, args, self.slot))
        return self.slot

    def run(self, fn, *args):
        slot = self.submit(fn, *args)
        slot["done"].wait()
        if "error" in slot:
            raise RuntimeError(slot["error"])
        return slot["result"]

    @property
    def busy(self) -> bool:
        return self.slot is not None and not self.slot["done"].is_set()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--policy-path", help="local policy checkpoint (this machine has the GPU)")
    p.add_argument(
        "--policy-server", help="tcp://HOST:PORT of sonic_policy_server.py (e.g. H100 over the VPN)"
    )
    p.add_argument("--policy-timeout-s", type=float, default=1.5, help="--policy-server request timeout")
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--backbone-dtype", choices=["bfloat16"], help="--policy-path GR00T: cast the frozen backbone"
    )
    p.add_argument("--deploy-host", default="localhost")
    p.add_argument("--action-port", type=int, default=5556)
    p.add_argument("--state-port", type=int, default=5557)
    p.add_argument(
        "--images", choices=["dataset", "zmq"], default="dataset", help="zmq: live head camera (real robot)"
    )
    p.add_argument("--dataset-root", help="--images dataset: recorded episode source")
    p.add_argument("--episode", type=int, help="--images dataset: recorded episode index")
    p.add_argument("--camera-host", default="localhost", help="--images zmq: LeRobot ImageServer host")
    p.add_argument("--camera-port", type=int, default=5555)
    p.add_argument(
        "--stereo-camera", default="head_camera", help="side-by-side stereo camera split into left/right"
    )
    p.add_argument("--task", help="instruction text (required with --images zmq; overrides the episode's)")
    p.add_argument("--replan-s", type=float, default=0.4)
    p.add_argument(
        "--rtc",
        action="store_true",
        help="real-time chunking in closed loop: each new chunk is generated from the unplayed tail of the current "
        "one (GR00T native overlap inpainting, Pi0.5 guided RTC), with the last inference latency's rows frozen",
    )
    p.add_argument(
        "--chunk-blend-s",
        type=float,
        default=0.0,
        help="cross-fade from the previous chunk to a newly arrived one over this many seconds (0 = switch at "
        "once, the evaluated setting); smooths chunk switches of stochastic policies such as GR00T",
    )
    p.add_argument(
        "--noise-seed",
        type=int,
        help="--policy-path: reseed the sampling noise before every chunk (GR00T flow matching; see the server)",
    )
    p.add_argument(
        "--noise-scale",
        type=float,
        help="--policy-path: scale the sampler's initial noise (temperature); 0 = deterministic mean-path chunks "
        "(open loop: smoother and more accurate for GR00T and Pi0.5). Default: 1 (unscaled)",
    )
    p.add_argument("--duration-s", type=float, help="default: the episode's length")
    p.add_argument("--gate-dir", type=Path, help="sim: wait for flag files instead of Enter")
    p.add_argument("--log", type=Path, help="write a per-tick .npz log")
    p.add_argument(
        "--max-chunk-age-s",
        type=float,
        default=1.0,
        help="end the episode if no new chunk arrives for this long (keep below the chunk's duration)",
    )
    p.add_argument(
        "--max-state-age-s",
        type=float,
        default=0.2,
        help="end the episode if the deploy's g1_debug robot state is older than this (telemetry lost)",
    )
    p.add_argument(
        "--max-token-step",
        type=float,
        default=0.0,
        help="slew limit for a first real-robot deployment only (e.g. 0.05): largest change of any token dimension "
        "per 20 ms tick; 0 = off (default), as the policies were trained on speed-limit-free tokens. Steps within a "
        "chunk stay below 0.04; a new chunk can jump by up to 0.25, which 0.05 spreads over ~0.1 s",
    )
    p.add_argument(
        "--max-arm-speed",
        type=float,
        default=6.0,
        help="watchdog: end the episode (normal shutdown path) if any measured arm joint moves faster than this "
        "(rad/s over 60 ms; teleop data p99.9 2.6, sim runs max 5.8; 0 = off)",
    )
    p.add_argument(
        "--arm-range-margin",
        type=float,
        default=0.15,
        help="watchdog: end the episode if a measured arm joint leaves the training data's range by more than this "
        "(rad)",
    )
    p.add_argument(
        "--state-source",
        choices=["robot", "dataset"],
        default="robot",
        help="robot: closed loop on the deploy's state; dataset: feed the recorded observation.state (evaluation)",
    )
    p.add_argument(
        "--handoff-blend-s",
        type=float,
        default=2.0,
        help="seconds to blend from the planner's current token to the standing token after the POSE switch "
        "(0 = jump). 2 s kept all feet down in 3/3 deploy runs; the jump unloaded a foot in 5/5; 4 s swayed in 1/3.",
    )
    p.add_argument(
        "--dex3-right-order",
        choices=["dataset", "swap"],
        default="dataset",
        help="swap: right hand index<->middle between dataset order and the deploy's slot order (NVIDIA MuJoCo)",
    )
    p.add_argument(
        "--end",
        choices=["planner", "stop"],
        default="planner",
        help="after the episode: planner = blend back to the planner token and leave the deploy running in planner "
        "mode (real robot); stop = stop control",
    )
    p.add_argument(
        "--startup-tokens",
        type=Path,
        help="table startup .npz from sonic_table_startup.py plan: after the POSE switch, play this table-safe arm "
        "path (planner stance -> fixed initial pose) instead of NVIDIA's standing token; after the episode, play it "
        "in reverse back to the stance",
    )
    p.add_argument(
        "--start",
        choices=["standing", "planner"],
        default="standing",
        help="episode start/end pose without --startup-tokens: standing = NVIDIA's standing token (hands rise to "
        "~0.8 m, 0.3 m forward); planner = stay in the planner stance (arms down) and blend from it into the policy "
        "and back, for policies whose episodes start with the arms low, e.g. Humanoid Everyday (table edge >= ~30 cm "
        "in front of the pelvis)",
    )
    p.add_argument(
        "--action-space",
        choices=["sonic78", "joint28", "joint31"],
        default="sonic78",
        help="joint28: the policy outputs arm + Dex3 hand joints (28D); joint31: + waist yaw/roll/pitch (motor 12-14). "
        "With --backend sonic each chunk is encoded to SONIC tokens here (needs --encoder-model, "
        "--observation-config, --robot-xml)",
    )
    p.add_argument(
        "--backend",
        choices=["sonic", "decoupled"],
        default="sonic",
        help="sonic: SONIC tokens to NVIDIA's deploy; decoupled: 50 Hz joint targets to decoupled_wbc_sim_host.py "
        "(GR00T decoupled WBC; joint action spaces only)",
    )
    p.add_argument(
        "--replay",
        action="store_true",
        help="stream the episode's recorded actions (--dataset-root) instead of a policy: chunks of --replay-horizon "
        "frames starting at the frame of each --replan-s time",
    )
    p.add_argument(
        "--replay-horizon", type=int, default=40, help="--replay chunk length in frames (GR00T: 40)"
    )
    p.add_argument(
        "--synthetic-waist",
        action="store_true",
        help="--action-space joint31: append wbc_common.synthetic_waist (16 s test track) to 28D chunks",
    )
    for name in ("encoder-model", "observation-config", "robot-xml"):
        p.add_argument(f"--{name}", type=Path, help="--action-space joint28: as for prepare_sonic_dataset.py")
    a = p.parse_args()
    if a.start == "planner" and (a.startup_tokens or a.handoff_blend_s <= 0):
        p.error(
            "--start planner excludes --startup-tokens and needs --handoff-blend-s > 0 (it reads the planner token)"
        )
    joint_space = a.action_space in ("joint28", "joint31")
    if a.replay:
        if a.policy_path or a.policy_server:
            p.error("--replay excludes --policy-path and --policy-server")
        if a.images != "dataset":
            p.error("--replay needs --images dataset")
    elif bool(a.policy_path) == bool(a.policy_server):
        p.error("give exactly one of --policy-path or --policy-server (or --replay)")
    if a.backend == "decoupled" and a.dex3_right_order == "swap":
        p.error("--backend decoupled excludes --dex3-right-order swap (C addresses joints by name)")
    if a.backend == "decoupled" and not joint_space:
        p.error("--backend decoupled needs --action-space joint28 or joint31")
    if a.backend == "decoupled" and (a.startup_tokens or a.max_token_step > 0):
        p.error("--backend decoupled excludes --startup-tokens and --max-token-step (token-only options)")
    if a.synthetic_waist and a.action_space != "joint31":
        p.error("--synthetic-waist needs --action-space joint31")
    joint28 = None  # (robot joint limits, SONIC encoder) for encoding joint chunks
    if joint_space and a.backend == "sonic":
        if not (a.encoder_model and a.observation_config and a.robot_xml):
            p.error(
                f"--action-space {a.action_space} needs --encoder-model, --observation-config and --robot-xml"
            )
        joint28 = (load_joint_limits(a.robot_xml), SonicEncoder(a.encoder_model, a.observation_config))
    if a.images == "dataset" and (a.dataset_root is None or a.episode is None):
        p.error("--images dataset needs --dataset-root and --episode")
    if a.images == "zmq" and (a.task is None or a.duration_s is None or a.state_source == "dataset"):
        p.error("--images zmq needs --task and --duration-s, and --state-source robot")
    startup = None
    if a.startup_tokens:
        if a.handoff_blend_s <= 0:
            p.error(
                "--startup-tokens needs --handoff-blend-s > 0 (it blends from the planner token into the path)"
            )
        startup = load_startup(a.startup_tokens)
    require_package("pyzmq", extra="pyzmq-dep", import_name="zmq")
    if a.dex3_right_order == "swap":
        RIGHT_ORDER[:] = RIGHT_SWAP

    ctx = zmq.Context()
    if a.replay:  # no policy: images are not needed, actions come from the dataset
        policy = SimpleNamespace(image_keys=[], shapes={}, n_obs=1, chunk=lambda *_: None)
    else:
        policy = (
            RemotePolicy(a.policy_server, a.policy_timeout_s)
            if a.policy_server
            else ChunkPolicy(a.policy_path, a.device, a.backbone_dtype, a.noise_seed, a.noise_scale)
        )
    if a.images == "zmq":
        images = LiveImages(
            ctx, a.camera_host, a.camera_port, policy.image_keys, policy.shapes, a.stereo_camera, a.task
        )
    else:
        images = DatasetImages(a.dataset_root, a.episode, policy.image_keys)
    replay = None
    if a.replay:
        replay = images.actions()
        check_replay_width(replay.shape[1], a.action_space, a.synthetic_waist)
    if a.synthetic_waist:
        check_synthetic_duration(a.duration_s or images.n / images.fps)
    worker = InferenceWorker()
    _, warm_imgs, warm_state, warm_task = images.frame(0.0)
    warm_state = np.zeros(28, np.float32) if warm_state is None else warm_state
    for _ in range(
        3
    ):  # warm up in the inference thread: VLA first calls take 2-5 s (and checks the server link)
        started = time.monotonic()
        worker.run(policy.chunk, [warm_state] * policy.n_obs, [warm_imgs] * policy.n_obs, a.task or warm_task)
    print(f"[streamer] policy ready: {time.monotonic() - started:.2f} s per chunk (warm)", flush=True)
    duration = a.duration_s or images.n / images.fps
    pub = ctx.socket(zmq.PUB)
    pub.bind(f"tcp://*:{a.action_port}")
    state = StateSubscriber(ctx, a.deploy_host, a.state_port)
    print(
        f"[streamer] policy {a.policy_path or a.policy_server} images {policy.image_keys} "
        f"{'live camera' if a.images == 'zmq' else f'episode {a.episode}'} {duration:.1f}s",
        flush=True,
    )

    def chunk_t0_of(k):
        """Replay chunk rows are dataset frames k, k+1, ...: their time base is frame k's time."""
        return k / images.fps

    history = []  # (t_ep, robot state) of recent episode ticks, for policies with n_obs_steps > 1
    log = {
        k: [] for k in ("wall", "t_ep", "phase", "frame", "token", "hands", "chunk_t0", "state", "joint_ref")
    }
    no_ref = np.full(31, np.nan, np.float32)

    def record(token, hands, phase, t_ep, k, chunk_t0, joint_ref):
        msg = state.latest()  # the ZMQ socket is only touched from this (main) thread
        ref = no_ref if joint_ref is None else np.asarray(joint_ref, np.float32)
        for key, val in (
            ("wall", time.time()),
            ("t_ep", t_ep),
            ("phase", phase),
            ("frame", k),
            ("token", token),
            ("hands", hands),
            ("chunk_t0", chunk_t0),
            ("state", observation_state(msg) if msg else np.full(28, np.nan)),
            ("joint_ref", ref),
        ):
            log[key].append(val)  # fmt: skip
        if msg and t_ep >= 0:
            history.append((t_ep, log["state"][-1]))
            del history[:-200]

    def arm_watchdog() -> str | None:
        """Why the episode must end, from the measured arm state of recent ticks; None while it is fine."""
        if len(history) <= ARM_SPEED_TICKS:
            return None
        (t0, s0), (t1, s1) = history[-1 - ARM_SPEED_TICKS], history[-1]
        arm = s1[:14]
        if a.max_arm_speed > 0 and t1 > t0:
            speed = np.abs(arm - s0[:14]) / (t1 - t0)
            j = int(np.argmax(speed))
            if speed[j] > a.max_arm_speed:
                return f"arm joint {j} measured at {speed[j]:.1f} rad/s > {a.max_arm_speed:g}"
        outside = np.maximum(ARM_LO - arm, arm - ARM_HI)
        j = int(np.argmax(outside))
        if outside[j] > a.arm_range_margin:
            return f"arm joint {j} at {arm[j]:.2f} rad, {outside[j]:.2f} rad outside the training range"
        return None

    def robot_states(t_ep):
        """Robot state at t_ep - i/fps (oldest first), nearest earlier tick; the current state if none yet."""
        now = observation_state(state.latest())
        out = []
        for i in reversed(range(policy.n_obs)):
            t = t_ep - i / images.fps
            earlier = [s for ts, s in history if ts <= t + 1e-9]
            out.append(earlier[-1] if earlier else now)
        return out

    def infer(t_ep, robot_hist, rtc_offset=None, rtc_delay=0):
        obs = [images.frame(max(t_ep - i / images.fps, 0.0)) for i in reversed(range(policy.n_obs))]
        k, task = obs[-1][0], a.task or obs[-1][3]
        if replay is not None:
            joints = replay_chunk(replay, k, a.replay_horizon)
        else:
            # --state-source dataset: recorded state (isolates execution from state feedback); robot: closed loop
            states = [o[2] for o in obs] if a.state_source == "dataset" else robot_hist
            joints = policy.chunk_rtc(states, [o[1] for o in obs], task, rtc_offset, rtc_delay)
        if a.synthetic_waist:
            times = (k + np.arange(len(joints))) / images.fps
            joints = np.hstack([joints[:, :28], synthetic_waist(times).astype(np.float32)])
        if not np.isfinite(joints).all():
            raise ValueError(f"rejected chunk at t={t_ep:.2f}s (nonfinite)")
        if a.action_space == "sonic78":
            chunk, joints = joints, None
        else:
            width = 31 if a.action_space == "joint31" else 28
            if joints.shape[1] != width:
                raise ValueError(
                    f"policy chunk is {joints.shape[1]}D, --action-space {a.action_space} needs {width}D"
                )
            # backend sonic: the 30 Hz joint chunk -> tokens + hands, exactly as before (A's baseline path)
            chunk = None if joint28 is None else joint_chunk_to_sonic(joints, *joint28, fps=images.fps)
            joints = to_joint_ref(joints)
        if chunk is not None and (not np.isfinite(chunk).all() or np.abs(chunk[:, :64]).max() > TOKEN_BOUND):
            raise ValueError(f"rejected chunk at t={t_ep:.2f}s (nonfinite or |token| > {TOKEN_BOUND})")
        return k, chunk, joints

    if a.backend == "decoupled":
        backend = DecoupledBackend(a, pub, state, record, images.fps)
    else:
        backend = SonicBackend(a, pub, state, record, images.fps, startup)
    backend.start()
    # First chunk: keep the 50 Hz stream on the rest pose (which the robot already holds) while it is computed,
    # instead of pausing the deploy's input (0.16 s on H100 locally, ~0.28 s from the robot PC over Wi-Fi).
    slot = worker.submit(infer, 0.0, robot_states(0.0))
    for _ in ticks(10**9):
        if slot["done"].is_set():
            break
        backend.wait_tick()
    if "error" in slot:
        raise RuntimeError(slot["error"])
    _, first, first_j = slot["result"]
    backend.begin(first, first_j)

    latencies, slot, slot_t = [], None, 0.0
    next_replan, last_chunk_t, chunk_t0 = a.replan_s, 0.0, 0.0
    last_accepted = True  # the first chunk is playing
    episode_end = "completed"
    for i in ticks(int(duration / TICK)):
        t_ep = i * TICK
        if slot is not None and slot["done"].is_set():
            last_accepted = "error" not in slot
            if "error" in slot:  # keep streaming the last good chunk; stop if it goes stale
                print(f"[streamer] {slot['error']}", flush=True)
            else:
                k_c, chunk, joints = slot["result"]
                t0 = chunk_t0_of(k_c) if replay is not None else slot_t
                backend.set_chunk(chunk, joints, t0, a.chunk_blend_s, t_ep)
                last_chunk_t, chunk_t0 = t_ep, t0
                latencies.append(slot["latency"])
            slot = None
        if t_ep - last_chunk_t > a.max_chunk_age_s:
            episode_end = f"no valid chunk for {a.max_chunk_age_s:g} s"
            print(f"[streamer] {episode_end}: ending episode", flush=True)
            break
        state.latest()
        if state.age() > a.max_state_age_s:  # never plan or stream on frozen joints
            episode_end = f"robot state {state.age():.2f} s old"
            print(f"[streamer] {episode_end}: ending episode", flush=True)
            break
        reason = arm_watchdog()
        if reason:
            episode_end = f"watchdog: {reason}"
            print(f"[streamer] {episode_end}: ending episode", flush=True)
            break
        if t_ep >= next_replan and not worker.busy:
            # rows of the current chunk played by now; rows that play during inference stay frozen
            lat = latencies[-1] if latencies else a.replan_s / 2
            rtc = rtc_args(a.rtc, last_accepted, t_ep, chunk_t0, lat, images.fps)
            slot, slot_t = worker.submit(infer, t_ep, robot_states(t_ep), *rtc), t_ep
            next_replan = t_ep + a.replan_s
        k = min(int(t_ep * images.fps), images.n - 1)
        backend.episode_tick(t_ep, k, chunk_t0)
    if latencies:
        print(
            f"[streamer] chunk inference s: median {np.median(latencies):.2f}, max {max(latencies):.2f}, n {len(latencies)}",
            flush=True,
        )
    backend.finish()
    if a.log:
        np.savez_compressed(
            a.log,
            **{k: np.asarray(v) for k, v in log.items()},
            episode=-1 if a.episode is None else a.episode,
            episode_end=episode_end,
            args=json.dumps(vars(a), default=str),  # the exact settings of this run
        )
    if a.gate_dir:
        write_done(a.gate_dir, episode_end)


if __name__ == "__main__":
    main()
