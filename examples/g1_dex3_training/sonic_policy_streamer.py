"""Stream a LeRobot SONIC-token policy (78D = token 64 + Dex3 hands 14, 30 Hz) to NVIDIA's g1_deploy_onnx_ref.

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
episodes then start from, and end back at, the path's final pose (hands above the table).
Right Dex3 hand in dataset order (thumb, index, middle), which the real robot follows; --dex3-right-order swap
only for NVIDIA's MuJoCo bridge.

Images: --images dataset feeds the recorded episode's camera frames at the elapsed time (evaluation without a
sim/real camera gap). Operator gates: --gate-dir waits for flag files (sim host), otherwise press Enter.
"""

from __future__ import annotations

import argparse
import json
import struct
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from lerobot.utils.import_utils import _zmq_available, require_package

if TYPE_CHECKING or _zmq_available:
    import zmq

try:
    from .sonic_token_stream import ChunkResampler
except ImportError:
    from sonic_token_stream import ChunkResampler

HEADER_SIZE = 1280  # gear_sonic zmq_planner_sender / zmq_packed_message_subscriber.hpp in our deploy image
TOKEN_BOUND = 1.25  # run_vla_inference rejects chunks whose token magnitude exceeds this
# gear_sonic/utils/inference/initial_poses.py
LATENT_INITIAL_MOTION_TOKEN = np.array(
    [-0.0625, 0.0, -0.0625, -0.125, -0.1875, -0.0625, 0.1875, 0.25, 0.1875, -0.125, 0.0625, -0.0625, -0.25, -0.25,
     -0.3125, -0.0625, 0.0, -0.0625, -0.125, -0.1875, 0.0, -0.25, 0.0, -0.25, -0.0625, 0.0625, 0.125, -0.125,
     0.25, 0.1875, 0.25, -0.125, 0.125, 0.1875, -0.0625, 0.0, -0.1875, -0.1875, 0.25, 0.0, 0.0, -0.125,
     0.0625, 0.0, -0.0625, -0.0625, 0.1875, -0.0625, 0.0, 0.0625, 0.125, 0.0625, 0.125, 0.0625, 0.125, 0.0,
     0.125, 0.1875, 0.0, 0.0, 0.0625, 0.0625, 0.1875, 0.0625],
    dtype=np.float32,
)  # fmt: skip
TICK = 0.02


# ---------------------------------------------------------------- deploy wire format
def _message(topic: str, fields: list[dict], payload: bytes, version: int) -> bytes:
    header = json.dumps({"v": version, "endian": "le", "count": 1, "fields": fields}, separators=(",", ":"))
    if len(header) > HEADER_SIZE:
        raise ValueError("header too large")
    return topic.encode() + header.encode().ljust(HEADER_SIZE, b"\x00") + payload


def command_message(start: bool, planner: bool) -> bytes:
    fields = [{"name": n, "dtype": "u8", "shape": [1]} for n in ("start", "stop", "planner")]
    return _message("command", fields, struct.pack("BBB", start, not start, planner), version=1)


def pose_message(token: np.ndarray, hands: np.ndarray, frame_index: int) -> bytes:
    arrays = {
        "token_state": np.asarray(token, "<f4").reshape(1, 64),
        "frame_index": np.array([frame_index], "<i8"),
        "left_hand_joints": np.asarray(hands[:7], "<f4").reshape(1, 7),
        "right_hand_joints": np.asarray(hands[7:], "<f4")[RIGHT_ORDER].reshape(1, 7),
    }
    fields = [
        {"name": k, "dtype": "i64" if v.dtype.kind == "i" else "f32", "shape": list(v.shape)}
        for k, v in arrays.items()
    ]
    return _message(
        "pose", fields, b"".join(np.ascontiguousarray(v).tobytes() for v in arrays.values()), version=4
    )


# Dataset (Unitree teleop) right-hand order is thumb 0-2, index 0-1, middle 0-1. NVIDIA's MuJoCo bridge maps Dex3
# slots in the XML's order, thumb, middle, index, like the left hand. --dex3-right-order swap converts both ways.
RIGHT_SWAP = np.array(
    [0, 1, 2, 5, 6, 3, 4]
)  # an involution: the same permutation converts in both directions
RIGHT_ORDER = np.arange(7)


def observation_state(msg: dict) -> np.ndarray:
    """28D joint28 state: arms (motor indices 15..28) + left Dex3 7 + right Dex3 7, as in the datasets."""
    body = np.asarray(msg["body_q"], np.float32)
    if body.shape != (29,):
        raise ValueError(f"expected 29 body joints, got {body.shape}")
    right = np.asarray(msg["right_hand_q"], np.float32)[RIGHT_ORDER]
    return np.concatenate([body[15:29], msg["left_hand_q"], right]).astype(np.float32)


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


def dataset_obs(ds, k: int, n: int, image_keys: list[str]) -> tuple[list[np.ndarray], list[dict], str]:
    """Last n dataset frames up to k (oldest first): states, images (HWC uint8), task. For offline diagnostics."""
    items = [ds[max(k - i, 0)] for i in reversed(range(n))]
    imgs = [
        {key: (it[key].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8) for key in image_keys}
        for it in items
    ]
    states = [it["observation.state"].numpy().astype(np.float32) for it in items]
    return states, imgs, str(items[-1].get("task", ""))


class ChunkPolicy:
    def __init__(self, path: str, device: str):
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.policies.factory import get_policy_class, make_pre_post_processors

        cfg = PreTrainedConfig.from_pretrained(path)
        cfg.device = device
        self.policy = get_policy_class(cfg.type).from_pretrained(path, config=cfg).to(device).eval()
        self.pre, self.post = make_pre_post_processors(
            policy_cfg=cfg,
            pretrained_path=path,
            preprocessor_overrides={"device_processor": {"device": device}},
        )
        self.device = torch.device(device)
        self.image_keys = [k for k in cfg.input_features if "image" in k]
        self.n_obs = int(getattr(cfg, "n_obs_steps", 1) or 1)  # diffusion conditions on 2 frames

    @torch.inference_mode()
    def chunk(self, states: list[np.ndarray], images: list[dict], task: str) -> np.ndarray:
        """states/images: the last n_obs observations, oldest first (1/fps apart); task: instruction text."""
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
        actions = self.policy.predict_action_chunk(self.pre(batch))
        return self.post(actions).squeeze(0).float().cpu().numpy()  # [T, 78]


def gate(name: str, gate_dir: Path | None) -> None:
    if gate_dir is None:
        input(f"[streamer] {name}: press Enter to continue ")
        return
    print(f"[streamer] waiting for {gate_dir / name}", flush=True)
    while not (gate_dir / name).exists():
        time.sleep(0.05)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--policy-path", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--deploy-host", default="localhost")
    p.add_argument("--action-port", type=int, default=5556)
    p.add_argument("--state-port", type=int, default=5557)
    p.add_argument("--images", choices=["dataset"], default="dataset")
    p.add_argument("--dataset-root", required=True)
    p.add_argument("--episode", type=int, required=True)
    p.add_argument("--replan-s", type=float, default=0.4)
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
        "--startup-tokens",
        type=Path,
        help="table startup .npz from sonic_table_startup.py plan: after the POSE switch, play this table-safe arm "
        "path (planner stance -> fixed initial pose) instead of NVIDIA's standing token; episodes start and end there",
    )
    a = p.parse_args()
    startup = None
    if a.startup_tokens:
        if a.handoff_blend_s <= 0:
            p.error(
                "--startup-tokens needs --handoff-blend-s > 0 (it blends from the planner token into the path)"
            )
        z = np.load(a.startup_tokens)
        startup = {"tokens": z["tokens"].astype(np.float32), "hands": z["hands"].astype(np.float32)}
        startup["fps"] = float(json.loads(str(z["meta"]))["fps"])
    rest_token = LATENT_INITIAL_MOTION_TOKEN if startup is None else startup["tokens"][-1]
    rest_hands = np.zeros(14, np.float32) if startup is None else startup["hands"][-1]
    require_package("pyzmq", extra="pyzmq-dep", import_name="zmq")
    if a.dex3_right_order == "swap":
        RIGHT_ORDER[:] = RIGHT_SWAP

    policy = ChunkPolicy(a.policy_path, a.device)
    images = DatasetImages(a.dataset_root, a.episode, policy.image_keys)
    duration = a.duration_s or images.n / images.fps
    ctx = zmq.Context()
    pub = ctx.socket(zmq.PUB)
    pub.bind(f"tcp://*:{a.action_port}")
    state = StateSubscriber(ctx, a.deploy_host, a.state_port)
    print(
        f"[streamer] policy {a.policy_path} images {policy.image_keys} episode {a.episode} {duration:.1f}s",
        flush=True,
    )

    gate("deploy_ready", a.gate_dir)
    time.sleep(0.5)  # let the deploy's SUB connect to our PUB before the first command
    pub.send(command_message(start=True, planner=True))
    print("[streamer] command: start control, PLANNER mode", flush=True)
    gate("settled", a.gate_dir)
    deadline = time.monotonic() + 10  # the deploy publishes g1_debug once control runs
    while state.latest() is None:
        if time.monotonic() > deadline:
            raise RuntimeError("no g1_debug robot state from the deploy within 10 s")
        time.sleep(0.05)

    zero_h = np.zeros(14, np.float32)
    frame = [0]
    history = []  # (t_ep, robot state) of recent episode ticks, for policies with n_obs_steps > 1
    log = {k: [] for k in ("wall", "t_ep", "phase", "frame", "token", "hands", "chunk_t0", "state")}

    def send(token, hands, phase, t_ep=-1.0, k=-1, chunk_t0=np.nan):
        pub.send(pose_message(token, hands, frame[0]))
        frame[0] += 1
        msg = state.latest()  # the ZMQ socket is only touched from this (main) thread
        for key, val in (
            ("wall", time.time()),
            ("t_ep", t_ep),
            ("phase", phase),
            ("frame", k),
            ("token", token),
            ("hands", hands),
            ("chunk_t0", chunk_t0),
            ("state", observation_state(msg) if msg else np.full(28, np.nan)),
        ):
            log[key].append(val)  # fmt: skip
        if msg and t_ep >= 0:
            history.append((t_ep, log["state"][-1]))
            del history[:-200]

    def robot_states(t_ep):
        """Robot state at t_ep - i/fps (oldest first), nearest earlier tick; the current state if none yet."""
        now = observation_state(state.latest())
        out = []
        for i in reversed(range(policy.n_obs)):
            t = t_ep - i / images.fps
            earlier = [s for ts, s in history if ts <= t + 1e-9]
            out.append(earlier[-1] if earlier else now)
        return out

    def ticks(n):
        start = time.monotonic()
        for i in range(n):
            yield i
            time.sleep(max(0.0, start + (i + 1) * TICK - time.monotonic()))

    def infer(t_ep, robot_hist):
        obs = [images.frame(max(t_ep - i / images.fps, 0.0)) for i in reversed(range(policy.n_obs))]
        k, task = obs[-1][0], obs[-1][3]
        # --state-source dataset: recorded state (isolates execution from state feedback); robot: closed loop
        states = [o[2] for o in obs] if a.state_source == "dataset" else robot_hist
        chunk = policy.chunk(states, [o[1] for o in obs], task)
        if not np.isfinite(chunk).all() or np.abs(chunk[:, :64]).max() > TOKEN_BOUND:
            raise ValueError(f"rejected chunk at t={t_ep:.2f}s (nonfinite or |token| > {TOKEN_BOUND})")
        return k, chunk

    if (
        a.handoff_blend_s > 0
    ):  # gradual: start POSE mode from the planner's current token (g1_debug token_state)
        planner = np.asarray(state.latest().get("token_state", []), np.float32)
        if planner.shape != (64,):
            raise RuntimeError("no 64D token_state from the deploy for the gradual handoff")
        for _ in ticks(50):
            send(planner, zero_h, "planner token")
        pub.send(command_message(start=True, planner=False))
        n = int(a.handoff_blend_s * 50)
        # table startup: switch into the table-safe arm path instead of NVIDIA's standing token, whose hands rise
        # to ~0.8 m, 0.3 m forward (into an 80 cm table)
        target = LATENT_INITIAL_MOTION_TOKEN if startup is None else startup["tokens"][0]
        for i in ticks(n):
            w = (i + 1) / n
            send((1 - w) * planner + w * target, zero_h, "handoff blend")
        if startup is not None:
            toks, hands = startup["tokens"], startup["hands"]
            for i in ticks(
                int((len(toks) - 1) / startup["fps"] * 50)
            ):  # 30 Hz path, look-ahead interp to 50 Hz
                x = i * TICK * startup["fps"]
                j, f = int(x), x - int(x)
                send(
                    toks[j] + f * (toks[j + 1] - toks[j]),
                    hands[j] + f * (hands[j + 1] - hands[j]),
                    "table startup",
                )
    else:
        for _ in ticks(50):  # latent initial token, then POSE mode
            send(LATENT_INITIAL_MOTION_TOKEN, zero_h, "initial")
        pub.send(command_message(start=True, planner=False))
    print("[streamer] command: POSE mode (streamed tokens)", flush=True)
    resampler = ChunkResampler(images.fps)
    _, first = infer(0.0, robot_states(0.0))
    resampler.set_chunk(first, t0=0.0)
    for i in ticks(
        50
    ):  # 1 s ease-in from the rest pose (standing token or table startup end) to the first chunk
        w = (i + 1) / 50
        send(
            (1 - w) * rest_token + w * first[0, :64],
            (1 - w) * rest_hands + w * first[0, 64:],
            "blend in",
            0.0,
            0,
        )

    pending, worker, latencies = {}, None, []

    def run_inference(t_obs, robot_state):
        try:
            started = time.monotonic()
            pending["result"] = (t_obs, *infer(t_obs, robot_state))
            latencies.append(time.monotonic() - started)
        except Exception as exc:  # keep streaming the last good chunk; stop if it goes stale
            pending["error"] = repr(exc)

    next_replan, last_chunk_t, chunk_t0 = a.replan_s, 0.0, 0.0
    for i in ticks(int(duration / TICK)):
        t_ep = i * TICK
        if "result" in pending:
            t_obs, _, chunk = pending.pop("result")
            resampler.set_chunk(chunk, t0=t_obs)
            last_chunk_t, chunk_t0 = t_ep, t_obs
        if "error" in pending:
            print(f"[streamer] {pending.pop('error')}", flush=True)
        if t_ep - last_chunk_t > a.max_chunk_age_s:
            print(f"[streamer] no valid chunk for {a.max_chunk_age_s:g} s: ending episode", flush=True)
            break
        state.latest()
        if state.age() > a.max_state_age_s:  # never plan or stream on frozen joints
            print(f"[streamer] robot state {state.age():.2f} s old: ending episode", flush=True)
            break
        if t_ep >= next_replan and (worker is None or not worker.is_alive()):
            snapshot = robot_states(t_ep)
            worker = threading.Thread(target=run_inference, args=(t_ep, snapshot), daemon=True)
            worker.start()
            next_replan = t_ep + a.replan_s
        out = resampler.token_at(t_ep)
        send(out[:64], out[64:], "episode", t_ep, min(int(t_ep * images.fps), images.n - 1), chunk_t0)
    if latencies:
        print(
            f"[streamer] chunk inference s: median {np.median(latencies):.2f}, max {max(latencies):.2f}, n {len(latencies)}",
            flush=True,
        )
    last_token, last_hands = log["token"][-1], log["hands"][-1]
    for i in ticks(50):  # 1 s blend back to the rest pose, 2 s hold, stop (at the table: hands stay above it)
        w = (i + 1) / 50
        send((1 - w) * last_token + w * rest_token, (1 - w) * last_hands + w * rest_hands, "blend out")
    for _ in ticks(100):
        send(rest_token, rest_hands, "hold rest")
    pub.send(command_message(start=False, planner=False))
    print("[streamer] command: stop", flush=True)
    if a.log:
        np.savez_compressed(a.log, **{k: np.asarray(v) for k, v in log.items()}, episode=a.episode)
    if a.gate_dir:
        (a.gate_dir / "done").touch()


if __name__ == "__main__":
    main()
