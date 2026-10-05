"""Whole-body backends of sonic_policy_streamer.py: startup, one send per 50 Hz tick, and shutdown.

SonicBackend: SONIC tokens + Dex3 hands to NVIDIA's g1_deploy_onnx_ref (command and pose v4 messages).
DecoupledBackend: 31D joint targets (wbc_common.pack_joint_message) to decoupled_wbc_sim_host.py.

The streamer drives either one the same way:
    start()                 gates and mode switches, until the robot holds the rest pose
    wait_tick()             one tick on the rest pose while the first chunk is computed
    begin(chunk, joints)    lead-in from the rest pose to the first chunk (time base 0)
    set_chunk(chunk, joints, t0, blend_s, now)   a newly arrived chunk
    episode_tick(t_ep, k, chunk_t0)               one episode tick from the latest chunk
    finish()                blend out, then return / hand back / stop
chunk is the 30 Hz token + hands chunk (None for decoupled), joints the 31D joint_ref chunk (None for sonic78).
record(token, hands, phase, t_ep, k, chunk_t0, joint_ref) is the streamer's per-tick log.
"""

from __future__ import annotations

import json
import struct
import time
from pathlib import Path

import numpy as np

try:
    from .sonic_targets import NOMINAL_BODY
    from .sonic_token_stream import ChunkResampler
    from .wbc_common import WAIST, pack_joint_message
except ImportError:
    from sonic_targets import NOMINAL_BODY
    from sonic_token_stream import ChunkResampler
    from wbc_common import WAIST, pack_joint_message

HEADER_SIZE = 1280  # gear_sonic zmq_planner_sender / zmq_packed_message_subscriber.hpp in our deploy image
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

# Dataset (Unitree teleop) right-hand order is thumb 0-2, index 0-1, middle 0-1. NVIDIA's MuJoCo bridge maps Dex3
# slots in the XML's order, thumb, middle, index, like the left hand. --dex3-right-order swap converts both ways.
RIGHT_SWAP = np.array(
    [0, 1, 2, 5, 6, 3, 4]
)  # an involution: the same permutation converts in both directions
RIGHT_ORDER = np.arange(7)


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


# ---------------------------------------------------------------- robot state, timing, operator gates
def observation_state(msg: dict) -> np.ndarray:
    """28D joint28 state: arms (motor indices 15..28) + left Dex3 7 + right Dex3 7, as in the datasets."""
    body = np.asarray(msg["body_q"], np.float32)
    if body.shape != (29,):
        raise ValueError(f"expected 29 body joints, got {body.shape}")
    right = np.asarray(msg["right_hand_q"], np.float32)[RIGHT_ORDER]
    return np.concatenate([body[15:29], msg["left_hand_q"], right]).astype(np.float32)


def measured_ref(msg: dict, action_space: str) -> np.ndarray:
    """31D joint_ref of the measured pose (arms, hands, waist); 28D runs get the nominal waist (backend C's
    lower_body waist location ignores it)."""
    ref = np.r_[observation_state(msg), np.asarray(msg["body_q"], np.float32)[12:15]].astype(np.float32)
    if action_space != "joint31":
        ref[WAIST] = NOMINAL_BODY[12:15]
    return ref


def ticks(n):
    """Yield 0..n-1 on a 50 Hz grid that starts now (late ticks are caught up by not sleeping)."""
    start = time.monotonic()
    for i in range(n):
        yield i
        time.sleep(max(0.0, start + (i + 1) * TICK - time.monotonic()))


def wait_state(state) -> None:
    deadline = time.monotonic() + 10  # the deploy / C host publishes g1_debug once control runs
    while state.latest() is None:
        if time.monotonic() > deadline:
            raise RuntimeError("no g1_debug robot state within 10 s")
        time.sleep(0.05)


GATE_PROMPTS = {
    "deploy_ready": "when the deploy has printed 'Init Done', press Enter to start control in PLANNER mode",
    "settled": "move the robot into position at the table (planner mode), then press Enter to start the episode",
}


def gate(name: str, gate_dir: Path | None) -> None:
    if gate_dir is None:
        input(f"[streamer] {GATE_PROMPTS.get(name, name)}: ")
        return
    print(f"[streamer] waiting for {gate_dir / name}", flush=True)
    while not (gate_dir / name).exists():
        time.sleep(0.05)


def load_startup(path: Path) -> dict:
    """--startup-tokens .npz from sonic_table_startup.py plan: tokens/hands (and rev_*) arrays and the path's fps."""
    z = np.load(path)
    startup = {k: z[k].astype(np.float32) for k in ("tokens", "hands", "rev_tokens", "rev_hands") if k in z}
    startup["fps"] = float(json.loads(str(z["meta"]))["fps"])
    return startup


# ---------------------------------------------------------------- backends
class SonicBackend:
    """Token source for NVIDIA's deploy. Start: PLANNER mode -> POSE mode with the planner's current token ->
    --handoff-blend-s blend to the rest token (NVIDIA's standing token, the table path's end, or the planner token
    itself with --start planner). Episode: ChunkResampler.token_at(t), the 30 Hz chunk interpolated to 50 Hz.
    End: 1 s blend back to the rest token, hold (or the table path reversed), then hand back to the planner or stop.
    For joint action spaces it also logs the 50 Hz joint_ref from the joint chunk (not sent)."""

    def __init__(self, a, pub, state, record, fps: float, startup: dict | None):
        self.a, self.pub, self.state, self.record, self.startup = a, pub, state, record, startup
        self.rest_token = LATENT_INITIAL_MOTION_TOKEN if startup is None else startup["tokens"][-1]
        self.rest_hands = np.zeros(14, np.float32) if startup is None else startup["hands"][-1]
        self.zero_h = np.zeros(14, np.float32)
        self.resampler = ChunkResampler(fps)
        self.jres = ChunkResampler(fps) if a.action_space != "sonic78" else None  # 50 Hz joint_ref
        self.planner = None  # the planner's token at the POSE switch (gradual handoff only)
        self.frame, self.last_sent, self.last_hands, self.slewed = 0, None, None, 0

    def send(self, token, hands, phase, t_ep=-1.0, k=-1, chunk_t0=np.nan, joint_ref=None):
        token = np.asarray(token, np.float32)
        step_max = self.a.max_token_step
        if step_max > 0 and self.last_sent is not None:  # slew limit on every phase
            step = token - self.last_sent
            if np.abs(step).max() > step_max:
                token = self.last_sent + np.clip(step, -step_max, step_max)
                self.slewed += 1
        self.last_sent, self.last_hands = token, hands
        self.pub.send(pose_message(token, hands, self.frame))
        self.frame += 1
        self.record(token, hands, phase, t_ep, k, chunk_t0, joint_ref)

    def _play(self, toks, hands, phase):
        """A 30 Hz table path (startup or its reverse), look-ahead interpolated to 50 Hz."""
        fps = self.startup["fps"]
        for i in ticks(int((len(toks) - 1) / fps * 50)):
            x = i * TICK * fps
            j, f = int(x), x - int(x)
            self.send(toks[j] + f * (toks[j + 1] - toks[j]), hands[j] + f * (hands[j + 1] - hands[j]), phase)

    def start(self):
        a, startup = self.a, self.startup
        gate("deploy_ready", a.gate_dir)
        time.sleep(0.5)  # let the deploy's SUB connect to our PUB before the first command
        self.pub.send(command_message(start=True, planner=True))
        print("[streamer] command: start control, PLANNER mode", flush=True)
        gate("settled", a.gate_dir)
        wait_state(self.state)
        if (
            a.handoff_blend_s > 0
        ):  # gradual: start POSE mode from the planner's current token (g1_debug token_state)
            planner = np.asarray(self.state.latest().get("token_state", []), np.float32)
            if planner.shape != (64,):
                raise RuntimeError("no 64D token_state from the deploy for the gradual handoff")
            self.planner = planner
            for _ in ticks(50):
                self.send(planner, self.zero_h, "planner token")
            self.pub.send(command_message(start=True, planner=False))
            n = int(a.handoff_blend_s * 50)
            # table startup: switch into the table-safe arm path instead of NVIDIA's standing token, whose hands rise
            # to ~0.8 m, 0.3 m forward (into an 80 cm table)
            target = LATENT_INITIAL_MOTION_TOKEN if startup is None else startup["tokens"][0]
            if a.start == "planner":  # rest on the planner stance itself: no standing token, no table path
                self.rest_token, n = planner.copy(), 0
            for i in ticks(n):
                w = (i + 1) / n
                self.send((1 - w) * planner + w * target, self.zero_h, "handoff blend")
            if startup is not None:
                self._play(startup["tokens"], startup["hands"], "table startup")
        else:
            for _ in ticks(50):  # latent initial token, then POSE mode
                self.send(LATENT_INITIAL_MOTION_TOKEN, self.zero_h, "initial")
            self.pub.send(command_message(start=True, planner=False))
        print("[streamer] command: POSE mode (streamed tokens)", flush=True)

    def wait_tick(self):
        self.send(self.rest_token, self.rest_hands, "wait first chunk")

    def begin(self, chunk, joints):
        self.resampler.set_chunk(chunk, t0=0.0)
        if self.jres is not None:
            self.jres.set_chunk(joints, t0=0.0)
        for i in ticks(
            50
        ):  # 1 s ease-in from the rest pose (standing token or table startup end) to the first chunk
            w = (i + 1) / 50
            self.send(
                (1 - w) * self.rest_token + w * chunk[0, :64],
                (1 - w) * self.rest_hands + w * chunk[0, 64:],
                "blend in",
                0.0,
                0,
            )

    def set_chunk(self, chunk, joints, t0, blend_s, now):
        self.resampler.set_chunk(chunk, t0=t0, blend_s=blend_s, now=now)
        if joints is not None:
            self.jres.set_chunk(joints, t0=t0, blend_s=blend_s, now=now)

    def episode_tick(self, t_ep, k, chunk_t0):
        out = self.resampler.token_at(t_ep)
        joint_ref = None if self.jres is None else self.jres.token_at(t_ep)
        self.send(out[:64], out[64:], "episode", t_ep, k, chunk_t0, joint_ref=joint_ref)

    def finish(self):
        a, startup = self.a, self.startup
        rest_token, rest_hands = self.rest_token, self.rest_hands
        print(f"[streamer] token slew limit active on {self.slewed} ticks so far", flush=True)
        last_token, last_hands = self.last_sent, self.last_hands
        for i in ticks(
            50
        ):  # 1 s blend back to the rest pose, 2 s hold, stop (at the table: hands stay above it)
            w = (i + 1) / 50
            self.send(
                (1 - w) * last_token + w * rest_token, (1 - w) * last_hands + w * rest_hands, "blend out"
            )
        if (
            startup is not None and "rev_tokens" in startup
        ):  # at the table: back to the stance on the path, reversed
            for _ in ticks(25):
                self.send(rest_token, rest_hands, "hold rest")
            toks, hands = startup["rev_tokens"], startup["rev_hands"]
            self._play(toks, hands, "table shutdown")
            for _ in ticks(50):
                self.send(toks[-1], hands[-1], "hold stance")
        else:
            for _ in ticks(100):
                self.send(rest_token, rest_hands, "hold rest")
        if a.end == "planner" and self.planner is not None:
            # hand back to the planner the way we took over: blend to the planner token recorded at the start, then
            # switch to planner mode and leave the deploy running (the operator stops it)
            held, held_h = self.last_sent, self.last_hands
            n = int(a.handoff_blend_s * 50)
            for i in ticks(n):
                w = (i + 1) / n
                self.send((1 - w) * held + w * self.planner, (1 - w) * held_h, "handback blend")
            self.pub.send(command_message(start=True, planner=True))
            print("[streamer] command: PLANNER mode (deploy left running)", flush=True)
        else:
            self.pub.send(command_message(start=False, planner=False))
            print("[streamer] command: stop", flush=True)


class DecoupledBackend:
    """50 Hz joint targets for decoupled_wbc_sim_host.py. Start: hold the measured pose. Episode: the 30 Hz joint
    chunk interpolated to 50 Hz. Lead-in: 2 s from the measured pose to the first chunk row, 1 s hold. End: 2 s back
    to the measured start pose."""

    no_token = np.full(64, np.nan, np.float32)

    def __init__(self, a, pub, state, record, fps: float):
        self.gate_dir, self.action_space = a.gate_dir, a.action_space
        self.pub, self.state, self.record = pub, state, record
        self.jres = ChunkResampler(fps)
        self.frame, self.rest_ref, self.last = 0, None, None

    def send(self, ref, phase, t_ep=-1.0, k=-1, chunk_t0=np.nan):
        """One joint message (wbc_common.pack_joint_message) per tick."""
        ref = np.asarray(ref, np.float32)
        self.pub.send(pack_joint_message(self.frame, phase, k, ref))
        self.frame += 1
        self.last = ref
        self.record(self.no_token, ref[14:28], phase, t_ep, k, chunk_t0, ref)

    def start(self):
        gate("deploy_ready", self.gate_dir)
        gate("settled", self.gate_dir)
        wait_state(self.state)
        self.rest_ref = measured_ref(self.state.latest(), self.action_space)

    def wait_tick(self):
        self.send(self.rest_ref, "wait first chunk")

    def begin(self, chunk, joints):
        self.jres.set_chunk(joints, t0=0.0)
        for i in ticks(100):  # 2 s lead-in in joint space from the measured pose
            w = (i + 1) / 100
            self.send((1 - w) * self.rest_ref + w * joints[0], "leadin")
        for _ in ticks(50):  # 1 s hold
            self.send(joints[0], "hold")

    def set_chunk(self, chunk, joints, t0, blend_s, now):
        self.jres.set_chunk(joints, t0=t0, blend_s=blend_s, now=now)

    def episode_tick(self, t_ep, k, chunk_t0):
        self.send(self.jres.token_at(t_ep), "episode", t_ep, k, chunk_t0)

    def finish(self):
        last = self.last
        for i in ticks(100):  # 2 s back to the measured start pose
            w = (i + 1) / 100
            self.send((1 - w) * last + w * self.rest_ref, "return")
