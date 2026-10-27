"""Whole-body backends of sonic_policy_streamer.py: startup, one send per 50 Hz tick, and shutdown.

SonicBackend: SONIC tokens + Dex3 hands to NVIDIA's g1_deploy_onnx_ref (command and pose v4 messages).
DecoupledBackend: 31D joint targets (wbc_common.pack_joint_message) to decoupled_wbc_sim_host.py.

The streamer drives either one the same way:
    start()                 gates and mode switches, until the robot holds the rest pose
                            (SonicBackend: connect() once + take_over() per episode, for --trials sessions)
    wait_tick()             one tick on the rest pose while the first chunk is computed
    begin(chunk, joints)    lead-in from the rest pose to the first chunk (time base 0)
    set_chunk(chunk, joints, t0, blend_s, now)   a newly arrived chunk
    episode_tick(t_ep, k, chunk_t0)               one episode tick from the latest chunk
    finish()                (SonicBackend, operator runs: hold the last pose until Enter) blend out, then return /
                            hand back / stop
chunk is the 30 Hz token + hands chunk (None for decoupled), joints the 31D joint_ref chunk (None for sonic78).
record(token, hands, phase, t_ep, k, chunk_t0, joint_ref) is the streamer's per-tick log.
"""

from __future__ import annotations

import contextlib
import json
import os
import select
import struct
import sys
import threading
import time
from collections import deque
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
    left, right = (np.asarray(msg[k], np.float32) for k in ("left_hand_q", "right_hand_q"))
    if left.shape != (7,) or right.shape != (7,):
        raise ValueError(f"expected 7 joints per Dex3 hand, got {left.shape} and {right.shape}")
    return np.concatenate([body[15:29], left, right[RIGHT_ORDER]]).astype(np.float32)


# Dex3 hand_q -> the robot model's joint order (index 0/1, middle 0/1, thumb 0/1/2) for both hands: what NVIDIA's data
# exporter stores (robot_model.get_configuration_from_actuated_joints, checked on marker values 2026-10-21).
OFFICIAL_HAND_ORDER = np.array([3, 4, 5, 6, 0, 1, 2])


def sonic46_state(msg: dict) -> np.ndarray:
    """46D state of the official GR00T UNITREE_G1_SONIC config (gr00t_official_server.py), in its key order: left and
    right leg 6+6, waist 3, left and right arm 7+7, left and right hand 7+7, projected gravity 3 (world [0, 0, -1] in the
    pelvis frame from the IMU base_quat w, x, y, z, as gear_sonic compute_projected_gravity)."""
    body = np.asarray(msg["body_q"], np.float64)
    left, right = (np.asarray(msg[k], np.float64)[OFFICIAL_HAND_ORDER] for k in ("left_hand_q", "right_hand_q"))
    quat = np.asarray(msg["base_quat"], np.float64)
    w, x, y, z = quat / np.linalg.norm(quat)
    gravity = [2 * (w * y - x * z), -2 * (y * z + w * x), 2 * (x * x + y * y) - 1]
    return np.concatenate([body[0:29], left, right, gravity]).astype(np.float32)


def say(text: str) -> None:
    """print() that cannot raise: a closed log pipe (tee gone) must not stop a blend out or handback."""
    with contextlib.suppress(OSError, ValueError):
        print(text, flush=True)


def token_echo_err(msg: dict | None, recent) -> float:
    """Smallest max-abs difference between the deploy's g1_debug token_state and a recently sent token. In POSE mode
    the deploy copies each received token into token_state; a deploy fed by another command source (or still in
    planner mode) reports other tokens. inf without a 64D token_state or sent tokens."""
    echo = np.asarray((msg or {}).get("token_state", []), np.float32)
    if echo.shape != (64,) or not recent:
        return float("inf")
    return float(np.abs(np.asarray(recent) - echo).max(axis=1).min())


def measured_ref(msg: dict, action_space: str) -> np.ndarray:
    """31D joint_ref of the measured pose (arms, hands, waist); 28D runs get the nominal waist (backend C's
    lower_body waist location ignores it)."""
    ref = np.r_[observation_state(msg), np.asarray(msg["body_q"], np.float32)[12:15]].astype(np.float32)
    if action_space != "joint31":
        ref[WAIST] = NOMINAL_BODY[12:15]
    return ref


def max_arm_speed(
    wall0: float, state0: np.ndarray, wall1: float, state1: np.ndarray
) -> tuple[int, float] | None:
    """(arm joint, rad/s) of the fastest of the 14 arm joints between two measured states; None if no time passed.
    Timed by the wall clock of the ticks, not t_ep: the 1 s blend in is logged at t_ep 0 (G1 run48: 3 ticks of
    blend-in motion over 20 ms of t_ep read 9.5 rad/s for a true 2.4 rad/s)."""
    if wall1 <= wall0:
        return None
    speed = np.abs(np.asarray(state1[:14]) - np.asarray(state0[:14])) / (wall1 - wall0)
    j = int(np.argmax(speed))
    return j, float(speed[j])


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
    "ready": "the robot holds the ready pose (hand over the table): place the bottle, then press Enter to start",
}


def gate(name: str, gate_dir: Path | None, prefix: str = "", consume: bool = False) -> None:
    """Wait for the operator (Enter) or, with gate_dir, for the flag file `name` (deleted once seen if consume:
    per-trial gates such as settled.3 are written again for every trial)."""
    if gate_dir is None:
        input(f"[streamer] {prefix}{GATE_PROMPTS.get(name.split('.')[0], name)}: ")
        return
    path = gate_dir / name
    print(f"[streamer] waiting for {path}", flush=True)
    while not path.exists():
        time.sleep(0.05)
    if consume:
        path.unlink()


def enter_pressed(pending: bytearray) -> bool | None:
    """Non-blocking poll, once per tick: True once a whole line (Enter) has arrived, None if stdin cannot deliver one
    (EOF, closed or not selectable). Reads only the bytes already there (os.read, not readline(), which waits for a
    newline and stalls the 50 Hz stream) and keeps a partial line in `pending`: a line cut off by EOF is no Enter."""
    try:
        fd = sys.stdin.fileno()
        if not select.select([fd], [], [], 0)[0]:
            return False
        data = os.read(fd, 4096)
    except (OSError, ValueError):
        return None
    if not data:
        return None
    pending += data
    return b"\n" in pending


def flush_stdin() -> None:
    """Drop Enters typed earlier (a double Enter at the operator stop must not pass the next prompt at once)."""
    with contextlib.suppress(OSError, ValueError, ImportError):
        if sys.stdin.isatty():
            import termios

            termios.tcflush(sys.stdin, termios.TCIFLUSH)


def load_ready(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """--ready-pose .npz from make_ready_pose.py: the recorded SONIC token (64) and Dex3 hand targets (14) of a ready
    pose, hand over the table (the 10-19 handover recordings return there between attempts)."""
    z = np.load(path)
    token, hands = z["token"].astype(np.float32), z["hands"].astype(np.float32)
    if token.shape != (64,) or hands.shape != (14,):
        raise ValueError(f"{path}: expected token (64,) and hands (14,), got {token.shape} and {hands.shape}")
    return token, hands


def load_ready_path(path: Path) -> dict | None:
    """The optional raise / lower paths of a --ready-pose .npz (make_ready_pose.py --raise-episode), 50 Hz rows:
    raise_tokens/raise_hands (recorded, played forward) and lower_tokens/lower_hands (the reversed joint path,
    re-encoded). None for a pose-only file."""
    z = np.load(path)
    keys = ("raise_tokens", "raise_hands", "lower_tokens", "lower_hands")
    if not all(k in z for k in keys):
        return None
    out = {k: z[k].astype(np.float32) for k in keys}
    for k in ("raise", "lower"):
        t, h = out[f"{k}_tokens"], out[f"{k}_hands"]
        if t.ndim != 2 or t.shape[1] != 64 or h.shape != (len(t), 14) or len(t) < 2:
            raise ValueError(f"{path}: bad {k} path, tokens {t.shape}, hands {h.shape}")
    return out


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
    End: hold the last token and hands until the operator presses Enter (not with --gate-dir), 1 s blend back to the
    rest token, hold (or the table path reversed), then hand back to the planner or stop.
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
        self.pose = False  # POSE mode commanded and not yet handed back: an error must still call finish()
        self.recent = deque(maxlen=25)  # tokens sent in the last 0.5 s: the deploy's echo lags by a few ticks
        # --ready-pose: --trials sessions wait between trials at this pose (hand over the table) instead of handing back
        ready = getattr(a, "ready_pose", None)
        self.ready = load_ready(Path(ready)) if ready else None
        self.ready_path = load_ready_path(Path(ready)) if ready else None
        self.at_ready = False
        self.stance: tuple[np.ndarray, np.ndarray] | None = (
            None  # the take-over rest pose, for the session end
        )

    def send(self, token, hands, phase, t_ep=-1.0, k=-1, chunk_t0=np.nan, joint_ref=None):
        token = np.asarray(token, np.float32)
        step_max = self.a.max_token_step
        if step_max > 0 and self.last_sent is not None:  # slew limit on every phase
            step = token - self.last_sent
            if np.abs(step).max() > step_max:
                token = self.last_sent + np.clip(step, -step_max, step_max)
                self.slewed += 1
        self.last_sent, self.last_hands = token, hands
        self.recent.append(token)
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
        self.connect()
        self.take_over()

    def connect(self):
        """Once per process: start control in PLANNER mode."""
        gate("deploy_ready", self.a.gate_dir)
        time.sleep(0.5)  # let the deploy's SUB connect to our PUB before the first command
        self.pub.send(command_message(start=True, planner=True))
        print("[streamer] command: start control, PLANNER mode", flush=True)

    def take_over(self, trial: int | None = None, n_trials: int | None = None):
        """Per episode: wait until the operator has the robot in position (planner mode), then hand over to POSE
        mode. --trials sessions repeat this after each finish() --end planner hand-back."""
        a, startup = self.a, self.startup
        self.last_sent = None  # no slew limit against the previous trial's last token
        if trial is None:
            gate("settled", a.gate_dir)
        else:
            gate(f"settled.{trial}", a.gate_dir, f"trial {trial + 1}/{n_trials}: ", consume=True)
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
            self.pose = True
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
            self.pose = True
        self.stance = (self.rest_token.copy(), self.rest_hands.copy())
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
        anchor_s = getattr(self.a, "chunk_anchor_s", 0.0) or 0.0
        # anchor only the 64 token dims (the hands are absolute joint targets with no slew limit) and from the token
        # actually sent, which the slew limit in send() can hold behind the resampler's output
        self.resampler.set_chunk(
            chunk, t0=t0, blend_s=blend_s, now=now, anchor_s=anchor_s, anchor_dims=64, current=self.last_sent
        )
        if joints is not None:
            self.jres.set_chunk(joints, t0=t0, blend_s=blend_s, now=now)

    def episode_tick(self, t_ep, k, chunk_t0):
        out = self.resampler.token_at(t_ep)
        joint_ref = None if self.jres is None else self.jres.token_at(t_ep)
        self.send(out[:64], out[64:], "episode", t_ep, k, chunk_t0, joint_ref=joint_ref)

    def _blend_to(self, token, hands, phase, n=50):
        last_token, last_hands = self.last_sent, self.last_hands
        for i in ticks(n):
            w = (i + 1) / n
            self.send((1 - w) * last_token + w * token, (1 - w) * last_hands + w * hands, phase)

    def _play50(self, tokens, hands, phase):
        """A 50 Hz path (one row per tick: the recordings' rate)."""
        for i in ticks(len(tokens)):
            self.send(tokens[i], hands[i], phase)

    def go_ready(self):
        """--ready-pose: 1 s blend from the current command to the ready pose, which then is the rest pose (the next
        trial's hold and blend in start there)."""
        token, hands = self.ready
        last_token, last_hands = self.last_sent, self.last_hands
        for i in ticks(50):
            w = (i + 1) / 50
            self.send((1 - w) * last_token + w * token, (1 - w) * last_hands + w * hands, "to ready")
        self.rest_token, self.rest_hands, self.at_ready = token, hands, True

    @contextlib.contextmanager
    def holding(self):
        """Keep streaming the last command at 50 Hz from a thread while the main thread blocks on the operator (the
        verdict prompt, the next trial's start): POSE mode needs a token every 20 ms. Not recorded (the trial log is
        being written meanwhile); the main thread sends nothing until it ends."""
        stop = threading.Event()
        token, hands = self.last_sent, self.last_hands

        def run():
            for _ in ticks(10**9):
                if stop.is_set():
                    return
                self.pub.send(pose_message(token, hands, self.frame))
                self.frame += 1

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        try:
            yield
        finally:
            stop.set()
            thread.join()

    def hold_for_operator(self, prompt: str | None = None) -> bool:
        """Keep streaming the last token and hands (hands stay closed on a held bottle) until the operator has cleared
        the space and pressed Enter: the blend back to the rest pose sweeps the raised arm down (G1: into the box on
        the table). Without usable stdin, hold until Ctrl-C. --gate-dir runs (sims) do not stop here.
        Returns True for Enter (or a gate-dir run) and False for Ctrl-C: a cancel is no clearance confirmation, so
        callers that would move into the box space (raise or go to the ready pose) must not move on it (PR #48)."""
        if self.a.gate_dir is not None:
            return True
        token, hands = self.last_sent, self.last_hands
        flush_stdin()
        say(
            prompt
            or "[streamer] holding the current pose: clear the space (move the box / take the bottle), then press "
            "Enter to return to the initial pose"
        )
        usable, pending = True, bytearray()
        try:
            for _ in ticks(10**9):
                self.send(token, hands, "hold for operator")
                if usable:
                    pressed = enter_pressed(pending)
                    if pressed:
                        return True
                    if pressed is None:
                        usable = False
                        say("[streamer] stdin unusable: holding the current pose until Ctrl-C")
        except KeyboardInterrupt:
            say("[streamer] Ctrl-C")
        return False

    def raise_to_ready(self, prompt: str):
        """--ready-pose session start: hold until Enter, then raise to the ready pose. Ctrl-C does not raise the arm: it
        raises KeyboardInterrupt into the session cleanup (hold, then stance and planner mode)."""
        if not self.hold_for_operator(prompt):
            raise KeyboardInterrupt
        if (
            self.ready_path is not None
        ):  # along the recorded raise (it cleared the box), not straight to the pose
            path = self.ready_path
            self._blend_to(path["raise_tokens"][0], path["raise_hands"][0], "to raise start")
            self._play50(path["raise_tokens"], path["raise_hands"], "raise")
        self.go_ready()

    def finish(self, to_ready: bool = False):
        """End of an episode. to_ready (--ready-pose, all but the last trial): hold, then go to the ready pose and stay
        in POSE mode. Otherwise (and at the session end): hold, back to the take-over rest pose, hand back."""
        a, startup = self.a, self.startup
        say(f"[streamer] token slew limit active on {self.slewed} ticks so far")
        if to_ready and self.ready is not None:
            if not self.hold_for_operator(
                "[streamer] holding the current pose: press Enter to move to the ready pose (hand over the table)"
            ):
                raise KeyboardInterrupt  # cancelled: the session cleanup holds again, then stance + planner
            self.go_ready()
            return
        lower = None
        if self.at_ready and self.stance is not None:  # session end: from the ready pose down to the stance
            self.rest_token, self.rest_hands = self.stance
            self.at_ready = False
            lower = self.ready_path
        rest_token, rest_hands = self.rest_token, self.rest_hands
        if not self.hold_for_operator():
            say("[streamer] returning to the initial pose")
        if (
            lower is not None
        ):  # retrace the recorded raise downward (re-encoded reverse), then blend to the stance
            self._blend_to(lower["lower_tokens"][0], lower["lower_hands"][0], "to lower start")
            self._play50(lower["lower_tokens"], lower["lower_hands"], "lower")
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
            say("[streamer] command: PLANNER mode (deploy left running)")
        else:
            self.pub.send(command_message(start=False, planner=False))
            say("[streamer] command: stop")
        self.pose = False


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

    def finish(self, to_ready: bool = False):  # to_ready: --ready-pose is sonic backend only
        last = self.last
        for i in ticks(100):  # 2 s back to the measured start pose
            w = (i + 1) / 100
            self.send((1 - w) * last + w * self.rest_ref, "return")
