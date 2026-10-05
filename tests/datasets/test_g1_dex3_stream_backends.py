"""stream_backends.py (the streamer's SONIC / decoupled start, per-tick send and finish), and golden equivalence of
sonic_policy_streamer.main() with the streamer before the restructure (fixtures_g1/sonic_policy_streamer_164fda2a.py).
"""

import contextlib
import importlib.util
import io
import itertools
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[2] / "examples" / "g1_dex3_training"))
import sonic_policy_streamer as new_streamer
import stream_backends
from sonic_token_stream import ChunkResampler
from stream_backends import (
    HEADER_SIZE,
    LATENT_INITIAL_MOTION_TOKEN,
    RIGHT_ORDER,
    DecoupledBackend,
    SonicBackend,
    measured_ref,
)
from wbc_common import unpack_joint_message

_spec = importlib.util.spec_from_file_location(
    "streamer_164fda2a", Path(__file__).parent / "fixtures_g1" / "sonic_policy_streamer_164fda2a.py"
)
old_streamer = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(old_streamer)


class FakePub:
    def __init__(self):
        self.sent = []

    def bind(self, addr):
        pass

    def send(self, raw):
        self.sent.append(bytes(raw))


class FakeState:
    """g1_debug messages whose arm and hand values drift with every read (so the number and order of reads matter)."""

    def __init__(self, drift=1e-4):
        self.n, self.drift = 0, drift

    def latest(self):
        self.n += 1
        d = self.n * self.drift
        body = np.zeros(29)
        body[15:29] = 0.1 + d + np.arange(14) * 0.001
        return {
            "body_q": body,
            "left_hand_q": np.full(7, 0.2 + d),
            "right_hand_q": np.arange(7) * 0.01 + d,
            "token_state": np.linspace(-0.2, 0.2, 64),
        }

    def age(self):
        return 0.0


class LateEvent:
    """threading.Event stand-in that reports done from its `delay`-th check on (a chunk that takes a few ticks)."""

    def __init__(self, delay):
        self.left = delay

    def set(self):
        pass

    def is_set(self):
        self.left -= 1
        return self.left <= 0


class SyncWorker:
    """InferenceWorker stand-in: runs the job at submit; the slot reports done on its 3rd check."""

    def __init__(self):
        self.slot = None

    def submit(self, fn, *args):
        self.slot = {"done": LateEvent(3), "latency": 0.125}
        try:
            self.slot["result"] = fn(*args)
        except Exception as exc:
            self.slot["error"] = repr(exc)
        return self.slot

    def run(self, fn, *args):
        return fn(*args)

    @property
    def busy(self):
        return self.slot is not None and not self.slot["done"].is_set()


def fake_images(actions):
    class Images:
        fps = 30

        def __init__(self, root, episode, keys):
            self.n = len(actions)

        def frame(self, t):
            k = min(max(int(t * self.fps), 0), self.n - 1)
            return k, {}, np.zeros(28, np.float32), "task"

        def actions(self):
            return actions.copy()

    return Images


def fake_encode(joints, limits, encoder, fps):
    tokens = 0.5 * np.tanh(joints[:, [0]] + np.linspace(-1, 1, 64))
    return np.hstack([tokens, joints[:, 14:28]]).astype(np.float32)


def episode_actions(width, n=45, nan_from=None, phase=0.0):
    t = np.arange(n)[:, None] / 30
    out = (0.3 * np.sin(2 * t + np.arange(width) * 0.1 + phase)).astype(np.float32)
    if nan_from is not None:
        out[nan_from:] = np.nan
    return out


def fake_policy(width):
    class Policy:
        """ChunkPolicy stand-in on two observations: the chunk depends on the robot states it is given."""

        image_keys, shapes, n_obs = [], {}, 2

        def __init__(self, *args):
            pass

        def chunk(self, states, images, task):
            return episode_actions(
                width, n=40, phase=100 * float(states[-1][:14].sum() - states[0][:14].sum())
            )

    return Policy


def run_main(mod, argv, actions, tmp):
    """mod.main() on a replayed episode with fake sockets and clocks: (sent messages, log arrays, stdout, done)."""
    pub, wall = FakePub(), itertools.count()
    fake_zmq = SimpleNamespace(Context=lambda: SimpleNamespace(socket=lambda kind: pub), PUB=1)
    patches = [
        mock.patch.object(mod, "zmq", fake_zmq, create=True),
        mock.patch.object(mod, "StateSubscriber", lambda *_: FakeState()),
        mock.patch.object(mod, "DatasetImages", fake_images(actions)),
        mock.patch.object(mod, "InferenceWorker", SyncWorker),
        mock.patch.object(mod, "ChunkPolicy", fake_policy(actions.shape[1])),
        mock.patch.object(mod, "require_package", lambda *a, **k: None),
        mock.patch.object(mod, "load_joint_limits", lambda path: None),
        mock.patch.object(mod, "SonicEncoder", lambda *a: None),
        mock.patch.object(mod, "joint_chunk_to_sonic", fake_encode),
        mock.patch("time.time", lambda: 1.9e9 + next(wall) * 0.001),
        mock.patch("time.sleep", lambda s: None),
        mock.patch("time.monotonic", lambda: 0.0),
    ]
    gate_dir = tmp / "gate"
    gate_dir.mkdir(exist_ok=True)
    for name in ("deploy_ready", "settled"):
        (gate_dir / name).touch()
    (gate_dir / "done").unlink(missing_ok=True)
    common = ["--dataset-root", "d", "--episode", "3", "--gate-dir", str(gate_dir)]
    common += [] if "--policy-path" in argv else ["--replay"]
    argv = ["streamer", *common, "--log", str(tmp / "log.npz"), *argv]
    out = io.StringIO()
    with contextlib.ExitStack() as stack:
        for p in [*patches, mock.patch.object(sys, "argv", argv), contextlib.redirect_stdout(out)]:
            stack.enter_context(p)
        try:
            mod.main()
        finally:
            RIGHT_ORDER[:] = np.arange(7)  # --dex3-right-order swap changes the module-level order in place
            old_streamer.RIGHT_ORDER[:] = np.arange(7)
    with np.load(tmp / "log.npz") as z:
        log = {k: z[k] for k in z.files}
    return pub.sent, log, out.getvalue(), (gate_dir / "done").read_text()


class Recorder:
    """The streamer's record() stand-in: one row per tick."""

    def __init__(self):
        self.rows = []

    def __call__(self, token, hands, phase, t_ep, k, chunk_t0, joint_ref):
        self.rows.append(
            SimpleNamespace(token=token, hands=hands, phase=phase, t_ep=t_ep, k=k, joint_ref=joint_ref)
        )


@contextlib.contextmanager
def backend_env():
    """Gate flag files present and no sleeping (ticks and the deploy connect pause): yields the gate dir."""
    with tempfile.TemporaryDirectory() as d, mock.patch.object(stream_backends.time, "sleep", lambda s: None):
        for name in ("deploy_ready", "settled"):
            (Path(d) / name).touch()
        with contextlib.redirect_stdout(io.StringIO()):
            yield Path(d)


def decode(raw):
    """('command', start, planner) or ('pose', token[64], frame_index, hands[14]) of a deploy message."""
    if raw.startswith(b"command"):
        start, _, planner = struct.unpack("BBB", raw[len("command") + HEADER_SIZE :])
        return "command", start, planner
    body = raw[len("pose") + HEADER_SIZE :]
    return (
        "pose",
        np.frombuffer(body[:256], "<f4"),
        struct.unpack("<q", body[256:264])[0],
        np.frombuffer(body[264:], "<f4"),
    )


def ramp(rows, width, scale):
    return (scale * (np.arange(rows)[:, None] + np.linspace(0, 1, width)) / rows).astype(np.float32)


def resampled(chunk, t0, t):
    r = ChunkResampler(30)
    r.set_chunk(chunk, t0=t0)
    return r.token_at(t)


class DecoupledBackendTests(unittest.TestCase):
    def test_phases_and_joint_targets(self):
        pub, rec, state = FakePub(), Recorder(), FakeState(drift=0.0)
        first, second = ramp(40, 31, 0.5), ramp(40, 31, -0.5)
        with backend_env() as gate_dir:
            b = DecoupledBackend(SimpleNamespace(gate_dir=gate_dir), pub, state, rec, 30)
            b.start()
            for _ in range(3):
                b.wait_tick()
            b.begin(None, first)
            b.set_chunk(None, second, 0.4, 0.0, 0.4)
            for i in range(5):
                b.episode_tick(0.4 + i * 0.02, 12 + i, 0.4)
            b.finish()
        msgs = [unpack_joint_message(m) for m in pub.sent]
        expected = (
            ["wait first chunk"] * 3 + ["leadin"] * 100 + ["hold"] * 50 + ["episode"] * 5 + ["return"] * 100
        )
        self.assertEqual([m["phase"] for m in msgs], expected)
        self.assertEqual([r.phase for r in rec.rows], expected)
        self.assertEqual([m["seq"] for m in msgs], list(range(len(msgs))))
        self.assertEqual([m["frame"] for m in msgs[153:158]], [12, 13, 14, 15, 16])
        self.assertTrue(all(m["frame"] == -1 for m in msgs[:153] + msgs[158:]))

        def ref(m):
            return np.r_[m["q_body"][15:29], m["q_hand"]]

        rest = measured_ref(state.latest())[:28]
        np.testing.assert_allclose(ref(msgs[0]), rest, atol=1e-6)  # wait: the measured pose
        np.testing.assert_allclose(
            ref(msgs[3]), 0.99 * rest + 0.01 * first[0, :28], atol=1e-6
        )  # leadin starts there
        np.testing.assert_allclose(
            ref(msgs[102]), first[0, :28], atol=1e-6
        )  # and ends at the first chunk row
        for m in msgs[103:153]:
            np.testing.assert_allclose(ref(m), first[0, :28], atol=1e-6)
        for i, m in enumerate(msgs[153:158]):
            np.testing.assert_allclose(ref(m), resampled(second, 0.4, 0.4 + i * 0.02)[:28], atol=1e-6)
        np.testing.assert_allclose(ref(msgs[-1]), rest, atol=1e-6)  # back to the measured start pose
        self.assertTrue(all(np.isnan(r.token).all() for r in rec.rows))
        np.testing.assert_allclose(rec.rows[0].joint_ref[:28], rest, atol=1e-6)


class SonicBackendTests(unittest.TestCase):
    def test_commands_handoff_blend_in_episode_and_handback(self):
        pub, rec, state = FakePub(), Recorder(), FakeState(drift=0.0)
        a = SimpleNamespace(
            handoff_blend_s=1.5, start="standing", end="planner", max_token_step=0.0, action_space="joint28"
        )
        first, second = ramp(40, 78, 0.5), ramp(40, 78, -0.5)
        joints = ramp(40, 31, 0.3)
        with backend_env() as a.gate_dir:
            b = SonicBackend(a, pub, state, rec, 30, None)
            b.start()
            for _ in range(3):
                b.wait_tick()
            b.begin(first, joints)
            b.set_chunk(second, joints, 0.4, 0.0, 0.4)
            for i in range(5):
                b.episode_tick(0.4 + i * 0.02, 12 + i, 0.4)
            b.finish()
        msgs = [decode(m) for m in pub.sent]
        commands = [(i, m[1:]) for i, m in enumerate(msgs) if m[0] == "command"]
        # start + PLANNER, 50 planner-token ticks, POSE mode; at the end blend back and hand back to PLANNER
        self.assertEqual(commands, [(0, (1, 1)), (51, (1, 0)), (len(msgs) - 1, (1, 1))])
        poses = [m for m in msgs if m[0] == "pose"]
        self.assertEqual([m[2] for m in poses], list(range(len(poses))))
        phases = (
            ["planner token"] * 50 + ["handoff blend"] * 75 + ["wait first chunk"] * 3 + ["blend in"] * 50
        )
        phases += ["episode"] * 5 + ["blend out"] * 50 + ["hold rest"] * 100 + ["handback blend"] * 75
        self.assertEqual([r.phase for r in rec.rows], phases)
        planner = state.latest()["token_state"].astype(np.float32)
        np.testing.assert_allclose(poses[0][1], planner)
        np.testing.assert_allclose(
            poses[124][1], LATENT_INITIAL_MOTION_TOKEN, atol=1e-7
        )  # handoff ends on rest
        np.testing.assert_allclose(poses[125][1], LATENT_INITIAL_MOTION_TOKEN)  # wait first chunk
        np.testing.assert_allclose(poses[125][3], 0)
        np.testing.assert_allclose(
            poses[128][1], 0.98 * LATENT_INITIAL_MOTION_TOKEN + 0.02 * first[0, :64], atol=1e-6
        )
        np.testing.assert_allclose(
            poses[177][1], first[0, :64], atol=1e-6
        )  # blend in ends on the first chunk
        np.testing.assert_allclose(poses[177][3], first[0, 64:], atol=1e-6)
        self.assertEqual((rec.rows[128].t_ep, rec.rows[128].k), (0.0, 0))
        for i in range(5):  # episode: the token resampler at t, joint_ref from the joint resampler
            out = resampled(second, 0.4, 0.4 + i * 0.02)
            np.testing.assert_allclose(poses[178 + i][1], out[:64], atol=1e-6)
            np.testing.assert_allclose(poses[178 + i][3], out[64:], atol=1e-6)
            np.testing.assert_allclose(
                rec.rows[178 + i].joint_ref, resampled(joints, 0.4, 0.4 + i * 0.02), atol=1e-6
            )
            self.assertEqual(rec.rows[178 + i].k, 12 + i)
        np.testing.assert_allclose(poses[-1][1], planner, atol=1e-6)  # handback ends on the planner token
        np.testing.assert_allclose(poses[-1][3], 0, atol=1e-6)


JOINT28 = ["--action-space", "joint28"]
ENCODER = ["--encoder-model", "e", "--observation-config", "o", "--robot-xml", "x"]
POLICY = ["--policy-path", "p", "--noise-seed", "0"]
GOLDEN_CASES = {
    "decoupled policy": (["--backend", "decoupled", *JOINT28, *POLICY], 28, None),
    "sonic policy": (POLICY, 78, None),
    "sonic joint28 policy": ([*JOINT28, *ENCODER, *POLICY, "--chunk-blend-s", "0.1"], 28, None),
    "decoupled": (["--backend", "decoupled", *JOINT28], 28, None),
    "decoupled chunk blend": (["--backend", "decoupled", *JOINT28, "--chunk-blend-s", "0.2"], 28, None),
    "decoupled stale chunk": (["--backend", "decoupled", *JOINT28, "--max-chunk-age-s", "0.5"], 28, 41),
    "sonic standing": ([], 78, None),
    "sonic start planner": (["--start", "planner"], 78, None),
    "sonic no handoff, end stop": (["--handoff-blend-s", "0", "--end", "stop"], 78, None),
    "sonic slew + chunk blend": (["--max-token-step", "0.05", "--chunk-blend-s", "0.2"], 78, None),
    "sonic table startup": (["--startup-tokens", "STARTUP"], 78, None),
    "sonic right swap": (["--dex3-right-order", "swap"], 78, None),
    "sonic joint28": ([*JOINT28, *ENCODER], 28, None),
    "sonic joint28 stale chunk": ([*JOINT28, *ENCODER, "--max-chunk-age-s", "0.5"], 28, 41),
}


class GoldenEquivalenceTests(unittest.TestCase):
    """The restructured main() sends the same bytes, logs the same arrays and prints the same lines as 164fda2a."""

    def test_same_messages_logs_and_output(self):
        for name, (argv, width, nan_from) in GOLDEN_CASES.items():
            with self.subTest(name), tempfile.TemporaryDirectory() as d:
                tmp = Path(d)
                if "STARTUP" in argv:
                    path = tmp / "startup.npz"
                    toks = np.linspace(0, 0.3, 6 * 64, dtype=np.float32).reshape(6, 64)
                    hands = np.linspace(0, 0.2, 6 * 14, dtype=np.float32).reshape(6, 14)
                    meta = json.dumps({"fps": 30})
                    np.savez(
                        path,
                        tokens=toks,
                        hands=hands,
                        rev_tokens=toks[::-1],
                        rev_hands=hands[::-1],
                        meta=meta,
                    )
                    argv = [str(path) if x == "STARTUP" else x for x in argv]
                actions = episode_actions(width, nan_from=nan_from)
                old = run_main(old_streamer, argv, actions, tmp)
                new = run_main(new_streamer, argv, actions, tmp)
                self.assertGreater(len(old[0]), 200)
                self.assertEqual(len(new[0]), len(old[0]))
                for i, (a, b) in enumerate(zip(old[0], new[0], strict=True)):
                    self.assertEqual(a, b, f"message {i}")
                self.assertEqual(sorted(new[1]), sorted(old[1]))
                for key in old[1]:
                    np.testing.assert_array_equal(new[1][key], old[1][key], err_msg=key)
                self.assertEqual(new[2], old[2])
                self.assertEqual(new[3], old[3])
                if nan_from is not None:
                    self.assertIn("no valid chunk", str(old[1]["episode_end"]))


if __name__ == "__main__":
    unittest.main()
