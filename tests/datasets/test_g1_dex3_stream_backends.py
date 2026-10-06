"""stream_backends.py: the streamer's SONIC / decoupled start, per-tick send and finish."""

import contextlib
import io
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[2] / "examples" / "g1_dex3_training"))
import stream_backends
from sonic_token_stream import ChunkResampler
from stream_backends import (
    HEADER_SIZE,
    LATENT_INITIAL_MOTION_TOKEN,
    DecoupledBackend,
    SonicBackend,
    measured_ref,
)
from wbc_common import unpack_joint_message


class FakePub:
    def __init__(self):
        self.sent = []

    def send(self, raw):
        self.sent.append(bytes(raw))


class FakeState:
    """A fixed g1_debug message."""

    def latest(self):
        body = np.zeros(29)
        body[15:29] = 0.1 + np.arange(14) * 0.001
        return {
            "body_q": body,
            "left_hand_q": np.full(7, 0.2),
            "right_hand_q": np.arange(7) * 0.01,
            "token_state": np.linspace(-0.2, 0.2, 64),
        }


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
        pub, rec, state = FakePub(), Recorder(), FakeState()
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
        pub, rec, state = FakePub(), Recorder(), FakeState()
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


if __name__ == "__main__":
    unittest.main()
