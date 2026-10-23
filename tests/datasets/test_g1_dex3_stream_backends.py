"""stream_backends.py: the streamer's SONIC / decoupled start, per-tick send and finish."""

import contextlib
import io
import os
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / "examples" / "g1_dex3_training"))
import stream_backends
from sonic_token_stream import ChunkResampler
from stream_backends import (
    HEADER_SIZE,
    LATENT_INITIAL_MOTION_TOKEN,
    DecoupledBackend,
    SonicBackend,
    max_arm_speed,
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
            b = DecoupledBackend(
                SimpleNamespace(gate_dir=gate_dir, action_space="joint28"), pub, state, rec, 30
            )
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

        rest = measured_ref(state.latest(), "joint28")[:28]
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
    def test_chunk_anchor_follows_the_slew_limited_command_and_leaves_hands_bounded(self):
        """--chunk-anchor-s with the token slew limit active (PR #46 review): a new chunk that asks for less must
        not make the sent token climb, and repeated anchored switches must not push the hands past their range."""
        pub, rec, state = FakePub(), Recorder(), FakeState()
        a = SimpleNamespace(
            handoff_blend_s=0.0,
            start="standing",
            end="planner",
            max_token_step=0.05,
            action_space="sonic78",
            chunk_anchor_s=1.0,
        )
        up = np.ones((40, 78), np.float32)  # token target 1.0
        # every chunk restarts its hand ramp 0 -> 1.0 (inside the +-1.0472 thumb limit)
        up[:, 64:] = np.linspace(0, 1.0, 40)[:, None]
        with backend_env() as a.gate_dir:
            b = SonicBackend(a, pub, state, rec, 30, None)
            b.start()
            b.begin(up, None)  # blends in to the first chunk (1.0)
            far = np.full((40, 78), 5.0, np.float32)
            b.set_chunk(far, None, 0.0, 0.3, 0.0)  # the anchored command now rises faster than 0.05 per tick
            for i in range(1, 11):
                b.episode_tick(i * 0.02, i, 0.0)
            sent = b.last_sent.copy()
            self.assertGreater(b.slewed, 0)  # the limiter held the sent token behind the resampler's output
            b.set_chunk(np.zeros((40, 78), np.float32), None, 0.2, 0.3, 0.2)
            b.episode_tick(0.22, 11, 0.2)
            # each dim heads to the new chunk (0) from what was sent: no dim climbs back toward the old 1.0
            self.assertTrue(np.all(np.abs(b.last_sent) <= np.abs(sent) + 1e-6))
            for k in range(1, 20):  # many anchored switches with valid, restarting hand ramps
                t0 = 0.2 + 0.4 * k
                b.set_chunk(up, None, t0, 0.3, t0 + 0.2)
                for j in range(20):
                    b.episode_tick(t0 + 0.2 + j * 0.02, j, t0)
                    self.assertLessEqual(float(np.max(b.last_hands)), 1.0 + 1e-6)

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

    def test_two_trials_connect_once_and_take_over_per_trial(self):
        pub, rec, state = FakePub(), Recorder(), FakeState()
        a = SimpleNamespace(
            handoff_blend_s=0.2, start="planner", end="planner", max_token_step=0.0, action_space="sonic78"
        )
        chunk = ramp(40, 78, 0.5)
        with backend_env() as a.gate_dir:
            (a.gate_dir / "settled.0").touch()
            (a.gate_dir / "settled.1").touch()
            b = SonicBackend(a, pub, state, rec, 30, None)
            b.connect()
            for trial in (0, 1):
                b.take_over(trial, 2)
                self.assertFalse((a.gate_dir / f"settled.{trial}").exists())  # consumed
                b.begin(chunk, None)
                b.episode_tick(0.02, 0, 0.0)
                b.finish()
        msgs = [decode(m) for m in pub.sent]
        commands = [m[1:] for m in msgs if m[0] == "command"]
        # one start in PLANNER mode; per trial: POSE mode, then back to PLANNER
        self.assertEqual(commands, [(1, 1), (1, 0), (1, 1), (1, 0), (1, 1)])
        phases = [r.phase for r in rec.rows]
        self.assertEqual(phases.count("planner token"), 100)
        self.assertEqual(phases.count("episode"), 2)

    def test_take_over_resets_the_slew_limit_between_trials(self):
        pub, rec, state = FakePub(), Recorder(), FakeState()
        a = SimpleNamespace(
            handoff_blend_s=0.2, start="planner", end="planner", max_token_step=0.05, action_space="sonic78"
        )
        with backend_env() as a.gate_dir:
            (a.gate_dir / "settled.0").touch()
            (a.gate_dir / "settled.1").touch()
            b = SonicBackend(a, pub, state, rec, 30, None)
            b.connect()
            b.take_over(0, 2)
            n = len(rec.rows)
            state.latest = lambda: {**FakeState.latest(state), "token_state": np.full(64, 1.0)}
            b.take_over(1, 2)
        first = next(r for r in rec.rows[n:] if r.phase == "planner token")
        np.testing.assert_allclose(first.token, 1.0)

    def _episode_then_finish(self, polls, rec):
        """One short episode with gates, then finish() as an operator run (no --gate-dir) with enter_pressed()
        answering `polls`. Returns the deploy messages."""
        pub, state = FakePub(), FakeState()
        a = SimpleNamespace(
            handoff_blend_s=0.2, start="planner", end="planner", max_token_step=0.0, action_space="sonic78"
        )
        with backend_env() as a.gate_dir:
            b = SonicBackend(a, pub, state, rec, 30, None)
            b.start()
            b.begin(ramp(40, 78, 0.5), None)
            b.episode_tick(0.4, 12, 0.0)
            a.gate_dir = None
            with mock.patch.object(stream_backends, "enter_pressed", side_effect=polls) as poll:
                b.finish()
        return [decode(m) for m in pub.sent], poll

    def test_finish_holds_the_last_pose_until_enter(self):
        rec = Recorder()
        msgs, poll = self._episode_then_finish([False, False, True], rec)
        self.assertEqual(poll.call_count, 3)
        phases = [r.phase for r in rec.rows]
        i = phases.index("hold for operator")
        self.assertEqual(phases[i - 1], "episode")
        self.assertEqual(phases[i : i + 4], ["hold for operator"] * 3 + ["blend out"])
        last = rec.rows[i - 1]
        # the last episode token and hands (closed on a bottle), not the rest pose
        for r in rec.rows[i : i + 3]:
            np.testing.assert_array_equal(r.token, last.token)
            np.testing.assert_array_equal(r.hands, last.hands)
        self.assertEqual(phases[-1], "handback blend")
        self.assertEqual([m[1:] for m in msgs if m[0] == "command"][-1], (1, 1))  # then PLANNER mode

    def test_finish_without_stdin_holds_until_ctrl_c(self):
        class InterruptAfterFiveHolds(Recorder):
            def __call__(self, token, hands, phase, *rest):
                super().__call__(token, hands, phase, *rest)
                if phase == "hold for operator" and [r.phase for r in self.rows].count(phase) == 5:
                    raise KeyboardInterrupt

        rec = InterruptAfterFiveHolds()
        msgs, poll = self._episode_then_finish([None], rec)
        self.assertEqual(poll.call_count, 1)  # stdin unusable: not polled again
        phases = [r.phase for r in rec.rows]
        self.assertEqual(phases.count("hold for operator"), 5)
        self.assertEqual(phases[-1], "handback blend")
        self.assertEqual([m[1:] for m in msgs if m[0] == "command"][-1], (1, 1))

    def test_enter_pressed_needs_a_whole_line_and_never_blocks(self):
        r, w = os.pipe()
        with os.fdopen(r) as stdin, mock.patch.object(sys, "stdin", stdin):
            pending = bytearray()
            self.assertFalse(stream_backends.enter_pressed(pending))  # nothing typed
            os.write(w, b"x")  # partial line, writer open: returns at once (readline() would block here)
            self.assertFalse(stream_backends.enter_pressed(pending))
            self.assertFalse(stream_backends.enter_pressed(pending))
            os.write(w, b"\n")
            self.assertTrue(stream_backends.enter_pressed(pending))
            os.close(w)
            self.assertIsNone(stream_backends.enter_pressed(bytearray()))  # EOF

    def test_enter_pressed_partial_line_then_eof_is_not_enter(self):
        r, w = os.pipe()
        with os.fdopen(r) as stdin, mock.patch.object(sys, "stdin", stdin):
            pending = bytearray()
            os.write(w, b"x")
            os.close(w)
            self.assertFalse(stream_backends.enter_pressed(pending))
            self.assertIsNone(stream_backends.enter_pressed(pending))


class MaxArmSpeedTests(unittest.TestCase):
    def test_times_by_wall_clock_not_episode_time(self):
        # G1 run48: blend-in ticks are logged at t_ep 0, so 3 ticks (60 ms) of motion ended 20 ms of t_ep after them
        s0, s1 = np.zeros(28), np.zeros(28)
        # 2.4 rad/s on right_wrist_roll over 60 ms; joint 20 is a hand joint, not an arm joint
        s1[11], s1[20] = 0.144, 5.0
        self.assertEqual(max_arm_speed(10.0, s0, 10.06, s1)[0], 11)
        self.assertAlmostEqual(max_arm_speed(10.0, s0, 10.06, s1)[1], 2.4)
        self.assertIsNone(max_arm_speed(10.0, s0, 10.0, s1))  # same tick: no speed, no division by zero


if __name__ == "__main__":
    unittest.main()


@pytest.mark.parametrize("side, n", [("left_hand_q", 6), ("right_hand_q", 6), ("left_hand_q", 8)])
def test_observation_state_rejects_a_hand_without_7_joints(side, n):
    msg = {"body_q": [0.0] * 29, "left_hand_q": [0.0] * 7, "right_hand_q": [0.0] * 7}
    msg[side] = [0.0] * n
    with pytest.raises(ValueError, match="7 joints"):
        stream_backends.observation_state(msg)
