import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[2] / "examples" / "g1_dex3_training"))
from sonic_targets import NOMINAL_BODY
from wbc_common import (
    REF_NAMES,
    FallDetector,
    check_replay_width,
    log_settings,
    pack_joint_message,
    read_done,
    replay_chunk,
    run_validity,
    to_joint_ref,
    unpack_joint_message,
    write_done,
)


class WbcCommonTests(unittest.TestCase):
    def test_joint_ref_pads_nominal_waist_and_passes_31d(self):
        c28 = np.arange(56, dtype=np.float32).reshape(2, 28)
        ref = to_joint_ref(c28)
        self.assertEqual(ref.shape, (2, 31))
        np.testing.assert_array_equal(ref[:, :28], c28)
        np.testing.assert_allclose(ref[:, 28:], np.tile(NOMINAL_BODY[12:15], (2, 1)))
        c31 = np.ones((3, 31), np.float32)
        np.testing.assert_array_equal(to_joint_ref(c31), c31)
        with self.assertRaises(ValueError):
            to_joint_ref(np.zeros((2, 30)))
        self.assertEqual(len(REF_NAMES), 31)

    def test_replay_chunk_pads_with_last_row(self):
        acts = np.arange(10, dtype=np.float32)[:, None] * np.ones((1, 28), np.float32)
        out = replay_chunk(acts, 7, 5)
        self.assertEqual(out.shape, (5, 28))
        np.testing.assert_array_equal(out[:, 0], [7, 8, 9, 9, 9])
        with self.assertRaises(ValueError):
            replay_chunk(acts, 10, 5)

    def test_startup_checks(self):
        check_replay_width(28, "joint28")
        check_replay_width(78, "sonic78")
        with self.assertRaises(ValueError):
            check_replay_width(78, "joint28")
        with self.assertRaises(ValueError):
            check_replay_width(28, "sonic78")

    @unittest.skipUnless(importlib.util.find_spec("msgpack"), "msgpack not installed")
    def test_joint_message_round_trip(self):
        ref = np.linspace(-1, 1, 31).astype(np.float32)
        msg = unpack_joint_message(pack_joint_message(7, "episode", 12, ref, t_wall=123.5))
        self.assertEqual((msg["seq"], msg["phase"], msg["frame"], msg["t_wall"]), (7, "episode", 12, 123.5))
        np.testing.assert_allclose(msg["q_body"][15:29], ref[:14], atol=1e-7)
        np.testing.assert_allclose(msg["q_body"][12:15], ref[28:31], atol=1e-7)
        np.testing.assert_allclose(msg["q_body"][:12], NOMINAL_BODY[:12], atol=1e-7)
        np.testing.assert_allclose(msg["q_hand"], ref[14:28], atol=1e-7)
        with self.assertRaises(ValueError):
            pack_joint_message(0, "x", -1, np.full(31, np.nan))
        with self.assertRaises(ValueError):
            unpack_joint_message(b"pose" + b"\x00")

    def test_done_gate_round_trip(self):
        with tempfile.TemporaryDirectory() as d:
            gate = Path(d)
            self.assertIsNone(read_done(gate))
            write_done(gate, "watchdog: arm joint 3 measured at 7.0 rad/s > 6")
            self.assertTrue(read_done(gate).startswith("watchdog"))
            (gate / "done").write_text("")
            self.assertEqual(read_done(gate), "unknown")

    def test_write_done_is_atomic(self):
        with tempfile.TemporaryDirectory() as d:
            write_done(Path(d), "completed")
            self.assertFalse((Path(d) / "done.tmp").exists())
            self.assertEqual(read_done(Path(d)), "completed")

    def test_fall_detector(self):
        f = FallDetector()
        tilted = [np.cos(np.radians(15)), np.sin(np.radians(15)), 0, 0]  # 30 deg tilt about x
        self.assertIsNone(f.update(0.0, tilted, 0.7, 8))  # not armed
        f.armed = True
        self.assertIsNone(f.update(1.0, tilted, 0.7, 8))
        self.assertIsNone(f.update(1.1, [1, 0, 0, 0], 0.7, 8))  # transient, reset
        self.assertIsNone(f.update(1.2, tilted, 0.7, 8))
        self.assertIsNotNone(f.update(1.41, tilted, 0.7, 8))
        g = FallDetector()
        g.armed = True
        g.update(0.0, [1, 0, 0, 0], 0.7, 0)
        self.assertIsNotNone(g.update(0.25, [1, 0, 0, 0], 0.7, 0))  # both feet off the floor

    def test_log_settings_defaults_for_old_logs(self):
        self.assertEqual(log_settings(json.dumps({"action_space": "sonic78"})), ("sonic", "sonic78"))
        self.assertEqual(log_settings(json.dumps({})), ("sonic", "sonic78"))
        self.assertEqual(
            log_settings(json.dumps({"backend": "decoupled", "action_space": "joint28"})),
            ("decoupled", "joint28"),
        )

    def test_run_validity(self):
        phase = np.array(["blend in"] + ["episode"] * 100 + ["return"])
        frame = np.r_[-1, np.arange(100), -1]
        wall = np.arange(102) * 0.02 + 10
        sim_wall = np.arange(0, 200) * 0.02 + 9
        ok = {"reason": "completed"}
        self.assertEqual(run_validity(ok, phase, frame, wall, sim_wall, 100), (True, "ok"))
        self.assertFalse(run_validity(None, phase, frame, wall, sim_wall, 100)[0])
        self.assertFalse(
            run_validity({"reason": "fell", "detail": "tilt"}, phase, frame, wall, sim_wall, 100)[0]
        )
        self.assertFalse(run_validity({"reason": "aborted"}, phase, frame, wall, sim_wall, 100)[0])
        self.assertFalse(run_validity(ok, phase, frame, wall, sim_wall, 110)[0])  # 100/110 < 98 %
        self.assertFalse(run_validity(ok, phase, frame, wall, sim_wall[:50], 100)[0])  # tail past the sim end
        self.assertFalse(run_validity(ok, phase, frame, wall, sim_wall, 100, held_fraction=0.03)[0])
        self.assertTrue(run_validity(ok, phase, frame, wall, sim_wall, 100, held_fraction=0.01)[0])


if __name__ == "__main__":
    unittest.main()
