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
    check_synthetic_duration,
    log_settings,
    pack_joint_message,
    read_done,
    replay_chunk,
    rpy_from_matrix,
    run_validity,
    synthetic_waist,
    to_joint_ref,
    unpack_joint_message,
    write_done,
)


def rot(roll, pitch, yaw):
    cr, sr, cp, sp, cy, sy = (
        np.cos(roll),
        np.sin(roll),
        np.cos(pitch),
        np.sin(pitch),
        np.cos(yaw),
        np.sin(yaw),
    )
    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return rz @ ry @ rx


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

    def test_synthetic_waist_track(self):
        t = np.arange(0, 20, 0.02)
        w = synthetic_waist(t)
        self.assertEqual(w.shape, (len(t), 3))
        for tt in (0.0, 5.0, 10.0, 16.0, 19.0):
            np.testing.assert_allclose(synthetic_waist([tt])[0], 0, atol=1e-9)
        self.assertAlmostEqual(synthetic_waist([1.25])[0, 0], 0.4, places=6)
        self.assertAlmostEqual(synthetic_waist([6.25])[0, 1], 0.15, places=6)
        self.assertAlmostEqual(synthetic_waist([11.0])[0, 2], 0.15, places=6)
        self.assertAlmostEqual(synthetic_waist([13.0])[0, 2], 0.3, places=6)
        self.assertLess(np.abs(np.diff(w, axis=0)).max(), 0.02)  # smooth at 50 Hz (no jumps)
        self.assertTrue(np.all((np.abs(w) > 1e-9).sum(1) <= 1))  # one axis at a time

    def test_startup_checks(self):
        check_replay_width(28, "joint28", False)
        check_replay_width(28, "joint31", True)
        check_replay_width(78, "sonic78", False)
        with self.assertRaises(ValueError):
            check_replay_width(78, "joint28", False)
        with self.assertRaises(ValueError):
            check_replay_width(28, "joint31", False)
        check_synthetic_duration(16.0)
        with self.assertRaises(ValueError):
            check_synthetic_duration(14.6)

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

    def test_rpy_from_matrix(self):
        np.testing.assert_allclose(rpy_from_matrix(rot(0.1, -0.2, 0.3)), [0.1, -0.2, 0.3], atol=1e-9)

    def test_log_settings_defaults_for_old_logs(self):
        self.assertEqual(log_settings(json.dumps({"action_space": "sonic78"})), ("sonic", "sonic78"))
        self.assertEqual(log_settings(json.dumps({})), ("sonic", "sonic78"))
        self.assertEqual(
            log_settings(json.dumps({"backend": "decoupled", "action_space": "joint31"})),
            ("decoupled", "joint31"),
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

    def test_fall_detector_latches_simulator_reset(self):
        f = FallDetector()
        f.latch("pelvis below 0.2 m")  # not armed: ignored (band hang, startup reset)
        f.armed = True
        self.assertIsNone(f.update(0.0, [1, 0, 0, 0], 0.79, 8))
        f.latch("pelvis below 0.2 m")
        # the robot is upright again after the simulator's reset; the latched fall still ends the run
        self.assertEqual(f.update(0.005, [1, 0, 0, 0], 0.79, 8), "pelvis below 0.2 m")


if __name__ == "__main__":
    unittest.main()
