import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[2] / "examples" / "g1_dex3_training"))
from wbc_common import FallDetector, read_done, write_done


class WbcCommonTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
