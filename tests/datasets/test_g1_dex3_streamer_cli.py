import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[2] / "examples" / "g1_dex3_training"))
from sonic_policy_streamer import measured_ref
from sonic_targets import NOMINAL_BODY

HERE = Path(__file__).parents[2] / "examples" / "g1_dex3_training"


def cli(*args):
    return subprocess.run(
        [sys.executable, str(HERE / "sonic_policy_streamer.py"), *args],
        capture_output=True,
        text=True,
        timeout=120,
    )


class StreamerCliTests(unittest.TestCase):
    def test_measured_ref_layout(self):
        body = np.arange(29, dtype=np.float32) / 100
        msg = {"body_q": body, "left_hand_q": np.full(7, 0.5), "right_hand_q": np.full(7, 0.7)}
        ref = measured_ref(msg, "joint31")
        np.testing.assert_allclose(ref[:14], body[15:29])
        np.testing.assert_allclose(ref[14:21], 0.5)
        np.testing.assert_allclose(ref[28:31], body[12:15])
        np.testing.assert_allclose(measured_ref(msg, "joint28")[28:31], NOMINAL_BODY[12:15])

    def test_argument_errors(self):
        cases = [
            (["--replay", "--policy-server", "tcp://x:1", "--dataset-root", "d", "--episode", "0"], "--replay excludes"),
            (["--backend", "decoupled", "--replay", "--dataset-root", "d", "--episode", "0"], "--backend decoupled needs"),
            (["--synthetic-waist", "--action-space", "joint28", "--replay", "--dataset-root", "d", "--episode", "0",
              "--backend", "decoupled"], "--synthetic-waist needs"),
            (["--dataset-root", "d", "--episode", "0"], "exactly one of"),
            (["--backend", "decoupled", "--dex3-right-order", "swap", "--replay", "--dataset-root", "d", "--episode", "0"], "excludes --dex3-right-order"),
        ]  # fmt: skip
        for args, text in cases:
            r = cli(*args)
            self.assertEqual(r.returncode, 2, args)
            self.assertIn(text, r.stderr, args)


if __name__ == "__main__":
    unittest.main()
