import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[2] / "examples" / "g1_dex3_training"))
from sonic_policy_streamer import frame_index, measured_ref, rtc_args, rtc_prefix
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
    def test_rtc_prefix(self):
        last = np.arange(40 * 3, dtype=np.float32).reshape(1, 40, 3)
        self.assertIsNone(rtc_prefix(None, 12))  # no chunk yet
        self.assertIsNone(rtc_prefix(last, None))  # first chunk of the episode
        self.assertIsNone(rtc_prefix(last, 40))  # whole chunk already played
        tail = rtc_prefix(last, 12)
        self.assertEqual(tail.shape, (1, 28, 3))
        self.assertEqual(tail[0, 0, 0], last[0, 12, 0])  # row 12 of the old chunk is row 0 of the new one

    def test_rtc_args_only_after_an_accepted_chunk(self):
        self.assertEqual(rtc_args(True, True, 1.2, 0.8, 0.22, 30), (12, 7))
        self.assertEqual(
            rtc_args(True, False, 1.2, 0.8, 0.22, 30), ()
        )  # last request rejected: plain request
        self.assertEqual(rtc_args(False, True, 1.2, 0.8, 0.22, 30), ())  # --rtc off

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
            (["--trials", "2", "--policy-server", "tcp://x:1", "--images", "zmq", "--task", "t", "--duration-s", "5"],
             "--trials needs --session-dir"),
            (["--trials", "2", "--session-dir", "s", "--policy-server", "tcp://x:1", "--dataset-root", "d",
              "--episode", "0"], "--trials needs --images zmq"),
        ]  # fmt: skip
        for args, text in cases:
            r = cli(*args)
            self.assertEqual(r.returncode, 2, args)
            self.assertIn(text, r.stderr, args)

    def test_resume_refuses_other_session_settings(self):
        import json
        import tempfile

        from eval_session import write_trial

        with tempfile.TemporaryDirectory() as session:
            log = {"wall": [0.0], "phase": ["episode"]}
            extra = {"episode_end": "operator stop", "verdict": "success", "reason": "", "note": "",
                     "args": json.dumps({"task": "pick bottle", "action_space": "joint28", "policy_server": "tcp://x:1"})}  # fmt: skip
            write_trial(Path(session), 0, log, extra, [])
            r = cli("--trials", "2", "--session-dir", session, "--policy-server", "tcp://x:1", "--images", "zmq",
                    "--task", "close laptop", "--duration-s", "5", "--action-space", "joint31")  # fmt: skip
            self.assertEqual(r.returncode, 2)
            self.assertIn("trial 1 used another task, action_space", r.stderr)


if __name__ == "__main__":
    unittest.main()


def test_frame_index_visits_every_frame_of_a_50hz_replay():
    # t_ep = i * 0.02 (stream_backends.TICK): int(0.58 * 50) is 28, which skipped frame 29 (scorer coverage 97.9%)
    assert [frame_index(i * 0.02, 50, 475) for i in range(475)] == list(range(475))
    assert [frame_index(i * 0.02, 30, 300) for i in range(0, 500, 5)] == list(range(0, 300, 3))
    assert frame_index(20.0, 50, 475) == 474 and frame_index(-0.1, 50, 475) == 0
