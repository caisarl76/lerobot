import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[2] / "examples" / "g1_dex3_training"))
import sonic_stream_eval


class StreamEvalValidityTest(unittest.TestCase):
    def test_missing_termination_is_invalid(self):
        stub = types.ModuleType("sonic_roundtrip_audit")
        stub.v3_meta = lambda src: ([{"episode_index": 3, "tasks": ["t"]}], {})
        stub.v3_episode = lambda *a, **k: {"action": np.zeros((10, 14))}
        stub.Robot = None  # must not be reached for an invalid run
        saved = sys.modules.get("sonic_roundtrip_audit")
        sys.modules["sonic_roundtrip_audit"] = stub
        try:
            with tempfile.TemporaryDirectory() as d:
                run = Path(d)
                (run / "sim").mkdir()
                np.savez(
                    run / "streamer.npz",
                    args=json.dumps({"backend": "decoupled", "action_space": "joint28"}),
                    phase=np.array(["episode"] * 10),
                    frame=np.arange(10),
                    wall=np.linspace(1.0, 2.0, 10),
                )
                np.savez(run / "sim" / "sim_state.npz", wall=np.linspace(0.0, 3.0, 50))
                out = sonic_stream_eval.score(run, "src", "conv", 3)
        finally:
            if saved is None:
                del sys.modules["sonic_roundtrip_audit"]
            else:
                sys.modules["sonic_roundtrip_audit"] = saved
        self.assertFalse(out["valid"])
        self.assertEqual(out["reason"], "no termination.json")
        self.assertEqual(out["backend"], "decoupled")


if __name__ == "__main__":
    unittest.main()
