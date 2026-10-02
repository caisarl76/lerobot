import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2] / "examples" / "g1_dex3_training"))
from wbc_compare import g0, g1, g2, syn


def r(orig50=3.0, orig95=8.0, ref95=8.0, tilt=3.0, hits=0, valid=True):
    return {"valid": valid, "palm_err_vs_original_cm": {"p50": orig50, "p95": orig95},
            "palm_err_vs_joint_ref_cm": {"p95": ref95}, "stability": {"tilt_max_deg": tilt}, "table": {"hits_max": hits}}  # fmt: skip


class CompareTests(unittest.TestCase):
    def test_g0_uses_spread_or_floor(self):
        runs = {"Astored": {1: [r(3.0), r(3.2), r(3.1)]}, "A28": {1: [r(3.4)] * 3}}
        self.assertTrue(g0(runs)["pass"])  # |3.4 - 3.1| = 0.3 <= 0.5
        runs["A28"] = {1: [r(3.8)] * 3}
        self.assertFalse(g0(runs)["pass"])  # 0.7 > max(0.5, 0.2)

    def test_g1_balance(self):
        runs = {"A28": {1: [r(tilt=3.0, hits=1)] * 3}, "C28": {1: [r(tilt=4.5, hits=1)] * 3}}
        self.assertTrue(g1(runs, "C28", "A28")["pass"])
        runs["C28"] = {1: [r(tilt=4.5, hits=1)] * 2 + [r(valid=False)]}
        self.assertFalse(g1(runs, "C28", "A28")["pass"])  # an invalid repeat fails G1
        runs["C28"] = {1: [r(tilt=5.5, hits=1)] * 3}
        self.assertFalse(g1(runs, "C28", "A28")["pass"])

    def test_g2_needs_mean_gain_and_four_of_six(self):
        a = {e: [r(ref95=10.0)] * 3 for e in range(6)}
        c = {e: [r(ref95=8.5 if e < 4 else 10.5)] * 3 for e in range(6)}
        self.assertFalse(g2({"A28": a, "C28": c})["pass"])  # mean gain 0.83 < 1.0
        c = {e: [r(ref95=8.0 if e < 4 else 10.2)] * 3 for e in range(6)}
        self.assertTrue(g2({"A28": a, "C28": c})["pass"])  # gain 1.27, 4/6 better

    def test_syn_within_two_cm_of_28d(self):
        runs = {"C28": {1: [r(ref95=8.0)] * 3}, "C31syn": {1: [r(ref95=9.5)] * 3}}
        self.assertTrue(syn(runs, "C31syn", "C28")["pass"])
        runs["C31syn"] = {1: [r(ref95=10.5)] * 3}
        self.assertFalse(syn(runs, "C31syn", "C28")["pass"])


if __name__ == "__main__":
    unittest.main()
