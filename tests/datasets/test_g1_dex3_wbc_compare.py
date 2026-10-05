import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2] / "examples" / "g1_dex3_training"))
from wbc_compare import g0, g1, g2, load, only, syn


def r(orig50=3.0, orig95=8.0, ref95=8.0, tilt=3.0, hits=0, valid=True, wrist=5.0):
    return {"valid": valid, "palm_err_vs_original_cm": {"p50": orig50, "p95": orig95},
            "palm_err_vs_joint_ref_cm": {"p95": ref95}, "stability": {"tilt_max_deg": tilt}, "table": {"hit_records": hits},
            "wrist_orientation_err_deg": {"p95": wrist}}  # fmt: skip


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

    def test_g1_missing_repeats_or_episodes_fail(self):
        a = {1: [r()] * 3, 2: [r()] * 3}
        self.assertFalse(g1({"A28": a, "C28": {1: [r()], 2: []}}, "C28", "A28")["pass"])
        self.assertFalse(g1({"A28": a, "C28": {1: [r()] * 3}}, "C28", "A28")["pass"])  # episode 2 missing
        self.assertTrue(g1({"A28": a, "C28": {1: [r()] * 3, 2: [r()] * 3}}, "C28", "A28")["pass"])

    def test_g1_table_contacts(self):
        runs = {"A28": {1: [r(hits=1)] * 3}, "C28": {1: [r(hits=2)] * 3}}
        self.assertFalse(g1(runs, "C28", "A28")["pass"])

    def test_g2_needs_four_better(self):
        a = {e: [r(ref95=10.0)] * 3 for e in range(6)}
        c = {e: [r(ref95=2.0 if e < 3 else 10.5)] * 3 for e in range(6)}
        out = g2({"A28": a, "C28": c})
        self.assertFalse(out["pass"])  # gain 3.75 but only 3/6 better
        self.assertEqual(out["better"], 3)
        self.assertEqual(out["rows"][0]["c_wrist_deg_p95"], 5.0)

    def test_syn_missing_fail_and_torso(self):
        base = {1: [r()] * 3}
        self.assertFalse(syn({"C28": base, "C31syn": {1: [r()] * 2}}, "C31syn", "C28")["pass"])
        self.assertFalse(syn({"C28": base, "C31syn": {1: [r()] * 3, 2: [r()] * 3}}, "C31syn", "C28")["pass"])
        t = r()
        t["torso_err_rad"] = {"roll": {"p95": 0.1}}
        row = syn({"C28": base, "C31syn": {1: [t] * 3}}, "C31syn", "C28")["rows"][0]
        self.assertEqual((row["torso_roll_rad_p95"], row["torso_yaw_rad_p95"]), (0.1, None))

    def test_load_counts_dir_without_json_as_invalid(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "WBC_A28_ep1_r1").mkdir()
            (Path(d) / "WBC_A28_ep1_r2").mkdir()
            (Path(d) / "WBC_A28_ep1_r2" / "stream_eval.json").write_text(json.dumps(r()))
            runs = load(d)
        self.assertEqual([x["valid"] for x in runs["A28"][1]], [False, True])

    def test_load_counts_unreadable_json_as_invalid(self):
        with tempfile.TemporaryDirectory() as d:
            for i, text in ((1, ""), (2, "{trunc")):
                (Path(d) / f"WBC_A28_ep1_r{i}").mkdir()
                (Path(d) / f"WBC_A28_ep1_r{i}" / "stream_eval.json").write_text(text)
            runs = load(d)
        self.assertEqual([x["reason"] for x in runs["A28"][1]], ["unreadable stream_eval.json"] * 2)

    def test_g2_episode_list_and_repeat_coverage(self):
        a = {e: [r(ref95=10.0)] * 3 for e in range(8)}
        c = {e: [r(ref95=8.0)] * 3 for e in range(8)}
        eps = [0, 1, 2, 3, 4, 5]
        self.assertTrue(g2(only({"A28": a, "C28": c}, eps), eps)["pass"])
        c3 = {e: c[e] for e in (0, 1, 2)}  # n=3 needs ceil(2) = 2 better
        self.assertTrue(g2(only({"A28": a, "C28": c3}, [0, 1, 2]), [0, 1, 2])["pass"])
        self.assertFalse(g2(only({"A28": a, "C28": c3}, eps), eps)["pass"])  # episodes 3-5 missing
        thin = {**c, 5: [r(ref95=8.0)] * 2}
        out = g2(only({"A28": a, "C28": thin}, eps), eps)
        self.assertFalse(out["pass"])
        self.assertIn("C28: 2 valid of 3", out["why"])
        self.assertEqual(sorted(only({"A28": a}, [1, 2])["A28"]), [1, 2])

    def test_requested_episodes_missing_fail(self):
        eps = [1293, 1300, 1455, 2207, 3555, 3600]
        full = {c: {1293: [r(ref95=8.0 if c == "C28" else 10.0)] * 3} for c in ("Astored", "A28", "C28")}
        runs = only(full, eps)
        self.assertFalse(g0(runs, eps)["pass"])
        out = g1(runs, "C28", "A28", eps)
        self.assertFalse(out["pass"])
        self.assertIn("no runs for C28", [x for x in out["rows"] if x["episode"] == 1300][0]["why"])
        self.assertFalse(g2(runs, eps)["pass"])

    def test_invalid_reference_repeats_fail(self):
        bad = [r(), r(valid=False), r(valid=False)]
        out = g1({"A28": {1: bad}, "C28": {1: [r()] * 3}}, "C28", "A28", [1])
        self.assertFalse(out["pass"])
        self.assertIn("A28: 1 valid of 3", out["rows"][0]["why"])
        out = g0({"Astored": {1: [r(), r(valid=False), r(valid=False)]}, "A28": {1: [r()] * 3}}, [1])
        self.assertFalse(out["pass"])
        self.assertIn("Astored: 1 valid of 3", out["rows"][0]["why"])
        out = syn({"C28": {1: bad}, "C31syn": {1: [r()] * 3}}, "C31syn", "C28")
        self.assertFalse(out["pass"])
        self.assertIn("C28: 1 valid of 3", out["rows"][0]["why"])

    def test_happy_path_with_requested_episodes(self):
        eps = [1, 2, 3, 4, 5, 6]
        mk = lambda v: {e: [r(ref95=v)] * 3 for e in eps}  # noqa: E731
        runs = {"Astored": mk(8.0), "A28": mk(10.0), "C28": mk(8.0), "C31syn": mk(8.0)}
        self.assertTrue(g0(runs, eps)["pass"])
        self.assertTrue(g1(runs, "C28", "A28", eps)["pass"])
        self.assertTrue(g2(runs, eps)["pass"])
        self.assertTrue(syn(runs, "C31syn", "C28")["pass"])


if __name__ == "__main__":
    unittest.main()
