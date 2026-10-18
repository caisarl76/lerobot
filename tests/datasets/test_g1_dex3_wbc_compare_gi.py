"""wbc_compare.gi: Isaac vs MuJoCo backend A replay parity (spec test 2)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2] / "examples" / "g1_dex3_training"))
import wbc_compare as wc


def res(p50, p95, tilt, valid=True):
    return {
        "valid": valid,
        "palm_err_vs_original_cm": {"p50": p50, "p95": p95},
        "stability": {"tilt_max_deg": tilt},
    }


def runs(i, a):
    return {"I28": {27: i}, "A28": {27: a}}


def test_gi_pass_and_each_failure():
    a = [res(2.0, 4.0, 5.0)] * 3
    assert wc.gi(runs([res(2.5, 5.5, 6.0)] * 3, a))["pass"]
    assert not wc.gi(runs([res(3.2, 5.5, 6.0)] * 3, a))["pass"]  # p50 diff 1.2 cm
    assert not wc.gi(runs([res(2.5, 6.5, 6.0)] * 3, a))["pass"]  # p95 diff 2.5 cm
    assert not wc.gi(runs([res(2.5, 5.5, 7.5)] * 3, a))["pass"]  # tilt +2.5 deg
    assert not wc.gi(runs([res(2.5, 5.5, 6.0)] * 2 + [res(0, 0, 0, valid=False)], a))["pass"]
    assert not wc.gi(runs([res(2.5, 5.5, 6.0)] * 2, a))["pass"]  # fewer than REPEATS
