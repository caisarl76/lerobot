"""Paired limiter vs no-limiter comparison on identical episodes (safety + palm error)."""

import json
import sys
from pathlib import Path

import numpy as np

out = Path(sys.argv[1])
rows = {
    (r["set"], r["episode"]): r
    for r in (json.loads(p.read_text()) for p in (out / "episodes").glob("*.json"))
}
rep = {}
for name in ("unitree", "humanoid_everyday"):
    pairs = [
        (rows[k], rows[(name + "_noslew", k[1])])
        for k in rows
        if k[0] == name and (name + "_noslew", k[1]) in rows
    ]
    pairs = [(a, b) for a, b in pairs if "error" not in a and "error" not in b]

    def f(r, *ks):
        return (lambda v: [v := v[k] for k in ks][-1])(r)

    def col(side, *ks, pairs=pairs):
        return np.array([f(p[side], *ks) for p in pairs], float)

    s = {}
    for label, ks in (
        ("palm_p95_vs_original_cm", ("palm_err_achieved_vs_raw_m", "p95")),
        ("palm_max_vs_original_cm", ("palm_err_achieved_vs_raw_m", "max")),
        ("world_palm_p95_cm", ("palm_err_world_vs_raw_m", "p95")),
    ):
        a, b = col(0, *ks) * 100, col(1, *ks) * 100
        s[label] = {
            "limited_median": round(float(np.median(a)), 2),
            "unlimited_median": round(float(np.median(b)), 2),
            "limited_worst": round(float(a.max()), 2),
            "unlimited_worst": round(float(b.max()), 2),
            "episodes_over_5cm": [int((a > 5).sum()), int((b > 5).sum())]
            if "p95" in label and "world" not in label
            else None,
        }
    for label in (
        "arm_joint_speed_max_rad_s",
        "arm_setpoint_rate_max_rad_s",
        "torso_tilt_max_deg",
        "torque_saturation_frac_arms",
        "torque_saturation_frac_legs_waist",
    ):
        a, b = col(0, "safety", label), col(1, "safety", label)
        s[label] = {
            "limited_median": round(float(np.median(a)), 4),
            "unlimited_median": round(float(np.median(b)), 4),
            "limited_max": round(float(a.max()), 4),
            "unlimited_max": round(float(b.max()), 4),
        }
    s["falls"] = [
        sum(p[0]["fell_at_s"] is not None for p in pairs),
        sum(p[1]["fell_at_s"] is not None for p in pairs),
    ]
    worst = sorted(pairs, key=lambda p: -p[1]["safety"]["arm_joint_speed_max_rad_s"])[:4]
    s["fastest_unlimited_episodes"] = [
        {
            "episode": b["episode"],
            "arm_speed_max": round(b["safety"]["arm_joint_speed_max_rad_s"], 1),
            "limited_arm_speed_max": round(a["safety"]["arm_joint_speed_max_rad_s"], 1),
            "tilt_deg": round(b["safety"]["torso_tilt_max_deg"], 1),
        }
        for a, b in worst
    ]
    s["episodes"] = len(pairs)
    rep[name] = s
(out / "paired.json").write_text(json.dumps(rep, indent=1))
print(json.dumps(rep, indent=1))
