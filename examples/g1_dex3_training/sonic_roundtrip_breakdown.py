"""Breakdown of round-trip errors: clipping vs slew vs decoder, per joint; Dex3 pass-through; pelvis drift."""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/code")
from sonic_roundtrip_audit import v3_episode, v3_meta
from sonic_targets import ACTION_NAMES, load_joint_limits

out = Path(sys.argv[1])
lim = load_joint_limits(Path("/gear_sonic_deploy/g1/g1_29dof_with_hand.xml"))
rep = {}
for name, src in (
    ("unitree", "/run-output/datasets/joint28"),
    ("humanoid_everyday", "/run-output/humanoid_everyday_g1_20260923/datasets/joint28"),
):
    rows, info = v3_meta(src)
    by = {r["episode_index"]: r for r in rows}
    eps = [json.loads(p.read_text()) for p in sorted((out / "episodes").glob(f"{name}__*.json"))]
    below, above, n, slew, big = np.zeros(28), np.zeros(28), 0, np.zeros(28), np.zeros(28)
    per_ep = []
    for e in eps:
        a = v3_episode(src, by[e["episode"]], info, ["frame_index", "action"])["action"].astype(np.float64)
        below += (a < lim[:, 0]).sum(0)
        above += (a > lim[:, 1]).sum(0)
        n += len(a)
        big = np.maximum(big, np.maximum(lim[:, 0] - a, a - lim[:, 1]).max(0))
        v = np.abs(np.diff(a, axis=0)) * 30
        slew += (v > np.r_[np.ones(14), np.full(14, 2.0)]).sum(0)
        d = np.load(Path(e["out"]).with_suffix(".npz"))
        per_ep.append(
            {
                "episode": e["episode"],
                "task": e.get("task"),
                "ach_raw_p95": e["palm_err_achieved_vs_raw_m"]["p95"],
                "ach_filt_p95": e["palm_err_decoded_vs_filtered_m"]["p95"]
                and float(np.percentile(np.linalg.norm(d["ach_palm"] - d["ref_palm"], axis=-1), 95)),
                "filt_raw_p95": e["palm_err_filtered_vs_raw_m"]["p95"],
                "pelvis_drift_m": e["pelvis_drift_m"],
                "world_p95": e["palm_err_world_vs_raw_m"]["p95"],
                "clip_frac": e["encoder_report"]["clipped_values"] / (28 * len(a)),
                "arm_speed_p99_rad_s": float(np.percentile(v[:, :14], 99)),
                "dex3_max": e.get("dex3_stored_vs_source", {}).get("max"),
            }
        )
    rep[name] = {
        "frames": n,
        "clip_rate_per_joint": {
            ACTION_NAMES[i]: round(float((below[i] + above[i]) / n), 4)
            for i in range(28)
            if below[i] + above[i]
        },
        "clip_below_above": {
            ACTION_NAMES[i]: [int(below[i]), int(above[i])] for i in range(28) if below[i] + above[i]
        },
        "max_exceedance_rad": {ACTION_NAMES[i]: round(float(big[i]), 3) for i in range(28) if big[i] > 0},
        "over_speed_limit_rate": {
            ACTION_NAMES[i]: round(float(slew[i] / n), 4) for i in range(28) if slew[i]
        },
        "worst": sorted(per_ep, key=lambda r: -r["ach_raw_p95"])[:6],
        "median_pelvis_drift_m": float(np.median([r["pelvis_drift_m"] for r in per_ep])),
    }
t43 = [json.loads(p.read_text()) for p in sorted((out / "episodes").glob("teleop43__*.json"))]
rep["teleop43_pelvis_drift_m"] = [round(e["pelvis_drift_m"], 3) for e in t43]
for e in t43[:3]:
    d = np.load(Path(e["out"]).with_suffix(".npz"))
    rep.setdefault("teleop43_leg_action_range_rad", []).append(
        np.ptp(d["raw_motor"][:, :15], 0).round(2).tolist()
    )
print(json.dumps(rep, indent=1))
