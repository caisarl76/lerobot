"""Initial-pose effect: episode-first-pose start vs SONIC default-pose start, same episodes."""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/code")
from sonic_roundtrip_audit import DEFAULT, Robot

out = Path(sys.argv[1])
robot = Robot()
default_palm = robot.fk_pelvis_frame(DEFAULT)
rows = {
    (r["set"], r["episode"]): r
    for r in (json.loads(p.read_text()) for p in (out / "episodes").glob("*.json"))
}
edges = [0, 0.5, 1, 2, 3, 5, 1e9]
rep = {}
for name in ("unitree", "humanoid_everyday"):
    acc = {"matched": [], "default": []}
    settle, dist, peak0, share = [], [], [], {"matched": [], "default": []}
    for (s, ep), r in rows.items():
        if s != name or (name + "_initdefault", ep) not in rows:
            continue
        for key, rr in (("matched", r), ("default", rows[(name + "_initdefault", ep)])):
            d = np.load(Path(rr["out"]).with_suffix(".npz"))
            e = np.linalg.norm(d["ach_palm"] - d["raw_palm"], axis=-1).max(-1)
            t = np.arange(len(e)) / rr["fps"]
            acc[key].append(
                [
                    e[(t >= a) & (t < b)].mean() if ((t >= a) & (t < b)).any() else np.nan
                    for a, b in zip(edges[:-1], edges[1:], strict=False)
                ]
            )
            over = e > 0.05
            share[key].append((over & (t < 2)).sum() / max(1, over.sum()) if over.any() else np.nan)
            if key == "default":
                dist.append(float(np.linalg.norm(d["raw_palm"][0] - default_palm, axis=-1).max()))
                peak0.append(float(e[t < 2].max()))
                ok = np.flatnonzero(e < 0.03)
                settle.append(float(t[ok[0]]) if len(ok) else np.nan)
    rep[name] = {
        "episodes": len(dist),
        "mean_palm_err_cm_by_time": {
            "bins_s": ["0-0.5", "0.5-1", "1-2", "2-3", "3-5", "5+"],
            **{k: (np.nanmean(np.array(v, float), 0) * 100).round(2).tolist() for k, v in acc.items()},
        },
        "share_of_over5cm_frames_in_first_2s": {k: round(float(np.nanmean(v)), 3) for k, v in share.items()},
        "first_frame_palm_distance_from_default_cm": {
            "median": round(float(np.median(dist)) * 100, 1),
            "p90": round(float(np.percentile(dist, 90)) * 100, 1),
            "max": round(max(dist) * 100, 1),
        },
        "default_start_peak_err_first_2s_cm": {
            "median": round(float(np.median(peak0)) * 100, 1),
            "max": round(max(peak0) * 100, 1),
        },
        "default_start_time_to_under_3cm_s": {
            "median": round(float(np.nanmedian(settle)), 2),
            "max": round(float(np.nanmax(settle)), 2),
        },
        "palm_p95_cm_matched_vs_default": [
            round(
                float(
                    np.median(
                        [
                            rows[(name, ep)]["palm_err_achieved_vs_raw_m"]["p95"]
                            for (s, ep) in rows
                            if s == name + "_initdefault"
                        ]
                    )
                )
                * 100,
                2,
            ),
            round(
                float(
                    np.median(
                        [
                            r["palm_err_achieved_vs_raw_m"]["p95"]
                            for (s, _), r in rows.items()
                            if s == name + "_initdefault"
                        ]
                    )
                )
                * 100,
                2,
            ),
        ],
    }
(out / "init.json").write_text(json.dumps(rep, indent=1))
print(json.dumps(rep, indent=1))
