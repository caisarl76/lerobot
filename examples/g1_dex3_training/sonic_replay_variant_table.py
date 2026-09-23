"""Tabulate a variant replay session: episode x variant palm/orientation error, lag and stability."""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/code")
from sonic_targets import NOMINAL_BODY

run = Path(sys.argv[1])
cmp = json.loads((run / "compare.json").read_text())
r = np.load(run / "replay.npz", allow_pickle=True)
rows = []
for i, e in enumerate(cmp["episodes"]):
    variant, episode = e["name"].split("__", 1)
    m = (r["episode"] == i) & (r["phase"] == "episode")
    q = r["pelvis"][m, 3:7]
    tilt = np.degrees(np.arccos(np.clip(1 - 2 * (q[:, 1] ** 2 + q[:, 2] ** 2), -1, 1)))
    rows.append(
        {
            "episode": episode,
            "variant": variant,
            "palm_p50_cm": e["palm_err_cm"]["p50"],
            "palm_p95_cm": e["palm_err_cm"]["p95"],
            "palm_max_cm": e["palm_err_cm"]["max"],
            "wrist_orient_p50_deg": e["wrist_orientation_err_deg"]["p50"],
            "wrist_orient_p95_deg": e["wrist_orientation_err_deg"]["p95"],
            "best_lag_frames": e["best_lag_frames"],
            "dex3_p95_rad": e["dex3_reached_vs_cmd_rad"]["p95"],
            "leg_dev_max_rad": round(float(np.abs(r["body_q"][m][:, :15] - NOMINAL_BODY[:15]).max()), 3),
            "tilt_max_deg": round(float(tilt.max()), 1),
            "floor_contacts_min": int(r["floor_contacts"][m].min()),
            "pass_5cm": e["verdict_5cm_p95"] == "PASS",
        }
    )
summary = {}
for v in dict.fromkeys(x["variant"] for x in rows):
    vs = [x for x in rows if x["variant"] == v]
    summary[v] = {
        "episodes_pass": f"{sum(x['pass_5cm'] for x in vs)}/{len(vs)}",
        "median_palm_p95_cm": round(float(np.median([x["palm_p95_cm"] for x in vs])), 2),
        "worst_palm_p95_cm": max(x["palm_p95_cm"] for x in vs),
        "median_wrist_orient_p95_deg": round(float(np.median([x["wrist_orient_p95_deg"] for x in vs])), 1),
        "max_tilt_deg": max(x["tilt_max_deg"] for x in vs),
        "max_leg_dev_rad": max(x["leg_dev_max_rad"] for x in vs),
        "min_floor_contacts": min(x["floor_contacts_min"] for x in vs),
    }
out = {"run": cmp["run"], "summary": summary, "rows": rows}
(run / "variant_table.json").write_text(json.dumps(out, indent=1))
print(json.dumps(summary, indent=1))
for x in rows:
    print(
        f"{x['episode']:26s} {x['variant']:15s} palm p50/p95/max {x['palm_p50_cm']:5.2f}/{x['palm_p95_cm']:6.2f}/{x['palm_max_cm']:6.2f}  "
        f"orient p95 {x['wrist_orient_p95_deg']:5.1f}  lag {x['best_lag_frames']}  legs {x['leg_dev_max_rad']:.2f}  tilt {x['tilt_max_deg']:4.1f}  "
        f"contacts {x['floor_contacts_min']}  {'PASS' if x['pass_5cm'] else 'FAIL'}"
    )
