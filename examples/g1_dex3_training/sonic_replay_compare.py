"""Compare an official-deploy token replay (sonic_official_replay.py) against the original joint actions.

Per episode, at each source frame (first 50 Hz tick of that frame): palm position error (pelvis frame, from
FK of the original arms on the robot's actual pelvis), wrist orientation error, per-joint arm error,
best lag, Dex3 hand command vs reached; plus stability (falls, tilt, floor contacts) for the whole run.
"""

import json
import sys
from pathlib import Path

import numpy as np

run = Path(sys.argv[1])
r = np.load(run / "replay.npz", allow_pickle=True)
eps = json.loads((run / "episodes.json").read_text())
names = ["shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw"]


def stats(x):
    x = np.asarray(x, float).ravel()
    return {
        "mean": round(float(x.mean()), 4),
        "p50": round(float(np.median(x)), 4),
        "p95": round(float(np.percentile(x, 95)), 4),
        "max": round(float(x.max()), 4),
    }


def rot_err_deg(ra, rb):
    c = (np.einsum("...ij,...ij->...", ra, rb) - 1) / 2
    return np.degrees(np.arccos(np.clip(c, -1, 1)))


quat = r["pelvis"][:, 3:7]
tilt = np.degrees(np.arccos(np.clip(1 - 2 * (quat[:, 1] ** 2 + quat[:, 2] ** 2), -1, 1)))
out = {
    "run": {
        "ticks": int(len(r["t"])),
        "pelvis_z_min": round(float(r["pelvis"][:, 2].min()), 4),
        "tilt_max_deg": round(float(tilt.max()), 2),
        "floor_contacts_min": int(r["floor_contacts"].min()),
        "fell": bool(r["pelvis"][:, 2].min() < 0.55),
    },
    "episodes": [],
}
for e_i, e in enumerate(eps):
    m = (r["episode"] == e_i) & (r["phase"] == "episode")
    idx = np.flatnonzero(m)
    frames = r["frame"][idx]
    first = idx[np.r_[True, np.diff(frames) != 0]]  # first tick of each source frame

    # pelvis-frame positions: R^T (p - pelvis_pos)
    def to_pelvis(p, rows):
        from_quat = []
        for q in r["pelvis"][rows, 3:7]:
            w, x, y, z = q
            from_quat.append(
                np.array(
                    [
                        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
                    ]
                )
            )
        rot = np.array(from_quat)
        return np.einsum("kji,kpj->kpi", rot, p - r["pelvis"][rows, None, :3])

    pa, pi = to_pelvis(r["palm"][first], first), to_pelvis(r["palm_int"][first], first)
    perr = np.linalg.norm(pa - pi, axis=-1)  # [n, 2]
    oerr = rot_err_deg(r["wrist_R"][first], r["wrist_R_int"][first])
    jerr = np.abs(r["body_q"][first][:, 15:] - r["orig_arms"][first])
    lags = {}
    for s in range(0, 16):
        a, b = pa[s:], pi[: len(pi) - s]
        if len(a) > 10:
            lags[s] = float(np.linalg.norm(a - b, axis=-1).mean())
    best = min(lags, key=lags.get)
    hand_err = np.abs(r["hand_q"][first] - r["hand_cmd"][first])
    out["episodes"].append(
        {
            **e,
            "frames_compared": int(len(first)),
            "palm_err_cm": {k: round(v * 100, 2) for k, v in stats(perr).items()},
            "palm_err_p95_cm_left_right": [
                round(float(np.percentile(perr[:, h], 95)) * 100, 2) for h in (0, 1)
            ],
            "palm_err_first_frame_cm": round(float(perr[0].max()) * 100, 2),
            "wrist_orientation_err_deg": {k: round(v, 1) for k, v in stats(oerr).items()},
            "arm_joint_err_rad": stats(jerr),
            "arm_joint_mae_rad": {
                f"{side}_{n}": round(float(jerr[:, i + 7 * j].mean()), 3)
                for j, side in enumerate(("L", "R"))
                for i, n in enumerate(names)
            },
            "best_lag_frames": best,
            "palm_mean_err_at_best_lag_cm": round(lags[best] * 100, 2),
            "palm_mean_err_lag0_cm": round(lags[0] * 100, 2),
            "dex3_reached_vs_cmd_rad": stats(hand_err),
            "verdict_5cm_p95": "PASS" if np.percentile(perr, 95) <= 0.05 else "FAIL",
        }
    )
print(json.dumps(out, indent=1))
(run / "compare.json").write_text(json.dumps(out, indent=1))
