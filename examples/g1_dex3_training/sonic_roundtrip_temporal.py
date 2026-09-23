"""How round-trip palm error evolves in time: drift (grows with t) vs bursts (tied to fast source motion).

Per frame (pelvis frame, worse hand): total = reached vs original, sonic = reached vs limited reference,
limiter = limited reference vs original. Reads episodes/*.json + .npz written by sonic_roundtrip_audit.py.
"""

import json
import sys
from pathlib import Path

import numpy as np

GATE = 0.05


def runs(mask):
    """(start, end) index pairs of True runs."""
    d = np.diff(np.r_[0, mask.astype(int), 0])
    return list(zip(np.flatnonzero(d == 1), np.flatnonzero(d == -1), strict=False))


def analyze(out):
    rows = [json.loads(p.read_text()) for p in sorted(Path(out, "episodes").glob("*.json"))]
    rep = {}
    for name in sorted({r["set"] for r in rows} - {"calib"}):
        dec = {k: [] for k in ("total", "sonic", "limiter")}
        secs = {k: [] for k in ("total", "sonic")}
        slopes, first_last, burst_runs, jumps = [], [], [], []
        speed_all, e_sonic_all, e_lim_all = [], [], []
        for r in (r for r in rows if r["set"] == name and "error" not in r):
            d = np.load(Path(r["out"]).with_suffix(".npz"))
            fps = r["fps"]

            def worst(a, b, d=d):
                return np.linalg.norm(d[a] - d[b], axis=-1).max(-1)

            tot, son, lim = (
                worst("ach_palm", "raw_palm"),
                worst("ach_palm", "ref_palm"),
                worst("ref_palm", "raw_palm"),
            )
            n = len(tot)
            t = np.arange(n) / fps
            speed = np.r_[
                0, np.abs(np.diff(d["raw_motor"][:, 15:], axis=0)).max(1) * fps
            ]  # max arm joint speed, rad/s
            speed_all.append(speed)
            e_sonic_all.append(son)
            e_lim_all.append(lim)
            dec_idx = np.minimum((np.arange(n) * 10) // n, 9)
            for k, e in (("total", tot), ("sonic", son), ("limiter", lim)):
                dec[k].append([e[dec_idx == i].mean() for i in range(10)])
            for k, e in (("total", tot), ("sonic", son)):
                edges = [0, 1, 5, 10, 20, 40, 1e9]
                secs[k].append(
                    [
                        e[(t >= a) & (t < b)].mean() if ((t >= a) & (t < b)).any() else np.nan
                        for a, b in zip(edges[:-1], edges[1:], strict=False)
                    ]
                )
            slopes.append(np.polyfit(t, son, 1)[0] * 10)  # m per 10 s
            q = max(1, n // 4)
            first_last.append((son[:q].mean(), son[-q:].mean()))
            win = int(fps)  # look back 1 s for a fast source motion
            for a, b in runs(tot > GATE):
                lo = max(0, a - win)
                burst_runs.append(
                    {
                        "set": name,
                        "episode": r["episode"],
                        "start_s": a / fps,
                        "dur_s": (b - a) / fps,
                        "peak_cm": float(tot[a:b].max() * 100),
                        "max_src_speed": float(speed[lo:b].max()),
                        "limiter_share": float(
                            (lim[a:b] ** 2).sum() / max(1e-12, (lim[a:b] ** 2).sum() + (son[a:b] ** 2).sum())
                        ),
                        "recovered": bool((tot[b : b + 2 * win] < 0.02).any()) if b < n else None,
                    }
                )
            for k in np.flatnonzero(speed > 6):  # ≥0.2 rad in one 30 Hz frame: teleop glitch / pose jump
                jumps.append(
                    {"episode": r["episode"], "t_s": round(k / fps, 2), "speed": round(float(speed[k]), 1)}
                )
        sp, es, el = map(np.concatenate, (speed_all, e_sonic_all, e_lim_all))
        bins = [0, 0.5, 1, 1.5, 2, 3, 1e9]
        by_speed = {
            f"{a}-{b if b < 1e9 else 'inf'} rad/s": {
                "frames": int(((sp >= a) & (sp < b)).sum()),
                "sonic_p95_cm": round(float(np.percentile(es[(sp >= a) & (sp < b)], 95) * 100), 2)
                if ((sp >= a) & (sp < b)).any()
                else None,
                "limiter_p95_cm": round(float(np.percentile(el[(sp >= a) & (sp < b)], 95) * 100), 2)
                if ((sp >= a) & (sp < b)).any()
                else None,
            }
            for a, b in zip(bins[:-1], bins[1:], strict=False)
        }
        fl = np.array(first_last)
        rep[name] = {
            "error_by_episode_decile_cm": {
                k: (np.nanmean(v, 0) * 100).round(2).tolist() for k, v in dec.items()
            },
            "error_by_elapsed_seconds_cm": {
                "bins": ["0-1", "1-5", "5-10", "10-20", "20-40", "40+"],
                **{k: (np.nanmean(np.array(v, float), 0) * 100).round(2).tolist() for k, v in secs.items()},
            },
            "sonic_err_slope_cm_per_10s": {
                "median": round(float(np.median(slopes) * 100), 3),
                "p90": round(float(np.percentile(slopes, 90) * 100), 3),
            },
            "sonic_err_first_vs_last_quarter_cm": [
                round(float(fl[:, 0].mean() * 100), 2),
                round(float(fl[:, 1].mean() * 100), 2),
            ],
            "episodes_where_last_quarter_worse": round(float((fl[:, 1] > fl[:, 0] + 0.005).mean()), 3),
            "sonic_error_by_source_arm_speed": by_speed,
            "corr_speed_vs_sonic_err": round(float(np.corrcoef(sp, es)[0, 1]), 3),
            "corr_speed_vs_limiter_err": round(float(np.corrcoef(sp, el)[0, 1]), 3),
            "bursts_over_5cm": {
                "count": len(burst_runs),
                "median_dur_s": round(float(np.median([b["dur_s"] for b in burst_runs])), 2)
                if burst_runs
                else None,
                "p90_dur_s": round(float(np.percentile([b["dur_s"] for b in burst_runs], 90)), 2)
                if burst_runs
                else None,
                "preceded_by_src_speed_over_1rad_s": round(
                    float(np.mean([b["max_src_speed"] > 1 for b in burst_runs])), 3
                )
                if burst_runs
                else None,
                "limiter_dominated": round(float(np.mean([b["limiter_share"] > 0.5 for b in burst_runs])), 3)
                if burst_runs
                else None,
                "recovered_below_2cm_within_2s": round(
                    float(np.mean([b["recovered"] for b in burst_runs if b["recovered"] is not None])), 3
                )
                if burst_runs
                else None,
                "longest": sorted(burst_runs, key=lambda b: -b["dur_s"])[:3],
            },
            "source_pose_jumps_over_6rad_s": {"count": len(jumps), "examples": jumps[:8]},
        }
    return rep


if __name__ == "__main__":
    rep = analyze(sys.argv[1])
    Path(sys.argv[1], "temporal.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))
