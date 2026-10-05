"""Comparison gates of the WBC backend comparison (spec: docs/superpowers/specs/2026-10-01-g1-wbc-backends-design.md)
from sonic_stream_eval.py results. Run dirs: WBC_<config>_ep<episode>_r<repeat> (Astored, A28, C28;
closed loop A28cl, C28cl, Anative). Statistics are per-episode means over valid repeats; spread = max - min.
Coverage rule (every gate): each side of a row needs >= REPEATS (3) runs and ALL of them valid, else the row fails with
a why ("A28: 1 valid of 3", "no runs for C28"). A run dir without stream_eval.json (crash) is an invalid run.
Given `episodes`, a gate iterates exactly those episodes (a requested episode without runs fails); without, the union
of the episodes present.
  G0  online A faithful: |A28 - Astored| <= max(0.5 cm, spread of Astored) on recorded-action palm p50, max(1.0, .) p95
  G1  equal balance: every repeat valid; mean max tilt <= reference + 2 deg; mean table.hit_records (contact records during the episode) <= reference's
  G2  C better: mean joint_ref palm p95 over the episodes >= 1.0 cm lower than A28, lower on >= 4 of 6 episodes
      (other episode counts n: all n paired, lower on >= ceil(2n/3)); every pair needs REPEATS runs on both sides
--episodes sets the gated episodes of G0/G1/G2/closed loop; G2 needs all of them.
Usage: python wbc_compare.py AUDIT_DIR [--out compare.json] [--episodes 1293,1300,1455,2207,3555,3600]
"""

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

REPEATS = 3
NAME = re.compile(r"WBC_(?P<cfg>[A-Za-z0-9]+)_ep(?P<ep>\d+)_r(?P<r>\d+)$")


def load(root) -> dict:
    runs = defaultdict(lambda: defaultdict(list))
    for d in sorted(Path(root).glob("WBC_*")):
        m = NAME.match(d.name)
        if not (m and d.is_dir()):
            continue
        f = d / "stream_eval.json"
        if not f.exists():
            res = {"valid": False, "reason": "no stream_eval.json"}
        else:
            try:
                res = json.loads(f.read_text())
            except ValueError:  # empty or truncated by a crash
                res = {"valid": False, "reason": "unreadable stream_eval.json"}
        runs[m["cfg"]][int(m["ep"])].append(res)
    return runs


def metric(results, *path) -> list:
    out = []
    for res in results:
        if not res.get("valid"):
            continue
        v = res
        for k in path:
            v = v.get(k) if isinstance(v, dict) else None
        if v is not None:
            out.append(float(v))
    return out


def mean(x):
    return round(float(np.mean(x)), 2) if x else None


def only(runs, episodes) -> dict:
    """runs restricted to the gated episodes (None keeps all)."""
    if episodes is None:
        return runs
    kept = {cfg: {e: v for e, v in eps.items() if e in episodes} for cfg, eps in runs.items()}
    return {cfg: eps for cfg, eps in kept.items() if eps}


def uncovered(**sides):
    """why string for the first side without REPEATS runs that are all valid, else None."""
    for name, rs in sides.items():
        if not rs:
            return f"no runs for {name}"
        nv = sum(bool(x.get("valid")) for x in rs)
        if len(rs) < REPEATS or nv < len(rs):
            return f"{name}: {nv} valid of {max(len(rs), REPEATS)}"
    return None


def episodes_of(runs, episodes, *cfgs):
    if episodes is not None:
        return sorted(episodes)
    return sorted(set().union(*(runs.get(c, {}) for c in cfgs)))


def g0(runs, episodes=None) -> dict:
    rows = []
    for ep in episodes_of(runs, episodes, "Astored", "A28"):
        s_runs, a_runs = runs.get("Astored", {}).get(ep, []), runs.get("A28", {}).get(ep, [])
        for q, floor in (("p50", 0.5), ("p95", 1.0)):
            s, a = metric(s_runs, "palm_err_vs_original_cm", q), metric(a_runs, "palm_err_vs_original_cm", q)
            if why := uncovered(Astored=s_runs, A28=a_runs):
                rows.append({"episode": ep, "q": q, "pass": False, "why": why})
            elif not s or not a:
                rows.append({"episode": ep, "q": q, "pass": False, "why": "no valid runs"})
            else:
                tol, diff = max(floor, max(s) - min(s)), abs(np.mean(a) - np.mean(s))
                rows.append(
                    {
                        "episode": ep,
                        "q": q,
                        "diff_cm": round(float(diff), 2),
                        "tol_cm": round(float(tol), 2),
                        "pass": bool(diff <= tol),
                    }
                )
    return {"pass": bool(rows) and all(x["pass"] for x in rows), "rows": rows}


def g1(runs, test="C28", ref="A28", episodes=None) -> dict:
    rows = []
    for ep in episodes_of(runs, episodes, test, ref):
        c, a = runs.get(test, {}).get(ep, []), runs.get(ref, {}).get(ep, [])
        ct, at = metric(c, "stability", "tilt_max_deg"), metric(a, "stability", "tilt_max_deg")
        ch, ah = metric(c, "table", "hit_records"), metric(a, "table", "hit_records")
        all_valid = bool(c) and all(x.get("valid") for x in c)
        row = {"episode": ep, "n": len(c), "n_ref": len(a), "all_valid": all_valid, "tilt": mean(ct), "tilt_ref": mean(at),
               "hit_records": mean(ch), "hit_records_ref": mean(ah)}  # fmt: skip
        if why := uncovered(**{test: c, ref: a}):
            row.update(why=why, **{"pass": False})
        else:
            ok = all_valid and bool(ct) and bool(at) and bool(ch) and bool(ah)
            ok = ok and np.mean(ct) <= np.mean(at) + 2 and np.mean(ch) <= np.mean(ah)
            row["pass"] = bool(ok)
        rows.append(row)
    return {"pass": bool(rows) and all(x["pass"] for x in rows), "rows": rows}


def g2(runs, episodes=None) -> dict:
    eps = sorted(set(runs.get("C28", {})) & set(runs.get("A28", {})))
    want = len(episodes) if episodes is not None else 6
    if episodes is not None:
        eps = [e for e in eps if e in episodes]
    c = [metric(runs["C28"][e], "palm_err_vs_joint_ref_cm", "p95") for e in eps]
    a = [metric(runs["A28"][e], "palm_err_vs_joint_ref_cm", "p95") for e in eps]
    pairs = [
        (e, float(np.mean(ci)), float(np.mean(ai))) for e, ci, ai in zip(eps, c, a, strict=True) if ci and ai
    ]
    if len(pairs) != want:
        return {
            "pass": False,
            "why": f"{len(pairs)} episodes with valid runs on both backends (need {want})",
            "rows": [],
        }
    thin = {e: w for e in eps if (w := uncovered(C28=runs["C28"][e], A28=runs["A28"][e]))}
    if thin:
        return {"pass": False, "why": f"coverage failed for episodes {thin}", "rows": []}
    gain = float(np.mean([ai - ci for _, ci, ai in pairs]))
    better = int(sum(ci < ai for _, ci, ai in pairs))
    need = 4 if want == 6 else math.ceil(2 * want / 3)
    rows = [{"episode": e, "c_p95": round(ci, 2), "a_p95": round(ai, 2),
             "c_wrist_deg_p95": mean(metric(runs["C28"][e], "wrist_orientation_err_deg", "p95")),
             "a_wrist_deg_p95": mean(metric(runs["A28"][e], "wrist_orientation_err_deg", "p95"))} for e, ci, ai in pairs]  # fmt: skip
    return {
        "pass": bool(gain >= 1.0 and better >= need),
        "gain_cm": round(gain, 2),
        "better": better,
        "rows": rows,
    }


def closed_loop(runs, episodes=None) -> list:
    rows = []
    for cfg in ("A28cl", "C28cl", "Anative"):
        for ep in episodes_of(runs, episodes, cfg):
            res = runs.get(cfg, {}).get(ep, [])
            why = uncovered(**{cfg: res})
            rows.append({"cfg": cfg, "episode": ep, "valid": int(sum(bool(x.get("valid")) for x in res)), "total": len(res),
                         "pass": why is None, **({"why": why} if why else {}),
                         **{f"{ref}_{q}": mean(metric(res, f"palm_err_vs_{ref}_cm", q))
                            for ref in ("joint_ref", "original") for q in ("p50", "p95")}})  # fmt: skip
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("root")
    p.add_argument("--out")
    p.add_argument(
        "--episodes", type=lambda s: [int(x) for x in s.split(",")], help="comma-separated gated episodes"
    )
    a = p.parse_args()
    all_runs = load(a.root)
    runs = only(all_runs, a.episodes)
    out = {"counts": {cfg: {ep: len(v) for ep, v in eps.items()} for cfg, eps in runs.items()},
           "G0": g0(runs, a.episodes), "G1": g1(runs, "C28", "A28", a.episodes), "G2": g2(runs, a.episodes)}  # fmt: skip
    if "C28cl" in runs:
        out["G1_closed"] = g1(runs, "C28cl", "A28cl", a.episodes)
    if any(c in runs for c in ("A28cl", "C28cl", "Anative")):
        out["closed_loop"] = closed_loop(runs, a.episodes)
    text = json.dumps(out, indent=1, default=float)
    print(text)
    if a.out:
        Path(a.out).write_text(text)


if __name__ == "__main__":
    main()
