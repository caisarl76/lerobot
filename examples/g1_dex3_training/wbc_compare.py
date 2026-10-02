"""Comparison gates of the WBC backend comparison (spec: docs/superpowers/specs/2026-10-01-g1-wbc-backends-design.md)
from sonic_stream_eval.py results. Run dirs: WBC_<config>_ep<episode>_r<repeat> (Astored, A28, C28, A31syn, C31syn;
closed loop A28cl, C28cl, Anative). Statistics are per-episode means over valid repeats; spread = max - min.
Coverage: every gated episode needs REPEATS (3) runs on both sides (G0, G1, syn); a run dir without stream_eval.json
(crash) counts as an invalid run, a missing repeat or episode fails the row.
  G0  online A faithful: |A28 - Astored| <= max(0.5 cm, spread of Astored) on recorded-action palm p50, max(1.0, .) p95
  G1  equal balance: every repeat valid; mean max tilt <= reference + 2 deg; mean table.hit_records (contact records during the episode) <= reference's
  G2  C better: mean joint_ref palm p95 over the episodes >= 1.0 cm lower than A28, lower on >= 4 of 6 episodes
  syn 31D synthetic waist: valid, joint_ref palm p95 <= the same backend's 28D replay + 2 cm
Usage: python wbc_compare.py AUDIT_DIR [--out compare.json]
"""

import argparse
import json
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
        runs[m["cfg"]][int(m["ep"])].append(json.loads(f.read_text()) if f.exists() else {"valid": False, "reason": "no stream_eval.json"})
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


def short(*sides):
    return any(len(s) < REPEATS for s in sides)


def g0(runs) -> dict:
    rows = []
    for ep in sorted(set(runs.get("Astored", {})) | set(runs.get("A28", {}))):
        S, A = runs.get("Astored", {}).get(ep, []), runs.get("A28", {}).get(ep, [])
        for q, floor in (("p50", 0.5), ("p95", 1.0)):
            s, a = metric(S, "palm_err_vs_original_cm", q), metric(A, "palm_err_vs_original_cm", q)
            if short(S, A):
                rows.append({"episode": ep, "q": q, "pass": False, "why": f"fewer than {REPEATS} runs ({len(S)} Astored, {len(A)} A28)"})
            elif not s or not a:
                rows.append({"episode": ep, "q": q, "pass": False, "why": "no valid runs"})
            else:
                tol, diff = max(floor, max(s) - min(s)), abs(np.mean(a) - np.mean(s))
                rows.append({"episode": ep, "q": q, "diff_cm": round(float(diff), 2), "tol_cm": round(float(tol), 2), "pass": bool(diff <= tol)})
    return {"pass": bool(rows) and all(x["pass"] for x in rows), "rows": rows}


def g1(runs, test="C28", ref="A28") -> dict:
    rows = []
    for ep in sorted(set(runs.get(test, {})) | set(runs.get(ref, {}))):
        c, a = runs.get(test, {}).get(ep, []), runs.get(ref, {}).get(ep, [])
        ct, at = metric(c, "stability", "tilt_max_deg"), metric(a, "stability", "tilt_max_deg")
        ch, ah = metric(c, "table", "hit_records"), metric(a, "table", "hit_records")
        all_valid = bool(c) and all(x.get("valid") for x in c)
        row = {"episode": ep, "n": len(c), "n_ref": len(a), "all_valid": all_valid, "tilt": mean(ct), "tilt_ref": mean(at),
               "hit_records": mean(ch), "hit_records_ref": mean(ah)}  # fmt: skip
        if short(c, a):
            row.update(why=f"fewer than {REPEATS} runs", **{"pass": False})
        else:
            ok = all_valid and bool(ct) and bool(at) and bool(ch) and bool(ah)
            ok = ok and np.mean(ct) <= np.mean(at) + 2 and np.mean(ch) <= np.mean(ah)
            row["pass"] = bool(ok)
        rows.append(row)
    return {"pass": bool(rows) and all(x["pass"] for x in rows), "rows": rows}


def g2(runs) -> dict:
    eps = sorted(set(runs.get("C28", {})) & set(runs.get("A28", {})))
    c = [metric(runs["C28"][e], "palm_err_vs_joint_ref_cm", "p95") for e in eps]
    a = [metric(runs["A28"][e], "palm_err_vs_joint_ref_cm", "p95") for e in eps]
    pairs = [(e, float(np.mean(ci)), float(np.mean(ai))) for e, ci, ai in zip(eps, c, a, strict=True) if ci and ai]
    if len(pairs) < 6:
        return {"pass": False, "why": f"{len(pairs)} episodes with valid runs on both backends (need 6)", "rows": []}
    gain = float(np.mean([ai - ci for _, ci, ai in pairs]))
    better = int(sum(ci < ai for _, ci, ai in pairs))
    rows = [{"episode": e, "c_p95": round(ci, 2), "a_p95": round(ai, 2),
             "c_wrist_deg_p95": mean(metric(runs["C28"][e], "wrist_orientation_err_deg", "p95")),
             "a_wrist_deg_p95": mean(metric(runs["A28"][e], "wrist_orientation_err_deg", "p95"))} for e, ci, ai in pairs]  # fmt: skip
    return {"pass": bool(gain >= 1.0 and better >= 4), "gain_cm": round(gain, 2), "better": better, "rows": rows}


def syn(runs, cfg="C31syn", base="C28") -> dict:
    rows = []
    for ep in sorted(runs.get(cfg, {})):
        s, b = runs[cfg][ep], runs.get(base, {}).get(ep, [])
        sv, bv = metric(s, "palm_err_vs_joint_ref_cm", "p95"), metric(b, "palm_err_vs_joint_ref_cm", "p95")
        all_valid = bool(s) and all(x.get("valid") for x in s)
        row = {"episode": ep, "n": len(s), "n_base": len(b), "all_valid": all_valid, "p95": mean(sv), "base_p95": mean(bv),
               **{f"torso_{ax}_rad_p95": mean(metric(s, "torso_err_rad", ax, "p95")) for ax in ("roll", "pitch", "yaw")}}  # fmt: skip
        if short(s, b):
            row.update(why=f"fewer than {REPEATS} runs", **{"pass": False})
        else:
            row["pass"] = bool(all_valid and sv and bv and np.mean(sv) <= np.mean(bv) + 2)
        rows.append(row)
    return {"pass": bool(rows) and all(x["pass"] for x in rows), "rows": rows}


def closed_loop(runs) -> list:
    rows = []
    for cfg in ("A28cl", "C28cl", "Anative"):
        for ep, res in sorted(runs.get(cfg, {}).items()):
            rows.append({"cfg": cfg, "episode": ep, "valid": int(sum(bool(x.get("valid")) for x in res)), "total": len(res),
                         **{f"{ref}_{q}": mean(metric(res, f"palm_err_vs_{ref}_cm", q))
                            for ref in ("joint_ref", "original") for q in ("p50", "p95")}})  # fmt: skip
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("root")
    p.add_argument("--out")
    a = p.parse_args()
    runs = load(a.root)
    out = {"counts": {cfg: {ep: len(v) for ep, v in eps.items()} for cfg, eps in runs.items()},
           "G0": g0(runs), "G1": g1(runs, "C28", "A28"), "G2": g2(runs),
           "syn_A": syn(runs, "A31syn", "A28"), "syn_C": syn(runs, "C31syn", "C28")}  # fmt: skip
    if "C28cl" in runs:
        out["G1_closed"] = g1(runs, "C28cl", "A28cl")
    if any(c in runs for c in ("A28cl", "C28cl", "Anative")):
        out["closed_loop"] = closed_loop(runs)
    text = json.dumps(out, indent=1, default=float)
    print(text)
    if a.out:
        Path(a.out).write_text(text)


if __name__ == "__main__":
    main()
