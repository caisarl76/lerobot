"""Comparison gates of the WBC backend comparison (spec: docs/superpowers/specs/2026-10-01-g1-wbc-backends-design.md)
from sonic_stream_eval.py results. Run dirs: WBC_<config>_ep<episode>_r<repeat> (Astored, A28, C28, A31syn, C31syn;
closed loop A28cl, C28cl, Anative). Statistics are per-episode means over valid repeats; spread = max - min.
  G0  online A faithful: |A28 - Astored| <= max(0.5 cm, spread of Astored) on recorded-action palm p50, max(1.0, .) p95
  G1  equal balance: every repeat valid; mean max tilt <= reference + 2 deg; table hits <= reference's
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

NAME = re.compile(r"WBC_(?P<cfg>[A-Za-z0-9]+)_ep(?P<ep>\d+)_r(?P<r>\d+)$")


def load(root) -> dict:
    runs = defaultdict(lambda: defaultdict(list))
    for p in sorted(Path(root).glob("WBC_*/stream_eval.json")):
        m = NAME.match(p.parent.name)
        if m:
            runs[m["cfg"]][int(m["ep"])].append(json.loads(p.read_text()))
    return runs


def metric(results, *path) -> list:
    out = []
    for res in results:
        if not res.get("valid"):
            continue
        v = res
        for k in path:
            v = None if v is None else v.get(k)
        if v is not None:
            out.append(float(v))
    return out


def g0(runs) -> dict:
    rows = []
    for ep in sorted(set(runs.get("Astored", {})) | set(runs.get("A28", {}))):
        for q, floor in (("p50", 0.5), ("p95", 1.0)):
            s = metric(runs.get("Astored", {}).get(ep, []), "palm_err_vs_original_cm", q)
            a = metric(runs.get("A28", {}).get(ep, []), "palm_err_vs_original_cm", q)
            if not s or not a:
                rows.append({"episode": ep, "q": q, "pass": False, "why": "no valid runs"})
                continue
            tol, diff = max(floor, max(s) - min(s)), abs(np.mean(a) - np.mean(s))
            rows.append({"episode": ep, "q": q, "diff_cm": round(diff, 2), "tol_cm": round(tol, 2), "pass": diff <= tol})
    return {"pass": bool(rows) and all(x["pass"] for x in rows), "rows": rows}


def g1(runs, test="C28", ref="A28") -> dict:
    rows = []
    for ep in sorted(runs.get(test, {})):
        c, a = runs[test][ep], runs.get(ref, {}).get(ep, [])
        ct, at = metric(c, "stability", "tilt_max_deg"), metric(a, "stability", "tilt_max_deg")
        ch, ah = metric(c, "table", "hits_max"), metric(a, "table", "hits_max")
        all_valid = bool(c) and all(x.get("valid") for x in c)
        ok = all_valid and bool(ct) and bool(at) and np.mean(ct) <= np.mean(at) + 2 and max(ch) <= max(ah)
        rows.append({"episode": ep, "all_valid": all_valid, "tilt": round(float(np.mean(ct)), 2) if ct else None,
                     "tilt_ref": round(float(np.mean(at)), 2) if at else None, "hits": max(ch) if ch else None,
                     "hits_ref": max(ah) if ah else None, "pass": bool(ok)})  # fmt: skip
    return {"pass": bool(rows) and all(x["pass"] for x in rows), "rows": rows}


def g2(runs) -> dict:
    eps = sorted(set(runs.get("C28", {})) & set(runs.get("A28", {})))
    c = [metric(runs["C28"][e], "palm_err_vs_joint_ref_cm", "p95") for e in eps]
    a = [metric(runs["A28"][e], "palm_err_vs_joint_ref_cm", "p95") for e in eps]
    pairs = [(e, np.mean(ci), np.mean(ai)) for e, ci, ai in zip(eps, c, a, strict=True) if ci and ai]
    if len(pairs) < 6:
        return {"pass": False, "why": f"{len(pairs)} episodes with valid runs on both backends (need 6)", "rows": []}
    gain = float(np.mean([ai - ci for _, ci, ai in pairs]))
    better = sum(ci < ai for _, ci, ai in pairs)
    rows = [{"episode": e, "c_p95": round(ci, 2), "a_p95": round(ai, 2)} for e, ci, ai in pairs]
    return {"pass": gain >= 1.0 and better >= 4, "gain_cm": round(gain, 2), "better": better, "rows": rows}


def syn(runs, cfg="C31syn", base="C28") -> dict:
    rows = []
    for ep in sorted(runs.get(cfg, {})):
        s, b = runs[cfg][ep], runs.get(base, {}).get(ep, [])
        sv, bv = metric(s, "palm_err_vs_joint_ref_cm", "p95"), metric(b, "palm_err_vs_joint_ref_cm", "p95")
        all_valid = bool(s) and all(x.get("valid") for x in s)
        ok = all_valid and bool(sv) and bool(bv) and np.mean(sv) <= np.mean(bv) + 2
        rows.append({"episode": ep, "all_valid": all_valid, "p95": round(float(np.mean(sv)), 2) if sv else None,
                     "base_p95": round(float(np.mean(bv)), 2) if bv else None, "pass": bool(ok)})  # fmt: skip
    return {"pass": bool(rows) and all(x["pass"] for x in rows), "rows": rows}


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
    text = json.dumps(out, indent=1, default=float)
    print(text)
    if a.out:
        Path(a.out).write_text(text)


if __name__ == "__main__":
    main()
