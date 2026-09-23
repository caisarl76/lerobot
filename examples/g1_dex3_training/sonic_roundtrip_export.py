"""Export representative episodes (original vs SONIC round trip) as compact JSON for the viewer page."""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/code")
from sonic_roundtrip_audit import v3_episode, v3_meta
from sonic_targets import ACTION_NAMES

out = Path(sys.argv[1])
rows = {
    (r["set"], r["episode"]): r
    for r in (json.loads(p.read_text()) for p in (out / "episodes").glob("*.json"))
}
MAX_FRAMES = 1800
SRC = {
    "unitree": ("/run-output/datasets/joint28", "/run-output/datasets/sonic78"),
    "humanoid_everyday": (
        "/run-output/humanoid_everyday_g1_20260923/datasets/joint28",
        "/run-output/humanoid_everyday_g1_20260923/datasets/sonic78",
    ),
}


def r3(a):
    return np.round(np.asarray(a, float), 3).tolist()


episodes = []
for name, (src, conv) in SRC.items():
    cands = [
        r
        for (s, _), r in rows.items()
        if s == name
        and (name + "_noslew", r["episode"]) in rows
        and (name + "_initdefault", r["episode"]) in rows
    ]
    by_p95 = sorted(cands, key=lambda r: r["palm_err_achieved_vs_raw_m"]["p95"])

    def speed(r):
        return float(
            np.abs(np.diff(np.load(Path(r["out"]).with_suffix(".npz"))["raw_motor"][:, 15:], axis=0)).max()
            * r["fps"]
        )

    picks, seen = [], set()
    for why, r in (
        ("typical (median error)", by_p95[len(by_p95) // 2]),
        ("worst error", by_p95[-1]),
        ("fastest source motion", max(cands, key=speed)),
    ):
        if r["episode"] in seen:
            continue
        seen.add(r["episode"])
        picks.append((why, r))
    rows_meta, info = v3_meta(src)
    meta = {m["episode_index"]: m for m in rows_meta}
    for why, r in picks:
        ep = r["episode"]

        def load(s, ep=ep):
            return np.load(Path(rows[(s, ep)]["out"]).with_suffix(".npz"))

        base, fast, dflt = load(name), load(name + "_noslew"), load(name + "_initdefault")
        n = min(len(base["raw_motor"]), MAX_FRAMES)

        def err(d, n=n):
            return np.linalg.norm(d["ach_palm"] - d["raw_palm"], axis=-1).max(-1)[:n] * 100

        src_a = v3_episode(src, meta[ep], info, ["frame_index", "action"])["action"]
        stored = v3_episode(conv, meta[ep], info, ["frame_index", "action"])["action"]
        episodes.append(
            {
                "dataset": name,
                "episode": ep,
                "task": r["task"],
                "why": why,
                "fps": r["fps"],
                "frames": int(len(base["raw_motor"])),
                "shown": int(n),
                "p95_cm": {
                    "current": round(r["palm_err_achieved_vs_raw_m"]["p95"] * 100, 2),
                    "nolimit": round(
                        rows[(name + "_noslew", ep)]["palm_err_achieved_vs_raw_m"]["p95"] * 100, 2
                    ),
                    "defaultstart": round(
                        rows[(name + "_initdefault", ep)]["palm_err_achieved_vs_raw_m"]["p95"] * 100, 2
                    ),
                },
                "original": r3(base["raw_motor"][:n, 15:]),
                "limited": r3(base["ref_motor"][:n, 15:]),
                "sonic": r3(base["achieved"][:n, 15:]),
                "sonic_nolimit": r3(fast["achieved"][:n, 15:]),
                "sonic_defaultstart": r3(dflt["achieved"][:n, 15:]),
                "palm_err_cm": {
                    "current": r3(err(base)),
                    "nolimit": r3(err(fast)),
                    "defaultstart": r3(err(dflt)),
                },
                "tokens16": np.rint(stored[:n, :64] * 16).astype(int).tolist(),
                "hand_original": r3(src_a[:n, 14:]),
                "hand_stored": r3(stored[:n, 64:]),
            }
        )
(out / "viz.json").write_text(
    json.dumps(
        {"arm_names": list(ACTION_NAMES[:14]), "hand_names": list(ACTION_NAMES[14:]), "episodes": episodes},
        separators=(",", ":"),
    )
)
print([(e["dataset"], e["episode"], e["why"], e["shown"], e["p95_cm"]) for e in episodes])
