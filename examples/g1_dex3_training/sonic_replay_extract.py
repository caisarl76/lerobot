"""Extract stored sonic78 tokens/hands + original joint28 actions for deploy replay (one npz per episode)."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/code")
from sonic_roundtrip_audit import v3_episode, v3_meta

SRC = {
    "unitree": ("/run-output/datasets/joint28", "/run-output/datasets/sonic78"),
    "humanoid_everyday": (
        "/run-output/humanoid_everyday_g1_20260923/datasets/joint28",
        "/run-output/humanoid_everyday_g1_20260923/datasets/sonic78",
    ),
}
out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
for spec in sys.argv[2:]:  # dataset:episode
    name, ep = spec.split(":")
    ep = int(ep)
    src, conv = SRC[name]
    rows, info = v3_meta(src)
    row = next(r for r in rows if r["episode_index"] == ep)
    a = v3_episode(src, row, info, ["frame_index", "action"])["action"].astype(np.float32)
    s = v3_episode(conv, row, info, ["frame_index", "action"])["action"].astype(np.float32)
    assert s.shape == (len(a), 78)
    np.savez(
        out / f"{name}__{ep}.npz",
        tokens=s[:, :64],
        hands=s[:, 64:],
        original=a,
        fps=np.float32(info["fps"]),
        task=row["tasks"][0],
    )
    print(name, ep, len(a), row["tasks"][0])
