"""Offline check of sonic_policy_streamer.py --action-space joint28 against the stored sonic78_nolimit tokens.

For each episode: (1) the whole recorded 28D action sequence through joint_chunk_to_sonic must reproduce the stored
78D actions exactly (same code path as prepare_sonic_dataset, limits off); (2) streamer simulation: replan every
REPLAN frames, encode actions[k:k+H] and use its first REPLAN rows, for several chunk lengths H.
Run where the datasets and SONIC v1.1 model are mounted (H100 prep container: /run-output, /sonic-model).
Result 2026-09-28 (20 Unitree episodes): whole episodes exact; chunks of 40+ frames 99.98% identical tokens
(differences are single 1/16 grid steps), 30 frames 99.57%; hands equal to float rounding (<=2.4e-7 rad).
"""

import json
import random
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

try:
    from .sonic_targets import SonicEncoder, joint_chunk_to_sonic, load_joint_limits
except ImportError:
    from sonic_targets import SonicEncoder, joint_chunk_to_sonic, load_joint_limits

D = Path("/run-output/datasets")
REPLAN = 12  # 0.4 s at 30 Hz (streamer default --replan-s)
limits = load_joint_limits(Path("/run-output/environment/g1_29dof_with_hand.xml"))
encoder = SonicEncoder(Path("/sonic-model/model_encoder.onnx"), Path("/sonic-model/observation_config.yaml"))


def actions(root, eps):
    out = {}
    for f in sorted((root / "data").rglob("*.parquet")):
        t = pq.read_table(f, columns=["episode_index", "frame_index", "action"])
        e = t.column("episode_index").to_numpy()
        for ep in set(e) & eps:
            m = e == ep
            order = np.argsort(t.column("frame_index").to_numpy()[m])
            out.setdefault(ep, []).append(
                np.stack(t.column("action").to_numpy(zero_copy_only=False)[m])[order]
            )
    return {k: np.concatenate(v) for k, v in out.items()}


eps = {2155, 6} | set(random.Random(0).sample(range(3152), 18)) - {2202}
j28, s78 = actions(D / "joint28", eps), actions(D / "sonic78_nolimit", eps)
full_exact, rows = [], {h: [] for h in (30, 40, 50, 100)}
for ep in sorted(eps):
    a, stored = j28[ep].astype(np.float32), s78[ep].astype(np.float32)
    assert len(a) == len(stored)
    full = joint_chunk_to_sonic(a, limits, encoder)
    full_exact.append(float(np.mean(full == stored)))
    for h in rows:
        for k in range(0, len(a) - REPLAN, REPLAN):
            used = joint_chunk_to_sonic(a[k : k + h], limits, encoder)[:REPLAN]
            ref = stored[k : k + len(used)]
            rows[h].append(
                (
                    np.abs(used[:, :64] - ref[:, :64]),
                    np.allclose(used[:, 64:], ref[:, 64:], atol=1e-6, rtol=0),
                )
            )

token_std = float(np.concatenate([s[:, :64] for s in s78.values()]).std())
res = {
    "episodes": len(eps),
    "full_episode_exact_fraction": round(float(np.mean(full_exact)), 6),
    "token_std": round(token_std, 4),
    "replan_frames": REPLAN,
}
for h, r in rows.items():
    d = np.concatenate([x for x, _ in r])
    res[f"chunk_{h}"] = {
        "token_exact_fraction": round(float(np.mean(d == 0)), 4),
        "token_mae_over_std": round(float(d.mean() / token_std), 4),
        "token_max_abs": float(d.max()),
        "hands_identical": bool(all(ok for _, ok in r)),
    }
print(json.dumps(res, indent=1))
