"""Open-loop smoothness of predicted chunks on held-out episodes (no robot, no closed loop).

Every 12 frames (0.4 s replan at 30 Hz) predict a chunk from the recorded observation, then measure, in units of the
per-dimension action std (78D: tokens 0:64, hands 64:78; 28D: arms 0:14, hands 14:28):
  overlap  - disagreement between consecutive chunks over their shared future frames
  seam     - jump at each seam of the chunks spliced as executed (first 12 frames of each), vs the recorded step
  step     - frame-to-frame step inside the spliced sequence away from seams, and the recording's step
  curv     - second difference inside each predicted chunk vs the recording's
Usage: python openloop_smooth.py POLICY_PATH DATASET_ROOT MODE(random|fixed) EP [EP ...]
"""

import json
import sys

import numpy as np
import torch
from sonic_policy_streamer import ChunkPolicy, dataset_obs

from lerobot.datasets.lerobot_dataset import LeRobotDataset

path, root, mode = sys.argv[1], sys.argv[2], sys.argv[3]
episodes = [int(e) for e in sys.argv[4:]]
R = 12
cp = ChunkPolicy(path, "cuda")
with open(f"{root}/meta/stats.json") as f:
    std = np.asarray(json.load(f)["action"]["std"], np.float32).clip(1e-6)
parts = (
    {"tokens": slice(0, 64), "hands": slice(64, 78)}
    if len(std) == 78
    else {"arms": slice(0, 14), "hands": slice(14, 28)}
)
acc = {p: {k: [] for k in ("overlap", "seam", "step", "gt_step", "curv", "gt_curv")} for p in parts}
for ep in episodes:
    ds = LeRobotDataset("local/x", root=root, episodes=[ep], video_backend="torchcodec")
    gt = np.stack([ds[i]["action"].numpy() for i in range(len(ds))]) / std
    chunks, starts = [], []
    for k in range(0, len(ds) - 40, R):
        states, imgs, task = dataset_obs(ds, k, cp.n_obs, cp.image_keys)
        if mode == "fixed":
            torch.manual_seed(0)
        cp.policy.reset()
        chunks.append(cp.chunk(states, imgs, task) / std)
        starts.append(k)
    H = min(len(c) for c in chunks)
    spliced = np.concatenate([c[:R] for c in chunks])
    for p, sl in parts.items():
        a = acc[p]
        for c0, c1 in zip(chunks[:-1], chunks[1:], strict=True):
            a["overlap"].append(np.abs(c1[: H - R, sl] - c0[R:H, sl]).mean())
        d = np.abs(np.diff(spliced[:, sl], axis=0)).mean(1)
        seam = np.arange(R - 1, len(d), R)
        a["seam"] += d[seam].tolist()
        a["step"] += np.delete(d, seam).tolist()
        g = gt[starts[0] : starts[0] + len(spliced), sl]
        a["gt_step"] += np.abs(np.diff(g, axis=0)).mean(1).tolist()
        for c in chunks:
            a["curv"] += np.abs(c[2:H, sl] - 2 * c[1 : H - 1, sl] + c[: H - 2, sl]).mean(1).tolist()
        a["gt_curv"] += np.abs(g[2:] - 2 * g[1:-1] + g[:-2]).mean(1).tolist()
print(f"{cp.policy.config.type} {mode}: {len(episodes)} held-out eps, chunk {H}")
for p, a in acc.items():
    m = {k: float(np.mean(v)) for k, v in a.items()}
    print(
        f"  {p:6s} overlap disagreement {m['overlap']:.3f} | seam jump {m['seam']:.3f} vs step {m['step']:.3f}"
        f" (recording step {m['gt_step']:.3f}) | within-chunk curvature {m['curv']:.4f} (recording {m['gt_curv']:.4f})"
    )
