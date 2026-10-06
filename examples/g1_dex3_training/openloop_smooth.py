"""Open-loop smoothness of predicted chunks on held-out episodes (no robot, no closed loop).

Every 12 frames (0.4 s replan at 30 Hz) predict a chunk from the recorded observation, then measure, in units of the
per-dimension action std (78D: tokens 0:64, hands 64:78; 28D: arms 0:14, hands 14:28; dimensions with zero std in the
dataset are skipped):
  overlap  - disagreement between consecutive chunks over their shared future frames
  seam     - jump at each seam of the chunks spliced as executed (first 12 frames of each), vs the recorded step
  step     - frame-to-frame step inside the spliced sequence away from seams, and the recording's step
  curv     - second difference inside each predicted chunk vs the recording's
Usage: python openloop_smooth.py POLICY_PATH DATASET_ROOT MODE(random|fixed|rtc) EP [EP ...]
  random - fresh sampling noise per chunk; fixed - the same noise seed for every chunk;
  rtc    - real-time chunking: each chunk is generated from the previous chunk's unexecuted tail (normalized), with
           RTC_DELAY frames (default 4, ~0.13 s) frozen; GR00T uses its native overlap inpainting, Pi0.5 LeRobot's
           guided RTC (enabled here). Random noise.
  temp:<t> - initial flow noise scaled by t (temp:0 = deterministic "mean-path" sample); avg:<K> - mean of K samples.
  Modes combine with "+", e.g. rtc+fixed, rtc+temp:0.5.
Also reports err: mean |executed - recorded| over the spliced executed frames (accuracy, std units).
REPLAN=<frames> overrides the 12-frame replan interval (needed when a policy's chunk is shorter, e.g. VLA-JEPA's 7).
BACKBONE_DTYPE=bfloat16 casts GR00T's frozen backbone (fits a 12 GB GPU).
"""

import json
import os
import sys

import numpy as np
import torch
from sonic_policy_streamer import ChunkPolicy, dataset_obs

from lerobot.datasets.lerobot_dataset import LeRobotDataset

path, root, mode = sys.argv[1], sys.argv[2], sys.argv[3]
flags = {m.split(":")[0]: (m.split(":")[1] if ":" in m else None) for m in mode.split("+")}
episodes = [int(e) for e in sys.argv[4:]]
R = int(os.environ.get("REPLAN", 12))
D = int(os.environ.get("RTC_DELAY", 4))
cp = ChunkPolicy(
    path,
    "cuda",
    os.environ.get("BACKBONE_DTYPE"),
    noise_scale=float(flags["temp"]) if "temp" in flags else None,
)
K = int(flags["avg"]) if "avg" in flags else 1
if "rtc" in flags and cp.policy.config.type == "pi05":
    from lerobot.policies.rtc.configuration_rtc import RTCConfig

    cp.policy.config.rtc_config = RTCConfig()
    cp.policy.init_rtc_processor()
with open(f"{root}/meta/stats.json") as f:
    std = np.asarray(json.load(f)["action"]["std"], np.float32)
# Dimensions that never move in the dataset (std 0, e.g. a hand joint left at 0) are left out of every metric:
# divided by a clipped std, any tiny prediction offset (e.g. state-relative XR-1 hands) would dominate the means.
live = std > 1e-6
std = std.clip(1e-6)
split = {"tokens": (0, 64), "hands": (64, 78)} if len(std) == 78 else {"arms": (0, 14), "hands": (14, 28)}
parts = {p: a + np.flatnonzero(live[a:b]) for p, (a, b) in split.items()}
acc = {p: {k: [] for k in ("overlap", "seam", "step", "gt_step", "curv", "gt_curv", "err")} for p in parts}
for ep in episodes:
    ds = LeRobotDataset("local/x", root=root, episodes=[ep], video_backend="torchcodec")
    gt = np.stack([ds[i]["action"].numpy() for i in range(len(ds))]) / std
    chunks, starts = [], []
    for k in range(0, len(ds) - 40, R):
        states, imgs, task = dataset_obs(ds, k, cp.n_obs, cp.image_keys)
        if "fixed" in flags:
            torch.manual_seed(0)
        cp.policy.reset()
        rtc = {}
        if "rtc" in flags and chunks:
            prev = cp.last_raw[:, R:]
            rtc = {"prev_chunk_left_over": prev, "inference_delay": D, "execution_horizon": prev.shape[1]}
        chunks.append(np.mean([cp.chunk(states, imgs, task, **rtc) for _ in range(K)], axis=0) / std)
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
        a["err"] += np.abs(spliced[: len(g), sl] - g).mean(1).tolist()
        for c in chunks:
            a["curv"] += np.abs(c[2:H, sl] - 2 * c[1 : H - 1, sl] + c[: H - 2, sl]).mean(1).tolist()
        a["gt_curv"] += np.abs(g[2:] - 2 * g[1:-1] + g[:-2]).mean(1).tolist()
print(f"{cp.policy.config.type} {mode}: {len(episodes)} held-out eps, chunk {H}")
for p, a in acc.items():
    m = {k: float(np.mean(v)) for k, v in a.items()}
    print(
        f"  {p:6s} overlap disagreement {m['overlap']:.3f} | seam jump {m['seam']:.3f} vs step {m['step']:.3f}"
        f" (recording step {m['gt_step']:.3f}) | within-chunk curvature {m['curv']:.4f} (recording {m['gt_curv']:.4f})"
        f" | err {m['err']:.3f}"
    )
