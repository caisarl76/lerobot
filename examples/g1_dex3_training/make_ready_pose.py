"""Ready pose for sonic_policy_streamer.py --ready-pose from a recorded sonic78 episode: the recorded SONIC token (64)
and Dex3 hand targets (14) at the stillest frame of the episode's first 0.5 s, where the operator holds the ready
pose (hand over the table) before an attempt. The 10-19 handover recordings return there between attempts.

Usage: python make_ready_pose.py DATASET_ROOT EPISODE OUT.npz
  e.g. make_ready_pose.py /mnt/data/jihun/datasets/G1_WBT_GR00T/handover_bottle_merged_sonic78_50hz_261020 76 \\
         ready_pose_261020_ep76.npz   (episode 76: the attempt start closest to the median of the 39 cut 10-19 attempts)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def ready_pose(root: Path, episode: int) -> dict:
    files = sorted(root.glob("data/*/*.parquet"))
    df = pd.concat(pd.read_parquet(f) for f in files)
    g = df[df.episode_index == episode]
    if g.empty:
        raise SystemExit(f"episode {episode} not in {root}")
    q, act, t = (
        np.stack(g["observation.state"].to_numpy()),
        np.stack(g["action"].to_numpy()),
        g["timestamp"].to_numpy(),
    )
    n = int(np.searchsorted(t - t[0], 0.5))
    speed = np.r_[
        0, np.linalg.norm(np.diff(q[:, :14], axis=0) / np.maximum(np.diff(t), 1e-3)[:, None], axis=1)
    ]
    k = int(np.argmin(speed[1:n])) + 1
    return {
        "token": act[k, :64].astype(np.float32),
        "hands": act[k, 64:78].astype(np.float32),
        "state": q[k].astype(np.float32),
        "meta": json.dumps(
            {"dataset": str(root), "episode": episode, "frame": k, "arm_speed": float(speed[k])}
        ),
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("root", type=Path)
    p.add_argument("episode", type=int)
    p.add_argument("out", type=Path)
    a = p.parse_args()
    pose = ready_pose(a.root, a.episode)
    np.savez(a.out, **pose)
    print(f"{a.out}: {pose['meta']}")
