"""Ready pose for sonic_policy_streamer.py --ready-pose from a recorded sonic78 episode: the recorded SONIC token (64)
and Dex3 hand targets (14) at the stillest frame of the episode's first 0.5 s, where the operator holds the ready
pose (hand over the table) before an attempt. The 10-19 handover recordings return there between attempts.

--raise-episode adds the session start/end paths (2026-10-24): the recorded raise of an episode that starts standing
(frame 0 up to the frame closest to the ready pose before its first grasp), played forward as recorded
(raise_tokens/raise_hands), and that measured joint path reversed and re-encoded with the SONIC encoder for lowering
(lower_tokens/lower_hands). A reversed token sequence is not a reversed motion (tokens encode ~1 s of future motion),
so the reverse is encoded from joints, as the table shutdown path is.

Usage: python make_ready_pose.py DATASET_ROOT EPISODE OUT.npz [--raise-episode E] [--sonic-dir DIR]
  e.g. make_ready_pose.py /mnt/data/jihun/datasets/G1_WBT_GR00T/handover_bottle_merged_sonic78_50hz_261020 76 \\
         ready_pose_261020_ep76.npz --raise-episode 11
       (episode 76: the attempt start closest to the median of the 39 cut 10-19 attempts; episode 11: the old-set
        raise that comes closest to that ready pose, 0.50 rad at 1.64 s)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from lerobot.utils.import_utils import _pandas_available, require_package

if TYPE_CHECKING or _pandas_available:
    import pandas as pd


def load_episode(root: Path, episode: int):
    require_package("pandas", extra="dataset")
    df = pd.concat(pd.read_parquet(f) for f in sorted(root.glob("data/*/*.parquet")))
    return df[df.episode_index == episode]


def raise_path(root: Path, episode: int, ready_state: np.ndarray, sonic_dir: Path) -> dict:
    """Recorded raise of `episode` up to its frame closest to `ready_state` (arms) before the first grasp, and the
    reversed measured joint path re-encoded for lowering, both at the dataset's 50 Hz."""
    from sonic_targets import SonicEncoder, joint_chunk_to_sonic, load_joint_limits

    g = load_episode(root, episode)
    if g.empty:
        raise SystemExit(f"episode {episode} not in {root}")
    q, act = np.stack(g["observation.state"].to_numpy()), np.stack(g["action"].to_numpy())
    closed = q[:, 21:28].mean(1) > 0.19
    grasp = np.flatnonzero(closed & (np.cumsum(~closed) > 0))  # first close after the hand was open
    end = int(grasp[0]) if len(grasp) else len(q)
    k = int(np.argmin(np.linalg.norm(q[:end, :14] - ready_state[:14], axis=1)))
    encoder = SonicEncoder(
        sonic_dir / "policy/sonic_v1_1/model_encoder.onnx",
        sonic_dir / "policy/sonic_v1_1/observation_config.yaml",
    )
    limits = load_joint_limits(sonic_dir / "g1/g1_29dof_with_hand.xml")
    down = q[: k + 1, :28][::-1].astype(np.float32)
    lower = joint_chunk_to_sonic(
        down, limits, encoder, fps=50
    )  # the last pose is held past the end (look-ahead)
    return {
        "raise_tokens": act[: k + 1, :64].astype(np.float32),
        "raise_hands": act[: k + 1, 64:78].astype(np.float32),
        "raise_state": q[: k + 1].astype(np.float32),
        "lower_tokens": lower[:, :64].astype(np.float32),
        "lower_hands": lower[:, 64:78].astype(np.float32),
        "path_meta": json.dumps(
            {
                "raise_episode": episode,
                "frames": [0, k],
                "fps": 50,
                "closest_to_ready_rad": float(np.linalg.norm(q[k, :14] - ready_state[:14])),
                "lower": "reversed measured joints, re-encoded (joint_chunk_to_sonic, speed limits off)",
            }
        ),
    }


def ready_pose(root: Path, episode: int) -> dict:
    g = load_episode(root, episode)
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
    p.add_argument(
        "--raise-episode", type=int, help="add the raise / lower paths from this standing-start episode"
    )
    p.add_argument(
        "--sonic-dir",
        type=Path,
        default=Path.home() / "work/GR00T-WholeBodyControl/gear_sonic_deploy",
        help="gear_sonic_deploy checkout holding policy/sonic_v1_1 (encoder) and g1/g1_29dof_with_hand.xml",
    )
    a = p.parse_args()
    pose = ready_pose(a.root, a.episode)
    if a.raise_episode is not None:
        pose |= raise_path(a.root, a.raise_episode, pose["state"], a.sonic_dir)
    np.savez(a.out, **pose)
    print(f"{a.out}: {pose['meta']}" + (f" | {pose['path_meta']}" if "path_meta" in pose else ""))
