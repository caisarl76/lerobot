"""Normalization statistics for Xiaomi-Robotics-1 training on a LeRobot dataset.

XR-1 (`xr1/tools/compute_normalize.py`) normalizes each step k of the 30-step chunk with its own mean/std of the
packed relative actions a[t+k] - s[t], over all complete windows, and the state with its q01/q99. Here each
action dim is made relative to the state dim given by --relative (-1 keeps it absolute, e.g. SONIC tokens,
which have no state). Held-out episodes are excluded, and so are windows whose anchor state is out of range: the
Unitree joint28 state has 266 corrupt right-hand frames (|value| up to ~3000 rad) in 39 episodes, which would
otherwise dominate the std of the relative hand actions. The valid state range is stored as `state_min/max`; the
model clamps the state to it before making actions relative.

Usage:
  python xr1_action_stats.py --root /run-output/datasets/joint28 --out stats.json \
      --exclude-json /run-output/configs/heldout_sonic78_nolimit_5pct.json --relative joint28
  --relative: "joint28" (all 28 dims relative), "sonic78" (tokens absolute, hands 64:78 relative to state 14:28),
              "none", or a comma-separated list of state indices / -1.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def relative_map(spec: str, action_dim: int) -> list[int]:
    if spec == "joint28":
        return list(range(action_dim))
    if spec == "sonic78":
        return [-1] * 64 + list(range(14, 28))
    if spec == "none":
        return [-1] * action_dim
    return [int(x) for x in spec.split(",")]


def load_columns(root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    episodes, states, actions = [], [], []
    for path in sorted((root / "data").rglob("*.parquet")):
        table = pq.read_table(path, columns=["episode_index", "frame_index", "observation.state", "action"])
        episodes.append(table.column("episode_index").to_numpy())
        order = table.column("frame_index").to_numpy()
        states.append(
            np.stack(table.column("observation.state").to_numpy(zero_copy_only=False)).astype(np.float32)
        )
        actions.append(np.stack(table.column("action").to_numpy(zero_copy_only=False)).astype(np.float32))
        if np.any(np.diff(order[episodes[-1] == episodes[-1][0]]) < 0):
            raise ValueError(f"{path}: frames are not sorted within an episode")
    return np.concatenate(episodes), np.concatenate(states), np.concatenate(actions)


def compute(root: Path, exclude: set[int], rel: list[int], chunk: int, max_abs_state: float) -> dict:
    episode_index, states, actions = load_columns(root)
    if len(rel) != actions.shape[1]:
        raise ValueError(f"--relative has {len(rel)} entries for {actions.shape[1]} action dims")
    rel_arr = np.array(rel)
    is_rel = rel_arr >= 0
    total = np.zeros((chunk, actions.shape[1]), np.float64)
    total_sq = np.zeros_like(total)
    windows = skipped = 0
    keep_frames = np.zeros(len(episode_index), bool)
    valid_state = (np.abs(states) <= max_abs_state).all(axis=1)
    boundaries = np.flatnonzero(np.diff(episode_index)) + 1
    starts = np.concatenate([[0], boundaries])
    ends = np.concatenate([boundaries, [len(episode_index)]])
    for start, end in zip(starts, ends, strict=True):
        if int(episode_index[start]) in exclude:
            continue
        keep_frames[start:end] = True
        a, s = actions[start:end].astype(np.float64), states[start:end].astype(np.float64)
        n = len(a) - chunk + 1
        if n <= 0:
            continue
        ok = valid_state[start : start + n]
        offset = np.zeros((n, a.shape[1]))
        offset[:, is_rel] = s[:n, rel_arr[is_rel]]
        for k in range(chunk):
            x = (a[k : k + n] - offset)[ok]
            total[k] += x.sum(0)
            total_sq[k] += (x * x).sum(0)
        windows += int(ok.sum())
        skipped += int((~ok).sum())
    mean = total / windows
    std = np.sqrt(np.maximum(total_sq / windows - mean * mean, 0.0))
    kept = states[keep_frames & valid_state]
    return {
        "dataset_root": str(root),
        "windows": int(windows),
        "windows_skipped_bad_state": int(skipped),
        "max_abs_state": max_abs_state,
        "frames": int(keep_frames.sum()),
        "excluded_episodes": sorted(exclude),
        "relative_action_state_indices": rel,
        "action_mean": mean.tolist(),
        "action_std": std.tolist(),
        "state_q01": np.quantile(kept, 0.01, axis=0).tolist(),
        "state_q99": np.quantile(kept, 0.99, axis=0).tolist(),
        "state_min": kept.min(axis=0).tolist(),
        "state_max": kept.max(axis=0).tolist(),
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--exclude-json", type=Path, help="held-out split JSON ('episodes', 'always_excluded')"
    )
    parser.add_argument("--exclude", type=int, nargs="*", default=[])
    parser.add_argument("--relative", default="joint28")
    parser.add_argument("--chunk", type=int, default=30)
    parser.add_argument("--max-abs-state", type=float, default=3.2, help="rad; frames beyond it are corrupt")
    args = parser.parse_args()
    exclude = set(args.exclude)
    if args.exclude_json:
        split = json.loads(args.exclude_json.read_text())
        exclude |= set(split["episodes"]) | set(split.get("always_excluded", []))
    info = json.loads((args.root / "meta/info.json").read_text())
    rel = relative_map(args.relative, info["features"]["action"]["shape"][0])
    stats = compute(args.root, exclude, rel, args.chunk, args.max_abs_state)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(stats) + "\n")
    std = np.array(stats["action_std"])
    print(
        f"{args.out}: {stats['windows']} windows, {stats['frames']} frames, {len(exclude)} excluded episodes"
    )
    print("std step 0 / 29 (min, max):", std[0].min(), std[0].max(), std[-1].min(), std[-1].max())


if __name__ == "__main__":
    main()
