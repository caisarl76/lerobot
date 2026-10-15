"""Convert GR00T-WBC SONIC teleop recordings (LeRobot v2.1, 50 Hz) to one sonic78 training dataset, kept at 50 Hz.

The recording already holds the SONIC motion tokens that drove the real robot (action.motion_token, 64) and the
measured robot state, which is the SONIC-reached pose, so no relabeling is needed. Joints are matched by name:
  observation.state (28)  = arms 14 + Dex3 hands 14 from observation.state, in sonic_targets.ACTION_NAMES order
  action (78)             = action.motion_token 64 + hand targets 14 from action.wbc, same hand order
  observation.images.egocentric = ego_view (the HE models' camera key, so HE checkpoints can be fine-tuned)
Sources are written in the given order. --annotations takes the dataset visualizer's review (workspace
runs/<id>/run.json, matched to a source by its source_root): Failed episodes are dropped and every kept frame range
between excluded intervals becomes its own episode, so no action chunk spans a cut. Writes a new LeRobot v3 dataset
(videos re-encoded, stats with quantiles), <out>.segments.json (source episode and frame range of each output
episode) and <out>.heldout.json (--heldout, or 5 % of the episodes, seed 0) for the configs' dataset.exclude_episodes.

Usage: python prepare_wbt_sonic78.py SRC [SRC ...] OUT [--annotations RUN_JSON ...] [--task TEXT] [--heldout EP ...]
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import av
import numpy as np
import pandas as pd
from sonic_targets import ACTION_NAMES

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def kept_segments(length: int, review: dict | None) -> list[tuple[int, int]]:
    """Half-open frame ranges left after the review's excluded intervals; [] for a Failed episode."""
    if review is None:
        return [(0, length)]
    if review.get("decision") == "delete":
        return []
    segments, start = [], 0
    for iv in sorted(review.get("excluded_intervals", []), key=lambda iv: iv["start_frame"]):
        if iv["start_frame"] > start:
            segments.append((start, iv["start_frame"]))
        start = max(start, iv["end_frame"])
    if start < length:
        segments.append((start, length))
    return segments


p = argparse.ArgumentParser()
p.add_argument("sources", nargs="+", type=Path)
p.add_argument("out", type=Path)
p.add_argument("--repo-id")
p.add_argument("--annotations", nargs="*", type=Path, default=[])
p.add_argument("--task", help="replace every episode's task text")
p.add_argument("--heldout", nargs="*", type=int, help="output episode indices to hold out")
args = p.parse_args()

reviews = {}
for path in args.annotations:
    run = json.loads(path.read_text())
    src = Path(run["source_root"]).resolve()
    meta = src / "meta/episodes.jsonl"
    assert (
        hashlib.sha256(meta.read_bytes()).hexdigest() == run["source_file_hashes"]["meta/episodes.jsonl"]
    ), f"{meta} changed since it was reviewed in {path}"
    reviews[src] = {int(ep): r for ep, r in run["episodes"].items()}
assert set(reviews) <= {s.resolve() for s in args.sources}, "an annotation run matches no source"

infos = [json.loads((src / "meta/info.json").read_text()) for src in args.sources]
h, w, _ = infos[0]["features"]["observation.images.ego_view"]["shape"]
ds = LeRobotDataset.create(
    args.repo_id or f"local/{args.out.name}",
    fps=50,
    root=args.out,
    robot_type="unitree_g1_dex3_sonic",
    streaming_encoding=True,
    features={
        "observation.state": {"dtype": "float32", "shape": (28,), "names": list(ACTION_NAMES)},
        "action": {
            "dtype": "float32",
            "shape": (78,),
            "names": [f"sonic_token_{i}" for i in range(64)] + list(ACTION_NAMES[14:]),
        },
        "observation.images.egocentric": {
            "dtype": "video",
            "shape": (h, w, 3),
            "names": ["height", "width", "channels"],
        },
    },
)
segments_out = []
for src, info in zip(args.sources, infos, strict=True):
    assert info["fps"] == 50, (src, info["fps"])
    state_names = info["features"]["observation.state"]["names"]
    wbc_names = info["features"]["action.wbc"]["names"]
    state_idx = [state_names.index(n) for n in ACTION_NAMES]
    hand_idx = [wbc_names.index(n) for n in ACTION_NAMES[14:]]
    tasks = {
        t["task_index"]: t["task"]
        for t in map(json.loads, (src / "meta/tasks.jsonl").read_text().splitlines())
    }
    review = reviews.get(src.resolve())
    for ep in range(info["total_episodes"]):
        df = pd.read_parquet(src / info["data_path"].format(episode_chunk=ep // 1000, episode_index=ep))
        segments = kept_segments(len(df), None if review is None else review[ep])
        if not segments:
            print(f"{src.name} episode {ep}: dropped (Failed)", flush=True)
            continue
        video = src / info["video_path"].format(
            episode_chunk=ep // 1000, video_key="observation.images.ego_view", episode_index=ep
        )
        with av.open(str(video)) as c:
            frames = [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]
        assert len(frames) == len(df), (src, ep, len(frames), len(df))
        state = np.stack(df["observation.state"])[:, state_idx].astype(np.float32)
        action = np.concatenate(
            [np.stack(df["action.motion_token"]), np.stack(df["action.wbc"])[:, hand_idx]], axis=1
        ).astype(np.float32)
        for start, end in segments:
            for i in range(start, end):
                ds.add_frame(
                    {
                        "observation.state": state[i],
                        "action": action[i],
                        "observation.images.egocentric": frames[i],
                        "task": args.task or tasks[int(df["task_index"].iloc[i])],
                    }
                )
            ds.save_episode()
            segments_out.append({"source": str(src), "episode": ep, "start_frame": start, "end_frame": end})
            print(f"{src.name} episode {ep} [{start}, {end}) -> {len(segments_out) - 1}", flush=True)
ds.finalize()

n_eps = len(segments_out)
heldout = (
    sorted(args.heldout)
    if args.heldout is not None
    else sorted(np.random.default_rng(0).choice(n_eps, max(1, round(0.05 * n_eps)), replace=False).tolist())
)
Path(f"{args.out}.segments.json").write_text(json.dumps(segments_out, indent=1) + "\n")
Path(f"{args.out}.heldout.json").write_text(json.dumps({"exclude_episodes": heldout}) + "\n")
print(f"done: {n_eps} episodes -> {args.out}; held out {heldout}")
