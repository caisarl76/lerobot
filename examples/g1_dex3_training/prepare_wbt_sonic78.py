"""Convert a GR00T-WBC SONIC teleop recording (LeRobot v2.1, 50 Hz) to the sonic78 training format, kept at 50 Hz.

The recording already holds the SONIC motion tokens that drove the real robot (action.motion_token, 64) and the
measured robot state, which is the SONIC-reached pose, so no relabeling is needed. Joints are matched by name:
  observation.state (28)  = arms 14 + Dex3 hands 14 from observation.state, in sonic_targets.ACTION_NAMES order
  action (78)             = action.motion_token 64 + hand targets 14 from action.wbc, same hand order
  observation.images.egocentric = ego_view (the HE models' camera key, so HE checkpoints can be fine-tuned)
Writes a new LeRobot v3 dataset (videos re-encoded, stats with quantiles) and <out>.heldout.json (5 % of the
episodes, seed 0) for the configs' dataset.exclude_episodes.

Usage: python prepare_wbt_sonic78.py SRC OUT [REPO_ID]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import av
import numpy as np
import pandas as pd
from sonic_targets import ACTION_NAMES

from lerobot.datasets.lerobot_dataset import LeRobotDataset

src, out = Path(sys.argv[1]), Path(sys.argv[2])
repo_id = sys.argv[3] if len(sys.argv) > 3 else f"local/{out.name}"
info = json.loads((src / "meta/info.json").read_text())
assert info["fps"] == 50, info["fps"]
state_names = info["features"]["observation.state"]["names"]
wbc_names = info["features"]["action.wbc"]["names"]
state_idx = [state_names.index(n) for n in ACTION_NAMES]
hand_idx = [wbc_names.index(n) for n in ACTION_NAMES[14:]]
tasks = {
    t["task_index"]: t["task"] for t in map(json.loads, (src / "meta/tasks.jsonl").read_text().splitlines())
}

h, w, _ = info["features"]["observation.images.ego_view"]["shape"]
ds = LeRobotDataset.create(
    repo_id,
    fps=50,
    root=out,
    robot_type="unitree_g1_dex3_sonic",
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
n_eps = info["total_episodes"]
for ep in range(n_eps):
    df = pd.read_parquet(src / info["data_path"].format(episode_chunk=ep // 1000, episode_index=ep))
    video = src / info["video_path"].format(
        episode_chunk=ep // 1000, video_key="observation.images.ego_view", episode_index=ep
    )
    with av.open(str(video)) as c:
        frames = [f.to_ndarray(format="rgb24") for f in c.decode(video=0)]
    assert len(frames) == len(df), (ep, len(frames), len(df))
    state = np.stack(df["observation.state"])[:, state_idx].astype(np.float32)
    action = np.concatenate(
        [np.stack(df["action.motion_token"]), np.stack(df["action.wbc"])[:, hand_idx]], axis=1
    ).astype(np.float32)
    for i in range(len(df)):
        ds.add_frame(
            {
                "observation.state": state[i],
                "action": action[i],
                "observation.images.egocentric": frames[i],
                "task": tasks[int(df["task_index"].iloc[i])],
            }
        )
    ds.save_episode()
    print(f"episode {ep}: {len(df)} frames", flush=True)
ds.finalize()

heldout = sorted(np.random.default_rng(0).choice(n_eps, max(1, round(0.05 * n_eps)), replace=False).tolist())
Path(f"{out}.heldout.json").write_text(json.dumps({"exclude_episodes": heldout}) + "\n")
print(f"done: {n_eps} episodes -> {out}; held out {heldout}")
