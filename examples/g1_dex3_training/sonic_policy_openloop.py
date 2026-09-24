"""Open-loop check of a SONIC-token policy on one recorded episode, plus the streamed run's state inputs.

1) Feed the dataset's own observation.state + images at each frame; compare the policy's first chunk action to the
   stored 78D action (tokens and Dex3 hands). This isolates policy quality from the robot/deploy.
2) If a streamer log is given, compare the robot state the streamer fed the policy with the dataset's
   observation.state at the same frame (closed-loop input drift).
Usage: python sonic_policy_openloop.py POLICY_PATH SONIC_ROOT EPISODE [STREAMER_NPZ]
"""

import json
import sys

import numpy as np
import torch

sys.path.insert(0, "/workspace/lerobot/examples/g1_dex3_training")
from sonic_policy_streamer import ChunkPolicy, dataset_obs

from lerobot.datasets.lerobot_dataset import LeRobotDataset

path, root, ep = sys.argv[1], sys.argv[2], int(sys.argv[3])
policy = ChunkPolicy(path, "cuda" if torch.cuda.is_available() else "cpu")
ds = LeRobotDataset("local/g1_dex3_all_sonic78", root=root, episodes=[ep], video_backend="torchcodec")
names = ds.meta.features["observation.state"]["names"]
tok_err, hand_err, states = [], [], []
for k in range(0, len(ds), 5):
    hist_states, hist_imgs, task = dataset_obs(ds, k, policy.n_obs, policy.image_keys)
    states.append(hist_states[-1])
    pred = policy.chunk(hist_states, hist_imgs, task)[0]
    target = ds[k]["action"].numpy()
    tok_err.append(np.abs(pred[:64] - target[:64]))
    hand_err.append(np.abs(pred[64:] - target[64:]))


def stats(x):
    x = np.asarray(x, float).ravel()
    return {"p50": round(float(np.median(x)), 3), "p95": round(float(np.percentile(x, 95)), 3)}


out = {
    "state_names": names,
    "openloop_token_abs": stats(tok_err),
    "openloop_hands_abs_rad": stats(hand_err),
    "openloop_hands_abs_rad_per_joint_p50": np.median(np.array(hand_err), 0).round(3).tolist(),
}
if len(sys.argv) > 4:
    s = np.load(sys.argv[4], allow_pickle=True)
    m = s["phase"] == "episode"
    frames, fed = s["frame"][m], s["state"][m]
    ds_state = np.stack([ds[int(k)]["observation.state"].numpy() for k in frames[::10]])
    diff = np.abs(fed[::10] - ds_state)
    out["closedloop_state_vs_dataset_abs_rad"] = {"arms": stats(diff[:, :14]), "hands": stats(diff[:, 14:])}
    out["closedloop_state_abs_rad_per_joint_p50"] = np.median(diff, 0).round(3).tolist()
print(json.dumps(out, indent=1))
