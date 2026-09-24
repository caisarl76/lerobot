"""Which part of the closed-loop robot state throws a SONIC-token policy off? (open-loop counterfactual)

At each scored frame the policy gets the dataset images plus observation.state built four ways: dataset state;
dataset hands with the streamed run's sim arm state; dataset arms with the sim hand state; the full sim state.
Error = |policy first action - stored 78D action| (tokens, Dex3 hands). Right thumb_0 is also tested alone.
Usage: python sonic_state_ablation.py POLICY_PATH SONIC_ROOT EPISODE STREAMER_NPZ
"""

import json
import sys

import numpy as np
import torch

sys.path.insert(0, "/workspace/lerobot/examples/g1_dex3_training")
from sonic_policy_streamer import ChunkPolicy

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def _obs(ds, k, n):
    """Last n dataset frames up to k (oldest first): states, images, task."""
    items = [ds[max(k - i, 0)] for i in reversed(range(n))]
    imgs = [
        {key: (it[key].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8) for key in policy.image_keys}
        for it in items
    ]
    return (
        [it["observation.state"].numpy().astype(np.float32) for it in items],
        imgs,
        str(items[-1].get("task", "")),
    )


path, root, ep, log = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
policy = ChunkPolicy(path, "cuda" if torch.cuda.is_available() else "cpu")
ds = LeRobotDataset("local/g1_dex3_all_sonic78", root=root, episodes=[ep], video_backend="torchcodec")
z = np.load(log, allow_pickle=True)
m = z["phase"] == "episode"
sim_state = {int(k): s for k, s in zip(z["frame"][m], z["state"][m], strict=True)}  # last tick per frame
variants = {
    "dataset state": lambda d, s: d,
    "sim arms": lambda d, s: np.r_[s[:14], d[14:]],
    "sim hands": lambda d, s: np.r_[d[:14], s[14:]],
    "sim right thumb_0 only": lambda d, s: np.r_[d[:21], s[21:22], d[22:]],
    "full sim state": lambda d, s: s,
}
err = {v: {"tokens": [], "hands": []} for v in variants}
for k in sorted(sim_state)[::5]:
    hist_states, hist_imgs, task = _obs(ds, k, policy.n_obs)
    d = hist_states[-1]
    target = ds[k]["action"].numpy()
    for name, build in variants.items():  # only the newest state is swapped; history stays recorded
        states = hist_states[:-1] + [build(d, sim_state[k]).astype(np.float32)]
        pred = policy.chunk(states, hist_imgs, task)[0]
        err[name]["tokens"].append(np.abs(pred[:64] - target[:64]).mean())
        err[name]["hands"].append(np.abs(pred[64:] - target[64:]).mean())
out = {
    name: {part: round(float(np.median(v)), 3) for part, v in e.items()} | {"n": len(e["tokens"])}
    for name, e in err.items()
}
print(json.dumps(out, indent=1))
