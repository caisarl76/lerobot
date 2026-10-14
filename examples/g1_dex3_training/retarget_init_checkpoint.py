"""Point a fine-tuning start checkpoint at a new dataset's normalization statistics, in place.

lerobot_train only replaces the stats of the standard (un)normalizer steps when it starts from a checkpoint, so
some policies would keep the old dataset's statistics: GR00T keeps them in its own processor steps and XR-1 in
persistent model buffers. This rebuilds the checkpoint's processors from the new dataset's stats (as a run from
the base weights would build them) and, for XR-1, reloads the model's stats from XR1_STATS and re-saves it.

Usage (training container): python retarget_init_checkpoint.py INIT_DIR DATASET_ROOT [XR1_STATS]
"""

from __future__ import annotations

import sys
from pathlib import Path

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.factory import get_policy_class, make_pre_post_processors

init, root = Path(sys.argv[1]), sys.argv[2]
cfg = PreTrainedConfig.from_pretrained(init)
cfg.device = "cpu"
stats = LeRobotDatasetMetadata("local/retarget", root=root).stats
if cfg.type == "xiaomi_robotics":
    cfg.stats_path = sys.argv[3]
    policy = get_policy_class(cfg.type).from_pretrained(init, config=cfg)
    policy.model.load_stats(cfg.stats_path)
    policy.save_pretrained(init)
pre, post = make_pre_post_processors(cfg, dataset_stats=stats)
for f in init.glob("policy_*processor*"):
    f.unlink()
pre.save_pretrained(init)
post.save_pretrained(init)
print(
    f"{init}: processors rebuilt from {root}"
    + (f", XR-1 stats from {cfg.stats_path}" if len(sys.argv) > 3 else "")
)
