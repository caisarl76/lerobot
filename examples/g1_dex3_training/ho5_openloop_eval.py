"""Open-loop eval of a ho5 policy (any of the 7 types, Unitree or HE): held-out episodes vs an equal-size training sample.

Feeds recorded observations (with the policy's own trained observation window) every STRIDE frames, predicts one chunk,
and compares it to the recorded 78D action chunk (64 SONIC tokens + 14 Dex3 hand joints) at several horizons.
Errors are raw and divided by the per-dim dataset std. Baseline: repeating the previous recorded action.
Usage: python ho5_openloop_eval.py RUN_DIR OUT_JSON [STRIDE] [BATCH]
"""

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.factory import resolve_delta_timestamps
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.processor.rename_processor import rename_batch_keys

HORIZONS = (
    0,
    6,
    10,
    29,
    49,
    99,
)  # chunk sizes: vla_jepa 7, molmoact2 30, diffusion/fastwam 32, groot 40, pi05 50, act 100
run, out_path = Path(sys.argv[1]), Path(sys.argv[2])
STRIDE = int(sys.argv[3]) if len(sys.argv) > 3 else 5
BATCH = int(sys.argv[4]) if len(sys.argv) > 4 else 32
ckpt = run / "checkpoints/last/pretrained_model"
train_cfg = json.loads((ckpt / "train_config.json").read_text())
ROOT, REPO = Path(train_cfg["dataset"]["root"]), train_cfg["dataset"]["repo_id"]
rename_map = train_cfg.get("rename_map") or {}
COMBINED = (
    ROOT / "meta/combined_provenance.json"
).is_file()  # Unitree episodes, then HE from he_episode_offset
if COMBINED:
    split_path = Path("/run-output/configs/heldout_combined_sonicstate_5pct.json")
else:
    split_root = Path(
        "/run-output/humanoid_everyday_g1_20260923" if "humanoid_everyday" in str(ROOT) else "/run-output"
    )
    split_path = split_root / "configs/heldout_sonic78_nolimit_5pct.json"
split = json.loads(split_path.read_text())
heldout = sorted(split["episodes"])
excluded = set(train_cfg["dataset"]["exclude_episodes"])
assert set(heldout) <= excluded, "held-out episodes were not excluded from training"

cfg = PreTrainedConfig.from_pretrained(ckpt)
cfg.device = "cuda"
if cfg.type == "vla_jepa":
    cfg.reinit_modules = None
if cfg.type == "pi05":
    cfg.adapt_action_projections = False
policy = get_policy_class(cfg.type).from_pretrained(ckpt, config=cfg).to("cuda").eval()
pre, post = make_pre_post_processors(
    cfg, pretrained_path=ckpt, postprocessor_overrides={"device_processor": {"device": "cpu"}}
)
mixed = (
    train_cfg.get("accelerator", {}).get("mixed_precision")
    if isinstance(train_cfg.get("accelerator"), dict)
    else None
)
amp_dtype = torch.float16 if mixed == "fp16" else torch.bfloat16

meta = LeRobotDataset(REPO, root=ROOT, episodes=[0], video_backend="torchcodec").meta
delta = resolve_delta_timestamps(cfg, meta, rename_map) or {}
if cfg.type in {
    "fastwam",
    "vla_jepa",
}:  # windows hold future video/state targets; inference uses only the first frame
    delta = {k: v[:1] for k, v in delta.items()}
pool = [e for e in range(meta.total_episodes) if e not in excluded]


def action_std(stats):
    return torch.tensor(np.asarray(stats["action"]["std"]), dtype=torch.float32).clamp_min(1e-6)


if COMBINED:
    # Score each source separately, normalized by that source's own action std so the numbers compare with the
    # single-dataset ho5 runs (the combined std mixes both sources).
    prov = json.loads((ROOT / "meta/combined_provenance.json").read_text())
    offset = prov["he_episode_offset"]
    SPLITS = {}
    for source, keep in (("unitree", lambda e: e < offset), ("he", lambda e: e >= offset)):
        src_std = action_std(json.loads((Path(prov["sources"][source]) / "meta/stats.json").read_text()))
        src_heldout = [e for e in heldout if keep(e)]
        src_pool = [e for e in pool if keep(e)]
        SPLITS[f"heldout_{source}"] = (src_heldout, src_std)
        SPLITS[f"train_sample_{source}"] = (
            sorted(random.Random(0).sample(src_pool, len(src_heldout))),
            src_std,
        )
else:
    std = action_std(meta.stats)
    SPLITS = {
        "heldout": (heldout, std),
        "train_sample": (sorted(random.Random(0).sample(pool, len(heldout))), std),
    }
task_of = {e: t for t, eps in split["by_task"].items() for e in eps}
H = None  # chunk length, known after the first prediction


def observation(b):
    obs = {k: v for k, v in b.items() if k not in ("action", "action_is_pad") and not k.endswith("_is_pad")}
    if cfg.type in {
        "fastwam",
        "vla_jepa",
    }:  # trained on windows starting at delta 0; deploy sees only the current frame
        for key, feature in cfg.input_features.items():
            if key in obs and obs[key].ndim == len(feature.shape) + 2:
                obs[key] = obs[key][:, 0]
    return obs


def evaluate(episodes, std):
    global H
    ds = LeRobotDataset(
        REPO,
        root=ROOT,
        episodes=episodes,
        video_backend="torchcodec",
        delta_timestamps={
            **{k: v for k, v in delta.items() if k != "action"},
            "action": [i / meta.fps for i in range(-1, 100)],
        },
    )
    lengths = [meta.episodes[e]["length"] for e in episodes]
    offsets = np.cumsum([0, *lengths])
    assert offsets[-1] == len(ds)
    idx = [int(o) + f for o, n in zip(offsets, lengths, strict=True) for f in range(0, n, STRIDE)]
    dl = torch.utils.data.DataLoader(torch.utils.data.Subset(ds, idx), batch_size=BATCH, num_workers=8)
    acc, raw, per_ep = defaultdict(list), defaultdict(list), defaultdict(list)
    for b in dl:
        assert (b["frame_index"] % STRIDE == 0).all()
        policy.reset()
        with (
            torch.inference_mode(),
            torch.autocast("cuda", dtype=amp_dtype, enabled=mixed in ("bf16", "fp16")),
        ):
            pred = post(policy.predict_action_chunk(pre(rename_batch_keys(observation(b), rename_map))))
        pred = pred.float().cpu()
        H = pred.shape[1]
        assert H <= 100, f"chunk {H} longer than the 100-step target window"
        gt, pad = b["action"][:, 1 : H + 1], b["action_is_pad"][:, 1 : H + 1]
        prev = b["action"][:, :1].where(
            ~b["action_is_pad"][:, :1, None], gt[:, :1]
        )  # first frame: no previous
        for name, p in (("model", pred), ("hold_prev", prev.expand_as(gt))):
            err = (p - gt).abs()
            for h in (h for h in HORIZONS if h < H):
                ok = ~pad[:, h]
                for part, sl in (("tokens", slice(0, 64)), ("hands", slice(64, 78))):
                    acc[name, part, h].append((err[ok, h, sl] / std[sl]).mean(1))
                    raw[name, part, h].append(err[ok, h, sl].mean(1))
            if name == "model":
                for e, v in zip(b["episode_index"].tolist(), (err[:, 0] / std).mean(1).tolist(), strict=True):
                    per_ep[e].append(v)

    def s(x):
        x = torch.cat(x).numpy()
        return {
            "mean": round(float(x.mean()), 4),
            "p50": round(float(np.median(x)), 4),
            "p95": round(float(np.percentile(x, 95)), 4),
        }

    res = {"episodes": len(episodes), "samples": len(idx)}
    for (name, part, h), v in sorted(acc.items(), key=lambda kv: kv[0][2]):
        res.setdefault(name, {}).setdefault(part, {})[f"h{h}"] = {"norm": s(v), "raw": s(raw[name, part, h])}
    ep_err = {e: float(np.mean(v)) for e, v in per_ep.items()}
    by_task = defaultdict(list)
    for e, v in ep_err.items():
        by_task[task_of.get(e, "train")].append(v)
    return res, ep_err, by_task


result = {
    "run": str(run),
    "policy": cfg.type,
    "dataset": str(ROOT),
    "checkpoint_step": (run / "checkpoints/last").resolve().name,
    "stride": STRIDE,
    "note": "norm = |err| / per-dim dataset std, averaged over dims; h = chunk offset",
    "split": str(split_path),
}
if COMBINED:
    result["note"] += "; combined dataset: std of each split's source dataset"
for split_name, (episodes, std) in SPLITS.items():
    result[split_name], ep_err, by_task = evaluate(episodes, std)
    if split_name.startswith("heldout"):
        result[f"{split_name}_worst_episodes"] = sorted(ep_err.items(), key=lambda kv: -kv[1])[:10]
        result[f"{split_name}_by_task_h0_norm"] = dict(
            sorted(((t, round(float(np.mean(v)), 4)) for t, v in by_task.items()), key=lambda kv: -kv[1])
        )
result["chunk_size"] = H
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(result, indent=1))
for split_name in SPLITS:
    r = result[split_name]
    print(split_name, r["episodes"], "eps", r["samples"], "samples")
    for name in ("model", "hold_prev"):
        for part in ("tokens", "hands"):
            print(
                f"  {name:9s} {part:6s}",
                "  ".join(f"{h}: {v['norm']['mean']:.3f}" for h, v in r[name][part].items()),
            )
