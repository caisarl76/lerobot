"""Write the official-recipe training configs for Psi0 and Xiaomi-Robotics-1 on the G1 Dex3 datasets.

2 models x {joint28, sonic78 (sonicstate)} x {Unitree, Humanoid Everyday}, each as `_smoke` and `_full`, in the
style of the `*_official_{smoke,full}` configs (`official_queue.sh` runs them; a failed smoke skips its full run).
Run inside a training container (paths below are container paths):

  python examples/g1_dex3_training/write_psi0_xr1_configs.py --run-output /run-output [--only psi0_joint28 ...]

Recipes (see docs/research/2026-09-30-psi0-xr1-design.md for sources and departures):
* Psi0 (both spaces): `finetune-real-sonic-psi0-2.8B-sonic1.1-robust.sh`: VLM + 12-block header from
  `postpre.sonic1.1.unifolm.2609181726.40k`, 80-D (64 tokens + 14 hands + 2 unused neck), tuned VLM
  (language 1e-6, vision 1e-5, merger 1e-4), CLIP-L pooled task embedding, state as action token with a learned
  null token (drop 0.1), state noise 0.05, 270x480 images with view and colour augmentation, 128 x 40K.
  joint28 uses the same checkpoint (user decision; the official joint recipe is the AMO-era
  `finetune-real-psi0.sh`) with the token-space action in/out layers re-initialised at width 28.
* XR-1 (both spaces): `xr1/configs` post-training: full fine-tune of Qwen3-VL-4B + DiT from
  `Xiaomi-Robotics-1-5B`, batch 48 x 10K, AdamW(0.9, 0.95) wd 0.1 clip 1, lr 2e-5 cosine to 5e-6, warmup 500,
  relative actions with per-step mean/std (tokens stay absolute), frequency + choice losses, async training.
"""

import argparse
import json
from copy import deepcopy
from pathlib import Path

HE = "humanoid_everyday_g1_20260923"
PSI = "/psi-weights"
XR1 = "/xr1-weights/Xiaomi-Robotics-1-5B/model_states.pt"
STATS = "/run-output/psi0_xr1/stats"

DATASETS = {
    ("unitree", "joint28"): ("local/g1_dex3_all_joint28", "/run-output/datasets/joint28"),
    ("unitree", "sonic78"): (
        "local/g1_dex3_all_sonic78_nolimit_sonicstate",
        "/run-output/datasets/sonic78_nolimit_sonicstate",
    ),
    ("he", "joint28"): ("local/humanoid_everyday_g1_joint28_g2", f"/run-output/{HE}/datasets/joint28_g2"),
    ("he", "sonic78"): (
        "local/humanoid_everyday_g1_sonic78_nolimit_sonicstate",
        f"/run-output/{HE}/datasets/sonic78_nolimit_sonicstate",
    ),
}
CAMERAS = {"unitree": ["cam_left_high", "cam_right_high"], "he": ["egocentric"]}
SPACE_NAME = {"joint28": "joint28", "sonic78": "sonic78sonicstate"}
# Our 28-D state is [left arm 7, right arm 7, left hand 7, right hand 7] (Dex3 order, same as UnifoLM).
XR1_STATE_SLOTS = list(range(0, 7)) + list(range(8, 15)) + list(range(16, 30))
UNITREE_STATE_QUANTILES = False
PSI0_SONIC_STATE_SLOTS = list(range(15, 43))  # body29 (legs 0-11, waist 12-14, arms 15-28) + hands 29-42


def psi0_policy(space: str, dataset: str) -> dict:
    """Psi0 policy fields; `batch` is (micro, accumulation) with 128 samples per update."""
    common = {
        "type": "psi0",
        "chunk_size": 30,
        "n_action_steps": 30,
        "num_inference_steps": 10,
        "view_feature_dim": 2048,
        "hidden_dim": 1536,
        "num_heads": 24,
        "attention_head_dim": 64,
        "optimizer_lr": 1e-4,
        "optimizer_betas": [0.95, 0.999],
        "optimizer_weight_decay": 1e-6,
        "optimizer_grad_clip_norm": 1.0,
        "img_aug": True,
        # Psi0 "bounds" (min/max) as released. Before fix_unitree_corrupt_state.py the Unitree state stats hold
        # corrupt right-hand frames (+-3000); --unitree-state-quantiles then switches that state to q01/q99.
        "normalization_mapping": {
            "VISUAL": "IDENTITY",
            "STATE": "QUANTILES" if dataset == "unitree" and UNITREE_STATE_QUANTILES else "MIN_MAX",
            "ACTION": "MIN_MAX",
        },
    }
    return {
        **common,
        "vlm_path": f"{PSI}/psi0/postpre.sonic1.1.unifolm.2609181726.40k",
        "action_header_path": f"{PSI}/psi0/postpre.sonic1.1.unifolm.2609181726.40k",
        # 78D: the whole 80-D header loads (64 tokens + 14 hands + 2 masked neck dims).
        # 28D (user decision 2026-09-30, departure from the AMO-era real-G1 recipe): same SONIC v1.1 checkpoint and
        # architecture; the header's token-space action in/out layers are re-initialised at width 28, every other
        # tensor (VLM, blocks, time/CLIP embedding, state token, positional table) loads.
        "model_action_dim": 80 if space == "sonic78" else 28,
        "action_header_load": "official" if space == "sonic78" else "matching",
        "model_state_dim": 45,
        "state_slots": PSI0_SONIC_STATE_SLOTS,
        "num_blocks": 12,
        "vlm_layer_indices": [3, 5, 8, 10, 12, 14, 17, 19, 21, 23, 26, 28],
        "qk_norm": "rms_norm",
        "combined_temb": True,
        "pooled_projection_dim": 768,
        "pooled_text_encoder_path": "openai/clip-vit-large-patch14",
        "state_as_action_token": True,
        "state_null_token": True,
        "state_drop_prob": 0.1,
        "state_noise_std": 0.05,
        "dropout": 0.0,
        "state_feature_dropout": 0.0,
        "rtc": False,
        "tune_vlm": True,
        "lang_backbone_lr": 1e-6,
        "vision_tower_lr": 1e-5,
        "mm_projector_lr": 1e-4,
        "gradient_checkpointing": True,
        "image_size": [270, 480],
        "view_aug": True,
        "view_aug_min_scale": 0.85,
        "view_aug_prob": 1.0,
        "optimizer_foreach": False,
    }


def xr1_policy(space: str, dataset: str) -> dict:
    rel = list(range(28)) if space == "joint28" else [-1] * 64 + list(range(14, 28))
    return {
        "type": "xiaomi_robotics",
        "chunk_size": 30,
        "n_action_steps": 30,
        "pretrained_weights_path": XR1,
        "vlm_name": "Qwen/Qwen3-VL-4B-Instruct",
        "model_action_dim": 60 if space == "joint28" else 78,
        "model_state_dim": 60,
        "state_slots": XR1_STATE_SLOTS,
        "relative_action_state_indices": rel,
        "stats_path": f"{STATS}/{dataset}_{space}.json",
        "view_names": ["Ego View"]
        if dataset == "he"
        else ["Ego View (left camera)", "Ego View (right camera)"],
        "freq_excluded_dims": [],
        "training_repeat": 4,
        "async_train": True,
        "optimizer_lr": 2e-5,
        "optimizer_betas": [0.9, 0.95],
        "optimizer_weight_decay": 0.1,
        "optimizer_grad_clip_norm": 1.0,
        # bf16 weights (as XR-1) + bf16 Adam moments, both with stochastic rounding: fp32 moments (38 GB for
        # 4.7B trainable parameters) do not fit next to weights, gradients and activations on one 80 GB GPU.
        "optimizer_state_dtype": "bfloat16",
        "scheduler_decay_lr": 5e-6,
    }


# (micro-batch, accumulation, updates, warmup updates) per model and space.
RECIPES = {
    ("psi0", "joint28"): (16, 8, 40_000, 1_000),
    ("psi0", "sonic78"): (16, 8, 40_000, 1_000),
    ("xr1", "joint28"): (16, 3, 10_000, 500),
    ("xr1", "sonic78"): (16, 3, 10_000, 500),
}
SMOKE_MICRO_STEPS = {"psi0": 200, "xr1": 90}


def write(run_output: Path, only: set[str] | None, smoke_steps: dict[str, int]) -> list[Path]:
    written = []
    for dataset, root in (("unitree", run_output), ("he", run_output / HE)):
        split = json.loads((root / "configs/heldout_sonic78_nolimit_5pct.json").read_text())
        exclude = sorted(set(split["episodes"]) | set(split.get("always_excluded", [])))
        for model in ("psi0", "xr1"):
            for space in ("joint28", "sonic78"):
                name = f"{model}_{SPACE_NAME[space]}"
                if only and name not in only:
                    continue
                micro, accum, updates, warmup = RECIPES[model, space]
                if model == "xr1" and dataset == "unitree":
                    # Two cameras double the VLM tokens: 16 x 3 ran out of memory; keep 48 samples per update.
                    micro, accum = 8, 6
                repo_id, ds_root = DATASETS[dataset, space]
                policy = psi0_policy(space, dataset) if model == "psi0" else xr1_policy(space, dataset)
                # LeRobot steps the scheduler per micro-batch: steps, warmup and decay are micro-batch counts.
                policy["scheduler_warmup_steps"] = warmup * accum
                if model == "xr1":
                    policy["scheduler_decay_steps"] = updates * accum
                width = 28 if space == "joint28" else 78
                policy.update(
                    device="cuda",
                    push_to_hub=False,
                    input_features={
                        "observation.state": {"type": "STATE", "shape": [28]},
                        **{
                            f"observation.images.{cam}": {"type": "VISUAL", "shape": [3, 480, 640]}
                            for cam in CAMERAS[dataset]
                        },
                    },
                    output_features={"action": {"type": "ACTION", "shape": [width]}},
                )
                for stage in ("smoke", "full"):
                    job = f"{name}_ho5_official_{stage}"
                    steps = smoke_steps[model] if stage == "smoke" else updates * accum
                    config = {
                        "dataset": {
                            "repo_id": repo_id,
                            "root": ds_root,
                            "episodes": None,
                            "video_backend": "torchcodec",
                            "eval_split": 0.0,
                            "exclude_episodes": exclude,
                            # Both models augment inside the policy (Psi0 view/colour, XR-1 colour).
                            "image_transforms": {"enable": False},
                        },
                        "policy": deepcopy(policy),
                        "output_dir": f"{'/run-output' if dataset == 'unitree' else f'/run-output/{HE}'}/runs/{job}",
                        "job_name": job,
                        "seed": 1000,
                        "num_workers": 8,
                        "batch_size": micro,
                        "steps": steps,
                        "env_eval_freq": 0,
                        "eval_steps": 0,
                        "log_freq": 10 if stage == "smoke" else 200,
                        "tolerance_s": 0.001,
                        "save_checkpoint": True,
                        "save_freq": steps if stage == "smoke" else max(steps // 2, 1),
                        "wandb": {"enable": False},
                        "accelerator": {"mixed_precision": "bf16", "gradient_accumulation": {"steps": accum}},
                        "ema": {"enable": False},
                        # The presets carry the per-module learning rates / weight-decay groups.
                        "use_policy_training_preset": True,
                    }
                    path = root / "configs" / f"{job}.json"
                    path.write_text(json.dumps(config, indent=2) + "\n")
                    written.append(path)
                    print(path)
    return written


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--run-output", type=Path, default=Path("/run-output"))
    parser.add_argument("--only", nargs="*", help="e.g. psi0_joint28 xr1_sonic78sonicstate")
    parser.add_argument(
        "--unitree-state-quantiles",
        action="store_true",
        help="Psi0 Unitree state q01/q99 (uncorrected datasets)",
    )
    parser.add_argument("--smoke-steps-psi0", type=int, default=SMOKE_MICRO_STEPS["psi0"])
    parser.add_argument("--smoke-steps-xr1", type=int, default=SMOKE_MICRO_STEPS["xr1"])
    args = parser.parse_args()
    UNITREE_STATE_QUANTILES = args.unitree_state_quantiles
    write(
        args.run_output,
        set(args.only) if args.only else None,
        {"psi0": args.smoke_steps_psi0, "xr1": args.smoke_steps_xr1},
    )
