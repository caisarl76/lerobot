"""Prepare explicit fine-tuning configurations for the five foundation policies."""

import argparse
import json
from copy import deepcopy
from pathlib import Path

from huggingface_hub import hf_hub_download


def write_configs(run_root: Path):
    revisions = {
        row["repo_id"]: row["revision"]
        for row in json.loads((run_root / "pretrained-model-inventory.json").read_text())
    }
    pretrained = {
        name: json.loads(Path(hf_hub_download(repo, "config.json", revision=revisions[repo])).read_text())
        for name, repo in (("pi05", "lerobot/pi05_base"), ("vla_jepa", "lerobot/VLA-JEPA-Pretrain"))
    }
    for space, width in (("joint28", 28), ("sonic78", 78)):
        for stage in ("smoke", "full"):
            base = json.loads((run_root / "configs" / f"act_{space}_{stage}.json").read_text())
            cameras = [
                key for key in base["policy"]["input_features"] if key.startswith("observation.images.")
            ]
            specs = {
                "pi05": {
                    **deepcopy(pretrained["pi05"]),
                    "pretrained_path": "lerobot/pi05_base",
                    "pretrained_revision": revisions["lerobot/pi05_base"],
                    "max_action_dim": max(32, width),
                    "max_state_dim": 32,
                    "adapt_action_projections": width == 78,
                    "dtype": "bfloat16",
                    "gradient_checkpointing": True,
                    "use_relative_actions": False,
                },
                "groot": {
                    "type": "groot",
                    "base_model_path": "nvidia/GR00T-N1.7-3B",
                    "embodiment_tag": "new_embodiment",
                    "action_decode_transform": "none",
                    "use_relative_actions": False,
                    "use_flash_attention": False,
                    "tune_llm": False,
                    "tune_visual": False,
                },
                "molmoact2": {
                    "type": "molmoact2",
                    "checkpoint_path": "allenai/MolmoAct2",
                    "checkpoint_revision": revisions["allenai/MolmoAct2"],
                    "expected_max_action_dim": max(32, width),
                    "adapt_action_projections": width == 78,
                    "image_keys": cameras,
                    "setup_type": "Unitree G1 with two Dex3 hands and stereo head cameras",
                    "control_mode": "absolute arm and hand joint positions"
                    if width == 28
                    else "SONIC motion tokens and absolute hand joint positions",
                    "normalize_gripper": False,
                    "action_mode": "continuous",
                    "inference_action_mode": "continuous",
                    "train_mode_vlm": "lora",
                    "gradient_checkpointing": True,
                    "dtype": "bfloat16",
                },
                "vla_jepa": {
                    **deepcopy(pretrained["vla_jepa"]),
                    "pretrained_path": "lerobot/VLA-JEPA-Pretrain",
                    "pretrained_revision": revisions["lerobot/VLA-JEPA-Pretrain"],
                    "action_dim": width,
                    "state_dim": 28,
                    "enable_world_model": True,
                    "freeze_qwen": False,
                    "world_model_loss_weight": 0.1,
                    "world_model_num_views": 2,
                    "reinit_modules": [
                        "model.action_model.action_encoder",
                        "model.action_model.action_decoder",
                        "model.action_model.state_encoder",
                    ],
                    "binarize_gripper_action": False,
                    "pre_snap_gripper_action": False,
                    "gripper_joint_names": [],
                    "use_relative_actions": False,
                },
                "fastwam": {
                    "type": "fastwam",
                    "pretrained_revision": revisions["lerobot/fastwam_base"],
                    "action_dim": width,
                    "proprio_dim": 28,
                    "use_gradient_checkpointing": True,
                    "freeze_video_expert": False,
                    "loss": {"lambda_video": 1.0, "lambda_action": 1.0},
                    "toggle_action_dimensions": [],
                    "torch_dtype": "bfloat16",
                },
            }
            for name, policy in specs.items():
                config = deepcopy(base)
                policy.update(
                    device="cuda",
                    push_to_hub=False,
                    input_features=deepcopy(base["policy"]["input_features"]),
                    output_features=deepcopy(base["policy"]["output_features"]),
                )
                if name == "fastwam":
                    # FastWAM's native processor resizes each view before concatenation.
                    for key in cameras:
                        policy["input_features"][key]["shape"] = [3, 224, 224]
                config["policy"] = policy
                config["batch_size"] = {"pi05": 4, "groot": 4, "molmoact2": 2, "vla_jepa": 1, "fastwam": 1}[
                    name
                ]
                config["job_name"] = f"{name}_{space}_{stage}"
                config["output_dir"] = str(run_root / "runs" / config["job_name"])
                path = run_root / "configs" / f"{config['job_name']}.json"
                path.write_text(json.dumps(config, indent=2) + "\n")
                print(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    write_configs(parser.parse_args().run_root)
