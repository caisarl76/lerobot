"""Write ACT/diffusion smoke and full-corpus configs; does not start training."""

import argparse
import json
from pathlib import Path


def write_configs(run_root: Path):
    manifest = json.loads((run_root / "datasets/joint28/meta/provenance.json").read_text())
    destination = run_root / "configs"
    destination.mkdir(exist_ok=True)
    for policy in ("act", "diffusion"):
        for space, width in (("joint28", 28), ("sonic78", 78)):
            for stage in ("smoke", "full"):
                name = f"{policy}_{space}_{stage}"
                config = {
                    "dataset": {
                        "repo_id": f"local/g1_dex3_all_{space}",
                        "root": str(run_root / "datasets" / space),
                        "episodes": manifest["smoke_episodes"] if stage == "smoke" else None,
                        "video_backend": "pyav",
                        "eval_split": 0.0,
                    },
                    "policy": {
                        "type": policy,
                        "device": "cuda",
                        "push_to_hub": False,
                        "input_features": {
                            "observation.state": {"type": "STATE", "shape": [28]},
                            **{
                                key: {"type": "VISUAL", "shape": [3, 480, 640]}
                                for key in manifest["camera_keys"]
                            },
                        },
                        "output_features": {"action": {"type": "ACTION", "shape": [width]}},
                    },
                    "output_dir": str(run_root / "runs" / name),
                    "job_name": name,
                    "seed": 1000,
                    "num_workers": 4,
                    "batch_size": 8 if stage == "smoke" else 32,
                    "steps": 20 if stage == "smoke" else 40000,
                    "env_eval_freq": 0,
                    "eval_steps": 0,
                    "log_freq": 1 if stage == "smoke" else 100,
                    "tolerance_s": 1e-3,
                    "save_checkpoint": True,
                    "save_freq": 0,
                    "wandb": {"enable": False},
                    "accelerator": {"mixed_precision": "bf16"},
                }
                if policy == "diffusion":
                    config["policy"].update(resize_shape=[256, 256], crop_ratio=0.875)
                path = destination / f"{name}.json"
                path.write_text(json.dumps(config, indent=2) + "\n")
                print(path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    write_configs(parser.parse_args().run_root)
