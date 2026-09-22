"""Decode both ends of smoke episodes from every source through the native reader."""

import argparse
import json
from pathlib import Path

import torch

from lerobot.configs.default import DatasetConfig
from lerobot.configs.train import TrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.policies.act.configuration_act import ACTConfig


def validate(root: Path, report_path: Path):
    manifest = json.loads((root / "meta/provenance.json").read_text())
    source = manifest.get("source_provenance", manifest)
    episodes = source["smoke_episodes"]
    cfg = TrainPipelineConfig(
        dataset=DatasetConfig(
            repo_id=manifest.get("repo_id", "local/g1_dex3_all_sonic78"),
            root=str(root),
            episodes=episodes,
            video_backend="pyav",
        ),
        policy=ACTConfig(device="cpu", push_to_hub=False),
        tolerance_s=1e-3,
    )
    dataset = make_dataset(cfg)
    action_dim = dataset.meta.features["action"]["shape"][0]
    reports = []
    offset = 0
    for episode in episodes:
        length = int(dataset.meta.episodes[episode]["length"])
        for index in (offset, offset + length - 1):
            item = dataset[index]
            assert int(item["episode_index"]) == episode
            assert item["action"].shape == (cfg.policy.chunk_size, action_dim)
            assert item["observation.state"].shape == (28,)
            assert torch.isfinite(item["action"]).all()
            cameras = {}
            for key in source["camera_keys"]:
                frame = item[key]
                assert frame.shape == (3, 480, 640)
                assert frame.dtype == torch.uint8
                assert frame.max() > frame.min(), f"constant video frame: {episode} {key}"
                cameras[key] = {"shape": list(frame.shape), "min": int(frame.min()), "max": int(frame.max())}
            reports.append(
                {
                    "episode_index": episode,
                    "frame_index": int(item["frame_index"]),
                    "task": item["task"],
                    "cameras": cameras,
                }
            )
        offset += length
    assert offset == len(dataset)
    report = {
        "dataset": str(root),
        "action_dim": action_dim,
        "episodes": episodes,
        "frames": offset,
        "samples": reports,
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Decoded {len(reports)} samples from {len(episodes)} episodes across all sources", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    validate(args.root, args.report)
