"""Verify a completed G1 training model; optionally discard its training checkpoints.

Run inside the training container, for example:
python examples/g1_dex3_training/finalize_baseline.py \
    --run-dir /run-output/runs/act_joint28_full --device cuda --prune
Omit --prune to write the validation report without deleting anything.
"""

import argparse
import hashlib
import importlib
import json
import os
import shutil
import tempfile
from pathlib import Path

RUNS_ROOT = Path("/run-output/runs")
POLICY_TYPES = {"act", "diffusion", "pi05", "groot", "molmoact2", "vla_jepa", "fastwam"}


def _check_tree(path: Path):
    """Reject links and special files before considering any recursive deletion."""
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"Expected a real directory: {path}")
    for directory, dirs, files in os.walk(path, followlinks=False):
        for name in dirs + files:
            entry = Path(directory) / name
            if entry.is_symlink() or not (entry.is_dir() or entry.is_file()):
                raise ValueError(f"Unexpected symlink or special file: {entry}")


def inspect_completed_checkpoint(run_dir: Path, *, allowed_root: Path = RUNS_ROOT):
    """Check completion evidence and constrain every pruning target to this run."""
    run_dir = Path(os.path.abspath(run_dir))
    allowed_root = Path(os.path.abspath(allowed_root))
    if allowed_root.resolve() != allowed_root or run_dir.parent != allowed_root:
        raise ValueError(f"Run must be a direct child of {allowed_root}")
    if run_dir.is_symlink() or run_dir.resolve() != run_dir or not run_dir.is_dir():
        raise ValueError(f"Run must be a real directory: {run_dir}")
    exit_path = allowed_root.parent / "logs" / f"{run_dir.name}.exit"
    if exit_path.is_symlink() or exit_path.resolve() != exit_path:
        raise ValueError(f"Exit marker must not traverse symlinks: {exit_path}")
    if exit_path.read_text().strip() != "0":
        raise ValueError("Training has not completed with exit code zero")
    checkpoints = run_dir / "checkpoints"
    if checkpoints.is_symlink() or not checkpoints.is_dir():
        raise ValueError("Expected a real checkpoints directory")
    steps = []
    for entry in checkpoints.iterdir():
        if entry.name == "last":
            continue
        if not entry.name.isascii() or not entry.name.isdigit() or len(entry.name) < 6:
            raise ValueError(f"Unexpected checkpoint entry: {entry}")
        _check_tree(entry)
        if {p.name for p in entry.iterdir()} - {"pretrained_model", "training_state"}:
            raise ValueError(f"Unexpected checkpoint contents: {entry}")
        steps.append(entry)
    if not steps:
        raise ValueError("No numeric checkpoints found")
    final = max(steps, key=lambda p: int(p.name))
    last = checkpoints / "last"
    if not last.is_symlink() or last.resolve(strict=True) != final:
        raise ValueError("checkpoints/last must point to the final checkpoint inside this run")
    model = final / "pretrained_model"
    for name in (
        "config.json",
        "train_config.json",
        "model.safetensors",
        "policy_preprocessor.json",
        "policy_postprocessor.json",
    ):
        if not (model / name).is_file():
            raise ValueError(f"Missing saved model file: {model / name}")
    config = json.loads((model / "train_config.json").read_text())
    step = config["steps"]
    saved_step = json.loads((final / "training_state/training_step.json").read_text())["step"]
    if type(step) is not int or step <= 0 or saved_step != step or final.name != f"{step:06d}":
        raise ValueError("Saved training step, final directory, and configured steps must agree")
    if Path(config["output_dir"]).resolve() != run_dir:
        raise ValueError("Saved output_dir does not match this run")
    return final, steps


def _weights_digest(model: Path):
    with (model / "model.safetensors").open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _write_report(run_dir: Path, report: dict):
    target = run_dir / "final_model_validation.json"
    if target.is_symlink():
        raise ValueError("Validation report must not be a symlink")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=run_dir, delete=False) as stream:
            temporary = Path(stream.name)
            json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _inference_observation(batch, cfg):
    observation = {key: value for key, value in batch.items() if key not in {"action", "action_is_pad"}}
    if cfg.type in {"fastwam", "vla_jepa"}:
        # Their training windows start at delta=0 and include future video/state targets.
        # Deployment receives only the current observation. Diffusion retains its past history.
        for key, feature in cfg.input_features.items():
            value = observation.get(key)
            if value is not None and value.ndim == len(feature.shape) + 2:
                observation[key] = value[:, 0]
            observation.pop(f"{key}_is_pad", None)
    return observation


def validate_model(run_dir: Path, device: str):
    """Reload the actual saved model and processors, then evaluate one native batch."""
    final, _ = inspect_completed_checkpoint(run_dir)
    model = final / "pretrained_model"
    digest = _weights_digest(model)

    import torch
    from torch.utils.data import default_collate

    from lerobot.configs import PreTrainedConfig
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.datasets.factory import make_dataset
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors
    from lerobot.processor.rename_processor import rename_batch_keys

    if device not in {"cpu", "cuda"} or (device == "cuda" and not torch.cuda.is_available()):
        raise ValueError(f"Requested device is unavailable: {device}")
    policy_type = json.loads((model / "config.json").read_text())["type"]
    if policy_type not in POLICY_TYPES:
        raise ValueError(f"Unsupported policy: {policy_type}")
    importlib.import_module(f"lerobot.policies.{policy_type}.configuration_{policy_type}")
    cfg = PreTrainedConfig.from_pretrained(model, local_files_only=True)
    cfg.device = device
    if cfg.type == "vla_jepa":
        # A completed checkpoint must retain every learned head, with no fine-tuning resets.
        cfg.reinit_modules = None
    if cfg.type == "pi05":
        cfg.adapt_action_projections = False
    policy = get_policy_class(cfg.type).from_pretrained(model, config=cfg, strict=True, local_files_only=True)
    policy.eval()
    for name, tensor in policy.state_dict().items():
        if not torch.isfinite(tensor).all().item():
            raise ValueError(f"Non-finite saved parameter or buffer: {name}")
    preprocessor, postprocessor = make_pre_post_processors(
        cfg,
        pretrained_path=model,
        preprocessor_overrides={"device_processor": {"device": device}},
        postprocessor_overrides={"device_processor": {"device": "cpu"}},
    )
    train_cfg = TrainPipelineConfig.from_pretrained(model, local_files_only=True)
    train_cfg.policy = cfg
    manifest = json.loads((Path(train_cfg.dataset.root) / "meta/provenance.json").read_text())
    provenance = manifest.get("source_provenance", manifest)
    episode = provenance["smoke_episodes"][0]
    train_cfg.dataset.episodes = [episode]
    train_cfg.dataset.exclude_episodes = None
    dataset = make_dataset(train_cfg)
    batch = default_collate([dataset[0]])
    # Match lerobot_train._preprocess_dataset_batch, without importing the training CLI.
    for key in dataset.meta.camera_keys:
        if key in batch and batch[key].dtype == torch.uint8:
            batch[key] = batch[key].to(dtype=torch.float32) / 255.0
    batch = preprocessor(rename_batch_keys(batch, train_cfg.rename_map))
    width = cfg.output_features["action"].shape[0]
    if width not in {28, 78}:
        raise ValueError(f"Unexpected action width: {width}")
    precision = train_cfg.accelerator.mixed_precision
    with (
        torch.inference_mode(),
        torch.autocast(
            device_type=device,
            dtype=torch.float16 if precision == "fp16" else torch.bfloat16,
            enabled=device == "cuda" and precision in {"bf16", "fp16"},
        ),
    ):
        loss, metrics = policy.forward(dict(batch))
        if loss.numel() != 1 or not torch.isfinite(loss).all().item():
            raise ValueError("Forward loss is not a finite scalar")
        policy.reset()
        # Inference must not receive the ground-truth action chunk.
        observation = _inference_observation(batch, cfg)
        actions = policy.predict_action_chunk(observation)
        if cfg.type in {"act", "vla_jepa"}:
            expected_horizon = cfg.chunk_size
        elif cfg.type == "fastwam":
            expected_horizon = cfg.action_horizon
        else:
            expected_horizon = cfg.n_action_steps
        expected = (1, expected_horizon, width)
        if tuple(actions.shape) != expected or not torch.isfinite(actions).all().item():
            raise ValueError(f"Invalid predicted actions: expected {expected}, got {tuple(actions.shape)}")
        output = postprocessor(actions)
        if tuple(output.shape) != expected or not torch.isfinite(output).all().item():
            raise ValueError("Postprocessed actions have invalid shape or non-finite values")
    if _weights_digest(model) != digest:
        raise ValueError("Weights changed during validation")
    report = {
        "verified": True,
        "checkpoint_path": str(final),
        "final_step": int(final.name),
        "policy_type": cfg.type,
        "device": device,
        "weights_sha256": digest,
        "weights_size_bytes": (model / "model.safetensors").stat().st_size,
        "episode": episode,
        "loss": loss.item(),
        "loss_metrics": {
            key: float(value)
            for key, value in (metrics or {}).items()
            if isinstance(value, (int, float)) or (isinstance(value, torch.Tensor) and value.numel() == 1)
        },
        "predicted_shape": list(actions.shape),
        "postprocessed_shape": list(output.shape),
        "postprocessed_min": output.min().item(),
        "postprocessed_max": output.max().item(),
        "pruned_paths": [],
    }
    _write_report(final.parent.parent, report)
    return report


def prune_verified_checkpoint(run_dir: Path, report: dict, *, allowed_root: Path = RUNS_ROOT):
    """Prune only after validation; recheck completion, paths, and the weights digest."""
    final, steps = inspect_completed_checkpoint(run_dir, allowed_root=allowed_root)
    model = final / "pretrained_model"
    if (
        report.get("verified") is not True
        or report.get("checkpoint_path") != str(final)
        or report.get("final_step") != int(final.name)
        or report.get("weights_sha256") != _weights_digest(model)
        or report.get("weights_size_bytes") != (model / "model.safetensors").stat().st_size
    ):
        raise ValueError("Validation report does not match the current final weights")
    targets = [step for step in steps if step != final] + [final / "training_state"]
    for target in targets:
        shutil.rmtree(target)
    return [str(target) for target in targets]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--prune", action="store_true")
    args = parser.parse_args()
    report = validate_model(args.run_dir, args.device)
    if args.prune:
        report["pruned_paths"] = prune_verified_checkpoint(args.run_dir, report)
        _write_report(args.run_dir, report)
    print(json.dumps(report, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
