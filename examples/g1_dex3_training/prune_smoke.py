"""Remove superseded smoke checkpoints after a verified full training run."""

import hashlib
import json
import shutil
from pathlib import Path

from .finalize_baseline import _check_tree

POLICIES = ("act", "diffusion", "pi05", "groot", "molmoact2", "vla_jepa", "fastwam")
SPACES = {"joint28": 28, "sonic78": 78}


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _real_direct_child(path: Path, parent: Path) -> None:
    if path.parent != parent or path.is_symlink() or not path.is_dir() or path.resolve() != path:
        raise ValueError(f"Expected a real direct run directory: {path}")


def prune_superseded_smoke(run_root: Path, full_name: str) -> None:
    root = Path(run_root).absolute()
    if root.resolve() != root or not root.is_dir():
        raise ValueError(f"Run root must be a real absolute directory: {root}")
    names = {f"{policy}_{space}_full": (policy, space) for policy in POLICIES for space in SPACES}
    if full_name not in names:
        raise ValueError(f"Unknown full run: {full_name}")
    policy, space = names[full_name]
    width = SPACES[space]
    runs = root / "runs"
    _real_direct_child(runs / full_name, runs)
    smoke_name = f"{policy}_{space}_smoke"
    smoke = runs / smoke_name
    _real_direct_child(smoke, runs)
    logs = root / "logs"
    for marker in (logs / f"{full_name}.exit", logs / f"{full_name}-verify.exit"):
        if marker.resolve() != marker or not marker.is_file() or marker.read_text().strip() != "0":
            raise ValueError(f"Missing successful full-run marker: {marker}")

    full_report_path = runs / full_name / "final_model_validation.json"
    if full_report_path.resolve() != full_report_path:
        raise ValueError("Full-run validation report must not traverse symlinks")
    full_report = json.loads(full_report_path.read_text())
    full_checkpoint = runs / full_name / "checkpoints" / "040000"
    full_model = full_checkpoint / "pretrained_model" / "model.safetensors"
    if (
        full_report.get("verified") is not True
        or full_report.get("final_step") != 40000
        or full_report.get("policy_type") != policy
        or full_report.get("predicted_shape", [None])[-1] != width
        or Path(full_report.get("checkpoint_path", "")).absolute() != full_checkpoint
        or full_model.resolve() != full_model
        or not full_model.is_file()
        or full_report.get("weights_sha256") != _digest(full_model)
    ):
        raise ValueError("Full-run validation proof is invalid")
    checkpoints = smoke / "checkpoints"
    if not checkpoints.exists() and not checkpoints.is_symlink():
        return
    if checkpoints.is_symlink() or not checkpoints.is_dir() or checkpoints.resolve() != checkpoints:
        raise ValueError("Smoke checkpoints must be a real directory")
    entries = list(checkpoints.iterdir())
    if {entry.name for entry in entries} != {"000020", "last"}:
        raise ValueError("Unexpected smoke checkpoint entries")
    smoke_checkpoint = checkpoints / "000020"
    _check_tree(smoke_checkpoint)
    last = checkpoints / "last"
    if not last.is_symlink() or last.resolve(strict=True) != smoke_checkpoint:
        raise ValueError("Smoke checkpoints/last must point to 000020")
    smoke_report_path = smoke / "final_model_validation.json"
    if smoke_report_path.resolve() != smoke_report_path:
        raise ValueError("Smoke validation report must not traverse symlinks")
    smoke_report = json.loads(smoke_report_path.read_text())
    if (
        smoke_report.get("verified") is not True
        or smoke_report.get("final_step") != 20
        or smoke_report.get("policy_type") != policy
        or smoke_report.get("predicted_shape", [None])[-1] != width
        or Path(smoke_report.get("checkpoint_path", "")).absolute() != smoke_checkpoint
    ):
        raise ValueError("Smoke validation proof is invalid")
    shutil.rmtree(checkpoints)
    print(f"deleted {checkpoints}", flush=True)
