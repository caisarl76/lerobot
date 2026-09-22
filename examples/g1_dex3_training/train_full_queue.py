"""Run the fourteen G1 Dex3 full training jobs, one at a time."""

import argparse
import fcntl
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

from examples.g1_dex3_training.prune_smoke import prune_superseded_smoke

POLICIES = ("act", "diffusion", "pi05", "groot", "molmoact2", "vla_jepa", "fastwam")
SPACES = (("joint28", 28), ("sonic78", 78))
STEPS = 40_000
MIN_FREE = 80 * 1024**3


def _json(path: Path):
    return json.loads(path.read_text())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _smoke_guard(root: Path, name: str, width: int) -> None:
    report = _json(root / "runs" / f"{name}_smoke" / "final_model_validation.json")
    if report.get("verified") is not True or report.get("final_step") != 20:
        raise ValueError(f"Invalid smoke report for {name}")
    shape = report.get("predicted_shape")
    if not isinstance(shape, list) or not shape or shape[-1] != width:
        raise ValueError(f"Smoke action shape does not match {name}: {shape}")


def _config_guard(root: Path, name: str, space: str, width: int) -> Path:
    path = root / "configs" / f"{name}_{space}_full.json"
    config = _json(path)
    dataset = config.get("dataset", {})
    policy = config.get("policy", {})
    if policy.get("type") != name:
        raise ValueError(f"Wrong policy type in {path}")
    if dataset.get("episodes") is not None:
        raise ValueError(f"Full config has episodes for {name}_{space}")
    if dataset.get("root") != str(root / "datasets" / space):
        raise ValueError(f"Wrong dataset root in {path}")
    if dataset.get("exclude_episodes") is not None or dataset.get("eval_split") != 0.0:
        raise ValueError(f"Invalid dataset split settings in {path}")
    if policy.get("input_features", {}).get("observation.state", {}).get("shape") != [28]:
        raise ValueError(f"Wrong state width in {path}")
    if config.get("steps") != STEPS or config.get("output_dir") != str(
        root / "runs" / f"{name}_{space}_full"
    ):
        raise ValueError(f"Invalid full config settings in {path}")
    if policy.get("output_features", {}).get("action", {}).get("shape") != [width]:
        raise ValueError(f"Wrong action width in {path}")
    return path


def _completed(root: Path, name: str) -> bool:
    run = root / "runs" / name
    report_path = run / "final_model_validation.json"
    if not run.exists():
        return False
    if not report_path.is_file():
        raise ValueError(f"Output run already exists without completion report: {run}")
    report = _json(report_path)
    checkpoint = Path(report.get("checkpoint_path", ""))
    model = checkpoint / "pretrained_model" / "model.safetensors"
    expected_checkpoint = run / "checkpoints" / f"{STEPS:06d}"
    if (
        report.get("verified") is not True
        or report.get("final_step") != STEPS
        or checkpoint.resolve() != expected_checkpoint.resolve()
        or not checkpoint.is_dir()
        or not model.is_file()
        or report.get("weights_sha256") != _sha256(model)
    ):
        raise ValueError(f"Existing run is incomplete or failed validation: {run}")
    return True


def _run(command: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log:
        completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=False)
    return completed.returncode


def run_queue(root: Path) -> int:
    root = root.resolve()
    if not root.is_dir():
        raise ValueError(f"Run root is not a directory: {root}")
    lock_path = root / "full-training.lock"
    with lock_path.open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another full-training queue is already running") from exc

        jobs = []
        for policy in POLICIES:
            for space, width in SPACES:
                name = f"{policy}_{space}_full"
                _smoke_guard(root, f"{policy}_{space}", width)
                config = _config_guard(root, policy, space, width)
                jobs.append((name, config))
        for name, config in jobs:
            if _completed(root, name):
                print(f"{name}: already complete, skipping", flush=True)
                prune_superseded_smoke(root, name)
                continue
            if shutil.disk_usage(root).free < MIN_FREE:
                raise RuntimeError(f"Insufficient free disk space before {name}; need at least 80 GiB")
            print(f"{name}: starting", flush=True)
            code = _run(
                [sys.executable, "-m", "lerobot.scripts.lerobot_train", f"--config_path={config}"],
                root / "logs" / f"{name}.log",
            )
            (root / "logs" / f"{name}.exit").write_text(f"{code}\n")
            print(f"{name}: training finished with exit {code}", flush=True)
            if code != 0:
                return code
            verify_log = root / "logs" / f"{name}-verify.log"
            verify_code = _run(
                [
                    sys.executable,
                    "-m",
                    "examples.g1_dex3_training.finalize_baseline",
                    "--run-dir",
                    str(root / "runs" / name),
                    "--device",
                    "cuda",
                    "--prune",
                ],
                verify_log,
            )
            (root / "logs" / f"{name}-verify.exit").write_text(f"{verify_code}\n")
            if verify_code != 0:
                return verify_code
            print(f"{name}: verification finished with exit {verify_code}", flush=True)
            prune_superseded_smoke(root, name)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=Path("/run-output"))
    try:
        return run_queue(parser.parse_args().run_root)
    except Exception as exc:
        print(f"train queue failed: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
