"""Registry of the trained G1 Dex3 models: where the weights are, which config and dataset trained them, and how.

Two steps, so nothing has to be installed on the hosts:

  collect  (in a container on each GPU host, /run-output mounted): one JSON line per run folder that has
           checkpoints/last/pretrained_model/train_config.json
    docker run --rm -i -v /mnt/data01/jhkim/model_weight/g1_dex3_20260922:/run-output:ro \
      --entrypoint /run-output/environment/venv/bin/python 4cbe2a3f7fc6 - collect h100 < model_registry.py > h100.jsonl
  render   (on the workstation): merge the hosts' JSON lines with the local model copies into a table
    python model_registry.py render --local /mnt/data/jihun/g1_models --out docs/research/g1-model-registry \
      h100.jsonl h100_174.jsonl

Columns: run name, dataset group (base folder), policy type, action space, dataset root, held-out episodes
excluded, samples = micro-batch x steps (LeRobot steps count micro-batches; accumulation only groups them into updates), final step, exit status, finished (UTC), weight size, host path,
config path, workstation copy.
"""

from __future__ import annotations

import csv
import glob
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

RUN_GLOBS = ("/run-output/runs/*", "/run-output/*/runs/*")
COLUMNS = (
    "run",
    "group",
    "policy",
    "action",
    "dataset",
    "excluded_eps",
    "batch",
    "accum",
    "steps",
    "samples",
    "final_step",
    "exit",
    "finished_utc",
    "weights_gb",
    "host",
    "weights",
    "config",
    "workstation_copy",
)


def collect(host: str) -> None:
    for run in sorted(p for pattern in RUN_GLOBS for p in glob.glob(pattern)):
        run = Path(run)
        model = run / "checkpoints" / "last" / "pretrained_model"
        train_config = model / "train_config.json"
        if not train_config.is_file():
            continue
        cfg = json.loads(train_config.read_text())
        base = run.parent.parent  # /run-output or /run-output/<group>
        exit_file = base / "logs" / f"{run.name}.exit"
        config_file = base / "configs" / f"{run.name}.json"
        policy = cfg.get("policy") or {}
        action = ((policy.get("output_features") or {}).get("action") or {}).get("shape") or [None]
        dataset = cfg.get("dataset") or {}
        accum = ((cfg.get("accelerator") or {}).get("gradient_accumulation") or {}).get("steps", 1)
        files = [f for f in model.rglob("*") if f.is_file()]
        record = {
            "run": run.name,
            "group": base.name if base.name != "run-output" else "unitree",
            "policy": policy.get("type"),
            "action": f"{action[0]}D" if action[0] else "",
            "dataset": dataset.get("root") or dataset.get("repo_id"),
            "excluded_eps": len(dataset.get("exclude_episodes") or []),
            "batch": cfg.get("batch_size"),
            "accum": accum,
            "steps": cfg.get("steps"),
            "samples": (cfg.get("batch_size") or 0) * (cfg.get("steps") or 0),
            "final_step": (run / "checkpoints" / "last").resolve().name,
            "exit": exit_file.read_text().strip() if exit_file.is_file() else "",
            "finished_utc": datetime.fromtimestamp(max(f.stat().st_mtime for f in files), UTC).strftime(
                "%Y-%m-%d %H:%M"
            ),
            "weights_gb": round(sum(f.stat().st_size for f in files) / 1e9, 1),
            "host": host,
            "weights": str(model).replace("/run-output", "/mnt/data01/jhkim/model_weight/g1_dex3_20260922"),
            "config": str(config_file).replace(
                "/run-output", "/mnt/data01/jhkim/model_weight/g1_dex3_20260922"
            )
            if config_file.is_file()
            else "",
        }
        print(json.dumps(record))


def local_copies(roots: list[str]) -> dict[tuple[str, str], list[str]]:
    """(job_name, dataset root) -> workstation model folders. HE and Unitree runs share job names, so the
    dataset decides which run a copy belongs to."""
    copies: dict[tuple[str, str], list[str]] = {}
    for root in roots:
        for train_config in sorted(Path(root).glob("*/pretrained_model*/train_config.json")):
            cfg = json.loads(train_config.read_text())
            dataset = (cfg.get("dataset") or {}).get("root") or (cfg.get("dataset") or {}).get("repo_id")
            if cfg.get("job_name"):
                copies.setdefault((cfg["job_name"], dataset), []).append(str(train_config.parent.parent))
    return copies


def render(paths: list[str], local: list[str], out: str) -> None:
    records = [
        json.loads(line) for path in paths for line in Path(path).read_text().splitlines() if line.strip()
    ]
    copies = local_copies(local)
    for r in records:
        r["workstation_copy"] = " ".join(copies.get((r["run"], r["dataset"]), []))
    records.sort(key=lambda r: (r["group"], r["policy"] or "", r["run"], r["host"]))
    with open(f"{out}.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows({k: r.get(k, "") for k in COLUMNS} for r in records)
    lines = [
        "# G1 Dex3 model registry",
        "",
        f"Generated {datetime.now(UTC):%Y-%m-%d %H:%M} UTC by `examples/g1_dex3_training/model_registry.py`"
        " (regenerate it after training or deleting runs; the CSV next to this file has the same rows).",
        "Paths are host paths under `/mnt/data01/jhkim/model_weight/g1_dex3_20260922` on the named host.",
        "",
    ]
    for group in sorted({r["group"] for r in records}):
        lines += [
            f"## {group}",
            "",
            "| Run | Policy | Action | Samples (micro-batch × steps, accum) | Excl. eps | Final | Exit "
            "| Finished (UTC) | GB | Host | Workstation copy |",
            "|" + " --- |" * 11,
        ]
        for r in (r for r in records if r["group"] == group):
            lines.append(
                f"| `{r['run']}` | {r['policy']} | {r['action']} | {r['samples']:,} ({r['batch']} × {r['steps']}, "
                f"accum {r['accum']}) | {r['excluded_eps']} | {r['final_step']} | {r['exit'] or '–'} | {r['finished_utc']} "
                f"| {r['weights_gb']} | {r['host']} | {'yes' if r['workstation_copy'] else ''} |"
            )
        lines.append("")
    matched = {(r["run"], r["dataset"]) for r in records}
    orphans = sorted(path for key, paths in copies.items() if key not in matched for path in paths)
    if orphans:
        lines += ["## Workstation copies without a run on any host", ""]
        lines += ["Their host runs were deleted (e.g. old recipes or the uncorrected Unitree data).", ""]
        lines += [f"- `{path}`" for path in orphans] + [""]
    lines += [
        "Dataset, weight and config paths per run are in the CSV (`dataset`, `weights`, `config`,"
        " `workstation_copy`)."
    ]
    Path(f"{out}.md").write_text("\n".join(lines) + "\n")
    print(f"{len(records)} runs -> {out}.md, {out}.csv")


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "collect":
        collect(sys.argv[2])
    elif len(sys.argv) >= 2 and sys.argv[1] == "render":
        args = sys.argv[2:]
        local = [args[i + 1] for i, a in enumerate(args) if a == "--local"]
        out = args[args.index("--out") + 1]
        skip = {i for i, a in enumerate(args) if a in ("--local", "--out")}
        paths = [a for i, a in enumerate(args) if i not in skip and i - 1 not in skip]
        render(paths, local, out)
    else:
        sys.exit(__doc__)
