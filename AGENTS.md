This file provides guidance to AI agents when working with code in this repository.

> **User-facing help → [`AGENT_GUIDE.md`](./AGENT_GUIDE.md)** (SO-101 setup, recording, picking a policy, training duration, eval — with copy-pasteable commands).

## Project Overview

LeRobot is a PyTorch-based library for real-world robotics, providing datasets, pretrained policies, and tools for training, evaluation, data collection, and robot control. It integrates with Hugging Face Hub for model/dataset sharing.

## Tech Stack

Python 3.12+ · PyTorch · Hugging Face (datasets, Hub, accelerate) · draccus (config/CLI) · Gymnasium (envs) · uv (package management)

## Development Setup

```bash
uv sync --locked                            # Base dependencies
uv sync --locked --extra test --extra dev   # Test + dev tools
uv sync --locked --extra all                # Everything
git lfs install && git lfs pull             # Test artifacts
```

## Key Commands

```bash
uv run pytest tests -svv --maxfail=10                 # All tests
DEVICE=cuda make test-end-to-end                      # All E2E tests
pre-commit run --all-files                           # Lint + format (ruff, typos, bandit, etc.)
```

## Architecture (`src/lerobot/`)

- **`scripts/`** — CLI entry points (`lerobot-train`, `lerobot-eval`, `lerobot-record`, etc.), mapped in `pyproject.toml [project.scripts]`.
- **`configs/`** — Dataclass configs parsed by draccus. `train.py` has `TrainPipelineConfig` (top-level). `policies.py` has `PreTrainedConfig` base. Polymorphism via `draccus.ChoiceRegistry` with `@register_subclass("name")` decorators.
- **`policies/`** — Each policy in its own subdir. All inherit `PreTrainedPolicy` (`nn.Module` + `HubMixin`) from `pretrained.py`. Factory with lazy imports in `factory.py`.
- **`processor/`** — Data transformation pipeline. `ProcessorStep` base with registry. `DataProcessorPipeline` / `PolicyProcessorPipeline` chain steps.
- **`datasets/`** — `LeRobotDataset` (episode-aware sampling + video decoding) and `LeRobotDatasetMetadata`.
- **`envs/`** — `EnvConfig` base in `configs.py`, factory in `factory.py`. Each env subclass defines `gym_kwargs` and `create_envs()`.
- **`robots/`, `motors/`, `cameras/`, `teleoperators/`** — Hardware abstraction layers.
- **`types.py`** and **`configs/types.py`** — Core type aliases and feature type definitions.

## Repository Structure (outside `src/`)

- **`tests/`** — Pytest suite organized by module. Fixtures in `tests/fixtures/`, mocks in `tests/mocks/`. Hardware tests use skip decorators from `tests/utils.py`. E2E tests via `Makefile` write to `tests/outputs/`.
- **`.github/workflows/`** — CI: `quality.yml` (pre-commit), `fast_tests.yml` (base deps, every PR), `full_tests.yml` (all extras + E2E + GPU, post-approval), `latest_deps_tests.yml` (daily lockfile upgrade), `security.yml` (TruffleHog), `release.yml` (PyPI publish on tags).
- **`docs/source/`** — HF documentation (`.mdx` files). Per-policy READMEs, hardware guides, tutorials. Built separately via `docs-requirements.txt` and CI workflows.
- **`examples/`** — End-user tutorials and scripts organized by use case (dataset creation, training, hardware setup).
- **`docker/`** — Dockerfiles for user (`Dockerfile.user`) and CI (`Dockerfile.internal`).
- **`benchmarks/`** — Performance benchmarking scripts.
- **Root files**: `pyproject.toml` (single source of truth for deps, build, tool config), `Makefile` (E2E test targets), `uv.lock`, `CONTRIBUTING.md` & `README.md` (general information).

## Notes

- **Mypy is gradual**: strict only for `lerobot.envs`, `lerobot.configs`, `lerobot.optim`, `lerobot.model`, `lerobot.cameras`, `lerobot.motors`, `lerobot.transport`. Add type annotations when modifying these modules.
- **Imports**: prefer top-level imports; relative (`from .sibling import X`) across sibling files within a module, absolute (`from lerobot.module import X`) across modules.
- **Optional dependencies**: many policies, envs, and robots are behind extras (e.g., `lerobot[aloha]`, see `pyproject.toml`). Guard optional imports with `TYPE_CHECKING or _foo_available` at module top + a `require_package(...)` check at use time. Reuse the `_foo_available` flags in `utils/import_utils.py`; don't call `is_package_available`.
- **Video decoding**: datasets can store observations as video files. `LeRobotDataset` handles frame extraction, but tests need ffmpeg installed.
- **Prioritize use of `uv run`** to execute Python commands (not raw `python` or `pip`).
- **G1 Dex3 → SONIC token conversion** (`examples/g1_dex3_training/`): read [`docs/research/2026-09-23-sonic-roundtrip-audit.md`](./docs/research/2026-09-23-sonic-roundtrip-audit.md) first. It covers the confirmed decisions (speed limits off, 30 Hz training with deploy-side 50 Hz token interpolation), the official sim2sim startup/replay procedure, and pitfalls (Dex3 hands bypass SONIC; the decoder output is a PD setpoint, not a pose). To continue the real-robot work, start from [`docs/research/2026-09-27-sonic-real-robot-handover.md`](./docs/research/2026-09-27-sonic-real-robot-handover.md), then [`docs/research/2026-09-28-sonic-28d-and-combined-handover.md`](./docs/research/2026-09-28-sonic-28d-and-combined-handover.md) (28D joint policies through SONIC, combined Unitree + HE training, held-out evaluation, next steps). Real-robot pipeline (workstation policy server + G1 PC2 streamer), head camera, safeguards, `--start planner` for Humanoid Everyday policies and runs 01–16: [`docs/research/2026-09-29-sonic-real-robot-first-runs.md`](./docs/research/2026-09-29-sonic-real-robot-first-runs.md). **Open issue, handle next:** GR00T predicts jerky token chunks although the no-limit dataset replays smoothly; the model output itself is not smooth (the training procedure may have a problem): [`docs/research/2026-09-29-issue-groot-jerky-predictions.md`](./docs/research/2026-09-29-issue-groot-jerky-predictions.md). **Current work, start here:** every policy (GR00T, Pi0.5, VLA-JEPA, MolmoAct2, FastWAM, Diffusion, ACT) is being retrained with its official recipe on HE and Unitree via per-GPU queues on the H100 (GPUs 0 and 6). State, pitfalls (the scheduler steps per micro-batch) and next checks: [`docs/research/2026-09-30-official-retraining-handover.md`](./docs/research/2026-09-30-official-retraining-handover.md). Psi0 and Xiaomi-Robotics-1 as native policies (recipes, open-loop and closed-loop sim results, first Psi0 robot run, Psi0 state temporal jitter; XR-1 drifts in closed loop through its state input; **to continue the Psi0 robot trials, read its "Handover: real-robot continuation" section**): [`docs/research/2026-09-30-psi0-xr1-design.md`](./docs/research/2026-09-30-psi0-xr1-design.md).
