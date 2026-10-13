# G1 Data Efficiency on a Balanced Own Dataset — Implementation Plan

> **Status 2026-10-13: shelved.** The user switched to the existing 43-episode handover-bottle data (20 vs 41 training
> episodes, base vs HE init, GR00T and Pi0.5) before collecting new data; this plan applies if the balanced study resumes.

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure real-robot success against own-episode count (10 / 25 / 50 per task) for GR00T and Pi0.5 started from public weights (`base`) and from HE checkpoints (`he`), on a newly collected balanced dataset.

**Architecture:** Small additions to the existing `examples/g1_dex3_training/` pipeline: a watchdog fix in the streamer, a converter for the new recordings sharing the sonic78 frame mapping, position-stratified nested subsets, a generalized fine-tune grid, and a blind trial ledger with Wilson intervals. The rest is operations: training on h100_174 in containers, delivery through private HF repos, and robot trials run by the user with the reference inference setting.

**Tech Stack:** Python 3.12+, LeRobot datasets v3, numpy, unittest (run with pytest), draccus train configs, Docker on h100 / h100_174, ZMQ streamer on PC2, Hugging Face Hub.

**Spec:** `docs/superpowers/specs/2026-10-13-g1-data-efficiency-balanced-design.md` (branch `docs/g1-balanced-data-efficiency`).

## Global Constraints

- Tasks (exact prompts): `Place the apple from the plate into the basket.` and `Place the yellow block into the gray cup.`
- 5 positions per task (p1–p5), 10 episodes per position, 50 per task.
- Subsets: 10 / 25 / 50 episodes per task, nested, 2 / 5 / 10 per position, fixed seed 0.
- VLAs `groot`, `pi05`; inits `base`, `he`; same sample budget at every size: GR00T 32 × 5,000 = 160K, Pi0.5 4 × 8 × 60,000 = 240K (EMA).
- One normalization stats file (from the 50-episode dataset) for every cell.
- Robot setting: noise 0, `--chunk-blend-s 0.3`, planner start/end, `POLICY_FPS=50`, `--max-token-step 0.05`, `--hold-rest-s 1.5`; workstation GPU 1, port 5560; Pi0.5 `POLICY_SUBDIR=pretrained_model_ema BACKBONE_DTYPE=`.
- 10 trials per task per cell, 2 per position, order p1 … p5, p1 … p5. Gate: `base` 50 ≥ 6/10 per task.
- Hosts: h100_174 GPUs 4 and 5 first, then h100 GPUs 0 and 6; **never h100 GPU 2**. Train only in containers (`-e LEROBOT_VIDEO_DECODER_CACHE_SIZE=5000`, a `--memory` cap); never install or write on the H100 hosts outside containers and mounts; never install into PC2's system Python or conda envs.
- HF transfers with `HF_HUB_DISABLE_XET=1`; delete any token file copied to a host when its transfer ends.
- Confirm with the user before merges to main, deletions of weights or datasets, and anything on the robot.
- Test command (worktree venvs are empty; use the main checkout's interpreter):
  `PYTHONPATH=$PWD/src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --python /home/jihun/work/lerobot/.venv/bin/python --no-project --with pytest pytest <tests> -q`

## Review Focus

1. A task/position with fewer usable episodes than a subset needs (a deleted retake, a < 3 s fragment): the split must fail and name the task and position, never return an unbalanced subset. → Task 4 test `test_too_few_at_a_position_raises`.
2. A source episode whose task text is not one of the two prompts (other wording, trailing space, case): the converter must stop and list those episodes, unless `--task SRC=PROMPT` overrides the source. → Task 3 test `test_match_task`.
3. A LeRobot v2.1 recording (the handover data was v2.1): the converter must stop with the conversion command instead of failing inside the loader. → Task 3 test `test_check_source_rejects_v21`.
4. Repeated or late state messages (equal timestamps) at the start of an episode: the watchdog must neither divide by zero nor inflate the speed. → Task 2 tests.
5. Trial ledger rows marked `void`, and a cell with no valid trial yet: excluded from n, no crash, interval reported as empty. → Task 6 test `test_summary_counts_and_void`.

---

### Task 1: Integration branch

**Files:** none new; merges only.

**Interfaces:**
- Produces: branch `feat/g1-balanced-de` with `feat/g1-combined-eval`, `feat/g1-pretrain-finetune`, `caisarl76/main` and `docs/g1-balanced-data-efficiency` merged; all later tasks commit here.

- [ ] **Step 1: Create the worktree from feat/g1-combined-eval**

```bash
cd /home/jihun/work/lerobot
git fetch caisarl76
git worktree add -b feat/g1-balanced-de .claude/worktrees/g1-balanced-impl feat/g1-combined-eval
cd .claude/worktrees/g1-balanced-impl
```

- [ ] **Step 2: Merge the three branches (all merged cleanly in a trial on 2026-10-13)**

```bash
git merge --no-edit feat/g1-pretrain-finetune
git merge --no-edit caisarl76/main
git merge --no-edit docs/g1-balanced-data-efficiency
git log --oneline -5
```

Expected: three merge commits, no conflicts.

- [ ] **Step 3: HF upload code.** `examples/g1_dex3_training/model_registry.py upload` (used in Task 9) exists only as uncommitted changes in the main checkout `/home/jihun/work/lerobot` (written by the HF-upload session). Ask the user whether to commit it there (with the registry `.md`/`.csv` it re-rendered) and merge `feat/g1-combined-eval` again. If the user declines, Task 9 runs the upload with that file copied from the main checkout.

- [ ] **Step 4: Run the existing tests**

```bash
PYTHONPATH=$PWD/src PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --python /home/jihun/work/lerobot/.venv/bin/python --no-project --with pytest \
  pytest tests/datasets/test_g1_dex3_wbt_splits.py tests/datasets/test_g1_dex3_finetune_grid.py tests/datasets/test_g1_dex3_stream_backends.py tests/datasets/test_g1_dex3_streamer_cli.py -q
```

Expected: all pass.

---

### Task 2: Watchdog speed from wall-clock time

Root cause (run48, 2026-10-13): `SonicBackend.begin` logs every blend-in tick with `t_ep = 0.0`, so they enter the streamer's `history`; at the first episode ticks the 3-tick window divides ~80 ms of motion by 20 ms. From the run48 log: watchdog 9.5 rad/s, true 2.4 rad/s. The same false stop ended the first run42 (6.8 rad/s on joint 11).

**Files:**
- Modify: `examples/g1_dex3_training/stream_backends.py` (add `max_arm_speed` after `ticks`)
- Modify: `examples/g1_dex3_training/sonic_policy_streamer.py` (`record`, `arm_watchdog`, `robot_states`)
- Test: `tests/datasets/test_g1_dex3_stream_backends.py`

**Interfaces:**
- Produces: `stream_backends.max_arm_speed(samples: list[tuple[float, np.ndarray]], span: int) -> tuple[int, float] | None` — samples are (wall time s, 14 arm joints); returns (joint, rad/s) over the last `span` intervals, `None` when there are too few samples or no elapsed time.

- [ ] **Step 1: Write the failing tests** (append to the test module, above `if __name__ == "__main__":`)

```python
class MaxArmSpeedTests(unittest.TestCase):
    def test_uses_wall_time_not_episode_time(self):
        # blend-in ticks share t_ep = 0; wall time still advances 20 ms per tick
        samples = [(0.02 * i, np.full(14, 0.05 * i)) for i in range(5)]
        joint, speed = stream_backends.max_arm_speed(samples, 3)
        self.assertAlmostEqual(speed, 2.5)  # 0.15 rad over 60 ms
        self.assertEqual(joint, 0)

    def test_too_few_samples_or_no_elapsed_time(self):
        self.assertIsNone(stream_backends.max_arm_speed([(0.0, np.zeros(14))] * 3, 3))
        same_time = [(1.0, np.full(14, float(i))) for i in range(5)]  # repeated state message
        self.assertIsNone(stream_backends.max_arm_speed(same_time, 3))

    def test_picks_fastest_joint(self):
        a, b = np.zeros(14), np.zeros(14)
        b[11] = 0.3
        samples = [(0.0, a), (0.02, a), (0.04, a), (0.06, b)]
        self.assertEqual(stream_backends.max_arm_speed(samples, 3), (11, 5.0))
```

- [ ] **Step 2: Run to verify they fail**

Run: the test command with `tests/datasets/test_g1_dex3_stream_backends.py -k MaxArmSpeed`
Expected: FAIL, `AttributeError: module 'stream_backends' has no attribute 'max_arm_speed'`.

- [ ] **Step 3: Implement** — in `stream_backends.py`, after `def ticks(n):`

```python
def max_arm_speed(samples: list[tuple[float, np.ndarray]], span: int) -> tuple[int, float] | None:
    """Fastest arm joint over the last `span` intervals of (wall time, 14 arm joints) samples, in rad/s.

    Wall time, not episode time: the blend-in ticks before an episode all carry t_ep = 0, which made the first
    episode ticks divide several ticks of motion by one (G1 run48: 9.5 rad/s computed, 2.4 rad/s true)."""
    if len(samples) <= span:
        return None
    (w0, a0), (w1, a1) = samples[-1 - span], samples[-1]
    if w1 <= w0:
        return None
    speed = np.abs(np.asarray(a1) - np.asarray(a0)) / (w1 - w0)
    j = int(np.argmax(speed))
    return j, float(speed[j])
```

In `sonic_policy_streamer.py`:
- add `max_arm_speed` to the `from stream_backends import (...)` list;
- in `record`, store the wall time: `history.append((t_ep, log["state"][-1], log["wall"][-1]))`;
- in `robot_states`, unpack three fields: `earlier = [s for ts, s, _ in history if ts <= t + 1e-9]`;
- in `arm_watchdog`, replace the speed block:

```python
        if len(history) <= ARM_SPEED_TICKS:
            return None
        arm = history[-1][1][:14]
        fastest = max_arm_speed([(w, s[:14]) for _, s, w in history[-1 - ARM_SPEED_TICKS :]], ARM_SPEED_TICKS)
        if a.max_arm_speed > 0 and fastest and fastest[1] > a.max_arm_speed:
            return f"arm joint {fastest[0]} measured at {fastest[1]:.1f} rad/s > {a.max_arm_speed:g}"
```

(the range check below it stays unchanged and keeps using `arm`).

- [ ] **Step 4: Run all streamer tests**

Run: the test command with `tests/datasets/test_g1_dex3_stream_backends.py tests/datasets/test_g1_dex3_streamer_cli.py`
Expected: PASS.

- [ ] **Step 5: Offline check on the run48 log** — the fixed computation must give the true speed:

```bash
uv run --no-sync python - <<'EOF'
import sys, numpy as np
sys.path.insert(0, "examples/g1_dex3_training")
from stream_backends import max_arm_speed
z = np.load("/home/jihun/g1_runs/run48_dex_groot_he25_block_t1.npz", allow_pickle=True)
keep = z["t_ep"] >= 0
s, w = z["state"][keep], z["wall"][keep]
worst = max(max_arm_speed([(w[j], s[j, :14]) for j in range(i - 3, i + 1)], 3)[1] for i in range(3, len(s)))
print(f"max watchdog speed over the run: {worst:.1f} rad/s")
EOF
```

Expected: about 4 rad/s (the measured blend-in peak, 3.95), below 6.

- [ ] **Step 6: Commit**

```bash
git add examples/g1_dex3_training/stream_backends.py examples/g1_dex3_training/sonic_policy_streamer.py tests/datasets/test_g1_dex3_stream_backends.py
git commit -m "fix(g1-dex3): arm-speed watchdog uses wall-clock time (blend-in ticks share t_ep 0; run48 false stop)"
```

---

### Task 3: Converter for the balanced recordings

**Files:**
- Create: `examples/g1_dex3_training/wbt_sonic78.py` (shared sonic78 frame mapping)
- Modify: `examples/g1_dex3_training/prepare_wbt_multitask_sonic78.py` (use `wbt_sonic78`)
- Create: `examples/g1_dex3_training/prepare_wbt_balanced_sonic78.py`
- Test: `tests/datasets/test_g1_dex3_balanced_convert.py`

**Interfaces:**
- Consumes: `wbt_splits.make_position_splits` (Task 4) — the converter's last step; implement Task 4 first or in the same session.
- Produces:
  - `wbt_sonic78.check_source(info: dict) -> None` (raises `ValueError`)
  - `wbt_sonic78.index_maps(info: dict) -> tuple[list[int], list[int]]` (state 28 indices, hand 14 indices into `action.wbc`)
  - `wbt_sonic78.create(out: Path, repo_id: str, h: int, w: int) -> LeRobotDataset`
  - `wbt_sonic78.frame(item: dict, state_idx: list[int], hand_idx: list[int], task: str) -> dict`
  - `prepare_wbt_balanced_sonic78.TASKS: tuple[str, str]`, `match_task(text: str) -> str`, `round_robin(keys_by_task: dict[str, list[str]], n: int = 5) -> dict[str, str]`
  - Output files: `OUT/` (LeRobot v3 dataset) and `OUT.splits.json` with keys `train`, `task`, `position`, `source_episode`.

- [ ] **Step 1: Write the failing tests**

```python
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2] / "examples" / "g1_dex3_training"))
from prepare_wbt_balanced_sonic78 import TASKS, match_task, round_robin
from wbt_sonic78 import check_source


class BalancedConvertTests(unittest.TestCase):
    def test_match_task(self):
        self.assertEqual(match_task("  place the apple from the plate into the basket "), TASKS[0])
        self.assertEqual(match_task("Place the yellow block into the gray cup."), TASKS[1])
        with self.assertRaises(ValueError):
            match_task("Place the yellow block into the grey mug.")  # other wording: not silently mapped

    def test_round_robin(self):
        keys = {TASKS[0]: [f"a/{i}" for i in range(7)], TASKS[1]: ["b/0", "b/1"]}
        pos = round_robin(keys)
        self.assertEqual([pos[f"a/{i}"] for i in range(7)], ["p1", "p2", "p3", "p4", "p5", "p1", "p2"])
        self.assertEqual(pos["b/1"], "p2")  # each task counts on its own

    def test_check_source_rejects_v21(self):
        with self.assertRaisesRegex(ValueError, "convert_dataset_v21_to_v30"):
            check_source({"codebase_version": "v2.1", "fps": 50})
        with self.assertRaisesRegex(ValueError, "fps"):
            check_source({"codebase_version": "v3.0", "fps": 30})
        check_source({"codebase_version": "v3.0", "fps": 50})


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run to verify they fail**

Run: the test command with `tests/datasets/test_g1_dex3_balanced_convert.py`
Expected: FAIL, `ModuleNotFoundError: No module named 'prepare_wbt_balanced_sonic78'`.

- [ ] **Step 3: Write `wbt_sonic78.py`** (the mapping moves here from `prepare_wbt_multitask_sonic78.py` unchanged)

```python
"""sonic78 frames from a GR00T-WBC teleop recording (LeRobot v3, 50 Hz), kept at 50 Hz:
  observation.state (28)  = arms 14 + Dex3 hands 14 from observation.state, by joint name (sonic_targets.ACTION_NAMES)
  action (78)             = action.motion_token 64 + hand targets 14 from action.wbc, same hand order
  observation.images.egocentric = ego_view (the HE models' camera key, so HE checkpoints can be fine-tuned)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sonic_targets import ACTION_NAMES

from lerobot.datasets.lerobot_dataset import LeRobotDataset

FPS = 50


def check_source(info: dict) -> None:
    version = str(info.get("codebase_version", ""))
    if not version.startswith("v3"):
        raise ValueError(
            f"source is LeRobot {version or '?'}; convert a copy first: python -m "
            "lerobot.scripts.convert_dataset_v21_to_v30 --repo-id local/src --root COPY --push-to-hub false"
        )
    if info["fps"] != FPS:
        raise ValueError(f"source fps {info['fps']}, expected {FPS}")


def index_maps(info: dict) -> tuple[list[int], list[int]]:
    state_names = info["features"]["observation.state"]["names"]
    wbc_names = info["features"]["action.wbc"]["names"]
    return [state_names.index(n) for n in ACTION_NAMES], [wbc_names.index(n) for n in ACTION_NAMES[14:]]


def create(out: Path, repo_id: str, h: int, w: int) -> LeRobotDataset:
    return LeRobotDataset.create(
        repo_id,
        fps=FPS,
        root=out,
        robot_type="unitree_g1_dex3_sonic",
        features={
            "observation.state": {"dtype": "float32", "shape": (28,), "names": list(ACTION_NAMES)},
            "action": {"dtype": "float32", "shape": (78,), "names": [f"sonic_token_{i}" for i in range(64)] + list(ACTION_NAMES[14:])},
            "observation.images.egocentric": {"dtype": "video", "shape": (h, w, 3), "names": ["height", "width", "channels"]},
        },
    )  # fmt: skip


def frame(item: dict, state_idx: list[int], hand_idx: list[int], task: str) -> dict:
    return {
        "observation.state": item["observation.state"].numpy()[state_idx].astype(np.float32),
        "action": np.concatenate([item["action.motion_token"].numpy(), item["action.wbc"].numpy()[hand_idx]]).astype(np.float32),
        "observation.images.egocentric": (item["observation.images.ego_view"].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8),
        "task": task,
    }  # fmt: skip
```

- [ ] **Step 4: Point `prepare_wbt_multitask_sonic78.py` at it.** Replace its `state_names`/`wbc_names`/`state_idx`/`hand_idx` lines with `check_source(info); state_idx, hand_idx = index_maps(info)` (drop its `assert info["fps"] == 50`), replace the `LeRobotDataset.create(...)` call with `ds = create(out, repo_id, h, w)`, and the `ds.add_frame({...})` body with `ds.add_frame(frame(source[i], state_idx, hand_idx, prompts[old]))`. Import `from wbt_sonic78 import check_source, create, frame, index_maps`; drop now-unused imports (`ACTION_NAMES`, `np` if unused). Behaviour is unchanged.

- [ ] **Step 5: Write `prepare_wbt_balanced_sonic78.py`**

```python
"""Convert the balanced G1 WBT teleop recordings (2 tasks x 5 positions x 10 episodes, LeRobot v3, 50 Hz) to sonic78
and write OUT.splits.json: nested 10/25/50-episode-per-task training lists stratified by position (wbt_splits).

Task per episode = the source episode's task text, which must be one of TASKS (case and surrounding space ignored);
--task SRC_NAME=PROMPT sets it for a whole source instead. Position per episode = --positions FILE (JSON
{"SRC_NAME/EPISODE": "p3"}), or by default round robin p1..p5 in recording order per task (the collection protocol).
Episodes under MIN_S are dropped before positions are assigned.

Usage: python prepare_wbt_balanced_sonic78.py OUT SRC [SRC ...] [--task SRC_NAME=PROMPT ...] [--positions FILE]
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from wbt_sonic78 import FPS, check_source, create, frame, index_maps
from wbt_splits import make_position_splits

from lerobot.datasets.lerobot_dataset import LeRobotDataset

TASKS = ("Place the apple from the plate into the basket.", "Place the yellow block into the gray cup.")
MIN_S = 3.0


def _norm(text: str) -> str:
    return text.strip().lower().rstrip(".")


def match_task(text: str) -> str:
    for t in TASKS:
        if _norm(text) == _norm(t):
            return t
    raise ValueError(f"task {text!r} is none of {TASKS}; fix the recording or pass --task SRC_NAME=PROMPT")


def round_robin(keys_by_task: dict[str, list[str]], n: int = 5) -> dict[str, str]:
    return {k: f"p{i % n + 1}" for keys in keys_by_task.values() for i, k in enumerate(keys)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("out", type=Path)
    p.add_argument("sources", type=Path, nargs="+")
    p.add_argument("--task", action="append", default=[], help="SRC_NAME=PROMPT")
    p.add_argument("--positions", type=Path)
    p.add_argument("--repo-id")
    a = p.parse_args()
    overrides = dict(t.split("=", 1) for t in a.task)

    episodes, bad = [], []  # (key, source dataset, episode, task) in recording order
    for src in a.sources:
        info = json.loads((src / "meta/info.json").read_text())
        check_source(info)
        source = LeRobotDataset(f"local/{src.name}", root=src, video_backend="torchcodec")
        meta = source.meta.episodes
        for e in range(len(meta["length"])):
            if meta["length"][e] / FPS < MIN_S:
                continue
            text = overrides.get(src.name) or meta["tasks"][e][0]
            try:
                episodes.append((f"{src.name}/{e}", source, e, match_task(text)))
            except ValueError:
                bad.append(f"{src.name}/{e}: {text!r}")
    if bad:
        raise SystemExit("episodes with an unknown task:\n  " + "\n  ".join(bad))
    by_task = defaultdict(list)
    for key, _, _, task in episodes:
        by_task[task].append(key)
    positions = json.loads(a.positions.read_text()) if a.positions else round_robin(by_task)

    first = episodes[0][1]
    h, w, _ = first.meta.info["features"]["observation.images.ego_view"]["shape"]
    state_idx, hand_idx = index_maps(first.meta.info)
    ds = create(a.out, a.repo_id or f"local/{a.out.name}", h, w)
    lengths, labels, source_episode = [], {}, {}
    for new, (key, source, e, task) in enumerate(episodes):
        start, end = source.meta.episodes["dataset_from_index"][e], source.meta.episodes["dataset_to_index"][e]
        for i in range(start, end):
            ds.add_frame(frame(source[i], state_idx, hand_idx, task))
        ds.save_episode()
        labels[new], source_episode[new] = (task, positions[key]), key
        lengths.append((end - start) / FPS)
        print(f"episode {new} ({key}, {positions[key]}): {end - start} frames", flush=True)
    ds.finalize()

    splits = make_position_splits(labels)
    splits["source_episode"] = source_episode
    Path(f"{a.out}.splits.json").write_text(json.dumps(splits, indent=1) + "\n")
    counts = defaultdict(int)
    for task, pos in labels.values():
        counts[task, pos] += 1
    for (task, pos), n in sorted(counts.items()):
        print(f"{task} {pos}: {n}")
    p95 = float(np.percentile(lengths, 95))
    print(f"done: {len(labels)} episodes -> {a.out}; p95 length {p95:.1f} s -> episode limit {2 * p95:.0f} s")


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run the tests**

Run: the test command with `tests/datasets/test_g1_dex3_balanced_convert.py tests/datasets/test_g1_dex3_wbt_splits.py`
Expected: PASS.

- [ ] **Step 7: Regression check of the refactored multitask converter** on 3 episodes of the existing draft is not possible without its source on this machine; instead diff its first output frame against the existing dataset after Task 7 if the source is present, else skip and say so in the commit message. Byte-compare is not expected (video re-encode); compare `observation.state` and `action` of episode 0 frame 0 exactly.

- [ ] **Step 8: Commit**

```bash
git add examples/g1_dex3_training/wbt_sonic78.py examples/g1_dex3_training/prepare_wbt_multitask_sonic78.py examples/g1_dex3_training/prepare_wbt_balanced_sonic78.py tests/datasets/test_g1_dex3_balanced_convert.py
git commit -m "feat(g1-dex3): converter for the balanced 2-task WBT recordings; sonic78 frame mapping shared with the multi-task converter"
```

---

### Task 4: Position-stratified nested subsets

**Files:**
- Modify: `examples/g1_dex3_training/wbt_splits.py` (add `make_position_splits`)
- Test: `tests/datasets/test_g1_dex3_wbt_splits.py`

**Interfaces:**
- Produces: `wbt_splits.make_position_splits(episodes: dict[int, tuple[str, str]], sizes=(10, 25, 50), seed=0) -> dict` with `{"train": {"10": [...], "25": [...], "50": [...]}, "task": {ep: task}, "position": {ep: position}}`; sizes are episodes per task.

- [ ] **Step 1: Write the failing tests** (add to `WbtSplitTests`; import `make_position_splits`)

```python
    def _balanced(self, per_position=10):
        eps, e = {}, 0
        for task in ("apple", "block"):
            for p in range(1, 6):
                for _ in range(per_position):
                    eps[e] = (task, f"p{p}")
                    e += 1
        return eps

    def test_position_splits_nested_and_stratified(self):
        eps = self._balanced()
        s = make_position_splits(eps)
        self.assertEqual(s, make_position_splits(eps))
        t10, t25, t50 = (set(s["train"][k]) for k in ("10", "25", "50"))
        self.assertTrue(t10 <= t25 <= t50)
        self.assertEqual(t50, set(eps))
        for size, chosen in ((10, t10), (25, t25)):
            for task in ("apple", "block"):
                for p in range(1, 6):
                    n = sum(eps[e] == (task, f"p{p}") for e in chosen)
                    self.assertEqual(n, size // 5)
        self.assertNotEqual(sorted(t10)[:2], [0, 1])  # drawn with the seed, not the first recorded

    def test_too_few_at_a_position_raises(self):
        eps = self._balanced()
        del eps[3]  # one p1 apple episode lost: 9 left, size 50 needs 10
        with self.assertRaisesRegex(ValueError, "apple.*p1"):
            make_position_splits(eps)
```

- [ ] **Step 2: Run to verify they fail**

Run: the test command with `tests/datasets/test_g1_dex3_wbt_splits.py`
Expected: FAIL, `ImportError: cannot import name 'make_position_splits'`.

- [ ] **Step 3: Implement** (append to `wbt_splits.py`; add `from collections import defaultdict` at the top)

```python
def make_position_splits(episodes: dict[int, tuple[str, str]], sizes=(10, 25, 50), seed=0) -> dict:
    """episodes: {episode: (task, position)}; sizes are episodes per task. Per (task, position), sorted, a seeded
    shuffle; each size takes the first size / n_positions of it, so sizes are nested and every position gets the
    same share. Raises if a size does not divide by the task's positions or a position has too few episodes."""
    rng = np.random.default_rng(seed)
    groups = defaultdict(list)
    for e, key in sorted(episodes.items()):
        groups[key].append(e)
    n_pos = defaultdict(int)
    for task, _ in groups:
        n_pos[task] += 1
    train = {str(s): [] for s in sizes}
    for (task, pos), members in sorted(groups.items()):
        order = [members[i] for i in rng.permutation(len(members))]
        for s in sizes:
            k, rest = divmod(s, n_pos[task])
            if rest:
                raise ValueError(f"size {s} is not a multiple of the {n_pos[task]} positions of {task!r}")
            if len(order) < k:
                raise ValueError(f"{task!r} {pos}: {len(order)} episodes, size {s} needs {k}")
            train[str(s)] += order[:k]
    return {
        "train": {k: sorted(v) for k, v in train.items()},
        "task": {e: t for e, (t, _) in episodes.items()},
        "position": {e: p for e, (_, p) in episodes.items()},
    }
```

- [ ] **Step 4: Run the tests**

Run: the test command with `tests/datasets/test_g1_dex3_wbt_splits.py`
Expected: PASS (old and new tests).

- [ ] **Step 5: Commit**

```bash
git add examples/g1_dex3_training/wbt_splits.py tests/datasets/test_g1_dex3_wbt_splits.py
git commit -m "feat(g1-dex3): position-stratified nested 10/25/50 subsets for the balanced WBT data"
```

---

### Task 5: Fine-tune grid for the balanced study

**Files:**
- Modify: `examples/g1_dex3_training/make_finetune_grid.py`
- Test: `tests/datasets/test_g1_dex3_finetune_grid.py`

**Interfaces:**
- Consumes: `OUT.splits.json` from Task 3 (`train` keyed by size).
- Produces: `grid(templates, splits, dataset_root, repo_id, job_root, inits=INITS, sizes=FRACTIONS, prefix="dex") -> dict[str, dict]` (defaults keep the old grid), `smoke(config: dict) -> dict`; job names `{model}_bal{size}_{init}_full` with `--prefix bal`; CLI flags `--models --inits --sizes --prefix --smoke-for`.

- [ ] **Step 1: Write the failing test** (add to `GridTests`; import `smoke`)

```python
    def test_balanced_grid_and_smoke(self):
        splits = {"train": {"10": [1], "25": [1, 2], "50": [1, 2, 3]}}
        templates = {("groot", "base"): TEMPLATE, ("groot", "he"): TEMPLATE}
        g = grid(templates, splits, "/new", "local/new", "/jobs", inits=("base", "he"), sizes=("10", "25", "50"), prefix="bal")
        self.assertEqual(sorted(g), sorted(f"groot_bal{n}_{i}_full" for n in (10, 25, 50) for i in ("base", "he")))
        self.assertEqual(g["groot_bal25_he_full"]["dataset"]["episodes"], [1, 2])
        self.assertEqual(g["groot_bal10_he_full"]["policy"]["pretrained_path"], "/jobs/init/groot_he")
        s = smoke(g["groot_bal50_base_full"])
        self.assertEqual((s["job_name"], s["steps"]), ("groot_bal50_base_smoke", 20))
        self.assertEqual(s["output_dir"], "/jobs/runs/groot_bal50_base_smoke")
        self.assertEqual(g["groot_bal50_base_full"]["steps"], 5000)  # smoke copies, does not modify
```

- [ ] **Step 2: Run to verify it fails**

Run: the test command with `tests/datasets/test_g1_dex3_finetune_grid.py`
Expected: FAIL, `ImportError: cannot import name 'smoke'`.

- [ ] **Step 3: Implement.** In `grid`, add the parameters and use them:

```python
def grid(templates, splits, dataset_root, repo_id, job_root, inits=INITS, sizes=FRACTIONS, prefix="dex"):
    out = {}
    for (model, kind), template in templates.items():
        for init in ("base",) if kind == "base" else tuple(i for i in inits if i != "base"):
            if init not in inits:
                continue
            for pct in sizes:
                name = f"{model}_{prefix}{pct}_{init}_full"
                ...  # body unchanged
```

Add after `grid`:

```python
def smoke(config: dict) -> dict:
    """A 20-step copy of a full run; official_queue.sh skips the full run when its smoke run fails."""
    c = copy.deepcopy(config)
    name = c["job_name"].removesuffix("_full") + "_smoke"
    c.update(steps=20, save_freq=20, job_name=name, output_dir=str(Path(c["output_dir"]).with_name(name)))
    return c
```

Replace `main` with an argparse version (positional arguments as before; defaults reproduce the dex grid):

```python
def main() -> None:
    import argparse

    p = argparse.ArgumentParser()
    for name in ("template_dir", "splits", "root", "repo_id", "job_root", "out_dir"):
        p.add_argument(name)
    p.add_argument("gpus", nargs="+")
    p.add_argument("--models", nargs="+", default=list(MODELS))
    p.add_argument("--inits", nargs="+", default=list(INITS))
    p.add_argument("--sizes", nargs="+", default=list(FRACTIONS))
    p.add_argument("--prefix", default="dex")
    p.add_argument("--smoke-for", nargs="*", default=[], help="full job names that also get a _smoke config")
    a = p.parse_args()
    templates = {
        (m, k): json.loads((Path(a.template_dir) / f"{m}_wbt50_{k}_full.json").read_text())
        for m in a.models
        for k in ("base", "he")
    }
    configs = grid(templates, json.loads(Path(a.splits).read_text()), a.root, a.repo_id, a.job_root, a.inits, a.sizes, a.prefix)
    configs.update({smoke(configs[n])["job_name"]: smoke(configs[n]) for n in a.smoke_for})
    Path(a.out_dir).mkdir(parents=True, exist_ok=True)
    for name, c in configs.items():
        (Path(a.out_dir) / f"{name}.json").write_text(json.dumps(c, indent=1) + "\n")
    names = sorted((n for n in configs if n.endswith("_full")), key=lambda n: (n.startswith("pi05"), n))
    for i, gpu in enumerate(a.gpus):
        (Path(a.out_dir) / f"queue_gpu{gpu}.txt").write_text(
            "".join(f"{Path(a.job_root).name}/{n}\n" for n in names[i :: len(a.gpus)])
        )
    print(f"{len(configs)} configs, queues for GPUs {a.gpus}")
```

Update the module docstring's usage line to show the flags and the balanced example.

- [ ] **Step 4: Run the tests**

Run: the test command with `tests/datasets/test_g1_dex3_finetune_grid.py`
Expected: PASS (old `test_grid` still gives 9 configs with the defaults).

- [ ] **Step 5: Commit**

```bash
git add examples/g1_dex3_training/make_finetune_grid.py tests/datasets/test_g1_dex3_finetune_grid.py
git commit -m "feat(g1-dex3): fine-tune grid takes models, inits, sizes and a name prefix; smoke copies for the first runs"
```

---

### Task 6: Blind trial codes and the result table

**Files:**
- Create: `examples/g1_dex3_training/robot_trials.py`
- Test: `tests/datasets/test_g1_dex3_robot_trials.py`

**Interfaces:**
- Produces:
  - `assign_codes(cells: list[str], seed: int) -> dict[str, str]` — `{"M1": cell, ...}` in a seeded random order.
  - `wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float] | None`
  - `summarize(rows: list[dict], codes: dict[str, str]) -> dict[tuple[str, str], dict]` — key (cell, task); value `{"n", "success", "partial", "fail", "void", "ci"}`; `n` excludes void.
  - CLI: `robot_trials.py codes CODES_JSON CELL ...` (refuses to overwrite), `robot_trials.py table TRIALS_CSV CODES_JSON` (markdown table + gate lines).
  - Ledger CSV columns: `run,code,task,position,outcome,note`; outcome ∈ success | partial | fail | void.

- [ ] **Step 1: Write the failing tests**

```python
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2] / "examples" / "g1_dex3_training"))
from robot_trials import assign_codes, gate, summarize, wilson


class RobotTrialsTests(unittest.TestCase):
    def test_wilson(self):
        lo, hi = wilson(8, 10)
        self.assertAlmostEqual(lo, 0.490, places=3)
        self.assertAlmostEqual(hi, 0.943, places=3)
        self.assertEqual(wilson(0, 10)[0], 0.0)
        self.assertAlmostEqual(wilson(0, 10)[1], 0.278, places=3)
        self.assertEqual(wilson(10, 10)[1], 1.0)
        self.assertIsNone(wilson(0, 0))

    def test_codes_cover_cells_and_are_seeded(self):
        cells = ["groot_bal50_base", "groot_bal50_he", "pi05_bal50_base"]
        c = assign_codes(cells, seed=7)
        self.assertEqual(sorted(c), ["M1", "M2", "M3"])
        self.assertEqual(sorted(c.values()), sorted(cells))
        self.assertEqual(c, assign_codes(cells, seed=7))

    def test_summary_counts_and_void(self):
        codes = {"M1": "groot_bal50_base", "M2": "groot_bal50_he"}
        rows = [{"code": "M1", "task": "apple", "outcome": o} for o in ["success"] * 6 + ["fail"] * 3 + ["partial", "void"]]
        rows.append({"code": "M2", "task": "apple", "outcome": "void"})
        s = summarize(rows, codes)
        cell = s["groot_bal50_base", "apple"]
        self.assertEqual((cell["n"], cell["success"], cell["partial"], cell["void"]), (10, 6, 1, 1))
        self.assertIsNone(s["groot_bal50_he", "apple"]["ci"])  # only a void trial so far
        self.assertEqual(gate(s, "groot", "apple"), True)  # base 50: 6/10 passes

    def test_unknown_outcome_rejected(self):
        with self.assertRaises(ValueError):
            summarize([{"code": "M1", "task": "apple", "outcome": "succes"}], {"M1": "x"})


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run to verify they fail**

Run: the test command with `tests/datasets/test_g1_dex3_robot_trials.py`
Expected: FAIL, `ModuleNotFoundError: No module named 'robot_trials'`.

- [ ] **Step 3: Implement `robot_trials.py`**

```python
"""Blind robot-trial ledger for the balanced data-efficiency study.

Servers run under codes (M1, M2, ...) so the operator scores without knowing the model; the code -> cell map
(CODES_JSON, kept out of the operator's view) is read only by `table`. Cells are <vla>_bal<size>_<init>.
Ledger CSV columns: run,code,task,position,outcome,note; outcome success | partial | fail | void
(void = watchdog stop in the first 0.5 s of an episode, rerun once; excluded from n).

Usage: python robot_trials.py codes CODES_JSON CELL [CELL ...]
       python robot_trials.py table TRIALS_CSV CODES_JSON
"""

from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

OUTCOMES = ("success", "partial", "fail", "void")
GATE = 6  # base at the largest size must reach this many successes of 10


def assign_codes(cells: list[str], seed: int) -> dict[str, str]:
    order = np.random.default_rng(seed).permutation(len(cells))
    return {f"M{i + 1}": cells[j] for i, j in enumerate(order)}


def wilson(successes: int, n: int, z: float = 1.96) -> tuple[float, float] | None:
    if n == 0:
        return None
    p, d = successes / n, 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def summarize(rows: list[dict], codes: dict[str, str]) -> dict[tuple[str, str], dict]:
    out = defaultdict(lambda: dict.fromkeys(OUTCOMES, 0))
    for r in rows:
        if r["outcome"] not in OUTCOMES:
            raise ValueError(f"unknown outcome {r['outcome']!r} in {r}")
        out[codes[r["code"]], r["task"]][r["outcome"]] += 1
    for v in out.values():
        v["n"] = v["success"] + v["partial"] + v["fail"]
        v["ci"] = wilson(v["success"], v["n"])
    return dict(out)


def gate(summary: dict, vla: str, task: str) -> bool:
    sizes = [int(c.split("_bal")[1].split("_")[0]) for c, t in summary if c.startswith(f"{vla}_bal") and t == task]
    cell = summary.get((f"{vla}_bal{max(sizes)}_base", task)) if sizes else None
    return bool(cell) and cell["success"] >= GATE


def main() -> None:
    cmd = sys.argv[1]
    if cmd == "codes":
        path = Path(sys.argv[2])
        if path.exists():
            raise SystemExit(f"{path} exists; codes are assigned once")
        path.write_text(json.dumps(assign_codes(sys.argv[3:], seed=int.from_bytes(__import__("os").urandom(4))), indent=1) + "\n")
        print(f"codes written to {path} (do not show it to the operator)")
    elif cmd == "table":
        rows = list(csv.DictReader(open(sys.argv[2])))
        s = summarize(rows, json.loads(Path(sys.argv[3]).read_text()))
        print("| cell | task | success | partial | fail | void | n | 95 % CI |\n|---|---|---|---|---|---|---|---|")
        for (cell, task), v in sorted(s.items()):
            ci = "-" if v["ci"] is None else f"{v['ci'][0]:.2f}–{v['ci'][1]:.2f}"
            print(f"| {cell} | {task} | {v['success']} | {v['partial']} | {v['fail']} | {v['void']} | {v['n']} | {ci} |")
        for vla, task in sorted({(c.split("_bal")[0], t) for c, t in s}):
            print(f"gate {vla} {task}: {'pass' if gate(s, vla, task) else 'not passed'}")
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests**

Run: the test command with `tests/datasets/test_g1_dex3_robot_trials.py`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add examples/g1_dex3_training/robot_trials.py tests/datasets/test_g1_dex3_robot_trials.py
git commit -m "feat(g1-dex3): blind robot-trial codes and success table with Wilson intervals and the base-50 gate"
```

---

### Task 7: Data intake and conversion (after the user posts the dataset location)

**Files:** none in the repo; outputs under `/mnt/data/jihun/datasets/G1_WBT_GR00T/`.

**Interfaces:**
- Consumes: the user's recording(s) `SRC...`; Task 3 converter.
- Produces: `/mnt/data/jihun/datasets/G1_WBT_GR00T/wbt_balanced_sonic78_50hz_<YYMMDD>` (+ `.splits.json`) and the episode limit in seconds.

- [ ] **Step 1: Inspect each source** (read-only): `meta/info.json` (`codebase_version`, `fps`, features `observation.state`, `action.motion_token`, `action.wbc`, `observation.images.ego_view`), episode count, per-episode task text and length. If `codebase_version` is v2.1, copy the source and convert the copy: `uv run --no-sync python -m lerobot.scripts.convert_dataset_v21_to_v30 --repo-id local/src --root <copy> --push-to-hub false`.
- [ ] **Step 2: Ask the user** how positions were recorded if the episode counts per task are not multiples of 5 or the task texts differ from the prompts (answers decide `--positions` / `--task`).
- [ ] **Step 3: Convert**

```bash
cd examples/g1_dex3_training
OUT=/mnt/data/jihun/datasets/G1_WBT_GR00T/wbt_balanced_sonic78_50hz_$(date +%y%m%d)
uv run --no-sync python prepare_wbt_balanced_sonic78.py $OUT SRC [SRC ...] [--task ...] [--positions ...]
```

Expected: 100 episodes; 10 per (task, position) printed; `.splits.json` with 20 / 50 / 100 training episodes for sizes 10 / 25 / 50; the printed episode limit.

- [ ] **Step 4: Spot-check** one episode per task with the task-sheet frames (start / grasp / end) as on 2026-10-13, and check the hand used is the right hand in every episode (closure of the right Dex3 > left). Report any left-hand episode to the user.
- [ ] **Step 5: Copy to h100_174** under the job root `/mnt/data01/jhkim/model_weight/g1_dex3_20260922/g1_wbt_bal_<YYMMDD>/datasets/` through a container mount (rsync from the workstation over `ssh -A h100 'ssh kube@192.168.75.174 ...'`, as for earlier datasets). Verify the file count and total size match.

---

### Task 8: Stage 1 training (base 50, GR00T and Pi0.5)

**Files:** configs in the job root; none in the repo.

**Interfaces:**
- Consumes: Task 5 grid CLI, Task 7 dataset, templates `/run-output/g1_wbt_handover_20261002/configs/{groot,pi05}_wbt50_{base,he}_full.json` (the dex grid's templates).
- Produces: job root `g1_wbt_bal_<YYMMDD>` with `configs/`, `runs/groot_bal50_base_full`, `runs/pi05_bal50_base_full`, `.exit` files.

- [ ] **Step 1: Generate all configs** (in a training container on h100_174 with the job root mounted at `/run-output`):

```bash
J=/run-output/g1_wbt_bal_<YYMMDD>; D=$J/datasets/wbt_balanced_sonic78_50hz_<YYMMDD>
python examples/g1_dex3_training/make_finetune_grid.py /run-output/g1_wbt_handover_20261002/configs $D.splits.json $D \
  local/wbt_balanced_sonic78_50hz_<YYMMDD> $J $J/configs 4 5 \
  --models groot pi05 --inits base he --sizes 10 25 50 --prefix bal \
  --smoke-for groot_bal50_base_full pi05_bal50_base_full
```

Expected: `14 configs` (12 full + 2 smoke).

- [ ] **Step 2: Stage-1 queues by hand**: `$J/queue_stage1_gpu4.txt` = `g1_wbt_bal_<YYMMDD>/groot_bal50_base_smoke`, `g1_wbt_bal_<YYMMDD>/groot_bal50_base_full`; `$J/queue_stage1_gpu5.txt` = the same two lines for `pi05`.
- [ ] **Step 3: Start one container per GPU** (h100_174 GPUs 4 and 5, the image of the dex-grid containers — `docker images` / `docker inspect` an existing `jihun-lerobot-*-174-*` container for its id and mounts), with `--gpus device=<g>`, `--memory 200g`, `-e LEROBOT_VIDEO_DECODER_CACHE_SIZE=5000`, the job-root mount at `/run-output`, then run `official_queue.sh /run-output/g1_wbt_bal_<YYMMDD>/queue_stage1_gpu<g>.txt` inside it in tmux.
- [ ] **Step 4: Watch the smoke runs** (exit 0, loss finite) and the first 500 steps of each full run; report expected end times (GR00T ~2 h, Pi0.5 ~6 h on the dex grid).
- [ ] **Step 5: On finish**, check `.exit` = 0, `checkpoints/last/pretrained_model` (and `pretrained_model_ema` for Pi0.5) exist; delete the smoke run dirs.

---

### Task 9: Delivery to the workstation

**Files:** registry `.md`/`.csv` re-rendered in the repo.

**Interfaces:**
- Consumes: finished runs; `model_registry.py upload` (Task 1 Step 3).
- Produces: `/mnt/data/jihun/g1_models/wbt_bal_<run>/pretrained_model` (GR00T) or `/pretrained_model_ema` (Pi0.5); private repos `mncai/g1-wbt-bal-<run>`.

- [ ] **Step 1: Upload** from h100_174 in a container with a token file copied for the transfer only: `model_registry.py collect h100_174` piped into `upload h100_174 mncai <run> ...` with `HF_HUB_DISABLE_XET=1`. For Pi0.5 also upload `pretrained_model_ema/` into the same repo (the dex-grid upload missed it): `huggingface-cli upload mncai/g1-wbt-bal-<run> <run_dir>/checkpoints/last/pretrained_model_ema pretrained_model_ema --private`.
- [ ] **Step 2: Delete the token file and the upload container** on h100_174.
- [ ] **Step 3: Download** on the workstation: `HF_HUB_DISABLE_XET=1 uv run --no-sync python -c "from huggingface_hub import snapshot_download as s; s('mncai/g1-wbt-bal-<run>', local_dir='/mnt/data/jihun/g1_models/wbt_bal_<run>', allow_patterns=['pretrained_model/*'])"` (Pi0.5: `pretrained_model_ema/*`); compare file sizes with the Hub listing.
- [ ] **Step 4: Load check** on workstation GPU 1 with the launcher `server` step (GR00T bf16; Pi0.5 `POLICY_SUBDIR=pretrained_model_ema BACKBONE_DTYPE=`), then `stop`.
- [ ] **Step 5: Re-render the registry** (`model_registry.py render ...`) and commit `docs/research/g1-model-registry.{md,csv}`.

---

### Task 10: Robot gate (stage 1 trials)

**Files:** ledger `/mnt/data/jihun/g1_eval/balanced/trials.csv`, codes `/mnt/data/jihun/g1_eval/balanced/codes.json` (outside the repo, not shown to the operator).

**Interfaces:**
- Consumes: Task 2 fix, Task 6 CLI, Task 9 workstation copies.
- Produces: gate result per (VLA, task).

- [ ] **Step 1: Deploy the watchdog fix to PC2**: back up `~/g1_sonic_eval/code/{sonic_policy_streamer.py,stream_backends.py}` as `*.bak-<YYMMDD>`, copy the two files from the branch, then `diff` each against the branch copy.
- [ ] **Step 2: Assign codes once for all 12 cells**: `robot_trials.py codes /mnt/data/jihun/g1_eval/balanced/codes.json groot_bal{10,25,50}_{base,he} pi05_bal{10,25,50}_{base,he}` (brace-expanded); create `trials.csv` with the header `run,code,task,position,outcome,note`.
- [ ] **Step 3: Handoff check with the user** (no scoring): one 10 s run of `groot_bal50_base` at p1; confirm no watchdog stop at the episode start, and from the log that the blend-in peak of joint 11 is below 6 rad/s.
- [ ] **Step 4: Trials**: for each stage-1 cell, start the server, tell the user only its code, and give the stream commands for 10 trials per task (run names `runNN_<code>_<task>_p<pos>_t<k>`, next free NN from `ls ~/g1_sonic_eval/runs`, positions p1…p5, p1…p5, prompt per Global Constraints, `--duration-s` = Task 7 episode limit). Record each outcome the user reports as one ledger row; a watchdog stop within 0.5 s of the episode start is `void` and gets one rerun.
- [ ] **Step 5: Gate**: `robot_trials.py table trials.csv codes.json`; report the gate lines and the stage-1 table to the user, and unmask the two stage-1 cells. If a task fails the gate for a VLA, stop and decide with the user (more episodes, easier layout) before Task 11.

---

### Task 11: Stage 2 training and delivery

- [ ] **Step 1: HE inits**: copy `/run-output/g1_wbt_dex_261009/init/{groot_he,pi05_he}` to `$J/init/` and retarget them to the new dataset in the training container: `python examples/g1_dex3_training/retarget_init_checkpoint.py $J/init/groot_he $D` (and `pi05_he`).
- [ ] **Step 2: Start queue runners** on the generated `$J/configs/queue_gpu4.txt` and `queue_gpu5.txt` (stage-1 jobs are skipped because their `.exit` exists); add h100 GPUs 0 and 6 only if the user agrees (copy the dataset and inits there first).
- [ ] **Step 3: Monitor** to completion as in Task 8 Steps 4–5.
- [ ] **Step 4: Deliver** the 10 runs as in Task 9.

---

### Task 12: Stage 2 trials

- [ ] **Step 1: Order**: `he` 50 first; then size 25 and size 10, at each size the two inits in a random order (tell the user codes only). Skip a cell only when the larger size of the same (VLA, init, task) scored 0/10; record it in the results doc as 0/10 untested.
- [ ] **Step 2: Trials** exactly as Task 10 Step 4, same ledger.
- [ ] **Step 3: Table** after each cell; unmask each cell after its 10 trials are recorded.
- [ ] **Step 4: Extra trials**: when a headline comparison (`he` at size N vs `base` 50, or `he` vs `base` at the same N) has overlapping 95 % intervals and the user agrees, run 10 more trials for those cells only.

---

### Task 13: Results doc and PR

**Files:**
- Create: `docs/research/<YYYY-MM-DD>-g1-balanced-data-efficiency.md`
- Modify: `AGENTS.md` (pointer next to the existing G1 Dex3 research notes)

- [ ] **Step 1: Write the doc**: question, data (tasks, positions, counts, episode limit), training grid (12 runs, recipes, registry link), the trial table from `robot_trials.py table`, success-vs-episodes per (VLA, task) with intervals, the headline (smallest size at which `he` matches `base` 50), smoothness and watchdog counts from `robot_run_smoothness.py`, the earlier grid's apple-tray runs as a data gap, and the run list.
- [ ] **Step 2: Update `AGENTS.md`** with one line pointing to the doc.
- [ ] **Step 3: Run all touched tests** (Tasks 2–6 test files) and `pre-commit run --files <changed files>`.
- [ ] **Step 4: Commit and push** the branch; ask the user before opening the PR to `main` (it also carries `feat/g1-combined-eval` and `feat/g1-pretrain-finetune`).
