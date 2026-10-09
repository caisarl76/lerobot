# G1 VLA pretrain → fine-tune data efficiency Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Show how much less of our own teleop data a VLA needs when it is pretrained on open datasets: fine-tune
GR00T, Pi0.5 and XR-1 from three starting points (base, HE, HE+Unitree) on 25 / 50 / 100 % of our G1 multi-task
teleop dataset and measure them on the same held-out episodes.

**Architecture:** Convert the annotated teleop draft to our sonic78 50 Hz training format with one stable task prompt
per episode, a stratified held-out set and nested training fractions. Rebuild the combined HE+Unitree dataset from the
corrected Unitree data and pretrain the three VLAs on it. Generate the 27 fine-tune configs from the handover-bottle
configs (same per-model sample budget for every fraction), run them as per-GPU queues, then score every run open loop
and in the 50 Hz closed-loop SONIC sim against the measured joints, and tabulate the data-efficiency curve.

**Tech Stack:** LeRobot (v3 datasets, `lerobot_train`, `aggregate_datasets`), the `examples/g1_dex3_training/` tools
(`prepare_wbt_sonic78.py`, `retarget_init_checkpoint.py`, `xr1_action_stats.py`, `official_queue.sh`,
`openloop_smooth.py`, `sonic_official_sim_eval.sh`, `wbt_joint28_reference.py`, `sonic_stream_eval.py`), Docker on
the H100 hosts, Python 3.12 / numpy / pandas.

**Spec:** this document's "Design" section (decisions taken with the user on 2026-10-09) plus the precedent
`docs/research/2026-10-02-g1-wbt-handover-training.md` (branch `feat/g1-combined-eval`), whose pipeline this reuses.

## Design (decisions, 2026-10-09)

- **Goal (user):** not zero-shot accuracy. Pretrain on open data (HE, Unitree), fine-tune on our teleop data, and show
  HE pretraining needs far fewer of our episodes.
- **Dataset:** `/mnt/data/jihun/datasets/G1_WBT_GR00T/official_annotations/workspace/drafts/e3dad2ee0371440cb40ff4ff09369a96/episode_000101`
  (LeRobot v3, 50 Hz, 102 episodes, `ego_view` 640×480, `action.motion_token` 64, `action.wbc` / `observation.state`
  43 with URDF joint names). Prompts come from `meta/lerobot_annotations.json` (`atoms` with `style == "task_aug"`).
  - Use the **87 episodes with prompts, minus fragments shorter than 3 s** (episode 76, 2.9 s) → 86 episodes.
    Without prompts: 59, 68, 81, 90–101 (mostly 1–4 s).
  - Task text per episode = its **first** `task_aug` prompt (stable for training and evaluation). The 28 episodes
    whose prompt is the generic "Place the manipulated object on the table." keep it.
  - The draft is unreviewed (only episode 0). If the review changes prompts or episodes, re-run Task 1 and the grid.
- **VLAs:** GR00T, Pi0.5, XR-1 (XR-1 in its official form, with state input, for all three starting points, so the
  only difference between runs is the starting point and the data fraction).
- **Starting points:** `base` (public weights), `he` (existing `*_sonic78sonicstate_ho5_official_full` HE
  checkpoints), `heu` (new HE+Unitree pretraining on the corrected Unitree data; Task 3).
- **Fractions:** 25 / 50 / 100 % of the training episodes, nested (25 % ⊂ 50 % ⊂ 100 %), stratified by task family.
  Held out: ~10 % per family (seed 0), never trained on.
- **Budgets:** the handover-bottle recipes, the same sample count for every fraction (so smaller fractions see more
  passes; the curve compares data, not compute): GR00T 32 × 5,000 = 160K, Pi0.5 4 × 8 × 60,000 = 240K, XR-1
  16 × 3 × 7,500 = 120K. HE+Unitree pretraining: the HE official budgets (GR00T 640K, Pi0.5 960K, XR-1 480K).
- **Evaluation (same for every run):** open loop on the held-out episodes (`openloop_smooth.py`, `temp:0`), then the
  50 Hz closed-loop SONIC sim on the held-out episodes in dataset-image mode, scored against the measured joints
  (`wbt_joint28_reference.py`), with the reference inference setting (noise scale 0, `--chunk-blend-s 0.3`,
  `--max-token-step 0.05`, planner start). Robot trials only for the final comparison (Task 8).
- **Claim to test:** for each VLA, the smallest fraction at which `he` (and `heu`) matches `base` at 100 % on closed-loop
  palm error (p50 and p95) with all held-out runs valid.

## Global Constraints

- GPUs: h100 GPUs 0 and 6; h100_174 GPUs 4, 5, 6, 7. **Never h100 GPU 2.** Ask the user before using any other GPU.
- No installs on the H100 hosts or the PC2 system Python; all host writes to root-owned dirs go through containers.
- h100 and h100_174 do not share `/mnt/data01`; copy datasets and checkpoints between them over the LAN (`ssh -A h100`, then `kube@192.168.75.174`).
- Commits end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`; PR bodies end with `🤖 Generated with [Claude Code](https://claude.com/claude-code)`.
- The repo is public: no addresses, user names or credentials in docs.
- Python via `uv run` locally; tests with `.venv/bin/python -m unittest`; lint with `uvx ruff@0.14.1 check` / `format`.
- Steps in configs count micro-batches (the LeRobot scheduler steps per micro-batch).

## Review Focus

- A family with fewer than 5 episodes (e.g. pouring, 2): it gets no held-out episode and stays in every fraction; the split must not crash or hold out its only episodes.
- Nested fractions: every episode in the 25 % list must also be in the 50 % and 100 % lists, and no held-out episode in any of them.
- Re-running Task 1 on the same draft must give identical splits (seed 0), or old configs silently train on different episodes.
- The combined pretraining data must name its camera `observation.images.egocentric` (the fine-tune and HE key); the 09-27 build used `cam_left_high`.
- An HE or HE+Unitree init that is not retargeted keeps the pretraining dataset's normalization (GR00T processor steps, XR-1 model buffers): every `he`/`heu` run must pass through `retarget_init_checkpoint.py`.
- A closed-loop run scored against the commanded joints (`action.wbc`) instead of the measured ones reads ~3× worse; the scorer must use the `wbt_joint28_reference.py` reference.

---

### Task 1: Splits and conversion of the multi-task draft

**Files:**
- Create: `examples/g1_dex3_training/wbt_splits.py`
- Create: `examples/g1_dex3_training/prepare_wbt_multitask_sonic78.py`
- Test: `tests/datasets/test_g1_dex3_wbt_splits.py`

**Interfaces:**
- Produces: `family(prompt: str) -> str` (one of `blocks_cup`, `basket_move`, `pour`, `generic`, `transfer`);
  `make_splits(episodes: dict[int, str], heldout_frac=0.1, fractions=(25, 50, 100), seed=0) -> dict` with keys
  `heldout: list[int]`, `train: {"25": [...], "50": [...], "100": [...]}`, `family: {episode: family}`.
- Produces on disk: dataset `OUT` (sonic78, 50 Hz, episodes renumbered 0..85 in source order) and `OUT.splits.json`
  (the `make_splits` output in new episode numbers, plus `"source_episode": {new: old}`).

- [ ] **Step 1: Write the failing test**

```python
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2] / "examples" / "g1_dex3_training"))
from wbt_splits import family, make_splits


class WbtSplitTests(unittest.TestCase):
    def test_family(self):
        self.assertEqual(family("Place the yellow block into the gray cup."), "blocks_cup")
        self.assertEqual(family("Place the yellow cube into the grey mug."), "blocks_cup")
        self.assertEqual(family("Pick up the white basket and place it on the table."), "basket_move")
        self.assertEqual(family("Place the white basket on the table."), "basket_move")
        self.assertEqual(family("Pour water from the pitcher into the cup."), "pour")
        self.assertEqual(family("Place the manipulated object on the table."), "generic")
        self.assertEqual(family("Place the apple from the plate into the basket."), "transfer")
        self.assertEqual(family("Place the blue and pink objects into the white basket."), "transfer")

    def test_splits_nested_stratified_deterministic(self):
        eps = {i: "Place the yellow block into the gray cup." for i in range(20)}
        eps.update({20 + i: "Place the apple from the plate into the basket." for i in range(10)})
        eps.update({30: "Pour water from the pitcher into the cup.", 31: "Pour water from the pitcher into the blue cup."})
        s = make_splits(eps)
        self.assertEqual(s, make_splits(eps))  # same seed, same split
        held = set(s["heldout"])
        self.assertEqual(len(held & set(range(20))), 2)  # 10 % of 20
        self.assertEqual(len(held & set(range(20, 30))), 1)
        self.assertFalse(held & {30, 31})  # a family under 5 episodes is never held out
        t25, t50, t100 = (set(s["train"][k]) for k in ("25", "50", "100"))
        self.assertTrue(t25 <= t50 <= t100)
        self.assertEqual(t100, set(eps) - held)
        self.assertFalse(held & t100)
        self.assertTrue(t25 & {30, 31})  # small families stay in every fraction (ceil keeps at least one)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m unittest tests/datasets/test_g1_dex3_wbt_splits.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'wbt_splits'`

- [ ] **Step 3: Write minimal implementation** (`examples/g1_dex3_training/wbt_splits.py`)

```python
"""Task families, held-out set and nested training fractions for the G1 WBT multi-task teleop data."""

from __future__ import annotations

import math

import numpy as np

MIN_FAMILY_FOR_HELDOUT = 5


def family(prompt: str) -> str:
    p = prompt.lower()
    if "pour" in p:
        return "pour"
    if "manipulated object" in p or p.startswith(("put the object", "please place the item")):
        return "generic"
    if p.startswith("pick up the white basket") or p.startswith("place the white basket"):
        return "basket_move"
    if "cup" in p or "mug" in p:
        return "blocks_cup"
    return "transfer"


def make_splits(episodes: dict[int, str], heldout_frac=0.1, fractions=(25, 50, 100), seed=0) -> dict:
    """episodes: {episode: prompt}. Per family (sorted, seeded shuffle): hold out round(frac * n) when n >= 5, then
    take the first ceil(f % of the rest) for each fraction, so the fractions are nested."""
    rng = np.random.default_rng(seed)
    fam = {e: family(p) for e, p in episodes.items()}
    heldout, train = [], {str(f): [] for f in fractions}
    for name in sorted(set(fam.values())):
        members = sorted(e for e, f in fam.items() if f == name)
        order = [members[i] for i in rng.permutation(len(members))]
        k = round(heldout_frac * len(order)) if len(order) >= MIN_FAMILY_FOR_HELDOUT else 0
        heldout += order[:k]
        rest = order[k:]
        for f in fractions:
            train[str(f)] += rest[: math.ceil(f / 100 * len(rest))]
    return {"heldout": sorted(heldout), "train": {k: sorted(v) for k, v in train.items()}, "family": fam}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m unittest tests/datasets/test_g1_dex3_wbt_splits.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Write the converter** (`examples/g1_dex3_training/prepare_wbt_multitask_sonic78.py`)

```python
"""Convert the annotated G1 WBT multi-task teleop draft (LeRobot v3, 50 Hz) to sonic78 at 50 Hz with one prompt per
episode, and write OUT.splits.json (held-out episodes and nested 25/50/100 % training lists, new episode numbers).

Same features as prepare_wbt_sonic78.py (state 28 = arms + Dex3 hands by name; action 78 = motion_token 64 + hand
targets 14 from action.wbc; ego_view renamed to observation.images.egocentric). Episodes: those with a task_aug
prompt in meta/lerobot_annotations.json and at least 3 s long. Task = the episode's first task_aug prompt.

Usage: python prepare_wbt_multitask_sonic78.py SRC OUT [REPO_ID]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sonic_targets import ACTION_NAMES
from wbt_splits import make_splits

from lerobot.datasets.lerobot_dataset import LeRobotDataset

MIN_S = 3.0
src, out = Path(sys.argv[1]), Path(sys.argv[2])
repo_id = sys.argv[3] if len(sys.argv) > 3 else f"local/{out.name}"
info = json.loads((src / "meta/info.json").read_text())
assert info["fps"] == 50, info["fps"]
state_names = info["features"]["observation.state"]["names"]
wbc_names = info["features"]["action.wbc"]["names"]
state_idx = [state_names.index(n) for n in ACTION_NAMES]
hand_idx = [wbc_names.index(n) for n in ACTION_NAMES[14:]]
notes = json.loads((src / "meta/lerobot_annotations.json").read_text())["episodes"]
prompts = {}
for e, v in notes.items():
    aug = [a["content"] for a in v["atoms"] if a["style"] == "task_aug"]
    if aug:
        prompts[int(e)] = aug[0]

source = LeRobotDataset("local/wbt_src", root=src, video_backend="torchcodec")
lengths = source.meta.episodes["length"]
keep = [e for e in sorted(prompts) if lengths[e] / 50 >= MIN_S]
h, w, _ = info["features"]["observation.images.ego_view"]["shape"]
ds = LeRobotDataset.create(
    repo_id,
    fps=50,
    root=out,
    robot_type="unitree_g1_dex3_sonic",
    features={
        "observation.state": {"dtype": "float32", "shape": (28,), "names": list(ACTION_NAMES)},
        "action": {"dtype": "float32", "shape": (78,), "names": [f"sonic_token_{i}" for i in range(64)] + list(ACTION_NAMES[14:])},
        "observation.images.egocentric": {"dtype": "video", "shape": (h, w, 3), "names": ["height", "width", "channels"]},
    },
)  # fmt: skip
for new, old in enumerate(keep):
    start, end = source.meta.episodes["dataset_from_index"][old], source.meta.episodes["dataset_to_index"][old]
    for i in range(start, end):
        item = source[i]
        ds.add_frame(
            {
                "observation.state": item["observation.state"].numpy()[state_idx].astype(np.float32),
                "action": np.concatenate([item["action.motion_token"].numpy(), item["action.wbc"].numpy()[hand_idx]]).astype(np.float32),
                "observation.images.egocentric": (item["observation.images.ego_view"].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8),
                "task": prompts[old],
            }
        )  # fmt: skip
    ds.save_episode()
    print(f"episode {new} (source {old}): {end - start} frames", flush=True)
ds.finalize()

splits = make_splits({new: prompts[old] for new, old in enumerate(keep)})
splits["source_episode"] = {new: old for new, old in enumerate(keep)}
Path(f"{out}.splits.json").write_text(json.dumps(splits, indent=1) + "\n")
print(f"done: {len(keep)} episodes -> {out}; held out {splits['heldout']}")
```

- [ ] **Step 6: Run the conversion on the workstation**

```bash
uv run python examples/g1_dex3_training/prepare_wbt_multitask_sonic78.py \
  /mnt/data/jihun/datasets/G1_WBT_GR00T/official_annotations/workspace/drafts/e3dad2ee0371440cb40ff4ff09369a96/episode_000101 \
  /mnt/data/jihun/datasets/G1_WBT_GR00T/wbt_multitask_sonic78_50hz_261009
```
Expected: `done: 86 episodes -> …; held out [...]` (about 8 held out), `wbt_multitask_sonic78_50hz_261009.splits.json`
with `train` lists of about 20 / 39 / 78 episodes. Check one frame: state values equal the source `observation.state`
at the matched names, and the image is the `ego_view` frame.

- [ ] **Step 7: Lint and commit**

```bash
uvx ruff@0.14.1 check examples/g1_dex3_training/wbt_splits.py examples/g1_dex3_training/prepare_wbt_multitask_sonic78.py tests/datasets/test_g1_dex3_wbt_splits.py
uvx ruff@0.14.1 format examples/g1_dex3_training/wbt_splits.py examples/g1_dex3_training/prepare_wbt_multitask_sonic78.py tests/datasets/test_g1_dex3_wbt_splits.py
git add examples/g1_dex3_training/wbt_splits.py examples/g1_dex3_training/prepare_wbt_multitask_sonic78.py tests/datasets/test_g1_dex3_wbt_splits.py
git commit -m "feat(g1-dex3): multi-task WBT teleop to sonic78 50 Hz with per-episode prompts, stratified held-out and nested fractions"
```

### Task 2: Combined HE+Unitree dataset (corrected Unitree data)

The 2026-09-27 combined build (`/mnt/data01/jhkim/model_weight/g1_dex3_20260922/combined/build_combined.py` on h100,
not in the repo) predates the in-place Unitree correction of 2026-10-01 (`datasets/sonic78_nolimit_sonicstate`,
stats rewritten 10-01; the Unitree `*_official_full` runs used it) and named the single camera `cam_left_high`. The
fine-tune data and the HE checkpoints use `observation.images.egocentric`, so the rebuild renames Unitree's
`cam_left_high` to `egocentric` instead.

**Files:**
- Create: `examples/g1_dex3_training/build_combined_sonic78.py` (port of the h100 script)
- Test: `tests/datasets/test_g1_dex3_build_combined.py`

**Interfaces:**
- Produces: `merge_heldout(ut_split: dict, he_split: dict, n_ut: int) -> dict` (the union split, same JSON format as
  `heldout_sonic78_nolimit_5pct.json`: `episodes`, `always_excluded`, `by_task`, plus `he_episode_offset`); on disk
  `/run-output/datasets/combined_sonic78_1cam_261009` and `/run-output/configs/heldout_combined_sonic78_261009_5pct.json`.

- [ ] **Step 1: Write the failing test**

```python
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2] / "examples" / "g1_dex3_training"))
from build_combined_sonic78 import merge_heldout


class CombinedTests(unittest.TestCase):
    def test_merge_heldout_offsets_he(self):
        ut = {"episodes": [3, 1], "always_excluded": [7], "by_task": {"pick": [1, 3]}}
        he = {"episodes": [0, 5], "by_task": {"laptop": [0, 5]}}
        m = merge_heldout(ut, he, n_ut=10)
        self.assertEqual(m["episodes"], [1, 3, 10, 15])
        self.assertEqual(m["always_excluded"], [7])
        self.assertEqual(m["by_task"], {"unitree: pick": [1, 3], "he: laptop": [10, 15]})
        self.assertEqual(m["he_episode_offset"], 10)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m unittest tests/datasets/test_g1_dex3_build_combined.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'build_combined_sonic78'`

- [ ] **Step 3: Port the script.** Copy the h100 `build_combined.py` to `examples/g1_dex3_training/build_combined_sonic78.py`
  and make exactly these changes:
  - Move everything after the function definitions under `def main() -> None:` and end with
    `if __name__ == "__main__": main()`; replace `sys.path.insert(0, "/workspace/lerobot/examples/g1_dex3_training")`
    by a plain `from augment_joint_quantiles import augment` inside `main()` (the script runs from that directory).
  - Names: `OUT = RUN / "datasets/combined_sonic78_1cam_261009"`, `REPO = "local/g1_dex3_combined_sonic78_1cam_261009"`,
    `VIEWS = RUN / "datasets/_views_combined_261009"`, split file `RUN / "configs/heldout_combined_sonic78_261009_5pct.json"`.
  - Camera: `ut_eps = make_view(UT, VIEWS / "ut", {CAM: EGO}, {RIGHT})` and `he_eps = make_view(HE, VIEWS / "he", {}, set())`;
    provenance `"camera": f"{EGO} (Unitree left head cam renamed; HE egocentric)"`; docstring accordingly.
  - Replace the inline held-out block by a call to this function (defined at module level):

```python
def merge_heldout(ut_split: dict, he_split: dict, n_ut: int) -> dict:
    """Union of the per-dataset ho5 splits; HE episodes follow the Unitree ones in the combined dataset."""
    by_task = {f"unitree: {t}": eps for t, eps in ut_split["by_task"].items()}
    by_task |= {f"he: {t}": [e + n_ut for e in eps] for t, eps in he_split["by_task"].items()}
    episodes = sorted(ut_split["episodes"] + [e + n_ut for e in he_split["episodes"]])
    always = sorted(ut_split.get("always_excluded", []) + [e + n_ut for e in he_split.get("always_excluded", [])])
    return {"he_episode_offset": n_ut, "always_excluded": always, "num_heldout": len(episodes), "episodes": episodes, "by_task": by_task}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m unittest tests/datasets/test_g1_dex3_build_combined.py -v`
Expected: PASS

- [ ] **Step 5: Build on h100** in a training container (the script's paths are container paths under `/run-output`;
  `/run-output` = `/mnt/data01/jhkim/model_weight/g1_dex3_20260922`), cwd `examples/g1_dex3_training`:
  `python build_combined_sonic78.py`. Expected: 7216 episodes (3152 Unitree + 4064 HE), one camera
  `observation.images.egocentric`, the mapping assertion passes, held-out = 158 + 226 = 384 episodes.

- [ ] **Step 6: Lint and commit** (ruff on the script and its test; message
  `feat(g1-dex3): combined Unitree + HE sonic78 builder (corrected Unitree data, egocentric camera)`)

### Task 3: HE+Unitree pretraining (GR00T, Pi0.5, XR-1)

**Files:** host configs only (not in the repo):
`/run-output/combined_261009/configs/{groot,pi05,xr1}_combined_sonic78_ho5_official_{smoke,full}.json`.

**Interfaces:**
- Consumes: Task 2's dataset and `heldout_combined_sonic78_261009_5pct.json` (its `episodes` list is the configs' `dataset.exclude_episodes`).
- Produces: `runs/{groot,pi05,xr1}_combined_sonic78_ho5_official_full/checkpoints/last/pretrained_model` (Pi0.5 also
  `pretrained_model_ema`), used as the `heu` init in Task 4.

- [ ] **Step 1:** Copy each HE official config (`humanoid_everyday_g1_20260923/configs/{groot,pi05,xr1}_sonic78sonicstate_ho5_official_full.json`) and change only `dataset.repo_id`/`root` → the combined dataset, `dataset.exclude_episodes` → the `episodes` list of `heldout_combined_sonic78_261009_5pct.json`, `output_dir`, `job_name`. XR-1: rebuild its stats with `xr1_action_stats.py --relative sonic78` on the combined dataset excluding the held-out episodes. Smoke configs: 20 steps (Pi0.5 40).
- [ ] **Step 2:** Append `combined_261009/{model}_combined_sonic78_ho5_official_smoke` then `_full` to a queue file per GPU and start `official_queue.sh` in a training container (one model per GPU: h100 GPU 0 Pi0.5 (longest), GPU 6 GR00T then XR-1).
- [ ] **Step 3:** Check each `logs/*.exit` is 0 and the final loss is in the range of the HE run of the same model; copy the three `last` checkpoints to h100_174 if the fine-tunes run there.

### Task 4: Fine-tune grid (27 runs)

**Files:**
- Create: `examples/g1_dex3_training/make_finetune_grid.py`
- Test: `tests/datasets/test_g1_dex3_finetune_grid.py`

**Interfaces:**
- Consumes: `OUT.splits.json` (Task 1), the bottle-study configs `{model}_wbt50_{base,he}_full.json` as templates.
- Produces: `grid(templates: dict, splits: dict, dataset_root: str, repo_id: str, job_root: str) -> dict[str, dict]`
  mapping run name `{model}_dex{pct}_{init}_full` to its config; config files and one queue file per GPU.

- [ ] **Step 1: Write the failing test**

```python
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2] / "examples" / "g1_dex3_training"))
from make_finetune_grid import grid

TEMPLATE = {
    "dataset": {"repo_id": "old", "root": "/old", "episodes": None, "exclude_episodes": [27, 35]},
    "policy": {"type": "groot", "pretrained_path": "/old/init/groot_he"},
    "output_dir": "/old/runs/x", "job_name": "x", "steps": 5000, "batch_size": 32,
}  # fmt: skip
SPLITS = {"heldout": [9], "train": {"25": [1], "50": [1, 2], "100": [1, 2, 3]}}


class GridTests(unittest.TestCase):
    def test_grid(self):
        templates = {("groot", "base"): TEMPLATE, ("groot", "he"): TEMPLATE}
        g = grid(templates, SPLITS, "/new", "local/new", "/jobs")
        self.assertEqual(len(g), 9)  # 1 model x 3 inits x 3 fractions
        c = g["groot_dex25_heu_full"]
        self.assertEqual(c["dataset"]["episodes"], [1])
        self.assertIsNone(c["dataset"]["exclude_episodes"])
        self.assertEqual(c["policy"]["pretrained_path"], "/jobs/init/groot_heu")
        self.assertEqual(c["output_dir"], "/jobs/runs/groot_dex25_heu_full")
        self.assertEqual(g["groot_dex100_base_full"]["steps"], 5000)  # same budget at every fraction
        self.assertEqual(TEMPLATE["dataset"]["root"], "/old")  # templates are not modified


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m unittest tests/datasets/test_g1_dex3_finetune_grid.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'make_finetune_grid'`

- [ ] **Step 3: Write the implementation** (`examples/g1_dex3_training/make_finetune_grid.py`)

```python
"""Fine-tune configs for the data-efficiency grid: {groot,pi05,xr1} x {base,he,heu} x {25,50,100} % of the WBT
multi-task training episodes, from the handover-bottle configs (same sample budget at every fraction).

`base` uses the model's base template; `he` and `heu` use its he template with pretrained_path JOB_ROOT/init/<model>_<init>
(retargeted with retarget_init_checkpoint.py before training). XR-1 runs need per-fraction stats: any template string
equal to XR1_STATS is replaced by JOB_ROOT/stats/xr1_dex<pct>.json.

Usage: python make_finetune_grid.py TEMPLATE_DIR SPLITS_JSON DATASET_ROOT REPO_ID JOB_ROOT OUT_DIR GPU [GPU ...]
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

MODELS, INITS, FRACTIONS = ("groot", "pi05", "xr1"), ("base", "he", "heu"), ("25", "50", "100")
XR1_STATS = "/run-output/g1_wbt_handover_20261002/stats/xr1_sonic78.json"


def _replace(obj, old, new):
    if isinstance(obj, dict):
        return {k: _replace(v, old, new) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_replace(v, old, new) for v in obj]
    return new if obj == old else obj


def grid(templates: dict, splits: dict, dataset_root: str, repo_id: str, job_root: str) -> dict[str, dict]:
    out = {}
    for (model, kind), template in templates.items():
        for init in ("base",) if kind == "base" else ("he", "heu"):
            for pct in FRACTIONS:
                name = f"{model}_dex{pct}_{init}_full"
                c = copy.deepcopy(template)
                c["dataset"].update(repo_id=repo_id, root=dataset_root, episodes=splits["train"][pct], exclude_episodes=None)
                if init != "base":
                    c["policy"]["pretrained_path"] = f"{job_root}/init/{model}_{init}"
                c.update(output_dir=f"{job_root}/runs/{name}", job_name=name)
                if model == "xr1":
                    c = _replace(c, XR1_STATS, f"{job_root}/stats/xr1_dex{pct}.json")
                out[name] = c
    return out


def main() -> None:
    tdir, splits_path, root, repo_id, job_root, out_dir, *gpus = sys.argv[1:]
    templates = {(m, k): json.loads((Path(tdir) / f"{m}_wbt50_{k}_full.json").read_text()) for m in MODELS for k in ("base", "he")}
    configs = grid(templates, json.loads(Path(splits_path).read_text()), root, repo_id, job_root)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    for name, c in configs.items():
        (Path(out_dir) / f"{name}.json").write_text(json.dumps(c, indent=1) + "\n")
    names = sorted(configs, key=lambda n: (n.startswith("pi05"), n))  # long Pi0.5 runs last in each queue
    for i, gpu in enumerate(gpus):
        (Path(out_dir) / f"queue_gpu{gpu}.txt").write_text("".join(f"{Path(job_root).name}/{n}\n" for n in names[i :: len(gpus)]))
    print(f"{len(configs)} configs, queues for GPUs {gpus}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m unittest tests/datasets/test_g1_dex3_finetune_grid.py -v`
Expected: PASS

- [ ] **Step 5: Lint and commit** (ruff on the two files; message
`feat(g1-dex3): fine-tune config grid for the pretrain-to-fine-tune data-efficiency study`)

### Task 5: Prepare the job root and run the 27 fine-tunes

**Files:** host only: job root `/run-output/g1_wbt_dex_261009/` on h100_174 (`datasets/`, `configs/`, `init/`, `stats/`, `runs/`, `logs/`).

**Interfaces:**
- Consumes: Task 1 dataset + splits, Task 3 `heu` checkpoints, the HE checkpoints (`he`), Task 4 configs.
- Produces: `runs/{model}_dex{pct}_{init}_full/checkpoints/last/pretrained_model` for all 27 runs.

- [ ] **Step 1:** Copy the Task 1 dataset and `splits.json` to `datasets/` on h100_174 (and h100 if queues run there).
- [ ] **Step 2:** `init/{groot,pi05,xr1}_he` = the HE official checkpoints (Pi0.5: EMA); `init/{model}_heu` = Task 3's. Run `retarget_init_checkpoint.py` on every `init/*` against the new dataset; check GR00T/Pi0.5 processor stats equal the dataset's and XR-1's buffers were reloaded.
- [ ] **Step 3:** XR-1 stats per fraction: `xr1_action_stats.py --relative sonic78` on the new dataset restricted to each `train[pct]` list → `stats/xr1_dex{25,50,100}.json`.
- [ ] **Step 4:** `make_finetune_grid.py <bottle configs dir> datasets/wbt_multitask_sonic78_50hz_261009.splits.json /run-output/g1_wbt_dex_261009/datasets/wbt_multitask_sonic78_50hz_261009 local/wbt_multitask_sonic78_50hz /run-output/g1_wbt_dex_261009 /run-output/g1_wbt_dex_261009/configs 4 5 6 7`; add a 20-step smoke config for one run per model and put it first in its queue.
- [ ] **Step 5:** Start one training container per GPU (bottle-study settings: `--memory 200g --shm-size 16g`, `LEROBOT_VIDEO_DECODER_CACHE_SIZE=5000`, `HF_HUB_OFFLINE=1`, code `/mnt/data01/jhkim/code/lerobot-g1-psi0-xr1-dev`) running `official_queue.sh queue_gpu<N>.txt`. Expected time: GR00T ~1 h, XR-1 ~2.5 h, Pi0.5 ~7 h per run; ~95 GPU-hours for 27 runs.
- [ ] **Step 6:** Check all 27 `.exit` files are 0; regenerate the model registry (`model_registry.py`).

### Task 6: Open-loop and closed-loop evaluation

**Files:**
- Create: `examples/g1_dex3_training/data_efficiency_table.py`
- Test: `tests/datasets/test_g1_dex3_data_efficiency_table.py`

**Interfaces:**
- Consumes: closed-loop run dirs `WBT_{model}_dex{pct}_{init}_ep{E}_r{R}/stream_eval.json` (scored by `sonic_stream_eval.py` against the `wbt_joint28_reference.py` reference).
- Produces: `table(root: Path) -> dict` — per model, per init, per fraction: `valid`, `runs`, palm p50 / p95 (mean over valid runs); printed as Markdown.

- [ ] **Step 1: Write the failing test**

```python
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2] / "examples" / "g1_dex3_training"))
from data_efficiency_table import table


def run(root, name, valid, p50=None, p95=None):
    d = Path(root) / name
    d.mkdir()
    body = {"valid": valid}
    if valid:
        body["palm_err_vs_original_cm"] = {"p50": p50, "p95": p95}
    (d / "stream_eval.json").write_text(json.dumps(body))


class TableTests(unittest.TestCase):
    def test_table(self):
        with tempfile.TemporaryDirectory() as r:
            run(r, "WBT_groot_dex25_he_ep3_r1", True, 2.0, 5.0)
            run(r, "WBT_groot_dex25_he_ep4_r1", True, 4.0, 7.0)
            run(r, "WBT_groot_dex25_he_ep5_r1", False)
            (Path(r) / "WBT_groot_dex25_he_ep6_r1").mkdir()  # crashed run: no stream_eval.json
            t = table(Path(r))
            cell = t["groot"]["he"]["25"]
            self.assertEqual((cell["valid"], cell["runs"]), (2, 4))
            self.assertEqual((cell["p50"], cell["p95"]), (3.0, 6.0))


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m unittest tests/datasets/test_g1_dex3_data_efficiency_table.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'data_efficiency_table'`

- [ ] **Step 3: Write the implementation** (`examples/g1_dex3_training/data_efficiency_table.py`)

```python
"""Data-efficiency table from closed-loop sim runs WBT_<model>_dex<pct>_<init>_ep<E>_r<R>/stream_eval.json.

The scorer's "original" reference here is the wbt_joint28_reference.py dataset (the measured joints), so
palm_err_vs_original_cm is the palm error against the pose the real robot reached. A run dir without a readable
stream_eval.json counts as an invalid run.
Usage: python data_efficiency_table.py ROOT
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

NAME = re.compile(r"WBT_(?P<model>[a-z0-9]+)_dex(?P<pct>\d+)_(?P<init>[a-z]+)_ep\d+_r\d+$")


def table(root: Path) -> dict:
    cells = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for d in sorted(root.glob("WBT_*")):
        m = NAME.match(d.name)
        if not (m and d.is_dir()):
            continue
        try:
            res = json.loads((d / "stream_eval.json").read_text())
        except (OSError, json.JSONDecodeError):
            res = {"valid": False}
        cells[m["model"]][m["init"]][m["pct"]].append(res)
    out = {}
    for model, inits in cells.items():
        for init, pcts in inits.items():
            for pct, runs in pcts.items():
                ok = [r for r in runs if r.get("valid")]
                err = [r["palm_err_vs_original_cm"] for r in ok]
                out.setdefault(model, {}).setdefault(init, {})[pct] = {
                    "valid": len(ok), "runs": len(runs),
                    "p50": round(float(np.mean([e["p50"] for e in err])), 2) if err else None,
                    "p95": round(float(np.mean([e["p95"] for e in err])), 2) if err else None,
                }  # fmt: skip
    return out


def main() -> None:
    t = table(Path(sys.argv[1]))
    for model, inits in sorted(t.items()):
        print(f"\n### {model}\n\n| init | 25 % | 50 % | 100 % |\n| --- | --- | --- | --- |")
        for init in ("base", "he", "heu"):
            row = [inits.get(init, {}).get(p) for p in ("25", "50", "100")]
            print(f"| {init} | " + " | ".join(f"{c['p50']} / {c['p95']} ({c['valid']}/{c['runs']})" if c else "-" for c in row) + " |")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m unittest tests/datasets/test_g1_dex3_data_efficiency_table.py -v`
Expected: PASS

- [ ] **Step 5: Open loop.** For each of the 27 runs: `openloop_smooth.py <run>/checkpoints/last/pretrained_model <dataset root> temp:0 <held-out episodes>` in the training container (replan 20 frames = 0.4 s at 50 Hz, as in the bottle study); collect tokens / hands err and seam per run into `eval/openloop_dex.md`.
- [ ] **Step 6: Closed loop.** Build the scorer reference once: `wbt_joint28_reference.py <dataset root> <dataset root>_joint28ref`. For each run and each held-out episode (1 repeat; 3 repeats for the 25 % and 100 % points of `base` and `he` if their difference is within the repeat spread), launch `sonic_official_sim_eval.sh` in dataset-image mode: `SIM_HOST=h100_174 HF_CACHE=/mnt/data01/jhkim/huggingface POLICY_FPS=50 SERVER_EXTRA=--noise-scale=0 JOINT28_DIR=<reference> SONIC_DIR=<dataset root>`, run name `WBT_{model}_dex{pct}_{init}_ep{E}_r{R}`, streamer args `--start planner --max-token-step 0.05 --chunk-blend-s 0.3`, task = the episode's prompt (the dataset's task). Queue the runs per allowed GPU with the `run_batch` pattern used for the WBC rounds.
- [ ] **Step 7:** `data_efficiency_table.py <audit root>` → the per-model tables. Commit the tool (ruff first): `feat(g1-dex3): data-efficiency table for the pretrain-to-fine-tune grid`.

### Task 7: Results write-up

**Files:**
- Create: `docs/research/2026-10-09-g1-pretrain-finetune-data-efficiency.md`
- Modify: `AGENTS.md` (one pointer sentence in the G1 Dex3 note)

- [ ] **Step 1:** Write the doc: dataset and split (families, held-out list), recipes, the open-loop table, the closed-loop data-efficiency tables per model, and the answer to the claim ("`he` at X % ≈ `base` at 100 %" per model, or not), with run counts and repeat spread.
- [ ] **Step 2:** Add the pointer to `AGENTS.md`; commit `docs(g1-dex3): pretrain-to-fine-tune data-efficiency results`; open a PR to main.

### Task 8: Robot comparison (needs the user and the G1)

- [ ] **Step 1:** Pick per model the `base` 100 % run and the smallest-fraction `he` (or `heu`) run that matched it in sim; copy both to the workstation (`/mnt/data/jihun/g1_models/`).
- [ ] **Step 2:** For two task families with held-out episodes (e.g. blocks into cup, apple between plate and basket), run N = 5 trials per model with the reference setting (`g1_groot_real_run.sh`, `NOISE_SCALE=0`, `BLEND_S=0.3`, `POLICY_FPS=50`, the episode's prompt), record success / partial / fail per trial, and add the table to the results doc.
