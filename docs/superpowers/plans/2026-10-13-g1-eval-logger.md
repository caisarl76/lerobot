# G1 Real-Robot Evaluation Logger Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run 5–10 operator-judged VLA trials per session on the G1, record each trial (state, commands, head camera, verdict), convert the session to a LeRobot v3.0 dataset, and label/review the failure step with a VLM in the dataset visualizer.

**Architecture:** The PC2 streamer (`sonic_policy_streamer.py --trials N`) loops over trials in one process and writes raw trial files with a PC2-safe helper module (`eval_session.py`, stdlib + numpy). A workstation converter (`eval_session_to_lerobot.py`) builds the LeRobot dataset plus `meta/eval_session.json`. In the visualizer repo, a backend module (`eval_labels.py`) asks the configured VLM for subtask starts and the failure step, stores subtasks as existing `subtask` atoms and the failure label in `meta/eval_labels.json`; a frontend Evaluation panel shows and edits it, and the existing Mark reviewed flow covers it.

**Tech Stack:** Python 3.12 (lerobot, numpy, pyzmq, msgpack, OpenCV), LeRobot v3.0 datasets; visualizer: FastAPI backend (Python 3.13 venv, LeRobot annotation pipeline at its pinned revision), Next.js + TypeScript frontend, bun.

**Spec:** `docs/superpowers/specs/2026-10-13-g1-eval-logger-design.md`

## Global Constraints

- Nothing new installed on PC2: code that runs there (`sonic_policy_streamer.py`, `stream_backends.py`, `eval_session.py`) imports only the standard library, numpy, and what the streamer already imports (pyzmq, msgpack, cv2).
- Without `--trials`, the streamer behaves exactly as today; the existing tests `tests/datasets/test_g1_dex3_streamer_cli.py` and `tests/datasets/test_g1_dex3_stream_backends.py` must keep passing unchanged.
- `--trials` requires `--images zmq`, `--backend sonic`, `--end planner`, `--handoff-blend-s > 0`, and `--session-dir`; it excludes `--log`.
- Default failure reasons, in this order: `did not reach the object`, `grasp failed / missed`, `dropped the object`, `wrong object / target`, `collision (table, knocked over)`, `stalled / froze`, `safety stop (watchdog)`, `other`.
- The dataset is LeRobot **v3.0**, written with `LeRobotDataset.create` / `add_frame` / `save_episode` / `finalize`; frames at `--fps` (default 30); episode = policy-controlled phase (`phase == "episode"` ticks) only.
- The operator's verdict is authoritative: nothing in the VLM path changes `meta/eval_session.json`.
- VLM: 2 fps sampling, at most 64 frames per episode, the `/annotate` VLM endpoint config (`engine.parse_config(engine.default_config())`); no new model service.
- lerobot worktree: `/home/jihun/work/lerobot-g1-eval-logger` (branch `feat/g1-eval-logger`). Run its tests with `/home/jihun/work/lerobot/.venv/bin/python -m pytest …` from the worktree root (the main checkout's venv has lerobot, cv2, zmq, msgpack; test files put `examples/g1_dex3_training` on `sys.path` themselves). Lint: `pre-commit run --files <changed files>`.
- Visualizer worktree: `/home/jihun/work/lerobot-dataset-visualizer-eval` (branch `feat/eval-labels` from the visualizer's `main` HEAD; the main checkout has uncommitted changes that are not ours — do not touch them). Backend tests: `/home/jihun/work/lerobot-dataset-visualizer/backend/.venv/bin/python -m pytest -q backend/tests/<file>` from the worktree root. Frontend: `bun install` once in the worktree, then `bun test <file>` and finally `bun run format && bun run validate`. Use bun only, never npm/yarn.
- Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`. lerobot commits use the `feat(g1-dex3): …` / `docs(g1-dex3): …` style.

## Review Focus

1. An Enter pressed during the takeover (or a stale `stop.i` file in gate mode) must not end the next trial on its first tick — `start_trial_input()` clears both; tested in Task 1.
2. Restarting the streamer after a crash with the same `--session-dir` must continue at the next trial number, not overwrite `trial_000` — `next_trial()`; tested in Task 1 and exercised in Task 4.
3. A trial that never reached the episode phase (first chunk failed, stopped during the blend-in) is still written (NaN wall times) and the converter skips it with a warning, listing it under `skipped` — tested in Task 6.
4. A bad VLM reply (not an object, unknown or out-of-order subtask, frame out of range, failure fields missing or present for a success, or an exception from the client) fails only that episode, keeps the raw reply, and leaves other episodes labeled — tested in Task 7.
5. Editing the failure label after Mark reviewed makes the episode unreviewed again, while datasets without eval files keep their existing reviews valid — tested in Task 8.

---

## File Structure

lerobot worktree (`/home/jihun/work/lerobot-g1-eval-logger`):

| File                                                          | Responsibility                                                                                                                    |
| ------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------- |
| Create `examples/g1_dex3_training/eval_session.py`            | PC2-safe session helpers: failure reasons, verdict prompt, gate-dir verdict, operator stop, atomic trial writing, resume, summary |
| Modify `examples/g1_dex3_training/stream_backends.py`         | `gate()` prompt prefix + consume; `SonicBackend.start()` split into `connect()` + `take_over()`                                   |
| Modify `examples/g1_dex3_training/sonic_policy_streamer.py`   | `LiveImages` JPEG recording; `--trials/--session-dir/--failure-reasons`; per-trial loop                                           |
| Modify `examples/g1_dex3_training/g1_groot_real_run.sh`       | `session` and `pull` modes                                                                                                        |
| Create `examples/g1_dex3_training/eval_session_to_lerobot.py` | Session → LeRobot v3.0 dataset + `meta/eval_session.json`                                                                         |
| Create `tests/datasets/test_g1_dex3_eval_session.py`          | Unit tests for `eval_session.py`                                                                                                  |
| Modify `tests/datasets/test_g1_dex3_stream_backends.py`       | Two-trial `connect()`/`take_over()` test                                                                                          |
| Create `tests/datasets/test_g1_dex3_eval_trials.py`           | `LiveImages` recording test + fake-rig two-trial streamer run                                                                     |
| Create `tests/datasets/test_g1_dex3_eval_convert.py`          | Converter test on a synthetic session                                                                                             |
| Create `docs/research/2026-10-13-g1-eval-logger.md`           | Operator workflow (steps 1–10)                                                                                                    |
| Modify `CLAUDE.md`                                            | One-line pointer to the workflow doc                                                                                              |

Visualizer worktree (`/home/jihun/work/lerobot-dataset-visualizer-eval`):

| File                                                        | Responsibility                                                                              |
| ----------------------------------------------------------- | ------------------------------------------------------------------------------------------- |
| Create `backend/eval_labels.py`                             | Session/label files, frame sampling, prompt, reply validation, per-episode labeling         |
| Modify `backend/annotation_history.py`                      | `eval_label_hash()`; review validity includes it                                            |
| Modify `backend/app.py`                                     | `GET /api/eval/episodes/{ep}`, `POST /api/eval/label`, `POST /api/eval/episodes/{ep}/label` |
| Create `backend/tests/test_eval_labels.py`                  | Module + API tests                                                                          |
| Modify `src/utils/annotationsClient.ts`                     | Eval client functions and types                                                             |
| Create `src/components/evaluation-panel.tsx`                | Evaluation panel                                                                            |
| Modify `src/components/annotations-panel.tsx`               | Mount the panel                                                                             |
| Modify `src/components/annotation-review-control.tsx`       | Refetch review on `eval-label-changed`                                                      |
| Modify `src/components/annotations-timeline.tsx`            | Failure marker                                                                              |
| Create `src/components/__tests__/evaluation-panel.test.tsx` | Panel test                                                                                  |

---

### Task 1: PC2-safe session helpers (`eval_session.py`)

**Files:**

- Create: `examples/g1_dex3_training/eval_session.py`
- Test: `tests/datasets/test_g1_dex3_eval_session.py`

**Interfaces:**

- Produces:
  - `DEFAULT_FAILURE_REASONS: tuple[str, ...]`, `SAFETY_STOP = "safety stop (watchdog)"`
  - `parse_reasons(text: str | None) -> list[str]`
  - `ask_verdict(reasons: list[str], episode_end: str, ask=input) -> dict | None` — `{"verdict": "success"|"fail", "reason": str, "note": str}`; `None` = redo
  - `gate_verdict(gate_dir: Path, trial: int, reasons: list[str]) -> dict | None` — reads and deletes `verdict.{trial}` (JSON `{"verdict": "success"|"fail"|"redo", "reason": <1-based int>, "note": str}`)
  - `start_trial_input(gate_dir: Path | None, trial: int) -> None` — deletes a stale `stop.{trial}`, or flushes pending terminal input
  - `stop_requested(gate_dir: Path | None, trial: int) -> bool` — consumes `stop.{trial}`, or a typed line (Enter)
  - `next_trial(session_dir: Path) -> int` — number of lines in `results.jsonl` (0 if missing)
  - `write_trial(session_dir: Path, trial: int, log: dict[str, list], extra: dict[str, str], frames: list[tuple[float, str, bytes]]) -> dict` — writes `trial_{trial:03d}.npz`, `trial_{trial:03d}_frames.npz`, appends and returns the `results.jsonl` line
  - `load_frames(path: Path) -> tuple[np.ndarray, np.ndarray, list[bytes]]` — `(t_recv, names, jpegs)`
  - `summarize(session_dir: Path) -> str`

- [ ] **Step 1: Write the failing tests**

```python
"""eval_session.py: the operator verdict per trial and the session files of sonic_policy_streamer.py --trials."""

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / "examples" / "g1_dex3_training"))
import eval_session as es


def answers(*replies):
    it = iter(replies)
    return lambda prompt="": next(it)


def test_parse_reasons():
    assert es.parse_reasons(None) == list(es.DEFAULT_FAILURE_REASONS)
    assert es.parse_reasons(" a , b,,c ") == ["a", "b", "c"]
    with pytest.raises(ValueError):
        es.parse_reasons(" , ")


def test_ask_verdict_success_redo_and_fail():
    reasons = ["x", "y", es.SAFETY_STOP]
    assert es.ask_verdict(reasons, "completed", answers("S")) == {"verdict": "success", "reason": "", "note": ""}
    assert es.ask_verdict(reasons, "completed", answers("r")) is None
    # invalid inputs are asked again: "q" verdict, "0" / "9" / "" reason (no default without a watchdog stop)
    got = es.ask_verdict(reasons, "operator stop", answers("q", "f", "0", "9", "", "2", " slipped "))
    assert got == {"verdict": "fail", "reason": "y", "note": "slipped"}


def test_ask_verdict_watchdog_preselects_safety_stop():
    reasons = ["x", "y", es.SAFETY_STOP]
    got = es.ask_verdict(reasons, "watchdog: arm joint 3 measured at 7.0 rad/s > 6", answers("f", "", ""))
    assert got == {"verdict": "fail", "reason": es.SAFETY_STOP, "note": ""}


def test_gate_verdict_consumes_the_file(tmp_path):
    (tmp_path / "verdict.1").write_text(json.dumps({"verdict": "fail", "reason": 2, "note": "n"}))
    assert es.gate_verdict(tmp_path, 1, ["a", "b"]) == {"verdict": "fail", "reason": "b", "note": "n"}
    assert not (tmp_path / "verdict.1").exists()
    (tmp_path / "verdict.2").write_text(json.dumps({"verdict": "redo"}))
    assert es.gate_verdict(tmp_path, 2, ["a"]) is None


def test_stale_stop_file_is_cleared_and_stop_is_consumed(tmp_path):
    (tmp_path / "stop.0").touch()
    es.start_trial_input(tmp_path, 0)  # left over from before the trial: must not end it
    assert not es.stop_requested(tmp_path, 0)
    (tmp_path / "stop.0").touch()
    assert es.stop_requested(tmp_path, 0)
    assert not (tmp_path / "stop.0").exists()


def fake_log(n=20, episode=slice(5, 15), t0=1000.0):
    phase = ["planner token"] * n
    phase[episode] = ["episode"] * (episode.stop - episode.start)
    return {
        "wall": list(t0 + 0.02 * np.arange(n)),
        "phase": phase,
        "state": [np.full(28, i, np.float32) for i in range(n)],
    }


def test_write_trial_files_resume_and_summary(tmp_path):
    assert es.next_trial(tmp_path / "s") == 0
    frames = [(1000.1, "egocentric", b"\xff\xd8a"), (1000.2, "egocentric", b"\xff\xd8bb")]
    extra = {"episode_end": "operator stop", "args": "{}", "verdict": "fail", "reason": "y", "note": ""}
    line = es.write_trial(tmp_path / "s", 0, fake_log(), extra, frames)
    assert line["trial"] == 0 and line["verdict"] == "fail" and line["frames"] == 2
    assert line["wall_start"] == pytest.approx(1000.1) and line["wall_stop"] == pytest.approx(1000.28)
    npz = np.load(tmp_path / "s/trial_000.npz")
    assert str(npz["verdict"]) == "fail" and npz["state"].shape == (20, 28)
    t, names, jpegs = es.load_frames(tmp_path / "s/trial_000_frames.npz")
    assert list(t) == [1000.1, 1000.2] and list(names) == ["egocentric"] * 2 and jpegs == [b"\xff\xd8a", b"\xff\xd8bb"]
    no_episode = fake_log(episode=slice(0, 0))
    line = es.write_trial(tmp_path / "s", 1, no_episode, {**extra, "verdict": "success", "reason": ""}, [])
    assert math.isnan(line["wall_start"])
    assert es.next_trial(tmp_path / "s") == 2  # a restarted session continues at trial 2
    assert not list((tmp_path / "s").glob("*.tmp*"))
    assert es.summarize(tmp_path / "s") == "success 1/2; y: 1"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `/home/jihun/work/lerobot/.venv/bin/python -m pytest tests/datasets/test_g1_dex3_eval_session.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'eval_session'`

- [ ] **Step 3: Write the implementation**

`examples/g1_dex3_training/eval_session.py`:

```python
"""Evaluation sessions of sonic_policy_streamer.py --trials: the operator's verdict per trial and the trial files.

Runs on PC2: the standard library and numpy only. A session directory holds
  trial_{i:03d}.npz         the streamer's 50 Hz log + episode_end, args, verdict, reason, note, wall_start, wall_stop
                            (wall time of the first / last "episode" tick; NaN if the trial never reached it)
  trial_{i:03d}_frames.npz  head-camera JPEGs as received (RGB data, as g1_head_camera_server.py encodes it):
                            jpeg (all bytes concatenated, uint8), offsets (n + 1), t_recv (time.time()), names
  results.jsonl             one line per kept trial; its line count is the next trial number (restart = resume)
Operator input: a terminal (Enter stops a trial, then s / f / r), or with --gate-dir the files settled.i, stop.i
and verdict.i (JSON {"verdict": "success" | "fail" | "redo", "reason": <1-based number>, "note": "..."}), each
deleted once read.
"""

from __future__ import annotations

import json
import math
import os
import select
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

DEFAULT_FAILURE_REASONS = (
    "did not reach the object",
    "grasp failed / missed",
    "dropped the object",
    "wrong object / target",
    "collision (table, knocked over)",
    "stalled / froze",
    "safety stop (watchdog)",
    "other",
)
SAFETY_STOP = "safety stop (watchdog)"


def parse_reasons(text: str | None) -> list[str]:
    if text is None:
        return list(DEFAULT_FAILURE_REASONS)
    reasons = [r.strip() for r in text.split(",") if r.strip()]
    if not reasons:
        raise ValueError("--failure-reasons is empty")
    return reasons


def ask_verdict(reasons: list[str], episode_end: str, ask=input) -> dict | None:
    """The operator's verdict after a trial; None = redo (discard the trial and run it again)."""
    while (v := ask("[eval] [s]uccess / [f]ail / [r]edo: ").strip().lower()) not in ("s", "f", "r"):
        pass
    if v == "r":
        return None
    if v == "s":
        return {"verdict": "success", "reason": "", "note": ""}
    default = reasons.index(SAFETY_STOP) + 1 if episode_end.startswith("watchdog") and SAFETY_STOP in reasons else 0
    menu = "\n".join(f"  {i}. {r}" for i, r in enumerate(reasons, 1))
    hint = f" (Enter = {default})" if default else ""
    while True:
        raw = ask(f"[eval] failure reason:\n{menu}\nnumber{hint}: ").strip()
        if not raw and default:
            n = default
            break
        if raw.isdigit() and 1 <= int(raw) <= len(reasons):
            n = int(raw)
            break
    note = ask("[eval] note (Enter to skip): ").strip()
    return {"verdict": "fail", "reason": reasons[n - 1], "note": note}


def gate_verdict(gate_dir: Path, trial: int, reasons: list[str]) -> dict | None:
    path = Path(gate_dir) / f"verdict.{trial}"
    print(f"[eval] waiting for {path}", flush=True)
    while not path.exists():
        time.sleep(0.05)
    v = json.loads(path.read_text())
    path.unlink()
    if v["verdict"] == "redo":
        return None
    reason = reasons[int(v["reason"]) - 1] if v["verdict"] == "fail" else ""
    return {"verdict": v["verdict"], "reason": reason, "note": v.get("note", "")}


def start_trial_input(gate_dir: Path | None, trial: int) -> None:
    """Forget stop requests made before the trial started (an Enter typed during the takeover)."""
    if gate_dir is not None:
        (Path(gate_dir) / f"stop.{trial}").unlink(missing_ok=True)
    elif sys.stdin.isatty():
        import termios

        termios.tcflush(sys.stdin, termios.TCIFLUSH)


def stop_requested(gate_dir: Path | None, trial: int) -> bool:
    """Polled once per 50 Hz tick: the operator pressed Enter (or wrote stop.{trial})."""
    if gate_dir is not None:
        path = Path(gate_dir) / f"stop.{trial}"
        if path.exists():
            path.unlink()
            return True
        return False
    if select.select([sys.stdin], [], [], 0)[0]:
        sys.stdin.readline()
        return True
    return False


def next_trial(session_dir: Path) -> int:
    path = Path(session_dir) / "results.jsonl"
    return len(path.read_text().splitlines()) if path.exists() else 0


def _save_npz(path: Path, **arrays) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.savez_compressed(f, **arrays)
    os.replace(tmp, path)  # atomic: a crash never leaves a half-written trial


def write_trial(
    session_dir: Path, trial: int, log: dict[str, list], extra: dict[str, str], frames: list[tuple[float, str, bytes]]
) -> dict:
    session_dir = Path(session_dir)
    session_dir.mkdir(parents=True, exist_ok=True)
    wall = np.asarray(log["wall"], np.float64)
    episode = wall[np.asarray(log["phase"]) == "episode"]
    wall_start, wall_stop = (float(episode[0]), float(episode[-1])) if len(episode) else (math.nan, math.nan)
    base = f"trial_{trial:03d}"
    _save_npz(
        session_dir / f"{base}.npz",
        **{k: np.asarray(v) for k, v in log.items()},
        **extra,
        wall_start=wall_start,
        wall_stop=wall_stop,
    )
    blobs = [b for _, _, b in frames]
    _save_npz(
        session_dir / f"{base}_frames.npz",
        jpeg=np.frombuffer(b"".join(blobs), np.uint8),
        offsets=np.cumsum([0] + [len(b) for b in blobs]).astype(np.int64),
        t_recv=np.array([t for t, _, _ in frames], np.float64),
        names=np.array([n for _, n, _ in frames], dtype=str),
    )
    line = {
        "trial": trial,
        **{k: extra[k] for k in ("verdict", "reason", "note", "episode_end")},
        "wall_start": wall_start,
        "wall_stop": wall_stop,
        "frames": len(frames),
    }
    with open(session_dir / "results.jsonl", "a") as f:
        f.write(json.dumps(line) + "\n")
    return line


def load_frames(path: Path) -> tuple[np.ndarray, np.ndarray, list[bytes]]:
    z = np.load(path)
    raw, offsets = z["jpeg"].tobytes(), z["offsets"]
    return z["t_recv"], z["names"], [raw[a:b] for a, b in zip(offsets[:-1], offsets[1:], strict=True)]


def summarize(session_dir: Path) -> str:
    lines = [json.loads(s) for s in (Path(session_dir) / "results.jsonl").read_text().splitlines()]
    ok = sum(line["verdict"] == "success" for line in lines)
    reasons = Counter(line["reason"] for line in lines if line["verdict"] == "fail")
    return "; ".join([f"success {ok}/{len(lines)}", *(f"{r}: {n}" for r, n in reasons.most_common())])
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `/home/jihun/work/lerobot/.venv/bin/python -m pytest tests/datasets/test_g1_dex3_eval_session.py -v`
Expected: 6 passed

- [ ] **Step 5: Lint and commit**

```bash
pre-commit run --files examples/g1_dex3_training/eval_session.py tests/datasets/test_g1_dex3_eval_session.py
git add examples/g1_dex3_training/eval_session.py tests/datasets/test_g1_dex3_eval_session.py
git commit -m "feat(g1-dex3): eval_session.py, operator verdicts and atomic trial files for streamer evaluation sessions

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: Split `SonicBackend.start()` into `connect()` + `take_over()`

**Files:**

- Modify: `examples/g1_dex3_training/stream_backends.py` (`gate()` at lines ~125-131, `SonicBackend.start()` at lines ~181-214)
- Test: `tests/datasets/test_g1_dex3_stream_backends.py` (append to `SonicBackendTests`)

**Interfaces:**

- Produces:
  - `gate(name: str, gate_dir: Path | None, prefix: str = "", consume: bool = False) -> None` — prompt is looked up by `name.split(".")[0]`; `consume` deletes the flag file once seen.
  - `SonicBackend.connect() -> None` — once per process: `deploy_ready` gate, `start` command in PLANNER mode.
  - `SonicBackend.take_over(trial: int | None = None, n_trials: int | None = None) -> None` — per trial: `settled` gate (`settled.{trial}`, consumed, prompt prefixed `trial {trial+1}/{n_trials}: `), then today's handoff.
  - `SonicBackend.start()` unchanged in behavior: `connect()` then `take_over()`.

- [ ] **Step 1: Write the failing test** (append inside `class SonicBackendTests`)

```python
    def test_two_trials_connect_once_and_take_over_per_trial(self):
        pub, rec, state = FakePub(), Recorder(), FakeState()
        a = SimpleNamespace(
            handoff_blend_s=0.2, start="planner", end="planner", max_token_step=0.0, action_space="sonic78"
        )
        chunk = ramp(40, 78, 0.5)
        with backend_env() as a.gate_dir:
            (a.gate_dir / "settled.0").touch()
            (a.gate_dir / "settled.1").touch()
            b = SonicBackend(a, pub, state, rec, 30, None)
            b.connect()
            for trial in (0, 1):
                b.take_over(trial, 2)
                self.assertFalse((a.gate_dir / f"settled.{trial}").exists())  # consumed
                b.begin(chunk, None)
                b.episode_tick(0.02, 0, 0.0)
                b.finish()
        msgs = [decode(m) for m in pub.sent]
        commands = [m[1:] for m in msgs if m[0] == "command"]
        # one start in PLANNER mode; per trial: POSE mode, then back to PLANNER
        self.assertEqual(commands, [(1, 1), (1, 0), (1, 1), (1, 0), (1, 1)])
        phases = [r.phase for r in rec.rows]
        self.assertEqual(phases.count("planner token"), 100)
        self.assertEqual(phases.count("episode"), 2)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `/home/jihun/work/lerobot/.venv/bin/python -m pytest tests/datasets/test_g1_dex3_stream_backends.py -v -k two_trials`
Expected: FAIL with `AttributeError: 'SonicBackend' object has no attribute 'connect'`

- [ ] **Step 3: Implement**

Replace `gate()` in `stream_backends.py`:

```python
def gate(name: str, gate_dir: Path | None, prefix: str = "", consume: bool = False) -> None:
    """Wait for the operator (Enter) or, with gate_dir, for the flag file `name` (deleted once seen if consume:
    per-trial gates such as settled.3 are written again for every trial)."""
    if gate_dir is None:
        input(f"[streamer] {prefix}{GATE_PROMPTS.get(name.split('.')[0], name)}: ")
        return
    path = gate_dir / name
    print(f"[streamer] waiting for {path}", flush=True)
    while not path.exists():
        time.sleep(0.05)
    if consume:
        path.unlink()
```

In `SonicBackend`, replace the first lines of `start()` (through `gate("settled", a.gate_dir)`) so that the method becomes three methods; the body after `wait_state(self.state)` moves unchanged into `take_over()`:

```python
    def start(self):
        self.connect()
        self.take_over()

    def connect(self):
        """Once per process: start control in PLANNER mode."""
        gate("deploy_ready", self.a.gate_dir)
        time.sleep(0.5)  # let the deploy's SUB connect to our PUB before the first command
        self.pub.send(command_message(start=True, planner=True))
        print("[streamer] command: start control, PLANNER mode", flush=True)

    def take_over(self, trial: int | None = None, n_trials: int | None = None):
        """Per episode: wait until the operator has the robot in position (planner mode), then hand over to POSE
        mode. --trials sessions repeat this after each finish() --end planner hand-back."""
        a, startup = self.a, self.startup
        if trial is None:
            gate("settled", a.gate_dir)
        else:
            gate(f"settled.{trial}", a.gate_dir, f"trial {trial + 1}/{n_trials}: ", consume=True)
        wait_state(self.state)
        # ... the existing body from `if (a.handoff_blend_s > 0):` through
        # `print("[streamer] command: POSE mode (streamed tokens)", flush=True)` stays here unchanged
```

Also update the module docstring line `start()                 gates and mode switches, until the robot holds the rest pose` to:

```
    start()                 gates and mode switches, until the robot holds the rest pose
                            (SonicBackend: connect() once + take_over() per episode, for --trials sessions)
```

- [ ] **Step 4: Run all backend and streamer tests**

Run: `/home/jihun/work/lerobot/.venv/bin/python -m pytest tests/datasets/test_g1_dex3_stream_backends.py tests/datasets/test_g1_dex3_streamer_cli.py -v`
Expected: all pass (the existing single-episode test is unchanged)

- [ ] **Step 5: Lint and commit**

```bash
pre-commit run --files examples/g1_dex3_training/stream_backends.py tests/datasets/test_g1_dex3_stream_backends.py
git add examples/g1_dex3_training/stream_backends.py tests/datasets/test_g1_dex3_stream_backends.py
git commit -m "feat(g1-dex3): SonicBackend connect() once and take_over() per trial; consumable per-trial gates

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: `LiveImages` records the received JPEGs

**Files:**

- Modify: `examples/g1_dex3_training/sonic_policy_streamer.py` (`class LiveImages`, lines ~342-410)
- Test: `tests/datasets/test_g1_dex3_eval_trials.py` (new file; Task 4 adds the fake-rig test to it)

**Interfaces:**

- Produces: `LiveImages.start_recording() -> None`; `LiveImages.stop_recording() -> list[tuple[float, str, bytes]]` — `(time.time() at receipt, stream name, JPEG bytes)` for every received image while recording.

- [ ] **Step 1: Write the failing test**

```python
"""sonic_policy_streamer.py --trials: camera recording and a two-trial session against a fake rig."""

import base64
import json
import socket
import sys
import time
from pathlib import Path

import numpy as np
import pytest

zmq = pytest.importorskip("zmq")
cv2 = pytest.importorskip("cv2")
pytest.importorskip("msgpack")

HERE = Path(__file__).parents[2] / "examples" / "g1_dex3_training"
sys.path.insert(0, str(HERE))
from sonic_policy_streamer import LiveImages


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def jpeg(value: int) -> bytes:
    return cv2.imencode(".jpg", np.full((48, 64, 3), value, np.uint8))[1].tobytes()


def camera_message(value: int) -> str:
    return json.dumps({"images": {"egocentric": base64.b64encode(jpeg(value)).decode("ascii")}})


def test_live_images_records_received_jpegs():
    port, ctx = free_port(), zmq.Context()
    pub = ctx.socket(zmq.PUB)
    pub.bind(f"tcp://127.0.0.1:{port}")
    key = "observation.images.egocentric"
    images = LiveImages(ctx, "127.0.0.1", port, [key], {key: (3, 48, 64)}, "head_camera", "t")
    try:
        deadline = time.monotonic() + 10
        while not images.latest and time.monotonic() < deadline:  # PUB/SUB slow joiner
            pub.send_string(camera_message(10))
            time.sleep(0.05)
        assert images.stop_recording() == []  # nothing recorded before start_recording()
        images.start_recording()
        for v in (20, 30, 40):
            pub.send_string(camera_message(v))
            time.sleep(0.1)
        rec = images.stop_recording()
        pub.send_string(camera_message(50))
        time.sleep(0.1)
        assert images.stop_recording() == []
    finally:
        images.close()
        pub.close(0)
        ctx.term()
    assert [name for _, name, _ in rec] == ["egocentric"] * 3
    assert [b for _, _, b in rec] == [jpeg(20), jpeg(30), jpeg(40)]  # the bytes as sent, not re-encoded
    assert all(abs(t - time.time()) < 5 for t, _, _ in rec)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `/home/jihun/work/lerobot/.venv/bin/python -m pytest tests/datasets/test_g1_dex3_eval_trials.py -v`
Expected: FAIL with `AttributeError: 'LiveImages' object has no attribute 'stop_recording'`

- [ ] **Step 3: Implement**

In `LiveImages.__init__`, after `self.latest, self.lock = {}, threading.Lock()` add:

```python
        self.recording: list[tuple[float, str, bytes]] | None = None  # --trials: JPEGs as received
```

Replace the body of the `while` loop in `_recv` from `msg = json.loads(sock.recv_string())` to the `with self.lock:` block with:

```python
            msg = json.loads(sock.recv_string())
            t = time.time()
            raw = {name: base64.b64decode(b64) for name, b64 in msg.get("images", {}).items()}
            recording = self.recording  # one read: the main thread swaps the list
            if recording is not None:
                recording.extend((t, name, jpg) for name, jpg in raw.items())
            decoded = {
                name: cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR) for name, jpg in raw.items()
            }
            with self.lock:
                self.latest.update(decoded)
```

Add methods after `close()`:

```python
    def start_recording(self) -> None:
        self.recording = []

    def stop_recording(self) -> list[tuple[float, str, bytes]]:
        """(receive time, stream name, JPEG bytes) of every image since start_recording(); [] if not recording."""
        recording, self.recording = self.recording, None
        return recording or []
```

Extend the class docstring with: `start_recording() / stop_recording() keep the received JPEG bytes (no re-encoding) for --trials sessions.`

- [ ] **Step 4: Run the test**

Run: `/home/jihun/work/lerobot/.venv/bin/python -m pytest tests/datasets/test_g1_dex3_eval_trials.py -v`
Expected: 1 passed

- [ ] **Step 5: Lint and commit**

```bash
pre-commit run --files examples/g1_dex3_training/sonic_policy_streamer.py tests/datasets/test_g1_dex3_eval_trials.py
git add examples/g1_dex3_training/sonic_policy_streamer.py tests/datasets/test_g1_dex3_eval_trials.py
git commit -m "feat(g1-dex3): streamer LiveImages records the received head-camera JPEGs

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: Streamer `--trials` session loop

**Files:**

- Modify: `examples/g1_dex3_training/sonic_policy_streamer.py` (imports ~lines 61-100, module docstring, `main()` argparse ~575 and validation ~684, episode body ~767-936)
- Test: `tests/datasets/test_g1_dex3_eval_trials.py` (append), `tests/datasets/test_g1_dex3_streamer_cli.py` (argument errors)

**Interfaces:**

- Consumes: Task 1 (`parse_reasons`, `ask_verdict`, `gate_verdict`, `start_trial_input`, `stop_requested`, `next_trial`, `write_trial`, `summarize`), Task 2 (`connect`, `take_over`), Task 3 (`start_recording`, `stop_recording`).
- Produces: CLI `--trials N --session-dir DIR [--failure-reasons "a,b"]`; the session directory format of Task 1 with `extra = {"episode_end", "args", "verdict", "reason", "note"}`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/datasets/test_g1_dex3_streamer_cli.py`, inside `test_argument_errors`'s `cases` list:

```python
            (["--trials", "2", "--policy-server", "tcp://x:1", "--images", "zmq", "--task", "t", "--duration-s", "5"],
             "--trials needs --session-dir"),
            (["--trials", "2", "--session-dir", "s", "--policy-server", "tcp://x:1", "--dataset-root", "d",
              "--episode", "0"], "--trials needs --images zmq"),
```

Append to `tests/datasets/test_g1_dex3_eval_trials.py` (move the four new imports — `subprocess`, `threading`, `msgpack`, `eval_session` — to the import block at the top of the file; ruff rejects mid-file imports):

```python
import subprocess
import threading

import msgpack

import eval_session


def serve_policy(ctx, port, stop):
    rep = ctx.socket(zmq.REP)
    rep.bind(f"tcp://127.0.0.1:{port}")
    key = "observation.images.egocentric"
    while not stop.is_set():
        if not rep.poll(50):
            continue
        req = msgpack.unpackb(rep.recv(), raw=False)
        if req["op"] == "info":
            out = {"image_keys": [key], "n_obs": 1, "shapes": {key: [3, 48, 64]}}
        else:
            out = {"chunk": np.zeros((40, 78), np.float32).tobytes(), "shape": [40, 78]}
        rep.send(msgpack.packb(out, use_bin_type=True))
    rep.close(0)


def publish(ctx, port, stop, period, make):
    pub = ctx.socket(zmq.PUB)
    pub.bind(f"tcp://127.0.0.1:{port}")
    i = 0
    while not stop.is_set():
        make(pub, i)
        i += 1
        time.sleep(period)
    pub.close(0)


def deploy_state(pub, i):
    msg = {"body_q": [0.0] * 29, "left_hand_q": [0.0] * 7, "right_hand_q": [0.0] * 7, "token_state": [0.0] * 64}
    pub.send(b"g1_debug" + msgpack.packb(msg, use_bin_type=True))


def camera(pub, i):
    pub.send_string(camera_message(i % 250))


def wait_gone(path: Path, timeout=120):
    deadline = time.monotonic() + timeout
    while path.exists():
        assert time.monotonic() < deadline, f"{path} was never consumed"
        time.sleep(0.05)


def drive(gate: Path):
    """The operator: trial 0 redone once, then success at the duration cap; trial 1 stopped early and failed."""
    for name in ("deploy_ready", "settled.0"):
        (gate / name).touch()
    (gate / "verdict.0").write_text(json.dumps({"verdict": "redo"}))
    wait_gone(gate / "verdict.0")
    (gate / "settled.0").touch()
    (gate / "verdict.0").write_text(json.dumps({"verdict": "success"}))
    wait_gone(gate / "verdict.0")
    (gate / "settled.1").touch()
    wait_gone(gate / "settled.1")
    time.sleep(3.5)  # past the takeover and the 1 s blend-in: inside the episode
    (gate / "stop.1").touch()
    (gate / "verdict.1").write_text(json.dumps({"verdict": "fail", "reason": 2, "note": "x"}))


def test_two_trial_session_against_a_fake_rig(tmp_path):
    ports = {k: free_port() for k in ("policy", "state", "camera", "action")}
    ctx, stop = zmq.Context(), threading.Event()
    threads = [
        threading.Thread(target=serve_policy, args=(ctx, ports["policy"], stop)),
        threading.Thread(target=publish, args=(ctx, ports["state"], stop, 0.02, deploy_state)),
        threading.Thread(target=publish, args=(ctx, ports["camera"], stop, 1 / 30, camera)),
    ]
    gate, session = tmp_path / "gate", tmp_path / "session"
    gate.mkdir()
    threads.append(threading.Thread(target=drive, args=(gate,), daemon=True))
    for t in threads:
        t.start()
    try:
        r = subprocess.run(
            [sys.executable, str(HERE / "sonic_policy_streamer.py"),
             "--policy-server", f"tcp://127.0.0.1:{ports['policy']}", "--images", "zmq",
             "--camera-host", "127.0.0.1", "--camera-port", str(ports["camera"]),
             "--deploy-host", "127.0.0.1", "--state-port", str(ports["state"]), "--action-port", str(ports["action"]),
             "--task", "pick", "--duration-s", "6", "--start", "planner", "--end", "planner",
             "--handoff-blend-s", "0.2", "--trials", "2", "--session-dir", str(session), "--gate-dir", str(gate)],
            capture_output=True, text=True, timeout=240, stdin=subprocess.DEVNULL,
        )  # fmt: skip
    finally:
        stop.set()
        for t in threads[:3]:
            t.join()
        ctx.term()
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    lines = [json.loads(s) for s in (session / "results.jsonl").read_text().splitlines()]
    assert [(x["trial"], x["verdict"], x["episode_end"]) for x in lines] == [
        (0, "success", "completed"),
        (1, "fail", "operator stop"),
    ]
    assert lines[1]["reason"] == eval_session.DEFAULT_FAILURE_REASONS[1] and lines[1]["note"] == "x"
    assert 5.5 < lines[0]["wall_stop"] - lines[0]["wall_start"] < 6.1  # the duration cap
    assert 0.2 < lines[1]["wall_stop"] - lines[1]["wall_start"] < 5.0  # stopped by the operator
    t, names, jpegs = eval_session.load_frames(session / "trial_001_frames.npz")
    assert len(jpegs) > 30 and set(names) == {"egocentric"} and np.all(np.diff(t) > 0)
    assert cv2.imdecode(np.frombuffer(jpegs[0], np.uint8), cv2.IMREAD_COLOR).shape == (48, 64, 3)
    log = np.load(session / "trial_000.npz")
    assert str(log["verdict"]) == "success" and json.loads(str(log["args"]))["trials"] == 2
    assert "success 1/2" in r.stdout
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `/home/jihun/work/lerobot/.venv/bin/python -m pytest tests/datasets/test_g1_dex3_eval_trials.py tests/datasets/test_g1_dex3_streamer_cli.py -v`
Expected: `test_argument_errors` FAIL (`unrecognized arguments: --trials`), `test_two_trial_session_against_a_fake_rig` FAIL (non-zero return code, same error)

- [ ] **Step 3: Implement**

3a. Imports — add to both the relative and the absolute import branches (next to the `wbc_common` import):

```python
    from .eval_session import (
        ask_verdict,
        gate_verdict,
        next_trial,
        parse_reasons,
        start_trial_input,
        stop_requested,
        summarize,
        write_trial,
    )
```

(the `except ImportError:` branch gets the same list with `from eval_session import (...)`).

3b. Module docstring — append a paragraph:

```
Evaluation sessions (--trials N --session-dir DIR): one process runs N operator-judged trials. Per trial the operator
puts the robot in position (planner mode) and presses Enter, the policy runs until the operator presses Enter again
(or --duration-s, or a watchdog stop), the robot hands back to the planner, and the terminal asks for
success / fail (+ reason from --failure-reasons, + note) / redo. Each kept trial is written at once (eval_session.py:
the 50 Hz log, the head-camera JPEGs as received, results.jsonl); a restart with the same --session-dir continues at
the next trial. eval_session_to_lerobot.py turns the session into a LeRobot dataset.
```

3c. Arguments — after the `--log` argument:

```python
    p.add_argument(
        "--trials",
        type=int,
        help="evaluation session: run this many operator-judged trials in one process (needs --session-dir; "
        "--duration-s becomes the per-trial cap; Enter stops a trial)",
    )
    p.add_argument("--session-dir", type=Path, help="--trials: directory for the trial files and results.jsonl")
    p.add_argument(
        "--failure-reasons",
        help="--trials: comma-separated failure menu (default: eval_session.DEFAULT_FAILURE_REASONS)",
    )
```

3d. Validation — directly after `a = p.parse_args()`:

```python
    if a.trials is not None:
        if a.trials < 1 or a.session_dir is None or a.log:
            p.error("--trials needs --session-dir and a count >= 1 (it replaces --log)")
        if a.images != "zmq" or a.backend != "sonic" or a.end != "planner" or a.handoff_blend_s <= 0:
            p.error("--trials needs --images zmq, --backend sonic, --end planner and --handoff-blend-s > 0")
        parse_reasons(a.failure_reasons)  # fail before the robot moves
```

3e. Episode body — wrap everything from `backend.start()` through `backend.finish()` into a nested function, and replace the tail of `main()`. The code between `if a.backend == "decoupled": … backend = SonicBackend(...)` and `backend.start()` stays. The result (from `backend.start()` to the end of `main()`):

```python
    def run_episode(trial: int | None) -> tuple[str, list]:
        """One episode. In a --trials session (trial is not None) it takes the robot over from the planner, records
        the head camera and ends early when the operator asks. Returns how it ended and the recorded frames."""
        history.clear()
        for rows in log.values():
            rows.clear()
        if trial is None:
            backend.start()
        else:
            backend.take_over(trial, a.trials)
            start_trial_input(a.gate_dir, trial)
        # Optional hold on the rest pose ... (existing comment and loop, unchanged)
        for _ in ticks(int(a.hold_rest_s * 50)):
            backend.wait_tick()
        if trial is not None:
            images.start_recording()
        # First chunk: ... (existing comment, unchanged)
        slot = worker.submit(infer, 0.0, robot_states(0.0))
        for _ in ticks(10**9):
            if slot["done"].is_set():
                break
            backend.wait_tick()
        if "error" in slot:
            raise RuntimeError(slot["error"])
        _, first, first_j = slot["result"]
        backend.begin(first, first_j)

        latencies, slot, slot_t = [], None, 0.0
        next_replan, last_chunk_t, chunk_t0 = a.replan_s, 0.0, 0.0
        last_accepted = True  # the first chunk is playing
        episode_end = "completed"
        for i in ticks(int(duration / TICK)):
            t_ep = i * TICK
            if trial is not None and stop_requested(a.gate_dir, trial):
                episode_end = "operator stop"
                print(f"[streamer] {episode_end}: ending episode", flush=True)
                break
            # ... the rest of the existing loop body, unchanged
        frames = images.stop_recording() if trial is not None else []
        if latencies:
            print(
                f"[streamer] chunk inference s: median {np.median(latencies):.2f}, max {max(latencies):.2f}, n {len(latencies)}",
                flush=True,
            )
        backend.finish()
        return episode_end, frames

    if a.trials:
        reasons = parse_reasons(a.failure_reasons)
        backend.connect()
        trial = next_trial(a.session_dir)
        if trial:
            print(f"[eval] {a.session_dir} has {trial} trials: continuing with trial {trial + 1}", flush=True)
        while trial < a.trials:
            episode_end, frames = run_episode(trial)
            print(f"[eval] trial {trial + 1}/{a.trials} ended: {episode_end}", flush=True)
            verdict = (
                gate_verdict(a.gate_dir, trial, reasons) if a.gate_dir else ask_verdict(reasons, episode_end)
            )
            if verdict is None:
                print(f"[eval] trial {trial + 1} discarded: redo", flush=True)
                continue
            extra = {"episode_end": episode_end, "args": json.dumps(vars(a), default=str), **verdict}
            write_trial(a.session_dir, trial, log, extra, frames)
            print(f"[eval] trial {trial + 1}/{a.trials} saved: {verdict['verdict']} {verdict['reason']}".rstrip(), flush=True)
            trial += 1
        print(f"[eval] session {a.session_dir}: {summarize(a.session_dir)}", flush=True)
    else:
        episode_end, _ = run_episode(None)
        if a.log:
            np.savez_compressed(
                a.log,
                **{k: np.asarray(v) for k, v in log.items()},
                episode=-1 if a.episode is None else a.episode,
                episode_end=episode_end,
                args=json.dumps(vars(a), default=str),  # the exact settings of this run
            )
        if a.gate_dir:
            write_done(a.gate_dir, episode_end)
    if isinstance(images, LiveImages):
        images.close()
```

`history` and `log` are already defined above as a list and a dict of lists; `history.clear()` and `rows.clear()` reset them in place so the existing `record`, `arm_watchdog` and `robot_states` closures keep working.

- [ ] **Step 4: Run the tests**

Run: `/home/jihun/work/lerobot/.venv/bin/python -m pytest tests/datasets/test_g1_dex3_eval_trials.py tests/datasets/test_g1_dex3_streamer_cli.py tests/datasets/test_g1_dex3_stream_backends.py tests/datasets/test_g1_dex3_eval_session.py -v`
Expected: all pass (the fake-rig test takes ~30–40 s)

- [ ] **Step 5: Lint and commit**

```bash
pre-commit run --files examples/g1_dex3_training/sonic_policy_streamer.py tests/datasets/test_g1_dex3_eval_trials.py tests/datasets/test_g1_dex3_streamer_cli.py
git add examples/g1_dex3_training/sonic_policy_streamer.py tests/datasets/test_g1_dex3_eval_trials.py tests/datasets/test_g1_dex3_streamer_cli.py
git commit -m "feat(g1-dex3): streamer --trials evaluation sessions: operator stop, verdict prompt, per-trial files

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Launcher `session` and `pull` modes

**Files:**

- Modify: `examples/g1_dex3_training/g1_groot_real_run.sh`

**Interfaces:**

- Produces: `g1_groot_real_run.sh session SESSION [TASK] [extra streamer args…]` (env `TRIALS`, default 10; `MAX_S`, default 120); `g1_groot_real_run.sh pull SESSION` (env `SESSIONS`, default `/mnt/data/jihun/g1_eval_sessions`) → `$SESSIONS/$SESSION/raw/` and `$SESSIONS/$SESSION/$SESSION.log`; `g1_groot_real_run.sh sync` copies the streamer files to PC2.

- [ ] **Step 1: Add the usage lines** to the header comment, after the `stream` line:

```bash
#   g1_groot_real_run.sh session SESSION [TASK] [extra streamer args...]
#                                                # PC2: evaluation session (TRIALS, default 10; MAX_S per-trial cap,
#                                                # default 120): Enter at "Init Done", then per trial Enter in position,
#                                                # Enter to stop, s / f (+ reason, note) / r
#   g1_groot_real_run.sh pull SESSION            # copy the session from PC2 to $SESSIONS/SESSION/raw
#   g1_groot_real_run.sh sync                    # copy the streamer files to PC2 (~/g1_sonic_eval/code)
```

and update the `*)` case to print the new header length: `sed -n 2,21p "$0"; exit 1 ;;` (count the header lines after editing and use that range).

- [ ] **Step 2: Add the cases** before `stop)`:

```bash
session)
  SESSION=${2:?session name, e.g. s01_he_groot_laptop}; TASK=${3:-close a laptop g1}; shift $(($# < 3 ? $# : 3))
  ssh -t pc2_222 "mkdir -p ~/g1_sonic_eval/sessions && cd ~/g1_sonic_eval/code && ../.venv/bin/python -u sonic_policy_streamer.py \
    --policy-server tcp://$WS:$PORT --images zmq --camera-host localhost --task '$TASK' \
    --start planner --end planner --duration-s ${MAX_S:-120} --replan-s 0.4 --max-token-step 0.1 ${BLEND_S:+--chunk-blend-s $BLEND_S} \
    ${POLICY_FPS:+--policy-fps $POLICY_FPS} --trials ${TRIALS:-10} --session-dir ../sessions/$SESSION $* \
    2>&1 | tee -a ../sessions/$SESSION.log" ;;
pull)
  SESSION=${2:?session name}; DEST=${SESSIONS:-/mnt/data/jihun/g1_eval_sessions}/$SESSION
  mkdir -p "$DEST"
  rsync -a "pc2_222:g1_sonic_eval/sessions/$SESSION/" "$DEST/raw/"
  rsync -a "pc2_222:g1_sonic_eval/sessions/$SESSION.log" "$DEST/"
  echo "$DEST/raw" ;;
sync)
  scp sonic_policy_streamer.py stream_backends.py eval_session.py pc2_222:g1_sonic_eval/code/ ;;
```

- [ ] **Step 3: Check the script**

Run: `bash -n examples/g1_dex3_training/g1_groot_real_run.sh && bash examples/g1_dex3_training/g1_groot_real_run.sh 2>&1 | grep -c "session\|pull\|sync"`
Expected: no syntax error; the count is at least 3 (the usage text lists the new modes)

- [ ] **Step 4: Commit**

```bash
git add examples/g1_dex3_training/g1_groot_real_run.sh
git commit -m "feat(g1-dex3): launcher session, pull and sync modes for evaluation sessions

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Session → LeRobot dataset converter

**Files:**

- Create: `examples/g1_dex3_training/eval_session_to_lerobot.py`
- Test: `tests/datasets/test_g1_dex3_eval_convert.py`

**Interfaces:**

- Consumes: Task 1 (`write_trial` file format, `load_frames`, `summarize`); `sonic_targets.ACTION_NAMES` (28 names).
- Produces: `convert(session_dir: Path, out: Path, repo_id: str, subtasks: list[str], fps: int = 30) -> dict` (the `eval_session.json` content); `frame_rows(times: np.ndarray, at: np.ndarray) -> np.ndarray`; CLI `--session-dir --out --subtasks [--repo-id] [--fps] [--overwrite]`. `meta/eval_session.json` keys: `task, subtasks, failure_reasons, model, streamer_args, date, fps, episodes{"<ep>": {trial, verdict, reason, note, episode_end, wall_start, wall_stop}}, skipped[{trial, why}]`.

- [ ] **Step 1: Write the failing test**

```python
"""eval_session_to_lerobot.py: a --trials session becomes a LeRobot v3.0 dataset with meta/eval_session.json."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
sys.path.insert(0, str(Path(__file__).parents[2] / "examples" / "g1_dex3_training"))
import eval_session
from eval_session_to_lerobot import convert, frame_rows

from lerobot.datasets.lerobot_dataset import LeRobotDataset

ARGS = {"task": "pick the bottle", "action_space": "sonic78", "policy_server": "tcp://ws:5560",
        "failure_reasons": None, "trials": 3}  # fmt: skip


def trial(session, i, t0, n_episode, verdict, reason=""):
    """10 handoff ticks, n_episode episode ticks, 10 blend-out ticks at 50 Hz; camera at 30 Hz from t0 - 0.5 s.
    state[:, 0] and token[:, 0] hold the tick's wall time; each JPEG is a flat grey of value 10 * (index % 20)."""
    n = 20 + n_episode
    wall = t0 - 0.2 + 0.02 * np.arange(n)
    phase = ["handoff blend"] * 10 + ["episode"] * n_episode + ["blend out"] * 10
    log = {
        "wall": list(wall),
        "phase": phase,
        "state": [np.full(28, w, np.float32) for w in wall],
        "token": [np.full(64, w, np.float32) for w in wall],
        "hands": [np.zeros(14, np.float32) for _ in wall],
        "joint_ref": [np.full(31, np.nan, np.float32) for _ in wall],
    }
    t_cam = t0 - 0.5 + np.arange(int((n * 0.02 + 0.5) * 30)) / 30
    frames = [
        (t, "egocentric", cv2.imencode(".jpg", np.full((48, 64, 3), 10 * (k % 20), np.uint8))[1].tobytes())
        for k, t in enumerate(t_cam)
    ]
    extra = {"episode_end": "operator stop", "args": json.dumps(ARGS), "verdict": verdict, "reason": reason, "note": ""}
    eval_session.write_trial(session, i, log, extra, frames)
    return wall, t_cam


def test_frame_rows():
    times = np.array([1.0, 2.0, 3.0])
    assert list(frame_rows(times, np.array([0.5, 1.0, 2.5, 9.0]))) == [0, 0, 1, 2]


def test_convert_two_episodes_and_a_skipped_trial(tmp_path):
    session = tmp_path / "raw"
    wall0, cam0 = trial(session, 0, 1000.0, 60, "success")
    trial(session, 1, 2000.0, 0, "fail", "stalled / froze")  # never reached the episode phase
    wall2, _ = trial(session, 2, 3000.0, 45, "fail", "grasp failed / missed")
    meta = convert(session, tmp_path / "dataset", "local/test_eval", ["reach", "grasp"], fps=30)
    assert meta["skipped"] == [{"trial": 1, "why": "no episode ticks"}]
    assert [(e["trial"], e["verdict"]) for e in meta["episodes"].values()] == [(0, "success"), (2, "fail")]
    assert meta["subtasks"] == ["reach", "grasp"] and meta["model"] == "tcp://ws:5560"
    assert json.loads((tmp_path / "dataset/meta/eval_session.json").read_text()) == meta

    ds = LeRobotDataset("local/test_eval", root=tmp_path / "dataset")
    assert ds.meta.total_episodes == 2 and ds.fps == 30
    span0 = wall0[10 + 59] - wall0[10]
    assert ds.meta.episodes[0]["length"] == int(np.floor(span0 * 30)) + 1
    item = ds[5]  # episode 0, frame 5
    t = wall0[10] + 5 / 30
    expected_tick = wall0[frame_rows(wall0, np.array([t]))[0]]
    assert item["observation.state"][0].item() == pytest.approx(expected_tick, abs=1e-3)
    assert item["action"].shape == (78,) and item["action"][0].item() == pytest.approx(expected_tick, abs=1e-3)
    k = frame_rows(cam0, np.array([t]))[0]
    grey = item["observation.images.egocentric"].float().mean().item() * 255
    assert grey == pytest.approx(10 * (k % 20), abs=8)  # flat grey through the video codec
    assert item["task"] == "pick the bottle"


def test_refuses_existing_output_and_missing_frames(tmp_path):
    session = tmp_path / "raw"
    trial(session, 0, 1000.0, 30, "success")
    (tmp_path / "dataset").mkdir()
    with pytest.raises(FileExistsError):
        convert(session, tmp_path / "dataset", "local/x", ["reach"])
    (session / "trial_000_frames.npz").unlink()
    with pytest.raises(FileNotFoundError, match="trial_000_frames.npz"):
        convert(session, tmp_path / "other", "local/x", ["reach"])
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `/home/jihun/work/lerobot/.venv/bin/python -m pytest tests/datasets/test_g1_dex3_eval_convert.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'eval_session_to_lerobot'`

- [ ] **Step 3: Implement** `examples/g1_dex3_training/eval_session_to_lerobot.py`:

```python
"""Convert an evaluation session of sonic_policy_streamer.py --trials (pulled from PC2) to a LeRobot v3.0 dataset.

One episode per kept trial (results.jsonl order) over the policy-controlled phase, wall_start..wall_stop (the first /
last "episode" tick); trials that never reached it are skipped and listed under "skipped". Frame k is at
wall_start + k / fps and takes the latest camera JPEG and the latest 50 Hz log tick at or before that time:
  observation.images.<stream>  the head camera as received (the JPEGs hold RGB data, so no colour conversion)
  observation.state (28)       measured arms + Dex3 hands (sonic_targets.ACTION_NAMES order)
  action                       what the streamer sent: token 64 + hands 14 (sonic78), or joint_ref 28 / 31
meta/eval_session.json carries the operator's verdicts and the session settings for the visualizer's Evaluation
panel (the subtask list is given here: the VLM labels the failure step against it).

Usage: eval_session_to_lerobot.py --session-dir RAW --out DATASET --subtasks "reach,grasp,lift,handover"
                                  [--repo-id local/NAME] [--fps 30] [--overwrite]
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from eval_session import load_frames, parse_reasons, summarize
from sonic_targets import ACTION_NAMES

from lerobot.datasets.lerobot_dataset import LeRobotDataset

WAIST_NAMES = ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint")  # motors 12-14
MAX_CAMERA_GAP_S = 0.2


def frame_rows(times: np.ndarray, at: np.ndarray) -> np.ndarray:
    """Index of the latest of the sorted `times` at or before each of `at` (the first one before the first)."""
    return np.clip(np.searchsorted(times, at, side="right") - 1, 0, len(times) - 1)


def action_rows(log, action_space: str) -> tuple[np.ndarray, list[str]]:
    if action_space == "sonic78":
        names = [f"sonic_token_{i}" for i in range(64)] + list(ACTION_NAMES[14:])
        return np.concatenate([log["token"], log["hands"]], axis=1), names
    width = 31 if action_space == "joint31" else 28
    return log["joint_ref"][:, :width], (list(ACTION_NAMES) + list(WAIST_NAMES))[:width]


def convert(session_dir: Path, out: Path, repo_id: str, subtasks: list[str], fps: int = 30) -> dict:
    session_dir, out = Path(session_dir), Path(out)
    if out.exists():
        raise FileExistsError(f"{out} exists (use --overwrite)")
    lines = [json.loads(s) for s in (session_dir / "results.jsonl").read_text().splitlines()]
    trials = []
    for line in lines:
        base = session_dir / f"trial_{line['trial']:03d}"
        for path in (base.with_suffix(".npz"), Path(f"{base}_frames.npz")):
            if not path.exists():
                raise FileNotFoundError(f"missing {path.name} for trial {line['trial']}")
        trials.append((line, np.load(base.with_suffix(".npz")), load_frames(Path(f"{base}_frames.npz"))))
    args = json.loads(str(trials[0][1]["args"]))
    action_space = args.get("action_space", "sonic78")
    skipped, kept = [], []
    for line, log, frames in trials:
        if not np.isfinite(line["wall_start"]):
            print(f"[convert] trial {line['trial']}: no episode ticks, skipped")
            skipped.append({"trial": line["trial"], "why": "no episode ticks"})
        elif not len(frames[2]):
            print(f"[convert] trial {line['trial']}: no camera frames, skipped")
            skipped.append({"trial": line["trial"], "why": "no camera frames"})
        else:
            kept.append((line, log, frames))
    if not kept:
        raise ValueError("no trial reached the episode phase")
    cameras = sorted(set(kept[0][2][1]))
    shapes = {
        c: cv2.imdecode(np.frombuffer(kept[0][2][2][list(kept[0][2][1]).index(c)], np.uint8), cv2.IMREAD_COLOR).shape
        for c in cameras
    }
    _, action_names = action_rows(kept[0][1], action_space)
    features = {
        "observation.state": {"dtype": "float32", "shape": (28,), "names": list(ACTION_NAMES)},
        "action": {"dtype": "float32", "shape": (len(action_names),), "names": action_names},
        **{
            f"observation.images.{c}": {"dtype": "video", "shape": shapes[c], "names": ["height", "width", "channels"]}
            for c in cameras
        },
    }
    ds = LeRobotDataset.create(repo_id, fps=fps, root=out, robot_type="unitree_g1_dex3_sonic", features=features)
    episodes = {}
    for ep, (line, log, (t_recv, names, jpegs)) in enumerate(kept):
        n = int(np.floor((line["wall_stop"] - line["wall_start"]) * fps)) + 1
        at = line["wall_start"] + np.arange(n) / fps
        rows = frame_rows(log["wall"], at)
        state = log["state"][rows].astype(np.float32)
        action = action_rows(log, action_space)[0][rows].astype(np.float32)
        per_cam = {}
        for c in cameras:
            sel = np.flatnonzero(names == c)
            t_c = t_recv[sel]
            inside = t_c[(t_c >= line["wall_start"]) & (t_c <= line["wall_stop"])]
            for gap_at, gap in zip(inside[:-1], np.diff(inside), strict=True):
                if gap > MAX_CAMERA_GAP_S:
                    print(f"[convert] trial {line['trial']} {c}: camera gap {gap:.2f} s at {gap_at - line['wall_start']:.2f} s")
            per_cam[c] = sel[frame_rows(t_c, at)]
        for k in range(n):
            frame = {"observation.state": state[k], "action": action[k], "task": args["task"]}
            for c in cameras:
                frame[f"observation.images.{c}"] = cv2.imdecode(
                    np.frombuffer(jpegs[per_cam[c][k]], np.uint8), cv2.IMREAD_COLOR
                )
            ds.add_frame(frame)
        ds.save_episode()
        episodes[str(ep)] = {
            k: line[k] for k in ("trial", "verdict", "reason", "note", "episode_end", "wall_start", "wall_stop")
        }
    ds.finalize()
    meta = {
        "task": args["task"],
        "subtasks": subtasks,
        "failure_reasons": parse_reasons(args.get("failure_reasons")),
        "model": args.get("policy_server") or args.get("policy_path"),
        "streamer_args": args,
        "date": datetime.fromtimestamp(kept[0][0]["wall_start"]).date().isoformat(),
        "fps": fps,
        "episodes": episodes,
        "skipped": skipped,
    }
    (out / "meta/eval_session.json").write_text(json.dumps(meta, indent=2))
    print(f"[convert] {out}: {len(kept)} episodes; {summarize(session_dir)}")
    return meta


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--session-dir", type=Path, required=True, help="raw session (g1_groot_real_run.sh pull)")
    p.add_argument("--out", type=Path, required=True, help="new dataset directory")
    p.add_argument("--subtasks", required=True, help="the task's ordered subtasks, comma-separated")
    p.add_argument("--repo-id", help="default: local/<out directory name>")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--overwrite", action="store_true", help="replace an existing --out")
    a = p.parse_args()
    subtasks = [s.strip() for s in a.subtasks.split(",") if s.strip()]
    if not subtasks:
        p.error("--subtasks is empty")
    if a.overwrite and a.out.exists():
        shutil.rmtree(a.out)
    convert(a.session_dir, a.out, a.repo_id or f"local/{a.out.name}", subtasks, a.fps)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests**

Run: `/home/jihun/work/lerobot/.venv/bin/python -m pytest tests/datasets/test_g1_dex3_eval_convert.py -v`
Expected: 3 passed

- [ ] **Step 5: Convert the fake-rig session end to end** (catches format drift between Tasks 4 and 6)

Run (`$SCRATCH` = any empty scratch directory; the dataset is reused in Task 10):

```bash
/home/jihun/work/lerobot/.venv/bin/python -m pytest tests/datasets/test_g1_dex3_eval_trials.py -k fake_rig --basetemp=$SCRATCH/evalrig -q
RAW=$(ls -d $SCRATCH/evalrig/test_two_trial*/session)
cd examples/g1_dex3_training && /home/jihun/work/lerobot/.venv/bin/python eval_session_to_lerobot.py --session-dir $RAW --out $SCRATCH/evalrig/dataset --subtasks "reach,grasp"
```

Expected: `[convert] …/dataset: 2 episodes; success 1/2; grasp failed / missed: 1`

- [ ] **Step 6: Lint and commit**

```bash
pre-commit run --files examples/g1_dex3_training/eval_session_to_lerobot.py tests/datasets/test_g1_dex3_eval_convert.py
git add examples/g1_dex3_training/eval_session_to_lerobot.py tests/datasets/test_g1_dex3_eval_convert.py
git commit -m "feat(g1-dex3): eval_session_to_lerobot.py, evaluation session to LeRobot v3.0 dataset with verdict metadata

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: Visualizer backend — `eval_labels.py`

**Setup (once, before this task):**

```bash
cd /home/jihun/work/lerobot-dataset-visualizer
git worktree add ../lerobot-dataset-visualizer-eval -b feat/eval-labels main
cd ../lerobot-dataset-visualizer-eval && bun install
```

**Files:**

- Create: `backend/eval_labels.py`
- Test: `backend/tests/test_eval_labels.py`

**Interfaces:**

- Produces:
  - `PROMPT_VERSION = "eval-labels-v1"`, `SAMPLE_FPS = 2.0`, `MAX_FRAMES = 64`
  - `read_session(root: Path) -> dict | None` (the `meta/eval_session.json` content)
  - `read_labels(root: Path) -> dict[str, dict]`; `write_label(root: Path, episode: int, label: dict) -> None` (atomic; caller holds the edit lock)
  - `sample_frames(n_frames: int, fps: float) -> list[int]`
  - `build_messages(images: list, frame_indices: list[int], timestamps: list[float], session: dict, entry: dict) -> list[dict]`
  - `validate_reply(reply, subtasks: list[str], n_frames: int) -> dict` — raises `ValueError`
  - `label_episodes(records: list, frames, vlm, session: dict, model: str) -> dict[int, dict]` — per episode `{"ok": True, "label": {...}, "atoms": [...], "raw": reply}` or `{"ok": False, "error": str, "raw": reply | None}`; `label` = `{vlm_verdict, failed_subtask, failure_frame, rationale, source: "vlm", model, prompt_version}`; `atoms` = `subtask` atoms (`role="assistant", style="subtask", content=<name>, timestamp=<frame time>, camera=None, tool_calls=None`).

- [ ] **Step 1: Write the failing tests** (`backend/tests/test_eval_labels.py`)

```python
from types import SimpleNamespace

from lerobot.annotations.steerable_pipeline.vlm_client import StubVlmClient
import pytest

import eval_labels

SESSION = {
    "task": "pick the bottle",
    "subtasks": ["reach", "grasp", "lift"],
    "episodes": {"0": {"verdict": "fail", "reason": "grasp failed / missed", "note": ""}, "1": {"verdict": "success", "reason": "", "note": ""}},
}


class Frames:
    def frames_at(self, record, timestamps, camera_key=None):
        return [f"img@{t:.2f}" for t in timestamps]


def record(ep, n=90, fps=30):
    return SimpleNamespace(episode_index=ep, frame_timestamps=tuple(k / fps for k in range(n)))


GOOD_FAIL = {
    "subtask_starts": [{"subtask": "reach", "frame": 0}, {"subtask": "grasp", "frame": 40}],
    "vlm_verdict": "fail", "failed_subtask": "grasp", "failure_frame": 60, "rationale": "missed",
}


def test_sample_frames():
    assert eval_labels.sample_frames(90, 30) == list(range(0, 90, 15))
    many = eval_labels.sample_frames(30 * 120, 30)
    assert len(many) == eval_labels.MAX_FRAMES and many[0] == 0 and many[-1] == 30 * 120 - 1


def test_messages_label_each_frame_and_include_the_operator_verdict():
    msgs = eval_labels.build_messages(["a", "b"], [0, 15], [0.0, 0.5], SESSION, SESSION["episodes"]["0"])
    texts = [b["text"] for b in msgs[0]["content"] if b["type"] == "text"]
    assert texts[0] == "frame 0 (t=0.0 s)" and texts[1] == "frame 15 (t=0.5 s)"
    assert "1. reach" in texts[-1] and "grasp failed / missed" in texts[-1] and "pick the bottle" in texts[-1]
    assert [b["image"] for b in msgs[0]["content"] if b["type"] == "image"] == ["a", "b"]


@pytest.mark.parametrize(
    ("reply", "error"),
    [
        ("not json", "not a JSON object"),
        ({**GOOD_FAIL, "subtask_starts": []}, "non-empty"),
        ({**GOOD_FAIL, "subtask_starts": [{"subtask": "push", "frame": 0}]}, "unknown subtask"),
        ({**GOOD_FAIL, "subtask_starts": [{"subtask": "grasp", "frame": 0}, {"subtask": "reach", "frame": 5}]}, "order"),
        ({**GOOD_FAIL, "subtask_starts": [{"subtask": "reach", "frame": 0}, {"subtask": "grasp", "frame": 90}]}, "frame"),
        ({**GOOD_FAIL, "vlm_verdict": "maybe"}, "vlm_verdict"),
        ({**GOOD_FAIL, "failed_subtask": "lift"}, "failed_subtask"),
        ({**GOOD_FAIL, "failure_frame": 10}, "failure_frame"),
        ({**GOOD_FAIL, "vlm_verdict": "success"}, "null for a success"),
    ],
)
def test_validate_reply_rejects(reply, error):
    with pytest.raises(ValueError, match=error):
        eval_labels.validate_reply(reply, SESSION["subtasks"], 90)


def test_label_episodes_keeps_good_episodes_when_one_fails():
    replies = {0: GOOD_FAIL, 1: {"subtask_starts": [{"subtask": "reach", "frame": 0}], "vlm_verdict": "partial"}}

    def respond(messages):
        return replies[0] if "grasp failed / missed" in messages[-1]["content"][-1]["text"] else replies[1]

    out = eval_labels.label_episodes([record(0), record(1)], Frames(), StubVlmClient(respond), SESSION, "qwen")
    assert out[0]["ok"] and out[0]["label"] == {
        "vlm_verdict": "fail", "failed_subtask": "grasp", "failure_frame": 60, "rationale": "missed",
        "source": "vlm", "model": "qwen", "prompt_version": eval_labels.PROMPT_VERSION,
    }
    assert [(a["content"], a["timestamp"]) for a in out[0]["atoms"]] == [("reach", 0.0), ("grasp", 40 / 30)]
    assert out[0]["atoms"][0] == {"role": "assistant", "style": "subtask", "content": "reach", "timestamp": 0.0, "camera": None, "tool_calls": None}
    assert not out[1]["ok"] and "vlm_verdict" in out[1]["error"] and out[1]["raw"] == replies[1]


def test_client_exception_fails_only_that_episode():
    def respond(messages):
        raise ConnectionError("endpoint down")

    out = eval_labels.label_episodes([record(0)], Frames(), StubVlmClient(respond), SESSION, "qwen")
    assert out[0] == {"ok": False, "error": "ConnectionError('endpoint down')", "raw": None}


def test_labels_file_roundtrip(tmp_path):
    (tmp_path / "meta").mkdir()
    assert eval_labels.read_labels(tmp_path) == {} and eval_labels.read_session(tmp_path) is None
    eval_labels.write_label(tmp_path, 3, {"failed_subtask": "grasp"})
    assert eval_labels.read_labels(tmp_path) == {"3": {"failed_subtask": "grasp"}}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `/home/jihun/work/lerobot-dataset-visualizer/backend/.venv/bin/python -m pytest -q backend/tests/test_eval_labels.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'eval_labels'`

- [ ] **Step 3: Implement** `backend/eval_labels.py`:

```python
"""VLM labels for robot evaluation sessions (lerobot eval_session_to_lerobot.py datasets).

meta/eval_session.json (written by the converter, never changed here) holds the operator's verdict per episode and
the task's ordered subtask list. Per episode the VLM sees frames sampled at SAMPLE_FPS (each labelled with its frame
index) and returns subtask start frames, its own verdict, and for a failure the failed subtask and failure frame.
Subtask starts become the existing `subtask` atoms; the rest goes to meta/eval_labels.json. The operator's verdict
stays authoritative: a differing vlm_verdict is only shown as a disagreement.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from lerobot.annotations.steerable_pipeline.frames import to_image_blocks

PROMPT_VERSION = "eval-labels-v1"
SAMPLE_FPS = 2.0
MAX_FRAMES = 64

PROMPT = """You are checking one trial of a robot evaluation. Task: "{task}".
The task's subtasks, in order:
{subtasks}
The images above are frames of the trial, {fps:g} per second, each labelled with its frame index.
The human operator judged this trial: {verdict}.
Reply with JSON only:
{{"subtask_starts": [{{"subtask": "<name from the list>", "frame": <frame index where it starts>}}, ...],
 "vlm_verdict": "success" or "fail",
 "failed_subtask": "<name from the list>" or null,
 "failure_frame": <frame index> or null,
 "rationale": "<one sentence>"}}
List only the subtasks the robot started, in list order. If the trial failed, failed_subtask is the subtask during
which it failed and failure_frame the first frame where the failure is visible; both are null for a success."""


def read_session(root: Path) -> dict | None:
    path = Path(root) / "meta/eval_session.json"
    return json.loads(path.read_text()) if path.exists() else None


def read_labels(root: Path) -> dict[str, dict]:
    path = Path(root) / "meta/eval_labels.json"
    return json.loads(path.read_text())["episodes"] if path.exists() else {}


def write_label(root: Path, episode: int, label: dict) -> None:
    labels = read_labels(root)
    labels[str(episode)] = label
    path = Path(root) / "meta/eval_labels.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps({"version": 1, "episodes": labels}, indent=2))
    temporary.replace(path)


def sample_frames(n_frames: int, fps: float) -> list[int]:
    step = max(1, round(fps / SAMPLE_FPS))
    indices = list(range(0, n_frames, step))
    if len(indices) > MAX_FRAMES:
        indices = sorted({int(i) for i in np.linspace(0, n_frames - 1, MAX_FRAMES).round()})
    return indices


def build_messages(images: list, frame_indices: list[int], timestamps: list[float], session: dict, entry: dict) -> list[dict]:
    content: list[dict[str, Any]] = []
    for image, k, t in zip(images, frame_indices, timestamps, strict=True):
        content.append({"type": "text", "text": f"frame {k} (t={t:.1f} s)"})
        content.extend(to_image_blocks([image]))
    verdict = entry["verdict"]
    if verdict == "fail":
        verdict += f" ({entry['reason']}{'; ' + entry['note'] if entry.get('note') else ''})"
    subtasks = "\n".join(f"{i}. {s}" for i, s in enumerate(session["subtasks"], 1))
    content.append(
        {"type": "text", "text": PROMPT.format(task=session["task"], subtasks=subtasks, fps=SAMPLE_FPS, verdict=verdict)}
    )
    return [{"role": "user", "content": content}]


def _frame(value: Any, n_frames: int) -> bool:
    return type(value) is int and 0 <= value < n_frames


def validate_reply(reply: Any, subtasks: list[str], n_frames: int) -> dict:
    if not isinstance(reply, dict):
        raise ValueError("reply is not a JSON object")
    starts = reply.get("subtask_starts")
    if not isinstance(starts, list) or not starts or not all(isinstance(s, dict) for s in starts):
        raise ValueError("subtask_starts must be a non-empty list of objects")
    names, frames = [s.get("subtask") for s in starts], [s.get("frame") for s in starts]
    if any(n not in subtasks for n in names):
        raise ValueError(f"unknown subtask in {names}")
    order = [subtasks.index(n) for n in names]
    if order != sorted(set(order)):
        raise ValueError(f"subtasks out of list order or repeated: {names}")
    if not all(_frame(f, n_frames) for f in frames) or frames != sorted(set(frames)):
        raise ValueError(f"subtask frames must be increasing frame indices in [0, {n_frames}): {frames}")
    verdict = reply.get("vlm_verdict")
    if verdict not in ("success", "fail"):
        raise ValueError(f"vlm_verdict must be success or fail, got {verdict!r}")
    failed, failure_frame = reply.get("failed_subtask"), reply.get("failure_frame")
    if verdict == "fail":
        if failed not in names:
            raise ValueError(f"failed_subtask must be one of the started subtasks {names}, got {failed!r}")
        if not _frame(failure_frame, n_frames) or failure_frame < frames[names.index(failed)]:
            raise ValueError(f"failure_frame must be a frame index at or after the start of {failed!r}")
    elif failed is not None or failure_frame is not None:
        raise ValueError("failed_subtask and failure_frame must be null for a success")
    return {
        "subtask_starts": [{"subtask": n, "frame": f} for n, f in zip(names, frames, strict=True)],
        "vlm_verdict": verdict,
        "failed_subtask": failed,
        "failure_frame": failure_frame,
        "rationale": str(reply.get("rationale", "")),
    }


def label_episodes(records: list, frames, vlm, session: dict, model: str) -> dict[int, dict]:
    """One VLM call per episode, so one bad reply or failed call leaves the other episodes labelled."""
    out: dict[int, dict] = {}
    for record in records:
        ep, timestamps = record.episode_index, list(record.frame_timestamps)
        fps = (len(timestamps) - 1) / timestamps[-1] if len(timestamps) > 1 and timestamps[-1] > 0 else 30.0
        indices = sample_frames(len(timestamps), fps)
        reply = None
        try:
            images = frames.frames_at(record, [timestamps[k] for k in indices])
            if len(images) != len(indices):
                raise ValueError(f"decoded {len(images)} of {len(indices)} frames")
            messages = build_messages(images, indices, [timestamps[k] for k in indices], session, session["episodes"][str(ep)])
            reply = vlm.generate_json([messages])[0]
            parsed = validate_reply(reply, session["subtasks"], len(timestamps))
        except Exception as exc:  # reported per episode; the job goes on
            out[ep] = {"ok": False, "error": str(exc) if isinstance(exc, ValueError) else repr(exc), "raw": reply}
            continue
        atoms = [
            {"role": "assistant", "style": "subtask", "content": s["subtask"], "timestamp": timestamps[s["frame"]], "camera": None, "tool_calls": None}
            for s in parsed["subtask_starts"]
        ]
        label = {k: parsed[k] for k in ("vlm_verdict", "failed_subtask", "failure_frame", "rationale")}
        out[ep] = {"ok": True, "label": {**label, "source": "vlm", "model": model, "prompt_version": PROMPT_VERSION}, "atoms": atoms, "raw": reply}
    return out
```

Note: the client-exception test expects `repr(exc)` for non-`ValueError` exceptions and `str(exc)` for validation errors — keep that split.

- [ ] **Step 4: Run the tests**

Run: `/home/jihun/work/lerobot-dataset-visualizer/backend/.venv/bin/python -m pytest -q backend/tests/test_eval_labels.py`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add backend/eval_labels.py backend/tests/test_eval_labels.py
git commit -m "feat: VLM failure-step labels for robot evaluation sessions

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 8: Visualizer backend — API endpoints and review hash

**Files:**

- Modify: `backend/annotation_history.py` (`review_status`, `save_review`)
- Modify: `backend/app.py` (imports near the other local modules; new routes after `/api/episodes/{episode_index}/review`)
- Test: `backend/tests/test_eval_labels.py` (append)

**Interfaces:**

- Consumes: Task 7 (`read_session`, `read_labels`, `write_label`, `label_episodes`, `PROMPT_VERSION`).
- Produces:
  - `annotation_history.eval_label_hash(root: Path, episode: int) -> str` — hash of the episode's `eval_labels.json` entry (`None` when absent); reviews store it as `eval_label_sha256` and are valid only while it matches (missing field = hash of `None`, so old reviews and non-eval datasets stay valid).
  - `GET /api/eval/episodes/{ep}?local_path|repo_id` → `{"entry", "subtasks", "failure_reasons", "fps", "label", "label_sha256", "summary": {"episodes", "success", "reasons", "labeled", "disagreements", "reviewed"}}`; 404 when the dataset has no `meta/eval_session.json` or the episode is not in it.
  - `POST /api/eval/label` body `{local_path|repo_id, episode_indices?: int[]}` → `{"job_id", "status"}`; job result `{"episodes": {"<ep>": {"ok", "error"?}}}`; poll with the existing `GET /api/annotation/jobs/{job_id}`.
  - `POST /api/eval/episodes/{ep}/label` body `{local_path|repo_id, failed_subtask: str|null, failure_frame: int|null, expected_label_sha256: str}` → same shape as GET; 409 on a stale hash; 422 on an unknown subtask or frame out of range.

- [ ] **Step 1: Write the failing tests** (append to `backend/tests/test_eval_labels.py`; put the new imports — `asyncio`, `json`, `annotation_history as history`, `app`, `httpx` — into the import block at the top of the file)

```python
import asyncio
import json

import annotation_history as history
import app
import httpx


def eval_dataset(root):
    (root / "meta/eval_session.json").write_text(
        json.dumps({**SESSION, "fps": 10, "failure_reasons": ["grasp failed / missed"], "episodes": {"0": {**SESSION["episodes"]["0"], "trial": 0}}})
    )
    return root


def test_review_includes_the_eval_label(tmp_path):
    atoms = [{"role": "assistant", "style": "subtask", "content": "reach", "timestamp": 0}]
    (tmp_path / "meta").mkdir()
    status = history.review_status(tmp_path, 0, atoms)
    history.save_review(tmp_path, 0, atoms, True, status["annotation_sha256"])
    assert history.review_status(tmp_path, 0, atoms)["status"] == "reviewed"  # no eval files: unchanged behaviour
    eval_labels.write_label(tmp_path, 0, {"failed_subtask": "grasp", "failure_frame": 1})
    assert history.review_status(tmp_path, 0, atoms)["status"] == "unreviewed"
    history.save_review(tmp_path, 0, atoms, True, status["annotation_sha256"])
    assert history.review_status(tmp_path, 0, atoms)["status"] == "reviewed"
    eval_labels.write_label(tmp_path, 0, {"failed_subtask": "grasp", "failure_frame": 2})  # reviewer edit
    assert history.review_status(tmp_path, 0, atoms)["status"] == "unreviewed"


def test_eval_api_get_and_human_edit(clear_dataset_state, legacy_v31_dataset):
    eval_dataset(legacy_v31_dataset)
    identity = {"local_path": str(legacy_v31_dataset)}

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app.app), base_url="http://test") as client:
            got = (await client.get("/api/eval/episodes/0", params=identity)).json()
            assert got["entry"]["verdict"] == "fail" and got["label"] is None and got["subtasks"] == SESSION["subtasks"]
            assert got["summary"] == {"episodes": 1, "success": 0, "reasons": {"grasp failed / missed": 1}, "labeled": 0, "disagreements": 0, "reviewed": 0}
            body = {**identity, "failed_subtask": "grasp", "failure_frame": 2, "expected_label_sha256": got["label_sha256"]}
            saved = (await client.post("/api/eval/episodes/0/label", json=body)).json()
            assert saved["label"]["source"] == "human" and saved["label"]["failure_frame"] == 2
            assert (await client.post("/api/eval/episodes/0/label", json=body)).status_code == 409  # stale hash
            bad = {**body, "expected_label_sha256": saved["label_sha256"], "failure_frame": 3}  # 3 frames: 0..2
            assert (await client.post("/api/eval/episodes/0/label", json=bad)).status_code == 422
            bad = {**body, "expected_label_sha256": saved["label_sha256"], "failed_subtask": "push"}
            assert (await client.post("/api/eval/episodes/0/label", json=bad)).status_code == 422
            assert (await client.get("/api/eval/episodes/7", params=identity)).status_code == 404

    asyncio.run(scenario())


def test_eval_label_job_writes_atoms_and_labels(clear_dataset_state, legacy_v31_dataset, monkeypatch):
    eval_dataset(legacy_v31_dataset)
    reply = {"subtask_starts": [{"subtask": "reach", "frame": 0}, {"subtask": "grasp", "frame": 1}],
             "vlm_verdict": "fail", "failed_subtask": "grasp", "failure_frame": 2, "rationale": "r"}  # fmt: skip
    monkeypatch.setattr(app, "_eval_vlm", lambda root: (StubVlmClient(lambda m: reply), Frames(), "stub"))
    identity = {"local_path": str(legacy_v31_dataset)}

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app.app), base_url="http://test") as client:
            job = (await client.post("/api/eval/label", json=identity)).json()
            for _ in range(200):
                status = (await client.get(f"/api/annotation/jobs/{job['job_id']}")).json()
                if status["status"] in ("completed", "failed"):
                    break
                await asyncio.sleep(0.05)
            assert status["status"] == "completed", status
            assert status["result"]["episodes"]["0"]["ok"]
            atoms = (await client.get("/api/episodes/0/atoms", params=identity)).json()["atoms"]
            assert [(a["style"], a["content"]) for a in atoms] == [("subtask", "reach"), ("subtask", "grasp")]
            got = (await client.get("/api/eval/episodes/0", params=identity)).json()
            assert got["label"]["model"] == "stub" and got["summary"]["labeled"] == 1 and got["summary"]["disagreements"] == 0

    asyncio.run(scenario())
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `/home/jihun/work/lerobot-dataset-visualizer/backend/.venv/bin/python -m pytest -q backend/tests/test_eval_labels.py`
Expected: the three new tests FAIL (`eval_label_hash` missing, 404/405 on `/api/eval/...`, `_eval_vlm` missing)

- [ ] **Step 3: Implement the review hash** in `backend/annotation_history.py`:

```python
def eval_label_hash(root: Path, episode: int) -> str:
    """Hash of the episode's robot-evaluation label (meta/eval_labels.json), so a review covers it too; None when
    absent, which keeps reviews of other datasets and of reviews saved before this field valid."""
    path = Path(root) / "meta/eval_labels.json"
    label = json.loads(path.read_text())["episodes"].get(str(episode)) if path.exists() else None
    return sha256(json.dumps(label, sort_keys=True).encode()).hexdigest()


_NO_EVAL_LABEL = sha256(json.dumps(None).encode()).hexdigest()
```

In `review_status`, add to the `reviewed = (...)` condition:

```python
        and review.get("eval_label_sha256", _NO_EVAL_LABEL) == eval_label_hash(root, episode)
```

In `save_review`, add to the stored dict:

```python
        # ponytail: binds to the label on disk at save time, not the one the browser showed; add an expected hash
        # (as for exclusions) if several reviewers edit one session at once
        "eval_label_sha256": eval_label_hash(root, episode),
```

- [ ] **Step 4: Implement the routes** in `backend/app.py`. Import `eval_labels` the way the file imports its other sibling modules (check the top of `app.py`; e.g. `import eval_labels` next to `import annotation_history`). Add after `set_episode_review`:

```python
# --- Robot evaluation sessions (meta/eval_session.json) ----------------------


class EvalLabelRequest(DatasetRef):
    episode_indices: list[int] | None = None


class EvalLabelEdit(DatasetRef):
    failed_subtask: str | None = None
    failure_frame: int | None = None
    expected_label_sha256: str


def _eval_vlm(root: Path):
    """(VLM client, frame provider, model id) from the /annotate VLM configuration."""
    from lerobot.annotations.steerable_pipeline.frames import make_frame_provider
    from lerobot.annotations.steerable_pipeline.vlm_client import make_vlm_client

    engine = _official_engine()
    config = engine.parse_config(engine.default_config())
    frames = make_frame_provider(root, camera_key=config.vlm.camera_key, video_backend=config.video_backend)
    return make_vlm_client(config.vlm), frames, config.vlm.model_id


def _eval_payload(state: DatasetState, session: dict, episode_index: int) -> dict:
    entry = session["episodes"].get(str(episode_index))
    if entry is None:
        raise HTTPException(404, "Episode is not part of this evaluation session")
    labels = eval_labels.read_labels(state.root)
    reviewed = 0
    for ep in session["episodes"]:
        atoms = json.loads(get_episode_atoms(int(ep), local_path=str(state.root)).body)["atoms"]
        reviewed += annotation_history.review_status(state.root, int(ep), atoms)["status"] == "reviewed"
    reasons: dict[str, int] = {}
    for e in session["episodes"].values():
        if e["verdict"] == "fail":
            reasons[e["reason"]] = reasons.get(e["reason"], 0) + 1
    disagreements = sum(
        1 for ep, e in session["episodes"].items() if ep in labels and labels[ep].get("vlm_verdict") not in (None, e["verdict"])
    )
    return {
        "entry": entry,
        "subtasks": session["subtasks"],
        "failure_reasons": session.get("failure_reasons", []),
        "fps": session.get("fps", 30),
        "label": labels.get(str(episode_index)),
        "label_sha256": annotation_history.eval_label_hash(state.root, episode_index),
        "summary": {
            "episodes": len(session["episodes"]),
            "success": sum(e["verdict"] == "success" for e in session["episodes"].values()),
            "reasons": reasons,
            "labeled": sum(ep in labels for ep in session["episodes"]),
            "disagreements": disagreements,
            "reviewed": reviewed,
        },
    }


def _eval_session(state: DatasetState) -> dict:
    session = eval_labels.read_session(state.root)
    if session is None:
        raise HTTPException(404, "Not an evaluation session dataset (no meta/eval_session.json)")
    return session


@app.get("/api/eval/episodes/{episode_index}")
def get_eval_episode(
    episode_index: int, repo_id: str | None = None, revision: str | None = None, local_path: str | None = None
) -> dict:
    state = _ensure_state(DatasetRef(repo_id=repo_id, revision=revision, local_path=local_path))
    return _eval_payload(state, _eval_session(state), episode_index)


@app.post("/api/eval/episodes/{episode_index}/label")
def set_eval_label(episode_index: int, payload: EvalLabelEdit) -> dict:
    state = _ensure_state(payload)
    session = _eval_session(state)
    with _annotation_edit_lock:
        _require_editable(state.root)
        if annotation_history.eval_label_hash(state.root, episode_index) != payload.expected_label_sha256:
            raise HTTPException(409, "Evaluation label changed; reload before saving")
        n_frames = len(_frame_timestamps(state, episode_index))
        if payload.failed_subtask is not None and payload.failed_subtask not in session["subtasks"]:
            raise HTTPException(422, f"Unknown subtask {payload.failed_subtask!r}")
        if payload.failure_frame is not None and not 0 <= payload.failure_frame < n_frames:
            raise HTTPException(422, f"failure_frame must be in [0, {n_frames})")
        label = eval_labels.read_labels(state.root).get(str(episode_index)) or {
            "vlm_verdict": None, "rationale": "", "model": None, "prompt_version": None,
        }
        label = {**label, "failed_subtask": payload.failed_subtask, "failure_frame": payload.failure_frame, "source": "human"}
        eval_labels.write_label(state.root, episode_index, label)
    return _eval_payload(state, session, episode_index)


@app.post("/api/eval/label")
def start_eval_labels(req: EvalLabelRequest) -> dict:
    state = _ensure_state(req)
    session = _eval_session(state)
    _require_editable(state.root)
    wanted = sorted(set(req.episode_indices or [int(ep) for ep in session["episodes"]]))

    def operation(_unused_output: Path) -> dict:
        from lerobot.annotations.steerable_pipeline.reader import iter_episodes

        vlm, frames, model = _eval_vlm(state.root)
        records = list(iter_episodes(state.root, only_episodes=tuple(wanted)))
        results = eval_labels.label_episodes(records, frames, vlm, session, model)
        ok = {ep for ep, r in results.items() if r["ok"]}
        with _annotation_edit_lock:
            for ep in sorted(ok):
                current = json.loads(get_episode_atoms(ep, local_path=str(state.root)).body)["atoms"]
                kept = [a for a in current if a.get("style") != "subtask"]
                state.annotations[ep] = EpisodeAnnotations(atoms=kept + results[ep]["atoms"])
                eval_labels.write_label(state.root, ep, results[ep]["label"])
            if ok:
                _save_annotations(state)
                annotation_history.snapshot_predictions(
                    state.root, state.root, ok, {}, {"eval_labels": eval_labels.PROMPT_VERSION, "vlm": {"model_id": model}},
                    eval_labels.PROMPT_VERSION,
                    episode_results={str(ep): {"raw": r["raw"], **({"label": r["label"]} if r["ok"] else {"error": r["error"]})} for ep, r in results.items()},
                )
        return {"episodes": {str(ep): {"ok": r["ok"], **({} if r["ok"] else {"error": r["error"]})} for ep, r in results.items()}}

    return _start_annotation_job(operation)
```

Check while implementing: `get_episode_atoms` takes `(episode_index, repo_id=None, revision=None, local_path=None)` positionally — call it with keywords as above; the `_ensure_state` cache key for `local_path=str(state.root)` must match the state the request used (it does for `local_path` requests; for `repo_id` local aliases, pass the same identity the request used — store `req`'s fields and pass them through instead of `local_path=str(state.root)` if the alias test fails).

- [ ] **Step 5: Run the backend tests**

Run: `/home/jihun/work/lerobot-dataset-visualizer/backend/.venv/bin/python -m pytest -q backend/tests/test_eval_labels.py backend/tests/test_annotation_history.py backend/tests/test_review_api.py`
Expected: all pass

- [ ] **Step 6: Run the full backend gate**

Run: `/home/jihun/work/lerobot-dataset-visualizer/backend/.venv/bin/python -m pytest -q backend/tests`
Expected: no new failures compared with `main` (run the same command on `main` first if anything fails, and compare)

- [ ] **Step 7: Commit**

```bash
git add backend/app.py backend/annotation_history.py backend/tests/test_eval_labels.py
git commit -m "feat: evaluation session API (labels, human edits, summary); reviews cover the eval label

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 9: Visualizer frontend — Evaluation panel

**Files:**

- Modify: `src/utils/annotationsClient.ts` (append)
- Create: `src/components/evaluation-panel.tsx`
- Modify: `src/components/annotations-panel.tsx` (render `<EvaluationPanel />` as the first child of `<div className="annotation-workbench">`, line ~561)
- Modify: `src/components/annotation-review-control.tsx` (listen to `eval-label-changed`)
- Modify: `src/components/annotations-timeline.tsx` (failure marker next to the playhead, line ~737)
- Test: `src/components/__tests__/evaluation-panel.test.tsx`

**Interfaces:**

- Consumes: Task 8 routes; `useAnnotations()` (`episodeId`, `ident`, `backendEnabled`, `snap`, `activeVideoEl`, `frameTimestamps`); `useTime()` (`currentTime`, `seek`); existing `getAnnotationJob(jobId)`.
- The spec's session summary (success rate, reasons, disagreements, reviewed) is shown in the panel header rather than in a separate dataset view: the panel is on every episode page, so no new view is needed.
- Produces: `fetchEvalEpisode`, `saveEvalLabel`, `startEvalLabeling`, types `EvalEpisode`, `EvalLabel`; window events `eval-label-changed` (after a save) and `eval-failure-changed` (`CustomEvent<{ time: number | null }>`, the failure marker time).

- [ ] **Step 1: Write the failing test** (`src/components/__tests__/evaluation-panel.test.tsx`)

```tsx
import React from "react";
import { afterEach, beforeEach, expect, mock, spyOn, test } from "bun:test";
import { fireEvent, render, waitFor } from "@testing-library/react";
import * as client from "../../utils/annotationsClient";
import {
  AnnotationsProvider,
  useAnnotations,
} from "../../context/annotations-context";
import { TimeProvider } from "../../context/time-context";
import { EvaluationPanel } from "../evaluation-panel";

const base: client.EvalEpisode = {
  entry: {
    trial: 2,
    verdict: "fail",
    reason: "grasp failed / missed",
    note: "slipped",
    episode_end: "operator stop",
  },
  subtasks: ["reach", "grasp", "lift"],
  failure_reasons: ["grasp failed / missed"],
  fps: 10,
  label: {
    vlm_verdict: "success",
    failed_subtask: null,
    failure_frame: null,
    rationale: "looks fine",
    source: "vlm",
    model: "qwen",
    prompt_version: "eval-labels-v1",
  },
  label_sha256: "h1",
  summary: {
    episodes: 5,
    success: 3,
    reasons: { "grasp failed / missed": 2 },
    labeled: 5,
    disagreements: 1,
    reviewed: 2,
  },
};

beforeEach(() => {
  spyOn(client, "isAnnotateBackendEnabled").mockReturnValue(true);
  spyOn(client, "fetchEpisodeAtomsWithHash").mockResolvedValue({
    atoms: [],
    annotation_sha256: "sha",
  });
  spyOn(client, "fetchFrameTimestamps").mockResolvedValue([
    0, 0.1, 0.2, 0.3, 0.4,
  ]);
  spyOn(client, "fetchEvalEpisode").mockResolvedValue(base);
});
afterEach(() => mock.restore());

function Episode() {
  const { setEpisode } = useAnnotations();
  React.useEffect(() => {
    setEpisode(1, { localPath: "/d" }, []);
  }, [setEpisode]);
  return <EvaluationPanel />;
}

const view = () =>
  render(
    <TimeProvider duration={1}>
      <AnnotationsProvider>
        <Episode />
      </AnnotationsProvider>
    </TimeProvider>,
  );

test("shows the operator verdict, the VLM disagreement and the session summary", async () => {
  const { findByText, getByText } = view();
  await findByText(/Operator: fail/);
  getByText(/grasp failed \/ missed/);
  getByText(/slipped/);
  getByText(/VLM: success/);
  getByText(/disagrees/i);
  getByText(/3\/5 success/);
});

test("saves the failed subtask and the playhead frame as a human label", async () => {
  const saved = {
    ...base,
    label: {
      ...base.label!,
      failed_subtask: "grasp",
      failure_frame: 0,
      source: "human" as const,
    },
    label_sha256: "h2",
  };
  const save = spyOn(client, "saveEvalLabel").mockResolvedValue(saved);
  const events: string[] = [];
  const listener = () => events.push("changed");
  window.addEventListener("eval-label-changed", listener);
  const { findByLabelText, getByText } = view();
  fireEvent.change(await findByLabelText("Failed subtask"), {
    target: { value: "grasp" },
  });
  fireEvent.click(getByText("Set to playhead"));
  fireEvent.click(getByText("Save label"));
  await waitFor(() => expect(save).toHaveBeenCalled());
  expect(save.mock.calls[0][2]).toEqual({
    failed_subtask: "grasp",
    failure_frame: 0,
    expected_label_sha256: "h1",
  });
  await waitFor(() => expect(events).toEqual(["changed"]));
  window.removeEventListener("eval-label-changed", listener);
});

test("renders nothing for a dataset without an evaluation session", async () => {
  spyOn(client, "fetchEvalEpisode").mockResolvedValue(null);
  const { container } = view();
  await waitFor(() => expect(container.textContent).toBe(""));
});
```

Before running, check `src/context/time-context.tsx` for the provider's exact export name and props (`TimeProvider` and its required props) and adjust the wrapper to match; copy the wrapper used by an existing test that renders timeline components if one exists (`grep -rl TimeProvider src/components/__tests__`).

- [ ] **Step 2: Run the test to verify it fails**

Run: `bun test src/components/__tests__/evaluation-panel.test.tsx`
Expected: FAIL (cannot resolve `../evaluation-panel`)

- [ ] **Step 3: Add the client functions** (append to `src/utils/annotationsClient.ts`)

```ts
export type EvalLabel = {
  vlm_verdict: "success" | "fail" | null;
  failed_subtask: string | null;
  failure_frame: number | null;
  rationale: string;
  source: "vlm" | "human";
  model: string | null;
  prompt_version: string | null;
};

export type EvalEpisode = {
  entry: {
    trial: number;
    verdict: "success" | "fail";
    reason: string;
    note: string;
    episode_end: string;
  };
  subtasks: string[];
  failure_reasons: string[];
  fps: number;
  label: EvalLabel | null;
  label_sha256: string;
  summary: {
    episodes: number;
    success: number;
    reasons: Record<string, number>;
    labeled: number;
    disagreements: number;
    reviewed: number;
  };
};

/** Evaluation data of an episode; null when the dataset is not an evaluation session. */
export async function fetchEvalEpisode(
  episodeId: number,
  ident: DatasetIdent,
): Promise<EvalEpisode | null> {
  const res = await fetch(buildUrl(`/api/eval/episodes/${episodeId}`, ident));
  if (res.status === 404) return null;
  if (!res.ok)
    throw new AnnotationRequestError(res.status, `evaluation: ${res.status}`);
  return (await res.json()) as EvalEpisode;
}

export async function saveEvalLabel(
  episodeId: number,
  ident: DatasetIdent,
  body: {
    failed_subtask: string | null;
    failure_frame: number | null;
    expected_label_sha256: string;
  },
): Promise<EvalEpisode> {
  const res = await fetch(
    buildUrl(`/api/eval/episodes/${episodeId}/label`, ident),
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        repo_id: ident.repoId || null,
        local_path: ident.localPath || null,
        revision: ident.revision || null,
        ...body,
      }),
    },
  );
  if (!res.ok)
    throw new AnnotationRequestError(
      res.status,
      (await res.text().catch(() => "")) || `evaluation label: ${res.status}`,
    );
  return (await res.json()) as EvalEpisode;
}

export async function startEvalLabeling(
  ident: DatasetIdent,
): Promise<{ job_id: string }> {
  return annotationRequest("/api/eval/label", {
    method: "POST",
    body: JSON.stringify({
      repo_id: ident.repoId || null,
      local_path: ident.localPath || null,
      revision: ident.revision || null,
    }),
  });
}
```

- [ ] **Step 4: Write the panel** (`src/components/evaluation-panel.tsx`)

```tsx
"use client";

import React from "react";
import { useAnnotations } from "../context/annotations-context";
import { useTime } from "../context/time-context";
import {
  fetchEvalEpisode,
  getAnnotationJob,
  saveEvalLabel,
  startEvalLabeling,
  type EvalEpisode,
} from "../utils/annotationsClient";

function announceFailure(time: number | null) {
  window.dispatchEvent(
    new window.CustomEvent("eval-failure-changed", { detail: { time } }),
  );
}

/** Operator verdict, VLM label and its human correction for robot evaluation sessions (meta/eval_session.json). */
export function EvaluationPanel() {
  const { episodeId, ident, backendEnabled, snap, activeVideoEl } =
    useAnnotations();
  const { currentTime, seek } = useTime();
  const [data, setData] = React.useState<EvalEpisode | null>(null);
  const [subtask, setSubtask] = React.useState<string>("");
  const [frame, setFrame] = React.useState<number | null>(null);
  const [busy, setBusy] = React.useState<string | null>(null);
  const [error, setError] = React.useState<string | null>(null);

  const load = React.useCallback(() => {
    if (!backendEnabled || episodeId == null) return;
    fetchEvalEpisode(episodeId, ident)
      .then((value) => {
        setData(value);
        setSubtask(value?.label?.failed_subtask ?? "");
        setFrame(value?.label?.failure_frame ?? null);
        setError(null);
      })
      .catch((cause) => setError(String(cause?.message ?? cause)));
  }, [backendEnabled, episodeId, ident]);

  React.useEffect(load, [load]);
  React.useEffect(() => {
    announceFailure(data && frame != null ? frame / data.fps : null);
  }, [data, frame]);
  React.useEffect(() => () => announceFailure(null), [episodeId]);

  if (!data)
    return error ? <div className="text-xs text-red-300">{error}</div> : null;
  const { entry, label, summary } = data;
  const disagrees =
    label?.vlm_verdict != null && label.vlm_verdict !== entry.verdict;

  const setToPlayhead = () => {
    const t = snap(activeVideoEl?.currentTime ?? currentTime);
    setFrame(Math.round(t * data.fps));
  };
  const save = async () => {
    if (episodeId == null) return;
    setBusy("Saving…");
    try {
      const next = await saveEvalLabel(episodeId, ident, {
        failed_subtask: subtask || null,
        failure_frame: frame,
        expected_label_sha256: data.label_sha256,
      });
      setData(next);
      window.dispatchEvent(new window.Event("eval-label-changed"));
      setError(null);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
    } finally {
      setBusy(null);
    }
  };
  const runVlm = async () => {
    setBusy("VLM labelling all episodes…");
    try {
      const { job_id } = await startEvalLabeling(ident);
      for (;;) {
        const job = await getAnnotationJob(job_id);
        if (job.status === "failed")
          throw new Error(job.error ?? "labelling failed");
        if (job.status === "completed") break;
        await new Promise((r) => setTimeout(r, 2000));
      }
      window.location.reload(); // subtask atoms changed on the server: reload them with the episode
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : String(cause));
      setBusy(null);
    }
  };

  return (
    <div className="rail-group text-xs" aria-label="Evaluation">
      <div className="rail-group-head">
        Evaluation · {summary.success}/{summary.episodes} success ·{" "}
        {summary.labeled} labelled · {summary.disagreements} disagreements ·{" "}
        {summary.reviewed} reviewed
      </div>
      <div>
        Operator: {entry.verdict}
        {entry.verdict === "fail" && ` — ${entry.reason}`}
        {entry.note && ` (${entry.note})`} · trial {entry.trial + 1} ·{" "}
        {entry.episode_end}
      </div>
      {label && (
        <div>
          VLM: {label.vlm_verdict ?? "—"}
          {disagrees && (
            <span className="ml-2 px-1 rounded bg-amber-500/20 text-amber-200">
              disagrees with the operator
            </span>
          )}
          {label.rationale && ` — ${label.rationale}`} ({label.source})
        </div>
      )}
      <div className="flex items-center gap-2 mt-1">
        <label>
          Failed subtask{" "}
          <select
            aria-label="Failed subtask"
            value={subtask}
            onChange={(e) => setSubtask(e.target.value)}
          >
            <option value="">—</option>
            {data.subtasks.map((s) => (
              <option key={s} value={s}>
                {s}
              </option>
            ))}
          </select>
        </label>
        <span>frame {frame ?? "—"}</span>
        <button type="button" onClick={setToPlayhead}>
          Set to playhead
        </button>
        {frame != null && (
          <button
            type="button"
            onClick={() => seek(frame / data.fps, "external")}
          >
            Go to
          </button>
        )}
        <button type="button" disabled={busy != null} onClick={save}>
          Save label
        </button>
        <button type="button" disabled={busy != null} onClick={runVlm}>
          Run VLM labels
        </button>
      </div>
      {busy && <div>{busy}</div>}
      {error && <div className="text-red-300">{error}</div>}
    </div>
  );
}
```

Check `getAnnotationJob`'s return type in `annotationsClient.ts` (line ~110) and adapt the `status` / `error` field access to it.

- [ ] **Step 5: Wire it in**

`src/components/annotations-panel.tsx`: `import { EvaluationPanel } from "./evaluation-panel";` and render `<EvaluationPanel />` as the first child inside `<div className="annotation-workbench">`.

`src/components/annotation-review-control.tsx`, in the effect that listens for `annotation-clipping-changed`, also listen for `eval-label-changed`:

```tsx
window.addEventListener("annotation-clipping-changed", changed);
window.addEventListener("eval-label-changed", changed);
return () => {
  window.removeEventListener("annotation-clipping-changed", changed);
  window.removeEventListener("eval-label-changed", changed);
};
```

`src/components/annotations-timeline.tsx`: inside the component, next to the other state hooks:

```tsx
const [failureTime, setFailureTime] = React.useState<number | null>(null);
React.useEffect(() => {
  const onFailure = (e: Event) =>
    setFailureTime((e as CustomEvent<{ time: number | null }>).detail.time);
  window.addEventListener("eval-failure-changed", onFailure);
  return () => window.removeEventListener("eval-failure-changed", onFailure);
}, []);
```

and directly before `<div className="tl-playhead" style={{ left: playheadLeft }} />`:

```tsx
{
  failureTime != null && (
    <div
      className="tl-playhead"
      title="failure frame"
      style={{
        left: `${(failureTime / duration) * 100}%`,
        background: "#ef4444",
      }}
    />
  );
}
```

- [ ] **Step 6: Run the tests and the repo gate**

Run: `bun test src/components/__tests__/evaluation-panel.test.tsx src/components/__tests__/annotation-review-control.test.tsx`
Expected: all pass

Run: `bun run format && bun run validate`
Expected: type-check, lint, format check and tests all pass

- [ ] **Step 7: Commit**

```bash
git add src/utils/annotationsClient.ts src/components/evaluation-panel.tsx src/components/annotations-panel.tsx src/components/annotation-review-control.tsx src/components/annotations-timeline.tsx src/components/__tests__/evaluation-panel.test.tsx
git commit -m "feat: Evaluation panel for robot evaluation sessions (operator verdict, VLM label, failure marker)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 10: Workflow doc and end-to-end check on the fake-rig dataset

**Files:**

- Create: `docs/research/2026-10-13-g1-eval-logger.md` (lerobot worktree)
- Modify: `CLAUDE.md` (lerobot worktree, the G1 Dex3 note paragraph)

- [ ] **Step 1: Write the workflow doc** with these sections, each with the exact commands:
  1. _Start_: `g1_groot_real_run.sh sync` (once per code change), `server`, `deploy`, `camera` as today.
  2. _Session_: `TRIALS=10 g1_groot_real_run.sh session s01_<model>_<task> "<task text>"`; operator keys: Enter at "Init Done"; per trial Enter in position, Enter to stop, `s` / `f` + reason number + note / `r`; a crash or Ctrl-C keeps finished trials, the same command continues at the next trial.
  3. _Pull and convert_: `g1_groot_real_run.sh pull s01_…`, then `eval_session_to_lerobot.py --session-dir $SESSIONS/s01_…/raw --out $SESSIONS/s01_…/dataset --subtasks "reach,grasp,…"`.
  4. _Label and review_: in the visualizer, `/annotate` → prepare the dataset directory (local path) → open the draft → Evaluation panel → **Run VLM labels** → per episode check subtasks on the timeline, set the failed subtask / **Set to playhead** / **Save label**, then **Mark reviewed**.
  5. _Files_: the session layout (Task 1 docstring) and `meta/eval_session.json` / `meta/eval_labels.json` keys.
  6. _Limits_: labels are ~0.5 s accurate (2 fps); the VLM sees the operator verdict when it judges (its verdict is a consistency check, not an independent one).

- [ ] **Step 2: Add the pointer to `CLAUDE.md`**, at the end of the G1 Dex3 paragraph:

```
Real-robot evaluation sessions (operator-judged trials, LeRobot export, VLM failure-step labels reviewed in the dataset visualizer): [`docs/research/2026-10-13-g1-eval-logger.md`](./docs/research/2026-10-13-g1-eval-logger.md).
```

- [ ] **Step 3: End-to-end check in the visualizer** with the fake-rig dataset from Task 6 Step 5 (`$SCRATCH/evalrig/dataset`): start the visualizer backend and `bun dev` from the visualizer worktree as described in its `docs/local-setup.md`, prepare the dataset in `/annotate`, open episode 1, and confirm: the Evaluation panel shows `Operator: fail — grasp failed / missed (x)`; **Save label** with `grasp` at the playhead shows a red marker on the timeline; **Mark reviewed** works and becomes unreviewed after changing the label again. **Run VLM labels** needs the configured VLM endpoint; if it is not reachable from this machine, record that and leave it for the first real session.

- [ ] **Step 4: Lint and commit**

```bash
pre-commit run --files docs/research/2026-10-13-g1-eval-logger.md CLAUDE.md
git add docs/research/2026-10-13-g1-eval-logger.md CLAUDE.md
git commit -m "docs(g1-dex3): real-robot evaluation session workflow (record, convert, VLM label, review)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

- [ ] **Step 5: Real robot (operator present, user-run)** — one short session, `TRIALS=2`, before relying on it: check that Enter stops a trial, the verdict prompt appears after the hand-back, both trials are in `results.jsonl`, and the converted dataset opens in the visualizer.
