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
# every trial of a session shares these (a resumed session must not change them)
SESSION_KEYS = ("task", "action_space", "policy_server", "policy_path")


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
    default = (
        reasons.index(SAFETY_STOP) + 1 if episode_end.startswith("watchdog") and SAFETY_STOP in reasons else 0
    )
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
    while True:
        time.sleep(0.05)
        try:
            v = json.loads(path.read_text())
        except (FileNotFoundError, json.JSONDecodeError):
            continue  # absent, or the writer is between create and write: keep waiting
        try:
            if not isinstance(v, dict) or v.get("verdict") not in ("success", "fail", "redo"):
                raise ValueError('"verdict" must be success, fail or redo')
            if v["verdict"] == "fail" and not (
                str(v.get("reason", "")).isdigit() and 1 <= int(v["reason"]) <= len(reasons)
            ):
                raise ValueError(f'a fail needs "reason" as a number 1..{len(reasons)}')
        except ValueError as e:
            print(f"[eval] {path}: {e}; fix the file", flush=True)
            path.unlink(missing_ok=True)
            continue
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


class HandoverDetector:
    """--end-on-handover: end the trial once one pick-and-handover attempt is complete, from the measured right hand
    (mean of its 7 Dex3 joints: ~0 open, ~0.38 closed around the bottle). It fires when the hand, after being seen
    open, stayed closed for at least ``min_hold_s`` (a grasp, not a flick) and then stayed open for ``open_s`` (the
    release; a shorter reopen is a regrasp). A hand that starts closed (the startup fist) and only opens is not a
    handover. One attempt per trial: in s17 (2026-10-23) second attempts within a trial were slower and less accurate,
    because the training episodes end about 2 s after the release.

    Durations use real elapsed time (``time.monotonic()``; ``now`` only for tests), never the streamer's scheduled
    episode time: ticks() catches up late ticks without sleeping, so after a stall the scheduled time races ahead and a
    brief closure would look like a held grasp (PR #47 review)."""

    def __init__(self, open_s: float = 1.0, min_hold_s: float = 0.5, closed_above: float = 0.19):
        self.open_s, self.min_hold_s, self.closed_above = open_s, min_hold_s, closed_above
        self.seen_open, self.held = False, False
        self.closed_since: float | None = None
        self.open_since: float | None = None

    def update(self, right_hand: float, now: float | None = None) -> bool:
        t = time.monotonic() if now is None else now
        if not math.isfinite(right_hand):
            return False
        if right_hand > self.closed_above:
            self.open_since = None
            if self.seen_open:
                if self.closed_since is None:
                    self.closed_since = t
                if t - self.closed_since >= self.min_hold_s:
                    self.held = True
            return False
        self.seen_open, self.closed_since = True, None
        if not self.held:
            return False
        if self.open_since is None:
            self.open_since = t
        return t - self.open_since >= self.open_s


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


def session_mismatch(first: dict, args: dict) -> list[str]:
    """The SESSION_KEYS where `args` differs from `first` (the session's trial 0 args)."""
    return [k for k in SESSION_KEYS if first.get(k) != args.get(k)]


def first_trial_args(session_dir: Path) -> dict | None:
    path = Path(session_dir) / "trial_000.npz"
    return json.loads(str(np.load(path)["args"])) if path.exists() else None


def _save_npz(path: Path, compress: bool = True, **arrays) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        (np.savez_compressed if compress else np.savez)(f, **arrays)
    os.replace(tmp, path)  # atomic: a crash never leaves a half-written trial


def write_trial(
    session_dir: Path,
    trial: int,
    log: dict[str, list],
    extra: dict[str, str],
    frames: list[tuple[float, str, bytes]],
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
        compress=False,  # JPEG bytes are already compressed
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
        "model": extra.get("model") or None,
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
